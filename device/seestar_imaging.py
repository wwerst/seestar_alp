#
# seestar_imaging - performances image-related tasks with a Seestar
#
# Config settings that shape the MJPEG stream served to the browser
# (all read from the [stream_quality] section of config.toml; both are
# BROWSER-HOP knobs — they do not change what the telescope pushes over
# its wifi uplink):
# .   experimental
# .   max_stream_fps  — cap on frames/s served to the browser (0 = uncapped)
# .   jpeg_quality    — cv2 JPEG encode quality for served frames (default 95)
#
import datetime
import os
import threading
from collections import deque
from time import monotonic, sleep, time
from typing import Optional

from flask import Flask, Response
import numpy as np
import cv2
from blinker import signal

import sys

from device import log
from device.analysis.snr_analysis import SNRAnalysis
from device.protocols.imager import SeestarImagerProtocol, ExposureModes
from device.config import Config


# view modes:
#   star: 3PPA, ContinuousExposure, Stack

# https://stackoverflow.com/questions/8554282/creating-a-png-file-in-python
# https://docs.astropy.org/en/stable/visualization/normalization.html#stretching

# Port 4700
#   Star:
# {  "id" : 112,  "method" : "iscope_start_view",  "params" : {    "mode" : "star"  }}
#   Moon:
# {  "id" : 254,  "method" : "iscope_start_view",  "params" : {    "mode" : "moon"  }}
# {  "id" : 255,  "method" : "start_scan_planet"}

# Port 4800
#   {  "id" : 21,  "method" : "begin_streaming"}
# Star:
#   {  "id" : 23,  "method" : "get_stacked_img"}


def table(rows):
    """Simple HTML table on a single row"""
    return "".join(
        [
            '<div class="row">'
            + "".join([f'<div class="col">{col}</div>' for col in row])
            + "</div>"
            for row in rows
        ]
    )


class SeestarImaging:
    # Rolling window (seconds) over which video_throughput() averages
    # bytes/s and frames/s served to the browser.
    THROUGHPUT_WINDOW_S = 10.0

    # ---- Stream-quality knobs (BROWSER-HOP only) -----------------------
    # These trim the MJPEG stream this imaging server re-encodes and pushes
    # to the *browser*; they do NOT change what the telescope pushes over
    # its wifi uplink (the RTSP H.264 bitrate is fixed by firmware), so they
    # do not relieve mount-command latency from a saturated uplink. See
    # docs/operations_runbook.md. Defaults preserve the historical behavior:
    #   max_stream_fps = 0   -> uncapped (serve loop spins at ~1 ms)
    #   jpeg_quality   = 95  -> OpenCV's default JPEG quality
    MAX_STREAM_FPS_DEFAULT = 0.0
    MAX_STREAM_FPS_CAP = 60.0
    JPEG_QUALITY_DEFAULT = 95
    JPEG_QUALITY_MIN = 10
    JPEG_QUALITY_MAX = 100

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, logger, host, port, device_name, device_num, device=None):
        logger.info(
            f"Initialize new instance of Seestar imager: {host}:{port}, name:{device_name}"
        )

        self.host = host
        self.port = port
        self.device_name = device_name
        self.device_num = device_num
        self.logger = logger
        # self.raw_img = None
        self.raw_img_size = [None, None]
        self.s = None
        self.is_connected = False
        self.is_streaming = False
        self.is_gazing = False
        self.is_live_viewing = False
        self.sent_subscription = False
        self.mode = None
        self.exposure_mode = None  # "stream"  # None | preview | stack | stream
        self.received_frame = 0
        self.sent_frame = 0
        self.last_frame = 0
        self.get_image_thread = None
        self.get_stream_thread = None
        self.heartbeat_msg_thread = None
        self.device = device
        self.lock = threading.RLock()
        self.eventbus = signal(f"{device_name}.eventbus")
        self.eventbus.connect(self.event_handler)
        self.BOUNDARY = b"\r\n--frame\r\n"
        # self.trace = MessageTrace(self.device_num, self.port, False)
        self.comm = SeestarImagerProtocol(
            logger=logger,
            device_name=device_name,
            device_num=device_num,
            host=host,
            port=port,
        )
        self.comm.start()

        # Star imaging metrics
        self.snr = None

        # Metrics
        self.last_stat_time = None
        self.last_stat_frames = None
        self.last_live_view_time = None
        self.last_stacking_frame = None

        # Video-throughput metering (bytes/s + frames/s over a rolling
        # window). Lock-free by construction: the frame-serve hot path
        # (get_frame) only appends to the deque and bumps counters, while
        # the stats reader (video_throughput) takes a defensive snapshot.
        # Each sample is (monotonic_ts, nbytes, is_new_frame); the Chromium
        # double-yield duplicate is recorded with is_new_frame=0 so frames/s
        # counts distinct frames while bytes/s reflects real wire bytes. The
        # maxlen bounds memory regardless of serve rate.
        self._serve_samples: deque = deque(maxlen=2048)
        self._serve_metering_start = None
        self.served_bytes_total = 0
        self.served_frames_total = 0

        # Stream-quality knobs (browser-hop only): max frames/s and JPEG
        # encode quality of the MJPEG stream served to the browser. Seeded
        # from the [stream_quality] config section (defaults preserve the
        # historical uncapped / quality-95 behavior); mutable at runtime via
        # the /api/<id>/stream_quality endpoint. The serve loop re-reads these
        # each iteration, so a change takes effect on the next frame.
        sq = self.default_stream_quality()
        self.max_stream_fps = sq["max_stream_fps"]
        self.jpeg_quality = sq["jpeg_quality"]

    def __repr__(self):
        return f"{type(self).__name__}(host={self.host}, port={self.port})"

    def event_handler(self, event):
        try:
            match event["Event"]:
                case "Stack":
                    stacked_frame = event["stacked_frame"] + event["dropped_frame"]
                    # xxx change to just stacked frame _or_ initial request?
                    if (
                        self.comm.is_connected()
                        and stacked_frame != self.last_stacking_frame
                        and stacked_frame > 0
                        and self.is_live_viewing
                    ):
                        self.logger.debug(
                            "Received Stack event.  Fetching stacked image"
                        )  # xxx trace
                        # If we get a stack event, we're going to assume we're stacking!
                        self.request_stacked_image()
                    self.last_stacking_frame = stacked_frame
                case _:
                    pass
        except:
            pass
        # print(f'Event handler: {event}')

    def request_stacked_image(self):
        with self.lock:
            self.comm.send_message('{"id": 23, "method": "get_stacked_img"}' + "\r\n")

    def blank_frame(self, message="Loading", timestamp=False):
        # load the gif image
        gif_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), Config.loading_gif
        )

        if message == "Loading":
            try:
                with open(gif_path, "rb") as gif_file:
                    gif_data = gif_file.read()

                    return b"Content-Type: image/gif\r\n\r\n" + gif_data + self.BOUNDARY
            except Exception:
                pass

        blank_image = np.ones((1920, 1080, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(
            blank_image,
            message,
            (200, 900),
            # (300, 1850),
            font,
            5,
            (128, 128, 128),
            4,
            cv2.LINE_8,
        )
        # image = cv2.imread('img/blank.jpg')
        if timestamp:
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]

            w = 1080
            h = 1920
            image = cv2.putText(
                np.copy(image),
                dt,  # f'{dt} {self.received_frame}',
                (int(w / 2 - 240), h - 70),
                font,
                1,
                (210, 210, 210),
                4,
                cv2.LINE_8,
            )
        imgencode = cv2.imencode(
            ".jpeg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        )[1]
        stringData = imgencode.tobytes()
        return b"Content-Type: image/jpeg\r\n\r\n" + stringData + self.BOUNDARY

    # render the template?
    # print("get_live_status:",  self.device.ra, self.device.dec)
    # status = f"RA: {self.device.ra} Dec: {self.device.dec}".encode('utf-8')
    # deprecated!
    def get_live_status(self):
        while True:
            self.update_live_status()
            # print(self.device.event_state)
            status = table(
                [["RA", "%.3f" % self.device.ra], ["Dec", "%.3f" % self.device.dec]]
            ).encode("utf-8")
            # status = "Testing..."
            frame = b"data: " + status + b"\n\n"
            yield frame
            sleep(5)

    def update_live_status(self):
        with self.lock:
            self.is_live_viewing = True
            self.last_live_view_time = int(time())

    def get_video_status(self):
        while True:
            status = f"Frame: {self.last_frame}".encode("utf-8")
            frame = b"data: " + status + b"\n\n"
            yield frame
            sleep(5)

    def is_working(self):
        view_state = self.device.view_state
        return view_state.get("state") == "working"

    def is_idle(self):
        return not self.is_working()

    def compare_set_exposure_mode(self) -> Optional[ExposureModes]:
        exposure_mode = None
        view_state = self.device.view_state
        # print("comparing exposure mode", view_state)
        # state = view_state.get("state")
        stage = view_state.get("stage")
        # mode = view_state.get('mode')
        # print(f"Compare And Set Exposure Mode {stage=} {self.exposure_mode=}")
        if self.is_idle():
            return None

        if stage == "RTSP":
            # if self.is_working():
            exposure_mode = "stream"
            # if self.exposure_mode != exposure_mode:
            #    self.start(exposure_mode)
        elif stage == "ContinuousExposure":
            exposure_mode = "preview"
            # if self.exposure_mode != exposure_mode:
            #    self.start(exposure_mode)
        elif stage == "Stack":
            # If stage is stack, leave exposure mode alone UNLESS exposure mode isn't set.
            # if self.exposure_mode is None and  the number of stacked exposures is > 2:
            exposure_mode = "stack"
            # if self.exposure_mode != exposure_mode:
            #    self.start(exposure_mode)

        # xxx what other exposure modes?
        return exposure_mode

    def build_frame_bytes(self, image: np.ndarray, width: int, height: int):
        font = cv2.FONT_HERSHEY_COMPLEX

        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
        # print("Emiting frame", dt)

        w = width or self.raw_img_size[0] or 1080
        h = height or self.raw_img_size[1] or 1920
        image = cv2.putText(
            np.copy(image),
            dt,  # f'{dt} {self.received_frame}',
            (int(w / 2 - 240), h - 70),
            font,
            1,
            (210, 210, 210),
            4,
            cv2.LINE_8,
        )
        imgencode = cv2.imencode(
            ".jpeg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        )[1]
        stringData = imgencode.tobytes()
        frame = b"Content-Type: image/jpeg\r\n\r\n" + stringData + self.BOUNDARY

        return frame

    def _meter_serve(self, nbytes: int, is_new_frame: bool) -> None:
        """Record bytes/frames served for throughput metering, aggregated
        across every concurrent ``/vid`` viewer of this telescope.

        Called on the frame-serve hot path (``get_frame``). Lock-free and
        best-effort: with multiple concurrent viewers the unsynchronized
        ``+=`` bumps can occasionally under-count — acceptable for a stats
        readout, and the alternative (a lock on the hot path) is not.
        ``is_new_frame`` is False for the Chromium double-yield duplicate so
        frames/s counts distinct frames while bytes/s reflects the real bytes
        pushed onto the wire.
        """
        now = monotonic()
        if self._serve_metering_start is None:
            self._serve_metering_start = now
        self._serve_samples.append((now, nbytes, 1 if is_new_frame else 0))
        self.served_bytes_total += nbytes
        if is_new_frame:
            self.served_frames_total += 1

    def _snapshot_serve_samples(self):
        """Lock-free read of the producer's sample deque.

        A concurrent append (mutation) can invalidate an in-progress
        iteration (``RuntimeError: deque mutated during iteration``); retry a
        few times, then give up with an empty snapshot rather than block the
        serve loop or raise into the stats endpoint.
        """
        for _ in range(5):
            try:
                return list(self._serve_samples)
            except RuntimeError:
                continue
        return []

    def video_throughput(self, window=None):
        """Rolling bytes/s + frames/s served to the browser.

        Averages over the last ``window`` seconds (default
        :attr:`THROUGHPUT_WINDOW_S`). During warm-up (before a full window of
        data exists) and while the stream is winding down, the denominator is
        clamped to the elapsed metering time so the reported rate neither
        spikes on the first frame nor understates a short-lived stream.
        """
        window = float(window or self.THROUGHPUT_WINDOW_S)
        now = monotonic()
        cutoff = now - window
        win_bytes = 0
        win_frames = 0
        for ts, nbytes, is_frame in self._snapshot_serve_samples():
            if ts >= cutoff:
                win_bytes += nbytes
                win_frames += is_frame
        if self._serve_metering_start is None:
            denom = window
        else:
            denom = min(window, max(now - self._serve_metering_start, 1.0))
        return {
            "window_s": window,
            "bytes_per_s": win_bytes / denom,
            "frames_per_s": win_frames / denom,
            "bytes_total": self.served_bytes_total,
            "frames_total": self.served_frames_total,
        }

    @staticmethod
    def _clamp_max_stream_fps(value):
        """Coerce a max-serve-fps value to a float in [0, MAX_STREAM_FPS_CAP].

        ``0`` (and any non-positive or NaN input) means *uncapped*. Raises
        ``TypeError``/``ValueError`` on values that aren't numbers so the
        settings endpoint can reject them with a 400.
        """
        fps = float(value)
        if fps != fps or fps <= 0:  # NaN or non-positive -> uncapped
            return 0.0
        return min(SeestarImaging.MAX_STREAM_FPS_CAP, fps)

    @staticmethod
    def _clamp_jpeg_quality(value):
        """Coerce a JPEG-quality value to an int in [JPEG_QUALITY_MIN, MAX].

        Raises ``TypeError``/``ValueError`` on non-numeric input so the
        settings endpoint can reject it with a 400.
        """
        q = int(round(float(value)))
        return max(
            SeestarImaging.JPEG_QUALITY_MIN,
            min(SeestarImaging.JPEG_QUALITY_MAX, q),
        )

    @classmethod
    def default_stream_quality(cls):
        """Stream-quality defaults from the ``[stream_quality]`` config section.

        Tolerant of a missing section or garbage values (falls back to the
        class defaults) so a bad config never blocks imager start-up.
        """
        try:
            fps = cls._clamp_max_stream_fps(
                Config.get_toml(
                    "stream_quality", "max_stream_fps", cls.MAX_STREAM_FPS_DEFAULT
                )
            )
        except (TypeError, ValueError):
            fps = float(cls.MAX_STREAM_FPS_DEFAULT)
        try:
            quality = cls._clamp_jpeg_quality(
                Config.get_toml(
                    "stream_quality", "jpeg_quality", cls.JPEG_QUALITY_DEFAULT
                )
            )
        except (TypeError, ValueError):
            quality = int(cls.JPEG_QUALITY_DEFAULT)
        return {"max_stream_fps": fps, "jpeg_quality": quality}

    def stream_quality(self):
        """Current runtime stream-quality knob values for this imager."""
        return {
            "max_stream_fps": self.max_stream_fps,
            "jpeg_quality": self.jpeg_quality,
        }

    def set_stream_quality(self, max_stream_fps=None, jpeg_quality=None):
        """Apply new (clamped) stream-quality knob values to this imager.

        Either field may be omitted (``None``) to leave it unchanged. Takes
        effect on the next served frame — the serve loop re-reads
        ``self.max_stream_fps`` / ``self.jpeg_quality`` each iteration.
        """
        if max_stream_fps is not None:
            self.max_stream_fps = self._clamp_max_stream_fps(max_stream_fps)
        if jpeg_quality is not None:
            self.jpeg_quality = self._clamp_jpeg_quality(jpeg_quality)
        return self.stream_quality()

    def _serve_delay(self, base_delay):
        """Inter-frame serve delay honoring the ``max_stream_fps`` cap.

        ``base_delay`` is the mode's natural cadence (0.001 s streaming,
        0.1 s preview). When ``max_stream_fps`` is set we never sleep *less*
        than ``1 / fps`` between served frames, throttling the browser-hop
        rate. A 0 cap returns ``base_delay`` unchanged (historical behavior).
        """
        fps = self.max_stream_fps
        if fps and fps > 0:
            return max(base_delay, 1.0 / fps)
        return base_delay

    def get_frame(self):
        # xxx : We want to be able to manually switch between preview and stack modes.
        #       If stage is RTSP, we force switch to stream exposure mode.
        # .      If stage is Stack, and exposure mode preview, leave it alone.
        # .      If stage is ContinuousExposure, switch to preview.
        # todo : don't send these if we already have an image and we have an exposure mode
        #
        # We send each frame twice because of a very long term bug in Chromium.  Yes, seriously.
        #   We will only send it when not in RTSP-backed modes.  (The idea being that with
        #   higher FPS being one frame behind isn't noticeable.)
        #
        # Some of the related issues:
        # - https://issues.chromium.org/issues/40791855 "multipart/x-mixed-replace images have 1 frame delay" from 2021
        # - https://issues.chromium.org/issues/41199053 "mjpeg image always shows the second to last image" from 2015
        # - https://issues.chromium.org/issues/40277613 "multipart/x-mixed-replace no longer working reliably" from 2012!
        yield b"\r\n--frame\r\n"
        image, width, height = self.comm.get_image()
        if image is not None:
            # image, _, _ = self.get_image(self.exposure_mode)
            frame = self.build_frame_bytes(image, width, height)
            self._meter_serve(len(frame), True)
            yield frame
            self._meter_serve(len(frame), False)
            yield frame
        else:
            yield self.blank_frame("Loading", True)
            yield self.blank_frame("Loading", True)

        # view_state = self.device.view_state
        # self.logger.info(f"mode: {self.mode} {type(self.mode)} view_state: {view_state}")

        exiting = False
        first_image = False
        while not self.is_idle():
            self.comm.set_exposure_mode(self.compare_set_exposure_mode())
            image, width, height = self.comm.get_image()

            if self.comm.is_streaming():
                delay = 0.001
                snr = -1
            else:
                raw_image, _, _ = self.comm.get_unprocessed_image()
                delay = 0.1
                snr = SNRAnalysis().analyze(raw_image)
            # Honor the max_stream_fps browser-hop cap (no-op when 0).
            delay = self._serve_delay(delay)

            if image is not None:
                # print("get_frame image!")
                try:
                    received_frame = self.comm.received_frame()
                    if self.last_frame != received_frame:
                        frame = self.build_frame_bytes(image, width, height)
                        # print("sending frame bytes=", len(stringData))

                        # Update stats!
                        self.sent_frame += 1

                        now = int(time())
                        if self.last_stat_time != now:
                            if (
                                self.last_stat_time is not None
                                and self.last_stat_frames is not None
                                and self.last_stat_frames is not None
                            ):
                                elapsed = now - self.last_stat_time
                                frames = self.sent_frame - self.last_stat_frames
                                self.logger.debug(
                                    f"Sent frames: {frames} in {elapsed} seconds.  FPS: {frames / elapsed}.  Received frame total: {self.received_frame}"
                                )

                            self.last_stat_time = now
                            self.last_stat_frames = self.sent_frame

                        self.last_frame = received_frame
                        self.snr = snr

                        first_image = True
                        # ts = time()
                        self._meter_serve(len(frame), True)
                        yield frame
                        # te = time()
                        # print(f'imaging yield1 took {te - ts:2.4f} seconds')
                        if not self.comm.is_streaming():
                            # ts = time()
                            self._meter_serve(len(frame), False)
                            yield frame
                            # te = time()
                            # print(f'imaging yield2 took {te - ts:2.4f} seconds')
                        # if not self.is_gazing:
                        #    yield frame
                    else:
                        pass
                        # self.logger.info("skipping send")
                except GeneratorExit:
                    # with self.lock:
                    #     self.raw_img = None
                    #     self.raw_img_size = [None, None]
                    exiting = True
                    break
                except Exception as e:
                    # print(traceback.format_exc())
                    self.logger.info(f"exception encoding frame. skipping {e=}")

                    # with self.lock:
                    #    self.raw_img = None
                    #    self.raw_img_size = [None, None]
            else:
                # print("Did not get frame!")
                if not first_image:
                    yield self.blank_frame("Loading", True)
                    yield self.blank_frame("Loading", True)
            sleep(delay)

        self.comm.set_exposure_mode(self.compare_set_exposure_mode())

        if not exiting:
            yield self.blank_frame("Idle")


if __name__ == "__main__":
    app = Flask(__name__)

    host, port, device_num, listen_port = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
    )
    logger = log.init_logging()
    imager = SeestarImaging(logger, host, port, "SeestarB", device_num)

    @app.route("/vid/<mode>")
    def vid(mode):
        return Response(
            imager.get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    app.run(host="localhost", port=listen_port, debug=True)  # , threaded=True)
