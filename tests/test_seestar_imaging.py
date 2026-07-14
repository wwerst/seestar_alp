import numpy as np

from device import seestar_imaging


class DummyLogger:
    def __init__(self):
        self.records = []

    def info(self, msg):
        self.records.append(("info", msg))

    def debug(self, msg):
        self.records.append(("debug", msg))

    def error(self, msg):
        self.records.append(("error", msg))


class DummyDevice:
    def __init__(self):
        self.view_state = {"state": "working", "stage": "RTSP"}
        self.ra = 10.123
        self.dec = 20.456


class FakeComm:
    def __init__(self, **_kwargs):
        self._connected = True
        self.exposure_mode = None
        self.set_modes = []
        self.sent = []
        self._streaming = False

    def start(self):
        return None

    def is_connected(self):
        return self._connected

    def send_message(self, msg):
        self.sent.append(msg)
        return True

    def set_exposure_mode(self, mode):
        self.exposure_mode = mode
        self.set_modes.append(mode)

    def get_image(self):
        return np.ones((2, 2, 3), dtype=np.uint8), 2, 2

    def get_unprocessed_image(self):
        return np.ones((2, 2, 3), dtype=np.uint8), 2, 2

    def is_streaming(self):
        return self._streaming

    def received_frame(self):
        return 1


def make_imager(monkeypatch):
    monkeypatch.setattr(seestar_imaging, "SeestarImagerProtocol", FakeComm)
    return seestar_imaging.SeestarImaging(
        logger=DummyLogger(),
        host="127.0.0.1",
        port=9999,
        device_name="scope",
        device_num=1,
        device=DummyDevice(),
    )


def test_table_renders_rows():
    html = seestar_imaging.table([["A", "1"], ["B", "2"]])
    assert 'class="row"' in html
    assert "A" in html and "B" in html


def test_compare_set_exposure_mode_paths(monkeypatch):
    imager = make_imager(monkeypatch)

    imager.device.view_state = {"state": "idle", "stage": "RTSP"}
    assert imager.compare_set_exposure_mode() is None

    imager.device.view_state = {"state": "working", "stage": "RTSP"}
    assert imager.compare_set_exposure_mode() == "stream"

    imager.device.view_state = {"state": "working", "stage": "ContinuousExposure"}
    assert imager.compare_set_exposure_mode() == "preview"

    imager.device.view_state = {"state": "working", "stage": "Stack"}
    assert imager.compare_set_exposure_mode() == "stack"


def test_event_handler_stack_requests_stacked_image(monkeypatch):
    imager = make_imager(monkeypatch)
    imager.is_live_viewing = True
    imager.last_stacking_frame = None

    called = {"stack": 0}
    monkeypatch.setattr(
        imager,
        "request_stacked_image",
        lambda: called.__setitem__("stack", called["stack"] + 1),
    )

    imager.event_handler({"Event": "Stack", "stacked_frame": 1, "dropped_frame": 1})
    assert called["stack"] == 1
    assert imager.last_stacking_frame == 2

    imager.event_handler({"Event": "Unknown", "stacked_frame": 5, "dropped_frame": 0})
    assert called["stack"] == 1


def test_request_stacked_image_emits_protocol_message(monkeypatch):
    imager = make_imager(monkeypatch)

    imager.request_stacked_image()

    assert imager.comm.sent == ['{"id": 23, "method": "get_stacked_img"}\r\n']


def test_blank_frame_and_build_frame_bytes(monkeypatch):
    imager = make_imager(monkeypatch)

    # Force fallback image path for blank frame
    monkeypatch.setattr(
        seestar_imaging,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
        raising=False,
    )
    monkeypatch.setattr(
        seestar_imaging.cv2, "putText", lambda image, *_args, **_kwargs: image
    )

    class Encoded:
        def tobytes(self):
            return b"jpeg"

    monkeypatch.setattr(
        seestar_imaging.cv2, "imencode", lambda *_args, **_kwargs: (True, Encoded())
    )

    frame = imager.blank_frame("Loading", timestamp=True)
    assert frame.startswith(b"Content-Type: image/jpeg")

    img = np.ones((2, 2, 3), dtype=np.uint8)
    frame2 = imager.build_frame_bytes(img, 2, 2)
    assert frame2.startswith(b"Content-Type: image/jpeg")


def test_get_frame_yields_initial_and_idle_frames(monkeypatch):
    imager = make_imager(monkeypatch)

    monkeypatch.setattr(imager, "build_frame_bytes", lambda *_args, **_kwargs: b"FRAME")
    monkeypatch.setattr(
        imager,
        "blank_frame",
        lambda message="", timestamp=False: f"BLANK:{message}:{timestamp}".encode(),
    )

    imager.device.view_state = {"state": "idle", "stage": "RTSP"}

    frames = list(imager.get_frame())

    assert frames[0] == b"\r\n--frame\r\n"
    assert frames[1] == b"FRAME"
    assert frames[2] == b"FRAME"
    assert frames[-1] == b"BLANK:Idle:False"


def test_get_live_and_video_status_generators(monkeypatch):
    imager = make_imager(monkeypatch)

    monkeypatch.setattr(seestar_imaging, "sleep", lambda _s: None)

    live = imager.get_live_status()
    live_frame = next(live)
    assert live_frame.startswith(b"data: ")

    video = imager.get_video_status()
    video_frame = next(video)
    assert video_frame.startswith(b"data: Frame:")


def test_repr_and_event_handler_exception_path(monkeypatch):
    imager = make_imager(monkeypatch)
    assert repr(imager) == "SeestarImaging(host=127.0.0.1, port=9999)"

    # Missing keys raises internally but should be swallowed.
    imager.event_handler({"Event": "Stack"})


def test_blank_frame_uses_gif_when_available(monkeypatch):
    imager = make_imager(monkeypatch)

    class DummyFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"GIF89a"

    monkeypatch.setattr(
        seestar_imaging, "open", lambda *_a, **_k: DummyFile(), raising=False
    )
    frame = imager.blank_frame("Loading", timestamp=False)
    assert frame.startswith(b"Content-Type: image/gif")


def test_get_frame_loop_streaming_branch(monkeypatch):
    imager = make_imager(monkeypatch)
    monkeypatch.setattr(seestar_imaging, "sleep", lambda _s: None)
    monkeypatch.setattr(imager, "build_frame_bytes", lambda *_a, **_k: b"FRAME")
    monkeypatch.setattr(
        imager,
        "blank_frame",
        lambda message="", timestamp=False: f"B:{message}:{timestamp}".encode(),
    )
    monkeypatch.setattr(seestar_imaging.SNRAnalysis, "analyze", lambda self, _img: 42)

    state = {"count": 0}

    def fake_idle():
        state["count"] += 1
        return state["count"] > 1

    monkeypatch.setattr(imager, "is_idle", fake_idle)
    monkeypatch.setattr(imager, "compare_set_exposure_mode", lambda: "stream")
    imager.comm._streaming = True
    imager.comm.received_frame = lambda: 2

    frames = list(imager.get_frame())
    assert frames[0] == b"\r\n--frame\r\n"
    assert b"FRAME" in frames
    assert imager.snr == -1


def test_get_frame_loop_non_streaming_and_stats(monkeypatch):
    imager = make_imager(monkeypatch)
    monkeypatch.setattr(seestar_imaging, "sleep", lambda _s: None)
    monkeypatch.setattr(imager, "build_frame_bytes", lambda *_a, **_k: b"F2")
    monkeypatch.setattr(
        imager,
        "blank_frame",
        lambda message="", timestamp=False: f"B:{message}:{timestamp}".encode(),
    )
    monkeypatch.setattr(seestar_imaging.SNRAnalysis, "analyze", lambda self, _img: 77)

    times = iter([10, 11, 11, 12])
    monkeypatch.setattr(seestar_imaging, "time", lambda: next(times))

    state = {"count": 0}

    def fake_idle():
        state["count"] += 1
        return state["count"] > 2

    monkeypatch.setattr(imager, "is_idle", fake_idle)
    monkeypatch.setattr(imager, "compare_set_exposure_mode", lambda: "preview")
    imager.comm._streaming = False
    imager.comm.received_frame = lambda: 3 if state["count"] == 1 else 3

    frames = list(imager.get_frame())
    assert frames.count(b"F2") >= 2
    assert imager.snr == 77


def test_meter_serve_counts_bytes_and_distinct_frames(monkeypatch):
    imager = make_imager(monkeypatch)

    imager._meter_serve(100, True)  # distinct frame
    imager._meter_serve(100, False)  # Chromium double-yield: bytes only
    imager._meter_serve(50, True)  # another distinct frame

    assert imager.served_bytes_total == 250
    assert imager.served_frames_total == 2

    tp = imager.video_throughput()
    assert tp["bytes_total"] == 250
    assert tp["frames_total"] == 2
    # frames/s counts distinct frames (2), bytes/s counts all wire bytes (250),
    # so the byte rate must exceed the frame rate for this sample set.
    assert tp["bytes_per_s"] > tp["frames_per_s"]
    assert tp["window_s"] == seestar_imaging.SeestarImaging.THROUGHPUT_WINDOW_S


def test_video_throughput_empty_is_zeroed(monkeypatch):
    imager = make_imager(monkeypatch)
    tp = imager.video_throughput()
    assert tp["bytes_per_s"] == 0
    assert tp["frames_per_s"] == 0
    assert tp["bytes_total"] == 0
    assert tp["frames_total"] == 0
    assert tp["window_s"] == 10.0


def test_video_throughput_window_excludes_old_samples(monkeypatch):
    imager = make_imager(monkeypatch)
    clock = {"t": 1000.0}
    monkeypatch.setattr(seestar_imaging, "monotonic", lambda: clock["t"])

    imager._meter_serve(1000, True)  # first sample at t=1000
    clock["t"] = 1000.0 + 30.0  # advance 30s (well past the 10s window)
    imager._meter_serve(500, True)  # only this one is inside the window

    tp = imager.video_throughput(window=10.0)
    # denom clamps to min(window, elapsed-since-first-sample) = 10s.
    assert tp["bytes_per_s"] == 500 / 10.0
    assert tp["frames_per_s"] == 1 / 10.0
    # Cumulative totals still include the evicted-from-window sample.
    assert tp["bytes_total"] == 1500
    assert tp["frames_total"] == 2


def test_video_throughput_no_startup_spike(monkeypatch):
    imager = make_imager(monkeypatch)
    clock = {"t": 500.0}
    monkeypatch.setattr(seestar_imaging, "monotonic", lambda: clock["t"])
    # A single large frame recorded within the first second must not divide
    # by a near-zero denominator and report an absurd rate.
    imager._meter_serve(1_000_000, True)
    tp = imager.video_throughput(window=10.0)
    assert tp["bytes_per_s"] == 1_000_000 / 1.0  # denom floored to 1.0s


def test_snapshot_serve_samples_survives_mutation_error(monkeypatch):
    imager = make_imager(monkeypatch)

    class Boom:
        def __iter__(self):
            raise RuntimeError("deque mutated during iteration")

    imager._serve_samples = Boom()
    assert imager._snapshot_serve_samples() == []
    # video_throughput must still return a well-formed zeroed reading.
    tp = imager.video_throughput()
    assert tp["bytes_per_s"] == 0
    assert tp["frames_per_s"] == 0


def test_get_frame_records_throughput(monkeypatch):
    imager = make_imager(monkeypatch)
    monkeypatch.setattr(seestar_imaging, "sleep", lambda _s: None)
    monkeypatch.setattr(imager, "build_frame_bytes", lambda *_a, **_k: b"FRAMEBYTES")
    monkeypatch.setattr(
        imager, "blank_frame", lambda message="", timestamp=False: b"BL"
    )
    # Idle immediately so only the initial frame block runs (image present).
    imager.device.view_state = {"state": "idle", "stage": "RTSP"}

    list(imager.get_frame())

    # The initial content frame is metered once as a distinct frame plus the
    # Chromium duplicate, so bytes cover both copies.
    assert imager.served_frames_total >= 1
    assert imager.served_bytes_total >= len(b"FRAMEBYTES")
    assert imager.video_throughput()["frames_total"] >= 1


def test_get_frame_handles_encode_exception_and_loading(monkeypatch):
    imager = make_imager(monkeypatch)
    monkeypatch.setattr(seestar_imaging, "sleep", lambda _s: None)
    monkeypatch.setattr(
        imager, "blank_frame", lambda message="", timestamp=False: b"BL"
    )

    # Initial image missing -> loading blanks
    imager.comm.get_image = lambda: (None, None, None)
    monkeypatch.setattr(imager, "is_idle", lambda: True)
    frames = list(imager.get_frame())
    assert frames[1] == b"BL"
    assert frames[2] == b"BL"

    # Loop image present but frame encoding fails -> handled by exception path
    state = {"count": 0}

    def fake_idle():
        state["count"] += 1
        return state["count"] > 1

    calls = {"n": 0}

    def seq_image():
        calls["n"] += 1
        if calls["n"] == 1:
            return (None, None, None)
        return (np.ones((2, 2, 3), dtype=np.uint8), 2, 2)

    imager.comm.get_image = seq_image
    imager.comm.get_unprocessed_image = lambda: (
        np.ones((2, 2, 3), dtype=np.uint8),
        2,
        2,
    )
    imager.comm.received_frame = lambda: 10
    imager.comm._streaming = False
    monkeypatch.setattr(imager, "is_idle", fake_idle)
    monkeypatch.setattr(imager, "compare_set_exposure_mode", lambda: "preview")
    monkeypatch.setattr(
        imager,
        "build_frame_bytes",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("encode")),
    )
    monkeypatch.setattr(seestar_imaging.SNRAnalysis, "analyze", lambda self, _img: 3)
    list(imager.get_frame())
