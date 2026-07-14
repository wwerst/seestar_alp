@../AGENTS.md

## Agent Model Policy
- All agents and subagents (both foreground and background) must run on Claude Opus (preferred, pass `model: 'opus'`) or Claude Sonnet — no other models — unless the user explicitly requests a different model.
- Ultracode/multi-agent workflow runs must spawn at most 20 agents total per run (across all phases, including verification fan-outs) unless the user explicitly requests a higher ceiling. Prefer fewer, stronger agents over wide fan-outs.
