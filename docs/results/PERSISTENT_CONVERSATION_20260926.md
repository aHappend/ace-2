# Persistent conversation checkpoint - September 26, 2026

**The three-turn RTL diagnostic is still running, not finally qualified.**
Mission `04bdbc62467f`, namespace `0008`, had reached its third turn at the
snapshot. No final result or final independent review existed. Exact observation
time and latest output path are in
[progress-2026-09-26.json](../continuation/progress-2026-09-26.json).

The last accepted milestone remains two turns with four generated tokens per
turn: eight full-vocabulary heads, 840 layer checks across 35 input positions,
24 retained KV layers, no prefix recomputation, no fallback and zero recorded
integer/logit/token mismatches. Input positions are not generated-token counts.
Its original result and review hashes remain in the progress summary.

This publication adds only the changed persistent-conversation driver and its
focused tests, with hashes in the
[incremental source manifest](../continuation/source-manifest-2026-09-26.json).
They are **WIP source**, not a claim that the current three-turn experiment
passed. Existing source and prior failures remain preserved.

Publication uses a separate clone and does not restart the diagnostic, mutate
its working tree/index, or copy its live state. Raw simulation output, models,
credentials and active claims remain host-only. This is local RTL simulation
and host/oracle work, not FPGA, silicon or hardware performance evidence.
