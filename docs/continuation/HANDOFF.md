# ACE-2 continuation handoff

Snapshot date: **2026-09-26**. See
[progress-2026-09-26.json](progress-2026-09-26.json) for exact observation time.
Live state may advance. The operational guide is
[HANDOFF.zh-CN.md](HANDOFF.zh-CN.md).

The equal mirrors are <https://github.com/aHappend/ace-2> and
<https://github.com/Argus-AiTeam/ace-2>, branch `main`, checkpoint ref
`refs/tags/checkpoint-2026-09-26-source`. Publish as `aHappend` to both, then
verify identical commit/tree IDs.

The reviewed two-turn, four-generated-tokens-per-turn result remains the
accepted milestone. Three-turn mission `04bdbc62467f`, diagnostic namespace
`0008`, was still running at capture; no final result or final review existed.
The two-file driver/test increment is explicitly WIP, not a qualification claim.
See [PERSISTENT_CONVERSATION_20260926.md](../results/PERSISTENT_CONVERSATION_20260926.md).

On the original host, run `python tools/argus_continuation.py inspect` first.
Discover actual runtime, active job and output rather than restarting the old
`0006` or `0007` namespaces. Do not interrupt a live job to publish a snapshot.
Normal independent review and native completion are required before describing
the three-turn run as accepted. Input positions are not generated-token counts.

This is computer-local RTL and host/oracle work, not FPGA, silicon or hardware
performance measurement. See [HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md):
the repository restores curated source, not full models, raw results, credentials,
live claims or scheduled supervision.
