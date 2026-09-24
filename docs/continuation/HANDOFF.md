# ACE-2 continuation handoff

Source snapshot: **2026-09-24T16:50:59Z**; live state observed at
**2026-09-24T16:58:55Z**. Live state may have advanced. The canonical
operator guide is [HANDOFF.zh-CN.md](HANDOFF.zh-CN.md).

The equal mirrors are <https://github.com/aHappend/ace-2> and
<https://github.com/Argus-AiTeam/ace-2>, branch `main`. The dated source ref is
`refs/tags/checkpoint-2026-09-24-source` at
`da5f71ab4160344db3680e2865648ca199796f8c`.

On the original host, use `python tools/argus_continuation.py inspect` before
acting. Do not restart a live daemon or duplicate an active mission. Continue
through the normal Manager; verify issuer, normal Reviewer, and native terminal
receipts. For disk loss, this repository restores curated source and
documentation only; see [HOST_ONLY_DEPENDENCIES.md](HOST_ONLY_DEPENDENCIES.md).
