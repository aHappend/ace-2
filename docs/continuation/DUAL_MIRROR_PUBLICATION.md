# Equal dual-mirror publication

Personal and team repositories are equal mirrors. There is no primary or
secondary. Every release must land as the same per-project commit and tree on:

- `https://github.com/aHappend/ace-2`
- `https://github.com/Argus-AiTeam/ace-2`

## Identity and safety

Use the user-supplied credential externally, per command:

```bash
GH_CONFIG_DIR=/home/argustest/argustest2/.gh-config gh api user
```

The login must be `aHappend`. Never print/copy the credential file, put a token
in a URL/script/log, change the global gh profile, or conflate GitHub identity
with local Copilot state.

Never prepare a publication in the dirty scientific worktree. Do not switch its
branch, reset, stash, clean, or stage its mixed index. Create an isolated clone
or worktree, fetch both current `main` tips, and preserve both histories. ACE-2
tips may have different commits even when their trees match; merge normally
rather than rewriting either history.

## Repeatable procedure

1. Fetch both tips and record their commit and tree IDs. Stop on unexpected
   history or protection instead of force-pushing.
2. Build an explicit curated manifest from a coherent read of the live source.
   Hash selected live files before and after copying; retry if any changed.
3. Include implementations, RTL, host/oracle code, tests, contracts, scripts,
   build/documentation dependencies and essential small fixtures. Clearly label
   current WIP as unaccepted.
4. Exclude credentials, model weights without approved licensing/storage, raw
   provider/operator transcripts, active locks/PIDs/claims, caches, build trees,
   large raw evidence and generated bulk that is not operational source.
5. Update README/AGENTS/handoff/progress/host-only inventory. Sanitized summaries
   must cite original hashes and must not masquerade as authoritative receipts.
6. Independently review links, source hashes, default read-only continuation
   behavior, current-state claims, sensitive strings, oversized files and the
   staged diff. Do not rerun one-shot science or expensive old validations.
7. Commit normally, preserving divergent parents if needed. Add a dated
   checkpoint ref. Never force-push or delete remote history.
8. Push the exact same commit and checkpoint ref to both repositories. Pushes are
   sequential and non-transactional: after each push, query the remote SHA/tree.
9. If one push fails, record **partial sync**, leave successful history intact,
   correct the normal failure, and finish the missing mirror. Do not claim dual
   success until both remote commit and tree IDs are identical.

The publication credential is not a backup dependency of this repository.
