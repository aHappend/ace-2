# V13 runtime-namespace failure and V15 additive successor

Date: 2026-08-14

The sole V13 action `ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z` was invoked once and returned 2.  The
immutable mode-0555 package root prevented both package-local live-state
creation and fallback first-terminal publication.  The recorded invocation
used `/bin/bash -c`, not the accepted shell-free argv contract.  No immutable
V13 terminal exists, no sealed tensor was opened, and no G8/G4/G2/G1 evaluator
run occurred.  V13 is permanently retired and remains byte-unchanged.

The additive V15 action is `ace2:qk-gbfp8-base-v15:execute-once:c2dfe170:20260814T084500Z`.  Its package root is immutable and
contains no live directory.  All lifecycle paths are instead bound to the
separately named `/home/argustest/ace-2/runtime/qk_gbfp8_head64_granularity_sweep_execution_v15_c2dfe170` tree.  The tree is precreated empty with mode
0700 and owner `1000:1000`.  Production preflight and the
independent verifier both check every parent with lstat, reject symlinks or
owner/mode drift, prove O_EXCL create-once behavior in the primary and fallback
terminal parents, remove the probes, and require all lifecycle files absent
before authority publication.

The future invocation is represented only as exact argv, cwd, explicit
environment, `os.posix_spawn`, and `shell=false`.  This static package creates
no execution authority.  Fresh-L2 static acceptance is required before any
separately authorized later execution mission.

The preserved V14 pre-review candidate is not an execution attempt and did not
create acceptance or authority.  Its inert verifier rejected the package after
finding a missing digest cross-link; the V15 verifier adds that cross-link and
binds the V14 failure record without modifying the V14 bytes.
