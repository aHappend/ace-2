# Base V9 canonical-byte preflight failure and V10 additive repair

Date: 2026-08-14

The retired action `ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z` is sealed by its immutable
`PREFLIGHT_FAILED_TERMINAL/READ_ONLY_PREFLIGHT_FAILED` record.  Its reproduced
failure detail is `ExecutionError:accepted V8 package: noncanonical bytes`.
No controller, evaluator, C02 parser, tensor payload, or G8/G4/G2/G1 run
occurred.  V9 must never be retried, replayed, resumed, repaired, or replaced
in place.

Measured byte cause:

- accepted V8 package predecessor: 11345 bytes, SHA-256 `b7c758c7141d7ac8037502da4a00fe9e0780d30501e2af483e9b8c571451493a`;
- exact production-canonical package mirror: 9109 bytes, SHA-256 `3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832`;
- accepted V8 result-schema predecessor: 21047 bytes, SHA-256 `fc5cf29d7ec7486b106edac592787b5888557c4b1b7453cb0a668b02fca113ec`;
- exact production-canonical result-schema mirror: 16502 bytes, SHA-256 `07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720`.

The predecessor objects are semantically equal to their mirrors.  Only JSON
serialization changes: sorted keys, compact separators, strict ASCII, and one
terminal newline.  All predecessor roots and terminal evidence remain
byte-unchanged.

The additive successor action is `ace2:qk-gbfp8-base-v10:execute-once:f34a97cc:20260814T075052Z`.  Its production launcher keeps
the exact V9 canonical reader and points the accepted package/schema bindings
to the new canonical mirrors.  The independent verifier extracts those exact
reader function bodies by AST, runs the exact file read on both bound mirrors,
checks the read-only-preflight call occurs before any durable live-state create,
and rejects adversarial noncanonical encodings.  Static review grants no
execution authority; a later separately authorized mission is required for any
transport or launcher invocation.
