# Base V8 sealed-action forensic report

Date: 2026-08-13

Scope: static AST/source inspection and retained lifecycle-record inspection only. No transport, launcher, controller, evaluator, or C02 parser was invoked. The sealed tensor was not opened, read, hashed, mapped, or copied.

The permitted current `lstat` observation was performed on the exact `SEALED_TENSOR` path frozen into the retained V8 launcher: `/home/argustest/ace-2/reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin`. It found a regular 1,305,797-byte file with mode `0400`, matching the retained launcher's preflight expectation. No content access was performed, and this metadata observation does not identify the V8 exception.

## Sealed evidence

- Retired action: `ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z`
- V8 authority file SHA-256: `248a308dda3813c3593c26ec528278ae9e56619933dc00083e821e5a2947479e`
- V8 authority canonical hash: `d745038d194b4be1e4bd7643344e402875da4d4e510d1a0ea64cccbadaeebf48`
- V8 ledger file SHA-256: `b566f7f48b8c613f59ce6fcc215046038a9358fa547313c6a635b90802d5481d`
- V8 ledger canonical hash: `ec2f49c233e1e66fa6c49ba1fe39c26d83bdee0eab9055b3dd6528ee7a32db22`
- V8 terminal file SHA-256: `ba92fd9eedbd39662325ad73ad1a85d34f59a901840cd38e67bcb95531630143`
- V8 terminal canonical hash: `9f30c0e28e488bc0d90db1ae2fcfafe754e882f03f7e5ef4c27105a4d604ae65`
- V8 terminal: `CONSUMED_ORPHAN/POST_CONSUMPTION_AMBIGUITY`
- V8 failure-detail SHA-256: `7fe1602a1cd89139fc712070f5961dce30d1bc5be2f13153130498de18661373`
- V8 counters: `invocation_count=0`, `payload_open_count=0`, `evaluator_calls=0`
- V8 credential and result: absent
- Prior V7 ledger file SHA-256: `0b4485abb861a8aeda82a283ecab7814045599e49a7d7e5256eea7a7cc6d0708`
- Prior V7 terminal file SHA-256: `5d6243a9fbd830334829bd96a63a0be7da965a22a55cab93a6500f2a0463c0c8`

## Static localization

AST inspection of the retained V8 launcher proves this successful-prefix order in `main`: publish authority, publish credential, publish consumed ledger, set `ledger_published = True`, call `_durable_unlink(credential)`, and after that call returns enter `_run_consumed`.

The retained credential is absent, proving that `_durable_unlink`'s `os.unlink(path)` side effect took effect. The next possible failure is its parent-directory `fsync`. If that durability tail completed, control entered `_run_consumed`, where the invocation counter remained zero until after controller/evaluator/C02 module loading, accepted package and result-schema reads, controller package validation, tensor-binding construction, and result-context construction. Either exception path publishes the observed `CONSUMED_ORPHAN/POST_CONSUMPTION_AMBIGUITY` shape with zero invocation and payload counters. The retained records do not distinguish those paths.

## Evidence-bound conclusion

The narrowest proven failure phase is after durable ledger publication and after credential unlink took effect, but before the retained launcher incremented its invocation counter, invoked the evaluator, or opened the payload. This unresolved interval includes either the credential parent-directory durability tail or pre-evaluator module/package validation and result-context work inside `_run_consumed`; it cannot be localized further from retained evidence.

No retained artifact exposes the exception text. The failure-detail value is only a SHA-256 digest, and no exact text has been proven to hash to it. Therefore this report does not name or guess an exception.

## V9 design consequence

Action `ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z` must complete every avoidable fallible load, dependency check, callable check, package/schema read and canonical validation, controller package validation, result-validator construction, and runtime-binding freeze during read-only preflight. After consumed-ledger publication, only credential durable unlink and the irreducible payload/evaluator/result/terminal sequence may remain. Any exception after ledger publication permanently seals `CONSUMED_ORPHAN`; no retry, replay, resume, repair, or replacement is permitted.

This static report grants no execution authority and is not a Fresh-L2 acceptance record.
