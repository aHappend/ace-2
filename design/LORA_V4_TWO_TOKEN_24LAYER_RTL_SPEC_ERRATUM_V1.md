# LoRA V4 two-token 24-layer RTL specification cycle erratum V1

- Erratum version: `1.0`
- Effective contract revision: `2.1`
- Published: `2026-08-11T21:00:34Z`
- Status: `NORMATIVE_ERRATUM_ACTIVE`
- Scope: simulator-cycle oracle only

## Normative precedence

This additive erratum leaves `design/SPEC.md` version 2.0 immutable. It
supersedes only the cycle derivation and exact cycle values in
`SPEC.md` section **Frozen workload and simulator-cycle expectations**.
It also supersedes the corresponding cycle fields in
`design/BENCHMARK_INTERFACE.json` version 2.0 and
`design/LORA_V4_TWO_TOKEN_24LAYER_RTL_CONTRACT.json` version 2.0.

For cycle fields, this erratum and
`design/LORA_V4_TWO_TOKEN_24LAYER_RTL_INTERFACE_ERRATUM_V1.json` are
normative and win any conflict with the repeated-layer-0 table. All other
behavior, interface declarations, transactions, clock/reset/handshake rules,
scenario coverage, claim boundaries, and acceptance requirements in the base
artifacts remain normative and unchanged.

The stale table assumed the accepted layer-0 cycle schedule repeated for all
24 layers. That assumption is withdrawn. Projection, RMSNorm, RoPE, SiLU, and
residual cycles are workload dependent. Attention score, softmax, and compose
cycles are invariant for this frozen workload.

## Bound pre-attempt oracle

The replacement oracle is the source-derived predictor frozen before the
attempt-0003 run. Its normative inputs are:

- `tools/run_lora_v4_two_token_24layer_rtl_cycle_repair.py`, SHA-256
  `f8bf671c6c980cff7a121d215b66a05078cd2552c45cb6c45386bc2aa35fef26`;
- cycle-repair preflight `SHA256SUMS`, SHA-256
  `452f6c48d939d825984c24db872dc497ca60a44f0be8a364fd301bb4bac65823`;
- cycle source bindings, SHA-256
  `be7291cd753c85b6e592336505a60972a89518fd4d426b86405d3e5646fdc0a4`;
- focused layer-0/layer-1 discrimination, SHA-256
  `bca8d44ac48eaeadbf3e7ea69ead7f1e97664ffb62a011ca9a23a3c54625800a`;
- attempt-0003 pre-run `authorization.json`, SHA-256
  `d32d36227efb6e6ab9e63d22727d14ce9f397f9432f078efc3f7f892671f17d0`;
- attempt-0003 pre-run `freeze.json`, SHA-256
  `baf91bb8b3290e553a17fa38267cfb268aca4630ff2f79578b1ac4059c2be488`.

The exact per-family formulas and source hashes are machine-readable in the
interface erratum. The exact per-layer values below are the result of applying
that bound oracle to each hash-indexed frozen layer vector set.

## Replacement exact simulator-cycle schedule

These remain deterministic family-local simulator cycles, not wall-clock time
or accelerator latency. Cases remain serialized within each family.

| Layer | Projection | RMSNorm | RoPE | Score | Softmax | Compose | SiLU | Residual | Aggregate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 26,764,922 | 249,060 | 757,397 | 5,633 | 3,071 | 34,023 | 1,764,423 | 3,623 | 29,582,152 |
| 1 | 26,786,410 | 248,984 | 758,465 | 5,633 | 3,071 | 34,023 | 1,769,287 | 3,631 | 29,609,504 |
| 2 | 26,771,681 | 249,038 | 758,792 | 5,633 | 3,071 | 34,023 | 1,822,791 | 3,623 | 29,648,652 |
| 3 | 26,704,046 | 248,856 | 757,460 | 5,633 | 3,071 | 34,023 | 1,842,247 | 3,615 | 29,598,951 |
| 4 | 26,700,964 | 248,836 | 759,704 | 5,633 | 3,071 | 34,023 | 1,764,423 | 3,615 | 29,520,269 |
| 5 | 26,708,739 | 248,836 | 758,276 | 5,633 | 3,071 | 34,023 | 1,803,335 | 3,615 | 29,565,528 |
| 6 | 26,702,918 | 248,836 | 757,658 | 5,633 | 3,071 | 34,023 | 1,774,151 | 3,615 | 29,529,905 |
| 7 | 26,702,314 | 248,836 | 758,966 | 5,633 | 3,071 | 34,023 | 1,754,695 | 3,615 | 29,511,153 |
| 8 | 26,700,030 | 248,836 | 757,382 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,497,557 |
| 9 | 26,717,326 | 248,836 | 757,682 | 5,633 | 3,071 | 34,023 | 1,749,831 | 3,615 | 29,520,017 |
| 10 | 26,701,585 | 248,836 | 757,709 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,499,439 |
| 11 | 26,711,625 | 248,836 | 758,714 | 5,633 | 3,071 | 34,023 | 1,754,695 | 3,615 | 29,520,212 |
| 12 | 26,709,225 | 248,836 | 757,154 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,506,524 |
| 13 | 26,710,218 | 248,836 | 758,228 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,508,591 |
| 14 | 26,715,857 | 248,836 | 758,393 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,514,395 |
| 15 | 26,712,266 | 248,836 | 758,168 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,510,579 |
| 16 | 26,706,474 | 248,836 | 758,390 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,505,009 |
| 17 | 26,709,874 | 248,836 | 758,360 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,508,379 |
| 18 | 26,716,639 | 248,836 | 758,459 | 5,633 | 3,071 | 34,023 | 1,744,967 | 3,615 | 29,515,243 |
| 19 | 26,724,003 | 248,836 | 757,850 | 5,633 | 3,071 | 34,023 | 1,783,879 | 3,615 | 29,560,910 |
| 20 | 26,719,698 | 248,838 | 758,204 | 5,633 | 3,071 | 34,023 | 1,774,151 | 3,615 | 29,547,233 |
| 21 | 26,713,144 | 248,840 | 759,332 | 5,633 | 3,071 | 34,023 | 1,837,383 | 3,615 | 29,605,041 |
| 22 | 26,747,743 | 248,860 | 757,598 | 5,633 | 3,071 | 34,023 | 1,798,471 | 3,623 | 29,599,022 |
| 23 | 26,761,676 | 248,896 | 755,825 | 5,633 | 3,071 | 34,023 | 1,808,199 | 3,623 | 29,620,946 |
| **All 24** | **641,319,377** | **5,972,748** | **18,194,166** | **135,192** | **73,704** | **816,552** | **42,506,664** | **86,808** | **709,105,211** |

The prior aggregate `709,971,648` is obsolete for acceptance. The normative
replacement aggregate is `709,105,211`, a delta of `-866,437` cycles.

## Acceptance and fail-closed readback

Exact cycle reproduction now means exact equality with every per-layer family
value above and with all family and total aggregates. A checker must reject:

- a changed base artifact, oracle wrapper, preflight artifact, source binding,
  frozen vector, authorization, freeze, run summary, or accepted check;
- a missing, duplicate, reordered, or extra layer/family field;
- any formula, per-layer value, family aggregate, or total disagreement;
- any attempt to use the obsolete repeated-layer-0 table for acceptance.

The required read-only command is:

```text
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_lora_v4_two_token_24layer_rtl_erratum_v1.py
```

It must emit `PASS_ERRATUM_V1_FAIL_CLOSED_CONSISTENCY` and exit zero. Any
exception or inconsistency exits nonzero. The checker reads consumed attempts
0001 through 0003 only; this erratum grants no authority to relabel, replay,
overwrite, or otherwise modify them.
