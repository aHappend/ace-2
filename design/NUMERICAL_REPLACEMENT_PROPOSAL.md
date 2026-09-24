# Planner proposal: shared token/group dynamic Scale32

## Authority and decision request

The Planner recommends `shared_token_group_dynamic_scale32_v1` as the next
structurally distinct architecture successor. This document is not a Manager
freeze, implementation authorization, or evidence that quality, area, timing,
power, or the complete workload passes. `research/PIPELINE_STATE.json` remains
Manager-owned and unchanged at `architecture`.

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`; mode remains `ADVANCE`. The 2.0 mm2 non-SRAM cap, 100 MHz
floor, 128-bit memory boundary, W4A8 contract, and complete-workload quality
limits are unchanged.

## Why this direction

The retained no-go evidence shows that attention-local representations,
per-head Q/K changes, native tagged attention, and Q/K or V residual terms can
improve layer-0 score or attention-value error while still regressing C4 final
`lm_head`. The sealed cross-layer carry direction never produced an admissible
candidate execution and may not be repaired or reused. The next high-leverage
move is therefore a recurrent activation-boundary contract, not another local
attention correction.

This proposal dynamically selects one power-of-two Scale32 exponent delta per
activation group at every layer boundary. It retains signed-int8 activations and
the existing normalized Scale32 representation. It changes neither weights nor
operator-owned targets.

## Frozen arithmetic proposed for Manager review

Each group has an immutable baseline Scale32 significand/exponent pair and a
runtime signed exponent delta. The effective scale keeps the baseline
significand and adds the delta to the exponent. Effective Scale32 exponents must
remain in the existing legal range `[-24,+4]`.

For exact wide values `x[i]` expressed in the baseline group's units, select the
smallest legal integer delta such that every

```
m[i] = RNE(x[i] / 2^delta)
```

is in `[-127,+127]`. `RNE` is round-to-nearest, ties-to-even. All-zero groups
use delta zero. Signed-int8 value `-128` is reserved and invalid in candidate
mode. If no legal delta exists, the operation fails with `numeric_overflow` and
publishes no successful completion or destination.

Grouping is fixed by tensor family:

- hidden and MLP-intermediate tensors use 128 lanes per group;
- Q, K, and V tensors use one 64-lane group per attention head so every RoPE
  pair has one identical scale;
- the 896-lane hidden width therefore has seven groups, the 4,864-lane MLP
  width has 38 groups, Q has 14 groups, and K/V have two groups each.

Projection forms signed-32 partial sums per 128-input group. Each partial is
tagged with the exact product of the input effective Scale32 and immutable W4
weight Scale32. One shared wide tagged accumulator aligns partials and rounds
only at final requantization. Residual add aligns both input group scales before
addition and selects a fresh output delta. RMSNorm aligns squared group sums by
twice the effective exponent before applying the frozen integer root/gain rule.
SiLU-gate adds operand exponents around the existing integer approximation and
selects a fresh output delta. KV-cache records remain signed-int8 with the
already-required four-byte dynamic Scale32 record per layer, token, and K/V
head.

No stochastic rounding, learned gate, prompt control, dataset control,
floating-point RTL, silent exponent clipping, or silent saturation is allowed.

## Admission and stop rules

The architecture contract is bound to the independently certified
`PUBLIC_CERTIFICATION.json` and its exact report, manifest, L2-review, and
Manager-seal hashes recorded in `design/ARCHITECTURE.md`. The harness guarantees
stage authorization, a durable exactly-one ledger, atomic artifact plus
companion-hash commit, exact provenance, candidate-entrypoint interlock, crash
recovery, and fail-closed behavior. It has executed no candidate or model and
does not authorize implementation.

Verification is split into two separately reviewed tasks. First, exactly one
disabled-mode baseline must reproduce the frozen baseline bit-for-bit through
the certified harness. Only after an independent PASS may exactly one candidate
run the same mechanism across all 24 layers and final RMSNorm/lm-head. On both
frozen datasets it must strictly improve layer-0 Q/K/V, score, attention-value,
post-MLP, and final `lm_head` relative-L2 without dataset-dependent behavior.
Complete-workload W4A8 perplexity and accuracy targets remain unchanged.

Failure on either dataset seals the direction. Group size, exponent range, and
sidecar rules may not be retuned under the same approval. Verified Fmax below
100 MHz blocks advancement. Fresh PPA at or above a 3% area increment triggers
the operator-owned global reuse/folding review; exceeding 2.0 mm2 enters
`GLOBAL_COMPRESSION` without relaxing the cap.

## Implementation boundary

After independent architecture review and a Manager freeze/advance in order,
the RTL stage is `stage_closing=false` and may run only reference/vector
generation, lint, elaboration, bit-exact simulation, interface checks, and
minimal formal. Baseline execution, candidate execution, full shell regression,
and canonical PPA are prohibited in RTL.

Full shell regression requires an independent candidate-quality GO plus fresh
separate Manager authorization. Fresh canonical SKY130 PPA requires shell
acceptance plus a separate Manager transition to `ppa`. Prototype, full
benchmark, and signoff remain reserved for a complete-workload or model/system
release milestone. No execution is claimed by this proposal.
