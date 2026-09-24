# ACE-2 architecture: dynamic Scale32 successor proposal

## Planner proposal and authority boundary

The Planner recommends `shared_token_group_dynamic_scale32_v1` for the next
Manager freeze decision. This is a complete architecture proposal, not a
successor freeze or implementation authorization. The Manager-owned stage
remains `architecture`; the accepted prefix remains through `layer_0.v_proj`,
first unsupported remains `layer_0.rope_q`, and mode remains `ADVANCE`.

The candidate dynamically selects one power-of-two Scale32 exponent delta per
64-lane Q/K/V head or 128-lane hidden/MLP group. It applies across all 24 layers
and final RMSNorm/lm-head, attacking recurrent activation quantization rather
than another attention-local tensor. Signed-int8 mantissas, W4 weights, the
128-bit external stream, the 2.0 mm2 non-SRAM cap, and the 100 MHz floor remain
unchanged.

### Certified harness binding and execution interlocks

This proposal is explicitly bound to
`evidence/baseline_harness_hardening/synthetic_harness_v1/PUBLIC_CERTIFICATION.json`
(SHA-256 `49c8cb699ee559a09f0298ec16a1c4e6168cd812aff11357b7c49c68f9d1d315`).
The certification binds the harness report
`88e08ad61aa2f7969ea7e08c74ac4875dc62b915646a618717428513087636b7`,
manifest `224dd28455c5cdda567e2b9f499acdb9da8be3ef5c54e228f65661c29f3a3e1e`,
independent L2 review
`2fca2c7c8759d82294cc9ec52cdb9d95fb22a163faf29f302d4231ef718e491f`,
and Manager recovery seal
`5dfd2fd991f3e0237a9ffdc60660bd2df04cd19ffd8f083eb39d306c6a182d3f`.
The certified harness made zero candidate/model executions and grants no
implementation authority.

The seven required harness properties are, in execution order:

1. stage authorization before execution;
2. a durable exactly-one reservation and run ledger;
3. atomic artifact and companion-hash commit;
4. exact provenance binding;
5. a candidate-entrypoint interlock;
6. crash recovery; and
7. fail-closed handling of missing or mismatched artifacts.

### Evidence-gated selection

| Retained direction | Material result | Architecture implication |
| --- | --- | --- |
| Tile-BFP and native tagged attention | Layer-0 score/value improved, but paired or C4 final output regressed | Do not spend another approval on attention-score storage |
| Q/K and V residual terms | Local metrics improved or stayed flat while C4 `lm_head` failed | A local cross term has insufficient end-to-end leverage |
| Down-projection residual fusion | Technical and execution-integrity no-go | Do not retune that family |
| Cross-layer error carry | Terminal baseline protocol/provenance no-go; mechanism sealed | Do not repair or reuse QECR |
| Dynamic Scale32 activation groups | Not yet executed | Highest recurrent Amdahl reach because every activation boundary and final output is covered |

### Compute, dataflow, and roofline model

The representative shape is one batch-1 decoder-layer token, excluding the
context-length-dependent attention dot products. The seven W4 projections
perform 14,909,440 MACs and read 7,454,720 packed-W4 weight bytes. Four shared
MAC lanes require 3,727,360 zero-stall cycles. Weight-only arithmetic intensity
is exactly 2.0 MAC/byte, so the compute/memory balance point is 2 bytes/cycle.

| External bandwidth | Weight DMA cycles including 64-cycle setup hook | Non-overlapped projection+DMA cycles | Roofline classification |
| ---: | ---: | ---: | --- |
| 1 byte/cycle | 7,454,784 | 11,182,144 | memory-bound |
| 2 bytes/cycle | 3,727,424 | 7,454,784 | balance point |
| 4 bytes/cycle | 1,863,744 | 5,591,104 | compute-bound |
| 8 bytes/cycle | 931,904 | 4,659,264 | compute-bound |
| 16 bytes/cycle | 465,984 | 4,193,344 | compute-bound |

Each layer produces 181 dynamic groups. A shared 16-lane max/exponent/round
path requires a 1,557-cycle planning hook for the second-pass group finalizers,
0.0418% of projection cycles. Projections create 116,480 tagged partial events
per layer, only 3.124% average utilization of a one-event-per-cycle shared
aligner. A partial is separated from the next partial by at least 32 MAC cycles,
so the selected aligner is intended to hide under the MAC schedule. These are
architecture estimates, not measured RTL latency or throughput.

The dataflow is:

1. stream a 64- or 128-lane wide-result group into one ping-pong buffer while
   computing max absolute value;
2. choose the smallest legal exponent delta that maps all RNE mantissas to
   `[-127,+127]` and keeps effective Scale32 exponent in `[-24,+4]`;
3. emit the signed-int8 payload and its 64-byte sidecar atomically at successful
   completion;
4. attach input effective Scale32 values to projection partial sums, residual
   operands, RMSNorm square sums, SiLU products, RoPE heads, and KV records;
5. align with one shared tagged accumulator and round only at the operator's
   architecturally defined output boundary.

No internal NoC is added. Existing DMA, eight SRAM banks, descriptor engine,
completion scoreboard, interrupt path, and error path are reused.

### Memory hierarchy and lifetime

The preserved noncandidate live set is 377,600 bytes of the 524,288-byte SRAM.
The proposal adds one 1,280-byte 128-lane wide ping-pong buffer, at most thirteen
64-byte live tensor sidecars (832 bytes), and 256 bytes of shared runtime state.
Planned peak SRAM is therefore 379,968 bytes, leaving 144,320 bytes. Mutually
exclusive operators and all 24 layers reuse the same allocation.

Transient sidecars add at most 832 external bytes per layer only if every
intermediate is spilled. KV-cache storage adds no new scale records: the frozen
workload already requires one four-byte Scale32 record per layer, token, and K/V
head; candidate mode makes those records dynamic.

### Lifetime and resource-sharing plan

| Resource/lifetime | Selected shared implementation | Dedicated alternative | Expected cycle cost | Area/storage consequence |
| --- | --- | --- | --- | --- |
| W4A8 MACs | Existing four projection lanes | Candidate-local array | none | avoids duplicate MAC array |
| Tagged align/accumulate | One pipelined wide aligner/accumulator across projections, residual, RMSNorm, and SiLU | One per operator | hidden under >=32 MAC cycles/group; 3.124% mean use | avoids duplicated shifters and wide accumulators |
| Max/exponent select | One 16-lane max and leading-bit unit | One per producer | included in 1,557 cycles/layer | all producers are lifetime-exclusive |
| Wide group storage | One 1,280-byte ping-pong buffer | Private operator buffers | second-pass quantization hook | avoids per-operator wide SRAM/register files |
| Round/saturate/vector SFU | Existing vector round/saturate, RMSNorm, and SiLU resources | Candidate-local units | included above | avoids duplicate SFU/control |
| DMA/SRAM/control | Existing DMA, eight banks, descriptor/interrupt/error engines | Private sidecar path | sidecar transfers only on spills | avoids new ports, reorder state, and CSRs |

Dedicated-versus-shared area, timing, and power remain unmeasured until legal
RTL and fresh canonical PPA exist.

### Leverage, risks, verification, and fallbacks

The proposal has recurrent Amdahl leverage: it changes every activation
quantization boundary in every block and the final output path. That is broader
than prior directions whose first material divergence was confined to attention
score or value.

Highest risks are final-output non-monotonicity on C4, exact mixed-Scale32
accumulation without repeated rounding, RMSNorm squared-exponent width bounds,
sidecar atomicity under stalls/reset, and timing of the shared shifter/leading-
bit path at 100 MHz. After a Manager freeze and stage advance, the RTL stage is
limited to reference/vector generation, lint, elaboration, bit-exact simulation,
interface checks, and minimal formal. It may not execute a disabled-mode
baseline, candidate quality run, full shell regression, or canonical PPA.

Verification is two separately reviewed tasks in strict order. Task 1 is exactly
one disabled-mode baseline through the certified harness. Task 2 is exactly one
all-24-layer two-dataset candidate quality run and may begin only after an
independent Task-1 PASS. Full shell regression remains prohibited until an
independent candidate-quality GO and fresh separate Manager authorization. One
fresh canonical SKY130 PPA remains prohibited until shell acceptance and a
separate Manager transition to `ppa`. This proposal claims no execution.

Failure on either frozen dataset seals this direction. Group-size, exponent,
or sidecar changes require a fresh architecture review. Structural fallbacks,
not automatically authorized, are uniform 128-lane groups, an explicit sidecar
stream, or a wider/two-pass exact tagged accumulator. No fallback may relax the
2.0 mm2 cap, 100 MHz floor, or complete-workload quality limits.

## Historical sealed architecture record

The remainder of this file preserves the prior
`cross_layer_quantization_error_carry_final_output_v1` architecture record as
negative provenance only. That contract is sealed and is not the proposed
successor.

## Authority, frontier, and claim boundary

The Manager rolled `verification` back to `architecture` at
`2026-08-02T01:44:31.998340Z`. The independently accepted disposition for
`shared_down_projection_residual_fusion_v1` is an integrity no-go bound to
`evidence/shared_down_projection_residual_fusion_v1/latest/VERIFICATION_DECISION.json`
(SHA-256 `9655ca8a724fc04c6e32c26f7883f07d3ff117b82d2aa04ebbc68b15de26d8c9`).
That predecessor is sealed. Its paired smoke, shell admission, candidate PPA,
physical design, prototype, benchmark, and signoff remain unrun.

The Manager-owned `current_stage` remains `architecture`. The accepted ordered
prefix remains `layer_0.input_rmsnorm`, `layer_0.q_proj`, `layer_0.k_proj`,
`layer_0.v_proj`; first unsupported remains `layer_0.rope_q`. `ADVANCE` mode,
the 128-bit abstract streaming-memory boundary, the 2.0 mm2 non-SRAM cap, and
the 100 MHz floor are unchanged. The latest accepted PPA frontier remains
historical: 62,199 cells, 0.6108746272 mm2 non-SRAM area, and +0.1502 ns setup
slack at 100 MHz. No successor area, timing, power, RTL, verification, FPGA,
GDS, tapeout-readiness, or silicon result is claimed here.

## Selected structurally distinct successor

Exactly one successor is frozen:
`cross_layer_quantization_error_carry_final_output_v1`.

It is not a Q/K residual, V residual, or parameter-only variant of the sealed
down-projection fusion direction. It adds a stateful error sideband across
transformer-block boundaries and a final-output consumption point. The emitted
hidden-state tensor remains signed int8. The sideband is consumed exactly once
before the next input RMSNorm; the layer-23 sideband is consumed exactly once
before final RMSNorm and lm-head. It is never added to the residual bypass,
attention score, K/V cache, or post-attention residual path.

Layer 0 begins with an implicit all-zero carry. For layer `l` in `0..23`, the
wide post-MLP value is formed from the authoritative signed-32 down-projection
accumulator and signed-int8 post-attention residual using the existing Scale32
metadata and a common exponent. Let:

```
A[l,j] = signed-32 down-projection accumulator including bias
R[l,j] = signed-int8 post-attention residual
Sa[l,j], Sr[l], So[l] = normalized Scale32 values
e[l,j] = min(exp(Sa[l,j]), exp(Sr[l]), exp(So[l]))
N[l,j] = A[l,j] * sig(Sa[l,j]) << (exp(Sa[l,j]) - e[l,j])
       + R[l,j] * sig(Sr[l])   << (exp(Sr[l])   - e[l,j])
D[l,j] = sig(So[l]) << (exp(So[l]) - e[l,j])
Q[l,j] = RNE_signed(N[l,j] / D[l,j])
H[l,j] = sat_s8(Q[l,j])
E[l,j] = N[l,j] - H[l,j] * D[l,j]
C[l,j] = RNE_signed(E[l,j] * 2^15 / D[l,j])
```

`C` is a signed-16 Q0.15 carry measured in units of the emitted hidden-state
Scale32 value. Successful carry-mode completion requires `Q` to be in signed
int8 range and `C` to be in `[-16384, 16384]`. A final-output saturation or a
larger remainder is `numeric_overflow`: provisional payload is invalid, carry
valid remains clear, and no successful completion is published. This preserves
the exact fixed-point difference to within one half of a Q0.15 carry LSB; there
is no independent source-term rounding and no silent carry clipping.

Before input RMSNorm for layer `l+1`, or before final RMSNorm for producer layer
23, each lane is reconstructed once as:

```
Xq15[j] = (sign_extend(H[l,j]) << 15) + sign_extend(C[l,j])
```

`Xq15` is signed Q8.15 in a checked signed-24 workspace. For the 896-lane
vector and existing signed-Q7.8 gain `Gq8`, carry-aware RMSNorm is exactly:

```
SSq30 = sum_j(Xq15[j] * Xq15[j])
MSq30 = RNE_unsigned(SSq30 / 896)
Rq15  = max(1, ceil_sqrt(MSq30))
Iq30  = floor(2^45 / Rq15)
Y[j]  = sat_s8(RNE_signed(Xq15[j] * Gq8[j] * Iq30 / 2^53))
```

`2^45` compensates the Q8.15 root while producing a Q30 reciprocal; the final
`2^53` removes 15 input, 8 gain, and 30 reciprocal fractional bits. No epsilon
term is added beyond the existing `max(1, root)` baseline rule. The carry is
cleared when the consuming RMSNorm descriptor completes successfully and is
not forwarded after use.

## Compute, dataflow, and memory model

Qwen2.5-0.5B has 24 sequential transformer layers, hidden width 896, and MLP
intermediate width 4,864. One down projection performs 4,358,144 W4A8 MACs per
token per layer. The existing four projection lanes have a zero-stall floor of
1,089,536 cycles.

The selected area-first carry finalizer reuses one serial Scale32 align/divide
engine. Its architecture schedule is 10 cycles for the already demonstrated
integer quotient/remainder path plus 16 fractional continuation/guard cycles
for exact Q0.15 carry rounding: 26 cycles per output lane, 23,296 cycles per
layer, and 559,104 cycles across 24 layers. These are frozen architecture
planning hooks, not measured successor RTL. The down-projection plus carry
schedule is 1,112,832 cycles/layer; carry generation is 2.138157895% of the
four-lane dot schedule and 2.093397746% of their combined schedule.

Carry-aware RMSNorm folds the pre-add and square through one shared signed
24x24 multiplier, one lane per cycle. Its square pass is 896 cycles rather than
the historical 56-cycle 16-lane planning hook. The reciprocal-root and output
scale phases remain serialized by the existing shared divider/SFU plan. The
added 840 square-pass cycles are below 0.08% of the down-projection-plus-carry
slice and avoid a dedicated 16-lane wide-square array. Exact RTL latency must be
bound later before any throughput claim.

Per layer, streamed token-dependent external traffic remains 2,185,728 bytes:
2,179,072 down-projection weight bytes, 4,864 input bytes, 896 residual bytes,
and 896 output bytes. Arithmetic intensity is 1.9939095807 MAC/external-byte.
The carry remains on chip and adds no K/V or external-memory payload. It adds
1,792 SRAM bytes of live state and 86,016 SRAM bytes/token of aggregate carry
traffic: 24 writes plus 23 next-layer reads plus one final-RMSNorm read.

The successor reuses the predecessor's 86,592 immutable Scale32 image layout
but requires new `QECR` schema-1 headers and a new full-image digest. The fixed
canonical SRAM-tagged base remains `0x800000000005c300` (offset `0x5c300`).
Immutable metadata occupies `[0x5c300,0x71540)`. The 896-entry signed-16 carry
buffer occupies `[0x71540,0x71c40)`, followed by 128 bytes of validity, producer
layer/token identity, counters, and error state in `[0x71c40,0x71cc0)`. Total
candidate allocation is 88,512 bytes. With the preserved 377,600-byte
noncandidate live set, planned peak SRAM is 466,112 of 524,288 bytes, leaving
58,176 bytes. These are logical storage budgets, not SRAM macro area results.

The existing DMA copies exactly 86,592 immutable bytes at model load and zeroes
the 1,920 mutable bytes. Software validates normalized Scale32 records, all 24
headers, layer/lane order, model revision, and the full SHA-256 before setting
`qecr_metadata_valid`. Reset, unload, model switch, loader error, or overwrite
clears metadata and carry validity. No internal NoC or clock-domain change is
introduced.

## Roofline and Amdahl leverage

The architecture DMA hook remains:

```
DMA_cycles(B) = ceil(2,185,728 / B) + 64
```

The 64-cycle term remains an unmeasured common descriptor/DMA setup estimate.
Using the 1,112,832-cycle compute hook, the raw-payload balance point is
1.964113182 bytes/cycle. The slice is memory-bound at 1 byte/cycle and
compute-bound at 2, 4, 8, and 16 bytes/cycle. At 16 bytes/cycle, payload and
setup consume 136,672 cycles and the non-overlapped layer total is 1,249,504
cycles. Removing all carry-finalizer cycles could improve this slice by at most
1.021381579x; the mechanism is selected for end-to-end numerical leverage, not
local speedup.

The Amdahl leverage is recurrent: one bounded error vector influences the next
layer's normalized attention branch, repeats at every layer boundary, and the
last vector reaches final RMSNorm/lm-head. The mandatory discriminator therefore
requires both frozen datasets, ordered all-24-layer execution, final-output
metrics, and strict improvement over a fresh bit-identical baseline. A local
layer metric cannot admit the candidate.

## Interface and control contract

Descriptor size, CSR map, interrupts, error codes, reset, clock, external
ready/valid channels, and completion tags do not change. Flag bit 7 selects the
active carry contract; historical candidate bits 4, 5, and 6 are zero.

- A carry-producing `W4A8_PROJ` requires `layer_id=0..23`, `m=1`, `n=896`,
  `k=4864`, bit 7 set, and zero in all historical/reserved mode bits.
- `src0_addr`, `src1_addr`, `scratch_addr`, `dst_addr`, and `scale_addr` retain
  the down-projection input, packed W4 weights, post-attention residual, emitted
  hidden output, and ordinary projection metadata meanings.
- A carry-consuming `RMSNORM` requires bit 7 set and `layer_id=1..24`. The
  producer must be `layer_id-1`, the token/chain identity must match, and the
  carry must be valid. Layer-0 input RMSNorm requires bit 7 clear.
- Carry production cannot start while another carry is valid. Carry consumption
  clears validity only on successful RMSNorm completion. `CHAIN_END` with a
  valid carry is a `descriptor_error`.
- Pre-start opcode, shape, layer, metadata identity, address, dependency,
  producer/consumer order, and carry-valid checks issue no payload on failure.
- After start, reset, memory error, overflow, or watchdog termination publishes
  no success. Accepted output beats are provisional and software must discard
  them. Carry state is invalidated on every unsuccessful terminal result.
- Backpressure may pause metadata, accumulator, carry write/read, square,
  reciprocal, scale, or output phases without changing lane order or arithmetic.
- Reset asserts asynchronously at the accelerator boundary and deasserts
  synchronously. The core remains one clock domain; any future asynchronous
  bridge reopens architecture and CDC verification.

## Lifetime and resource-sharing plan

| Resource/lifetime | Selected shared implementation | Dedicated alternative | Expected cycle cost | Area/storage consequence |
| --- | --- | --- | ---: | --- |
| Projection MACs | Existing four W4A8 lanes | Candidate dot array | none | avoids a full duplicated projection array |
| Wide post-MLP value | Consume signed-32 accumulator in place | Copy all accumulators | none | avoids 3,584 bytes/token and extra ports |
| Align/divide/remainder | One serial Scale32 engine across all lanes/layers | Four parallel engines | 23,296 vs 5,824 cycles/layer | avoids four wide divider/remainder datapaths |
| Carry buffer | One 1,792-byte ping-free buffer, consumed before overwrite | Per-layer buffers | none | avoids 41,216 extra bytes for 23 redundant buffers |
| RMSNorm square | Reconfigure the carry engine's signed 24x24 multiplier | Dedicated 16-lane square array | +840 cycles/RMSNorm | avoids sixteen wide multipliers and mux/control |
| Sum/divide/root | Reuse vector/SFU accumulator and divider | Candidate-local RMSNorm | serialized planning hook | avoids duplicate 56-bit accumulator/divider/control |
| Round/saturate/error | Reuse common ties-even, clamp, sticky-error, and completion state | Candidate-local block | arbitration only | avoids duplicate CSR-visible state |
| DMA/SRAM/control | Existing DMA, eight banks, command engine | Private path | none | avoids new external ports and reorder state |
| Layer lifetime | One engine and one carry buffer across 24 sequential layers | One copy per layer | none | avoids 23 engine/control copies |

The selected alternative intentionally spends bounded cycles to preserve area
reserve. Dedicated-versus-shared area, timing, and power remain unmeasured until
authorized RTL and a fresh canonical SKY130 flow. If the next fresh PPA exceeds
2.0 mm2, the operator policy requires global compression; no architecture text
can waive that contract.

## Risks, verification strategy, and stop rules

Highest risks are baseline reproducibility, exact quotient/remainder sign rules,
Q0.15 tie handling, carry saturation, stale producer identity, double
consumption, widened RMSNorm square/root widths, and a new timing path through
the shared wide multiplier/divider.

Before RTL, an independent reference must prove the scalar equations, exhaustive
tie and sign boundaries, 24-bit reconstruction bounds, 56-bit sum-of-squares
bound, reciprocal scaling, reset/error behavior, and metadata/order checks.
After Manager-authorized RTL, the single bounded fast-loop task must include
implementation, full regression, canonical SKY130 PPA, and evidence binding
with `stage_closing=false`. No separate verification or PPA closeout task is
permitted for this intermediate uplift.

Candidate evaluation stops before shell/PPA if any of these occurs: fresh
zero-carry baseline mismatch; any layer-0 pre-injection difference; missing or
duplicate carry production/consumption; noncausal downstream differences;
overflow on frozen traces; score/post-MLP/final-output equality or regression on
either dataset; missing all-24-layer/final-RMSNorm execution; or construct/hash
mismatch. The 1.05x perplexity and two-point accuracy targets remain unmet until
complete executable evidence satisfies them.
