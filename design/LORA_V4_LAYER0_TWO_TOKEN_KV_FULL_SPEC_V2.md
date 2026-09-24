# LoRA V4 layer-0 two-position KV full-path specification V2

## Status and authority

This is the additive specification-closure artifact for mission
`extend-layer0-two-token-kv-rtl`. It inherits every requirement of
`design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_SPEC.md` at SHA-256
`b913b397b4b15d89a5f63647c257da5f169c68be100ba12ae42e8228eed6aa70`
and closes only the interface-range, reset-observability, simulator
latency/throughput, scenario-outcome, and benchmark-applicability gaps found
after accepted `attempt-0001` was sealed. The predecessor specification,
interface, frozen contract, and `attempt-0001` remain immutable.

The V2 execution is governed by a separate `repair-0002-contract.json` and
must be written only to `attempt-0002`. It is a Fresh-L2 recompilation and
reproduction, not an in-place repair or relabeling of accepted evidence.

## Benchmark applicability

No external benchmark applies to this mission. This is a deterministic,
tokenizer-derived, bounded layer-0 RTL reproduction against an independent
fixed-point golden. It produces no benchmark score, ranking, throughput
comparison, model-quality certification, product acceptance, accelerator
latency, FPGA/U280 result, synthesis/PPA result, or arbitrary-text chat claim.
The mission-scoped machine-readable statement is in
`design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_INTERFACE_V2.json`.

## Legal semantic ranges

Packed signed fields use two's-complement interpretation. Values outside a
field's semantic range are illegal even when they fit the HDL bit width.

| Interface | Legal inputs for this bounded contract |
| --- | --- |
| Common ready/valid/control | Every one-bit field is `0` or `1`. Payload is sampled only on an accepting rising edge. Payload must remain stable while `valid=1` and `ready=0`. |
| Projection | `last_group_i` is `0..1215`; this workload uses `223` for 896-element reductions and `1215` for 4864-element reductions. Each activation lane is signed int8 `[-128,127]`; each weight lane is signed int4 `[-8,7]`; multiplier and bias are signed int32; right shift is `0..63`; zero point is signed int8 and is `0` in this workload. Exactly `last_group_i+1` pair handshakes precede one metadata handshake and one output. |
| RMSNorm | `HIDDEN_SIZE=896`, `LANES=16`, and exactly 56 collection beats plus 56 scale/gain beats are required. Activation lanes are signed int8. Gain lanes are signed 16-bit Q7.8. No extra or partial beat is legal. |
| RoPE | Exactly 56 beats per tensor. Activation and paired-activation lanes are signed int8. Scale, paired-scale, cosine, and sine lanes are signed 16-bit; trigonometric lanes are Q15. `second_half_i` is boolean and must match split-half pairing. Positions are exactly 0 and 1. |
| Attention score | Exactly 64 signed-int8 Q/K pair handshakes per token. Multiplier is signed int32; shift is `0..63`. Output score is signed int16. Contexts are one token at position 0 and two tokens at position 1. |
| Softmax | `context_count_i` is `1..8`; this workload uses only `1` and `2`. Active scores are signed int16. Inactive score lanes are `-32768`. Active unsigned-Q15 probabilities are `0..32768` and sum to `32768` subject to the maintained integer reference. |
| Attention compose | Commands are `MAX_FIRST=0x00`, `MAX_MORE=0x01`, `SUM_FIRST=0x02`, `SUM_MORE=0x03`, `VALUE_FIRST=0x04`, `VALUE_MORE=0x05`, and `VALUE_LAST=0x06`. `tile_count_i` is `1..8`; cumulative context is `1..32768`; this workload uses protocol context 9 with two real causal entries and seven padding entries. `start_authorized_i` must be 1 for acceptance. Value lanes are signed int8. |
| SiLU gate | `lane_count_i` is `1..8`; this workload uses 8 except a legal final partial beat when required by tensor length. Gate/up lanes are signed 16-bit Q6.9. Multiplier is signed int32; shift is `0..63`; zero point is signed int8 and is `0` here. |
| Residual shell replay | Opcode is `ACE2_OPCODE_RESIDUAL_ADD=0x08`; flags are `0x09`; layer is `0..23` and is 0 here; `m=1`, `n=896`, `k=0`, sequence position is 0, and all source/destination addresses are 16-byte aligned. Completion tags are 16-bit and must return unchanged. |

Integer payload bit patterns within these ranges are legal; saturation is an
observable numerical result, not an illegal-input condition. Unsupported
opcodes, invalid command order, zero or excessive counts, wrong shapes,
misaligned addresses, and unauthorized compose starts are illegal controls.

## Observable reset contract

Reset observations assume all non-clock inputs are held at zero, including
all source valids and sink readies. After asynchronous assertion has
propagated:

| Family | Required observable values |
| --- | --- |
| Projection | `start_ready=1`; pair/meta ready and output valid are 0; output data, accumulator, overflow, and saturation are 0. |
| RMSNorm | `start_ready=1`; input/gain/scale ready, output valid, and done valid are 0; output data, sumsq, inverse RMS, and saturation are 0. |
| RoPE | `start_ready=1`; beat ready and output valid are 0; output data and saturation are 0. |
| Attention score | `start_ready=1`; pair ready and output valid are 0; score, accumulator, and saturation are 0. |
| Softmax | `start_ready=1`; output valid is 0; all probability lanes and saturation are 0. |
| Attention compose | With zero command/tile inputs, `command_allowed=0`, `start_ready=1`, value ready/output valid/output last/command done are 0, output data is 0, saturation is 0, and context count is 0. |
| SiLU gate | `start_ready=1`, `beat_ready=1`, output valid is 0, and output data/saturation are 0. |
| Residual shell | CSR ready is 1; command ready, busy, IRQ, command-done valid/error/saturation, memory request/write/read-response ready valids, and SRAM request valids are 0; completion tag, sumsq, inverse RMS, and CSR response data/error are 0. Payloads whose valid is 0 are ignored and need not be zero. |

The first legal transaction after reset release must complete exactly. Core
testbenches provide the settle intervals frozen in the V1 interface. The shell
remains command-not-ready until its normal CSR enable sequence completes.
Successful first transactions plus source-hash-bound reset assignments are
the reset/recovery evidence for this bounded run; no mid-command reset claim
is made.

## Pre-execution simulator latency and throughput expectations

Cycle counts are deterministic testbench clock counts at 10 ns and are not
hardware or accelerator latency. Because sources, generated vectors, and
testbenches are hash-bound, Fresh-L2 must reproduce these exact family totals:

| Family | Expected cycles | Accepted-output/transaction accounting |
| --- | ---: | --- |
| Projection | 26,764,922 | 7,454,720 pair beats, 25,344 metadata beats, 25,344 outputs |
| RMSNorm | 249,060 | 224 collect beats, 224 gain/scale beats, 224 output beats, 4 done handshakes |
| RoPE | 757,397 | 224 input/output beat transactions |
| Attention score | 5,633 | 2,688 pair beats and 42 score outputs |
| Softmax | 3,071 | 28 starts and 28 probability outputs |
| Attention compose | 34,023 | 168 commands, 1,008 value beats, 112 output beats |
| SiLU gate | 1,764,423 | 1,216 input beats and 1,216 output handshakes |
| Residual shell | 3,623 | 4 commands and 224 output beats |

The aggregate expected total is 29,582,152 family-local simulator cycles.
Cases are serialized within each family; no overlapping-case or sustained
hardware-throughput claim is permitted. Output backpressure may hold a valid
payload indefinitely and must not alter it. Per-family timeout guards remain
hard failure bounds, not expected latency.

## Scenario outcomes

| Class | Required in attempt-0002 | Observable acceptance outcome |
| --- | --- | --- |
| Normal | Yes | Both positions complete every listed operator with zero fixed-point mismatches and all PASS markers. |
| Boundary | Yes | Position 0 uses singleton causal context and identity RoPE; position 1 uses context 2, non-identity RoPE, both cache entries, grouped-query mapping, and the cache counterfactual changes compose output. Final/partial packing boundaries match exactly. |
| Reset | Yes | The source-bound reset map above is unchanged and every family completes its first post-reset case; no pre-command output-valid or error is observed. |
| Output stall | Yes | The sink initially withholds ready; output valid and payload remain stable until acceptance. |
| Recovery | Yes | Sequential cases complete without inter-case reset, each core returns ready/idle, and shell completion tags remain ordered and exact. |
| Illegal compose control | No | If exercised, unsupported command, invalid protocol order, zero/excessive tile count, or unauthorized start must not be accepted; `command_allowed=0` where applicable. Current outcome is `NOT_EXERCISED_NO_CLAIM`. |
| Illegal shell descriptor | No | If exercised, wrong opcode/shape/layer/alignment must terminate as descriptor error with no successful payload completion. Current outcome is `NOT_EXERCISED_NO_CLAIM`. |
| Random stalls | No | Current outcome is `NOT_EXERCISED_NO_CLAIM`; only deterministic output backpressure is covered. |
| Mid-command clear/reset | No | Current outcome is `NOT_EXERCISED_NO_CLAIM`. |
| Injected memory/tag error recovery | No | Current outcome is `NOT_EXERCISED_NO_CLAIM`. |

## Fresh-L2 repair acceptance additions

In addition to V1 acceptance items A1-A8, `attempt-0002` passes only if:

1. all 143 members of preserved `attempt-0001` verify before contract freeze,
   before execution, and during the decisive check;
2. the V2 spec, V2 interface, V2 runner, parent frozen contract, parent
   spec/interface, and preserved `attempt-0001/SHA256SUMS` are hash-bound by a
   separate repair contract;
3. all eight simulator families are recompiled into `attempt-0002/sim` from
   `attempt-0002/execution_sources`; no V1 simulator binary is reused;
4. all compile and simulation return codes are zero, all expected cycle totals
   reproduce exactly, all integer mismatch counts are zero, and both final
   output hashes reproduce exactly; and
5. immutable readback of `attempt-0002` passes with the V2 repair contract and
   mission-scoped non-benchmark statement intact.

All original non-goals and the post-preview float-fidelity erratum remain in
force.
