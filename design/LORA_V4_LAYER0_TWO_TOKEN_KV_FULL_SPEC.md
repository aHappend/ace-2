# LoRA V4 layer-0 two-position KV full-path specification

## Status and authority

This document is the normative specification for mission
`extend-layer0-two-token-kv-rtl`. It is a bounded, computer-local RTL
verification contract. It does not modify or supersede the productization
contract in `design/SPEC.md`, and it grants no full-model, Stage-1, FPGA,
U280, synthesis, PPA, accelerator-latency, or model-quality authority.

The accepted token-0 layer-0 attempt and the accepted two-position attention
frontier remain immutable inputs. The full-layer attempt defined here is a
separate namespace and may not rewrite or relabel either predecessor.

## Frozen identities

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, model SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Adapter: canonical LoRA V4 checkpoint-176, adapter SHA-256
  `c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28`,
  checkpoint tree SHA-256
  `8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee`.
- Tokenizer-derived prompt alias:
  `canonical_lora_v4_layer0_adaptation_prompt`; positions 0 and 1 are token
  IDs `5576` and `12` from the frozen token sequence.
- Geometry: hidden 896, intermediate 4864, 14 query heads, 2 KV heads,
  head dimension 64, grouped-query mapping `kv_head = query_head // 7`.
- Predecessor attention contract SHA-256
  `d1e55d8a473b01e25322ce7127e156995512f72a3bc334d57b6339bda202def0`;
  predecessor attention attempt `SHA256SUMS` SHA-256
  `389291ee4d985c857aeeb02209471d851d6c13c0ea0cec776dc0bb6628d10c28`.
- Accepted token-0 full-layer contract SHA-256
  `8b2d26099c8ae84cc2bb965543e01d51e22847e1ec6dadfd227054e7ebf575a4`;
  accepted attempt `SHA256SUMS` SHA-256
  `d19892f2bb05ef250e8875087ae85fd2ac14de0d9bd6e13166a81c67e8c4e238`.

## Required execution

The official attempt executes both positions through this ordered layer-0
operator path:

1. input RMSNorm;
2. Q, K, and V W4A8 projections;
3. Q and K RoPE;
4. causal attention score, softmax, and attention compose;
5. O W4A8 projection;
6. attention residual add;
7. post-attention RMSNorm;
8. gate and up W4A8 projections;
9. SiLU-gate product;
10. down W4A8 projection;
11. final residual add and 896-byte signed-int8 layer output.

Position 0 is prefill and writes one K/V entry. Position 1 is decode and must
read both the cached position-0 entry and its new position-1 entry. Position-1
Q and K RoPE must be non-identity. Every query head must use the frozen
grouped-query mapping. The counterfactual obtained by zeroing cached
position-0 V must change at least one position-1 compose result, and at least
one position-1 head must assign nonzero softmax probability to position 0.

## Fixed-point and scaling contract

- Projection weights are signed W4, symmetric per output channel, using the
  merged `W_base_fp32 + (alpha/r) * (B_fp32 @ A_fp32)` tensors.
- Activations and cache entries are signed int8. Projection accumulators are
  signed int32 complete reductions. Requantization uses signed Q31
  multipliers, nonnegative right shifts, zero output zero-point, zero bias,
  and round-to-nearest ties-to-even followed by int8 saturation.
- Q/K/V weight tensors, output scales, multipliers, and shifts are reused
  unchanged from the accepted token-0 attempt for both positions. This is the
  static KV-cache scale contract.
- O, gate, up, and down weights are shared across positions. Position 0 must
  reproduce the accepted token-0 metadata and tensors exactly. Position 1
  freezes deterministic per-operator output scales from the independently
  computed merged-float path; its multipliers and shifts are then derived
  before RTL execution and recorded in the attempt.
- RMSNorm uses int8 input and signed Q7.8 gain metadata. Residual inputs are
  requantized to a declared common tensor scale and added with int8
  saturation. SiLU inputs are signed Q6.9 and its output is signed int8.
- RoPE uses theta 1,000,000, signed Q9 activation scale one, signed Q15
  cosine/sine coefficients, split-half pairing, identity at position 0, and
  non-identity at position 1.
- Attention score and compose use the maintained independent Python reference
  models. Decode context count is 2. The compose testbench presents two real
  causal entries followed by seven `-32768` score/zero-value padding entries,
  exercising FIRST/LAST protocol tiles without changing the mathematical
  result.

## Clock, reset, handshake, stall, and recovery protocol

- Simulation clock is 10 ns, toggled every 5 ns.
- All cores use active-low asynchronous `rst_ni`. Core testbenches assert
  reset for four rising edges. RMSNorm, RoPE, score, and softmax then wait two
  additional rising edges; projection waits two; SiLU and compose begin after
  reset deassertion as implemented by their maintained testbenches. The shell
  residual replay asserts reset for five rising edges and waits two after
  deassertion.
- `clear_i` is tied low for this attempt. Mid-command clear is not exercised.
- A source asserts `valid` only after observing `ready`, holds payload stable
  through the accepting edge, and deasserts on the following falling edge.
  A sink initially holds `ready` low, waits for `valid`, then accepts the
  output and deasserts `ready`. This intentional wait exercises output hold
  under backpressure.
- The shell residual replay keeps all SRAM request-ready bits asserted and
  injects no memory error or tag error. Random stalls are not used. Timeout
  guards are the exact values in the interface manifest and any timeout is a
  failed attempt with no RTL-correctness conclusion.
- Multiple cases execute sequentially without resetting between cases. Each
  operator must return to its ready/idle protocol state after output
  acceptance. The two shell residual commands per position must complete in
  order and preserve the completion tag. Error recovery after an injected
  fault is outside this bounded attempt.

## Exact workload and cycle-accounting table

The exact elaborated module interfaces, parameter values, ports, testbench
specializations, timeout guards, and handshake counts are frozen in
`design/LORA_V4_LAYER0_TWO_TOKEN_KV_FULL_INTERFACE.json`.

The official attempt must record simulator cycle counts for projection,
RMSNorm, RoPE, attention score, softmax, attention compose, SiLU, and shell
residual replay. Cycle counts are descriptive simulator evidence, not
accelerator latency. Required transaction totals are:

| Family | Cases/commands | Required accepted data transactions |
| --- | ---: | ---: |
| Projection | 14 operator-position cases, 25,344 channels | 7,454,720 pair beats, 25,344 metadata beats, 25,344 outputs |
| RMSNorm | 4 cases | 224 collect beats, 224 gain/scale beats, 224 output beats, 4 done handshakes |
| RoPE | 4 Q/K tensor cases | 224 independent 16-lane beat transactions and outputs |
| Attention score | 28 head-position cases / 42 causal tokens | 2,688 scalar Q/K pair beats and 42 outputs |
| Softmax | 28 head-position cases | 28 starts and 28 probability outputs |
| Attention compose | 28 cases, fixed protocol context 9 | 168 commands, 1,008 value beats, 112 output beats |
| SiLU gate | 2 cases | 1,216 input beats and 1,216 output handshakes |
| Residual shell | 4 commands | 224 output beats: attention and final residual for both positions |

## Acceptance matrix

The official attempt passes only if all entries below pass in one sealed
attempt namespace.

| ID | Requirement |
| --- | --- |
| A1 | Revalidate all frozen source, checkpoint, predecessor, spec, and interface hashes before creating the official attempt. |
| A2 | Position 0 reproduces every accepted token-0 operator tensor, all seven projection accumulators/outputs, both residual outputs, and final layer output exactly. |
| A3 | Position 1 uses cached position-0 K/V, non-identity Q/K RoPE, grouped-query mapping, context-2 score/softmax/compose, and the cache counterfactual checks. |
| A4 | RTL and independent Python fixed-point outputs have zero mismatches at every operator boundary, every projection accumulator/output, all K/V cache bytes, and both final 896-byte outputs. |
| A5 | All maintained testbench PASS markers are present, no timeout or overflow occurs, and the exact projection result count is 25,344. |
| A6 | The attempt records raw logs, generated vectors, tensors, execution sources, cycle counts, commands, elapsed time, source hashes, a machine-readable comparison, manifest, and `SHA256SUMS`. |
| A7 | A fresh `--check` verifies every immutable member and reproduces the exact zero-mismatch status without rerunning RTL. |
| A8 | Float-vs-W4A8 errors are reported per position as descriptive values only. No float-fidelity or model-quality PASS is permitted; the separate post-preview erratum remains bound. |

## Failure classification and claim boundary

A missing dependency, source/hash mismatch, compile failure, simulator
failure, timeout, malformed PASS marker, nonzero mismatch, cache-reuse failure,
or final-output mismatch fails the attempt. Infrastructure or harness failure
does not establish an RTL defect unless the failing operator transaction
completed with valid comparable output.

Success establishes exactly two tokenizer positions through one complete
layer-0 W4A8 RTL operator path on the local simulator. It does not establish
the full 24-layer model, arbitrary-text RTL chat, Stage-1 completion, FPGA or
U280 execution, synthesis/PPA, routed timing, hardware latency, or acceptable
model quality.
