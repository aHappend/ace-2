# ACE-2 workload definition

This file freezes the definition-stage workload for ACE-2. It is a contract for
later architecture, RTL, verification, PPA, and benchmark work; it is not a
claim that those downstream artifacts already exist.

## Model boundary

| Field | Frozen value |
| --- | --- |
| Model | Qwen2.5-0.5B |
| Public config source | `https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json` retrieved 2026-07-25T13:57:38Z, SHA-256 `479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b` |
| Public license | Apache-2.0 (`apache-2.0` in Hugging Face model metadata for `Qwen/Qwen2.5-0.5B`, checked 2026-07-25) |
| Architecture | `Qwen2ForCausalLM` |
| Workload mode | Batch-1 causal decoder inference |
| Layers | 24 |
| Hidden size | 896 |
| Attention heads | 14 query heads, 2 KV heads |
| Head dimension | 64 |
| MLP intermediate size | 4864 |
| Maximum context | 32768 tokens |
| RoPE theta | 1000000.0 |
| RMSNorm epsilon | 1e-6 |
| Vocab size | 151936 |
| Embedding/lm-head | Tied weights |
| Sliding window | Disabled by the frozen config |

The accelerator scope includes all transformer-block math, final RMSNorm, and
the tied lm-head projection as a tiled streaming projection. The host/runtime
owns tokenization, chat template formatting, sampling, detokenization, external
memory allocation, and application I/O.

## Required supported operations

ACE-2 must support the operations needed to execute the frozen model boundary:

| Class | Required operations |
| --- | --- |
| Control | CSR read/write, job start/stop, descriptor queueing, completion tags, status, errors, interrupts |
| Memory | External-memory stream read/write, backpressure handling, banked SRAM tiling, KV-cache spill/fill |
| Linear algebra | W4A8 tiled projections for Q, K, V, output, gate, up, down, and lm-head weights |
| Attention | RoPE, grouped-query attention, causal mask, score scaling, softmax, value accumulation |
| Vector/SFU | RMSNorm, SiLU, residual add, requantization, saturation |
| Runtime | Deterministic command stream generation for prefill and decode traces |

The first delivery does not support training, batch sizes greater than 1,
speculative decoding, beam search, multimodal models, sliding-window attention,
dynamic model shape changes, hard external-memory PHYs, or floating-point RTL
datapaths for the primary workload.

## Ordered layer/operator frontier

Capability progress is tracked as an ordered implementation prefix over the
frozen Qwen2.5-0.5B workload. A layer/operator is counted as supported only
after its RTL, full verification, canonical SKY130 PPA, and evidence binding
have completed for the current source and constraint hashes.

For each transformer layer `layer_0` through `layer_23`, the canonical order is:

1. `input_rmsnorm`
2. `q_proj`
3. `k_proj`
4. `v_proj`
5. `rope_q`
6. `rope_k`
7. `kv_write`
8. `attention_score`
9. `softmax`
10. `attention_value`
11. `o_proj`
12. `attention_residual_add`
13. `post_attention_rmsnorm`
14. `mlp_gate_proj`
15. `mlp_up_proj`
16. `silu_gate`
17. `mlp_down_proj`
18. `mlp_residual_add`

After `layer_23.mlp_residual_add`, the final sequence is `final_rmsnorm` and
`lm_head`.

For the current `cross_layer_quantization_error_carry_final_output_v1`
architecture contract, the accepted implementation-supported prefix remains
`layer_0.input_rmsnorm`, `layer_0.q_proj`, `layer_0.k_proj`, and
`layer_0.v_proj`; `layer_0.rope_q` remains first unsupported. The frozen
successor emits a signed-16 Q0.15 post-MLP quantization-error carry, consumes it
exactly once before the next layer's input RMSNorm, and consumes the layer-23
carry before final RMSNorm/lm-head. The residual backbone remains signed int8.
The architecture freeze does not advance the ordered implementation prefix,
and evidence beyond `v_proj` remains historical until later authorized
implementation, verification, and canonical PPA.

## Numerical workload contract

The baseline is W4A8 with signed two's-complement int4 weights, signed
two's-complement int8 activations, and signed int32 dot-product accumulators.
Weights use one symmetric scale with zero point 0 per output channel across the
complete projection reduction. The signed-int4 stream is packed and consumed in
128-element blocks, two weights per byte, low nibble first; those blocks do not
reset or subdivide the output-channel scale.

All hardware-visible requantization uses integer multiplier/shift metadata:

`y = saturate_int8(round_to_nearest_even((acc * multiplier_s32) / 2^right_shift_u6) + output_zero_point)`

Tiled partial sums must be exactly equivalent to the independent fixed-point
reference before requantization. KV-cache K and V tensors are stored as int8
values with per-layer, per-token, per-KV-head scale metadata. RoPE, softmax,
RMSNorm, and SiLU use the fixed-point formats recorded in
`design/CHIP_SCOPE.json`.

For the current architecture contract, ordinary Q/K/V, RoPE, score, softmax,
attention-value, attention-output, and residual-bypass arithmetic remain
unchanged. At each MLP output, the successor derives a bounded Q0.15 error from
the wide common-exponent post-MLP value and the emitted signed-int8 hidden state.
That error is applied only to the immediately following input RMSNorm, with the
last error applied to final RMSNorm. Exact equations, width bounds, ordering,
validation, and address rules are specified in `design/ARCHITECTURE.md` and
`design/SPEC.md`; immutable quality targets remain unmet until executable
reference, RTL, full verification, and canonical PPA evidence prove them.

## Required trace set

Correctness and benchmark work must cover both prefill and decode. Synthetic
edge-case traces are allowed for debug, but they do not replace full-shape
traces.

| Trace class | Frozen lengths | Required evidence |
| --- | --- | --- |
| Prefill | 16, 128, 512, 2048, 8192, 32768 prompt tokens | Reference outputs, RTL outputs, cycle counts, memory bytes, stall breakdown |
| Decode | Context lengths 1, 128, 1024, 4096, 8192, 32768; generated-token runs of 1, 16, and 128 tokens | Reference outputs, RTL outputs, cycle/token, bytes/token, KV-cache traffic, stall breakdown |

The maximum-context traces may be slow, but they remain part of the frozen
definition because the public model config supports 32768 positions.

## Memory boundary

The accelerator observes an abstract, backpressured external-memory stream:

| Property | Frozen value |
| --- | --- |
| Address width | 64 bits |
| Primary data width | 128 bits |
| Byte strobes | 16 |
| Tags | 8 bits |
| Burst length field | 16 bits |
| Flow control | Independent ready/valid request, write-data, read-data, and response channels |

Benchmarks sweep effective external-memory bandwidth by injecting ready stalls
at 1, 2, 4, 8, and 16 bytes/cycle. The interface is intentionally not bound to
a DRAM PHY, cache-coherent fabric, or specific board-level memory controller.

## Quality workload

The quality gate compares the W4A8 fixed-point reference against the public BF16
Qwen2.5-0.5B reference using the same tokenizer and prompts. Later stages must
freeze raw prompt manifests and hashes before running the gate.

| Metric | Acceptance target |
| --- | --- |
| WikiText-2 raw validation perplexity | W4A8 <= 1.05x BF16 |
| C4-en frozen 512-prompt validation perplexity | W4A8 <= 1.05x BF16 |
| Frozen open lm-eval task average normalized accuracy | W4A8 drop <= 2.0 percentage points versus BF16 |

The default open lm-eval task set for the definition contract is PIQA,
HellaSwag, Winogrande, ARC-Easy, and ARC-Challenge. If a dataset, license, or
tooling issue prevents one of these tasks, the omission must be recorded as a
blocker or limitation rather than replaced silently.

## Benchmark outputs

For each trace and each memory-bandwidth point, later benchmark artifacts must
report:

| Output | Unit |
| --- | --- |
| Sustained throughput | tokens/s at 100 MHz |
| Compute cost | cycles/token |
| External-memory traffic | bytes/token and bytes/s |
| SRAM use | logical bank accesses/token and peak live bytes |
| Backpressure impact | cycles stalled by request, write-data, read-data, and response channels |
| Reuse impact | MAC utilization, shared-SFU utilization, and mux/control overhead when applicable |

These are measurement requirements, not pre-existing ACE-2 results.
