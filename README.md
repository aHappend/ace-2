# ACE-2: ARGUS Compute Engine

> **Continue or recover this work:** start with [AGENTS.md](AGENTS.md) and the
> [Chinese continuation handoff](docs/continuation/HANDOFF.zh-CN.md). The dated
> [progress snapshot](docs/continuation/progress-2026-09-24.json) separates
> accepted evidence, running work, and host-only dependencies.

ACE-2 is an evidence-driven, pre-tapeout ASIC accelerator project for
batch-1 Qwen2.5-0.5B inference. It explores a complete W4A8 system boundary:
the numerical reference, synthesizable SystemVerilog, command/runtime
integration, verification, and reproducible physical-design evidence.

ACE-2 is developed with **[Argus](https://github.com/lbx154/Argus)**, the
open-source long-running agent harness used to plan, execute, review, and
preserve evidence across this engineering campaign.

## Project contract

| Item | Target |
| --- | --- |
| Model | Qwen2.5-0.5B, batch 1 |
| Quantization | W4A8 with explicit scale, rounding, saturation, accumulator, and KV-cache formats |
| Primary flow | SKY130 with public, reproducible tooling |
| Timing floor | 100 MHz |
| Non-SRAM area cap | 2.0 mm² |
| Delivery level | Computer-verifiable pre-tapeout design |

The design covers host-visible control, command processing, external-memory
streaming, banked SRAM interfaces, projection/attention/MLP datapaths,
normalization, RoPE, KV cache, softmax, residual and requantization paths, plus
an executable reference and RTL verification flow.

## Current status

ACE-2 is active research, not a tapeout, fabricated chip, or completed FPGA
deployment. The local Stage-1 attempt 0002 passed an arbitrary UTF-8,
two-generated-token chat run through 35 causal positions and 840 RTL layer
executions with persistent K/V state and zero integer-reference mismatches.
See [the Stage-1 result](docs/results/STAGE1_RTL_CHAT_V74.md).

The demonstrated model boundary is a BF16 Qwen2.5-0.5B-Instruct parent plus
the trained ACE-2 LoRA v4 checkpoint-176 adapter, mapped into ACE-2 W4A8
integer/RTL execution semantics. It is not a separately trained W4A8
checkpoint. No synthesis, PPA, bitstream, PCIe/XRT, Alveo U280, or
deployed-hardware result is claimed.

## Repository map

| Path | Contents |
| --- | --- |
| `design/` | Architecture, specification, interfaces, targets, and evidence contracts |
| `rtl/` | Synthesizable SystemVerilog implementation |
| `reference/` | Independent numerical reference |
| `verification/` | Testbenches, Verilator harnesses, vectors, and checks |
| `tools/` | Artifact generation, audits, runtime, and evidence-binding utilities |
| `flow/`, `constraints/`, `ppa/` | Synthesis, timing, physical-design, and PPA flow |
| `evidence/` | Raw and bound evidence supporting accepted claims |
| `benchmark/` | Benchmark definitions and comparison artifacts |

Start with [the mission](MISSION.md), [architecture](design/ARCHITECTURE.md),
and [specification](design/SPEC.md). The specification is authoritative for
the current product and acceptance boundaries.

## Reproducible entry points

```bash
# CPU_SOFTWARE_REFERENCE_ONLY_NOT_RTL_COMPLETION: run the local CPU-only
# Qwen2.5-0.5B-Instruct tokenizer/forward/KV-cache/decode reference.
python3 tools/qwen25_cpu_reference_chat.py \
  --prompt 'Explain in one sentence why cycle-accurate verification matters.' \
  --output-dir build/cpu_reference_chat/operator

# Run two deterministic smoke prompts plus arbitrary operator text in one load,
# then verify the machine-readable evidence.
python3 tools/qwen25_cpu_reference_chat.py \
  --smoke \
  --prompt 'Give one concise reason to preserve failing seeds.' \
  --output-dir build/cpu_reference_chat/latest
python3 verification/verify_qwen25_cpu_reference_chat.py \
  --manifest build/cpu_reference_chat/latest/manifest.json

# Run one fresh arbitrary UTF-8 prompt through canonical Qwen chat tokenization,
# one causal Icarus-backed W4A8 session, persistent RTL-published K/V, sealed
# trace verification, and the independent full-chain oracle. Omitting
# ACE2_CHAT_MAX_NEW_TOKENS uses the four-token default with three decode
# transitions. All three output paths must be fresh.
make ace2-chat \
  ACE2_CHAT_ARGS="--prompt 'Why test digital systems?'" \
  ACE2_CHAT_OUTPUT=reports/verification/ace2-chat-four-token-interface-activation-0001 \
  ACE2_CHAT_VERIFICATION_OUTPUT=reports/verification/ace2-chat-four-token-interface-verification-0001 \
  ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT=reports/verification/ace2-chat-four-token-interface-full-chain-oracle-0001

# Run the two-turn persistent-K/V RTL diagnostic. The output namespace must be
# fresh. Omitting the token-count variable requests four tokens per turn.
make persistent-conversation-rtl-diagnostic \
  PERSISTENT_CONVERSATION_RTL_OUTPUT=reports/verification/<fresh-namespace>

# Equivalent direct CLI with an explicit supported count.
python3 scripts/run_persistent_conversation_rtl_diagnostic.py \
  --output reports/verification/<fresh-namespace> \
  --max-new-tokens 8

# Non-interactive prompt input without placing prompt text in argv.
python3 tools/ace2_chat_demo.py \
  --prompt-file prompt.txt \
  --max-new-tokens 4 \
  --output build/ace2_chat_demo/stage1-product-file-fresh

# Diagnostic tokenizer/package preflight only; this is not product acceptance.
make ace2-chat-prepare < prompt.txt

# Reproduce the active specification-stage and V4 aggregate-diagnosis checks.
make specification-stage-audit

# Core RTL checks
make rtl-lint
make rtl-sim

# Build the Verilated full-Qwen runtime
make full-qwen-runtime-build

# Offline file-backed grouped-W4A8 generation. Prompt text is read as strict
# UTF-8 from stdin and never needs to appear in argv.
python3 tools/run_file_backed_w4a8_chat.py \
  --package /path/to/model.ace2w4m1 \
  --max-new-tokens 8 < prompt.txt

# Equivalent prompt-file input.
python3 tools/run_file_backed_w4a8_chat.py \
  --package /path/to/model.ace2w4m1 \
  --prompt-file prompt.txt \
  --max-new-tokens 8

# Check publication/evidence integrity
make publication-integrity-check

# Validate generated Qwen2.5 0.5B/1.5B/3B/7B hardware descriptors.
# Larger-model results are structural checks, not RTL execution claims.
make model-hardware-contract-check
```

The CPU command is an offline software reference only. It defaults to the
already-local immutable Qwen2.5-0.5B-Instruct snapshot and records the exact
command, model and source hashes, runtime/CPU versions, prompts and token IDs,
explicit one-token KV-cache reuse, decoded output, latency, peak RSS, stdout,
and smoke status. It does not establish Stage-1 completion, W4A8/RTL
agreement, accelerator latency, or S7/U280 execution.

The file-backed W4A8 command is also CPU-host software, not an RTL-completion
or hardware-performance claim. It performs no network access: it verifies the
complete `ACE2W4M1` framing and all 169 grouped projection payloads before
constructing a fresh runtime from the pinned local Qwen2.5-0.5B-Instruct
snapshot and frozen calibration contract. Success is one JSON object containing
decoded text, input and generated token IDs, termination reason, per-step
integer-logit hashes, and ordered 24-layer key/value/Scale32 cache records.
Malformed, reordered, trailing, or truncated packages and invalid UTF-8 input
exit nonzero without emitting a success object.

The Stage-1 chat path uses the canonical pinned Qwen2.5-0.5B-Instruct
tokenizer/chat template and a single persistent prefill/decode session.
Simulator-emitted layer-23 corrected-V bytes replace the host cache rows before
downstream positions. Omitting `ACE2_CHAT_MAX_NEW_TOKENS` from the primary
`ace2-chat` target requires exactly four readable generated tokens and three
decode transitions. Setting `ACE2_CHAT_MAX_NEW_TOKENS=8` selects the supported
eight-token contract with seven decode transitions; other counts are rejected.
Both modes require continuous K/V growth across all 24 layers, no software
transformer or logits fallback, and zero independent full-chain oracle
mismatches. The fresh output directories record separate measured Icarus
compile, host model/orchestration, simulation, total wall latency, sealed-trace
verification, and independent-oracle results. The target rejects software
fallback and unreadable output rather than printing either as a product
response.

The persistent-conversation Make target defaults
`PERSISTENT_CONVERSATION_RTL_MAX_NEW_TOKENS` to `4`; the direct CLI defaults
`--max-new-tokens` to the same value. Both accept only integers from `2`
through `8`, require `PERSISTENT_CONVERSATION_RTL_OUTPUT`/`--output` to name a
path that does not exist, and fail before model or RTL execution for an
unsupported count. A successful run generates exactly the requested count on
each of two turns while retaining all 24 K/V layers without prefix
recomputation or software transformer/logits fallback.

Authenticated sealed 4-, 6-, and 8-token evidence can be compared without
replaying any attempt:

```bash
make ace2-chat-token-scaling-profile \
  ACE2_CHAT_TOKEN_SCALING_OUTPUT=build/ace2_chat_token_scaling/profile.json
```

The deterministic JSON separates raw compile, host/orchestration,
RTL-simulation, independent-oracle, wall-time, authenticated storage, and
available retained peak-RSS metrics. Additive metrics are also divided by
generated tokens and by decode transitions (`generated_tokens - 1`), with
explicit units. Every value is labeled as computer-local simulation/host
profiling; the command fails closed on a missing seal or member, mixed attempt
or source binding, fallback, token/timing mismatch, or incomplete coverage.

The deterministic prompt-suite regression runs three distinct prompts only in
fresh per-case namespaces:

```bash
make prompt-suite-rtl-oracle-regression \
  ACE2_CHAT_PROMPT_SUITE_OUTPUT=reports/verification/<fresh-suite-namespace>
```

Suite `PASS` requires 12 generated tokens, continuous 24-layer K/V growth,
complete nonzero independent-oracle comparisons, zero byte and selected-token
mismatches, and `software_transformer_or_logits_fallback=false`. See
`verification/RTL_CHAT_ORACLE_REGRESSION.md` for the focused compile/test
command and the simulation/host-only evidence boundary.

Some flows require tools or licensed hardware that are not bundled with this
repository. A missing tool, model artifact, FPGA board, or external service is
reported as a prerequisite rather than represented as a successful run.

## Argus

Argus provides the long-horizon engineering loop behind ACE-2: backlog and
budget supervision, reusable skill matching, engineer execution, independent
review, checkpoints, and evidence-aware replanning.

- Source: <https://github.com/lbx154/Argus>
- ACE-2 remains the hardware project; Argus is the general agent harness used
  to develop and supervise it.
