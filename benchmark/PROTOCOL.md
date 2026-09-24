# ACE-2 current-stage benchmark protocol

Generated: `2026-07-30T18:23:31Z`

This protocol binds the current `benchmark` stage to the accepted ACE-2
pre-tapeout candidate evidence. The stage remains blocked: cycle-accurate RTL
kernel scenarios and architecture estimates do not satisfy the frozen full-shape
trace requirement, and the BF16/W4A8 quality gate is blocked by the active
numerical-evidence binding. No fair open-baseline win/loss, quality pass, FPGA,
GDS, signoff, tapeout, or silicon claim is made.

## Frozen candidate and workload

| Item | Pinned value |
| --- | --- |
| Current stage source | `research/PIPELINE_STATE.json` (`current_stage=benchmark`; Manager-owned) |
| Candidate top | `ace2_shell` |
| Supported prefix | 434 ordered layer/operators through `lm_head` |
| First unsupported layer/operator | `none` |
| Model boundary | Qwen2.5-0.5B, batch 1 causal decoder inference |
| Quantization | W4A8 signed int4 weights, signed int8 activations, int32 accumulators, fixed-point requantization |
| Clock for reported rates | 100 MHz |
| Primary technology/resource scope | SKY130 HD mapped standard-cell PPA, no routed parasitics |
| Host boundary | Tokenization, prompt formatting, sampling, detokenization, application I/O, and external allocation remain host-owned |

## Fairness rules

All candidate and future baseline rows must use the same frozen workload,
quantization/quality contract, abstract 128-bit memory interface, 1/2/4/8/16
bytes-per-cycle bandwidth sweep, 100 MHz reporting point, host-offload boundary,
and evidence-retention rules. A row is excluded from win/loss comparison unless
its raw logs, source/config hashes, and unsupported operators are recorded.

The pinned Gemmini `TransposerUnitTest` subset executed two 8-bit tests
successfully. It is explicitly incompatible with the frozen model,
quantization, memory, host, technology, and measurement boundaries, so it is
engineering context only and excluded from win/loss comparison. VTA and NVDLA
remain optional unexecuted baselines. Commercial market data is not used as
direct evidence.

## Frozen gate status

| Gate | Status | Decisive evidence |
| --- | --- | --- |
| Full-shape cycle-accurate traces | **Blocked** | RTL rejects attention-score, softmax, and attention-value descriptors above context 8 and has no cross-tile exact-softmax merge command/state; `benchmark/raw/latest/trace_capability_audit.json` |
| BF16/W4A8 quality | **Blocked, not executed officially** | Active quality blocker is bound in `benchmark/raw/latest/quality_gate_preflight.json` and `benchmark/raw/latest/quality_results.json`; official evaluation remains prohibited while the blocker is active. |
| Required Gemmini evidence | **Executed incompatible subset** | Two pinned 8-bit transposer tests passed; not a fair comparison; `benchmark/raw/latest/gemmini_subset_manifest.json` |
| Makefile provenance | **Blocked** | Current Makefile no longer reconstructs to the immutable PPA/prototype packet hashes; `benchmark/raw/latest/makefile_provenance.json` |

## Measurement and estimate methods

| Scope | Method | Synchronization / repetitions | Power and energy |
| --- | --- | --- | --- |
| Kernel scenarios | Existing Icarus cycle-accurate RTL logs parsed from `verification/RESULTS.json` | Command acceptance to completion acceptance where logged; deterministic vector counts in each scenario; no statistical warmup | Energy = cycles / 100 MHz * current SKY130 mapped power |
| End-to-end prefill/decode | `design/MEMORY_MODEL.json` roofline-style model bound to the current PPA area/power | Deterministic calculation for the frozen prompt/context sets; no measured runtime repetitions | Same constant-power estimate; explicitly not full workload activity power |
| Gemmini subset | Pinned upstream Chisel transposer tests through Treadle | Two tests, one run; source/config/log hashes retained | No power or energy comparison |

## Raw evidence

| Artifact | SHA-256 |
| --- | --- |
| `benchmark/raw/latest/benchmark_binding.json` | `571a5601712c13a9ac8763be6b20d8872d5a968fc9733070775457114fb95a59` |
| `verification/RESULTS.json` | `7b774feda72dfb778f406516f57e32d373f4bec9279f778224f1319f16b8c869` |
| `ppa/RESULTS.json` | `b8974d6ea7bbdcefe508d28091269c82285362d29d5a369f95a565dc8655d86a` |
| `prototype/RESULTS.json` | `cb56e8e5560828dd3f0ffc7815aea68bf1d60861ff9280b430f6dda21df6c88a` |
| `design/MEMORY_MODEL.json` | `3bc90c54e3e7bc229d0b3613351d5283169715e6343b7d6e47cef747d5466d77` |
| `benchmark/raw/latest/trace_capability_audit.json` | `abe5c720773bdf3af377a16ff667e241ed253aa75d4775b725ccd0f7265df870` |
| `benchmark/raw/latest/quality_gate_preflight.json` | `6ddd8415b9d5b57e811425427fe82a9ae94f0e7ca15ac7e62bbb66c4c62534e0` |
| `benchmark/raw/latest/quality_results.json` | `64e15111cb02f1d20ea6ca4aeb3e13189088290a8590e62f18395dc96439292b` |
| `benchmark/raw/latest/gemmini_subset_manifest.json` | `fb47fa4ad94ee56a892e1f131c4e67f35e554caeea20e0c512c2333b83f4dea4` |
| `benchmark/raw/latest/makefile_provenance.json` | `491196ae99831a162d31c4aff21fef385f3e5a30d280f4885ab705a8309e2396` |
| `benchmark/raw/latest/packet_integrity_audit.json` | `1be8f2603f17414b1fbb32f0ba012f27f01db88b8b7bc662b8181f6d04df43e3` |

## Claim boundary

The benchmark packet reports kernel measurements, architecture estimates, one
incompatible Gemmini subset, and explicit blockers only. Architecture estimates
are not relabeled as cycle-accurate traces. A quality pass, fair open-accelerator
win/loss, FPGA measurement, routed GDS/signoff, tapeout readiness, and
fabricated-silicon measurement are not claimed.
