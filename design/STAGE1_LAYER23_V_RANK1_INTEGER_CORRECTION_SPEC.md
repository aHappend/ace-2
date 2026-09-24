# Stage 1 layer-23 V rank-1 integer correction RTL microarchitecture

This is a bounded successor path for the Fresh-L2-reviewed nonofficial
`stage1vrank1int01` candidate. It does not modify or reclassify the certified
attempt-0002 baseline, does not enter the official evaluator, and does not
authorize PPA, FPGA/U280, XRT, or product-completion claims.

## Bound evidence

| Item | Path | SHA-256 |
| --- | --- | --- |
| Candidate freeze | `evidence/candidates/stage1-layer23-v-rank1-integer-correction-v1/nonofficial-candidate-0001/candidate-freeze.json` | `c00e607a567ebc1f02854e9763f9722eeb985bfaf97962401f48414c73dfd0f4` |
| Candidate runner | `tools/run_stage1_layer23_v_rank1_integer_candidate.py` | `e29892e9fe9af5b22959c37121364cc6b83608892098692464baf20fa6839c0a` |
| Candidate result | `evidence/candidates/stage1-layer23-v-rank1-integer-correction-v1/nonofficial-candidate-0001/result.json` | `fa4547a5884a070966a5c0c1fea8997b2aef51d22d40431df34f719cce6ca037` |
| Fresh-L2 review | `/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/32444a7fc5ab/round-0001.json` | `80881054b5dc3b06f83027dd548446a8b177b7c56add0cdfd306c4f737cbfc06` |
| Sidecar integration evidence | `evidence/verification/stage1-layer23-v-rank1-integration-v1/latest/RESULT.json` | `36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1` |
| Sidecar Fresh-L2 review | `/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/271efb2d9974/round-0001.json` | `52b06dcbbd21e00131582bc2deea7c35bae64da53fb9dbdaf979961c508f1fa9` |

Fresh-L2 status is `done` and independently reproduced the rank-1 integer
candidate. The successor therefore freezes only the audited integer datapath:
signed-int8 input and rank factors, signed-int32 accumulators, frozen
Scale32-derived signed-int32 multipliers and u6 right shifts, signed
round-to-nearest ties-to-even requantization, signed-int8 saturation, and final
saturating add in the retained V-output quanta.

## RTL interface

Top module: `ace2_layer23_v_rank1_integer_correction_core`.

| Port | Direction | Width/signedness | Meaning |
| --- | --- | --- | --- |
| `clk_i` | input | 1 | Single rising-edge clock |
| `rst_ni` | input | 1 | Active-low asynchronous reset |
| `clear_i` | input | 1 | Synchronous transaction/output clear; preserves configured tables |
| `enable_i` | input | 1 | Enables start acceptance only after configuration is complete |
| `cfg_valid_i`/`cfg_ready_o` | input/output | 1 | Configuration write handshake, legal only while disabled and idle |
| `cfg_kind_i` | input | 3 | `0=input_to_rank`, `1=rank_to_channel`, `2=second_multiplier`, `3=second_shift`, `4=first_multiplier`, `5=first_shift` |
| `cfg_index_i` | input | 10 unsigned | Element index for vector tables; scalar configs require index 0 |
| `cfg_data_i` | input | signed 32 | Sign/zero-extended configuration payload |
| `cfg_error_o` | output | 1 | Sticky invalid configuration write until clear/reset |
| `config_complete_o` | output | 1 | All required factor, multiplier, and shift records loaded with no config error |
| `start_valid_i`/`start_ready_o` | input/output | 1 | Starts one 896-to-128 correction when enabled and fully configured |
| `input_valid_i`/`input_ready_o` | input/output | 1 | Streams 896 signed-int8 input payload bytes in index order |
| `input_s8_i` | input | signed 8 | Current input byte in retained candidate input quanta |
| `baseline_valid_i`/`baseline_ready_o` | input/output | 1 | Streams 128 unchanged baseline V output bytes in channel order |
| `baseline_v_s8_i` | input | signed 8 | Current baseline V output byte in retained V-output quanta |
| `output_valid_o`/`output_ready_i` | output/input | 1 | Corrected output stream handshake |
| `output_channel_o` | output | 7 unsigned | Channel index `0..127` for the current output |
| `output_last_o` | output | 1 | High only on channel 127 |
| `rank_accumulator_s32_o` | output | signed 32 | First-stage dot-product accumulator |
| `rank_rounded_s32_o` | output | signed 32 | First-stage post-multiplier rounded value before int8 saturation |
| `rank_intermediate_s8_o` | output | signed 8 | Saturated rank scalar |
| `rank_valid_o` | output | 1 | Rank scalar is valid while output channels are being emitted |
| `correction_accumulator_s32_o` | output | signed 32 | Per-channel rank scalar times rank-to-channel factor |
| `correction_rounded_s32_o` | output | signed 32 | Per-channel post-multiplier rounded correction |
| `correction_s8_o` | output | signed 8 | Per-channel saturated correction byte |
| `corrected_v_s8_o` | output | signed 8 | Saturating add of baseline V and correction bytes |
| `input_saturation_o` | output | 1 | Always false for this already-quantized input stream |
| `rank_saturation_o` | output | 1 | Rank-intermediate int8 saturation indicator |
| `correction_saturation_o` | output | 1 | Current channel correction int8 saturation indicator |
| `add_saturation_o` | output | 1 | Current channel final add saturation indicator |
| `descriptor_error_o` | output | 1 | Invalid configuration/transaction boundary |
| `numeric_overflow_o` | output | 1 | Internal signed-width violation |

## Cycle behavior

1. Configuration writes occur only when `enable_i=0`, the core is idle, and
   `cfg_ready_o=1`. `clear_i` clears sticky errors and in-flight state while
   preserving configured tables. `rst_ni` clears all configuration and state.
2. A transaction is accepted only on `start_valid_i && start_ready_o`; the core
   then consumes exactly 896 input bytes, one per cycle when
   `input_valid_i && input_ready_o`.
3. The first-stage accumulator is requantized with the frozen first multiplier
   and right shift using signed ties-to-even rounding, then saturated to one
   signed-int8 rank scalar. `rank_valid_o` is asserted for the output phase.
4. The core then accepts 128 baseline V bytes, one channel at a time, and emits
   per-channel correction accumulator, rounded correction, correction byte, and
   corrected V byte. `output_valid_o` holds all output payload fields stable
   until `output_ready_i=1`.
5. After channel 127 is accepted, the core returns to idle and `start_ready_o`
   may accept the next configured transaction.

## Acceptance matrix

| Scenario | Observable acceptance |
| --- | --- |
| Nominal | Synthetic positive vector produces exact rank, correction, and corrected output without saturation or errors |
| Negative/tie | Synthetic negative half-LSB tie rounds to even and matches the software integer reference |
| Saturation | Positive and negative synthetic vectors assert rank/correction/final-add saturation and clamp to signed-int8 bounds |
| Frozen 34 positions | Every frozen position matches the reviewed payload byte-for-byte for rank scalar, 128 correction bytes, and 128 corrected V bytes |
| Backpressure | Output payload and `output_last_o` hold stable while `output_ready_i=0` |
| Reset/clear/config | Reset clears configuration; start is not accepted before complete config; invalid config raises `cfg_error_o`; clear returns the core to a clean boundary |

## Descriptor-bound sidecar integration

The shell integration is an explicitly enabled, descriptor-bound sidecar path for
Qwen2.5-0.5B layer 23 `self_attn.v_proj` output in fused QKV execution. It
preserves the public `ace2_shell` port contract and adds only elaboration
parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ENABLE_LAYER23_V_RANK1_SIDECAR` | `0` | Disabled bypass by default; when `0`, the projection write data path is byte-identical to the pre-integration baseline |
| `LAYER23_V_RANK1_CONFIG_VALID` | `1` | Configuration-complete guard for the frozen generated payload; when `0` and the exact guarded descriptor is presented, the shell rejects the descriptor rather than silently applying a partial correction |

The sidecar admits only this exact descriptor/profile and V phase:

| Field | Required value |
| --- | --- |
| Opcode | `ACE2_OPCODE_FUSED_QKV` (`8'h0b`) |
| Flags | `8'h00` |
| Layer | `8'd23` |
| Shape | `m=1`, `n=896`, `k=896` |
| Profile | `HIDDEN_SIZE=896`, `QKV_KV_OUTPUTS=128`, `ACE2_NUM_LAYERS=24`, `LANES=16`, `ACT_WIDTH=8` |
| `src0` | `64'h0000001000000700` |
| `src1` | `64'h000000010a384000` |
| `dst` | `64'h0000001000000a80` |
| `scale` | `64'h0000000200472800` |
| `scratch` | `64'h0000000000000000` |
| QKV phase | V only (`QKV_PHASE_V`) |

The frozen factor and Scale32 metadata payload is generated from the accepted
integer candidate into
`rtl/generated/ace2_layer23_v_rank1_integer_correction_payload.svh`; no refit or
rescale is performed. The sidecar consumes the cached 896-byte activation used
by fused QKV, corrects only the 128-byte V projection beats, and feeds the
projection write data mux before downstream KV/attention consumption. Nonmatching
layers, descriptors, profiles, Q/K phases, disabled sidecar elaborations, and
invalid configuration all preserve the baseline V beat.

## Integration evidence

Persistent nonofficial evidence for the lint-clean descriptor-bound integration
is recorded at
`evidence/verification/stage1-layer23-v-rank1-integration-v1/latest/RESULT.json`,
SHA-256
`36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1`; its
`SHA256SUMS` hashes to
`980ae593d53ce9a833b520f4d992b7a29aec8063b588d21632c05e7617f8afa2`.

| Check | Result |
| --- | --- |
| Software reference | Passes all 34 frozen positions and all eight accepted multi-prompt records reconstructed as 290 position vectors |
| Isolated core Icarus | Passes four synthetic nominal/tie/saturation cases, invalid config, backpressure hold, and all 34 frozen positions x 128 outputs |
| Sidecar Icarus | Passes disabled bypass, invalid-config fail-closed, non-layer guard, held-output stability, all 34 frozen positions, and all 290 generalization positions x 8 beats |
| Shell guard Icarus | With the sidecar enabled and config invalid, the exact guarded descriptor completes with error and no memory traffic |
| Strict Verilator lint | Isolated core and sidecar have no diagnostics; shell elaborations with default-disabled, enabled/config-valid, and enabled/config-invalid sidecar are clean when known pre-existing shell `DECLFILENAME`/`UNUSED` baseline warning classes are suppressed |
| Baseline shell drift | Existing `o_proj` Icarus smoke regression passes with the sidecar present and default-disabled |

Fresh-L2 mission `271efb2d9974` returned `done` for this integration. The
reviewer independently reproduced RESULT SHA-256
`36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1`, byte-identical
scratch regeneration for the complete generated corpus, software and RTL
agreement over 34 frozen and 290 generalization positions, default-disabled
baseline preservation, exact V-phase descriptor guarding, invalid-config
fail-closed behavior before memory traffic, and zero new/changed shell-lint
diagnostics beyond the preserved six-warning baseline.

The full unsuppressed `ace2_shell` strict lint log still exits on pre-existing
baseline warnings (`DECLFILENAME` for the local attention accumulator module and
`UNUSED` warnings in existing shell/SILU code). No unresolved warning is
attributable to the new sidecar path in the focused or baseline-filtered lint
logs. This evidence did not run official preflight/chat/attempt-0003, did not
replay attempt-0002, and did not run synthesis, PPA, OpenSTA, U280, XRT, or
Stage 2 work. The reviewed sidecar evidence is not a successful arbitrary-text
Stage 1 product demo and does not grant official execution or deployment
authority.
