# ACE-2 Phase-2 model hardware parameterization contract

This is a specification-stage, additive successor contract. It preserves
the accepted 0.5B v1 model-hardware descriptors and records no RTL, model,
simulator, synthesis, FPGA, U280, PPA, timing, or product-completion result.

## Unified descriptor matrix

| Model | Primary tier | Layers | Hidden | MLP | Q heads/KV heads | Head dim | Vocab | Max context | Compute tiles/memory channels | Current gaps |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| qwen2.5-0.5b | ace2_nano | 24 | 896 | 4864 | 14/2 | 64 | 151936 | 32768 | 1/1 | 0 |
| qwen2.5-1.5b | ace2_nano | 28 | 1536 | 8960 | 12/2 | 128 | 151936 | 131072 | 2/2 | 10 |
| qwen2.5-3b | ace2_edge | 36 | 2048 | 11008 | 16/2 | 128 | 151936 | 32768 | 4/4 | 9 |
| qwen2.5-7b | ace2_pro | 28 | 3584 | 18944 | 28/4 | 128 | 152064 | 131072 | 8/8 | 12 |

## Bound 0.5B hard-coded inventory

| Surface | Source | Symbol | Current binding |
| --- | --- | --- | --- |
| rtl | rtl/generated/ace2_model_parameters.svh:6 | ACE2_HIDDEN_SIZE | 896 |
| rtl | rtl/generated/ace2_model_parameters.svh:12 | ACE2_VOCAB_SIZE | 151936 |
| rtl | rtl/ace2_shell.sv:136 | MLP_INTERMEDIATE_SIZE | 4864 |
| rtl | rtl/ace2_shell.sv:157 | QKV_KV_OUTPUTS | 128 |
| rtl | rtl/ace2_shell.sv:167 | ATTN_HEAD_DIM | 64 |
| rtl | rtl/ace2_shell.sv:187 | ROPE_MAX_SEQUENCE_POSITION | 32767 |
| rtl | rtl/ace2_rmsnorm_core.sv:4 | HIDDEN_SIZE parameter default | 896 |
| rtl | rtl/ace2_w4a8_proj_core.sv:4 | K_SIZE parameter default | 896 |
| rtl | rtl/ace2_attention_score_core.sv:4 | HEAD_DIM parameter default | 64 |
| rtl | rtl/ace2_attention_compose_core.sv:5 | HEAD_DIM parameter default | 64 |
| rtl | rtl/ace2_attention_compose_core.sv:6 | CONTEXT_MAX parameter default | 32768 |
| rtl | rtl/ace2_qk_residual_cross_term_core.sv:441 | HEAD_DIM parameter default | 64 |
| rtl | rtl/ace2_v_residual_value_correction_core.sv:216 | CONTEXT_MAX parameter default | 32768 |
| rtl | rtl/ace2_absolute_rope_online_attention_core.sv:9 | LANE_COUNT parameter default | 64 |
| rtl | rtl/ace2_native_accumulator_tagged_attention_core.sv:283 | MAX_LANES parameter default | 64 |
| rtl | rtl/ace2_tile_bfp_score_attention_core.sv:8 | MAX_KEYS parameter default | 64 |
| rtl | rtl/ace2_tile_max_delta_attention_core.sv:8 | MAX_KEYS parameter default | 64 |
| rtl | rtl/ace2_cross_layer_error_carry_core.sv:371 | ace2_error_carry_state_core HIDDEN_SIZE default | 896 |
| rtl | rtl/ace2_cross_layer_error_carry_core.sv:571 | ace2_carry_aware_rmsnorm_core HIDDEN_SIZE default | 896 |
| rtl | rtl/ace2_layer16_post_attention_rmsnorm_runtime_metadata_core.sv:4 | HIDDEN_SIZE parameter default | 896 |
| rtl | rtl/ace2_layer16_group1_scale32_core.sv:309 | HIDDEN_SIZE parameter default | 896 |
| rtl | rtl/ace2_layer16_group4_scale32_core.sv:309 | HIDDEN_SIZE parameter default | 896 |
| rtl | rtl/ace2_layer17_rmsnorm_output_grouped_scale32_core.sv:4 | HIDDEN_SIZE parameter default | 896 |
| model_packer | tools/build_full_qwen_image_v2.py:52 | Qwen2.5-0.5B source and core dimensions | Qwen/Qwen2.5-0.5B, hidden=896, head_dim=64, kv_heads=2, layers=24 |
| model_packer | tools/build_full_qwen_image_v2.py:81 | EXPECTED_REGION_BYTES | 0.5B region byte totals |
| model_packer | tools/verify_full_qwen_image_v2.py:49 | independent verifier source and dimensions | Qwen/Qwen2.5-0.5B, hidden=896, head_dim=64, kv_heads=2, layers=24 |
| model_packer | tools/audit_full_qwen_image_contract.py:172 | RMSNorm/gain and accepted command-count audit | 896 gains, 13,914 accepted two-token commands |
| command_generation | tools/run_full_qwen_command_schedule_runtime.py:37 | accepted runtime model path and package dimensions | Qwen2.5-0.5B, [151936, 896], layers=24, tiles=4748 |
| command_generation | tools/ace2_chat_demo.py:118 | ACE2RT2 package and fused-QKV constants | hidden=896, max_context=32768, kv_bytes=272, LM tiles=4748 |
| command_generation | tools/ace2_chat_demo.py:103 | FUSED_QKV offsets/strides | 0.5B Q/K/V byte offsets and 0x71c000 layer stride |
| command_generation | tools/run_fused_qkv_runtime_prefix.py:252 | fused-prefix runtime_preflight geometry | [151936, 896], max_context=32768, kv_stride=272 |
| host_runtime | verification/verilator/ace2_shell_runtime_main.cpp:82 | ACE2RT2 C++ geometry constants | layers=24, hidden=896, vocab=151936, max_context=32768 |
| host_runtime | verification/verilator/ace2_shell_runtime_main.cpp:613 | C++ schedule loops and fused-QKV legality | 24 layers, 14 heads, fused n/k=896, fixed spans |
| host_runtime | verification/verilator/ace2_runtime_identity_profile.h:29 | runtime identity profile embedding offset | 0.5B image/model hashes and embedding_offset=32288 |

## Narrow successor implementation plan

| Order | Task | Decisive acceptance check |
| ---: | --- | --- |
| 1 | descriptor-bound-sv-parameter-package | 0.5B focused regressions still pass and generated parameter values match design/model_hardware_contracts/phase2/qwen2.5-0.5b.json. |
| 2 | qkv-gqa-and-attention-geometry-parameterization | Single-layer Q/K/V, RoPE, attention-score, attention-value, and fused-QKV reference checks pass for all four descriptors. |
| 3 | packer-and-command-layout-parameterization | No-execution package/layout checks generate consistent address maps for 0.5B, 1.5B, 3B, and 7B and reject a descriptor/image mismatch. |
| 4 | host-runtime-contract-hash-binding | Runtime package metadata reconstructs the descriptor hash and fails closed on hidden_size, kv_width, max_context, or image-region drift. |
| 5 | representative-multisize-rtl-reference-checks | Each descriptor passes representative focused RTL/reference checks and existing 0.5B regressions remain green after every RTL change. |

## Claim boundary

The descriptor and inventory checks are static and deterministic. Larger-model
records are configuration contracts only until successor RTL, packer,
command generation, host runtime, and focused RTL/reference checks are
implemented and independently reviewed.
