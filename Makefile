PYTHON ?= python3
ARGUS_SKILL_PYTHON ?= $(PYTHON)
ACE2_AMLT_BIN ?= amlt
ACE2_AMLT_PROJECT ?=
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
ORFS_IMAGE ?= openroad/orfs@sha256:3bc303869d5e4caac8f72c854f2b1614c726b2961bbb372f54bc8fbc0e725e71
ACE2_CHAT_OUTPUT ?= reports/verification/ace2-chat-four-token-interface-activation-0001
ACE2_CHAT_VERIFICATION_OUTPUT ?= reports/verification/ace2-chat-four-token-interface-verification-0001
ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT ?= reports/verification/ace2-chat-four-token-interface-full-chain-oracle-0001
ACE2_CHAT_MAX_NEW_TOKENS ?= 4
ACE2_CHAT_TIMEOUT_CYCLES ?= 100000000
ACE2_CHAT_TIMEOUT_SECONDS ?= 72000
ACE2_CHAT_ARGS ?=
ACE2_CHAT_STORAGE_CHECK_OUTPUT ?= build/ace2_chat_demo/storage-lifecycle-check
ACE2_CHAT_ORACLE_REGRESSION_OUTPUT ?= reports/rtl-chat-independent-oracle-regression-0006
ACE2_CHAT_MULTITOKEN_OUTPUT ?= reports/verification/persistent-kv-multitoken-rtl-chat-0001
ACE2_CHAT_MULTITOKEN_VERIFICATION_OUTPUT ?= reports/verification/persistent-kv-multitoken-rtl-chat-verification-0001
ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT ?= reports/verification/persistent-kv-multitoken-full-chain-independent-oracle-0001
ACE2_CHAT_MULTITOKEN_MAX_NEW_TOKENS ?= 4
ACE2_CHAT_MULTITOKEN_TIMEOUT_SECONDS ?= 72000
ACE2_CHAT_MULTITOKEN_ARGS ?=
ACE2_CHAT_PROMPT_SUITE ?= tests/prompt_suite_rtl_oracle_strict_fresh_v1.json
ACE2_CHAT_PROMPT_SUITE_OUTPUT ?= reports/verification/prompt-suite-rtl-oracle-strict-fresh-0001
ACE2_CHAT_PROMPT_SUITE_TIMEOUT_SECONDS ?= 72000
ACE2_FRESH_MULTITOKEN_DIAGNOSTIC_OUTPUT ?= reports/verification/eight-token-rtl-oracle-diagnostic-0005
ACE2_FRESH_MULTITOKEN_DIAGNOSTIC_TIMEOUT_SECONDS ?= 72000
ACE2_FRESH_MULTITOKEN_DIAGNOSTIC_ARGS ?=
ACE2_CHAT_TOKEN_SCALING_OUTPUT ?= build/ace2_chat_token_scaling/profile.json
ACE2_CHAT_TOKEN_SCALING_4_ATTEMPT ?= reports/verification/ace2-chat-four-token-interface-activation-0001
ACE2_CHAT_TOKEN_SCALING_4_ORACLE ?= reports/verification/ace2-chat-four-token-interface-full-chain-oracle-0001
ACE2_CHAT_TOKEN_SCALING_6_ATTEMPT ?= reports/verification/six-token-rtl-oracle-round11/rtl-attempt
ACE2_CHAT_TOKEN_SCALING_6_ORACLE ?= reports/verification/six-token-rtl-oracle-round11/full-chain-oracle
ACE2_CHAT_TOKEN_SCALING_8_ATTEMPT ?= reports/verification/eight-token-rtl-oracle-diagnostic-0005/rtl-attempt
ACE2_CHAT_TOKEN_SCALING_8_ORACLE ?= reports/verification/eight-token-rtl-oracle-diagnostic-0005/independent-oracle
PRODUCTION_W4A8_PACKAGE ?= build/production-w4a8-package-20260921T171100Z/qwen2.5-0.5b-instruct.ace2w4m1
PRODUCTION_W4A8_PACKAGE_SHA256 ?= e0e5e8bbd78a6741fb556dd6bb8725e824eef476c7b73b5f58c8a942290e5ada
PRODUCTION_W4A8_BATCH_QUALIFICATION_OUTPUT ?= reports/verification/production-w4a8-batch-qualification-0001
PRODUCTION_W4A8_BATCH_MAX_NEW_TOKENS ?= 2
PRODUCTION_W4A8_BATCH_TIMEOUT_SECONDS ?= 7200
PRODUCTION_W4A8_CONVERSATION_QUALIFICATION_OUTPUT ?= reports/verification/production-w4a8-conversation-qualification-0001
PRODUCTION_W4A8_CONVERSATION_MAX_NEW_TOKENS ?= 2
PERSISTENT_CONVERSATION_RTL_OUTPUT ?= reports/verification/persistent-conversation-active-rtl-diagnostic-0005
PERSISTENT_CONVERSATION_RTL_MAX_NEW_TOKENS ?= 4
PERSISTENT_CONVERSATION_RTL_TURNS ?= 2
RTL_SOURCES := rtl/ace2_pkg.sv rtl/ace2_rmsnorm_core.sv rtl/ace2_dynamic_scale32_core.sv rtl/ace2_w4a8_proj_core.sv rtl/ace2_rope_core.sv rtl/ace2_dynamic_rope_head_core.sv rtl/ace2_fixed_q7_rope_score_core.sv rtl/ace2_relative_rope_score_fusion_core.sv rtl/ace2_attention_score_core.sv rtl/ace2_softmax_core.sv rtl/ace2_attention_compose_core.sv rtl/ace2_silu_gate_core.sv rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv rtl/ace2_shell.sv
RTL_GENERATED_HEADERS := rtl/generated/ace2_model_parameters.svh
RTL_INCLUDE_FLAGS := -Irtl
VERILATOR_SHELL_OPROJ_BIN := build/verilator_shell_oproj/Vace2_shell_oproj_harness
VERILATOR_FULL_QWEN_RUNTIME_BIN := build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness
ATTN_RETIME_EVIDENCE_DIR := evidence/rtl_attn_max_predicate_retime_v1
DYNAMIC_SCALE32_BOUNDARY_EVIDENCE_DIR := evidence/verification/chat-v2-dynamic-scale32-qkv-current-tree-v2
ATTN_RETIME_PHASE ?= postchange
SHELL_SMOKE_OPCODE ?= o_proj
SHELL_SMOKE_OPCODE_HEX_o_proj := 01
SHELL_SMOKE_OPCODE_HEX_vector_family := 08
SHELL_SMOKE_OPCODE_HEX := $(SHELL_SMOKE_OPCODE_HEX_$(SHELL_SMOKE_OPCODE))

.PHONY: model-hardware-contract-check grouped-w4a8-payload-conformance file-backed-w4a8-chat-test production-w4a8-batch-qualification production-w4a8-batch-qualification-test production-w4a8-conversation-qualification production-w4a8-conversation-qualification-test architecture-contract-check architecture-proposal-bind architecture-proposal-check architecture-proposal-review architecture-proposal-review-check architecture-successor-freeze architecture-successor-freeze-check baseline-harness-certification-bind baseline-harness-certification-check cross-layer-error-carry-architecture-review cross-layer-error-carry-architecture-review-check cross-layer-error-carry-vectors cross-layer-error-carry-metadata cross-layer-error-carry-rtl-preflight cross-layer-error-carry-rtl-bind cross-layer-error-carry-rtl-review cross-layer-error-carry-rtl-review-check dynamic-scale32-rtl-preflight dynamic-scale32-rtl-bind dynamic-scale32-rtl-bind-check dynamic-scale32-rtl-review dynamic-scale32-rtl-review-check dynamic-scale32-rtl-engineer-recovery dynamic-scale32-rtl-engineer-recovery-check environment-contract-check environment-preflight vectors vector-rmsnorm vector-residual vector-mlp-residual vector-post-attention-rmsnorm vector-projection vector-rope vector-dynamic-rope vector-attention-score vector-attention-compose vector-qk-residual-cross-term vector-v-residual-value-correction vector-down-projection-residual-fusion \
	vector-softmax vector-attention-value vector-silu-gate vector-absolute-rope-online vector-native-accumulator-tagged rtl-lint rtl-sim rtl-sim-rmsnorm \
	rtl-sim-projection rtl-sim-rope rtl-sim-dynamic-rope rtl-sim-rope-shell rtl-sim-attention-score rtl-sim-softmax rtl-sim-attention-compose \
	rtl-sim-silu-gate rtl-sim-shell rtl-sim-attention-score-shell rtl-sim-attn-max-predicate-focused rtl-sim-attn-max-predicate-shell-trace rtl-sim-qproj-stride rtl-sim-mlp-up rtl-sim-mlp-down rtl-sim-mlp-residual rtl-sim-silu-shell rtl-sim-layer-sweep rtl-sim-shell-smoke rtl-sim-shell-verilator rtl-sim-shell-agreement rtl-runtime-package-test rtl-sim-runtime-rmsnorm-min-latency full-qwen-runtime-build \
	rtl-sim-runtime-first-kv-reference rtl-sim-runtime-dynamic-scale32-first-boundary chat-first-residual-rtl-boundary dynamic-scale32-first-boundary-synth-sta \
	rtl-sim-final-rmsnorm rtl-sim-lm-head dynamic-rope-focused fixed-q7-focused fixed-q7-fidelity-focused relative-rope-score-focused absolute-rope-online-focused projection-shadow-focused tile-max-delta-focused tile-bfp-focused native-accumulator-tagged-focused qk-residual-cross-term-preflight v-residual-value-correction-preflight down-projection-residual-fusion-preflight down-projection-residual-fusion-authority-rebind down-projection-residual-fusion-l2-audit down-projection-residual-fusion-integrity-check verification-stage rtl-synth-sky130 rtl-sta-sky130 ppa-power-vcd ppa-power-sky130 ppa-stage prototype-stage benchmark-stage host-rtl-persistence-batch-microbenchmark rtl-fast-loop clean

.PHONY: ace2-chat ace2-chat-prepare ace2-chat-storage-lifecycle-check ace2-chat-token-scaling-profile ace2-chat-token-scaling-profile-test rtl-chat-oracle-regression persistent-kv-multitoken-rtl-chat persistent-kv-multitoken-rtl-chat-verify persistent-kv-multitoken-full-chain-oracle persistent-conversation-rtl-diagnostic persistent-conversation-rtl-diagnostic-test prompt-suite-rtl-oracle-regression prompt-suite-rtl-oracle-regression-test prompt-recovery-coordinator-test fresh-multitoken-rtl-oracle-diagnostic fresh-multitoken-rtl-oracle-diagnostic-test lora-pilot-package-audit specification-checklist-audit specification-checklist-test specification-stage-audit specification-stage-history-audit specification-stage-s5-audit specification-stage-v3-preexecution-audit specification-stage-v3-frozen-l2-audit qwen25-lora-v4-aggregate-diagnostic-check v4-execution-package-self-test v4-execution-package-audit v4-execution-package-closure-audit v8-full-finetune-freeze-check v8-full-finetune-self-test-check v8-full-finetune-l2-check s1-transport-terminal-audit-check s1-transport-terminal-review-check s2-transport-package-audit-check s4-full-finetune-terminal-audit-check s4-full-finetune-terminal-review-check publication-integrity-ordering-regression publication-integrity-check publication-authorization-preflight final-product-publication final-product-publication-check
.PHONY: rtl-stage-contract-interface-audit rtl-stage-contract-audit rtl-stage-contract-review fused-qkv-focused fused-qkv-synth-sta fused-qkv-runtime-prefix

FUSED_QKV_ATTEMPT ?= attempt-0001

fused-qkv-focused:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_fused_qkv_benchmark_v2.py run \
		--attempt "$(FUSED_QKV_ATTEMPT)"

fused-qkv-synth-sta:
	mkdir -p build/fused-qkv-v2/sky130 evidence/verification/fused-qkv-v2/sky130
	flock -n build/fused-qkv-v2/sky130/synthesis.lock docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'source /OpenROAD-flow-scripts/env.sh && yosys -l evidence/verification/fused-qkv-v2/sky130/yosys.log -s flow/yosys/sky130_fused_qkv.ys'
	sed -E 's/\b(wire|reg|input|output) signed\b/\1/g' build/fused-qkv-v2/sky130/ace2_shell_mapped.v > build/fused-qkv-v2/sky130/ace2_shell_mapped_sta.v
	docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'set -o pipefail && source /OpenROAD-flow-scripts/env.sh && sta -exit flow/yosys/sky130_fused_qkv_sta.tcl | tee evidence/verification/fused-qkv-v2/sky130/sta.log'
	docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'set -o pipefail && source /OpenROAD-flow-scripts/env.sh && sta -exit flow/yosys/sky130_fused_qkv_paths.tcl | tee evidence/verification/fused-qkv-v2/sky130/cache_paths_sta.log'

fused-qkv-runtime-prefix: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_fused_qkv_runtime_prefix.py

model-hardware-contract-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/model_hardware_contract.py --check
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/model_hardware_contract_phase2.py --check
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/quantization_policy.py --check
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/grouped_w4a8_payload_conformance.py --check
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) -m unittest \
		verification.test_model_hardware_contract \
		verification.test_model_hardware_contract_phase2 \
		verification.test_grouped_w4a8_payload_conformance \
		verification.test_quantization_policy \
		verification.test_ace2_chat_demo.Ace2ChatDemoTest.test_pinned_base_completion_preflight_authorizes_product_candidate

grouped-w4a8-payload-conformance:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/grouped_w4a8_payload_conformance.py --check
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) -m unittest verification.test_grouped_w4a8_payload_conformance -v

file-backed-w4a8-chat-test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) -m unittest verification.test_file_backed_w4a8_chat -v

production-w4a8-batch-qualification:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) tools/qualify_file_backed_w4a8_batch.py \
		--package "$(PRODUCTION_W4A8_PACKAGE)" \
		--expected-package-sha256 "$(PRODUCTION_W4A8_PACKAGE_SHA256)" \
		--output "$(PRODUCTION_W4A8_BATCH_QUALIFICATION_OUTPUT)" \
		--max-new-tokens "$(PRODUCTION_W4A8_BATCH_MAX_NEW_TOKENS)" \
		--timeout-seconds "$(PRODUCTION_W4A8_BATCH_TIMEOUT_SECONDS)"

production-w4a8-batch-qualification-test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) -m unittest \
		verification.test_file_backed_w4a8_chat \
		verification.test_qualify_file_backed_w4a8_batch -v

production-w4a8-conversation-qualification:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) tools/qualify_file_backed_w4a8_conversation.py \
		--package "$(PRODUCTION_W4A8_PACKAGE)" \
		--expected-package-sha256 "$(PRODUCTION_W4A8_PACKAGE_SHA256)" \
		--output "$(PRODUCTION_W4A8_CONVERSATION_QUALIFICATION_OUTPUT)" \
		--max-new-tokens "$(PRODUCTION_W4A8_CONVERSATION_MAX_NEW_TOKENS)"

production-w4a8-conversation-qualification-test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) -m unittest \
		verification.test_qualify_file_backed_w4a8_conversation -v

production-w4a8-package-test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools $(PYTHON) -m unittest verification.test_export_production_w4a8_package -v

rtl-stage-contract-interface-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest verification.test_audit_rtl_stage_contract
	$(PYTHON) tools/audit_rtl_stage_contract.py

rtl-stage-contract-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_rtl_stage_audit.py \
		--iverilog "$(IVERILOG)" \
		--verilator "$(VERILATOR)" \
		$(RTL_STAGE_AUDIT_ARGS)

rtl-stage-contract-review:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_rtl_stage_audit_review.py \
		$(RTL_STAGE_REVIEW_ARGS)

ace2-chat:
	@case "$(ACE2_CHAT_MAX_NEW_TOKENS)" in \
		4|5|6|7|8) ;; \
		*) echo "ACE2_CHAT_MAX_NEW_TOKENS must be an integer from 4 through 8" >&2; exit 2 ;; \
	esac
	@if [ "$(ACE2_CHAT_MAX_NEW_TOKENS)" != "4" ]; then \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/run_fresh_multitoken_rtl_oracle_diagnostic.py \
			--attempt \
			--attempt-output "$(ACE2_CHAT_OUTPUT)" \
			--max-new-tokens "$(ACE2_CHAT_MAX_NEW_TOKENS)" \
			--timeout-seconds "$(ACE2_CHAT_TIMEOUT_SECONDS)" \
			$(ACE2_CHAT_ARGS); \
	else \
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ace2_v73_stage1_chat_attempt.py \
		--output "$(ACE2_CHAT_OUTPUT)" \
		--max-new-tokens "$(ACE2_CHAT_MAX_NEW_TOKENS)" \
		--timeout-seconds "$(ACE2_CHAT_TIMEOUT_SECONDS)" \
		$(ACE2_CHAT_ARGS); \
	fi
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_persistent_kv_multitoken_rtl_chat.py \
		--attempt "$(ACE2_CHAT_OUTPUT)" \
		--output "$(ACE2_CHAT_VERIFICATION_OUTPUT)" \
		--expected-generated-tokens "$(ACE2_CHAT_MAX_NEW_TOKENS)"
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_full_chain_independent_oracle.py \
		--attempt "$(ACE2_CHAT_OUTPUT)" \
		--output "$(ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT)" \
		--expected-generated-tokens "$(ACE2_CHAT_MAX_NEW_TOKENS)"

ace2-chat-prepare:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ace2_chat_demo.py \
		--prepare-only \
		--output "$(ACE2_CHAT_OUTPUT)" \
		--max-new-tokens "$(ACE2_CHAT_MAX_NEW_TOKENS)" \
		--timeout-cycles "$(ACE2_CHAT_TIMEOUT_CYCLES)"

ace2-chat-storage-lifecycle-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/rtl_arbitrary_text_generation_backend.py \
		--storage-lifecycle-check \
		--output "$(ACE2_CHAT_STORAGE_CHECK_OUTPUT)"

ace2-chat-token-scaling-profile:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/profile_ace2_chat_token_scaling.py \
		--case 4 "$(ACE2_CHAT_TOKEN_SCALING_4_ATTEMPT)" "$(ACE2_CHAT_TOKEN_SCALING_4_ORACLE)" \
		--case 6 "$(ACE2_CHAT_TOKEN_SCALING_6_ATTEMPT)" "$(ACE2_CHAT_TOKEN_SCALING_6_ORACLE)" \
		--case 8 "$(ACE2_CHAT_TOKEN_SCALING_8_ATTEMPT)" "$(ACE2_CHAT_TOKEN_SCALING_8_ORACLE)" \
		--output "$(ACE2_CHAT_TOKEN_SCALING_OUTPUT)"

ace2-chat-token-scaling-profile-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s tests -p 'test_ace2_chat_token_scaling_profiler.py'

rtl-chat-oracle-regression:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verification/run_stage1_chat_oracle_regression.py \
		--output "$(ACE2_CHAT_ORACLE_REGRESSION_OUTPUT)"

persistent-kv-multitoken-rtl-chat:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ace2_v73_stage1_chat_attempt.py \
		--output "$(ACE2_CHAT_MULTITOKEN_OUTPUT)" \
		--max-new-tokens "$(ACE2_CHAT_MULTITOKEN_MAX_NEW_TOKENS)" \
		--timeout-seconds "$(ACE2_CHAT_MULTITOKEN_TIMEOUT_SECONDS)" \
		$(ACE2_CHAT_MULTITOKEN_ARGS)

persistent-kv-multitoken-rtl-chat-verify:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_persistent_kv_multitoken_rtl_chat.py \
		--attempt "$(ACE2_CHAT_MULTITOKEN_OUTPUT)" \
		--output "$(ACE2_CHAT_MULTITOKEN_VERIFICATION_OUTPUT)"

persistent-kv-multitoken-full-chain-oracle:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_full_chain_independent_oracle.py \
		--attempt "$(ACE2_CHAT_MULTITOKEN_OUTPUT)" \
		--output "$(ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT)"

persistent-conversation-rtl-diagnostic:
	@case "$(PERSISTENT_CONVERSATION_RTL_MAX_NEW_TOKENS)" in \
		2|3|4|5|6|7|8) ;; \
		*) echo "PERSISTENT_CONVERSATION_RTL_MAX_NEW_TOKENS must be an integer from 2 through 8" >&2; exit 2 ;; \
	esac
	@case "$(PERSISTENT_CONVERSATION_RTL_TURNS)" in \
		2|3) ;; \
		*) echo "PERSISTENT_CONVERSATION_RTL_TURNS must be 2 or 3" >&2; exit 2 ;; \
	esac
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/run_persistent_conversation_rtl_diagnostic.py \
		--output "$(PERSISTENT_CONVERSATION_RTL_OUTPUT)" \
		--max-new-tokens "$(PERSISTENT_CONVERSATION_RTL_MAX_NEW_TOKENS)" \
		--turns "$(PERSISTENT_CONVERSATION_RTL_TURNS)"

persistent-conversation-rtl-diagnostic-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s tests -p 'test_persistent_conversation_rtl_diagnostic.py' -v

prompt-suite-rtl-oracle-regression:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/run_prompt_suite_rtl_oracle_regression.py \
		--config "$(ACE2_CHAT_PROMPT_SUITE)" \
		--output "$(ACE2_CHAT_PROMPT_SUITE_OUTPUT)" \
		--timeout-seconds "$(ACE2_CHAT_PROMPT_SUITE_TIMEOUT_SECONDS)"

prompt-suite-rtl-oracle-regression-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m py_compile \
		scripts/run_prompt_suite_rtl_oracle_regression.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s tests -p 'test_prompt_suite_rtl_oracle_regression.py'

prompt-recovery-coordinator-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m py_compile \
		scripts/run_prompt_recovery_coordinator.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s tests -p 'test_prompt_recovery_coordinator.py'

fresh-multitoken-rtl-oracle-diagnostic:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/run_fresh_multitoken_rtl_oracle_diagnostic.py \
		--output "$(ACE2_FRESH_MULTITOKEN_DIAGNOSTIC_OUTPUT)" \
		--timeout-seconds "$(ACE2_FRESH_MULTITOKEN_DIAGNOSTIC_TIMEOUT_SECONDS)" \
		$(ACE2_FRESH_MULTITOKEN_DIAGNOSTIC_ARGS)

fresh-multitoken-rtl-oracle-diagnostic-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s tests -p 'test_fresh_multitoken_rtl_oracle_diagnostic.py'
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s tests -p 'test_oracle_payload_retention.py'

lora-pilot-package-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_qwen25_05b_bf16_lora_pilot_package.py \
		--output build/qwen25_05b_bf16_lora_pilot_package_audit.json

qwen25-lora-v4-aggregate-diagnostic-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/build_qwen25_lora_v4_aggregate_failure_diagnostic.py --check

v8-full-finetune-freeze-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/freeze_qwen25_05b_bf16_full_finetune_product_v8_execution_package.py --check

v8-full-finetune-self-test-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) pilot/qwen25_05b_bf16_full_finetune_product_v8_runner/self_test.py --check

v8-full-finetune-l2-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_qwen25_05b_bf16_full_finetune_product_v8_execution_package_l2.py --check

s1-transport-terminal-audit-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_bf16_successor_s1_transport_terminal.py --check

s1-transport-terminal-review-check:
	PYTHONDONTWRITEBYTECODE=1 $(ARGUS_SKILL_PYTHON) tools/run_bf16_successor_s1_transport_terminal_review.py --check

s2-transport-package-audit-check:
	@test -n "$(ACE2_AMLT_PROJECT)" || { echo "ACE2_AMLT_PROJECT is required" >&2; exit 2; }
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) research/execution/qwen25_bf16_successor_s2_transport_20260808/audit_package.py \
		--amlt-bin "$(ACE2_AMLT_BIN)" \
		--amlt-project "$(ACE2_AMLT_PROJECT)"

s4-full-finetune-terminal-audit-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_bf16_full_finetune_successor_s4_terminal.py --check

s4-full-finetune-terminal-review-check:
	PYTHONDONTWRITEBYTECODE=1 $(ARGUS_SKILL_PYTHON) tools/run_bf16_full_finetune_successor_s4_terminal_review_followup.py --check

specification-stage-audit: specification-checklist-audit

specification-stage-history-audit: qwen25-lora-v4-aggregate-diagnostic-check s1-transport-terminal-audit-check s1-transport-terminal-review-check s4-full-finetune-terminal-audit-check s4-full-finetune-terminal-review-check
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_specification_stage.py

specification-checklist-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_productization_specification_checklist.py \
		--output research/raw/specification/specification-checklist-audit-20260812.json

specification-checklist-test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest \
		tools.test_audit_productization_specification_checklist

specification-stage-s5-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_specification_stage_s5.py \
		--output research/raw/specification/specification-stage-audit-s5-marker-free-20260808T143725Z.json

specification-stage-v3-preexecution-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_specification_stage_v3_preexecution.py \
		--output research/raw/specification/specification-stage-audit-v3-preexecution-20260807T230641Z.json

specification-stage-v3-frozen-l2-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_specification_stage_v3_frozen_l2.py \
		--output research/raw/specification/specification-stage-audit-v3-frozen-l2-20260807T233249Z.json

v4-execution-package-self-test:
	ACE2_LORA_DEVICE=cpu CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=26080704 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) pilot/qwen25_05b_bf16_lora_product_v4_runner/self_test.py

v4-execution-package-audit:
	@if [ -f build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/execution-package-l2-acceptance.json ]; then \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_qwen25_05b_bf16_lora_product_v4_execution_package_closure.py \
			--output build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/v4-execution-package-closure-audit.json; \
	else \
		ACE2_LORA_DEVICE=cpu CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=26080704 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 \
			PYTHONDONTWRITEBYTECODE=1 $(PYTHON) pilot/qwen25_05b_bf16_lora_product_v4_runner/self_test.py --check; \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_qwen25_05b_bf16_lora_product_v4_execution_package.py \
			--output build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/v4-execution-package-audit-latest.json; \
	fi

v4-execution-package-closure-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_qwen25_05b_bf16_lora_product_v4_execution_package_closure.py \
		--output build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/v4-execution-package-closure-audit.json

publication-integrity-ordering-regression:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/test_publication_integrity_ordering.py

publication-integrity-check: publication-integrity-ordering-regression
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/check_publication_integrity.py --check

publication-authorization-preflight: publication-integrity-check

final-product-publication:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/publish_final_product_certification.py --apply

final-product-publication-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/publish_final_product_certification.py --check

architecture-contract-check:
	@if [ -f evidence/shared_token_group_dynamic_scale32_v1/architecture/MANAGER_FREEZE.json ]; then \
		$(PYTHON) tools/freeze_token_group_dynamic_scale32_successor.py --check; \
	elif [ -f evidence/shared_token_group_dynamic_scale32_v1/architecture/PROPOSAL.json ]; then \
		$(PYTHON) tools/bind_token_group_dynamic_scale32_proposal.py --check; \
	elif [ -f evidence/review/architecture_stage_closing_cross_layer_quantization_error_carry_final_output_v1/decision.json ]; then \
		$(PYTHON) tools/run_cross_layer_error_carry_architecture_review.py --check; \
	else \
		$(PYTHON) tools/bind_cross_layer_error_carry_architecture.py --check; \
	fi

architecture-proposal-bind:
	$(PYTHON) tools/bind_token_group_dynamic_scale32_proposal.py

architecture-proposal-check:
	$(PYTHON) tools/bind_token_group_dynamic_scale32_proposal.py --check

architecture-proposal-review:
	$(PYTHON) tools/run_token_group_dynamic_scale32_architecture_review.py

architecture-proposal-review-check:
	$(PYTHON) tools/run_token_group_dynamic_scale32_architecture_review.py --check

architecture-successor-freeze:
	$(PYTHON) tools/freeze_token_group_dynamic_scale32_successor.py

architecture-successor-freeze-check:
	$(PYTHON) tools/freeze_token_group_dynamic_scale32_successor.py --check

baseline-harness-certification-bind:
	$(PYTHON) tools/bind_synthetic_baseline_harness_certification.py

baseline-harness-certification-check:
	$(PYTHON) tools/bind_synthetic_baseline_harness_certification.py --check

cross-layer-error-carry-architecture-review:
	$(PYTHON) tools/run_cross_layer_error_carry_architecture_review.py

cross-layer-error-carry-architecture-review-check:
	$(PYTHON) tools/run_cross_layer_error_carry_architecture_review.py --check

cross-layer-error-carry-vectors:
	$(PYTHON) tools/gen_cross_layer_error_carry_vectors.py

cross-layer-error-carry-metadata:
	$(PYTHON) tools/gen_cross_layer_error_carry_metadata.py

cross-layer-error-carry-rtl-preflight:
	$(PYTHON) tools/run_cross_layer_error_carry_preflight.py

cross-layer-error-carry-rtl-bind:
	$(PYTHON) tools/bind_cross_layer_error_carry_rtl.py

cross-layer-error-carry-rtl-review:
	$(PYTHON) tools/run_cross_layer_error_carry_rtl_review.py

cross-layer-error-carry-rtl-review-check:
	$(PYTHON) tools/run_cross_layer_error_carry_rtl_review.py --check

dynamic-scale32-rtl-preflight:
	mkdir -p build evidence/shared_token_group_dynamic_scale32_v1/rtl/latest verification/generated
	$(PYTHON) tools/gen_dynamic_scale32_vectors.py > evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/vector_generation.log 2>&1
	cat evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/vector_generation.log
	$(PYTHON) -m unittest verification.test_dynamic_scale32 -v > evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/reference_unittest.log 2>&1
	cat evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/reference_unittest.log
	$(IVERILOG) -g2012 -Wall -Iverification/generated -o build/ace2_dynamic_scale32_tb.vvp rtl/ace2_dynamic_scale32_core.sv verification/tb/ace2_dynamic_scale32_tb.sv > evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/iverilog.log 2>&1
	$(VVP) build/ace2_dynamic_scale32_tb.vvp > evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/rtl_simulation.log 2>&1
	cat evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/rtl_simulation.log
	@set -e; for top in ace2_dynamic_scale32_group_core ace2_dynamic_scale32_sidecar_builder_core ace2_dynamic_scale32_sidecar_validator_core ace2_scale32_tagged_accumulator_core; do \
		$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module $$top rtl/ace2_dynamic_scale32_core.sv; \
	done > evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/verilator_lint.log 2>&1
	cat evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/verilator_lint.log
	$(PYTHON) tools/run_dynamic_scale32_formal.py > evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/minimal_formal.log 2>&1
	cat evidence/shared_token_group_dynamic_scale32_v1/rtl/latest/minimal_formal.log

dynamic-scale32-rtl-bind: dynamic-scale32-rtl-preflight
	$(PYTHON) tools/bind_dynamic_scale32_rtl.py

dynamic-scale32-rtl-bind-check:
	$(PYTHON) tools/bind_dynamic_scale32_rtl.py --check

dynamic-scale32-rtl-review:
	$(PYTHON) tools/run_dynamic_scale32_rtl_review.py

dynamic-scale32-rtl-review-check:
	$(PYTHON) tools/run_dynamic_scale32_rtl_review.py --check

dynamic-scale32-rtl-engineer-recovery:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/recover_dynamic_scale32_rtl.py

dynamic-scale32-rtl-engineer-recovery-check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/recover_dynamic_scale32_rtl.py --check

environment-contract-check:
	$(PYTHON) tools/bind_environment_compatibility.py --check

environment-preflight:
	$(PYTHON) tools/run_environment_preflight.py --check

vectors:
	$(MAKE) -j11 vector-rmsnorm vector-residual vector-mlp-residual vector-post-attention-rmsnorm vector-projection vector-rope vector-attention-score vector-attention-compose vector-softmax vector-attention-value vector-silu-gate

vector-rmsnorm:
	$(PYTHON) tools/gen_rmsnorm_vectors.py

vector-residual:
	$(PYTHON) tools/gen_residual_vectors.py

vector-mlp-residual:
	$(PYTHON) tools/gen_mlp_residual_vectors.py

vector-post-attention-rmsnorm:
	$(PYTHON) tools/gen_post_attention_rmsnorm_vectors.py

vector-projection:
	$(PYTHON) tools/gen_projection_vectors.py

vector-rope:
	$(PYTHON) tools/gen_rope_vectors.py

vector-dynamic-rope:
	$(PYTHON) tools/gen_dynamic_rope_vectors.py

vector-attention-score:
	$(PYTHON) tools/gen_attention_score_vectors.py

vector-attention-compose:
	$(PYTHON) tools/gen_attention_compose_vectors.py

vector-softmax:
	$(PYTHON) tools/gen_softmax_vectors.py

vector-attention-value:
	$(PYTHON) tools/gen_attention_value_vectors.py

vector-silu-gate:
	$(PYTHON) tools/gen_silu_gate_vectors.py

vector-absolute-rope-online:
	$(PYTHON) tools/gen_absolute_rope_online_attention_vectors.py

vector-native-accumulator-tagged:
	$(PYTHON) tools/gen_native_accumulator_tagged_vectors.py

vector-qk-residual-cross-term:
	$(PYTHON) tools/gen_qk_residual_cross_term_vectors.py

vector-v-residual-value-correction:
	$(PYTHON) tools/gen_v_residual_value_correction_vectors.py

vector-down-projection-residual-fusion:
	$(PYTHON) tools/gen_down_projection_residual_fusion_vectors.py
	$(PYTHON) tools/gen_down_projection_residual_fusion_metadata.py

down-projection-residual-fusion-preflight:
	$(PYTHON) tools/run_down_projection_residual_fusion_preflight.py

down-projection-residual-fusion-authority-rebind:
	$(PYTHON) tools/bind_down_projection_residual_fusion_authoritative_discriminator.py

down-projection-residual-fusion-l2-audit:
	$(PYTHON) tools/run_down_projection_residual_fusion_focused_no_go_l2.py

down-projection-residual-fusion-integrity-check:
	$(PYTHON) tools/check_down_projection_residual_fusion_integrity_disposition.py

qk-residual-cross-term-preflight:
	$(PYTHON) tools/run_qk_residual_cross_term_preflight.py

v-residual-value-correction-preflight:
	$(PYTHON) tools/run_v_residual_value_correction_preflight.py

rtl-lint:
	mkdir -p evidence/frontier/latest
	@{ $(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal $(RTL_INCLUDE_FLAGS) --top-module ace2_shell $(RTL_SOURCES); status=$$?; if [ $$status -eq 0 ]; then echo ACE2_RTL_LINT_PASS; else echo "ACE2_RTL_LINT_FAIL status=$$status"; fi; exit $$status; } > evidence/frontier/latest/rtl_lint.log 2>&1
	cat evidence/frontier/latest/rtl_lint.log

rtl-sim: vectors
	mkdir -p build evidence/frontier/latest
	flock build/rtl-sim.lock $(MAKE) -j8 rtl-sim-rmsnorm rtl-sim-projection rtl-sim-rope rtl-sim-attention-score rtl-sim-softmax rtl-sim-attention-compose rtl-sim-silu-gate rtl-sim-shell

rtl-sim-rmsnorm:
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_rmsnorm_tb.vvp $(RTL_SOURCES) verification/tb/ace2_rmsnorm_tb.sv
	$(VVP) build/ace2_rmsnorm_tb.vvp > evidence/frontier/latest/rtl_core_sim.log
	cat evidence/frontier/latest/rtl_core_sim.log

rtl-sim-projection:
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_w4a8_proj_tb.vvp $(RTL_SOURCES) verification/tb/ace2_w4a8_proj_tb.sv
	$(VVP) build/ace2_w4a8_proj_tb.vvp > evidence/frontier/latest/rtl_proj_sim.log
	cat evidence/frontier/latest/rtl_proj_sim.log

rtl-sim-rope:
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_rope_tb.vvp $(RTL_SOURCES) verification/tb/ace2_rope_tb.sv
	$(VVP) build/ace2_rope_tb.vvp > evidence/frontier/latest/rtl_rope_sim.log
	cat evidence/frontier/latest/rtl_rope_sim.log

rtl-sim-dynamic-rope: vector-dynamic-rope
	mkdir -p build evidence/dynamic_rope_head_scale_v1/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -Iverification/generated -o build/ace2_dynamic_rope_head_tb.vvp $(RTL_SOURCES) verification/tb/ace2_dynamic_rope_head_tb.sv
	$(VVP) build/ace2_dynamic_rope_head_tb.vvp > evidence/dynamic_rope_head_scale_v1/latest/rtl_dynamic_rope_head.log
	cat evidence/dynamic_rope_head_scale_v1/latest/rtl_dynamic_rope_head.log

dynamic-rope-focused:
	mkdir -p evidence/dynamic_rope_head_scale_v1/latest
	$(PYTHON) -m unittest verification.test_dynamic_rope_head_scale > evidence/dynamic_rope_head_scale_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/dynamic_rope_head_scale_v1/latest/software_reference_unittest.log
	$(MAKE) rtl-sim-dynamic-rope

fixed-q7-focused:
	mkdir -p build evidence/layer0_fixed_q7_rope_score_v1/latest
	$(PYTHON) -m unittest verification.test_layer0_fixed_q7 > evidence/layer0_fixed_q7_rope_score_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_fixed_q7_rope_score_v1/latest/software_reference_unittest.log
	$(IVERILOG) -g2012 -Irtl -o build/ace2_fixed_q7_rope_score_tb.vvp rtl/ace2_fixed_q7_rope_score_core.sv verification/tb/ace2_fixed_q7_rope_score_tb.sv
	$(VVP) build/ace2_fixed_q7_rope_score_tb.vvp > evidence/layer0_fixed_q7_rope_score_v1/latest/rtl_fixed_q7_rope_score.log
	cat evidence/layer0_fixed_q7_rope_score_v1/latest/rtl_fixed_q7_rope_score.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module ace2_fixed_q7_rope_score_core rtl/ace2_fixed_q7_rope_score_core.sv > evidence/layer0_fixed_q7_rope_score_v1/latest/rtl_lint.log 2>&1
	cat evidence/layer0_fixed_q7_rope_score_v1/latest/rtl_lint.log

fixed-q7-fidelity-focused:
	mkdir -p build evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest
	$(PYTHON) -m unittest verification.test_layer0_fixed_q7 > evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest/software_reference_unittest.log
	$(IVERILOG) -g2012 -Irtl -o build/ace2_fixed_q7_rope_score_fidelity_tb.vvp rtl/ace2_fixed_q7_rope_score_core.sv verification/tb/ace2_fixed_q7_rope_score_tb.sv
	$(VVP) build/ace2_fixed_q7_rope_score_fidelity_tb.vvp > evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest/rtl_fixed_q7_rope_score.log
	cat evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest/rtl_fixed_q7_rope_score.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module ace2_fixed_q7_rope_score_core rtl/ace2_fixed_q7_rope_score_core.sv > evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest/rtl_lint.log 2>&1
	cat evidence/layer0_fixed_q7_rope_score_v1_fidelity_repair/latest/rtl_lint.log

relative-rope-score-focused:
	mkdir -p build evidence/layer0_relative_rope_score_fusion_v1/latest
	$(PYTHON) tools/check_global_valid_key_max_contract.py > evidence/layer0_relative_rope_score_fusion_v1/latest/global_valid_key_max_contract.json
	$(PYTHON) -m unittest verification.test_layer0_relative_rope_score_fusion > evidence/layer0_relative_rope_score_fusion_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_relative_rope_score_fusion_v1/latest/software_reference_unittest.log
	$(IVERILOG) -g2012 -Irtl -o build/ace2_relative_rope_score_fusion_tb.vvp rtl/ace2_relative_rope_score_fusion_core.sv verification/tb/ace2_relative_rope_score_fusion_tb.sv
	$(VVP) build/ace2_relative_rope_score_fusion_tb.vvp > evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_relative_rope_score_fusion.log
	cat evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_relative_rope_score_fusion.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module ace2_relative_rope_score_core rtl/ace2_relative_rope_score_fusion_core.sv > evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_arithmetic_lint.log 2>&1
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module ace2_global_score_center_core rtl/ace2_relative_rope_score_fusion_core.sv > evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_center_lint.log 2>&1
	cat evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_arithmetic_lint.log
	cat evidence/layer0_relative_rope_score_fusion_v1/latest/rtl_center_lint.log

absolute-rope-online-focused: vector-absolute-rope-online
	mkdir -p build evidence/layer0_absolute_rope_online_attention_v1/latest
	$(PYTHON) -m unittest verification.test_absolute_rope_online_attention -v > evidence/layer0_absolute_rope_online_attention_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_absolute_rope_online_attention_v1/latest/software_reference_unittest.log
	$(IVERILOG) -g2012 -Wall -Irtl/generated -Iverification/generated -o build/ace2_absolute_rope_score_tb.vvp rtl/ace2_absolute_rope_score_core.sv verification/tb/ace2_absolute_rope_score_tb.sv
	$(VVP) build/ace2_absolute_rope_score_tb.vvp > evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_absolute_rope_score.log
	cat evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_absolute_rope_score.log
	$(IVERILOG) -g2012 -Wall -Irtl/generated -o build/ace2_absolute_rope_online_attention_tb.vvp rtl/ace2_absolute_rope_online_attention_core.sv verification/tb/ace2_absolute_rope_online_attention_tb.sv
	$(VVP) build/ace2_absolute_rope_online_attention_tb.vvp > evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_online_attention.log
	cat evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_online_attention.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Irtl/generated --top-module ace2_absolute_rope_score_core rtl/ace2_absolute_rope_score_core.sv > evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_score_lint.log 2>&1
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Irtl/generated --top-module ace2_absolute_rope_online_attention_core rtl/ace2_absolute_rope_online_attention_core.sv > evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_online_lint.log 2>&1
	cat evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_score_lint.log
	cat evidence/layer0_absolute_rope_online_attention_v1/latest/rtl_online_lint.log

projection-shadow-focused:
	mkdir -p build evidence/layer0_projection_shadow_staged_attention_v1/latest verification/generated
	$(PYTHON) tools/gen_projection_shadow_vectors.py
	$(PYTHON) -m unittest verification.test_projection_shadow_staged_attention -v > evidence/layer0_projection_shadow_staged_attention_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_projection_shadow_staged_attention_v1/latest/software_reference_unittest.log
	$(IVERILOG) -g2012 -Wall -Iverification/generated -o build/ace2_projection_shadow_staged_attention_tb.vvp rtl/ace2_projection_shadow_staged_attention_core.sv verification/tb/ace2_projection_shadow_staged_attention_tb.sv
	$(VVP) build/ace2_projection_shadow_staged_attention_tb.vvp > evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_projection_shadow.log
	cat evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_projection_shadow.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall --top-module ace2_projection_shadow_core rtl/ace2_projection_shadow_staged_attention_core.sv > evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_projection_lint.log 2>&1
	$(VERILATOR) --lint-only --language 1800-2017 -Wall --top-module ace2_projection_shadow_score_core rtl/ace2_projection_shadow_staged_attention_core.sv > evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_score_lint.log 2>&1
	cat evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_projection_lint.log
	cat evidence/layer0_projection_shadow_staged_attention_v1/latest/rtl_score_lint.log

tile-max-delta-focused:
	mkdir -p build evidence/layer0_tile_max_delta_attention_v1/latest verification/generated
	$(PYTHON) tools/gen_tile_max_delta_attention_vectors.py
	$(PYTHON) -m unittest verification.test_tile_max_delta_attention -v > evidence/layer0_tile_max_delta_attention_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_tile_max_delta_attention_v1/latest/software_reference_unittest.log
	$(PYTHON) tools/ace2_full_model_fixed_point.py --mode self-test > evidence/layer0_tile_max_delta_attention_v1/latest/full_model_self_test.log 2>&1
	cat evidence/layer0_tile_max_delta_attention_v1/latest/full_model_self_test.log
	$(IVERILOG) -g2012 -Wall -Irtl -Iverification/generated -o build/ace2_tile_max_delta_attention_tb.vvp rtl/ace2_tile_max_delta_attention_core.sv verification/tb/ace2_tile_max_delta_attention_tb.sv
	$(VVP) build/ace2_tile_max_delta_attention_tb.vvp > evidence/layer0_tile_max_delta_attention_v1/latest/rtl_tile_max_delta.log
	cat evidence/layer0_tile_max_delta_attention_v1/latest/rtl_tile_max_delta.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal -Irtl --top-module ace2_tile_max_delta_score_core rtl/ace2_tile_max_delta_attention_core.sv > evidence/layer0_tile_max_delta_attention_v1/latest/rtl_score_lint.log 2>&1
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal -Irtl --top-module ace2_hierarchical_softmax_core rtl/ace2_tile_max_delta_attention_core.sv > evidence/layer0_tile_max_delta_attention_v1/latest/rtl_softmax_lint.log 2>&1
	cat evidence/layer0_tile_max_delta_attention_v1/latest/rtl_score_lint.log
	cat evidence/layer0_tile_max_delta_attention_v1/latest/rtl_softmax_lint.log

tile-bfp-focused:
	mkdir -p build evidence/layer0_tile_bfp_score_attention_v1/latest verification/generated
	$(PYTHON) tools/gen_tile_bfp_attention_vectors.py
	$(PYTHON) -m unittest verification.test_tile_bfp_attention -v > evidence/layer0_tile_bfp_score_attention_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/layer0_tile_bfp_score_attention_v1/latest/software_reference_unittest.log
	$(PYTHON) tools/ace2_full_model_fixed_point.py --mode self-test > evidence/layer0_tile_bfp_score_attention_v1/latest/full_model_self_test.log 2>&1
	cat evidence/layer0_tile_bfp_score_attention_v1/latest/full_model_self_test.log
	$(IVERILOG) -g2012 -Wall -Irtl -Iverification/generated -o build/ace2_tile_bfp_attention_tb.vvp rtl/ace2_tile_bfp_score_attention_core.sv verification/tb/ace2_tile_bfp_attention_tb.sv
	$(VVP) build/ace2_tile_bfp_attention_tb.vvp > evidence/layer0_tile_bfp_score_attention_v1/latest/rtl_tile_bfp.log
	cat evidence/layer0_tile_bfp_score_attention_v1/latest/rtl_tile_bfp.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal -Irtl --top-module ace2_tile_bfp_score_core rtl/ace2_tile_bfp_score_attention_core.sv > evidence/layer0_tile_bfp_score_attention_v1/latest/rtl_score_lint.log 2>&1
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal -Irtl --top-module ace2_bfp_hierarchical_softmax_core rtl/ace2_tile_bfp_score_attention_core.sv > evidence/layer0_tile_bfp_score_attention_v1/latest/rtl_softmax_lint.log 2>&1
	cat evidence/layer0_tile_bfp_score_attention_v1/latest/rtl_score_lint.log
	cat evidence/layer0_tile_bfp_score_attention_v1/latest/rtl_softmax_lint.log

native-accumulator-tagged-focused: vector-native-accumulator-tagged
	mkdir -p build evidence/shared_native_accumulator_tagged_attention_v1/latest
	$(PYTHON) -m unittest verification.test_native_accumulator_tagged_attention -v > evidence/shared_native_accumulator_tagged_attention_v1/latest/software_reference_unittest.log 2>&1
	cat evidence/shared_native_accumulator_tagged_attention_v1/latest/software_reference_unittest.log
	$(IVERILOG) -g2012 -Wall -Iverification/generated -o build/ace2_native_accumulator_tagged_attention_tb.vvp rtl/ace2_native_accumulator_tagged_attention_core.sv verification/tb/ace2_native_accumulator_tagged_attention_tb.sv
	$(VVP) build/ace2_native_accumulator_tagged_attention_tb.vvp > evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_simulation.log
	cat evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_simulation.log
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module ace2_native_accumulator_rope_core rtl/ace2_native_accumulator_tagged_attention_core.sv > evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_rope_lint.log 2>&1
	$(VERILATOR) --lint-only --language 1800-2017 -Wall -Wno-fatal --top-module ace2_tagged_attention_score_core rtl/ace2_native_accumulator_tagged_attention_core.sv > evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_score_lint.log 2>&1
	cat evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_rope_lint.log
	cat evidence/shared_native_accumulator_tagged_attention_v1/latest/rtl_score_lint.log

rtl-sim-rope-shell:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_rope_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_rope_tb.vvp +ROPE_ONLY > evidence/frontier/latest/rtl_rope_shell_sim.log
	cat evidence/frontier/latest/rtl_rope_shell_sim.log

rtl-sim-attention-score:
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_attention_score_tb.vvp $(RTL_SOURCES) verification/tb/ace2_attention_score_tb.sv
	$(VVP) build/ace2_attention_score_tb.vvp > evidence/frontier/latest/rtl_attention_score_sim.log
	cat evidence/frontier/latest/rtl_attention_score_sim.log

rtl-sim-softmax:
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_softmax_tb.vvp $(RTL_SOURCES) verification/tb/ace2_softmax_tb.sv
	$(VVP) build/ace2_softmax_tb.vvp > evidence/frontier/latest/rtl_softmax_sim.log
	cat evidence/frontier/latest/rtl_softmax_sim.log

rtl-sim-attention-compose:
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_attention_compose_tb.vvp $(RTL_SOURCES) verification/tb/ace2_attention_compose_tb.sv
	$(VVP) build/ace2_attention_compose_tb.vvp > evidence/frontier/latest/rtl_attention_compose_sim.log
	cat evidence/frontier/latest/rtl_attention_compose_sim.log

rtl-sim-silu-gate:
	$(IVERILOG) -g2012 -Irtl -Iverification/tb -o build/ace2_silu_gate_tb.vvp $(RTL_SOURCES) verification/tb/ace2_silu_gate_tb.sv
	$(VVP) build/ace2_silu_gate_tb.vvp > evidence/frontier/latest/rtl_silu_gate_core_sim.log
	cat evidence/frontier/latest/rtl_silu_gate_core_sim.log

rtl-sim-shell:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_tb.vvp > evidence/frontier/latest/rtl_shell_sim.log
	cat evidence/frontier/latest/rtl_shell_sim.log

rtl-sim-attention-score-shell:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_attn_score_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_attn_score_tb.vvp +ATTN_SCORE_ONLY > evidence/frontier/latest/rtl_attention_score_shell_sim.log
	cat evidence/frontier/latest/rtl_attention_score_shell_sim.log

rtl-sim-attn-max-predicate-focused:
	mkdir -p build $(ATTN_RETIME_EVIDENCE_DIR)
	$(IVERILOG) -g2012 -Wall -DACE2_ATTN_MAX_RETIME_POSTCHANGE $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_attn_max_predicate_tb.vvp $(RTL_SOURCES) verification/tb/ace2_attn_max_predicate_tb.sv > $(ATTN_RETIME_EVIDENCE_DIR)/focused_compile.log 2>&1
	$(VVP) build/ace2_attn_max_predicate_tb.vvp > $(ATTN_RETIME_EVIDENCE_DIR)/focused_simulation.log 2>&1
	cat $(ATTN_RETIME_EVIDENCE_DIR)/focused_compile.log
	cat $(ATTN_RETIME_EVIDENCE_DIR)/focused_simulation.log

rtl-sim-attn-max-predicate-shell-trace:
	mkdir -p build $(ATTN_RETIME_EVIDENCE_DIR)
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_attn_retime_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv > $(ATTN_RETIME_EVIDENCE_DIR)/$(ATTN_RETIME_PHASE)_shell_compile.log 2>&1
	$(VVP) build/ace2_shell_attn_retime_tb.vvp +ATTN_SCORE_ONLY +ATTN_RETIME_TRACE > $(ATTN_RETIME_EVIDENCE_DIR)/$(ATTN_RETIME_PHASE)_attention_score_shell.log 2>&1
	cat $(ATTN_RETIME_EVIDENCE_DIR)/$(ATTN_RETIME_PHASE)_shell_compile.log
	tail -n 12 $(ATTN_RETIME_EVIDENCE_DIR)/$(ATTN_RETIME_PHASE)_attention_score_shell.log

rtl-sim-qproj-stride:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_qproj_stride_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_qproj_stride_tb.vvp +QPROJ_STRIDE_ONLY > evidence/frontier/latest/rtl_qproj_stride_sim.log
	cat evidence/frontier/latest/rtl_qproj_stride_sim.log

rtl-sim-mlp-up:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_mlp_up_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_mlp_up_tb.vvp +MLP_UP_ONLY > evidence/frontier/latest/rtl_mlp_up_sim.log
	cat evidence/frontier/latest/rtl_mlp_up_sim.log

rtl-sim-mlp-down:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_mlp_down_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_mlp_down_tb.vvp +MLP_DOWN_ONLY > evidence/frontier/latest/rtl_mlp_down_sim.log
	cat evidence/frontier/latest/rtl_mlp_down_sim.log

rtl-sim-mlp-residual: vector-mlp-residual
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_mlp_residual_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_mlp_residual_tb.vvp +MLP_RESIDUAL_ONLY > evidence/frontier/latest/rtl_mlp_residual_sim.log
	cat evidence/frontier/latest/rtl_mlp_residual_sim.log

rtl-sim-silu-shell:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 -Irtl -Iverification/tb -o build/ace2_shell_silu_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_silu_tb.vvp +SILU_ONLY > evidence/frontier/latest/rtl_silu_gate_sim.log
	cat evidence/frontier/latest/rtl_silu_gate_sim.log

rtl-sim-layer-sweep:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_layer_sweep_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_layer_sweep_tb.vvp +LAYER_SWEEP_ONLY > evidence/frontier/latest/rtl_layer_sweep_sim.log
	cat evidence/frontier/latest/rtl_layer_sweep_sim.log

rtl-sim-final-rmsnorm:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_final_rmsnorm_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_final_rmsnorm_tb.vvp +FINAL_RMSNORM_ONLY > evidence/frontier/latest/rtl_final_rmsnorm_sim.log
	cat evidence/frontier/latest/rtl_final_rmsnorm_sim.log

rtl-sim-lm-head:
	mkdir -p build evidence/frontier/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_lm_head_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_lm_head_tb.vvp +LM_HEAD_ONLY > evidence/frontier/latest/rtl_lm_head_sim.log
	cat evidence/frontier/latest/rtl_lm_head_sim.log

rtl-sim-shell-smoke:
	@test -n "$(SHELL_SMOKE_OPCODE_HEX)" || { echo "unsupported SHELL_SMOKE_OPCODE=$(SHELL_SMOKE_OPCODE)" >&2; exit 2; }
	mkdir -p build evidence/verification/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_smoke_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv
	$(VVP) build/ace2_shell_smoke_tb.vvp +SMOKE_OPCODE=$(SHELL_SMOKE_OPCODE_HEX) > evidence/verification/latest/rtl_shell_smoke_$(SHELL_SMOKE_OPCODE).log
	cat evidence/verification/latest/rtl_shell_smoke_$(SHELL_SMOKE_OPCODE).log

$(VERILATOR_SHELL_OPROJ_BIN): $(RTL_SOURCES) $(RTL_GENERATED_HEADERS) \
		verification/verilator/ace2_shell_oproj_harness.sv \
		verification/verilator/ace2_shell_oproj_main.cpp
	mkdir -p build/verilator_shell_oproj evidence/verification/latest
	$(VERILATOR) --cc --exe --build -j 8 --output-split 10000 \
		--output-split-cfuncs 100 --language 1800-2017 -Wno-fatal \
		--top-module ace2_shell_oproj_harness --Mdir build/verilator_shell_oproj \
		-CFLAGS "-O1" $(RTL_INCLUDE_FLAGS) -Iverification/tb $(RTL_SOURCES) \
		verification/verilator/ace2_shell_oproj_harness.sv \
		$(CURDIR)/verification/verilator/ace2_shell_oproj_main.cpp

rtl-sim-shell-verilator: $(VERILATOR_SHELL_OPROJ_BIN)
	mkdir -p evidence/verification/latest
	$(VERILATOR_SHELL_OPROJ_BIN) > evidence/verification/latest/rtl_shell_verilator_oproj.log
	cat evidence/verification/latest/rtl_shell_verilator_oproj.log

rtl-sim-shell-agreement:
	$(PYTHON) tools/check_shell_throughput_agreement.py

$(VERILATOR_FULL_QWEN_RUNTIME_BIN): $(RTL_SOURCES) $(RTL_GENERATED_HEADERS) \
		verification/verilator/ace2_shell_runtime_harness.sv \
		verification/verilator/ace2_runtime_identity_profile.h \
		verification/verilator/ace2_shell_runtime_main.cpp
	mkdir -p build/verilator_full_qwen_runtime
	$(VERILATOR) --cc --exe --build -j 8 --output-split 10000 \
		--output-split-cfuncs 100 --language 1800-2017 -Wno-fatal \
		--top-module ace2_shell_runtime_harness \
		--Mdir build/verilator_full_qwen_runtime \
		-CFLAGS "-O3 -std=c++17" -LDFLAGS "-lcrypto" \
		$(RTL_INCLUDE_FLAGS) $(RTL_SOURCES) \
		verification/verilator/ace2_shell_runtime_harness.sv \
		$(CURDIR)/verification/verilator/ace2_shell_runtime_main.cpp

full-qwen-runtime-build: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)

host-rtl-persistence-batch-microbenchmark: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_host_rtl_persistence_batch_microbenchmark.py

rtl-runtime-package-test: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest verification.test_ace2_runtime_package -v

rtl-sim-runtime-rmsnorm-min-latency: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest verification.test_verilated_rmsnorm_runtime -v

rtl-sim-runtime-first-kv-reference: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest verification.test_verilated_first_kv_runtime -v

rtl-sim-runtime-dynamic-scale32-first-boundary: $(VERILATOR_FULL_QWEN_RUNTIME_BIN)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest verification.test_verilated_dynamic_scale32_boundary -v

chat-first-residual-rtl-boundary:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/run_chat_residual_boundary_rtl.py

dynamic-scale32-first-boundary-synth-sta:
	mkdir -p build/sky130_dynamic_scale32_first_boundary $(DYNAMIC_SCALE32_BOUNDARY_EVIDENCE_DIR)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/capture_chat_sky130_provenance.py before --evidence-dir $(DYNAMIC_SCALE32_BOUNDARY_EVIDENCE_DIR)
	flock -n build/sky130_dynamic_scale32_first_boundary/synthesis.lock docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'source /OpenROAD-flow-scripts/env.sh && yosys -l $(DYNAMIC_SCALE32_BOUNDARY_EVIDENCE_DIR)/sky130_yosys.log -s flow/yosys/sky130_dynamic_scale32_first_boundary.ys'
	sed -E 's/\b(wire|reg|input|output) signed\b/\1/g' build/sky130_dynamic_scale32_first_boundary/ace2_shell_mapped.v > build/sky130_dynamic_scale32_first_boundary/ace2_shell_mapped_sta.v
	docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'set -o pipefail && source /OpenROAD-flow-scripts/env.sh && sta -exit flow/yosys/sky130_dynamic_scale32_first_boundary_sta.tcl | tee $(DYNAMIC_SCALE32_BOUNDARY_EVIDENCE_DIR)/sky130_sta.log'
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/capture_chat_sky130_provenance.py after --evidence-dir $(DYNAMIC_SCALE32_BOUNDARY_EVIDENCE_DIR)

verification-stage:
	$(PYTHON) tools/run_down_projection_residual_fusion_verification.py

rtl-synth-sky130: publication-integrity-check
	mkdir -p build/sky130_rmsnorm evidence/frontier/latest
	flock -n build/sky130_rmsnorm/synthesis.lock docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'source /OpenROAD-flow-scripts/env.sh && yosys -l evidence/frontier/latest/sky130_yosys.log -s flow/yosys/sky130_rmsnorm.ys'

rtl-sta-sky130: rtl-synth-sky130
	sed -E 's/\b(wire|reg|input|output) signed\b/\1/g' build/sky130_rmsnorm/ace2_shell_mapped.v > build/sky130_rmsnorm/ace2_shell_mapped_sta.v
	docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'set -o pipefail && source /OpenROAD-flow-scripts/env.sh && sta -exit flow/yosys/sky130_sta.tcl | tee evidence/frontier/latest/sky130_sta.log'

ppa-power-vcd:
	mkdir -p build ppa/raw/latest
	$(IVERILOG) -g2012 $(RTL_INCLUDE_FLAGS) -Iverification/tb -o build/ace2_shell_power_tb.vvp $(RTL_SOURCES) verification/tb/ace2_shell_tb.sv ppa/harness/ace2_shell_power_dump_tb.sv -s ace2_shell_power_dump_tb
	$(VVP) build/ace2_shell_power_tb.vvp +LM_HEAD_ONLY > ppa/raw/latest/ace2_shell_lm_head_power_vcd.log
	cat ppa/raw/latest/ace2_shell_lm_head_power_vcd.log

ppa-power-sky130: rtl-sta-sky130 ppa-power-vcd
	docker run --rm -u $$(id -u):$$(id -g) -v "$$(pwd)":/work -w /work $(ORFS_IMAGE) bash -lc 'set -o pipefail && source /OpenROAD-flow-scripts/env.sh && sta -exit flow/yosys/sky130_power.tcl | tee ppa/raw/latest/sky130_power.log'

ppa-stage: ppa-power-sky130
	$(PYTHON) tools/run_ppa_stage.py

prototype-stage:
	$(PYTHON) tools/run_prototype_stage.py

benchmark-stage:
	$(PYTHON) tools/run_benchmark_stage.py

rtl-fast-loop: rtl-lint rtl-sim rtl-sim-mlp-up rtl-sim-mlp-down rtl-sim-mlp-residual rtl-sim-silu-shell rtl-sim-layer-sweep rtl-sim-final-rmsnorm rtl-sim-lm-head rtl-sta-sky130

clean:
	rm -rf build/ace2_rmsnorm_tb.vvp build/ace2_w4a8_proj_tb.vvp build/ace2_rope_tb.vvp build/ace2_attention_score_tb.vvp build/ace2_softmax_tb.vvp build/ace2_silu_gate_tb.vvp build/ace2_shell_tb.vvp build/ace2_shell_qproj_stride_tb.vvp build/ace2_shell_mlp_up_tb.vvp build/ace2_shell_mlp_down_tb.vvp build/ace2_shell_mlp_residual_tb.vvp build/ace2_shell_silu_tb.vvp build/ace2_shell_layer_sweep_tb.vvp build/ace2_shell_final_rmsnorm_tb.vvp build/ace2_shell_lm_head_tb.vvp build/ace2_shell_smoke_tb.vvp build/ace2_shell_power_tb.vvp build/verilator_shell_oproj build/sky130_rmsnorm

.PHONY: chat-all-residual-reference qwen-instruct-local-audit

chat-all-residual-reference:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/ace2_chat_all_residual_reference.py --output build/ace2_chat_diagnostics/all-residual-scale32-two-prompts-20260805.json --max-new-tokens 8 --top-k 10

qwen-instruct-local-audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_qwen_instruct_local_artifact.py --revision 7ae557604adf67be50417f59c2c2f167def9a775 --output build/ace2_chat_diagnostics/qwen2.5-0.5b-instruct-local-audit-20260805.json
