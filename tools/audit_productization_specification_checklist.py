#!/usr/bin/env python3
"""Audit only the live ACE-2 productization specification-stage checklist.

This verifier intentionally does not inspect, recreate, or depend on consumed
attempt state.  It binds the normative specification, public RTL declaration,
non-benchmark closure, Manager-owned pipeline stage, and read-only U280 host
inventory needed by the current specification gate. It also binds the current
resolved bounded-diagnosis gate without creating downstream authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_CHECKSUM = ROOT / "research/PIPELINE_STATE.sha256"
SHELL = ROOT / "rtl/ace2_shell.sv"
SHELL_AUDIT = ROOT / "tools/audit_rtl_stage_contract.py"

REQUIRED_ACCEPTANCE_CLASSES = (
    "normal",
    "boundary",
    "illegal",
    "reset",
    "stall",
    "recovery",
)

BEHAVIOR_FRAGMENTS = (
    "Command acceptance",
    "Prefill step `i`",
    "First decode decision",
    "Decode feedback step `j`",
    "KV visibility",
    "Termination",
    "Completion",
)

FUSED_QKV_SPEC_FRAGMENTS = (
    "default product schedule uses one fused opcode `0x0b` QKV command",
    "`src0=0x0000001000000700`",
    "`src1=0x0000000100000000 + L*0x000000000071c000`",
    "`scale=0x0000000200000000 + L*0x0000000000031800`",
    "`dst=0x0000001000000a80`",
    "Alignment and in-image range are necessary but not sufficient",
    "rejected before command execution",
)

PROTOCOL_FRAGMENTS = (
    "one clock domain, `clk_i`",
    "No internal CDC is permitted",
    "`rst_ni` is active low",
    "Assertion may occur asynchronously",
    "Deassertion at the accelerator boundary must be synchronous to `clk_i`",
    "Every ready/valid channel transfers only on `valid && ready`",
    "holds valid and its complete payload stable while ready is low",
    "same-cycle combinational response is unsupported",
    "Arbitrary finite backpressure is legal",
    "there is no finite latency guarantee",
    "at most one direct command at a time",
    "no fixed cycles/token or tokens/s value is claimed",
)

AUTHORITY_FRAGMENTS = (
    "`NONOFFICIAL_RTL_REFERENCE_INTEGRATION_ACCEPTED_OFFICIAL_CHAT_DENIED`",
    "previous status `BLOCKED_EXTERNAL_ATTESTATION`",
    "`current_stage=stage1_software_diagnosis`",
    "nonofficial Stage 1 diagnosis, candidate, focused RTL/reference, generalization, and descriptor-bound sidecar-integration evidence",
    "`bounded_diagnosis_authority_consumed=true`",
    "`fresh_l2_review_authorized_for_bounded_diagnosis=false`",
    "`official_rtl_backed_arbitrary_text_chat_authorized=false`",
    "does not authorize official preflight/run, `attempt-0003`, retry/replay/resume",
    "`rtl_authorized=false`, `stage_2_authorized=false`, `xrt_authorized=false`, and `u280_authorized=false`",
    "`diagnosis/stage1layer23qkvfamilies01/result.json`",
    "`evidence/diagnostics/stage1-layer23-v-residual-structure-v1/nonofficial-diagnosis-0001/result.json`",
    "`271efb2d9974`",
    "`DENY_NEXT_OFFICIAL_RTL_BACKED_ARBITRARY_TEXT_CHAT`",
    "`sealed_evidence_replayed=false`",
    "`stage1vprecision01`",
    "`stage1lmheadequiv01`",
    "No runtime namespace, payload open, evaluator call, target start",
    "Stage advancement remains a Manager action",
    "Manager-authorship provenance unresolved",
)

CURRENT_AUTHORIZATION = {
    "bounded_diagnosis_authority_consumed": True,
    "bounded_nonofficial_software_diagnosis_authorized": False,
    "checkpoint_176_official_execution_authorized": False,
    "fresh_l2_review_authorized_for_bounded_diagnosis": False,
    "official_rtl_backed_arbitrary_text_chat_authorized": False,
    "rtl_authorized": False,
    "stage_2_authorized": False,
    "u280_authorized": False,
    "xrt_authorized": False,
}

EXPECTED_STAGE1_EVIDENCE = {
    "bounded_diagnosis": {
        "classification": "bounded_nonofficial_software_only_layer23_qkv_diagnosis",
        "fresh_l2_review_request": {
            "path": "diagnosis/stage1layer23qkvfamilies01/fresh-l2-review-request.json",
            "sha256": "e1d691762bafd3d7a82789fa0d9dea548f0b5294a6924eadf02e6afc8f5a461e",
            "status": "PENDING_INDEPENDENT_REVIEW",
        },
        "result": {
            "bytes": 30392,
            "path": "diagnosis/stage1layer23qkvfamilies01/result.json",
            "sha256": "4e6fa9c101f2d2a4c09a80fb1883665043a5d8126ebaa7d4fa862bf602aa4856",
            "status": "COMPONENT_AND_QUANTIZER_FAMILY_CAUSALLY_LOCALIZED",
        },
        "runner": {
            "bytes": 34656,
            "path": "tools/diagnose_stage1_layer23_qkv_families.py",
            "sha256": "5b41665c5018183ab88ab3825e5d31c3301f97d7baa49327b0244efe08796402",
        },
    },
    "manager_operator_answer": {
        "continuation_id": "9e235ac035b2",
        "decision": "CLEAR_STALE_ATTESTATION_WAIT_AND_AUTHORIZE_ONE_BOUNDED_STAGE1_DIAGNOSIS",
        "plan_id": "plan-083dd935a5dc",
        "plan_version": 1,
    },
    "sealed_attempt_0002": {
        "manifest_sha256": "0259039f54ecc224a785ada1d4567b34158746fc06b3d3f3e316f5d277ddb470",
        "run_summary_sha256": "c496921d5b93f4840ec67c38c1c4d0afab034a8798a9a760c36a639ee80d4159",
        "status": "SEALED_RTL_W4A8_INTEGER_PARITY_READABLE_OUTPUT_QUALITY_FAIL",
    },
}

EXPECTED_NEXT_ACTION = {
    "action": "Do not start official RTL-backed arbitrary-text chat; remain at the bounded Stage 1 control-plane state unless separate fresh authority is granted",
    "authority_decision": "DENY_NEXT_OFFICIAL_RTL_BACKED_ARBITRARY_TEXT_CHAT",
    "candidate_promotion_authorized": False,
    "official_execution_authorized": False,
    "rtl_backed_arbitrary_text_chat_authorized": False,
    "software_candidate_execution_authorized": False,
    "stage": "stage1_software_diagnosis",
}

EXPECTED_REVIEWED_BOUNDED_EVIDENCE = {
    "stage1layer23qkvfamilies01": {
        "handoff": "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/9e235ac035b2/round-0002.json",
        "handoff_sha256": "70a428c83f8747f00ae1af08418d14c7d40a339612e740bb79ab105d5819eebd",
        "status": "done",
        "result_sha256": "4e6fa9c101f2d2a4c09a80fb1883665043a5d8126ebaa7d4fa862bf602aa4856",
        "product_claim": False,
    },
    "stage1vprecision01": {
        "handoff": "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/stage1vprecision01/round-0002.json",
        "handoff_sha256": "01490089e8ebebd19cbe0e15618244cef74429ac9680878ad96464a959349a37",
        "status": "done",
        "matrix_result_sha256": "ef6b1a37c9ea8df211f93358e05f1d3468e791084f7f8bb31c9c6c8fa9de38c9",
        "requirement_sha256": "924025be1faaadaf0319aee5dc690571921a0195683f1df87d419ecb9f9333d0",
        "selected_format": None,
        "negative_result": True,
        "current_rtl_support": False,
        "product_claim": False,
    },
    "stage1lmheadequiv01": {
        "handoff": "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/stage1lmheadequiv01/round-0001.json",
        "handoff_sha256": "075c14e87971230fb8a4fcc9bfb8d0391fa11fdf9130fa80459bce2e052e7096",
        "status": "done",
        "reviewer_acceptance_sha256": "b10e69c90bffe512fb54153c93a51db0d19fb3f7c4d74f12da96c770f09835e5",
        "engineer_result_sha256": "7a9009a51084871836be78278f96ce9b16bf1d88c1407996b00e65b25ddd2408",
        "successor_activation_reads_per_case": 1792,
        "successor_packed_w4_reads_per_case": 896,
        "metadata_reads_per_case": 32,
        "write_beats_per_case": 2,
        "weight_span_bytes_per_case": 14336,
        "weight_high_water_offset": 14320,
        "checked_logits": 64,
        "full_vocabulary_runtime_reexecuted": False,
        "synthesis_or_ppa_executed": False,
        "product_claim": False,
    },
    "stage1vjointscale01": {
        "handoff": "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/stage1vjointscale01/round-0001.json",
        "handoff_sha256": "e7688c3960f9663e4c1bba7521d4fd945a87eb57d7a865bb2499bb48d7cec9b4",
        "status": "done",
        "result_sha256": "1c07c32cf687fb382580fd6558a1bd0f83170fba331896374f77d46c3f95a4f4",
        "sha256s_sha256": "0469d5a0d2e4e22d67ed0818ad8c0e0a23d21e59389c558a92d6fb4cffcf7b6b",
        "classification": "bounded_nonofficial_software_only_joint_scale_search",
        "candidate_count": 5,
        "selected_candidate_id": None,
        "promotion_eligible": False,
        "negative_result": True,
        "baseline_target_token_rank": 15,
        "candidate_target_token_ranks": [14, 14, 16, 15, 20],
        "baseline_top8_overlap_count": 0,
        "candidate_top8_overlap_counts": [2, 2, 2, 2, 1],
        "baseline_jsd_nats": 0.6701229933515445,
        "candidate_jsd_nats": [
            0.6678304612691217,
            0.6670142652978546,
            0.6685694180913542,
            0.6668547670659924,
            0.6702937097823827,
        ],
        "baseline_held_out_v_relative_l2": 1.01390282266012,
        "best_candidate_held_out_v_relative_l2": 1.0155168651981596,
        "official_execution_authorized": False,
        "focused_or_full_rtl_executed": False,
        "synthesis_or_ppa_executed": False,
        "stage_2_executed": False,
        "product_claim": False,
    },
    "blocked_item_3d3e14c90193_closure": {
        "mission_id": "1421b599ab84",
        "handoff": "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/1421b599ab84/round-0001.json",
        "handoff_sha256": "d5ce2feb92c771dcef701244fa3f399697f166f7f66dd423515b1ac52d01bcf6",
        "status": "done",
        "operator_advanced_stage": False,
        "rtl_backed_arbitrary_text_chat_authorized": False,
        "official_preflight_or_run_authorized": False,
        "attempt_0003_authorized": False,
        "attestation_task_authorized": False,
        "stage_2_authorized": False,
        "repository_destructive_action": False,
        "product_claim": False,
    },
}

EXPECTED_FORBIDDEN_EFFECTS = [
    "official preflight/run",
    "attempt-0003",
    "retry/replay/resume",
    "checkpoint-176 official execution",
    "W4/RTL product promotion",
    "full Icarus generation as product evidence",
    "synthesis/PPA",
    "Stage 2",
    "XRT/U280/FPGA work",
    "product-completion claim",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def markdown_section(text: str, heading: str) -> str:
    match = re.search(rf"(?m)^{re.escape(heading)}\s*$", text)
    if match is None:
        return ""
    level = len(heading) - len(heading.lstrip("#"))
    following = text[match.end() :]
    next_heading = re.search(rf"(?m)^#{{1,{level}}}\s+", following)
    if next_heading is None:
        return following
    return following[: next_heading.start()]


def checksum_companion_matches(path: Path, companion: Path) -> bool:
    if not path.is_file() or not companion.is_file():
        return False
    fields = companion.read_text(encoding="ascii").strip().split()
    return bool(fields and fields[0] == sha256(path))


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def normalize_markdown_text(text: str) -> str:
    """Collapse Markdown line wrapping without changing visible tokens."""

    return re.sub(r"\s+", " ", text).strip()


def evaluate(
    *,
    spec_text: str,
    benchmark: dict[str, Any],
    pipeline: dict[str, Any],
    shell_report: dict[str, Any],
    pipeline_checksum_matches: bool,
    inventory_exists: bool,
    inventory_sha256: str | None,
) -> dict[str, Any]:
    cycle_section = markdown_section(
        spec_text, "### Stage 1 cycle-level prefill, KV, and decode behavior"
    )
    protocol_section = markdown_section(
        spec_text, "### Clock, reset, flow-control, latency, and throughput rules"
    )
    public_section = markdown_section(
        spec_text, "### Public `ace2_shell` parameter and port contract"
    )
    acceptance_section = markdown_section(spec_text, "### Stage 1 acceptance matrix")
    command_section = markdown_section(
        spec_text, "### Direct-command control and legal-value rules"
    )
    authority_section = markdown_section(
        spec_text, "### Stage 1 current authority and advancement gate"
    )
    benchmark_section = markdown_section(spec_text, "### Benchmark-interface closure")
    normalized_cycle = normalize_markdown_text(cycle_section)
    normalized_protocol = normalize_markdown_text(protocol_section)
    normalized_public = normalize_markdown_text(public_section)
    normalized_command = normalize_markdown_text(command_section)
    normalized_fused_spec = f"{normalized_cycle} {normalized_command}"
    normalized_authority = normalize_markdown_text(authority_section)
    normalized_benchmark = normalize_markdown_text(benchmark_section)

    active = benchmark.get("active_productization_contract", {})
    public = active.get("public_module", {}) if isinstance(active, dict) else {}
    authority_gate = (
        active.get("production_acceptance_authority_gate", {})
        if isinstance(active, dict)
        else {}
    )
    stage_1 = active.get("stage_1", {}) if isinstance(active, dict) else {}
    stage_2 = active.get("stage_2", {}) if isinstance(active, dict) else {}
    pipeline_blocker = pipeline.get("stage1_blocker", {})
    canonical_blocker = authority_gate.get("canonical_stage1_blocker", {})
    fused_qkv = benchmark.get("fused_qkv_contract", {})
    fused_command = fused_qkv.get("command", {})
    fused_descriptor = fused_command.get("legal_descriptor", {})
    fused_layout = fused_command.get("memory_layout", {})
    fused_schedule = fused_qkv.get("default_product_schedule", {})
    fused_illegal = next(
        (
            case
            for case in fused_qkv.get("acceptance_matrix", [])
            if case.get("id") == "fused_qkv_illegal_descriptor"
        ),
        {},
    )
    shell_checks = shell_report.get("checks", {})

    behavior_missing = [
        fragment for fragment in BEHAVIOR_FRAGMENTS if fragment not in normalized_cycle
    ]
    public_fragments = (
        "Parameters are signed SystemVerilog `integer` elaboration constants",
        "supports exactly the tuple below",
        "Every public port below is declared as an unsigned SystemVerilog scalar or packed bit vector",
        "there are no `signed` public port declarations",
        "| Parameter | Default | Legal value | Contract |",
        "| Port | Direction | Width (unsigned packed bits) | Contract |",
    )
    public_missing = [
        fragment for fragment in public_fragments if fragment not in normalized_public
    ]
    fused_spec_missing = [
        fragment
        for fragment in FUSED_QKV_SPEC_FRAGMENTS
        if fragment not in normalized_fused_spec
    ]
    fused_contract_ok = bool(
        fused_qkv.get("module") == "ace2_shell"
        and fused_qkv.get("source") == "rtl/ace2_shell.sv"
        and fused_command.get("opcode") == 11
        and fused_command.get("symbol") == "ACE2_OPCODE_FUSED_QKV"
        and fused_descriptor
        == {
            "dst_addr": 68719479424,
            "flags": 0,
            "k": 896,
            "layer_id": [0, 23],
            "m": 1,
            "n": 896,
            "scale_addr_formula": (
                "0x0000000200000000 + layer_id * 0x0000000000031800"
            ),
            "src0_addr": 68719478528,
            "src1_addr_formula": (
                "0x0000000100000000 + layer_id * 0x000000000071c000"
            ),
            "scratch_addr": 0,
        }
        and fused_layout.get("activation_bytes") == 896
        and fused_layout.get("activation_read_beats") == 56
        and fused_layout.get("weight_layer0_base") == 4294967296
        and fused_layout.get("weight_layer_stride") == 7454720
        and fused_layout.get("weight_span_bytes") == 516096
        and fused_layout.get("weight_byte_offsets_q_k_v") == [0, 401408, 458752]
        and fused_layout.get("metadata_layer0_base") == 8589934592
        and fused_layout.get("metadata_layer_stride") == 202752
        and fused_layout.get("metadata_span_bytes") == 18432
        and fused_layout.get("metadata_byte_offsets_q_k_v") == [0, 14336, 16384]
        and fused_layout.get("dst_span_bytes") == 1152
        and fused_layout.get("dst_byte_offsets_q_k_v") == [0, 896, 1024]
        and fused_schedule.get("mode") == "fused"
        and "Dynamic Scale32" in fused_schedule.get(
            "prompt_layer0_dynamic_scale32_exception", ""
        )
        and fused_illegal.get("required") is True
        and "noncanonical" in fused_illegal.get("acceptance", "")
        and "before_command_execution" in fused_illegal.get("acceptance", "")
    )
    interface_ok = bool(
        shell_report.get("status") == "PASS"
        and shell_checks.get("parameter_count_source") == 14
        and shell_checks.get("parameter_count_spec") == 14
        and shell_checks.get("port_count_source") == 64
        and shell_checks.get("port_count_spec") == 64
        and shell_checks.get("parameter_mismatches") == []
        and shell_checks.get("port_mismatches") == []
        and public.get("name") == "ace2_shell"
        and public.get("source") == "rtl/ace2_shell.sv"
        and public.get("parameter_count") == 14
        and public.get("port_count") == 64
        and public.get("signed_public_port_count") == 0
        and public.get("interface_repair_permitted_in_specification") is False
        and stage_1.get("status") == "specified_not_implemented"
        and stage_1.get("backend_required")
        == "rtl_backed_verilator_or_later_hw_emu_or_hw"
        and stage_1.get("software_only_fallback_qualifies") is False
        and stage_1.get("arbitrary_utf8_input_required") is True
        and stage_1.get("multi_turn_input_required") is True
        and stage_1.get("minimum_visible_nontermination_tokens") == 2
        and stage_1.get("legal_max_new_tokens_range") == [1, 256]
        and not behavior_missing
        and not public_missing
        and not fused_spec_missing
        and fused_contract_ok
    )

    protocol_missing = [
        fragment for fragment in PROTOCOL_FRAGMENTS if fragment not in normalized_protocol
    ]
    protocol_ok = bool(protocol_section and not protocol_missing)

    observed_classes = sorted(
        {
            match.group(1).strip().lower()
            for match in re.finditer(r"(?m)^\|\s*([^|]+?)\s*\|", acceptance_section)
            if match.group(1).strip().lower() not in {"class", "---"}
        }
    )
    missing_classes = [
        name for name in REQUIRED_ACCEPTANCE_CLASSES if name not in observed_classes
    ]
    acceptance_ok = bool(acceptance_section and not missing_classes)

    authority_missing = [
        fragment
        for fragment in AUTHORITY_FRAGMENTS
        if fragment not in normalized_authority
    ]
    stage_1_authority_gate_ok = bool(
        authority_section
        and not authority_missing
        and authority_gate.get("status")
        == "NONOFFICIAL_RTL_REFERENCE_INTEGRATION_ACCEPTED_OFFICIAL_CHAT_DENIED"
        and authority_gate.get("previous_status") == "BLOCKED_EXTERNAL_ATTESTATION"
        and authority_gate.get("source")
        == "research/PIPELINE_STATE.json#stage1_blocker"
        and authority_gate.get("authorized_scope") == pipeline_blocker.get("claim_boundary")
        and authority_gate.get("authorization") == CURRENT_AUTHORIZATION
        and authority_gate.get("forbidden_effects") == EXPECTED_FORBIDDEN_EFFECTS
        and canonical_blocker == pipeline_blocker
        and stage_1.get("advancement_blocked_by")
        == "production_acceptance_authority_gate_manager_authorship_and_official_chat_denial"
        and pipeline_blocker.get("status")
        == "NONOFFICIAL_RTL_REFERENCE_INTEGRATION_ACCEPTED_OFFICIAL_CHAT_DENIED"
        and pipeline_blocker.get("previous_status") == "BLOCKED_EXTERNAL_ATTESTATION"
        and pipeline_blocker.get("record_kind") == "ace2_stage1_blocker"
        and pipeline_blocker.get("resolved_at_utc") == "2026-08-20T13:08:29Z"
        and pipeline_blocker.get("authorization") == CURRENT_AUTHORIZATION
        and pipeline_blocker.get("evidence", {})
        .get("v_residual_structure_diagnosis", {})
        .get("result", {})
        .get("sha256")
        == "11dbd7dda4715827e3a16e4e530bdb6870fa51370fff86084a05c6d07ea3bc2d"
        and pipeline_blocker.get("evidence", {})
        .get("v_residual_structure_diagnosis", {})
        .get("authorization_consumption", {})
        .get("sha256")
        == "87bf833dfcda18d514c366d622a50be834a61422bab7cdf0dab096b3e55f96d9"
        and pipeline_blocker.get("evidence", {})
        .get("prior_bounded_qkv_diagnosis", {})
        .get("result", {})
        .get("sha256")
        == "4e6fa9c101f2d2a4c09a80fb1883665043a5d8126ebaa7d4fa862bf602aa4856"
        and pipeline_blocker.get("next_action", {}).get("stage")
        == "stage1_software_diagnosis"
        and pipeline_blocker.get("next_action", {}).get("authority_decision")
        == "DENY_NEXT_OFFICIAL_RTL_BACKED_ARBITRARY_TEXT_CHAT"
        and pipeline_blocker.get("next_action", {}).get("official_execution_authorized")
        is False
        and pipeline_blocker.get("next_action", {}).get(
            "rtl_backed_arbitrary_text_chat_authorized"
        )
        is False
        and pipeline_blocker.get("next_action", {}).get(
            "software_candidate_execution_authorized"
        )
        is False
        and pipeline_blocker.get("predecessor_and_terminal_evidence_preserved")
        is True
        and pipeline_blocker.get("evidence", {})
        .get("manager_reconciliation", {})
        .get("authority_decision")
        == "DENY_NEXT_OFFICIAL_RTL_BACKED_ARBITRARY_TEXT_CHAT"
        and pipeline_blocker.get("evidence", {})
        .get("manager_reconciliation", {})
        .get("sealed_evidence_replayed")
        is False
        and authority_gate.get("reviewed_bounded_evidence", {})
        .get("stage1_layer23_v_rank1_sidecar_integration", {})
        .get("handoff_sha256")
        == "52b06dcbbd21e00131582bc2deea7c35bae64da53fb9dbdaf979961c508f1fa9"
        and authority_gate.get("reviewed_bounded_evidence", {})
        .get("stage1_layer23_v_rank1_sidecar_integration", {})
        .get("result_sha256")
        == "36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1"
        and authority_gate.get("reviewed_bounded_evidence", {})
        .get("stage1_layer23_v_rank1_sidecar_integration", {})
        .get("product_claim")
        is False
        and authority_gate.get("manager_checksum_reconciliation", {}).get("required")
        is False
        and authority_gate.get("manager_checksum_reconciliation", {}).get(
            "performed_by_planner"
        )
        is False
        and authority_gate.get("manager_checksum_reconciliation", {}).get(
            "manager_authorship_proven"
        )
        is False
        and pipeline.get("stages", {}).get("specification", {}).get("status")
        == "done_advanced_to_bounded_stage1_software_diagnosis"
        and pipeline.get("stages", {})
        .get("stage1_software_diagnosis", {})
        .get("status")
        == "nonofficial_rtl_reference_integration_accepted_official_chat_denied"
    )

    fixed_external = active.get("fixed_external_benchmark", {})
    fixed_external_null = bool(
        isinstance(fixed_external, dict)
        and fixed_external.get("applies") is False
        and all(
            fixed_external.get(key) is None
            for key in ("prompt", "evaluator", "official_inputs", "tool_versions", "score_policy")
        )
    )
    non_benchmark_statement = benchmark.get("non_benchmark_statement", "")
    local = benchmark.get("local_contract", {})
    benchmark_ok = bool(
        benchmark.get("stage") == "specification"
        and benchmark.get("applies") is False
        and benchmark.get("external_contract") is None
        and fixed_external_null
        and isinstance(non_benchmark_statement, str)
        and "No external accelerator benchmark" in non_benchmark_statement
        and "hidden harness" in non_benchmark_statement
        and "hidden golden output" in non_benchmark_statement
        and local.get("top_module") == "ace2_shell"
        and local.get("entrypoint") == "python3 tools/ace2_chat_demo.py"
        and local.get("output_path") == "caller-selected directory passed through --output"
        and local.get("public_interface_source")
        == "design/SPEC.md#public-ace2_shell-parameter-and-port-contract"
        and "No external accelerator benchmark" in normalized_benchmark
        and "No fixed benchmark attempt is started or consumed" in normalized_benchmark
    )

    pipeline_stage_ok = bool(
        pipeline.get("current_stage") in {"specification", "stage1_software_diagnosis"}
        and benchmark.get("stage") == "specification"
    )
    pipeline_integrity_ok = bool(
        pipeline_stage_ok and pipeline_checksum_matches
    )

    u280 = benchmark.get("u280_environment_gate", {})
    u280_hold_ok = bool(
        inventory_exists
        and inventory_sha256 is not None
        and inventory_sha256 == u280.get("latest_inventory_sha256")
        and u280.get("toolchain_download_authorized") is False
        and u280.get("vivado_present_in_observed_host_inventory") is False
        and u280.get("vitis_present_in_observed_host_inventory") is False
        and u280.get("vpp_present_in_observed_host_inventory") is False
        and u280.get("xrt_utilities_present_in_observed_host_inventory") is False
        and u280.get("hardware_emulation_endpoint") is None
        and u280.get("local_u280_access_evidence_present") is False
        and u280.get("u280_hardware_conclusion") is None
        and stage_2.get("status") == "blocked_pending_stage_1_and_external_prerequisites"
        and benchmark.get("stage_order", {}).get("stage_2_u280") == "pending_after_stage_1"
    )

    checklist = {
        "spec.behavior-interface": {
            "status": _status(interface_ok),
            "shell_contract_audit": shell_report.get("status"),
            "parameter_count": shell_checks.get("parameter_count_source"),
            "port_count": shell_checks.get("port_count_source"),
            "signed_public_port_count": public.get("signed_public_port_count"),
            "missing_behavior_fragments": behavior_missing,
            "missing_public_contract_fragments": public_missing,
            "missing_fused_qkv_spec_fragments": fused_spec_missing,
            "fused_qkv_contract": _status(fused_contract_ok),
        },
        "spec.clock-reset-protocol": {
            "status": _status(protocol_ok),
            "clock_domains": 1 if protocol_ok else None,
            "missing_protocol_fragments": protocol_missing,
        },
        "spec.acceptance-matrix": {
            "status": _status(acceptance_ok),
            "required_classes": list(REQUIRED_ACCEPTANCE_CLASSES),
            "observed_classes": observed_classes,
            "missing_classes": missing_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": _status(benchmark_ok),
            "external_benchmark_applies": benchmark.get("applies"),
            "external_contract": benchmark.get("external_contract"),
            "local_top_module": local.get("top_module"),
            "local_entrypoint": local.get("entrypoint"),
            "local_output_path": local.get("output_path"),
        },
    }
    checklist_complete = all(item["status"] == "PASS" for item in checklist.values())
    # The checksum companion is Manager-owned control-plane integrity evidence,
    # not one of the four specification checklist criteria.  Report drift
    # prominently and deny stage-advancement eligibility, but do not turn four
    # passing specification criteria into a false specification failure.
    audit_complete = bool(
        checklist_complete
        and pipeline_stage_ok
        and stage_1_authority_gate_ok
        and u280_hold_ok
    )

    return {
        "schema_version": 1,
        "kind": "ace2_productization_specification_checklist_audit",
        "status": "PASS_SPECIFICATION_CHECKLIST_COMPLETE"
        if audit_complete
        else "FAIL_SPECIFICATION_CHECKLIST",
        "checklist": checklist,
        "supporting_constraints": {
            "pipeline_state_integrity": {
                "status": _status(pipeline_integrity_ok),
                "current_stage": pipeline.get("current_stage"),
                "stage_matches_specification": pipeline_stage_ok,
                "checksum_companion_matches": pipeline_checksum_matches,
                "manager_reconciliation_required": not pipeline_integrity_ok,
            },
            "u280_prerequisite_hold": {
                "status": _status(u280_hold_ok),
                "inventory": u280.get("latest_inventory"),
                "inventory_sha256": inventory_sha256,
                "toolchain_download_authorized": u280.get(
                    "toolchain_download_authorized"
                ),
                "hardware_emulation_endpoint": u280.get(
                    "hardware_emulation_endpoint"
                ),
                "local_u280_access_evidence_present": u280.get(
                    "local_u280_access_evidence_present"
                ),
                "u280_hardware_conclusion": u280.get(
                    "u280_hardware_conclusion"
                ),
            },
            "stage_1_current_authority_gate": {
                "status": _status(stage_1_authority_gate_ok),
                "blocker_status": pipeline_blocker.get("status"),
                "previous_status": pipeline_blocker.get("previous_status"),
                "missing_spec_fragments": authority_missing,
                "evidence": pipeline_blocker.get("evidence"),
                "authorization": authority_gate.get("authorization"),
                "forbidden_effects": authority_gate.get("forbidden_effects"),
                "reviewed_bounded_evidence": authority_gate.get(
                    "reviewed_bounded_evidence"
                ),
                "downstream_authority_available": False,
            },
        },
        "conclusions": {
            "specification_checklist_complete": checklist_complete,
            "current_stage": pipeline.get("current_stage"),
            "stage_1_status": stage_1.get("status"),
            "stage_2_status": stage_2.get("status"),
            "stage_1_authority_blocked": bool(stage_1_authority_gate_ok),
            "stage_1_authority_blocker": pipeline_blocker.get("status"),
            "manager_pipeline_reconciliation_required": not pipeline_integrity_ok,
            "stage_advancement_eligible": bool(
                audit_complete
                and pipeline_integrity_ok
                and not stage_1_authority_gate_ok
            ),
            "manager_authority_proven": False,
            "product_completion_eligible": False,
            "rtl_correctness_conclusion": None,
            "u280_deployment_conclusion": None,
            "downstream_stage_authority_created": False,
        },
    }


def audit() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "tools"))
    from audit_rtl_stage_contract import audit as audit_shell_contract

    spec_text = SPEC.read_text(encoding="utf-8")
    benchmark = load_json(BENCHMARK)
    pipeline = load_json(PIPELINE)
    shell_report = audit_shell_contract()
    inventory_rel = benchmark.get("u280_environment_gate", {}).get("latest_inventory")
    inventory = ROOT / inventory_rel if isinstance(inventory_rel, str) else None
    inventory_exists = bool(inventory is not None and inventory.is_file())
    inventory_digest = sha256(inventory) if inventory_exists and inventory is not None else None

    result = evaluate(
        spec_text=spec_text,
        benchmark=benchmark,
        pipeline=pipeline,
        shell_report=shell_report,
        pipeline_checksum_matches=checksum_companion_matches(
            PIPELINE, PIPELINE_CHECKSUM
        ),
        inventory_exists=inventory_exists,
        inventory_sha256=inventory_digest,
    )
    input_paths = [
        SELF,
        SPEC,
        BENCHMARK,
        PIPELINE,
        PIPELINE_CHECKSUM,
        SHELL,
        SHELL_AUDIT,
    ]
    if inventory_exists and inventory is not None:
        input_paths.append(inventory)
    result["input_bindings"] = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths
    }
    result["reproduction_command"] = (
        "PYTHONDONTWRITEBYTECODE=1 python3 "
        "tools/audit_productization_specification_checklist.py"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{sha256(output)}  {output.name}\n", encoding="ascii"
        )
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
