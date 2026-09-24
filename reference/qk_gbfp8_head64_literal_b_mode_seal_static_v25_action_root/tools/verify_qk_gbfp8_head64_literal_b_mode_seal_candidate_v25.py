#!/usr/bin/env python3
"""Literal-``-B`` and immutable-mode verifier for additive static-only V25."""

from __future__ import annotations

import sys

EARLY_DONT_WRITE_BYTECODE = sys.dont_write_bytecode is True
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from literal_b_launch_guard_v25 import prove_effective_suppression, validate_current_process


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build/v25-literal-b-mode-seal-static-0002"
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_PACKAGE.json"
REGRESSION_REPORT = ACTION_ROOT / "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json"
B0_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_B0_SIDECAR_SCHEMA.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_FRESH_L2_ACCEPTANCE_SCHEMA.json"
B0_VALID_FIXTURE = ACTION_ROOT / "fixtures/B0_VALID_SYNTHETIC_FIXTURE.json"
B0_MUTATIONS = ACTION_ROOT / "fixtures/B0_SYNTHETIC_NEGATIVE_MUTATIONS.json"
NORMAL_IMPORT_FIXTURE = ACTION_ROOT / "fixtures/V23_STYLE_NORMAL_IMPORT_INVENTORY_MUTATION.json"
LAUNCH_CATALOG = ACTION_ROOT / "fixtures/LITERAL_B_LAUNCH_NEGATIVE_CASES.json"
B0_VALIDATOR = ACTION_ROOT / "tools/validate_qk_gbfp8_head64_source_oracle_b0_v23.py"
POST_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_post_acceptance_v25.py"
LAUNCH_MATRIX = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_literal_b_launch_matrix_v25.py"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
MISSION = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/v25literalbmodeseal/mission.json")
V24_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_bytecode_free_verifier_static_v24_action_root"
V24_PACKAGE = V24_ROOT / "QK_GBFP8_HEAD64_BYTECODE_FREE_VERIFIER_STATIC_V24_PACKAGE.json"
V24_CANDIDATE_REPORT = PROJECT_ROOT / "build/v24-bytecode-free-verifier-static-0001/QK_GBFP8_HEAD64_BYTECODE_FREE_VERIFIER_STATIC_V24_CANDIDATE_REPORT.json"
V24_NEGATIVE_REPORT = PROJECT_ROOT / "build/v24-bytecode-free-verifier-static-0001/QK_GBFP8_HEAD64_BYTECODE_FREE_VERIFIER_STATIC_V24_NEGATIVE_FIXTURE_REPORT.json"
V24_REVIEW_HANDOFF = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/v24bytecodefreeverifier/round-0001.json")
V24_CHECKPOINT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/v24bytecodefreeverifier/CHECKPOINT.md")
V23_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_action_root"
V23_ACCEPTANCE = V23_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V23_PYC = V23_ROOT / "tools/__pycache__/verify_qk_gbfp8_head64_source_oracle_verifier_hardening_candidate_v23.cpython-313.pyc"
V22_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_source_oracle_attribution_static_v22_action_root"
V22_PACKAGE = V22_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_ATTRIBUTION_STATIC_V22_PACKAGE.json"
V22_REPORT = V22_ROOT / "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json"
V22_CANDIDATE_REPORT = PROJECT_ROOT / "verification/QK_GBFP8_HEAD64_SOURCE_ORACLE_ATTRIBUTION_STATIC_V22_CANDIDATE_REPORT.json"
V22_CANDIDATE_VERIFIER = V22_ROOT / "tools/verify_qk_gbfp8_head64_source_oracle_attribution_candidate_v22.py"
DIAGNOSIS = PROJECT_ROOT / "diagnosis/QK_GBFP8_HEAD64_V21_BASE_FAILURE_DIAGNOSIS_AND_V22_STATIC_SUCCESSOR.json"
V21_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root"
V21_RUNTIME = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba"
V21_RESULT = V21_RUNTIME / "primary/result/base/result.json"
OFFICIAL_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
OFFICIAL_RESULT_SCHEMA = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
BINDING_TABLE = V21_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
ENTRY_LAUNCH_PROOF = validate_current_process(Path(__file__), EARLY_DONT_WRITE_BYTECODE) if __name__ == "__main__" else None

EXPECTED_ACTION_ID = "ace2:qk-gbfp8-base-v25:literal-b-mode-seal:12b1f604:additive-0002"
EXPECTED_V24_ACTION_ID = "ace2:qk-gbfp8-base-v24:bytecode-free-verifier:ca93ecc0:additive-0001"
EXPECTED_V23_ACTION_ID = "ace2:qk-gbfp8-base-v23:source-oracle-verifier-hardening:ec58e011:additive-0001"
EXPECTED_V22_ACTION_ID = "ace2:qk-gbfp8-base-v22:source-oracle-attribution:4901c835:additive-0001"
EXPECTED_V21_ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
EXPECTED_CLAIM = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
EXPECTED_FROZEN = {
    "benchmark_package_sha256": "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832",
    "binding_table_sha256": "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655",
    "candidate_definitions": [
        {"bytes_per_head": 80, "exponent_bytes_per_head": 16, "group_count": 8, "group_size": 8, "label": "G8", "mantissa_bytes_per_head": 64},
        {"bytes_per_head": 96, "exponent_bytes_per_head": 32, "group_count": 16, "group_size": 4, "label": "G4", "mantissa_bytes_per_head": 64},
        {"bytes_per_head": 128, "exponent_bytes_per_head": 64, "group_count": 32, "group_size": 2, "label": "G2", "mantissa_bytes_per_head": 64},
        {"bytes_per_head": 192, "exponent_bytes_per_head": 128, "group_count": 64, "group_size": 1, "label": "G1", "mantissa_bytes_per_head": 64},
    ],
    "candidate_order": ["G8", "G4", "G2", "G1"],
    "diagnosis_file_sha256": "650f46edd9fba234c76948d679af76384e9d55a00369af252ced01f23cc57c2c",
    "failure_if_none_pass": "HARD_THRESHOLD_FAILED",
    "hard_gates": {
        "cross_lane_record_count_maximum": 0,
        "invalid_or_non_finite_value_count_maximum": 0,
        "normalization_rejection_count_maximum": 0,
        "positive_centered_realized_score_count_maximum": 0,
        "rank_margin_violation_count_maximum": 0,
        "saturation_event_count_maximum": 0,
        "top_key_matching_fraction_minimum": {"denominator": 1, "numerator": 1},
        "top_key_mismatch_count_maximum": 0,
        "unique_oracle_positive_margin_preserved_fraction_minimum": {"denominator": 1, "numerator": 1},
    },
    "input_token_ids_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
    "lane_metadata_byte_count": 20057,
    "lane_metadata_sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
    "model_identity_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
    "numerical_tensor_identities": [
        {"dtype": "torch.bfloat16", "sha256": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a", "shape": [1, 2, 41, 64], "tensor_name": "bf16.k_rope"},
        {"dtype": "torch.bfloat16", "sha256": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429", "shape": [1, 14, 41, 64], "tensor_name": "bf16.q_rope"},
        {"dtype": "torch.bfloat16", "sha256": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765", "shape": [1, 14, 41, 41], "tensor_name": "bf16.qk_scaled_scores"},
    ],
    "official_evaluator_sha256": "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    "official_result_schema_sha256": "07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720",
    "require_every_hard_gate": True,
    "score_error_affects_selection": False,
    "sealed_set_id": "w4a8-c02-attention-substage-trace-v2",
    "selected_tensor_names": ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"],
    "selection_policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
    "tensor_bundle_byte_count": 1305797,
    "tensor_bundle_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
    "tensor_record_count": 25,
    "v21_result_byte_count": 8767,
    "v21_result_file_sha256": "4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f",
}
EXPECTED_ACTION_IDENTITY = {
    "action_id": EXPECTED_ACTION_ID,
    "additive_successor": True,
    "predecessor_action_id": EXPECTED_V24_ACTION_ID,
    "predecessor_failed_immutable": True,
    "retry_replay_resume_repair_reinterpret_permitted": False,
    "runtime_namespace": None,
}
EXPECTED_CLAIM_BOUNDARY = {
    "authority_or_credential_materialized": False,
    "base_selection_permitted": False,
    "claim": EXPECTED_CLAIM,
    "evaluator_invocations": 0,
    "execution_authorized": False,
    "official_payload_open_count": 0,
    "official_target_process_starts": 0,
    "owner_or_ledger_materialized": False,
    "result_or_terminal_materialized": False,
    "rtl_or_hardware_activity": False,
    "runtime_namespace_materialized": False,
    "v21_or_predecessor_mutated": False,
    "v23_resealed_reinterpreted_or_mutated": False,
    "v24_accepted_resealed_reinterpreted_or_mutated": False,
    "v24_executed": False,
    "v24_preserved_byte_identical": True,
    "v25_executed": False,
}
EXPECTED_B0_IDENTITY = {
    "base_selection_permitted": False,
    "control_id": "B0_SOURCE_QK_IDENTITY",
    "future_separate_authority_required": True,
    "materialized_by_this_package": False,
    "nonselecting": True,
    "published_data_boundary": "SYNTHETIC_FIXTURES_ONLY_NO_OFFICIAL_TENSOR_VALUES",
    "quantization_completion": False,
    "residual_control_labels": ["B0", "G1", "G2", "G4", "G8"],
    "row_class_schema": ["ALL_CAUSAL_ROWS", "SINGLETON_ROWS", "ORACLE_TIED_TOP_ROWS", "ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS"],
    "schema_file_sha256": "3beb5cd5ac35c66bec7ebeac8f4bd4f7a510736f1efd619f24eba82b81b1826b",
    "schema_path": "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_B0_SIDECAR_SCHEMA.json",
    "software_identity_path": True,
    "validator_file_sha256": "2a0786cc2ae9a9a0bb70ab5329bee61bc0d846d1a6559e3a1ee48cb69bbe1a16",
    "validator_path": "tools/validate_qk_gbfp8_head64_source_oracle_b0_v23.py",
}
EXPECTED_ARITHMETIC = {
    "carried_forward_byte_identical": True,
    "official_evaluator_invocation_count": 0,
    "official_payload_open_count": 0,
    "official_target_process_starts": 0,
    "report_file_sha256": "baaba1e6a305deb5467ca016b74bd953673d9664d035098a1c6284d8762795f1",
    "report_path": "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json",
    "report_sha256": "8a7714d964de79d8de689db9687014b5d9bd1154d7b77dbc4a0384d91613a412",
    "source_v22_report_file_sha256": "baaba1e6a305deb5467ca016b74bd953673d9664d035098a1c6284d8762795f1",
    "source_v22_tool_sha256": "92eebe6ee52d471ef96fcfadfe986905e8497b83b2c2f3bd9a2c46f6868e7e7e",
}
EXPECTED_FRESH_L2 = {
    "acceptance_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
    "acceptance_schema_file_sha256": "2146882e8d2a50e69ba94e4f6c84825fbb8dce5fccd8d29215a69b01926949c3",
    "acceptance_schema_path": "reference/QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "decision_requested": "ACCEPT_STATIC_PACKAGE",
    "engineer_self_review_can_close": False,
    "launch_matrix_independent_reproduction_required": True,
    "required_role": "Fresh-L2",
    "static_acceptance_grants_execution_authority": False,
    "status": "PENDING_INDEPENDENT_REVIEW",
}
EXPECTED_MISSION = {
    "path": str(MISSION),
    "sha256": "2bba04e321f51d1eafc735c739687368065f28048aefe9f10a941bd357a125fb",
    "size": 3747,
}
EXPECTED_TOOLCHAIN = {
    "interpreter_path": "/home/argustest/miniconda3/bin/python3.13",
    "interpreter_sha256": "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad",
    "interpreter_version": "3.13.5",
    "jsonschema_version": "4.23.0",
}
EXPECTED_PREDECESSOR = {
    "v22_action_root": {"entry_count": 11, "file_count": 7, "root": str(V22_ROOT), "tree_sha256": "7bb7ffebf39c0a5fda8d85da24e05100e8cc6674d2f72a06be1321dd5b6f4dcb"},
    "v22_candidate_report": {"path": str(V22_CANDIDATE_REPORT), "sha256": "f5d2df7a134baaf4d01c8ab176cb12724e9c3eff1b9ede4f704977e123dc3583", "size": 1697},
    "v22_package": {"path": str(V22_PACKAGE), "sha256": "ec58e0114804e4c1091440db18129075174a207dcd92bf1bff1e78c8271be72b", "size": 8442},
    "v22_review_absent": str(V22_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
    "v21_action_root": {"entry_count": 30, "file_count": 25, "root": str(V21_ROOT), "tree_sha256": "f71d1652d9b488547386e3e476e5fd88a81e10e1a0739300e167acafb2bd28f6"},
    "v21_runtime_root": {"entry_count": 11, "file_count": 5, "root": str(V21_RUNTIME), "tree_sha256": "ac7cfdfddec92340e63f9892230c35f263da346b986fb843517da0ddaac7dd05"},
}
EXPECTED_V23_PRESERVATION = {
    "acceptance": {"path": str(V23_ACCEPTANCE), "sha256": "ca93ecc0dca75534ea344006f7fd763f7b189c4ac96eb42132e9e452d304cd1f", "size": 1562},
    "generated_bytecode_evidence": {"path": str(V23_PYC), "sha256": "6426aa2f4e4d7fd9c3402cca384eff0cc3f6b6ddb4b39e7132f980a169ed92a1", "size": 41134},
    "v23_action_root": {"entry_count": 17, "file_count": 11, "root": str(V23_ROOT), "tree_sha256": "35f705b9ae3a85eb62654dbea6fd2b624ba122cda1d4b6feee27188922ebe93a"},
}
EXPECTED_V24_PRESERVATION = {
    "acceptance_absent": str(V24_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
    "candidate_report": {"path": str(V24_CANDIDATE_REPORT), "sha256": "502faadde1d54e19ce67c22dd97d5e51dddd2109a731e30df331e7ee31ef9f25", "size": 14868},
    "checkpoint": {"path": str(V24_CHECKPOINT), "sha256": "6e119ccb559cc4ecd79e74063638e9290bda312f48e5bc2f8ab6725de8d6a972", "size": 5465},
    "negative_fixture_report": {"path": str(V24_NEGATIVE_REPORT), "sha256": "f657de13403ea0ba96224f1cbe35da45567f0238c411e470a3de93ce366308a9", "size": 24076},
    "owner_writable_modes_preserved": True,
    "package": {"path": str(V24_PACKAGE), "sha256": "12b1f6045e591dd0e16235aeb4347d205b5eac9ae461846efe53c972aeb19097", "size": 12737},
    "review_handoff": {"path": str(V24_REVIEW_HANDOFF), "sha256": "3973a334b1083a70d2153dafacccf9480df3d610011f37ebb72b84127e81ba2e", "size": 877},
    "terminal_fresh_l2_rejection": True,
    "v24_action_root": {"entry_count": 15, "file_count": 10, "root": str(V24_ROOT), "tree_sha256": "19e302a9cae9dc0f341253c45e3fad63a8ef5747b80ab786e32a394d27d5b672"},
}
EXPECTED_BYTECODE_CONTRACT = {
    "candidate_entrypoint": "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_candidate_v25.py",
    "candidate_exact_tree_before_after_required": True,
    "forbidden_inventory_artifact_components": ["__pycache__"],
    "forbidden_inventory_suffixes": [".pyc", ".pyo", ".swp", ".temp", ".tmp", "~"],
    "normal_import_mutation_must_be_detected": True,
    "normal_import_negative_fixture": "fixtures/V23_STYLE_NORMAL_IMPORT_INVENTORY_MUTATION.json",
    "post_acceptance_entrypoint": "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_post_acceptance_v25.py",
    "post_acceptance_exact_tree_before_after_required": True,
    "rejects_environment_only_suppression": True,
    "required_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
    "required_interpreter_flags": ["-B"],
    "sets_sys_dont_write_bytecode_before_in_tree_import": True,
}
EXPECTED_LITERAL_B_CONTRACT = {
    "argv_sources_must_agree": True,
    "candidate_entrypoint": "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_candidate_v25.py",
    "environment_only_v24_command_must_reject": True,
    "launch_guard": "tools/literal_b_launch_guard_v25.py",
    "launch_matrix": "tools/verify_qk_gbfp8_head64_literal_b_launch_matrix_v25.py",
    "negative_catalog": "fixtures/LITERAL_B_LAUNCH_NEGATIVE_CASES.json",
    "negative_case_ids": [
        "environment_only_exact_v24_command",
        "fake_b_after_script_path",
        "missing_script_path",
        "ambiguous_script_path",
        "argv_disagreement",
        "missing_pythondontwritebytecode",
        "non_null_pycacheprefix",
        "late_sys_dont_write_bytecode",
        "ineffective_suppression",
    ],
    "parse_only_tokens_before_resolved_script": True,
    "post_acceptance_entrypoint": "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_post_acceptance_v25.py",
    "proc_cmdline_required": "/proc/self/cmdline",
    "real_literal_b_token": "-B",
    "sys_flags_not_accepted_as_literal_b_evidence": True,
    "sys_orig_argv_required": True,
}
EXPECTED_FORBIDDEN_PATHS = [
    str(ACTION_ROOT / "live"),
    str(PROJECT_ROOT / "runtime/qk_gbfp8_head64_literal_b_mode_seal_v25_12b1f604"),
    str(BUILD_ROOT / "authority"),
    str(BUILD_ROOT / "credential.json"),
    str(BUILD_ROOT / "authority-ledger.json"),
    str(BUILD_ROOT / "owner-claim.json"),
    str(BUILD_ROOT / "result.json"),
    str(BUILD_ROOT / "first-terminal.json"),
]
EXPECTED_STATIC_POLICY = {
    "acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
    "allowed_relative_files": [
        "QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_PACKAGE.json",
        "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json",
        "fixtures/B0_SYNTHETIC_NEGATIVE_MUTATIONS.json",
        "fixtures/B0_VALID_SYNTHETIC_FIXTURE.json",
        "fixtures/LITERAL_B_LAUNCH_NEGATIVE_CASES.json",
        "fixtures/V23_STYLE_NORMAL_IMPORT_INVENTORY_MUTATION.json",
        "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_B0_SIDECAR_SCHEMA.json",
        "reference/QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_FRESH_L2_ACCEPTANCE_SCHEMA.json",
        "review/FRESH_L2_STATIC_ACCEPTANCE.json",
        "tools/validate_qk_gbfp8_head64_source_oracle_b0_v23.py",
        "tools/literal_b_launch_guard_v25.py",
        "tools/verify_qk_gbfp8_head64_literal_b_launch_matrix_v25.py",
        "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_candidate_v25.py",
        "tools/verify_qk_gbfp8_head64_literal_b_mode_seal_post_acceptance_v25.py",
    ],
    "inventory_policy": "EXACT_STATIC_FILES_PLUS_ONE_INDEPENDENT_FRESH_L2_ACCEPTANCE",
    "optional_before_acceptance": ["review/FRESH_L2_STATIC_ACCEPTANCE.json"],
}
AUDIT = {"official_payload_open_count": 0, "official_target_process_starts": 0}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value, raw


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"JSON object: {path}")
    return value


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash: {field}")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _open_is_mutating(args: tuple[Any, ...]) -> bool:
    if len(args) < 2:
        return False
    mode = args[1]
    if isinstance(mode, str):
        return any(character in mode for character in "wax+")
    if isinstance(mode, int):
        mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(mode & mask)
    return False


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0])
        except (TypeError, OSError):
            return
        if _inside(path, ACTION_ROOT) and _open_is_mutating(args):
            raise VerificationError(f"V25 action-tree write prohibited: {path}")
        if path.name in {"attention-substage-tensors.bin", "fixed-input-tensors.bin"}:
            AUDIT["official_payload_open_count"] += 1
            raise VerificationError("official payload open prohibited")
    if event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.replace", "os.chmod", "os.link", "os.symlink"}:
        for value in args[:2]:
            try:
                path = Path(value)
            except (TypeError, OSError):
                continue
            if _inside(path, ACTION_ROOT):
                raise VerificationError(f"V25 action-tree mutation prohibited: {event}:{path}")
    if event == "subprocess.Popen":
        AUDIT["official_target_process_starts"] += 1
        raise VerificationError("process start prohibited in inert V25 verifier")


def inventory_records(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"preservation root absent: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"preservation symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise VerificationError(f"unsupported preservation entry: {path}")
    return records


def inventory_digest(root: Path) -> dict[str, Any]:
    records = inventory_records(root)
    return {"entry_count": len(records), "file_count": sum(record["kind"] == "file" for record in records), "root": str(root), "tree_sha256": sha256_bytes(compact_bytes(records))}


def assert_no_forbidden_inventory(records: list[dict[str, Any]], contract: dict[str, Any], name: str) -> None:
    forbidden_components = set(contract["forbidden_inventory_artifact_components"])
    forbidden_suffixes = tuple(contract["forbidden_inventory_suffixes"])
    for record in records:
        relative = record["path"]
        parts = Path(relative).parts
        require(forbidden_components.isdisjoint(parts), f"{name} forbidden component: {relative}")
        require(not relative.endswith(forbidden_suffixes), f"{name} forbidden suffix: {relative}")


def verify_launch_contract(launch_proof: dict[str, Any] | None) -> dict[str, Any]:
    require(type(launch_proof) is dict, "literal -B launch proof absent")
    effectiveness = prove_effective_suppression()
    return {**launch_proof, **effectiveness}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"import spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_static_contract(package: dict[str, Any]) -> None:
    require(package["artifact_kind"] == "qk_gbfp8_head64_literal_b_mode_seal_static_v25_package", "package kind")
    require(package["authoritative_stage"] == "Base", "authoritative stage")
    require(package["root_id"] == "qk_gbfp8_head64_literal_b_mode_seal_static_v25_action_root", "root id")
    require(package["schema_version"] == 1, "schema version")
    require(package["action_identity"] == EXPECTED_ACTION_IDENTITY, "action identity")
    require(package["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "claim boundary")
    require(package["frozen_contract"] == EXPECTED_FROZEN, "frozen contract")
    require(package["b0_identity_control"] == EXPECTED_B0_IDENTITY, "B0 identity control")
    require(package["arithmetic_regressions"] == EXPECTED_ARITHMETIC, "arithmetic report contract")
    require(package["fresh_l2_review"] == EXPECTED_FRESH_L2, "Fresh-L2 contract")
    require(package["mission_contract"] == EXPECTED_MISSION, "mission contract")
    require(package["verification_toolchain"] == EXPECTED_TOOLCHAIN, "verification toolchain")
    require(package["predecessor_preservation"] == EXPECTED_PREDECESSOR, "predecessor preservation")
    require(package["v23_preservation"] == EXPECTED_V23_PRESERVATION, "failed V23 preservation")
    require(package["v24_preservation"] == EXPECTED_V24_PRESERVATION, "terminal V24 preservation")
    require(package["bytecode_free_verifier_contract"] == EXPECTED_BYTECODE_CONTRACT, "bytecode-free verifier contract")
    require(package["literal_b_launch_contract"] == EXPECTED_LITERAL_B_CONTRACT, "literal -B launch contract")
    require(package["forbidden_live_paths"] == EXPECTED_FORBIDDEN_PATHS, "forbidden paths")
    require(package["static_file_policy"] == EXPECTED_STATIC_POLICY, "static file policy")
    require(package["mode_seal"]["acceptance_slot"] == {"final_mode": "0444", "path": "review/FRESH_L2_STATIC_ACCEPTANCE.json", "pre_acceptance_state": "ABSENT", "schema_path": EXPECTED_FRESH_L2["acceptance_schema_path"]}, "acceptance slot")
    require(package["mode_seal"]["ordinary_directory_mode"] == "0555", "ordinary directory seal")
    require(package["mode_seal"]["pre_acceptance_review_directory_mode"] == "0755", "review create-once mode")
    require(package["mode_seal"]["sealed_file_mode"] == "0444", "sealed file mode")


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    validate_static_contract(package)
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    allowed = set(package["static_file_policy"]["allowed_relative_files"])
    acceptance_relative = package["static_file_policy"]["acceptance_relative_path"]
    observed = set()
    observed_directories = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"V25 symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            observed_directories.add(relative)
            continue
        require(stat.S_ISREG(info.st_mode), f"V25 non-regular file: {relative}")
        observed.add(relative)
    expected = set(allowed)
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    else:
        expected.remove(acceptance_relative)
        require(not os.path.lexists(ACCEPTANCE), "Engineer materialized Fresh-L2 acceptance")
    expected_directories = set()
    for relative in expected:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    expected_directories.add("review")
    require(observed == expected, f"V25 inventory expected={sorted(expected)} observed={sorted(observed)}")
    require(observed_directories == expected_directories, f"V25 directories expected={sorted(expected_directories)} observed={sorted(observed_directories)}")
    records = inventory_records(ACTION_ROOT)
    assert_no_forbidden_inventory(records, package["bytecode_free_verifier_contract"], "V25 inventory")
    root_mode = f"{stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode):04o}"
    require(root_mode == "0555", "V25 action root mode seal")
    for record in records:
        if record["kind"] == "directory":
            expected_mode = "0555" if allow_acceptance or record["path"] != "review" else "0755"
            require(record["mode"] == expected_mode, f"V25 directory mode seal: {record['path']}")
        else:
            require(record["mode"] == "0444", f"V25 file mode seal: {record['path']}")
    pre_records = [record for record in records if record["path"] != acceptance_relative]
    normalized_seal_records: list[dict[str, Any]] = [{"kind": "directory", "mode": root_mode, "path": "."}]
    for record in pre_records:
        item = dict(record)
        if allow_acceptance and item["kind"] == "directory" and item["path"] == "review":
            item["mode"] = "0755"
        if item["path"] == PACKAGE.name:
            item.pop("sha256")
            item["sha256_field"] = "package_content_sha256"
            item["sha256_scope"] = "canonical_object_without_package_content_sha256"
        normalized_seal_records.append(item)
    require(normalized_seal_records == package["mode_seal"]["pre_acceptance_records"], "V25 pre-acceptance mode-seal manifest")
    normalized = pre_records
    return {
        "candidate_action_tree_sha256": sha256_bytes(compact_bytes(normalized)),
        "mode_seal_exact": True,
        "pre_acceptance_record_count": len(normalized_seal_records),
        "review_directory_mode": "0555" if allow_acceptance else "0755",
        "inventory_policy": package["static_file_policy"]["inventory_policy"],
        "static_package_file_count": len(allowed) - 1,
    }


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file(), f"generated file absent: {relative}")
        require(path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"], f"generated file binding: {relative}")
        require(f"{stat.S_IMODE(os.lstat(path).st_mode):04o}" == "0444", f"generated file mode seal: {relative}")
    return {"generated_file_count": len(package["generated_files"]), "generated_files_sha256": sha256_bytes(compact_bytes(package["generated_files"]))}


def verify_mission_and_toolchain() -> dict[str, Any]:
    require(MISSION.stat().st_size == EXPECTED_MISSION["size"] and sha256_file(MISSION) == EXPECTED_MISSION["sha256"], "mission contract drift")
    executable = Path(EXPECTED_TOOLCHAIN["interpreter_path"])
    require(sha256_file(executable) == EXPECTED_TOOLCHAIN["interpreter_sha256"], "interpreter drift")
    require(sys.version.split()[0] == EXPECTED_TOOLCHAIN["interpreter_version"], "interpreter version")
    require(importlib.metadata.version("jsonschema") == EXPECTED_TOOLCHAIN["jsonschema_version"], "jsonschema version")
    return {"mission_sha256": EXPECTED_MISSION["sha256"], "interpreter_sha256": EXPECTED_TOOLCHAIN["interpreter_sha256"], "jsonschema_version": EXPECTED_TOOLCHAIN["jsonschema_version"]}


def verify_v23_preservation() -> dict[str, Any]:
    require(inventory_digest(V23_ROOT) == EXPECTED_V23_PRESERVATION["v23_action_root"], "failed V23 action tree drift")
    for key, path in (("acceptance", V23_ACCEPTANCE), ("generated_bytecode_evidence", V23_PYC)):
        record = EXPECTED_V23_PRESERVATION[key]
        require(path.stat().st_size == record["size"] and sha256_file(path) == record["sha256"], f"failed V23 preserved file drift: {key}")
    acceptance, _ = canonical(V23_ACCEPTANCE)
    require(acceptance["action_id"] == EXPECTED_V23_ACTION_ID and acceptance["accepted"] is True, "failed V23 acceptance identity")
    return {
        "acceptance_file_sha256": EXPECTED_V23_PRESERVATION["acceptance"]["sha256"],
        "generated_bytecode_evidence_sha256": EXPECTED_V23_PRESERVATION["generated_bytecode_evidence"]["sha256"],
        "v23_action_tree_sha256": EXPECTED_V23_PRESERVATION["v23_action_root"]["tree_sha256"],
    }


def verify_v24_preservation() -> dict[str, Any]:
    require(inventory_digest(V24_ROOT) == EXPECTED_V24_PRESERVATION["v24_action_root"], "terminal V24 action tree drift")
    require(not os.path.lexists(EXPECTED_V24_PRESERVATION["acceptance_absent"]), "terminal V24 acceptance unexpectedly exists")
    paths = {
        "candidate_report": V24_CANDIDATE_REPORT,
        "checkpoint": V24_CHECKPOINT,
        "negative_fixture_report": V24_NEGATIVE_REPORT,
        "package": V24_PACKAGE,
        "review_handoff": V24_REVIEW_HANDOFF,
    }
    for key, path in paths.items():
        record = EXPECTED_V24_PRESERVATION[key]
        require(path.stat().st_size == record["size"] and sha256_file(path) == record["sha256"], f"terminal V24 preserved file drift: {key}")
    package, _ = canonical(V24_PACKAGE)
    candidate_report, _ = canonical(V24_CANDIDATE_REPORT)
    review_handoff = json_object(V24_REVIEW_HANDOFF)
    require(package["action_identity"]["action_id"] == EXPECTED_V24_ACTION_ID, "terminal V24 action identity")
    require(candidate_report["status"] == "PASS_V24_STATIC_CANDIDATE_PENDING_FRESH_L2", "terminal V24 candidate status")
    require(review_handoff["review"]["status"] == "replan_requested", "terminal V24 review status")
    return {
        "acceptance_absent": True,
        "owner_writable_modes_preserved": True,
        "package_file_sha256": EXPECTED_V24_PRESERVATION["package"]["sha256"],
        "terminal_fresh_l2_rejection": True,
        "v24_action_tree_sha256": EXPECTED_V24_PRESERVATION["v24_action_root"]["tree_sha256"],
    }


def verify_launch_catalog() -> dict[str, Any]:
    catalog, raw = canonical(LAUNCH_CATALOG)
    observed = catalog["fixture_sha256"]
    unhashed = dict(catalog)
    unhashed.pop("fixture_sha256")
    require(observed == sha256_bytes(compact_bytes(unhashed)), "literal -B launch catalog self hash")
    require(catalog["artifact_kind"] == "qk_gbfp8_head64_literal_b_mode_seal_v25_launch_negative_catalog", "launch catalog kind")
    require(catalog["case_count"] == len(catalog["cases"]) == 9, "launch catalog count")
    require([case["id"] for case in catalog["cases"]] == EXPECTED_LITERAL_B_CONTRACT["negative_case_ids"], "launch catalog ids")
    require(catalog["claim_boundary"] == EXPECTED_CLAIM, "launch catalog claim")
    require(catalog["official_payload_open_count"] == catalog["official_evaluator_invocation_count"] == catalog["official_target_process_starts"] == 0, "launch catalog effects")
    return {"case_count": 9, "catalog_file_sha256": sha256_bytes(raw), "catalog_sha256": observed}


def verify_independent_sources() -> dict[str, Any]:
    official, official_raw = canonical(OFFICIAL_PACKAGE)
    require(sha256_bytes(official_raw) == EXPECTED_FROZEN["benchmark_package_sha256"], "benchmark package hash")
    inputs = official["official_benchmark"]["input_bindings"]
    expected_from_official = {
        "benchmark_package_sha256": sha256_bytes(official_raw),
        "candidate_definitions": official["candidates"],
        "candidate_order": official["selection_policy"]["order"],
        "failure_if_none_pass": official["selection_policy"]["failure_if_none_pass"],
        "hard_gates": official["hard_gates"],
        "input_token_ids_sha256": inputs["input_token_ids_sha256"],
        "lane_metadata_byte_count": inputs["lane_metadata"]["byte_count"],
        "lane_metadata_sha256": inputs["lane_metadata"]["sha256"],
        "model_identity_sha256": official["official_benchmark"]["model_identity_sha256"],
        "numerical_tensor_identities": [inputs["tensor_records"]["realized_key_source"], inputs["tensor_records"]["realized_query_source"], inputs["tensor_records"]["bf16_oracle_scores"]],
        "require_every_hard_gate": official["selection_policy"]["require_every_hard_gate"],
        "score_error_affects_selection": official["selection_policy"]["score_error_affects_selection"],
        "sealed_set_id": inputs["sealed_set_id"],
        "selection_policy": official["selection_policy"]["policy"],
        "tensor_bundle_byte_count": inputs["tensor_bundle"]["byte_count"],
        "tensor_bundle_sha256": inputs["tensor_bundle"]["sha256"],
    }
    for key, value in expected_from_official.items():
        require(EXPECTED_FROZEN[key] == value, f"official frozen source: {key}")
    require(sha256_file(OFFICIAL_EVALUATOR) == EXPECTED_FROZEN["official_evaluator_sha256"], "official evaluator hash")
    require(sha256_file(OFFICIAL_RESULT_SCHEMA) == EXPECTED_FROZEN["official_result_schema_sha256"], "official result schema hash")
    require(sha256_file(BINDING_TABLE) == EXPECTED_FROZEN["binding_table_sha256"], "binding table hash")
    binding = json_object(BINDING_TABLE)
    require(binding["record_count"] == EXPECTED_FROZEN["tensor_record_count"], "binding record count")
    require(binding["selected_tensor_names"] == EXPECTED_FROZEN["selected_tensor_names"], "selected tensor names")
    result, result_raw = canonical(V21_RESULT)
    require(len(result_raw) == EXPECTED_FROZEN["v21_result_byte_count"] and sha256_bytes(result_raw) == EXPECTED_FROZEN["v21_result_file_sha256"], "V21 result bytes")
    require(result["irreversible_action_id"] == EXPECTED_V21_ACTION_ID, "V21 result action")
    diagnosis_raw = DIAGNOSIS.read_bytes()
    require(sha256_bytes(diagnosis_raw) == EXPECTED_FROZEN["diagnosis_file_sha256"], "diagnosis hash")
    diagnosis = json.loads(diagnosis_raw.decode("ascii"))
    frozen = diagnosis["v22_static_successor"]["frozen_bindings"]
    require(frozen["candidate_order"] == EXPECTED_FROZEN["candidate_order"], "diagnosis candidate order")
    require(frozen["hard_gates"] == EXPECTED_FROZEN["hard_gates"], "diagnosis hard gates")
    return {"independent_source_count": 9, "v21_result_file_sha256": sha256_bytes(result_raw)}


def verify_v22_and_predecessors() -> dict[str, Any]:
    require(inventory_digest(V22_ROOT) == EXPECTED_PREDECESSOR["v22_action_root"], "V22 action tree drift")
    require(not os.path.lexists(EXPECTED_PREDECESSOR["v22_review_absent"]), "V22 review absence changed")
    for key in ("v22_package", "v22_candidate_report"):
        record = EXPECTED_PREDECESSOR[key]
        path = Path(record["path"])
        require(path.stat().st_size == record["size"] and sha256_file(path) == record["sha256"], f"predecessor file drift: {key}")
    v22 = load_module(V22_CANDIDATE_VERIFIER, "v25_bound_v22_candidate_verifier")
    fresh = v22.verify(False)
    prior, prior_raw = canonical(V22_CANDIDATE_REPORT)
    require(fresh == prior and compact_bytes(fresh) == prior_raw, "V22 candidate report recomputation")
    require(fresh["status"] == "PASS_V22_STATIC_CANDIDATE_PENDING_FRESH_L2", "V22 remains rejected candidate")
    require(fresh["predecessor_preservation"]["v21_action_tree_sha256"] == EXPECTED_PREDECESSOR["v21_action_root"]["tree_sha256"], "V21 action preservation")
    require(fresh["predecessor_preservation"]["v21_runtime_tree_sha256"] == EXPECTED_PREDECESSOR["v21_runtime_root"]["tree_sha256"], "V21 runtime preservation")
    return {"v22_action_tree_sha256": EXPECTED_PREDECESSOR["v22_action_root"]["tree_sha256"], "v22_candidate_report_file_sha256": sha256_bytes(prior_raw), "v22_acceptance_present": False}


def verify_regression_report() -> dict[str, Any]:
    carried = REGRESSION_REPORT.read_bytes()
    source = V22_REPORT.read_bytes()
    require(carried == source, "V22 arithmetic report bytes changed")
    report, raw = canonical(REGRESSION_REPORT)
    verify_self_hash(report, "report_sha256")
    require(sha256_bytes(raw) == EXPECTED_ARITHMETIC["report_file_sha256"], "arithmetic report file hash")
    require(report["official_payload_open_count"] == report["official_evaluator_invocation_count"] == report["official_target_process_starts"] == 0, "arithmetic report official effects")
    return {"byte_identical_to_v22": True, "report_file_sha256": sha256_bytes(raw), "report_sha256": report["report_sha256"]}


def _set_path(value: Any, path: list[Any], replacement: Any, delete: bool = False) -> None:
    target = value
    for part in path[:-1]:
        target = target[part]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = replacement


def _scalar_paths(value: Any, prefix: tuple[Any, ...] = ()) -> Iterable[tuple[Any, ...]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _scalar_paths(value[key], prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _scalar_paths(item, prefix + (index,))
    else:
        yield prefix


def _mutated_scalar(value: Any) -> Any:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if value is None:
        return "unexpected-runtime-namespace"
    if type(value) is str:
        return ("0" * 64) if len(value) == 64 and value != "0" * 64 else value + "__MUTATED"
    raise VerificationError(f"unsupported mutation scalar: {type(value)}")


def verify_package_mutations(package: dict[str, Any]) -> dict[str, Any]:
    source, source_raw = canonical(V24_NEGATIVE_REPORT)
    carried_paths = source["package_contract_mutations"]["rejected_mutation_paths"]
    require(len(carried_paths) == source["package_contract_mutations"]["rejected_mutation_count"] == 187, "V24 carried package mutation catalog")
    rejected = []
    for dotted in carried_paths:
        path: list[Any] = [int(part) if part.isdigit() else part for part in dotted.split(".")]
        changed = deepcopy(package)
        original: Any = changed
        for part in path:
            original = original[part]
        _set_path(changed, path, _mutated_scalar(original))
        try:
            validate_static_contract(changed)
        except VerificationError:
            rejected.append(dotted)
        else:
            raise VerificationError(f"fail-closed carried package mutation accepted: {dotted}")
    required = {
        "frozen_contract.input_token_ids_sha256",
        "frozen_contract.model_identity_sha256",
        "frozen_contract.sealed_set_id",
        "frozen_contract.v21_result_file_sha256",
        "frozen_contract.v21_result_byte_count",
        "frozen_contract.selection_policy",
        "frozen_contract.score_error_affects_selection",
        "b0_identity_control.quantization_completion",
        "claim_boundary.execution_authorized",
        "claim_boundary.runtime_namespace_materialized",
    }
    require(required.issubset(set(rejected)), "required frozen mutation coverage")
    require(len(rejected) == 187, f"carried V24 package mutation count: {len(rejected)}")
    return {"rejected_mutation_count": len(rejected), "rejected_mutation_paths": rejected, "required_v22_defect_paths_rejected": sorted(required), "source_v24_negative_report_file_sha256": sha256_bytes(source_raw)}


def verify_v25_extension_mutations(package: dict[str, Any]) -> dict[str, Any]:
    rejected = []
    for section in ("v23_preservation", "v24_preservation", "bytecode_free_verifier_contract", "literal_b_launch_contract", "mode_seal"):
        for relative in _scalar_paths(package[section]):
            changed = deepcopy(package)
            path = (section,) + relative
            original = changed
            for part in path:
                original = original[part]
            _set_path(changed, list(path), _mutated_scalar(original))
            try:
                validate_static_contract(changed)
                if section == "mode_seal":
                    verify_inventory(changed, False)
            except VerificationError:
                rejected.append(".".join(str(part) for part in path))
            else:
                raise VerificationError(f"fail-closed V25 extension mutation accepted: {path}")
    changed = deepcopy(package)
    fixture_index = changed["static_file_policy"]["allowed_relative_files"].index(EXPECTED_BYTECODE_CONTRACT["normal_import_negative_fixture"])
    path = ("static_file_policy", "allowed_relative_files", fixture_index)
    _set_path(changed, list(path), EXPECTED_BYTECODE_CONTRACT["normal_import_negative_fixture"] + "__MUTATED")
    try:
        validate_static_contract(changed)
    except VerificationError:
        rejected.append(".".join(str(part) for part in path))
    else:
        raise VerificationError("fail-closed V25 fixture inventory mutation accepted")
    changed = deepcopy(package)
    launch_fixture_index = changed["static_file_policy"]["allowed_relative_files"].index(EXPECTED_LITERAL_B_CONTRACT["negative_catalog"])
    launch_path = ("static_file_policy", "allowed_relative_files", launch_fixture_index)
    _set_path(changed, list(launch_path), EXPECTED_LITERAL_B_CONTRACT["negative_catalog"] + "__MUTATED")
    try:
        validate_static_contract(changed)
    except VerificationError:
        rejected.append(".".join(str(part) for part in launch_path))
    else:
        raise VerificationError("fail-closed V25 launch catalog inventory mutation accepted")
    return {"rejected_mutation_count": len(rejected), "rejected_mutation_paths": rejected}


def verify_b0_fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    validator = load_module(B0_VALIDATOR, "v23_bound_b0_validator")
    schema = json_object(B0_SCHEMA)
    valid, valid_raw = canonical(B0_VALID_FIXTURE)
    valid_summary = validator.validate_document(valid, schema, EXPECTED_FROZEN)
    catalog, catalog_raw = canonical(B0_MUTATIONS)
    rejected = []
    for case in catalog["cases"]:
        changed = deepcopy(valid)
        _set_path(changed, case["path"], case.get("value"), delete=case["operation"] == "delete")
        changed["sidecar_sha256"] = validator.self_hash(changed)
        try:
            validator.validate_document(changed, schema, EXPECTED_FROZEN)
        except validator.B0ValidationError as error:
            rejected.append({"id": case["id"], "reason": str(error)})
        else:
            raise VerificationError(f"fail-closed B0 mutation accepted: {case['id']}")
    required = {"duplicate_row_class", "omitted_row_class", "unexpected_row_class", "duplicate_residual_label", "omitted_residual_label", "unexpected_residual_label", "b0_as_quantization", "execution_authority_claim", "runtime_namespace_claim"}
    require(required.issubset({item["id"] for item in rejected}), "required B0 mutation coverage")
    summary = {
        "catalog_file_sha256": sha256_bytes(catalog_raw),
        "rejected_mutation_count": len(rejected),
        "rejected_mutations": rejected,
        "schema_file_sha256": sha256_file(B0_SCHEMA),
        "valid_fixture_file_sha256": sha256_bytes(valid_raw),
        "valid_fixture_summary": valid_summary,
        "validator_file_sha256": sha256_file(B0_VALIDATOR),
    }
    return summary, valid


def verify_v23_style_normal_import_fixture() -> dict[str, Any]:
    fixture, raw = canonical(NORMAL_IMPORT_FIXTURE)
    verify_self_hash(fixture, "fixture_sha256")
    expected = {
        "artifact_kind": "qk_gbfp8_head64_bytecode_free_verifier_v24_v23_style_normal_import_mutation_fixture",
        "claim_boundary": EXPECTED_CLAIM,
        "expected_added_component": "__pycache__",
        "expected_added_suffix": ".pyc",
        "fixture_sha256": fixture["fixture_sha256"],
        "module_name": "ace2_v24_negative_v23_style_normal_import",
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "schema_version": 1,
        "source_filename": "v23_style_candidate.py",
        "source_text": "VALUE = 23\n",
        "sys_dont_write_bytecode_during_negative_import": False,
    }
    require(fixture == expected, "V23-style normal-import fixture contract")
    with tempfile.TemporaryDirectory(prefix="ace2-v25-normal-import-negative-") as directory:
        root = Path(directory)
        source = root / fixture["source_filename"]
        source.write_bytes(fixture["source_text"].encode("ascii"))
        before = inventory_records(root)
        prior = sys.dont_write_bytecode
        require(prior is True, "negative fixture requires suppressed parent process")
        sys.dont_write_bytecode = False
        try:
            load_module(source, fixture["module_name"])
        finally:
            sys.dont_write_bytecode = prior
            sys.modules.pop(fixture["module_name"], None)
        after = inventory_records(root)
        before_paths = {record["path"] for record in before}
        added_paths = sorted(record["path"] for record in after if record["path"] not in before_paths)
        require(before != after, "V23-style normal import did not mutate inventory")
        require(any(fixture["expected_added_component"] in Path(path).parts for path in added_paths), "normal import __pycache__ mutation absent")
        require(any(path.endswith(fixture["expected_added_suffix"]) for path in added_paths), "normal import pyc mutation absent")
    require(sys.dont_write_bytecode is True, "negative fixture failed to restore bytecode suppression")
    return {
        "added_paths": added_paths,
        "fixture_file_sha256": sha256_bytes(raw),
        "fixture_sha256": fixture["fixture_sha256"],
        "inventory_mutation_detected": True,
        "normal_import_style": "IN_TREE_MODULE_IMPORT_WITHOUT_EARLY_SUPPRESSION",
    }


def verify_no_live_effects(package: dict[str, Any]) -> dict[str, Any]:
    for value in package["forbidden_live_paths"]:
        require(not os.path.lexists(value), f"forbidden V25 live path exists: {value}")
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "inert audit boundary")
    return {"forbidden_path_count": len(package["forbidden_live_paths"]), "official_evaluator_invocation_count": 0, **AUDIT}


def build_negative_report(package: dict[str, Any]) -> dict[str, Any]:
    b0, _ = verify_b0_fixtures()
    report = {
        "artifact_kind": "qk_gbfp8_head64_literal_b_mode_seal_v25_synthetic_negative_fixture_report",
        "b0_document_mutations": b0,
        "claim_boundary": EXPECTED_CLAIM,
        "literal_b_launch_negative_catalog": verify_launch_catalog(),
        "normal_import_inventory_mutation": verify_v23_style_normal_import_fixture(),
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_contract_mutations": verify_package_mutations(package),
        "v25_extension_mutations": verify_v25_extension_mutations(package),
        "status": "PASS_V25_LITERAL_B_MODE_SEAL_AND_SYNTHETIC_MUTATION_REJECTION",
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def verify(allow_acceptance: bool = False, launch_proof: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    AUDIT.update({"official_payload_open_count": 0, "official_target_process_starts": 0})
    launch = verify_launch_contract(launch_proof)
    root_mode_before = f"{stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode):04o}"
    action_before = inventory_records(ACTION_ROOT)
    package, package_raw = verify_manifest()
    assert_no_forbidden_inventory(action_before, package["bytecode_free_verifier_contract"], "V25 action tree before verifier")
    negative = build_negative_report(package)
    inventory = verify_inventory(package, allow_acceptance)
    report = {
        "artifact_kind": "qk_gbfp8_head64_literal_b_mode_seal_static_v25_candidate_verifier_report",
        "b0_schema_and_validator": negative["b0_document_mutations"],
        "bytecode_free_launch": launch,
        "claim_boundary": EXPECTED_CLAIM,
        "frozen_sources": verify_independent_sources(),
        "generated_files": verify_generated_files(package),
        "inventory": inventory,
        "live_effects": verify_no_live_effects(package),
        "mission_and_toolchain": verify_mission_and_toolchain(),
        "negative_fixture_report_sha256": negative["report_sha256"],
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_bytes(package_raw),
        "predecessor_preservation": verify_v22_and_predecessors(),
        "regressions": verify_regression_report(),
        "rejected_package_mutation_count": negative["package_contract_mutations"]["rejected_mutation_count"],
        "status": "PASS_V25_STATIC_CANDIDATE_PENDING_FRESH_L2",
        "v23_preservation": verify_v23_preservation(),
        "v24_preservation": verify_v24_preservation(),
        "v25_execution_authority_granted": False,
    }
    action_after = inventory_records(ACTION_ROOT)
    root_mode_after = f"{stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode):04o}"
    require(action_after == action_before and root_mode_after == root_mode_before, "V25 candidate verifier mutated exact action tree or root mode")
    normalized = [record for record in action_before if record["path"] != EXPECTED_STATIC_POLICY["acceptance_relative_path"]]
    bound_inventory = [{"kind": "directory", "mode": root_mode_before, "path": "."}, *normalized]
    report["inventory_guard"] = {
        "acceptance_excluded_from_bound_candidate_tree": True,
        "candidate_action_tree_before_sha256": sha256_bytes(compact_bytes(bound_inventory)),
        "candidate_action_tree_exact_match": True,
        "candidate_action_tree_after_sha256": sha256_bytes(compact_bytes(bound_inventory)),
        "root_mode": root_mode_before,
    }
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "final inert audit boundary")
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report, negative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BUILD_ROOT / "QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_CANDIDATE_REPORT.json")
    parser.add_argument("--negative-output", type=Path, default=BUILD_ROOT / "QK_GBFP8_HEAD64_LITERAL_B_MODE_SEAL_STATIC_V25_NEGATIVE_FIXTURE_REPORT.json")
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    report, negative = verify(False, ENTRY_LAUNCH_PROOF)
    for path in (args.output, args.negative_output):
        try:
            path.resolve(strict=False).relative_to(BUILD_ROOT.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise VerificationError(f"V25 report output must stay inside declared build root: {path}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.negative_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compact_bytes(report))
    args.negative_output.write_bytes(compact_bytes(negative))
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
