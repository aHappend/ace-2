#!/usr/bin/env python3
"""Acceptance-aware inert verifier for additive static-only V23."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build/v23-source-oracle-verifier-hardening-static-0001"
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_PACKAGE.json"
REGRESSION_REPORT = ACTION_ROOT / "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json"
B0_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_B0_SIDECAR_SCHEMA.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_FRESH_L2_ACCEPTANCE_SCHEMA.json"
B0_VALID_FIXTURE = ACTION_ROOT / "fixtures/B0_VALID_SYNTHETIC_FIXTURE.json"
B0_MUTATIONS = ACTION_ROOT / "fixtures/B0_SYNTHETIC_NEGATIVE_MUTATIONS.json"
B0_VALIDATOR = ACTION_ROOT / "tools/validate_qk_gbfp8_head64_source_oracle_b0_v23.py"
POST_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_source_oracle_verifier_hardening_post_acceptance_v23.py"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
MISSION = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/v23verifierhardening/mission.json")
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

EXPECTED_ACTION_ID = "ace2:qk-gbfp8-base-v23:source-oracle-verifier-hardening:ec58e011:additive-0001"
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
    "predecessor_action_id": EXPECTED_V22_ACTION_ID,
    "predecessor_rejected_immutable": True,
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
    "v22_accepted_reinterpreted_or_mutated": False,
    "v23_executed": False,
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
    "acceptance_schema_file_sha256": "94829657189b881850b9a7ce3f3bc88accd60566aea7c8d0e91be0e0cebdad29",
    "acceptance_schema_path": "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    "decision_requested": "ACCEPT_STATIC_PACKAGE",
    "engineer_self_review_can_close": False,
    "required_role": "Fresh-L2",
    "static_acceptance_grants_execution_authority": False,
    "status": "PENDING_INDEPENDENT_REVIEW",
}
EXPECTED_MISSION = {
    "path": str(MISSION),
    "sha256": "6817d716034f5562471cab4208540fd4f92daa098ae91d6c476b9dc8e66cd937",
    "size": 3075,
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
EXPECTED_FORBIDDEN_PATHS = [
    str(ACTION_ROOT / "live"),
    str(PROJECT_ROOT / "runtime/qk_gbfp8_head64_source_oracle_verifier_hardening_v23_ec58e011"),
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
        "QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_PACKAGE.json",
        "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json",
        "fixtures/B0_SYNTHETIC_NEGATIVE_MUTATIONS.json",
        "fixtures/B0_VALID_SYNTHETIC_FIXTURE.json",
        "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_B0_SIDECAR_SCHEMA.json",
        "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_FRESH_L2_ACCEPTANCE_SCHEMA.json",
        "review/FRESH_L2_STATIC_ACCEPTANCE.json",
        "tools/validate_qk_gbfp8_head64_source_oracle_b0_v23.py",
        "tools/verify_qk_gbfp8_head64_source_oracle_verifier_hardening_candidate_v23.py",
        "tools/verify_qk_gbfp8_head64_source_oracle_verifier_hardening_post_acceptance_v23.py",
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


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0])
        except (TypeError, OSError):
            return
        if path.name in {"attention-substage-tensors.bin", "fixed-input-tensors.bin"}:
            AUDIT["official_payload_open_count"] += 1
            raise VerificationError("official payload open prohibited")
    if event == "subprocess.Popen":
        AUDIT["official_target_process_starts"] += 1
        raise VerificationError("process start prohibited in inert V23 verifier")


def inventory_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"preservation root absent: {root}")
    records = []
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
    return {"entry_count": len(records), "file_count": sum(record["kind"] == "file" for record in records), "root": str(root), "tree_sha256": sha256_bytes(compact_bytes(records))}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"import spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_static_contract(package: dict[str, Any]) -> None:
    require(package["artifact_kind"] == "qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_package", "package kind")
    require(package["authoritative_stage"] == "Base", "authoritative stage")
    require(package["root_id"] == "qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_action_root", "root id")
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
    require(package["forbidden_live_paths"] == EXPECTED_FORBIDDEN_PATHS, "forbidden paths")
    require(package["static_file_policy"] == EXPECTED_STATIC_POLICY, "static file policy")


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    validate_static_contract(package)
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    allowed = set(package["static_file_policy"]["allowed_relative_files"])
    acceptance_relative = package["static_file_policy"]["acceptance_relative_path"]
    observed = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"V23 symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), f"V23 non-regular file: {relative}")
        require(not relative.endswith((".pyc", ".pyo")) and "__pycache__" not in relative, f"generated Python artifact: {relative}")
        observed.add(relative)
    expected = set(allowed)
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    else:
        expected.remove(acceptance_relative)
        require(not os.path.lexists(ACCEPTANCE), "Engineer materialized Fresh-L2 acceptance")
    require(observed == expected, f"V23 inventory expected={sorted(expected)} observed={sorted(observed)}")
    return {"inventory_policy": package["static_file_policy"]["inventory_policy"], "static_package_file_count": len(allowed) - 1}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file(), f"generated file absent: {relative}")
        require(path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"], f"generated file binding: {relative}")
    return {"generated_file_count": len(package["generated_files"]), "generated_files_sha256": sha256_bytes(compact_bytes(package["generated_files"]))}


def verify_mission_and_toolchain() -> dict[str, Any]:
    require(MISSION.stat().st_size == EXPECTED_MISSION["size"] and sha256_file(MISSION) == EXPECTED_MISSION["sha256"], "mission contract drift")
    executable = Path(EXPECTED_TOOLCHAIN["interpreter_path"])
    require(sha256_file(executable) == EXPECTED_TOOLCHAIN["interpreter_sha256"], "interpreter drift")
    require(sys.version.split()[0] == EXPECTED_TOOLCHAIN["interpreter_version"], "interpreter version")
    require(importlib.metadata.version("jsonschema") == EXPECTED_TOOLCHAIN["jsonschema_version"], "jsonschema version")
    return {"mission_sha256": EXPECTED_MISSION["sha256"], "interpreter_sha256": EXPECTED_TOOLCHAIN["interpreter_sha256"], "jsonschema_version": EXPECTED_TOOLCHAIN["jsonschema_version"]}


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
    v22 = load_module(V22_CANDIDATE_VERIFIER, "v23_bound_v22_candidate_verifier")
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
    sections = ["action_identity", "claim_boundary", "frozen_contract", "b0_identity_control", "arithmetic_regressions", "fresh_l2_review", "mission_contract", "verification_toolchain", "predecessor_preservation", "forbidden_live_paths", "static_file_policy"]
    rejected = []
    for section in sections:
        for relative in _scalar_paths(package[section]):
            changed = deepcopy(package)
            path = (section,) + relative
            original = changed
            for part in path:
                original = original[part]
            _set_path(changed, list(path), _mutated_scalar(original))
            try:
                validate_static_contract(changed)
            except VerificationError:
                rejected.append(".".join(str(part) for part in path))
            else:
                raise VerificationError(f"fail-closed package mutation accepted: {path}")
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
    return {"rejected_mutation_count": len(rejected), "rejected_mutation_paths": rejected, "required_v22_defect_paths_rejected": sorted(required)}


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


def verify_no_live_effects(package: dict[str, Any]) -> dict[str, Any]:
    for value in package["forbidden_live_paths"]:
        require(not os.path.lexists(value), f"forbidden V23 live path exists: {value}")
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "inert audit boundary")
    return {"forbidden_path_count": len(package["forbidden_live_paths"]), "official_evaluator_invocation_count": 0, **AUDIT}


def build_negative_report(package: dict[str, Any]) -> dict[str, Any]:
    b0, _ = verify_b0_fixtures()
    report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_verifier_hardening_v23_synthetic_negative_fixture_report",
        "b0_document_mutations": b0,
        "claim_boundary": EXPECTED_CLAIM,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_contract_mutations": verify_package_mutations(package),
        "status": "PASS_V23_EXHAUSTIVE_SYNTHETIC_MUTATION_REJECTION",
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def verify(allow_acceptance: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    AUDIT.update({"official_payload_open_count": 0, "official_target_process_starts": 0})
    package, package_raw = verify_manifest()
    negative = build_negative_report(package)
    report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_candidate_verifier_report",
        "b0_schema_and_validator": negative["b0_document_mutations"],
        "claim_boundary": EXPECTED_CLAIM,
        "frozen_sources": verify_independent_sources(),
        "generated_files": verify_generated_files(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "live_effects": verify_no_live_effects(package),
        "mission_and_toolchain": verify_mission_and_toolchain(),
        "negative_fixture_report_sha256": negative["report_sha256"],
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_bytes(package_raw),
        "predecessor_preservation": verify_v22_and_predecessors(),
        "regressions": verify_regression_report(),
        "rejected_package_mutation_count": negative["package_contract_mutations"]["rejected_mutation_count"],
        "status": "PASS_V23_STATIC_CANDIDATE_PENDING_FRESH_L2",
        "v23_execution_authority_granted": False,
    }
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "final inert audit boundary")
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report, negative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BUILD_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_CANDIDATE_REPORT.json")
    parser.add_argument("--negative-output", type=Path, default=BUILD_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_VERIFIER_HARDENING_STATIC_V23_NEGATIVE_FIXTURE_REPORT.json")
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    report, negative = verify(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.negative_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compact_bytes(report))
    args.negative_output.write_bytes(compact_bytes(negative))
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
