#!/usr/bin/env python3
"""Acceptance-aware inert verifier for the additive static-only V22 candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_ATTRIBUTION_STATIC_V22_PACKAGE.json"
REGRESSION_TOOL = ACTION_ROOT / "tools/qk_gbfp8_head64_source_oracle_attribution_regressions_v22.py"
REGRESSION_REPORT = ACTION_ROOT / "evidence/PUBLIC_SYNTHETIC_ARITHMETIC_REGRESSION_REPORT.json"
B0_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_ATTRIBUTION_STATIC_V22_B0_SIDECAR_SCHEMA.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_SOURCE_ORACLE_ATTRIBUTION_STATIC_V22_FRESH_L2_ACCEPTANCE_SCHEMA.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
POST_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_source_oracle_attribution_post_acceptance_v22.py"
DIAGNOSIS = PROJECT_ROOT / "diagnosis/QK_GBFP8_HEAD64_V21_BASE_FAILURE_DIAGNOSIS_AND_V22_STATIC_SUCCESSOR.json"
V21_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root"
V21_PACKAGE = V21_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
V21_RUNTIME = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba"
V21_RESULT = V21_RUNTIME / "primary/result/base/result.json"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
OFFICIAL_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
OFFICIAL_RESULT_SCHEMA = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
BINDING_TABLE = V21_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"

EXPECTED_ACTION_ID = "ace2:qk-gbfp8-base-v22:source-oracle-attribution:4901c835:additive-0001"
EXPECTED_PREDECESSOR_ACTION_ID = "ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001"
EXPECTED_CLAIM = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
EXPECTED_RESULT_SHA256 = "4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f"
EXPECTED_RESULT_BYTES = 8767
EXPECTED_HASHES = {
    "benchmark_package_sha256": "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832",
    "official_evaluator_sha256": "8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63",
    "official_result_schema_sha256": "07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720",
    "binding_table_sha256": "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655",
    "tensor_bundle_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
}
EXPECTED_CANDIDATES = ["G8", "G4", "G2", "G1"]
EXPECTED_HARD_GATES = {
    "cross_lane_record_count_maximum": 0,
    "invalid_or_non_finite_value_count_maximum": 0,
    "normalization_rejection_count_maximum": 0,
    "positive_centered_realized_score_count_maximum": 0,
    "rank_margin_violation_count_maximum": 0,
    "saturation_event_count_maximum": 0,
    "top_key_matching_fraction_minimum": {"numerator": 1, "denominator": 1},
    "top_key_mismatch_count_maximum": 0,
    "unique_oracle_positive_margin_preserved_fraction_minimum": {"numerator": 1, "denominator": 1},
}
EXPECTED_TENSORS = [
    {"dtype": "torch.bfloat16", "sha256": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a", "shape": [1, 2, 41, 64], "tensor_name": "bf16.k_rope"},
    {"dtype": "torch.bfloat16", "sha256": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429", "shape": [1, 14, 41, 64], "tensor_name": "bf16.q_rope"},
    {"dtype": "torch.bfloat16", "sha256": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765", "shape": [1, 14, 41, 41], "tensor_name": "bf16.qk_scaled_scores"},
]
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
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value, raw


def json_object_with_raw(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict, f"JSON object: {path}")
    return value, raw


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
        raise VerificationError("process start prohibited in inert V22 verifier")


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
    return {
        "entry_count": len(records),
        "file_count": sum(record["kind"] == "file" for record in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def load_regression_module() -> Any:
    spec = importlib.util.spec_from_file_location("v22_bound_public_regressions", REGRESSION_TOOL)
    require(spec is not None and spec.loader is not None, "regression import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v22_bound_public_regressions"] = module
    spec.loader.exec_module(module)
    return module


def validate_static_contract(package: dict[str, Any]) -> None:
    require(package["artifact_kind"] == "qk_gbfp8_head64_source_oracle_attribution_static_v22_package", "package kind")
    require(package["authoritative_stage"] == "Base", "authoritative Base stage")
    require(package["action_identity"] == {
        "action_id": EXPECTED_ACTION_ID,
        "additive_successor": True,
        "predecessor_action_id": EXPECTED_PREDECESSOR_ACTION_ID,
        "predecessor_consumed_terminal_immutable": True,
        "retry_replay_resume_repair_reinterpret_permitted": False,
    }, "V22 action identity")
    require(package["claim_boundary"] == {
        "claim": EXPECTED_CLAIM,
        "authority_or_credential_materialized": False,
        "base_selection_permitted": False,
        "evaluator_invocations": 0,
        "execution_authorized": False,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "owner_or_ledger_materialized": False,
        "result_or_terminal_materialized": False,
        "rtl_or_hardware_activity": False,
        "runtime_namespace_materialized": False,
        "v21_retried_replayed_resumed_repaired_reinterpreted_or_mutated": False,
        "v22_executed": False,
    }, "claim boundary")
    frozen = package["frozen_contract"]
    require(frozen["hashes"] == EXPECTED_HASHES, "frozen hashes")
    require(frozen["candidate_order"] == EXPECTED_CANDIDATES, "candidate order")
    require(frozen["hard_gates"] == EXPECTED_HARD_GATES, "hard gates")
    require(frozen["selection_policy"] == "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selection policy")
    require(frozen["score_error_affects_selection"] is False, "score tracking only")
    require(frozen["tensor_record_count"] == 25 and frozen["numerical_tensor_identities"] == EXPECTED_TENSORS, "tensor identities")
    b0 = package["b0_identity_control"]
    require(b0["control_id"] == "B0_SOURCE_QK_IDENTITY" and b0["nonselecting"] is True, "B0 nonselecting")
    require(b0["quantization_completion"] is False and b0["base_selection_permitted"] is False, "B0 cannot pass quantization")
    require(b0["materialized_by_this_package"] is False and b0["future_separate_authority_required"] is True, "B0 future-only")


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    validate_static_contract(package)
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    policy = package["static_file_policy"]
    allowed = set(policy["allowed_relative_files"])
    acceptance_relative = "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    require(policy["optional_before_acceptance"] == [acceptance_relative] and acceptance_relative in allowed, "acceptance inventory policy")
    observed = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"V22 symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), f"V22 non-regular file: {relative}")
        require(not relative.endswith((".pyc", ".pyo")) and "__pycache__" not in relative, f"generated Python artifact: {relative}")
        observed.add(relative)
    expected = set(allowed)
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    else:
        expected.remove(acceptance_relative)
        require(not os.path.lexists(ACCEPTANCE), "Engineer materialized Fresh-L2 acceptance")
    require(observed == expected, f"V22 static inventory expected={sorted(expected)} observed={sorted(observed)}")
    return {"acceptance_present": allow_acceptance, "file_count": len(observed)}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file() and path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"], f"generated file binding: {relative}")
    return {"generated_file_count": len(package["generated_files"])}


def verify_frozen_contract(package: dict[str, Any]) -> dict[str, Any]:
    diagnosis, diagnosis_raw = json_object_with_raw(DIAGNOSIS)
    result, result_raw = canonical(V21_RESULT)
    require(len(result_raw) == EXPECTED_RESULT_BYTES and sha256_bytes(result_raw) == EXPECTED_RESULT_SHA256, "immutable V21 result")
    require(result["irreversible_action_id"] == EXPECTED_PREDECESSOR_ACTION_ID, "V21 result action")
    require(result["terminal"]["status"] == "FAILED_TERMINAL" and result["terminal"]["retry_replay_resume_repair_permitted"] is False, "V21 terminal")
    successor = diagnosis["v22_static_successor"]
    require(successor["identity"]["action_id"] == EXPECTED_ACTION_ID, "diagnosis V22 identity")
    require(successor["frozen_bindings"]["candidate_order"] == EXPECTED_CANDIDATES, "diagnosis candidate order")
    require(successor["frozen_bindings"]["hard_gates"] == EXPECTED_HARD_GATES, "diagnosis hard gates")
    require(sha256_file(OFFICIAL_PACKAGE) == EXPECTED_HASHES["benchmark_package_sha256"], "benchmark package drift")
    require(sha256_file(OFFICIAL_EVALUATOR) == EXPECTED_HASHES["official_evaluator_sha256"], "official evaluator drift")
    require(sha256_file(OFFICIAL_RESULT_SCHEMA) == EXPECTED_HASHES["official_result_schema_sha256"], "result schema drift")
    require(sha256_file(BINDING_TABLE) == EXPECTED_HASHES["binding_table_sha256"], "V18 table drift")
    require(package["frozen_contract"]["diagnosis_file_sha256"] == sha256_bytes(diagnosis_raw), "diagnosis package binding")
    return {"diagnosis_file_sha256": sha256_bytes(diagnosis_raw), "v21_result_file_sha256": sha256_bytes(result_raw)}


def verify_regressions(package: dict[str, Any]) -> dict[str, Any]:
    bound, raw = canonical(REGRESSION_REPORT)
    verify_self_hash(bound, "report_sha256")
    module = load_regression_module()
    observed = module.run_regressions()
    require(observed == bound, "fresh public/synthetic regressions differ from bound report")
    require(bound["bf16_decode_and_g1"]["finite_word_count"] == 65280, "BF16 exhaustive count")
    require(bound["binding_resolution"]["permutation_count"] == 6, "six tensor permutations")
    require(bound["near_tie_argmax"]["g1_absolute_error_q12_20_lsb"] < bound["near_tie_argmax"]["g2_absolute_error_q12_20_lsb"], "near-tie aggregate ordering")
    require(bound["near_tie_argmax"]["g1_top_key"] != bound["near_tie_argmax"]["oracle_top_key"], "near-tie G1 argmax regression")
    require(bound["official_payload_open_count"] == bound["official_evaluator_invocation_count"] == 0, "regression inert boundary")
    require(package["arithmetic_regressions"]["report_file_sha256"] == sha256_bytes(raw), "regression report binding")
    return {"report_file_sha256": sha256_bytes(raw), "report_sha256": bound["report_sha256"]}


def verify_b0_schema(package: dict[str, Any]) -> dict[str, Any]:
    schema_raw = B0_SCHEMA.read_bytes()
    schema = json.loads(schema_raw.decode("ascii", "strict"))
    control = schema["properties"]["control"]["properties"]
    require(control["nonselecting"]["const"] is True and control["quantization_completion"]["const"] is False, "B0 schema nonselecting")
    require(control["base_selection_permitted"]["const"] is False and control["tensor_values_exposed"]["const"] is False, "B0 schema fail closed")
    require("selected_candidate" not in schema_raw.decode("ascii"), "B0 schema cannot select candidate")
    require(package["b0_identity_control"]["schema_file_sha256"] == sha256_bytes(schema_raw), "B0 schema binding")
    return {"schema_file_sha256": sha256_bytes(schema_raw), "row_class_count": 4}


def verify_preservation(package: dict[str, Any]) -> dict[str, Any]:
    v21, _ = canonical(V21_PACKAGE)
    verify_self_hash(v21, "package_content_sha256")
    declared = v21["preservation"]
    expected = package["predecessor_preservation"]
    require(expected["v21_declared_preservation_sha256"] == sha256_bytes(compact_bytes(declared)), "V21 declared preservation binding")
    for record in declared:
        require(inventory_digest(Path(record["root"])) == record, f"inherited predecessor drift: {record['root']}")
    require(inventory_digest(V21_ROOT) == expected["v21_action_root"], "V21 action root drift")
    require(inventory_digest(V21_RUNTIME) == expected["v21_runtime_root"], "V21 runtime drift")
    return {"inherited_root_count": len(declared), "v21_action_tree_sha256": expected["v21_action_root"]["tree_sha256"], "v21_runtime_tree_sha256": expected["v21_runtime_root"]["tree_sha256"]}


def verify_rejection_mutations(package: dict[str, Any]) -> dict[str, Any]:
    mutations = []
    for name, mutate in (
        ("changed_identity", lambda item: item["action_identity"].__setitem__("action_id", "changed")),
        ("threshold_loosening", lambda item: item["frozen_contract"]["hard_gates"].__setitem__("top_key_mismatch_count_maximum", 1)),
        ("candidate_reorder", lambda item: item["frozen_contract"].__setitem__("candidate_order", ["G1", "G2", "G4", "G8"])),
        ("payload_read_claim", lambda item: item["claim_boundary"].__setitem__("official_payload_open_count", 1)),
        ("execution_authority", lambda item: item["claim_boundary"].__setitem__("execution_authorized", True)),
        ("b0_as_quantization", lambda item: item["b0_identity_control"].__setitem__("quantization_completion", True)),
        ("tensor_identity_drift", lambda item: item["frozen_contract"]["numerical_tensor_identities"][0].__setitem__("sha256", "0" * 64)),
    ):
        changed = deepcopy(package)
        mutate(changed)
        try:
            validate_static_contract(changed)
        except VerificationError:
            mutations.append(name)
        else:
            raise VerificationError(f"fail-closed mutation accepted: {name}")
    return {"rejected_mutation_count": len(mutations), "rejected_mutations": mutations}


def verify_no_live_effects(package: dict[str, Any]) -> dict[str, Any]:
    for value in package["forbidden_live_paths"]:
        require(not os.path.lexists(value), f"forbidden V22 live path exists: {value}")
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "inert audit boundary")
    return {"forbidden_path_count": len(package["forbidden_live_paths"]), **AUDIT}


def verify(allow_acceptance: bool = False) -> dict[str, Any]:
    AUDIT.update({"official_payload_open_count": 0, "official_target_process_starts": 0})
    package, package_raw = verify_manifest()
    report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_attribution_static_v22_candidate_verifier_report",
        "b0_schema": verify_b0_schema(package),
        "claim_boundary": EXPECTED_CLAIM,
        "frozen_contract": verify_frozen_contract(package),
        "generated_files": verify_generated_files(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "live_effects": verify_no_live_effects(package),
        "package_file_sha256": sha256_bytes(package_raw),
        "package_content_sha256": package["package_content_sha256"],
        "predecessor_preservation": verify_preservation(package),
        "regressions": verify_regressions(package),
        "rejection_mutations": verify_rejection_mutations(package),
        "status": "PASS_V22_STATIC_CANDIDATE_PENDING_FRESH_L2",
        "v22_execution_authority_granted": False,
    }
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "final inert audit boundary")
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    report = verify(False)
    raw = compact_bytes(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
