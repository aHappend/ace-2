#!/usr/bin/env python3
"""Inert candidate verifier for the additive static-only V18 package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_STATIC_PACKAGE.json"
BINDING_TABLE = ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
CAUSE_REPORT = ACTION_ROOT / "evidence/V17_C02_BINDING_CAUSAL_CHAIN.json"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_C02_BINDING_FIXTURE_REPORT.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
REPAIR_MODULE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_c02_binding_repair_v18.py"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_c02_binding_fixture_v18.py"
POST_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_c02_binding_repair_post_acceptance_v18.py"
LANE_METADATA = PROJECT_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
ACCEPTED_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
V17_LAUNCHER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py"
PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
V17_TERMINAL = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31/primary/authority/base/first-terminal.json"
OFFICIAL_PAYLOADS = {
    (PROJECT_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin").resolve(),
    (PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin").resolve(),
}
EXPECTED_ACTION_ID = "ace2:qk-gbfp8-base-v18:static-c02-binding-repair:additive-0001"
EXPECTED_CLAIM_BOUNDARY = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
EXPECTED_V17_ACTION_ID = "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z"
EXPECTED_V17_TERMINAL_RAW_SHA256 = "659a876c65cee806d10735f6f3dcab099904bbaac2c1bbf26dd9779a5bcaf3c8"
EXPECTED_V17_TERMINAL_SELF_SHA256 = "d7ec39a07ba68390b0bd014f5363288424ab503293bfa526e339518f0fde6617"
EXPECTED_C02_ERROR_DIGEST = "9b58c4983e1043a8edec6799aa3d7b310050861661aa47175b250b1bdc358286"
EXPECTED_LANE_METADATA_SHA256 = "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a"
EXPECTED_BUNDLE_SHA256 = "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175"
EXPECTED_MODEL_SHA256 = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
AUDIT = {"official_payload_opens": 0, "official_target_starts": 0}


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
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
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
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path in OFFICIAL_PAYLOADS:
            AUDIT["official_payload_opens"] += 1
            raise VerificationError("official payload open prohibited")
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v17.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise VerificationError("official target start prohibited")


def inventory_digest(root: Path) -> dict[str, Any]:
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
    return {
        "entry_count": len(records),
        "file_count": sum(record["kind"] == "file" for record in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def load_repair_module() -> Any:
    spec = importlib.util.spec_from_file_location("v18_binding_repair_candidate", REPAIR_MODULE)
    require(spec is not None and spec.loader is not None, "repair import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixture_module() -> Any:
    sys.path.insert(0, str(FIXTURE.parent))
    try:
        spec = importlib.util.spec_from_file_location("v18_binding_fixture_candidate", FIXTURE)
        require(spec is not None and spec.loader is not None, "fixture import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def verify_manifest() -> tuple[dict[str, Any], bytes]:
    package, raw = canonical(MANIFEST)
    verify_self_hash(package, "package_content_sha256")
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_package", "package kind")
    require(package["root_id"] == ROOT_ID, "root id")
    require(package["action_identity"] == {
        "action_id": EXPECTED_ACTION_ID,
        "additive_successor": True,
        "predecessor_action_id": EXPECTED_V17_ACTION_ID,
        "predecessor_disposition": "CONSUMED_ORPHAN",
        "retry_replay_resume_repair_replacement_permitted": False,
        "v17_modified": False,
    }, "action identity")
    require(package["claim_boundary"] == {
        "claim": EXPECTED_CLAIM_BOUNDARY,
        "authority_materialized": False,
        "checkpoint_176_activity": False,
        "credential_materialized": False,
        "evaluator_invocations": 0,
        "execution_authorized": False,
        "hardware_or_rtl_activity": False,
        "ledger_materialized": False,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "result_materialized": False,
        "runtime_terminal_materialized": False,
        "software_fallback": False,
        "stage_transition": False,
        "v17_retried": False,
        "v18_executed": False,
    }, "claim boundary")
    return package, raw


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "action root absent")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "action root mode")
    policy = package["static_file_policy"]
    acceptance_relative = "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    require(policy["acceptance_relative_path"] == acceptance_relative, "acceptance policy path")
    allowed = set(policy["allowed_relative_files"])
    require(acceptance_relative in allowed and policy["optional_before_acceptance"] == [acceptance_relative], "acceptance policy inventory")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(ACTION_ROOT.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"action root symlink: {path}")
        relative = path.relative_to(ACTION_ROOT).as_posix()
        if stat.S_ISDIR(info.st_mode):
            observed_directories.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o555, f"directory mode: {relative}")
        else:
            require(stat.S_ISREG(info.st_mode), f"non-regular action root entry: {relative}")
            observed_files.add(relative)
            require(stat.S_IMODE(info.st_mode) == 0o444, f"file mode: {relative}")
            require(not relative.endswith((".pyc", ".pyo")), f"generated Python artifact: {relative}")
    expected = set(allowed)
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "Fresh-L2 acceptance absent")
    else:
        expected.remove(acceptance_relative)
        require(not os.path.lexists(ACCEPTANCE), "Engineer materialized Fresh-L2 acceptance")
        require("review" not in observed_directories, "pre-acceptance review directory exists")
    require(observed_files == expected, f"action root file inventory: expected={sorted(expected)} observed={sorted(observed_files)}")
    require(not any(directory == "live" or directory.startswith("live/") for directory in observed_directories), "live directory exists")
    return {"acceptance_present": allow_acceptance, "directory_count": 1 + len(observed_directories), "file_count": len(observed_files)}


def verify_bindings(package: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(LANE_METADATA) == EXPECTED_LANE_METADATA_SHA256, "lane metadata drift")
    lane = json_object(LANE_METADATA)
    accepted, accepted_raw = canonical(ACCEPTED_PACKAGE)
    repair = load_repair_module()
    expected = repair.exact_binding_table(lane)
    table, raw = canonical(BINDING_TABLE)
    verify_self_hash(table, "binding_table_sha256")
    require(table["records"] == expected and table["record_count"] == 25, "complete binding table")
    require(table["selected_tensor_names"] == list(repair.EXPECTED_SELECTED_NAMES), "binding table selection")
    require(table["source_metadata"] == {
        "byte_count": 20057,
        "path": str(LANE_METADATA),
        "sha256": EXPECTED_LANE_METADATA_SHA256,
    }, "binding metadata source")
    require(table["tensor_bundle"] == {
        "byte_count": 1305797,
        "record_count": 25,
        "sha256": EXPECTED_BUNDLE_SHA256,
    }, "binding tensor bundle identity")
    context = repair.prepare_static_context(accepted, lane)
    require(context["tensor_bindings"] == expected and len(context["numerical_tensor_names"]) == 3, "V18 prepared context")
    binding_manifest = package["complete_binding_table"]
    require(binding_manifest["file_sha256"] == sha256_bytes(raw), "manifest binding table hash")
    require(binding_manifest["binding_table_sha256"] == table["binding_table_sha256"], "manifest binding table self hash")
    require(package["frozen_official_identity"] == {
        "accepted_static_package_file_sha256": sha256_bytes(accepted_raw),
        "input_token_ids_sha256": accepted["official_benchmark"]["input_bindings"]["input_token_ids_sha256"],
        "lane_metadata_sha256": EXPECTED_LANE_METADATA_SHA256,
        "model_identity_sha256": EXPECTED_MODEL_SHA256,
        "static_package_id": accepted["package_id"],
        "tensor_bundle_sha256": EXPECTED_BUNDLE_SHA256,
    }, "frozen official identity")
    return {"binding_table_file_sha256": sha256_bytes(raw), "binding_table_sha256": table["binding_table_sha256"], "record_count": len(expected), "selected_count": len(context["numerical_tensor_names"])}


def verify_causal_chain(package: dict[str, Any]) -> dict[str, Any]:
    report, raw = canonical(CAUSE_REPORT)
    verify_self_hash(report, "report_sha256")
    require(package["causal_chain"]["file_sha256"] == sha256_bytes(raw), "causal report file binding")
    require(report["conclusion"] == {
        "failure_before_numerical_evaluation": True,
        "failure_message": "producer c02 binding exact keys",
        "failure_stage": "C02_BINDING_VALIDATION",
        "first_unselected_record_binding_is_none": True,
    }, "causal conclusion")
    terminal, terminal_raw = canonical(V17_TERMINAL)
    verify_self_hash(terminal, "first_terminal_sha256")
    require(sha256_bytes(terminal_raw) == EXPECTED_V17_TERMINAL_RAW_SHA256, "V17 terminal raw hash")
    require(terminal["first_terminal_sha256"] == EXPECTED_V17_TERMINAL_SELF_SHA256, "V17 terminal self hash")
    require(terminal["status"] == "CONSUMED_ORPHAN" and terminal["action_retired"] is True, "V17 retired terminal")
    require(terminal["payload_open_count"] == 1 and terminal["invocation_count_performed"] == 1 and terminal["official_target_process_starts"] == 1, "V17 irreversible counters")
    require(terminal["evaluator_diagnostic"]["process"] == {"return_code": 70, "started": True}, "V17 evaluator process")
    require(terminal["evaluator_diagnostic"]["stderr"]["preview_ascii"] == f"C02Error:{EXPECTED_C02_ERROR_DIGEST}\\x0a", "V17 stderr digest")

    launcher = V17_LAUNCHER.read_text(encoding="utf-8")
    parser = PARSER.read_text(encoding="utf-8")
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    require('records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]' in launcher, "V17 selected record source")
    require('for record in records.values()' in launcher and 'holder["tensor_bindings"]' in launcher, "V17 three-binding construction")
    require('binding = bindings.get(name)' in parser and 'set(binding) != {"dtype", "sha256", "shape"}' in parser, "parser exact binding requirement")
    require('raise C02Error("producer c02 binding exact keys")' in parser, "parser frozen failure message")
    require('candidate_results = evaluate_all_candidates(producer_records)' in evaluator, "numerical evaluation ordering")
    require('selected = _validated_selected_records(records)' in evaluator, "numerical selected records")
    require('query = selected[QUERY_RECORD_NAME]' in evaluator and 'key = selected[KEY_RECORD_NAME]' in evaluator and 'oracle = selected[ORACLE_RECORD_NAME]' in evaluator, "three numerical tensors")
    return {"failure_stage": report["conclusion"]["failure_stage"], "report_sha256": report["report_sha256"], "v17_terminal_raw_sha256": sha256_bytes(terminal_raw)}


def verify_fixture(package: dict[str, Any]) -> dict[str, Any]:
    module = load_fixture_module()
    observed = module.run_fixture()
    bound, raw = canonical(FIXTURE_REPORT)
    require(observed == bound, "fresh fixture differs from bound report")
    verify_self_hash(bound, "report_sha256")
    require(bound["status"] == "PASS_V18_C02_BINDING_REPAIR_SYNTHETIC_FIXTURE", "fixture status")
    require(bound["complete_binding_count"] == 25 and bound["cross_parser_record_count"] == 25, "fixture record closure")
    require(len(bound["cases"]) == 6 and all(case["passed"] for case in bound["cases"]), "fixture fail-closed cases")
    require(bound["numerical_tensor_names"] == ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"], "fixture numerical selection")
    require(bound["diagnostic_contract"] == {"exception_type": "C02Error", "failure_stage": "C02_BINDING_VALIDATION"}, "fixture diagnostic contract")
    require(bound["official_payload_open_count"] == 0 and bound["official_target_process_starts"] == 0, "fixture live effects")
    require(package["synthetic_fixture"]["report_file_sha256"] == sha256_bytes(raw), "fixture manifest file hash")
    require(package["synthetic_fixture"]["report_sha256"] == bound["report_sha256"], "fixture manifest self hash")
    return {"case_count": len(bound["cases"]), "report_file_sha256": sha256_bytes(raw), "report_sha256": bound["report_sha256"]}


def verify_generated_files(package: dict[str, Any]) -> dict[str, Any]:
    for relative, record in package["generated_files"].items():
        path = ACTION_ROOT / relative
        require(path.is_file() and sha256_file(path) == record["sha256"] and path.stat().st_size == record["size"], f"generated file binding: {relative}")
    require(package["post_acceptance_verifier"] == {
        "acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
        "inventory_policy": "EXACT_STATIC_FILES_PLUS_ONE_BOUND_FRESH_L2_ACCEPTANCE",
        "path": str(POST_VERIFIER),
        "sha256": sha256_file(POST_VERIFIER),
    }, "post-acceptance verifier binding")
    return {"generated_file_count": len(package["generated_files"]), "post_acceptance_verifier_sha256": sha256_file(POST_VERIFIER)}


def verify_preservation(package: dict[str, Any]) -> list[dict[str, Any]]:
    observed = []
    for expected in package["preservation"]:
        current = inventory_digest(Path(expected["root"]))
        require(current == expected, f"predecessor drift: {expected['root']}")
        observed.append(current)
    require(len(observed) >= 10, "V13-V17 preservation coverage")
    return observed


def verify_no_live_effects(package: dict[str, Any]) -> dict[str, Any]:
    for value in package["forbidden_live_paths"]:
        require(not os.path.lexists(value), f"forbidden V18 live path exists: {value}")
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "inert audit boundary")
    return {"forbidden_path_count": len(package["forbidden_live_paths"]), "official_payload_open_count": 0, "official_target_process_starts": 0}


def verify(allow_acceptance: bool = False) -> dict[str, Any]:
    AUDIT.update({"official_payload_opens": 0, "official_target_starts": 0})
    package, package_raw = verify_manifest()
    report = {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_candidate_verifier_report",
        "bindings": verify_bindings(package),
        "causal_chain": verify_causal_chain(package),
        "fixture": verify_fixture(package),
        "generated_files": verify_generated_files(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "live_effects": verify_no_live_effects(package),
        "package_file_sha256": sha256_bytes(package_raw),
        "package_content_sha256": package["package_content_sha256"],
        "preservation": verify_preservation(package),
        "status": "PASS_V18_C02_BINDING_REPAIR_STATIC_CANDIDATE",
    }
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
