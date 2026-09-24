#!/usr/bin/env python3
"""Build the additive static-only V18 C02 binding repair package.

The builder reads only public metadata and frozen text/JSON predecessors.  It
does not open the protected tensor payload, invoke the evaluator, create a
runtime namespace, or materialize Fresh-L2 acceptance.
"""

from __future__ import annotations

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
BUILD_ROOT = PROJECT_ROOT / "build/v18-c02-binding-repair-static-0001"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_STATIC_PACKAGE.json"
BINDING_TABLE = ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
CAUSE_REPORT = ACTION_ROOT / "evidence/V17_C02_BINDING_CAUSAL_CHAIN.json"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_C02_BINDING_FIXTURE_REPORT.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA.json"
ACCEPTANCE = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
REPAIR_MODULE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_c02_binding_repair_v18.py"
FIXTURE = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_c02_binding_fixture_v18.py"
CANDIDATE_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_c02_binding_repair_candidate_v18.py"
POST_VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_c02_binding_repair_post_acceptance_v18.py"
LANE_METADATA = PROJECT_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
ACCEPTED_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
V17_LAUNCHER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py"
PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
V17_TERMINAL = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31/primary/authority/base/first-terminal.json"
ACTION_ID = "ace2:qk-gbfp8-base-v18:static-c02-binding-repair:additive-0001"
V17_ACTION_ID = "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z"
CLAIM_BOUNDARY = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
V17_TERMINAL_RAW_SHA256 = "659a876c65cee806d10735f6f3dcab099904bbaac2c1bbf26dd9779a5bcaf3c8"
V17_TERMINAL_SELF_SHA256 = "d7ec39a07ba68390b0bd014f5363288424ab503293bfa526e339518f0fde6617"
C02_ERROR_DIGEST = "9b58c4983e1043a8edec6799aa3d7b310050861661aa47175b250b1bdc358286"
LANE_METADATA_SHA256 = "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a"
BUNDLE_SHA256 = "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175"
MODEL_SHA256 = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
OFFICIAL_PAYLOADS = {
    (PROJECT_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin").resolve(),
    (PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin").resolve(),
}
PRESERVATION_ROOTS = (
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v13_shellfree_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v14_shellfree_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_shellfree_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_action_root",
    PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root",
    PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v15_c2dfe170",
    PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_e4c05735",
    PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31",
)
AUDIT = {"official_payload_opens": 0, "official_target_starts": 0}


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


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


def write_canonical(path: Path, value: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = compact_bytes(value)
    path.write_bytes(raw)
    return raw


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict, f"JSON object: {path}")
    return value, raw


def self_seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed.pop(field, None)
    sealed[field] = sha256_bytes(compact_bytes(sealed))
    return sealed


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path in OFFICIAL_PAYLOADS:
            AUDIT["official_payload_opens"] += 1
            raise BuildError("official payload open prohibited")
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v17.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise BuildError("official target start prohibited")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module import spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            raise BuildError(f"unsupported preservation entry: {path}")
    return {
        "entry_count": len(records),
        "file_count": sum(record["kind"] == "file" for record in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def build_binding_table(repair: Any, lane: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    records = repair.exact_binding_table(lane)
    table = self_seal({
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_exact_binding_table",
        "record_count": len(records),
        "records": records,
        "selected_tensor_names": list(repair.EXPECTED_SELECTED_NAMES),
        "source_metadata": {"byte_count": LANE_METADATA.stat().st_size, "path": str(LANE_METADATA), "sha256": sha256_file(LANE_METADATA)},
        "tensor_bundle": {"byte_count": lane["tensor_bundle"]["bytes"], "record_count": lane["tensor_bundle"]["tensor_count"], "sha256": lane["tensor_bundle"]["sha256"]},
    }, "binding_table_sha256")
    raw = write_canonical(BINDING_TABLE, table)
    return table, raw


def build_causal_report(lane: dict[str, Any], accepted: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    terminal, terminal_raw = canonical(V17_TERMINAL)
    require(sha256_bytes(terminal_raw) == V17_TERMINAL_RAW_SHA256, "V17 terminal raw hash")
    require(terminal["first_terminal_sha256"] == V17_TERMINAL_SELF_SHA256, "V17 terminal self hash")
    require(terminal["evaluator_diagnostic"]["stderr"]["preview_ascii"] == f"C02Error:{C02_ERROR_DIGEST}\\x0a", "V17 C02 digest")
    selected_records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]
    selected_names = sorted(record["tensor_name"] for record in selected_records.values())
    all_names = sorted(lane["tensor_bundle"]["tensors"])
    unselected = [name for name in all_names if name not in selected_names]
    require(len(all_names) == 25 and len(selected_names) == 3 and len(unselected) == 22, "causal record cardinality")
    report = self_seal({
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_v17_c02_binding_causal_chain",
        "conclusion": {
            "failure_before_numerical_evaluation": True,
            "failure_message": "producer c02 binding exact keys",
            "failure_stage": "C02_BINDING_VALIDATION",
            "first_unselected_record_binding_is_none": True,
        },
        "frozen_parser": {
            "failure_message_sha256": sha256_bytes(b"producer c02 binding exact keys"),
            "path": str(PARSER),
            "sha256": sha256_file(PARSER),
        },
        "frozen_evaluator": {
            "numerical_entrypoint": "evaluate_all_candidates",
            "path": str(EVALUATOR),
            "selected_tensor_names": selected_names,
            "sha256": sha256_file(EVALUATOR),
        },
        "lane_metadata": {
            "path": str(LANE_METADATA),
            "record_count": len(all_names),
            "sha256": sha256_file(LANE_METADATA),
            "tensor_bundle_sha256": lane["tensor_bundle"]["sha256"],
        },
        "v17_prepare_execution": {
            "binding_count": len(selected_names),
            "binding_source": "official_benchmark.input_bindings.tensor_records",
            "first_unselected_record": unselected[0],
            "path": str(V17_LAUNCHER),
            "sha256": sha256_file(V17_LAUNCHER),
        },
        "v17_terminal": {
            "action_id": terminal["action_id"],
            "first_terminal_raw_sha256": sha256_bytes(terminal_raw),
            "first_terminal_self_sha256": terminal["first_terminal_sha256"],
            "invocation_count_performed": terminal["invocation_count_performed"],
            "official_target_process_starts": terminal["official_target_process_starts"],
            "payload_open_count": terminal["payload_open_count"],
            "return_code": terminal["evaluator_diagnostic"]["process"]["return_code"],
            "status": terminal["status"],
            "stderr_c02_error_digest": C02_ERROR_DIGEST,
        },
    }, "report_sha256")
    raw = write_canonical(CAUSE_REPORT, report)
    return report, raw


def build_acceptance_schema() -> bytes:
    sha = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    schema = {
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA_V1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "acceptance_sha256": sha,
            "accepted": {"const": True},
            "action_id": {"const": ACTION_ID},
            "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_fresh_l2_static_acceptance"},
            "binding_table_file_sha256": sha,
            "candidate_report_sha256": sha,
            "claim_boundary": {"const": CLAIM_BOUNDARY},
            "decision": {"const": "ACCEPT_STATIC_PACKAGE"},
            "fixture_report_file_sha256": sha,
            "manifest_file_sha256": sha,
            "official_payload_open_count": {"const": 0},
            "official_target_process_starts": {"const": 0},
            "reviewer_role": {"const": "Fresh-L2"},
            "static_acceptance_grants_execution_authority": {"const": False},
            "v17_retried": {"const": False},
            "v18_executed": {"const": False},
        },
        "required": [
            "acceptance_sha256", "accepted", "action_id", "artifact_kind",
            "binding_table_file_sha256", "candidate_report_sha256", "claim_boundary",
            "decision", "fixture_report_file_sha256", "manifest_file_sha256",
            "official_payload_open_count", "official_target_process_starts", "reviewer_role",
            "static_acceptance_grants_execution_authority", "v17_retried", "v18_executed",
        ],
        "type": "object",
    }
    return write_canonical(ACCEPTANCE_SCHEMA, schema)


def set_static_modes() -> None:
    for path in sorted(ACTION_ROOT.rglob("*"), reverse=True):
        if path.is_dir():
            os.chmod(path, 0o555)
        else:
            os.chmod(path, 0o444)
    os.chmod(ACTION_ROOT, 0o555)


def build() -> dict[str, Any]:
    require(not os.path.lexists(ACCEPTANCE), "Fresh-L2 acceptance already exists")
    for directory in (ACTION_ROOT, ACTION_ROOT / "tools", ACTION_ROOT / "bindings", ACTION_ROOT / "evidence", ACTION_ROOT / "reference", BUILD_ROOT):
        if directory.exists():
            os.chmod(directory, 0o755)
        else:
            directory.mkdir(parents=True, exist_ok=True)
    for source in (REPAIR_MODULE, FIXTURE, CANDIDATE_VERIFIER, POST_VERIFIER):
        require(source.is_file(), f"V18 source absent: {source}")

    require(sha256_file(LANE_METADATA) == LANE_METADATA_SHA256, "lane metadata hash")
    lane, _ = canonical(LANE_METADATA)
    accepted, accepted_raw = canonical(ACCEPTED_PACKAGE)
    require(lane["tensor_bundle"]["sha256"] == BUNDLE_SHA256, "tensor bundle metadata hash")
    require(accepted["official_benchmark"]["model_identity_sha256"] == MODEL_SHA256, "model identity")

    repair = load_module("v18_binding_repair_builder", REPAIR_MODULE)
    sys.path.insert(0, str(FIXTURE.parent))
    try:
        fixture = load_module("v18_binding_fixture_builder", FIXTURE)
    finally:
        sys.path.pop(0)

    binding_table, binding_raw = build_binding_table(repair, lane)
    causal_report, causal_raw = build_causal_report(lane, accepted)
    fixture_report = fixture.run_fixture()
    fixture_raw = write_canonical(FIXTURE_REPORT, fixture_report)
    acceptance_schema_raw = build_acceptance_schema()
    preservation = [inventory_digest(root) for root in PRESERVATION_ROOTS]

    generated_paths = (
        BINDING_TABLE,
        CAUSE_REPORT,
        FIXTURE_REPORT,
        ACCEPTANCE_SCHEMA,
        REPAIR_MODULE,
        FIXTURE,
        CANDIDATE_VERIFIER,
        POST_VERIFIER,
    )
    generated_files = {path.relative_to(ACTION_ROOT).as_posix(): file_record(path) for path in generated_paths}
    manifest_relative = MANIFEST.relative_to(ACTION_ROOT).as_posix()
    acceptance_relative = ACCEPTANCE.relative_to(ACTION_ROOT).as_posix()
    allowed_files = sorted(set(generated_files) | {manifest_relative, acceptance_relative})
    forbidden_live_paths = [
        str(ACTION_ROOT / "live"),
        str(PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static"),
        str(BUILD_ROOT / "authority"),
        str(BUILD_ROOT / "credential.json"),
        str(BUILD_ROOT / "authority-ledger.json"),
        str(BUILD_ROOT / "first-terminal.json"),
        str(BUILD_ROOT / "result.json"),
    ]
    package = self_seal({
        "action_identity": {
            "action_id": ACTION_ID,
            "additive_successor": True,
            "predecessor_action_id": V17_ACTION_ID,
            "predecessor_disposition": "CONSUMED_ORPHAN",
            "retry_replay_resume_repair_replacement_permitted": False,
            "v17_modified": False,
        },
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_package",
        "causal_chain": {"file_sha256": sha256_bytes(causal_raw), "path": str(CAUSE_REPORT), "report_sha256": causal_report["report_sha256"]},
        "claim_boundary": {
            "claim": CLAIM_BOUNDARY,
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
        },
        "complete_binding_table": {
            "binding_table_sha256": binding_table["binding_table_sha256"],
            "file_sha256": sha256_bytes(binding_raw),
            "path": str(BINDING_TABLE),
            "record_count": 25,
            "source_metadata_sha256": LANE_METADATA_SHA256,
        },
        "diagnostic_contract": {
            "exact_keys": ["exception_type", "failure_stage"],
            "message_text_retained": False,
            "stable_failure_stage": "C02_BINDING_VALIDATION",
        },
        "forbidden_live_paths": forbidden_live_paths,
        "fresh_l2_review": {
            "acceptance_path": str(ACCEPTANCE),
            "acceptance_present": False,
            "acceptance_schema_file_sha256": sha256_bytes(acceptance_schema_raw),
            "acceptance_schema_path": str(ACCEPTANCE_SCHEMA),
            "grants_execution_authority": False,
            "reviewer_role": "Fresh-L2",
            "status": "PENDING_INDEPENDENT_REVIEW",
        },
        "frozen_official_identity": {
            "accepted_static_package_file_sha256": sha256_bytes(accepted_raw),
            "input_token_ids_sha256": accepted["official_benchmark"]["input_bindings"]["input_token_ids_sha256"],
            "lane_metadata_sha256": LANE_METADATA_SHA256,
            "model_identity_sha256": MODEL_SHA256,
            "static_package_id": accepted["package_id"],
            "tensor_bundle_sha256": BUNDLE_SHA256,
        },
        "generated_files": generated_files,
        "numerical_selection": {
            "complete_parser_binding_count": 25,
            "numerical_tensor_count": 3,
            "tensor_names": list(repair.EXPECTED_SELECTED_NAMES),
        },
        "post_acceptance_verifier": {
            "acceptance_relative_path": acceptance_relative,
            "inventory_policy": "EXACT_STATIC_FILES_PLUS_ONE_BOUND_FRESH_L2_ACCEPTANCE",
            "path": str(POST_VERIFIER),
            "sha256": sha256_file(POST_VERIFIER),
        },
        "preservation": preservation,
        "required_disposition": "V18_C02_BINDING_REPAIR_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY",
        "root_id": ROOT_ID,
        "schema_version": 1,
        "static_file_policy": {
            "acceptance_relative_path": acceptance_relative,
            "allowed_relative_files": allowed_files,
            "optional_before_acceptance": [acceptance_relative],
        },
        "synthetic_fixture": {
            "case_count": len(fixture_report["cases"]),
            "official_payload_open_count": 0,
            "official_target_process_starts": 0,
            "report_file_sha256": sha256_bytes(fixture_raw),
            "report_path": str(FIXTURE_REPORT),
            "report_sha256": fixture_report["report_sha256"],
            "synthetic_bytes_only": True,
        },
    }, "package_content_sha256")
    manifest_raw = write_canonical(MANIFEST, package)
    set_static_modes()

    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "builder inert audit boundary")
    build_report = self_seal({
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_build_report",
        "binding_record_count": 25,
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture_case_count": len(fixture_report["cases"]),
        "manifest_file_sha256": sha256_bytes(manifest_raw),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "preserved_root_count": len(preservation),
        "status": "PASS_V18_C02_BINDING_REPAIR_STATIC_BUILD",
        "v17_retried": False,
        "v18_executed": False,
    }, "report_sha256")
    build_raw = write_canonical(BUILD_ROOT / "build-report.json", build_report)
    review_request = self_seal({
        "acceptance_path": str(ACCEPTANCE),
        "acceptance_schema_path": str(ACCEPTANCE_SCHEMA),
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_fresh_l2_review_request",
        "candidate_verifier_path": str(CANDIDATE_VERIFIER),
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest_file_sha256": sha256_bytes(manifest_raw),
        "manifest_path": str(MANIFEST),
        "post_acceptance_verifier_path": str(POST_VERIFIER),
        "reviewer_role": "Fresh-L2",
        "status": "PENDING_INDEPENDENT_REVIEW",
    }, "request_sha256")
    write_canonical(BUILD_ROOT / "fresh-l2-review-request.json", review_request)
    os.chmod(BUILD_ROOT / "build-report.json", 0o444)
    os.chmod(BUILD_ROOT / "fresh-l2-review-request.json", 0o444)
    return {
        "build_report_file_sha256": sha256_bytes(build_raw),
        "manifest_file_sha256": sha256_bytes(manifest_raw),
        "status": build_report["status"],
    }


def main() -> int:
    sys.addaudithook(audit_hook)
    print(compact_bytes(build()).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
