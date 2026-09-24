#!/usr/bin/env python3
"""Issue one create-only Fresh-L2 static acceptance for the inert V20 package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v20:execute-once:a81f2916:additive-0001"
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root"
BUILD_ROOT = PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_a81f2916"
PACKAGE = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA.json"
ACCEPTANCE_DIR = ACTION_ROOT / "review"
ACCEPTANCE = ACCEPTANCE_DIR / "FRESH_L2_STATIC_ACCEPTANCE.json"
FIXTURE_REPORT = ACTION_ROOT / "evidence/SYNTHETIC_BOUND_CORE_MANIFEST_PREFLIGHT_FIXTURE_REPORT.json"
PREFLIGHT = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v20.py"
VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py"
CANDIDATE_REPORT = BUILD_ROOT / "candidate-inert-verifier-report.json"
REVIEW_REQUEST = BUILD_ROOT / "fresh-l2-review-request.json"
REVIEW_REPORT = BUILD_ROOT / "fresh-l2-independent-review.json"
POST_REPORT = BUILD_ROOT / "fresh-l2-post-acceptance-inert-report.json"
V19_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_action_root"
V19_WRAPPER = V19_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py"
V19_VERIFIER = V19_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_exactly_once_v18_binding_v19.py"
V19_TERMINAL = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1/primary/authority/base/first-terminal.json"
V19_TERMINAL_SHA256 = "a81f2916fe3e5f2aa0489caee7d7a52ada1a8760b32ea7faf3a94bad6f4cb862"
V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
CORE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
CORE_MANIFEST = CORE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
V19_WRONG_CORE_MANIFEST = CORE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "TZ": "UTC"}
AUDIT = {"official_payload_opens": 0, "official_target_starts": 0}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    require(path != OFFICIAL_PAYLOAD, "official payload hashing prohibited")
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
    require(path != OFFICIAL_PAYLOAD, "official payload read prohibited")
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash: {field}")


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def write_once(path: Path, raw: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, f"short write: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_opens"] += 1
            raise ReviewError("official payload open prohibited")
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v20.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise ReviewError("official target start prohibited")


def tree_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"root absent: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise ReviewError(f"unsupported entry: {path}")
    return {
        "entry_count": len(records),
        "file_count": sum(item["kind"] == "file" for item in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def verify_invocations(package: dict[str, Any]) -> None:
    acceptance = str(ACCEPTANCE)
    package_path = str(PACKAGE)
    expected = {
        "launcher_invocation": [str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py"), "--package", package_path, "--acceptance", acceptance, "--irreversible-action-id", ACTION_ID],
        "transport_invocation": [str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py"), "--package", package_path, "--acceptance", acceptance, "--irreversible-action-id", ACTION_ID],
        "production_evaluator_invocation": [str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v20.py"), "--mode", "production", "--package", package_path, "--acceptance", acceptance, "--irreversible-action-id", ACTION_ID],
    }
    for key, argv in expected.items():
        record = package[key]
        require(record["argv"] == argv, f"exact argv: {key}")
        require(record["cwd"] == str(ACTION_ROOT), f"exact cwd: {key}")
        require(record["environment"] == ENVIRONMENT, f"exact environment: {key}")
        require(record["shell"] is False, f"shell disabled: {key}")
        candidate = dict(record)
        observed = candidate.pop("invocation_sha256")
        require(observed == sha256_bytes(compact_bytes(candidate)), f"invocation self hash: {key}")


def add_path(paths: set[str], path: Path, root: Path | None = None) -> None:
    path = path.resolve()
    paths.add(str(path))
    if root is None:
        return
    root = root.resolve()
    current = path.parent
    while current == root or root in current.parents:
        paths.add(str(current))
        if current == root:
            break
        current = current.parent


def expected_closure_paths() -> set[str]:
    core, _ = canonical(CORE_MANIFEST)
    accepted_path = CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
    accepted, _ = canonical(accepted_path)
    paths: set[str] = set()
    add_path(paths, CORE_ROOT)
    add_path(paths, CORE_MANIFEST)
    for relative in core["generated_files"]:
        add_path(paths, CORE_ROOT / relative, CORE_ROOT)
    for preserved in core["preservation"].values():
        root = Path(preserved["root"])
        add_path(paths, root)
        for relative in preserved.get("files", {}):
            add_path(paths, root / relative, root)

    def absolute_existing(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                absolute_existing(item)
        elif isinstance(value, list):
            for item in value:
                absolute_existing(item)
        elif isinstance(value, str) and value.startswith("/") and os.path.lexists(value):
            add_path(paths, Path(value))

    absolute_existing(core)
    for path in (
        PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/FRESH_L2_STATIC_ACCEPTANCE.json",
        PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/independent-inert-verifier-report.json",
        PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_STATIC_PACKAGE.json",
        PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root/bindings/C02_EXACT_BINDINGS_25.json",
        PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json",
        PROJECT_ROOT / "build/v18-c02-binding-repair-static-0001/fresh-l2-post-acceptance-inert-report.json",
        INTERPRETER,
        accepted_path,
    ):
        add_path(paths, path)
    static_root = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root"
    for label in ("c02_parser", "controller", "evaluator", "result_schema", "verifier"):
        add_path(paths, static_root / accepted["static_bindings"][label]["path"], static_root)
    g16_root = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
    add_path(paths, g16_root / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json")
    add_path(paths, OFFICIAL_PAYLOAD, g16_root)
    return paths


def verify_closure(package: dict[str, Any]) -> dict[str, Any]:
    closure = package["core_manifest_closure"]
    entries = closure["entries"]
    require(entries == sorted(entries, key=lambda item: item["path"]), "closure order")
    require(sha256_bytes(compact_bytes(entries)) == closure["entries_sha256"], "closure digest")
    observed_paths = {entry["path"] for entry in entries}
    expected_paths = expected_closure_paths()
    require(observed_paths == expected_paths, "independent transitive closure path set")
    full_hash_files = directory_count = deferred_count = 0
    for entry in entries:
        path = Path(entry["path"])
        require(os.path.lexists(path), f"closure path absent: {path}")
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"closure symlink: {path}")
        require(f"{stat.S_IMODE(info.st_mode):04o}" == entry["mode"], f"closure mode: {path}")
        if entry["kind"] == "directory":
            directory_count += 1
            require(stat.S_ISDIR(info.st_mode), f"closure directory: {path}")
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_size == entry["size"], f"closure file metadata: {path}")
        if entry["hash_policy"] == "FULL_READ_ONLY_PREFLIGHT":
            full_hash_files += 1
            require(sha256_file(path) == entry["sha256"], f"closure file hash: {path}")
        else:
            deferred_count += 1
            require(path == OFFICIAL_PAYLOAD and entry["hash_policy"] == "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN", "sole deferred payload")
    require(len(entries) == 132 and directory_count == 46 and full_hash_files == 85 and deferred_count == 1, "closure cardinalities")
    require(closure["core_manifest_path"] == str(CORE_MANIFEST), "core manifest exact path")
    require(closure["core_manifest_sha256"] == sha256_file(CORE_MANIFEST), "core manifest exact hash")
    require(closure["core_manifest_mode"] == "0444", "core manifest exact mode")
    return {"directory_count": directory_count, "entry_count": len(entries), "entries_sha256": closure["entries_sha256"], "full_hash_file_count": full_hash_files, "deferred_hash_file_count": deferred_count}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_real_preflight() -> dict[str, Any]:
    module = load_module("fresh_l2_v20_preflight_probe", PREFLIGHT)

    def hash_bytes(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    with tempfile.TemporaryDirectory(prefix="fresh-l2-v20-preflight-") as temporary:
        root = Path(temporary)
        manifest = root / "CORE_MANIFEST.json"
        artifact = root / "artifact.bin"
        payload = root / "payload.bin"
        manifest_value = {"artifact_kind": "fresh_l2_synthetic", "generated_files": {"artifact.bin": {"sha256": hash_bytes(b"ABCDEFGH"), "size": 8}}, "package_content_sha256": "1" * 64}
        manifest.write_bytes(compact_bytes(manifest_value))
        artifact.write_bytes(b"ABCDEFGH")
        payload.write_bytes(b"SYNTHETIC")
        for path in (manifest, artifact, payload):
            os.chmod(path, 0o444)

        def record(path: Path, *, canonical_json: bool = False, deferred: bool = False) -> dict[str, Any]:
            raw = path.read_bytes()
            return {
                "canonical_json": canonical_json,
                "hash_policy": "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN" if deferred else "FULL_READ_ONLY_PREFLIGHT",
                "kind": "file",
                "mode": "0444",
                "path": str(path),
                "semantic_sha256": hash_bytes(compact_bytes(json.loads(raw))) if canonical_json else None,
                "sha256": hash_bytes(raw),
                "size": len(raw),
                "sources": ["fresh-l2-independent-probe"],
            }

        entries = [
            {"canonical_json": False, "hash_policy": "NOT_APPLICABLE", "kind": "directory", "mode": f"{stat.S_IMODE(os.lstat(root).st_mode):04o}", "path": str(root), "semantic_sha256": None, "sha256": None, "size": None, "sources": ["fresh-l2-independent-probe"]},
            record(manifest, canonical_json=True),
            record(artifact),
            record(payload, deferred=True),
        ]
        entries.sort(key=lambda item: item["path"])
        manifest_raw = manifest.read_bytes()
        closure = {
            "core_manifest_mode": "0444",
            "core_manifest_path": str(manifest),
            "core_manifest_semantic_sha256": hash_bytes(manifest_raw),
            "core_manifest_sha256": hash_bytes(manifest_raw),
            "core_manifest_self_sha256": "1" * 64,
            "deferred_hash_paths": [str(payload)],
            "directory_count": 1,
            "entries": entries,
            "entries_sha256": hash_bytes(compact_bytes(entries)),
            "file_count": 3,
            "full_hash_file_count": 2,
            "official_evaluator_sha256": "2" * 64,
            "official_parser_sha256": "3" * 64,
            "package_key": "core_manifest_closure",
            "path_hash_mode_complete": True,
            "schema_version": 1,
        }
        positive = module.verify_core_manifest_closure(closure, expected_core_manifest_path=manifest)
        os.chmod(artifact, 0o644)
        artifact.write_bytes(b"12345678")
        os.chmod(artifact, 0o444)
        negative_stage = "NOT_REJECTED"
        try:
            module.verify_core_manifest_closure(closure, expected_core_manifest_path=manifest)
        except module.CoreManifestPreflightError as error:
            negative_stage = error.stage
        require(positive["status"] == "PASS_COMPLETE_BOUND_CORE_MANIFEST_CLOSURE" and negative_stage == "PATH_HASH", "real preflight positive/negative probe")
        return {"negative_stage": negative_stage, "positive_status": positive["status"]}


def verify_defect() -> dict[str, Any]:
    wrong = str(V19_WRONG_CORE_MANIFEST)
    correct = str(CORE_MANIFEST)
    wrapper_source = V19_WRAPPER.read_text(encoding="utf-8")
    verifier_source = V19_VERIFIER.read_text(encoding="utf-8")
    require(wrong in wrapper_source and not os.path.lexists(V19_WRONG_CORE_MANIFEST), "V19 missing manifest path diagnosis")
    require(correct not in wrapper_source and CORE_MANIFEST.is_file(), "V19 expected/canonical path distinction")
    require(wrong not in verifier_source and "CORE_MANIFEST" not in verifier_source, "V19 static verifier omitted executable path check")
    return {"actual_core_manifest_sha256": sha256_file(CORE_MANIFEST), "actual_core_manifest_mode": f"{stat.S_IMODE(os.lstat(CORE_MANIFEST).st_mode):04o}", "v19_expected_path_absent": True, "v19_static_verifier_checked_path": False}


def run_candidate_verifier() -> tuple[dict[str, Any], bytes]:
    candidate, candidate_raw = canonical(CANDIDATE_REPORT)
    command = [str(INTERPRETER), str(VERIFIER)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=ENVIRONMENT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(completed.returncode == 0 and completed.stderr == b"", "candidate verifier rerun")
    require(completed.stdout == candidate_raw, "candidate verifier byte equality")
    require(candidate["status"] == "PASS_V20_BOUND_PATH_REPAIR_CANDIDATE_INERT", "candidate verifier status")
    return candidate, completed.stdout


def create_acceptance(package: dict[str, Any], package_raw: bytes) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    fixture, fixture_raw = canonical(FIXTURE_REPORT)
    require(fixture["status"] == "PASS_V20_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE" and fixture["case_count"] == 29, "fixture status")
    value = sealed({
        "accepted": True,
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_fresh_l2_static_acceptance",
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "core_manifest_closure_entries_sha256": package["core_manifest_closure"]["entries_sha256"],
        "decision": "ACCEPT_STATIC_PACKAGE",
        "execution_package_file_sha256": sha256_bytes(package_raw),
        "fixture_report_file_sha256": sha256_bytes(fixture_raw),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "required_disposition": "V20_BOUND_PATH_REPAIR_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY",
        "reviewer_role": "Fresh-L2",
        "runtime_namespace_file_count": 0,
        "static_acceptance_grants_execution_authority": False,
        "v18_binding_table_file_sha256": V18_BINDING_SHA256,
        "v19_terminal_file_sha256": V19_TERMINAL_SHA256,
        "v20_executed": False,
    }, "acceptance_sha256")
    schema, _ = canonical(ACCEPTANCE_SCHEMA)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)
    verify_self_hash(value, "acceptance_sha256")
    raw = compact_bytes(value)
    require(not os.path.lexists(ACCEPTANCE_DIR) and not os.path.lexists(ACCEPTANCE), "acceptance path already exists")
    os.chmod(ACTION_ROOT, 0o755)
    try:
        os.mkdir(ACCEPTANCE_DIR, 0o700)
        write_once(ACCEPTANCE, raw)
        os.chmod(ACCEPTANCE, 0o444)
        os.chmod(ACCEPTANCE_DIR, 0o555)
        descriptor = os.open(ACTION_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.chmod(ACTION_ROOT, 0o555)
    return value


def run_post_acceptance_verifier() -> dict[str, Any]:
    command = [str(INTERPRETER), str(VERIFIER), "--allow-acceptance", "--report", str(POST_REPORT)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=ENVIRONMENT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(completed.returncode == 0 and completed.stderr == b"", "post-acceptance verifier")
    report, raw = canonical(POST_REPORT)
    require(raw == completed.stdout, "post-acceptance report byte equality")
    require(report["status"] == "PASS_V20_BOUND_PATH_REPAIR_POST_ACCEPTANCE_INERT" and report["acceptance"]["present"] is True, "post-acceptance status")
    return report


def accept() -> dict[str, Any]:
    sys.addaudithook(audit_hook)
    require(not os.path.lexists(ACCEPTANCE) and not os.path.lexists(ACCEPTANCE_DIR), "Fresh-L2 acceptance already exists")
    require(not os.path.lexists(REVIEW_REPORT) and not os.path.lexists(POST_REPORT), "Fresh-L2 review output already exists")
    require(not os.path.lexists(RUNTIME_ROOT), "V20 runtime namespace materialized")
    require(sha256_file(V19_TERMINAL) == V19_TERMINAL_SHA256, "V19 terminal drift")
    package, package_raw = canonical(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    request, _ = canonical(REVIEW_REQUEST)
    verify_self_hash(request, "request_sha256")
    require(tree_digest(ACTION_ROOT)["tree_sha256"] == request["action_root_tree_sha256"], "pre-acceptance action tree")
    require(package["action_identity"]["action_id"] == ACTION_ID, "action identity")
    require(package["claim_boundary"]["claim"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "claim boundary")
    require(package["numerical_selection"]["tensor_names"] == ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"], "three numerical selections")
    verify_invocations(package)
    closure = verify_closure(package)
    defect = verify_defect()
    probe = probe_real_preflight()
    candidate, candidate_stdout = run_candidate_verifier()
    preservation_mismatches = [expected["root"] for expected in package["preservation"] if tree_digest(Path(expected["root"])) != expected]
    require(not preservation_mismatches and len(package["preservation"]) == 17, "predecessor preservation")
    require(not os.path.lexists(RUNTIME_ROOT), "candidate checks materialized runtime")
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "pre-acceptance official effects")
    review = sealed({
        "accepted": True,
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_fresh_l2_independent_review",
        "candidate_report_file_sha256": sha256_bytes(candidate_stdout),
        "candidate_status": candidate["status"],
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "core_manifest_closure": closure,
        "defect_diagnosis": defect,
        "decision": "ACCEPT_STATIC_PACKAGE",
        "exact_invocation_contract_count": 3,
        "fixture_case_count": candidate["fixture"]["case_count"],
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "preflight_probe": probe,
        "preservation_root_count": len(package["preservation"]),
        "reviewer_role": "Fresh-L2",
        "runtime_namespace_file_count": 0,
        "v18_binding_record_count": 25,
        "v19_terminal_file_sha256": V19_TERMINAL_SHA256,
        "v20_executed": False,
    }, "review_sha256")
    write_once(REVIEW_REPORT, compact_bytes(review))
    acceptance = create_acceptance(package, package_raw)
    post = run_post_acceptance_verifier()
    require(not os.path.lexists(RUNTIME_ROOT), "post-acceptance runtime namespace materialized")
    preservation_mismatches = [expected["root"] for expected in package["preservation"] if tree_digest(Path(expected["root"])) != expected]
    require(not preservation_mismatches, "post-acceptance predecessor preservation")
    require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "post-acceptance official effects")
    return {
        "acceptance_file_sha256": sha256_file(ACCEPTANCE),
        "acceptance_self_sha256": acceptance["acceptance_sha256"],
        "post_acceptance_report_file_sha256": sha256_file(POST_REPORT),
        "post_acceptance_status": post["status"],
        "review_file_sha256": sha256_file(REVIEW_REPORT),
        "status": "PASS_V20_FRESH_L2_STATIC_ACCEPTANCE",
    }


def main() -> int:
    try:
        result = accept()
    except Exception as error:
        sys.stderr.write(f"V20_FRESH_L2_ACCEPT_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
