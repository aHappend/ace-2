#!/usr/bin/env python3
"""Diagnose the retired V19 CORE_MANIFEST preflight defect without execution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
V19_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_action_root"
V19_WRAPPER = V19_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py"
V17_WRAPPER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root/tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py"
V19_BUILDER = PROJECT_ROOT / "tools/build_v19_exactly_once_v18_binding.py"
V19_VERIFIER = PROJECT_ROOT / "tools/verify_v19_exactly_once_v18_binding_inert.py"
V19_PACKAGE = V19_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_PACKAGE.json"
V19_ACCEPTANCE = V19_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V19_POST_REPORT = PROJECT_ROOT / "build/v19-exactly-once-v18-binding-attempt-0001/fresh-l2-post-acceptance-inert-report.json"
V19_RUNTIME = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1"
V19_TERMINAL = V19_RUNTIME / "primary/authority/base/first-terminal.json"
V19_OWNER = V19_RUNTIME / "primary/authority/base/owner-claim.json"
V20_BATCH = PROJECT_ROOT / "design/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_V20_BOUND_PATH_REPAIR_REPLACEMENT_BATCH.json"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
EXPECTED_CORE_MANIFEST_SHA256 = "9e120e0e831ec736df45bd1b9f825217ce3368c5e22dbf3e4fc67ac537ea61f1"
EXPECTED_CORE_MANIFEST_MODE = 0o444
AUDIT = {"official_payload_opens": 0, "subprocess_starts": 0}


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


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
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash mismatch: {field}")


def tree_digest(root: Path) -> dict[str, Any]:
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
            raise DiagnosticError(f"unsupported preservation entry: {path}")
    return {
        "entry_count": len(records),
        "file_count": sum(record["kind"] == "file" for record in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def source_assignments(path: Path) -> tuple[ast.Module, dict[str, ast.AST]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
    return tree, assignments


def evaluate_assignment(assignments: dict[str, ast.AST], name: str) -> Any:
    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            require(node.id in assignments, f"unbound source assignment: {node.id}")
            return evaluate(assignments[node.id])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
            return Path(evaluate(node.args[0]))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return evaluate(node.left) / evaluate(node.right)
        raise DiagnosticError(f"unsupported source assignment expression for {name}: {ast.dump(node)}")

    require(name in assignments, f"source assignment absent: {name}")
    return evaluate(assignments[name])


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_opens"] += 1
            raise DiagnosticError("official payload open prohibited")
    if event == "subprocess.Popen":
        AUDIT["subprocess_starts"] += 1
        raise DiagnosticError("subprocess start prohibited")


def write_once(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = compact_bytes(report)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            require(written > 0, "short report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def diagnose() -> dict[str, Any]:
    package, package_raw = canonical(V19_PACKAGE)
    acceptance, acceptance_raw = canonical(V19_ACCEPTANCE)
    post_report, post_raw = canonical(V19_POST_REPORT)
    terminal, terminal_raw = canonical(V19_TERMINAL)
    v20_batch = json_object(V20_BATCH)
    verify_self_hash(package, "package_content_sha256")
    verify_self_hash(acceptance, "acceptance_sha256")
    verify_self_hash(post_report, "report_sha256")
    verify_self_hash(terminal, "first_terminal_sha256")

    v19_tree, v19_assignments = source_assignments(V19_WRAPPER)
    _, v17_assignments = source_assignments(V17_WRAPPER)
    _, builder_assignments = source_assignments(V19_BUILDER)
    verifier_tree, _ = source_assignments(V19_VERIFIER)
    generated_path = evaluate_assignment(v19_assignments, "CORE_MANIFEST")
    predecessor_path = evaluate_assignment(v17_assignments, "CORE_MANIFEST")
    builder_hash_path = evaluate_assignment(builder_assignments, "V17_CORE_MANIFEST")
    generated_hash = evaluate_assignment(v19_assignments, "EXPECTED_CORE_MANIFEST_SHA256")
    require(isinstance(generated_path, Path) and isinstance(predecessor_path, Path) and isinstance(builder_hash_path, Path), "manifest path source types")
    require(generated_hash == EXPECTED_CORE_MANIFEST_SHA256, "generated wrapper manifest hash")
    require(builder_hash_path == predecessor_path, "builder hash source path")
    require(not os.path.lexists(generated_path), "retired V19 defect path unexpectedly exists")
    require(predecessor_path.is_file() and not predecessor_path.is_symlink(), "correct core manifest absent")
    predecessor_info = os.lstat(predecessor_path)
    require(stat.S_IMODE(predecessor_info.st_mode) == EXPECTED_CORE_MANIFEST_MODE, "correct core manifest mode")
    require(sha256_file(predecessor_path) == EXPECTED_CORE_MANIFEST_SHA256, "correct core manifest hash")
    require(generated_path == Path(str(predecessor_path).replace("V17", "V19")), "broad rewrite reproduction")

    v19_wrapper_source = V19_WRAPPER.read_text(encoding="utf-8")
    builder_source = V19_BUILDER.read_text(encoding="utf-8")
    verifier_names = {node.id for node in ast.walk(verifier_tree) if isinstance(node, ast.Name)}
    wrapper_names = {node.id for node in ast.walk(v19_tree) if isinstance(node, ast.Name)}
    require("CORE_MANIFEST" in wrapper_names and "sha256_file(CORE_MANIFEST)" in v19_wrapper_source, "launcher manifest preflight absent")
    require('source = source.replace("V17", "V19").replace("execution_v17_exactly_once", "execution_v19_exactly_once")' in builder_source, "broad V17-to-V19 launcher rewrite absent")
    require("CORE_MANIFEST" not in verifier_names, "V19 inert verifier unexpectedly checks launcher manifest path")
    require("core_manifest_path" not in package["frozen_core"], "V19 package unexpectedly binds core manifest path")
    require(package["frozen_core"]["core_root"] == str(predecessor_path.parent), "V19 package core root")
    require(package["frozen_core"]["core_manifest_file_sha256"] == EXPECTED_CORE_MANIFEST_SHA256, "V19 package core manifest hash")
    require(acceptance["decision"] == "ACCEPT_STATIC_PACKAGE" and acceptance["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "V19 acceptance boundary")
    require(post_report["status"] == "PASS_V19_EXACTLY_ONCE_V18_BINDING_POST_ACCEPTANCE_INERT", "V19 post-acceptance report")

    require(terminal["status"] == "PREFLIGHT_FAILED_TERMINAL", "V19 terminal status")
    require(terminal["reason_code"] == "READ_ONLY_PREFLIGHT_FAILED", "V19 terminal reason")
    require(terminal["wrapper_diagnostic"]["exception_type"] == "FileNotFoundError", "V19 terminal exception")
    require(terminal["invocation_count_performed"] == 0, "V19 evaluator invocation count")
    require(terminal["payload_open_count"] == 0, "V19 payload open count")
    require(terminal["official_target_process_starts"] == 0, "V19 target start count")
    require(terminal["authority_sha256"] is None and terminal["credential_consumed"] is False, "V19 authority or credential effect")
    require(V19_OWNER.is_file(), "V19 owner claim absent")
    forbidden_runtime_files = [
        V19_RUNTIME / "primary/authority/base/authority.json",
        V19_RUNTIME / "primary/authority/base/credential.json",
        V19_RUNTIME / "primary/authority/base/authority-ledger.json",
        V19_RUNTIME / "primary/result/base/result.json",
        V19_RUNTIME / "fallback/first-terminal.json",
    ]
    require(not any(os.path.lexists(path) for path in forbidden_runtime_files), "unexpected retired V19 runtime effect")

    preservation_before: list[dict[str, Any]] = []
    for expected in package["preservation"]:
        observed = tree_digest(Path(expected["root"]))
        require(observed == expected, f"predecessor preservation drift: {expected['root']}")
        preservation_before.append(observed)
    v19_root_before = tree_digest(V19_ROOT)
    v19_runtime_before = tree_digest(V19_RUNTIME)

    replacement = v20_batch["replacement_nodes"][0]["new_identity"]
    v20_runtime = PROJECT_ROOT / replacement["runtime_namespace"]
    require(not os.path.lexists(v20_runtime), "V20 runtime namespace exists")

    preservation_after = [tree_digest(Path(expected["root"])) for expected in package["preservation"]]
    v19_root_after = tree_digest(V19_ROOT)
    v19_runtime_after = tree_digest(V19_RUNTIME)
    require(preservation_after == preservation_before, "predecessor roots changed during diagnosis")
    require(v19_root_after == v19_root_before, "V19 action root changed during diagnosis")
    require(v19_runtime_after == v19_runtime_before, "V19 runtime root changed during diagnosis")
    require(AUDIT == {"official_payload_opens": 0, "subprocess_starts": 0}, "diagnostic audit boundary")

    report: dict[str, Any] = {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v19_core_manifest_preflight_defect_diagnostic",
        "builder_root_cause": {
            "broad_version_rewrite": 'source.replace("V17", "V19")',
            "builder_hash_source_path": str(builder_hash_path),
            "builder_hash_source_sha256": sha256_file(builder_hash_path),
            "builder_source_sha256": sha256_file(V19_BUILDER),
            "generated_path_matches_broad_rewrite": generated_path == Path(str(predecessor_path).replace("V17", "V19")),
            "generated_wrapper_source_sha256": sha256_file(V19_WRAPPER),
            "predecessor_wrapper_source_sha256": sha256_file(V17_WRAPPER),
        },
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "core_manifest_binding": {
            "correct_file_mode": f"{stat.S_IMODE(predecessor_info.st_mode):04o}",
            "correct_file_path": str(predecessor_path),
            "correct_file_sha256": sha256_file(predecessor_path),
            "correct_file_size": predecessor_info.st_size,
            "generated_v19_expected_path": str(generated_path),
            "generated_v19_expected_path_exists": False,
            "predecessor_and_builder_path_identical": predecessor_path == builder_hash_path,
        },
        "failure_taxonomy": "PRECONSUMPTION_PREFLIGHT_BOUND_PATH_ABSENT",
        "official_effect_boundary": {
            "diagnostic_official_payload_open_count": 0,
            "diagnostic_subprocess_start_count": 0,
            "v20_runtime_namespace_absent": True,
        },
        "preservation": {
            "predecessor_root_count": len(preservation_before),
            "predecessor_roots": preservation_before,
            "v19_action_root": v19_root_before,
            "v19_runtime_root": v19_runtime_before,
            "verified_unchanged_before_after": True,
        },
        "static_acceptance_gap": {
            "acceptance_decision": acceptance["decision"],
            "acceptance_file_sha256": sha256_bytes(acceptance_raw),
            "core_manifest_path_bound_in_package": False,
            "inert_verifier_core_manifest_path_check_present": False,
            "package_file_sha256": sha256_bytes(package_raw),
            "post_acceptance_report_file_sha256": sha256_bytes(post_raw),
            "post_acceptance_status": post_report["status"],
            "verifier_source_sha256": sha256_file(V19_VERIFIER),
        },
        "status": "PASS_V19_CORE_MANIFEST_ZERO_INVOCATION_DEFECT_DIAGNOSED",
        "v19_terminal": {
            "action_id": terminal["action_id"],
            "action_retired": terminal["action_retired"],
            "authority_sha256": terminal["authority_sha256"],
            "credential_consumed": terminal["credential_consumed"],
            "exception_type": terminal["wrapper_diagnostic"]["exception_type"],
            "invocation_count_performed": terminal["invocation_count_performed"],
            "official_target_process_starts": terminal["official_target_process_starts"],
            "owner_claim_file_sha256": sha256_file(V19_OWNER),
            "payload_open_count": terminal["payload_open_count"],
            "reason_code": terminal["reason_code"],
            "status": terminal["status"],
            "terminal_file_sha256": sha256_bytes(terminal_raw),
            "terminal_self_sha256": terminal["first_terminal_sha256"],
        },
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    try:
        report = diagnose()
        if args.report is not None:
            write_once(args.report, report)
        sys.stdout.buffer.write(compact_bytes(report))
    except Exception as error:
        sys.stderr.write(f"V19_CORE_MANIFEST_DIAGNOSTIC_FAIL:{type(error).__name__}:{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
