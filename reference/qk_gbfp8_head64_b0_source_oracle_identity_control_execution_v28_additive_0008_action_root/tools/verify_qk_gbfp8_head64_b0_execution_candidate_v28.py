#!/home/argustest/miniconda3/bin/python3.13
"""Bytecode-free inert verifier for the sealed V28 execution package."""

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

sys.dont_write_bytecode = True

PROJECT_ROOT = Path("/home/argustest/ace-2")
ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_additive_0008_action_root"
PACKAGE = ROOT / "QK_GBFP8_HEAD64_B0_SOURCE_ORACLE_IDENTITY_CONTROL_EXECUTION_V28_PACKAGE.json"
ACCEPTANCE = ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
ACCEPTANCE_SCHEMA = ROOT / "reference/QK_GBFP8_HEAD64_B0_V28_FRESH_L2_ACCEPTANCE_SCHEMA.json"
STATIC_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0008_action_root"
STATIC_PACKAGE = STATIC_ROOT / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_PACKAGE.json"
STATIC_ACCEPTANCE = STATIC_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
BUILD_0014 = PROJECT_ROOT / "build/v28-b0-source-oracle-identity-control-static-0014/prepare_additive_0008_provenance_repair.py"
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0008_once"
EXTERNAL_AUTHORITY_ROOT = PROJECT_ROOT / "authority/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0008_once"
OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"

STATIC_PACKAGE_SHA256 = "c3052c9d72733a36fa83993aa8fb22e037e00f1f0b52c26eebd6c9664fe78237"
STATIC_ACCEPTANCE_SHA256 = "4e47a94ee3f6faed3ebba23d5cea499171c51303b4cfbe1d18884ddcf19e46c2"
BUILD_0014_SHA256 = "d584ef93a92a33b92a0a1ef9d1ba49376efeefb21b8a00c98a1a8ef96b164c31"
ACTION_ID = "ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control-execute-once:38571d38:additive-0008"
AUDIT = {"official_payload_open_count": 0, "process_start_count": 0}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise VerificationError(detail)


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
    value = json.loads(raw.decode("ascii", "strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    require(type(value) is dict, f"JSON object {path}")
    return value, raw


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash {field}")


def inventory(root: Path, include_root: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if include_root:
        info = os.lstat(root)
        records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": "."})
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"symlink {relative}")
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative,
                            "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise VerificationError(f"unsupported entry {relative}")
    return records


def tree_hash(root: Path, include_root: bool = True) -> str:
    return sha256_bytes(compact_bytes(inventory(root, include_root)))


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            candidate = Path(args[0]).resolve()
        except (TypeError, OSError, RuntimeError):
            candidate = None
        if candidate == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_open_count"] += 1
            raise VerificationError("official payload open prohibited")
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.posix_spawnp"}:
        AUDIT["process_start_count"] += 1
        raise VerificationError("process start prohibited")


def verify_preservation(package: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(STATIC_PACKAGE) == STATIC_PACKAGE_SHA256, "accepted additive-0008 package drift")
    require(sha256_file(STATIC_ACCEPTANCE) == STATIC_ACCEPTANCE_SHA256, "accepted additive-0008 acceptance drift")
    require(sha256_file(BUILD_0014) == BUILD_0014_SHA256, "unauthorized build-0014 draft drift")
    for path in STATIC_ROOT.rglob("*"):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), "additive-0008 symlink")
        require("__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}, "additive-0008 bytecode")
        if stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o555, f"additive-0008 directory mode {path}")
        elif stat.S_ISREG(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o444, f"additive-0008 file mode {path}")
    require(package["preservation"]["accepted_static_package_file_sha256"] == STATIC_PACKAGE_SHA256, "package preservation binding")
    require(package["preservation"]["accepted_static_acceptance_file_sha256"] == STATIC_ACCEPTANCE_SHA256, "acceptance preservation binding")
    require(package["preservation"]["unauthorized_build_0014_file_sha256"] == BUILD_0014_SHA256, "build-0014 preservation binding")
    return {"accepted_additive_0008_unchanged": True, "build_0014_untouched": True}


def verify_inventory(package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    observed = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file())
    expected = set(package["static_file_policy"]["allowed_relative_files"])
    acceptance_relative = package["static_file_policy"]["acceptance_relative_path"]
    if allow_acceptance:
        require(ACCEPTANCE.is_file(), "post-acceptance verification without acceptance")
    else:
        expected.remove(acceptance_relative)
        require(not ACCEPTANCE.exists(), "Engineer package contains Reviewer acceptance")
    require(observed == sorted(expected), "exact execution-root file inventory")
    for path in ROOT.rglob("*"):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"execution-root symlink {path}")
        if stat.S_ISDIR(info.st_mode):
            expected_mode = 0o755 if path == ROOT / "review" else 0o555
            require(stat.S_IMODE(info.st_mode) == expected_mode, f"directory mode {path}")
        elif stat.S_ISREG(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o444, f"file mode {path}")
    require(stat.S_IMODE(os.lstat(ROOT).st_mode) == 0o555, "root mode")
    if not allow_acceptance:
        require(not any((ROOT / "review").iterdir()), "review directory not empty")
    return {"acceptance_present": allow_acceptance, "file_count": len(observed)}


def verify_generated(package: dict[str, Any]) -> dict[str, Any]:
    for relative, identity in package["generated_files"].items():
        path = ROOT / relative
        require(path.is_file() and path.stat().st_size == identity["size"], f"generated size {relative}")
        require(sha256_file(path) == identity["sha256"], f"generated hash {relative}")
    return {"generated_file_count": len(package["generated_files"])}


def verify_sources() -> dict[str, Any]:
    wrapper_path = ROOT / "tools/execute_qk_gbfp8_head64_b0_source_oracle_identity_control_once_v28.py"
    adapter_path = ROOT / "tools/qk_gbfp8_head64_b0_evaluator_adapter_v28.py"
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    adapter_source = adapter_path.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_source)
    require("shell=False" in adapter_source, "shell-free adapter")
    require("os.O_EXCL" in wrapper_source or "durable_create" in wrapper_source, "create-once wrapper")
    require("EXTERNAL_MANAGER_ADMISSION" in wrapper_source and "EXTERNAL_OPERATOR_AUTHORITY" in wrapper_source, "external authority inputs")
    require("payload_base64" not in adapter_source and "preview_ascii" not in adapter_source, "diagnostic leakage exclusion")
    calls = {node.func.id for node in ast.walk(wrapper_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    require({"validate_external_records", "build_runtime_envelope", "invoke_evaluator", "publish_terminal"} <= calls, "wrapper lifecycle calls")
    return {"create_once": True, "external_authority_only": True, "shell_false": True}


def verify_regression(package: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "tools"))
    import test_qk_gbfp8_head64_b0_execution_v28 as regression
    regression.AUDIT.update({"official_payload_open_count": 0, "process_start_count": 0})
    observed = regression.run()
    report, report_raw = canonical(ROOT / "evidence/SYNTHETIC_V28_INERT_REGRESSION_REPORT.json")
    require(observed == report and compact_bytes(report) == report_raw, "regression report reproducibility")
    verify_self_hash(report, "report_sha256")
    require(report["status"] == "PASS_V28_INERT_PROTECTED_DATA_FREE_REGRESSION", "regression status")
    require(report["official_payload_open_count"] == report["official_evaluator_invocation_count"] == report["official_target_process_starts"] == 0, "regression inert counts")
    require(len(report["adapter_boundary_cases"]) == 10, "adapter boundary cardinality")
    require(report["schema_and_mutation"]["schema_negative_count"] >= 37 and report["schema_and_mutation"]["binding_mutation_negative_count"] == 8, "negative coverage")
    return {"adapter_boundary_case_count": 10, "binding_mutation_negative_count": 8, "schema_negative_count": report["schema_and_mutation"]["schema_negative_count"]}


def verify_acceptance(package: dict[str, Any], package_raw: bytes, allow_acceptance: bool) -> dict[str, Any]:
    if not allow_acceptance:
        return {"present": False}
    from jsonschema import Draft202012Validator
    schema, _ = canonical(ACCEPTANCE_SCHEMA)
    acceptance, raw = canonical(ACCEPTANCE)
    Draft202012Validator(schema).validate(acceptance)
    verify_self_hash(acceptance, "acceptance_sha256")
    require(acceptance["execution_package_file_sha256"] == sha256_bytes(package_raw), "acceptance package binding")
    require(acceptance["execution_package_content_sha256"] == package["package_content_sha256"], "acceptance content binding")
    require(acceptance["static_acceptance_grants_execution_authority"] is False, "acceptance authority boundary")
    return {"acceptance_file_sha256": sha256_bytes(raw), "present": True}


def verify(allow_acceptance: bool = False) -> dict[str, Any]:
    before = inventory(ROOT)
    package, package_raw = canonical(PACKAGE)
    verify_self_hash(package, "package_content_sha256")
    require(package["action_identity"]["action_id"] == ACTION_ID, "action identity")
    require(package["claim_boundary"] == "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "claim boundary")
    require(package["authority_inputs"]["namespace_relation"] == "STRICTLY_OUTSIDE_RUNTIME_NAMESPACE", "authority namespace boundary")
    require(package["official_effects"] == {"evaluator_invocations": 0, "payload_opens": 0, "target_process_starts": 0}, "zero official effects claim")
    require(not os.path.lexists(RUNTIME_ROOT), "runtime namespace materialized")
    require(not os.path.lexists(EXTERNAL_AUTHORITY_ROOT), "live external authority materialized")
    report = {
        "acceptance": verify_acceptance(package, package_raw, allow_acceptance),
        "artifact_kind": "qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_inert_verifier_report",
        "generated_files": verify_generated(package),
        "inventory": verify_inventory(package, allow_acceptance),
        "manifest_file_sha256": sha256_bytes(package_raw),
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "preservation": verify_preservation(package),
        "regression": verify_regression(package),
        "runtime_namespace_materialized": False,
        "sources": verify_sources(),
        "status": "PASS_V28_POST_ACCEPTANCE_INERT" if allow_acceptance else "PASS_V28_CANDIDATE_INERT_READY_FOR_FRESH_L2",
    }
    after = inventory(ROOT)
    require(before == after, "verifier changed candidate tree")
    require(AUDIT == {"official_payload_open_count": 0, "process_start_count": 0}, "verifier inert audit")
    report["candidate_tree_sha256"] = tree_hash(ROOT)
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-acceptance", action="store_true")
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    try:
        report = verify(args.allow_acceptance)
    except Exception as error:
        sys.stderr.write(f"B0_V28_INERT_VERIFY_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
