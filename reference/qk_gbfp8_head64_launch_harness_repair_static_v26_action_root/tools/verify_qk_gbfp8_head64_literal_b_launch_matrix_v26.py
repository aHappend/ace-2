#!/usr/bin/env python3
"""Execute the inert positive/negative literal-``-B`` launch matrix for V26."""

from __future__ import annotations

import sys

EARLY_DONT_WRITE_BYTECODE = sys.dont_write_bytecode is True
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _load_bound_launch_guard() -> Any:
    guard_path = Path(__file__).resolve(strict=True).with_name("literal_b_launch_guard_v26.py")
    raw = guard_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "ee624cdad6eb91b0e7a1efd9a99d10be7b2d0f773d38dc1f2c87233d4b43504f":
        raise RuntimeError("LITERAL_B_GUARD_HASH_MISMATCH")
    name = "ace2_v26_matrix_bound_literal_b_launch_guard"
    spec = importlib.util.spec_from_file_location(name, guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("LITERAL_B_GUARD_IMPORT_SPEC")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


_BOUND_LAUNCH_GUARD = _load_bound_launch_guard()
prove_effective_suppression = _BOUND_LAUNCH_GUARD.prove_effective_suppression
validate_current_process = _BOUND_LAUNCH_GUARD.validate_current_process


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build/v26-launch-harness-repair-static-0003"
CANDIDATE = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_launch_harness_repair_candidate_v26.py"
POST = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_launch_harness_repair_post_acceptance_v26.py"
CATALOG = ACTION_ROOT / "fixtures/LITERAL_B_LAUNCH_NEGATIVE_CASES.json"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
LAUNCH_PROOF = validate_current_process(Path(__file__), EARLY_DONT_WRITE_BYTECODE)
SUPPRESSION_PROOF = prove_effective_suppression()


class MatrixError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value, raw


def inventory_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"action-tree symlink: {relative}")
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": mode, "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": mode, "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise MatrixError(f"unsupported action-tree entry: {relative}")
    return records


def action_tree_sha256() -> str:
    return sha256_bytes(compact_bytes(inventory_records(ACTION_ROOT)))


def base_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment.pop("PYTHONPATH", None)
    return environment


def sitecustomize(directory: Path, text: str) -> None:
    (directory / "sitecustomize.py").write_text(text, encoding="ascii")


def command_for(case_id: str, entrypoint: Path, temporary: Path) -> tuple[list[str], dict[str, str]]:
    environment = base_environment()
    script = str(entrypoint)
    python = str(INTERPRETER)
    if case_id == "environment_only_exact_v24_command":
        return [python, script], environment
    if case_id == "fake_b_after_script_path":
        return [python, script, "-B"], environment
    if case_id == "missing_script_path":
        code = f"import runpy;runpy.run_path({script!r},run_name='__main__')"
        return [python, "-B", "-c", code], environment
    if case_id == "ambiguous_script_path":
        return [python, "-B", script, script], environment
    if case_id == "argv_disagreement":
        code = f"import runpy,sys;sys.orig_argv=[sys.executable,'-B',{script!r}];runpy.run_path({script!r},run_name='__main__')"
        return [python, "-B", "-c", code], environment
    if case_id == "missing_pythondontwritebytecode":
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        return [python, "-B", script], environment
    if case_id == "non_null_pycacheprefix":
        environment["PYTHONPYCACHEPREFIX"] = str(temporary / "pycache-prefix")
        return [python, "-B", script], environment
    if case_id == "late_sys_dont_write_bytecode":
        sitecustomize(temporary, "import sys\nsys.dont_write_bytecode = False\n")
        environment["PYTHONPATH"] = str(temporary)
        return [python, "-B", script], environment
    if case_id == "ineffective_suppression":
        sitecustomize(
            temporary,
            "import importlib.machinery\n"
            "from pathlib import Path\n"
            "_ace2_original_exec_module = importlib.machinery.SourceFileLoader.exec_module\n"
            "def _ace2_forced_exec_module(self, module):\n"
            "    _ace2_original_exec_module(self, module)\n"
            "    if Path(self.path).name == 'suppression_probe.py':\n"
            "        cache = Path(self.path).parent / '__pycache__'\n"
            "        cache.mkdir(exist_ok=True)\n"
            "        (cache / 'forced.pyc').write_bytes(b'forced')\n"
            "importlib.machinery.SourceFileLoader.exec_module = _ace2_forced_exec_module\n",
        )
        environment["PYTHONPATH"] = str(temporary)
        return [python, "-B", script], environment
    raise MatrixError(f"unknown launch case: {case_id}")


def run_negative(case: dict[str, Any], entrypoint: Path) -> dict[str, Any]:
    before = action_tree_sha256()
    with tempfile.TemporaryDirectory(prefix="ace2-v26-launch-negative-", dir=BUILD_ROOT) as directory:
        command, environment = command_for(case["id"], entrypoint, Path(directory))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    after = action_tree_sha256()
    stderr = completed.stderr.decode("utf-8", errors="replace")
    require(completed.returncode != 0, f"negative launch unexpectedly passed: {entrypoint.name}:{case['id']}")
    require(case["expected_error"] in stderr, f"negative launch wrong error: {entrypoint.name}:{case['id']}")
    require(before == after, f"negative launch mutated action tree: {entrypoint.name}:{case['id']}")
    return {
        "action_tree_unchanged": True,
        "entrypoint": entrypoint.name,
        "expected_error": case["expected_error"],
        "id": case["id"],
        "rejected": True,
    }


def run_positive(output_root: Path) -> dict[str, Any]:
    candidate_report = output_root / "QK_GBFP8_HEAD64_LAUNCH_HARNESS_REPAIR_STATIC_V26_CANDIDATE_REPORT.json"
    negative_report = output_root / "QK_GBFP8_HEAD64_LAUNCH_HARNESS_REPAIR_STATIC_V26_NEGATIVE_FIXTURE_REPORT.json"
    command = [str(INTERPRETER), "-B", str(CANDIDATE), "--output", str(candidate_report), "--negative-output", str(negative_report)]
    before = action_tree_sha256()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=base_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    after = action_tree_sha256()
    require(completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace"))
    require(before == after, "positive candidate launch mutated sealed action tree")
    candidate_value, candidate_raw = canonical(candidate_report)
    negative_value, negative_raw = canonical(negative_report)
    require(candidate_value["status"] == "PASS_V26_STATIC_CANDIDATE_PENDING_FRESH_L2", "candidate status")
    require(negative_value["status"] == "PASS_V26_LITERAL_B_MODE_SEAL_AND_SYNTHETIC_MUTATION_REJECTION", "negative status")
    return {
        "action_tree_unchanged": True,
        "candidate_report_file_sha256": sha256_bytes(candidate_raw),
        "candidate_report_sha256": candidate_value["report_sha256"],
        "literal_b": True,
        "negative_fixture_report_file_sha256": sha256_bytes(negative_raw),
        "negative_fixture_report_sha256": negative_value["report_sha256"],
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve(strict=False)
    try:
        output_root.relative_to(BUILD_ROOT.resolve(strict=True))
    except ValueError as error:
        raise MatrixError("launch-matrix output root must stay inside declared V26 build root") from error
    require(not output_root.exists(), "launch-matrix output root already exists")
    output_root.mkdir(parents=True)

    catalog, catalog_raw = canonical(CATALOG)
    fixture = dict(catalog)
    observed_self_hash = fixture.pop("fixture_sha256")
    require(observed_self_hash == sha256_bytes(compact_bytes(fixture)), "launch catalog self hash")
    require(catalog["case_count"] == len(catalog["cases"]) == 9, "launch catalog count")

    initial_tree = action_tree_sha256()
    case_results = []
    for case in catalog["cases"]:
        for entrypoint in (CANDIDATE, POST):
            case_results.append(run_negative(case, entrypoint))
    positive = run_positive(output_root)
    final_tree = action_tree_sha256()
    require(initial_tree == final_tree, "launch matrix mutated sealed action tree")
    report = {
        "artifact_kind": "qk_gbfp8_head64_launch_harness_repair_v26_launch_matrix_report",
        "candidate_positive": positive,
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "launch_catalog_file_sha256": sha256_bytes(catalog_raw),
        "launch_catalog_sha256": observed_self_hash,
        "launch_proof": {**LAUNCH_PROOF, **SUPPRESSION_PROOF},
        "negative_launch_count": len(case_results),
        "negative_launches": case_results,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "post_acceptance_positive_launch_deferred": True,
        "sealed_action_tree_after_sha256": final_tree,
        "sealed_action_tree_before_sha256": initial_tree,
        "status": "PASS_V26_LITERAL_B_LAUNCH_MATRIX_PENDING_FRESH_L2",
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    output = output_root / "QK_GBFP8_HEAD64_LAUNCH_HARNESS_REPAIR_STATIC_V26_LAUNCH_MATRIX_REPORT.json"
    output.write_bytes(compact_bytes(report))
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
