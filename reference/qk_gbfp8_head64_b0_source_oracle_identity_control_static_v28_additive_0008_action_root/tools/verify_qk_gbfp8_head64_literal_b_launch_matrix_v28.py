#!/usr/bin/env python3
"""Run V28's literal--B negative matrix and positive inert candidate."""

from __future__ import annotations

import sys
EARLY_DONT_WRITE_BYTECODE = sys.dont_write_bytecode is True
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build/v28-b0-source-oracle-identity-control-static-0013"
CANDIDATE = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_source_oracle_identity_control_candidate_v28.py"
POST = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_source_oracle_identity_control_post_acceptance_v28.py"
CATALOG = ACTION_ROOT / "fixtures/LITERAL_B_LAUNCH_NEGATIVE_CASES.json"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")


class MatrixError(RuntimeError):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise MatrixError(detail)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"symlink: {relative}")
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": mode, "path": relative})
        elif stat.S_ISREG(info.st_mode):
            raw = path.read_bytes()
            records.append({"kind": "file", "mode": mode, "path": relative, "sha256": sha256_bytes(raw), "size": len(raw)})
        else:
            raise MatrixError(f"unsupported: {relative}")
    return records


def tree_hash() -> str:
    return sha256_bytes(compact_bytes(inventory(ACTION_ROOT)))


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPYCACHEPREFIX", None)
    env.pop("PYTHONPATH", None)
    return env


def command(case: str, script: Path, temporary: Path) -> tuple[list[str], dict[str, str]]:
    env = environment()
    python = str(INTERPRETER)
    target = str(script)
    if case == "environment_only_exact_v24_command": return [python, target], env
    if case == "fake_b_after_script_path": return [python, target, "-B"], env
    if case == "missing_script_path": return [python, "-B", "-c", f"import runpy;runpy.run_path({target!r},run_name='__main__')"], env
    if case == "ambiguous_script_path": return [python, "-B", target, target], env
    if case == "argv_disagreement": return [python, "-B", "-c", f"import runpy,sys;sys.orig_argv=[sys.executable,'-B',{target!r}];runpy.run_path({target!r},run_name='__main__')"], env
    if case == "missing_pythondontwritebytecode":
        env.pop("PYTHONDONTWRITEBYTECODE", None); return [python, "-B", target], env
    if case == "non_null_pycacheprefix":
        env["PYTHONPYCACHEPREFIX"] = str(temporary / "pycache"); return [python, "-B", target], env
    if case == "late_sys_dont_write_bytecode":
        (temporary / "sitecustomize.py").write_text("import sys\nsys.dont_write_bytecode=False\n", encoding="ascii")
        env["PYTHONPATH"] = str(temporary); return [python, "-B", target], env
    if case == "ineffective_suppression":
        (temporary / "sitecustomize.py").write_text("import importlib.machinery\nfrom pathlib import Path\no=importlib.machinery.SourceFileLoader.exec_module\ndef f(self,module):\n o(self,module)\n if Path(self.path).name=='suppression_probe.py':\n  p=Path(self.path).parent/'__pycache__';p.mkdir(exist_ok=True);(p/'forced.pyc').write_bytes(b'x')\nimportlib.machinery.SourceFileLoader.exec_module=f\n", encoding="ascii")
        env["PYTHONPATH"] = str(temporary); return [python, "-B", target], env
    raise MatrixError(case)


def run_negative(case: dict[str, Any], script: Path) -> dict[str, Any]:
    before = tree_hash()
    with tempfile.TemporaryDirectory(prefix="ace2-v28-negative-", dir=BUILD_ROOT) as directory:
        argv, env = command(case["id"], script, Path(directory))
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    after = tree_hash()
    stderr = completed.stderr.decode("utf-8", "replace")
    require(completed.returncode != 0, f"negative passed: {case['id']}:{script.name}")
    require(case["expected_error"] in stderr, f"wrong negative error: {case['id']}:{stderr}")
    require(before == after, "negative mutated tree")
    return {"action_tree_unchanged": True, "entrypoint": script.name, "expected_error": case["expected_error"], "id": case["id"], "rejected": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve(strict=False)
    require(str(output).startswith(str(BUILD_ROOT.resolve())), "output outside build root")
    require(not output.exists(), "output exists")
    output.mkdir(parents=True)
    catalog = json.loads(CATALOG.read_text(encoding="ascii"))
    unhashed = dict(catalog); observed = unhashed.pop("fixture_sha256")
    require(observed == sha256_bytes(compact_bytes(unhashed)), "catalog self hash")
    initial = tree_hash()
    negatives = [run_negative(case, script) for case in catalog["cases"] for script in (CANDIDATE, POST)]
    candidate_path = output / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_CANDIDATE_REPORT.json"
    negative_path = output / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_NEGATIVE_FIXTURE_REPORT.json"
    completed = subprocess.run([str(INTERPRETER), "-B", str(CANDIDATE), "--output", str(candidate_path), "--negative-output", str(negative_path)], cwd=PROJECT_ROOT, env=environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(completed.returncode == 0, completed.stderr.decode("utf-8", "replace"))
    final = tree_hash()
    require(initial == final, "positive mutated tree")
    candidate = json.loads(candidate_path.read_text(encoding="ascii"))
    negative = json.loads(negative_path.read_text(encoding="ascii"))
    require(candidate["status"] == "PASS_V28_STATIC_CANDIDATE_PENDING_FRESH_L2", "candidate status")
    require(negative["status"] == "PASS_V28_IDENTITY_AND_AUTHORITY_MUTATION_REJECTION", "negative status")
    report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_static_v28_launch_matrix_report",
        "candidate_report_file_sha256": sha256_bytes(candidate_path.read_bytes()),
        "candidate_report_sha256": candidate["report_sha256"],
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "negative_fixture_report_file_sha256": sha256_bytes(negative_path.read_bytes()),
        "negative_fixture_report_sha256": negative["report_sha256"],
        "negative_launch_count": len(negatives),
        "negative_launches": negatives,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "post_acceptance_positive_launch_deferred": True,
        "sealed_action_tree_after_sha256": final,
        "sealed_action_tree_before_sha256": initial,
        "status": "PASS_V28_LITERAL_B_LAUNCH_MATRIX_PENDING_FRESH_L2",
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    report_path = output / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_LAUNCH_MATRIX_REPORT.json"
    report_path.write_bytes(compact_bytes(report))
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
