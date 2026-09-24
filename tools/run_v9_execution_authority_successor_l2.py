#!/usr/bin/env python3
"""Run/check one fresh read-only L2 review of the sealed V9 successor package."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import argus_skill
from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.core.knobs import (
    resolve_role_backend,
    resolve_role_model,
    resolve_role_reasoning_effort,
    resolve_runner_bin_setting,
)
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path("/home/argustest/ace-2")
PACKAGE_ROOT = ROOT / "build/v9-single-request-execution-authority-successor-package-v1-candidate-0004"
REVIEW_ROOT = ROOT / "build/v9-single-request-execution-authority-successor-package-v1-candidate-0004-review"
MANIFEST = PACKAGE_ROOT / "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE.json"
VERIFIER = PACKAGE_ROOT / "verify_package_independent.py"
REVIEW_CONTRACT = PACKAGE_ROOT / "REVIEW_CONTRACT.md"
REQUEST = REVIEW_ROOT / "REVIEW_REQUEST.json"
RERUN = REVIEW_ROOT / "INDEPENDENT_VERIFIER_RERUN.json"
DECISION = REVIEW_ROOT / "FRESH_L2_REVIEW.json"
DISPOSITION = "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE_READY_NO_AUTHORITY_CONSUMED"
MISSION_ID = "v9-single-request-execution-authority-successor-package-v1-fresh-l2"
RULINGS = (
    "v9.successor.provenance-chain",
    "v9.successor.request-output-descriptors",
    "v9.successor.one-consumption-terminal-publication",
    "v9.successor.interpreter-input-fd-binding",
    "v9.successor.no-production-state",
    "v9.successor.canonical-historical-verifier-not-run",
    "v9.successor.sealed-immutable",
)
EXACT_ENV = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC", "PATH": "/usr/bin:/bin"}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    require(not path.exists() and not sidecar.exists(), f"refusing to overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def companion_matches(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.is_file() and sidecar.read_text(encoding="ascii").split() == [sha256(path), path.name]


def package_inputs() -> dict[str, Any]:
    require(PACKAGE_ROOT.is_dir() and stat.S_IMODE(os.lstat(PACKAGE_ROOT).st_mode) == 0o555, "package root is not sealed")
    expected = {MANIFEST.name, MANIFEST.name + ".sha256", REVIEW_CONTRACT.name, VERIFIER.name}
    actual = {path.name for path in PACKAGE_ROOT.iterdir()}
    require(actual == expected, "package file set differs")
    for path in PACKAGE_ROOT.iterdir():
        require(path.is_file() and not path.is_symlink() and stat.S_IMODE(os.lstat(path).st_mode) == 0o444, f"package file not sealed: {path}")
    require(companion_matches(MANIFEST), "manifest sidecar differs")
    manifest = load(MANIFEST)
    require(manifest.get("acceptance_boundary") == DISPOSITION, "manifest disposition differs")
    require(manifest.get("review_status") == "PENDING_FRESH_L2", "manifest review status differs")
    return {
        "claim_boundary": "Fresh read-only L2 review of one sealed static successor package. No execution authority, credential, claim, request, socket, cgroup, PID namespace, live/production state, target process, evaluator, model, RTL, XRT, or U280 action is permitted.",
        "manifest": {"byte_count": MANIFEST.stat().st_size, "path": str(MANIFEST), "sha256": sha256(MANIFEST)},
        "review_contract": {"byte_count": REVIEW_CONTRACT.stat().st_size, "path": str(REVIEW_CONTRACT), "sha256": sha256(REVIEW_CONTRACT)},
        "verifier": {"byte_count": VERIFIER.stat().st_size, "path": str(VERIFIER), "sha256": sha256(VERIFIER)},
    }


def rerun_verifier() -> dict[str, Any]:
    result = subprocess.run(
        ["/usr/bin/python3", "-B", str(VERIFIER)],
        cwd=ROOT,
        env=EXACT_ENV,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    require(result.returncode == 0, f"independent verifier failed: {result.stdout}")
    payload = json.loads(result.stdout)
    require(payload.get("status") == "PASS_V9_SUCCESSOR_INDEPENDENT_STATIC_VERIFICATION", "verifier status differs")
    require(payload.get("disposition") == DISPOSITION, "verifier disposition differs")
    require(payload.get("authority_consumed") is False and payload.get("target_process_starts") == 0, "verifier execution boundary differs")
    return {
        "command": ["/usr/bin/python3", "-B", str(VERIFIER)],
        "environment": {key: EXACT_ENV[key] for key in ("LANG", "LC_ALL", "PYTHONHASHSEED", "TZ")},
        "exit_code": result.returncode,
        "output": payload,
        "rerun_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def seal_review_root() -> None:
    for path in REVIEW_ROOT.iterdir():
        require(path.is_file() and not path.is_symlink(), f"unexpected review entry: {path}")
        os.chmod(path, 0o444)
    os.chmod(REVIEW_ROOT, 0o555)


def check_existing() -> None:
    package_inputs()
    require(REVIEW_ROOT.is_dir() and stat.S_IMODE(os.lstat(REVIEW_ROOT).st_mode) == 0o555, "review root is not sealed")
    expected = {REQUEST.name, REQUEST.name + ".sha256", RERUN.name, RERUN.name + ".sha256", DECISION.name, DECISION.name + ".sha256"}
    require({path.name for path in REVIEW_ROOT.iterdir()} == expected, "review file set differs")
    for path in (REQUEST, RERUN, DECISION):
        require(companion_matches(path), f"review sidecar differs: {path}")
    decision = load(DECISION)
    require(decision.get("accepted") is True and decision.get("reviewer_status") == "done", "review is not accepted")
    require(decision.get("disposition") == DISPOSITION, "review disposition differs")
    require(decision.get("authority_consumed") is False and decision.get("target_process_starts") == 0, "review boundary differs")
    require(decision.get("kind") == "fresh_l2_reviewer_verdict" and decision.get("producer_role") == "reviewer", "review verdict identity differs")
    require(decision.get("raw_review", {}).get("status") == "done", "raw reviewer status differs")
    require(bool(decision.get("reviewer_thread_id")), "reviewer thread identity is absent")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("V9_SUCCESSOR_FRESH_L2_CHECK_PASS")
        return 0

    require(not REVIEW_ROOT.exists(), f"review root already exists: {REVIEW_ROOT}")
    verified = package_inputs()
    REVIEW_ROOT.mkdir(mode=0o700)
    write_once(REQUEST, {
        "claim_boundary": verified["claim_boundary"],
        "disposition_required": DISPOSITION,
        "mission_id": MISSION_ID,
        "requested_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "requested_rulings": list(RULINGS),
        "schema_version": 1,
        "status": "PENDING_FRESH_L2",
        "verified_inputs": verified,
    })
    verifier_rerun = rerun_verifier()
    write_once(RERUN, verifier_rerun)

    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(project_root=global_root() / "projects" / sid, global_root=global_root(), mission_id=MISSION_ID)
    reviewer_skill = (Path(argus_skill.__file__).resolve().parent / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(encoding="utf-8")
    objective = "Independently Fresh-L2-review the exact sealed static V9 single-request execution-authority successor package without granting or consuming authority."
    instruction = (
        "Inspect the exact sealed package, manifest, review contract, independent verifier, accepted execution-domain attempt-0004, direct-spawn, activation, broker, canonical V9 root, and preserved predecessors directly. "
        "Freshly rerun only this command from an empty environment: env -i LANG=C LC_ALL=C PYTHONHASHSEED=0 TZ=UTC PATH=/usr/bin:/bin /usr/bin/python3 -B "
        + str(VERIFIER)
        + ". Do not import candidate code. Do not run the old canonical verifier; its historical 'FAIL V9 exact static file set' is expected because the frozen policy predates review/FRESH_L2_STATIC_ACCEPTANCE.json. "
        "Do not start activation, broker, transport, launcher, evaluator, quantization, model, GPU, RTL, XRT, U280, cgroup, PID namespace, or any production/live/authority path. "
        "Adversarially inspect provenance replacement, request replacement, outer V9ED publication field/status replacement, inner canonical first-terminal schema/path/status replacement, cross-layer status conflation, replay/cardinality changes, and exact/dynamic interpreter-input FD binding. "
        "Confirm that the outer status vocabulary is exactly CRASH_TERMINAL_CONSUMED, REJECTED_TERMINAL_CONSUMED, SUCCEEDED_TERMINAL, UNKNOWN_TERMINAL_CONSUMED; the inner vocabulary is exactly CONSUMED_ORPHAN, FAILED_TERMINAL, PREFLIGHT_FAILED_TERMINAL, SUCCEEDED_TERMINAL; and no total status mapping is claimed. "
        "Return done only if the package is immutable, the verifier passes with 23 mutation rejections and zero target starts, every forbidden state path remains absent, both descriptors cross-check against the accepted publication verifier and canonical schema/manifest, and the accepted kernel-domain disposition is live. "
        "Cover these review areas in the reasoning: "
        + ", ".join(RULINGS)
        + f". Return done only for the exact static package; the sealed verdict writer will record disposition {DISPOSITION}. A done verdict grants no execution authority and does not advance the Manager-owned RTL stage."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=f"The sealed successor manifest is {sha256(MANIFEST)}; its inert verifier passed 23 adversarial mutations, independently cross-checked distinct outer/inner terminal descriptors, and reported zero authority consumption, production/live state, or target starts.",
        main_error=None,
        raw_evidence=json.dumps({"verified_inputs": verified, "independent_verifier_rerun": verifier_rerun}, indent=2, sort_keys=True),
        scope="v9_static_successor_package_fresh_l2_only",
        checkpoint_path="",
        escalate_hint="Real G8/G4/G2/G1 execution remains unauthorized and requires a separate later mission.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="read-only",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    accepted = (
        raw.get("status") == "done"
        and not raw.get("next_action")
        and bool(raw.get("thread_id"))
    )
    write_once(DECISION, {
        "accepted": accepted,
        "authority_consumed": False,
        "claim_boundary": verified["claim_boundary"],
        "disposition": DISPOSITION if accepted else "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE_REJECTED_IMMUTABLE_NO_AUTHORITY_CONSUMED",
        "independent_verifier_rerun_sha256": sha256(RERUN),
        "kind": "fresh_l2_reviewer_verdict",
        "manifest_raw_sha256": sha256(MANIFEST),
        "producer_role": "reviewer",
        "raw_review": raw,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_thread_id": raw.get("thread_id", ""),
        "reviewer_status": raw.get("status", "continue"),
        "schema_version": 1,
        "target_process_starts": 0,
    })
    seal_review_root()
    if not accepted:
        print(f"V9_SUCCESSOR_FRESH_L2_REJECTED status={raw.get('status')}")
        return 2
    check_existing()
    print(f"V9_SUCCESSOR_FRESH_L2_DONE thread={raw.get('thread_id')} manifest={sha256(MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
