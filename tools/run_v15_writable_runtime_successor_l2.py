#!/usr/bin/env python3
"""Run/check one static-only Fresh-L2 review of the immutable V15 successor."""

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
ACTION_ROOT = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root"
MANIFEST = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json"
VERIFIER = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree.py"
REPORT = ACTION_ROOT / "V13_RUNTIME_NAMESPACE_FAILURE_REPORT.md"
ACCEPTANCE_SCHEMA = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json"
ACCEPTANCE_DIR = ACTION_ROOT / "review"
ACCEPTANCE = ACCEPTANCE_DIR / "FRESH_L2_STATIC_ACCEPTANCE.json"
REVIEW_ROOT = ROOT / "build/v15-writable-runtime-successor-fresh-l2-review"
REQUEST = REVIEW_ROOT / "REVIEW_REQUEST.json"
RERUN = REVIEW_ROOT / "INDEPENDENT_VERIFIER_RERUN.json"
DECISION = REVIEW_ROOT / "FRESH_L2_REVIEW.json"
POST_ACCEPTANCE_RERUN = REVIEW_ROOT / "POST_ACCEPTANCE_VERIFIER_RERUN.json"
ACTION_ID = "ace2:qk-gbfp8-base-v15:execute-once:c2dfe170:20260814T084500Z"
DISPOSITION = "V15_WRITABLE_RUNTIME_SUCCESSOR_STATIC_PACKAGE_ACCEPTED_NO_EXECUTION_AUTHORITY"
MISSION_ID = "v15-writable-runtime-successor-static-fresh-l2"
EXACT_ENV = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC", "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
RULINGS = (
    "v15.external-writable-runtime-separation",
    "v15.parent-lstat-no-symlink-mode-owner",
    "v15.primary-fallback-create-once-before-authority",
    "v15.exact-production-reader-pass",
    "v15.shellfree-exact-argv-cwd-environment",
    "v15.adversarial-reader-and-manifest-rejection",
    "v15.retired-v13-and-preserved-v14-static-failure",
    "v15.zero-target-starts-and-empty-runtime",
    "v15.static-acceptance-schema-review-binding",
    "v15.sealed-immutable-file-set",
)


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, f"expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    require(not path.exists() and not sidecar.exists(), f"refusing to overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar.write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def companion_matches(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.is_file() and sidecar.read_text(encoding="ascii").split() == [sha256(path), path.name]


def package_inputs(*, acceptance_expected: bool) -> dict[str, Any]:
    require(ACTION_ROOT.is_dir() and not ACTION_ROOT.is_symlink(), "action root absent")
    require(stat.S_IMODE(os.lstat(ACTION_ROOT).st_mode) == 0o555, "action root is not sealed")
    require(MANIFEST.is_file() and VERIFIER.is_file() and REPORT.is_file(), "required package input absent")
    require(not os.path.lexists(ACTION_ROOT / "live"), "V15 live state exists")
    require(ACCEPTANCE.exists() is acceptance_expected, "acceptance presence differs")
    package = load(MANIFEST)
    require(package.get("action_identity", {}).get("future_action_id") == ACTION_ID, "action identity differs")
    require(package.get("claim_boundary", {}).get("execution_authorized") is False, "package grants execution")
    return {
        "acceptance_expected": acceptance_expected,
        "action_id": ACTION_ID,
        "claim_boundary": "Static-only Fresh-L2 review. No transport, launcher, controller, evaluator, C02 parser, tensor, G8/G4/G2/G1, authority, credential, ledger, result, terminal, RTL, XRT, FPGA, or U280 action is permitted.",
        "manifest": {"byte_count": MANIFEST.stat().st_size, "path": str(MANIFEST), "sha256": sha256(MANIFEST)},
        "report": {"byte_count": REPORT.stat().st_size, "path": str(REPORT), "sha256": sha256(REPORT)},
        "verifier": {"byte_count": VERIFIER.stat().st_size, "path": str(VERIFIER), "sha256": sha256(VERIFIER)},
        "acceptance_schema": {"byte_count": ACCEPTANCE_SCHEMA.stat().st_size, "path": str(ACCEPTANCE_SCHEMA), "sha256": sha256(ACCEPTANCE_SCHEMA)},
    }


def rerun_verifier(*, acceptance_expected: bool) -> dict[str, Any]:
    result = subprocess.run(
        ["/usr/bin/python3", str(VERIFIER)],
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
    require(payload.get("status") == "PASS_V15_WRITABLE_RUNTIME_SUCCESSOR_INDEPENDENT_STATIC_VERIFICATION", "verifier status differs")
    require(payload.get("adversarial_cases", 0) >= 51, "verifier adversarial count differs")
    require(payload.get("target_process_starts") == 0, "verifier target starts differ")
    require((payload.get("acceptance") is not None) is acceptance_expected, "verifier acceptance observation differs")
    return {
        "acceptance_expected": acceptance_expected,
        "command": ["/usr/bin/python3", str(VERIFIER)],
        "environment": EXACT_ENV,
        "exit_code": result.returncode,
        "output": payload,
        "rerun_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def create_acceptance(decision: dict[str, Any]) -> None:
    require(not ACCEPTANCE_DIR.exists() and not ACCEPTANCE.exists(), "acceptance path already exists")
    payload = {
        "action_id": ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_fresh_l2_static_acceptance",
        "decision": "ACCEPT_STATIC_PACKAGE",
        "execution_package_file_sha256": sha256(MANIFEST),
        "independent_verifier_rerun_sha256": sha256(RERUN),
        "reviewer_decision_sha256": sha256(DECISION),
        "reviewer_role": "Fresh-L2",
        "reviewer_thread_id": decision["reviewer_thread_id"],
        "static_acceptance_grants_execution_authority": False,
        "target_process_starts": 0,
    }
    payload["acceptance_sha256"] = sha256_bytes(compact_bytes(payload))
    import jsonschema

    schema = load(ACCEPTANCE_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(payload)
    raw = compact_bytes(payload)
    os.chmod(ACTION_ROOT, 0o755)
    try:
        os.mkdir(ACCEPTANCE_DIR, 0o700)
        descriptor = os.open(ACCEPTANCE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                require(written > 0, "short acceptance write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(ACCEPTANCE, 0o444)
        os.chmod(ACCEPTANCE_DIR, 0o555)
    finally:
        os.chmod(ACTION_ROOT, 0o555)


def seal_review_root() -> None:
    for path in REVIEW_ROOT.iterdir():
        require(path.is_file() and not path.is_symlink(), f"unexpected review entry: {path}")
        os.chmod(path, 0o444)
    os.chmod(REVIEW_ROOT, 0o555)


def check_existing() -> None:
    package_inputs(acceptance_expected=True)
    require(REVIEW_ROOT.is_dir() and stat.S_IMODE(os.lstat(REVIEW_ROOT).st_mode) == 0o555, "review root is not sealed")
    expected = {
        REQUEST.name,
        REQUEST.name + ".sha256",
        RERUN.name,
        RERUN.name + ".sha256",
        DECISION.name,
        DECISION.name + ".sha256",
        POST_ACCEPTANCE_RERUN.name,
        POST_ACCEPTANCE_RERUN.name + ".sha256",
    }
    require({path.name for path in REVIEW_ROOT.iterdir()} == expected, "review file set differs")
    for path in (REQUEST, RERUN, DECISION, POST_ACCEPTANCE_RERUN):
        require(companion_matches(path), f"review sidecar differs: {path}")
    decision = load(DECISION)
    require(decision.get("accepted") is True and decision.get("reviewer_status") == "done", "review is not accepted")
    require(decision.get("disposition") == DISPOSITION, "review disposition differs")
    require(decision.get("authority_consumed") is False and decision.get("target_process_starts") == 0, "review boundary differs")
    require(bool(decision.get("reviewer_thread_id")), "reviewer thread identity absent")
    post = load(POST_ACCEPTANCE_RERUN)
    require(post.get("output", {}).get("acceptance") is not None, "post-acceptance verification absent")
    acceptance = load(ACCEPTANCE)
    require(acceptance.get("reviewer_decision_sha256") == sha256(DECISION), "acceptance review binding differs")
    require(acceptance.get("independent_verifier_rerun_sha256") == sha256(RERUN), "acceptance verifier binding differs")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("V15_WRITABLE_RUNTIME_SUCCESSOR_FRESH_L2_CHECK_PASS")
        return 0

    require(not REVIEW_ROOT.exists(), f"review root already exists: {REVIEW_ROOT}")
    verified = package_inputs(acceptance_expected=False)
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
    verifier_rerun = rerun_verifier(acceptance_expected=False)
    write_once(RERUN, verifier_rerun)

    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(backend=backend_name, runner_bin=resolve_runner_bin_setting("reviewer") or None)
    sid = os.environ.get("ARGUS_SKILL_SESSION_ID", ROOT.name)
    runner.set_usage_context(project_root=global_root() / "projects" / sid, global_root=global_root(), mission_id=MISSION_ID)
    reviewer_skill = (Path(argus_skill.__file__).resolve().parent / "verticals/chip_design/skills/reviewer/chip-design-signoff-review.md").read_text(encoding="utf-8")
    objective = "Independently Fresh-L2-review the exact immutable V15 writable-runtime read-only-preflight successor without granting or consuming execution authority."
    instruction = (
        f"Inspect {ACTION_ROOT}, {MANIFEST}, {REPORT}, {VERIFIER}, {ACCEPTANCE_SCHEMA}, the exact V9 terminal, and all bound predecessor/canonical files directly. "
        "Freshly rerun only this inert command from the project root: env -i LANG=C LC_ALL=C PYTHONHASHSEED=0 TZ=UTC PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "
        + str(VERIFIER)
        + ". Do not invoke transport, launcher, controller, evaluator, C02 parser, tensor reads, G8/G4/G2/G1, model, RTL, XRT, FPGA, or U280 paths. "
        "Confirm that the original pretty V8 package and result schema are byte-preserved, their canonical mirrors are semantically identical, and the exact production launcher's AST-extracted decode_canonical_json/read_canonical_json functions accept both exact mirrors. "
        "Confirm the package root is immutable and has no package-local live directory, while the separately named V15 runtime tree is empty, mode 0700, owner-bound, outside the package root, and contains no authority, credential, ledger, result, primary terminal, or fallback terminal. "
        "Confirm every runtime parent is checked with lstat for directory type, no symlink, exact mode, uid, and gid; production preflight performs transient O_EXCL probes in both primary and fallback terminal parents, rejects a second create, removes both probes, and completes before the first authority create. "
        "Confirm _read_accepted_package_and_result_schema is called inside _preflight before the first authority _durable_create in main, and the exact invocation is an argv vector with exact cwd/environment, os.posix_spawn, shell=false, and no /bin/bash, /bin/sh, or -c token. "
        "Adversarially inspect all 51 rejection cases, including canonical digest cross-links, runtime-root substitution, primary/fallback aliasing, mode/owner changes, binding replacement, old-action substitution, shell wrappers, acceptance-schema omission, and target-start changes. "
        "Confirm the acceptance schema requires reviewer_decision_sha256, independent_verifier_rerun_sha256, reviewer_thread_id, and target_process_starts=0 while static_acceptance_grants_execution_authority remains false. "
        "Confirm V13 remains retired after its one exit-2 attempt with no tensor/evaluator/G8/G4/G2/G1 run, and the failed immutable V14 pre-review candidate remains unaccepted and authority-free with its verifier-gap record bound into V15. "
        "Cover these exact rulings in the reason: "
        + ", ".join(RULINGS)
        + f". Return done only if the exact static V15 package deserves disposition {DISPOSITION}. A done verdict does not authorize execution and does not advance research/PIPELINE_STATE.json."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=f"V15 manifest SHA-256 {sha256(MANIFEST)}; the bound inert verifier rejects 51 adversarial cases, proves both terminal parents writable/create-once before authority, preserves an empty runtime, validates exact canonical reads and shell-free argv/cwd/environment, and reports zero target starts.",
        main_error=None,
        raw_evidence=json.dumps({"verified_inputs": verified, "independent_verifier_rerun": verifier_rerun}, indent=2, sort_keys=True),
        scope="v15_writable_runtime_static_successor_fresh_l2_only",
        checkpoint_path="",
        escalate_hint="Any V15 transport/launcher execution remains separately operator-authorized and exactly once.",
        preselected_skill_block=reviewer_skill,
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort("ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="workspace-write",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    accepted = raw.get("status") == "done" and not raw.get("next_action") and bool(raw.get("thread_id"))
    decision = {
        "accepted": accepted,
        "authority_consumed": False,
        "claim_boundary": verified["claim_boundary"],
        "disposition": DISPOSITION if accepted else "V15_WRITABLE_RUNTIME_SUCCESSOR_STATIC_PACKAGE_REJECTED_NO_AUTHORITY",
        "independent_verifier_rerun_sha256": sha256(RERUN),
        "kind": "fresh_l2_reviewer_verdict",
        "manifest_raw_sha256": sha256(MANIFEST),
        "producer_role": "reviewer",
        "raw_review": raw,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": raw.get("status", "continue"),
        "reviewer_thread_id": raw.get("thread_id", ""),
        "schema_version": 1,
        "target_process_starts": 0,
    }
    write_once(DECISION, decision)
    if not accepted:
        seal_review_root()
        print(f"V15_WRITABLE_RUNTIME_SUCCESSOR_FRESH_L2_REJECTED status={raw.get('status')}")
        return 2

    create_acceptance(decision)
    post = rerun_verifier(acceptance_expected=True)
    write_once(POST_ACCEPTANCE_RERUN, post)
    seal_review_root()
    check_existing()
    print(f"V15_WRITABLE_RUNTIME_SUCCESSOR_FRESH_L2_DONE thread={raw.get('thread_id')} manifest={sha256(MANIFEST)} acceptance={sha256(ACCEPTANCE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
