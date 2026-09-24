#!/usr/bin/env python3
"""Run one marker-free independent L2 review of the repaired V5 package."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus_skill.adapters.agent_cli_backend import AgentCliBackend
from argus_skill.core.knobs import (
    resolve_role_backend,
    resolve_role_model,
    resolve_role_reasoning_effort,
    resolve_runner_bin_setting,
)
from argus_skill.core.paths import global_root
from argus_skill.reviewer import Reviewer, ReviewerConfig


ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V5_EXECUTION_PACKAGE_CONTRACT.json"
MANIFEST = OFFLINE / "v5-execution-package-manifest.json"
SELF_TEST = OFFLINE / "v5-execution-package-self-test.json"
REQUEST = OFFLINE / "execution-package-l2-review-request.json"
DECISION = OFFLINE / "execution-package-l2-decision.json"
ACCEPTANCE = OFFLINE / "execution-package-l2-acceptance.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
MISSION_ID = "qwen25-lora-v5-execution-package-repair-l2-v1"
RULINGS = (
    "v5.direct-train-whole-root-interlock",
    "v5.official-dev-predecessor-chain",
    "v5.marker-free-freeze",
)
SOURCE_FILES = (
    ROOT / "pilot/qwen25_05b_bf16_lora_product_v5_runner/runtime.py",
    ROOT / "pilot/qwen25_05b_bf16_lora_product_v5_runner/train.py",
    ROOT / "pilot/qwen25_05b_bf16_lora_product_v5_runner/evaluate_dev.py",
    ROOT / "pilot/qwen25_05b_bf16_lora_product_v5_runner/self_test.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact is absent: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    if not companion.is_file():
        return False
    fields = companion.read_text(encoding="ascii").split()
    return len(fields) == 2 and fields[0] == sha256(path) and fields[1] == path.name


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )


def session_id() -> str:
    configured = os.environ.get("ARGUS_SKILL_SESSION_ID")
    if configured:
        return configured
    session = ROOT / ".ace2-session.json"
    if session.is_file():
        value = load(session).get("sid")
        if isinstance(value, str) and value:
            return value
    return ROOT.name


def reviewer_supports(reason: str, identifier: str) -> bool:
    normalized = reason.lower().replace("`", "")
    return f"{identifier}: supported" in normalized


def verified_inputs() -> dict[str, Any]:
    require(load(PIPELINE).get("current_stage") == "specification", "current stage is not specification")
    require(not RUN_ROOT.exists(), "official V5 namespace exists")
    require(
        not list(ROOT.glob("build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5/attempt-*/ATTEMPT_CONSUMPTION_MARKER.json")),
        "V5 attempt marker exists",
    )
    for path in (CONTRACT, MANIFEST, SELF_TEST, REQUEST):
        require(companion_matches(path), f"checksum companion differs: {path}")
    manifest = load(MANIFEST)
    self_test = load(SELF_TEST)
    request = load(REQUEST)
    require(self_test.get("status") == "PASS", "V5 self-test is not PASS")
    require(self_test.get("check_count") == 25, "V5 self-test check count differs")
    require(all(self_test.get("checks", {}).values()), "one or more V5 self-test checks failed")
    require(self_test.get("official_namespace_exists") is False, "V5 self-test reports a namespace")
    require(request.get("status") == "PENDING_INDEPENDENT_L2", "V5 review request status differs")
    require(request.get("official_namespace_exists") is False, "V5 review request reports a namespace")
    require(request.get("execution_tree_sha256") == manifest.get("tree_sha256"), "request tree differs")
    require(
        request["exact_package"]["execution_manifest"]["sha256"] == sha256(MANIFEST),
        "request manifest binding differs",
    )
    require(
        request["exact_package"]["execution_self_test"]["sha256"] == sha256(SELF_TEST),
        "request self-test binding differs",
    )
    return {
        "claim_boundary": (
            "Independent exact-package review only; no V5 namespace, marker, training, "
            "official dev/holdout access, merge, quantization, RTL, PPA, FPGA, or product claim."
        ),
        "contract": artifact(CONTRACT),
        "execution_manifest": artifact(MANIFEST),
        "execution_self_test": artifact(SELF_TEST),
        "pipeline_state": artifact(PIPELINE),
        "review_request": artifact(REQUEST),
        "source_files": [artifact(path) for path in SOURCE_FILES],
        "tree_sha256": manifest["tree_sha256"],
    }


def check_existing() -> None:
    verified = verified_inputs()
    require(companion_matches(DECISION), "V5 L2 decision checksum differs")
    require(companion_matches(ACCEPTANCE), "V5 L2 acceptance checksum differs")
    decision = load(DECISION)
    acceptance = load(ACCEPTANCE)
    require(decision.get("reviewer_status") == "done", "V5 L2 decision is not done")
    require(acceptance.get("status") == "ACCEPTED_V5_EXACT_PACKAGE_NO_ATTEMPT", "V5 L2 status differs")
    require(acceptance.get("accepted") is True, "V5 L2 acceptance is false")
    require(acceptance.get("attempt_authority_granted") is False, "V5 L2 grants attempt authority")
    require(acceptance.get("independent_from_constructor") is True, "V5 L2 is not independent")
    require(acceptance.get("execution_tree_sha256") == verified["tree_sha256"], "V5 L2 tree differs")
    require(acceptance.get("execution_manifest_sha256") == sha256(MANIFEST), "V5 L2 manifest differs")
    require(acceptance.get("contract_sha256") == sha256(CONTRACT), "V5 L2 contract differs")
    require(acceptance.get("reviewer_decision_sha256") == sha256(DECISION), "V5 L2 decision binding differs")


def main() -> int:
    if "--check" in sys.argv[1:]:
        check_existing()
        print("ACE2_LORA_V5_EXECUTION_PACKAGE_L2_CHECK_PASS")
        return 0

    require(not DECISION.exists(), "V5 repaired-package L2 decision already exists")
    require(not ACCEPTANCE.exists(), "V5 repaired-package L2 acceptance already exists")
    verified = verified_inputs()
    backend_name = resolve_role_backend("reviewer")
    runner = AgentCliBackend(
        backend=backend_name,
        runner_bin=resolve_runner_bin_setting("reviewer") or None,
    )
    sid = session_id()
    runner.set_usage_context(
        project_root=global_root() / "projects" / sid,
        global_root=global_root(),
        mission_id=MISSION_ID,
    )
    objective = (
        "Independently review the exact marker-free ACE-2 Qwen2.5-0.5B V5 execution package "
        "after the two bounded fail-closed repairs. Decide whether direct train invocation now "
        "rejects any pre-existing official V5 run root, whether evaluate_dev authenticates the "
        "attempt marker and complete predecessor hash chain before official-dev access, and "
        "whether the regenerated package remains marker-free and self-consistent."
    )
    instruction = (
        "Inspect the exact review request, manifest, self-test result, runtime.py, train.py, "
        "evaluate_dev.py, self_test.py, the archived attempt-0001 package, and current pipeline "
        "state directly. You may run only read-only hashes, ruff/compile checks, "
        "self_test.py --check, and the execution-package freeze tool --check. Do not run the "
        "self-test in write mode, create a namespace or marker, train, open official dev or "
        "holdout bytes, merge, quantize, edit SPEC/RTL, synthesize, or perform U280 work. "
        "Return done only if the exact package closes both prior Reviewer defects and all 25 "
        "recorded checks are credible. In the reason, state each accepted ruling exactly as "
        "'v5.direct-train-whole-root-interlock: supported', "
        "'v5.official-dev-predecessor-chain: supported', and "
        "'v5.marker-free-freeze: supported'. A done verdict grants no attempt authority."
    )
    review = Reviewer(runner, memory_maintenance_enabled=False).evaluate(
        objective=objective,
        original_objective=objective,
        operator_messages=[instruction],
        planner_review_instruction=instruction,
        round_index=1,
        round_max=1,
        session_id=sid,
        main_summary=(
            "Planner repaired only the two authorized direct-worker interlocks, added negative "
            "regressions, archived the first reviewed package, and regenerated the marker-free "
            f"request for tree {verified['tree_sha256']}."
        ),
        main_error=None,
        raw_evidence=json.dumps(verified, indent=2, sort_keys=True),
        scope="v5_exact_execution_package_marker_free_l2_only",
        checkpoint_path="",
        escalate_hint=(
            "This bounded review never grants attempt authority and never advances the "
            "Manager-owned specification stage."
        ),
        config=ReviewerConfig(
            model=resolve_role_model("reviewer", role_env="ARGUS_SKILL_REVIEWER_MODEL"),
            reasoning_effort=resolve_role_reasoning_effort(
                "ARGUS_SKILL_REVIEWER_REASONING_EFFORT", default="high"
            ),
            skip_git_repo_check=True,
            full_auto=True,
            dangerous_yolo=False,
            sandbox_mode="workspace-write",
            isolate_workdir=False,
            working_dir=str(ROOT),
        ),
    )
    raw = asdict(review)
    status = raw.get("status", "continue")
    reason = raw.get("reason", "")
    if status == "done":
        require(all(reviewer_supports(reason, item) for item in RULINGS), "Reviewer reason lacks exact rulings")
        require(not raw.get("next_action"), "Reviewer returned done with a next action")
        require(bool(raw.get("thread_id")), "Reviewer execution identity is missing")
    decision = {
        "claim_boundary": verified["claim_boundary"],
        "execution_tree_sha256": verified["tree_sha256"],
        "reason": reason,
        "reviewed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_status": status,
        "reviewer_thread_id": raw.get("thread_id", ""),
        "schema_version": 1,
        "verified_evidence": verified,
        "raw_decision": raw,
    }
    write_json(DECISION, decision)
    if status != "done":
        print(f"ACE2_LORA_V5_EXECUTION_PACKAGE_L2_REPLAN status={status}")
        return 2
    acceptance = {
        "accepted": True,
        "attempt_authority_granted": False,
        "claim_boundary": verified["claim_boundary"],
        "contract_sha256": sha256(CONTRACT),
        "execution_manifest_sha256": sha256(MANIFEST),
        "execution_tree_sha256": verified["tree_sha256"],
        "independent_from_constructor": True,
        "reviewed_at_utc": decision["reviewed_at_utc"],
        "reviewer_decision_sha256": sha256(DECISION),
        "reviewer_status": "done",
        "reviewer_thread_id": raw["thread_id"],
        "schema_version": 1,
        "status": "ACCEPTED_V5_EXACT_PACKAGE_NO_ATTEMPT",
    }
    write_json(ACCEPTANCE, acceptance)
    check_existing()
    print(
        "ACE2_LORA_V5_EXECUTION_PACKAGE_L2_PASS "
        f"tree={verified['tree_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
