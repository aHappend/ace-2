#!/usr/bin/env python3
"""Bind the sealed S5 post-review handoff into project-local provenance."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus_skill.core.paths import global_root


ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = "f68cdf8ac2b5"
OUTPUT = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-full-finetune-successor-s5-postreview-fresh-review-20260808.json"
)
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S5_EXECUTION_PACKAGE_CONTRACT.json"
PACKAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s5/package-audit.json"
STAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s5/stage-closure-audit.json"
OFFLINE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5"
MANIFEST = OFFLINE / "s5-execution-package-manifest.json"
SELF_TEST = OFFLINE / "s5-execution-package-self-test.json"
ACCEPTANCE = OFFLINE / "execution-package-l2-acceptance.json"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-attempt-operator-authority.json"
SUBMISSION = ROOT / "build/bf16-full-finetune-successor-s5"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s5"

EXPECTED = {
    "contract": "6b9425f027bcc50f399fdded1ac91147fbdde91c9a041e10d1e00e6b3c9c0a54",
    "package_audit": "52328fd8507eba393e7cda1c0ab64cf1a41734c518234bb1ca8caab978ed7711",
    "manifest": "04fc24205d3a639d70e83bc37d6d433fa06fa398764015f6f05bac45b9169348",
    "self_test": "1131ffd2140778953fbdf0812a6d7e9fda4b2918ba027e487e4af63551b58f79",
    "acceptance": "ec3a08d3228afbcb01b895f9d8b488674971f75cfc3861d0ac17758c107b4241",
    "stage_audit": "3679639aae5d2267bfae86889f583da0f8fc72e12046050dee7e91f3eeb56bd5",
    "mission": "1ddb4c85ba31a8e2ba5d07b69ec033dc974247308fb66602a3b303e048321c30",
    "round": "1b48fd705b01805a32c178cc50be15333002ce77265a0b633afe9e19764b4ad9",
    "checkpoint": "74881136a51ec80326a1f2bb02cab8265302dffc414b026e27711f1ac1d6cda0",
}


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [
        sha256(path),
        path.name,
    ]


def artifact(path: Path, **extra: Any) -> dict[str, Any]:
    value = {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }
    value.update(extra)
    return value


def session_id() -> str:
    session = load(ROOT / ".ace2-session.json")
    value = session.get("sid")
    require(isinstance(value, str) and value, "active Argus session id is absent")
    return value


def sealed_paths() -> tuple[Path, Path, Path]:
    root = global_root() / "projects" / session_id() / "handoffs" / MISSION_ID
    return root / "mission.json", root / "round-0002.json", root / "CHECKPOINT.md"


def expected_record() -> dict[str, Any]:
    mission_path, round_path, checkpoint_path = sealed_paths()
    for label, path in (
        ("mission", mission_path),
        ("round", round_path),
        ("checkpoint", checkpoint_path),
    ):
        require(path.is_file(), f"sealed reviewer {label} is absent")
        require(sha256(path) == EXPECTED[label], f"sealed reviewer {label} hash differs")

    mission = load(mission_path)
    reviewed = load(round_path)
    checkpoint = checkpoint_path.read_text(encoding="utf-8")
    require(mission.get("mission_id") == MISSION_ID, "sealed mission id differs")
    require(mission.get("stage") == "specification", "sealed mission stage differs")
    require(reviewed.get("producer_role") == "reviewer", "sealed producer is not Reviewer")
    require(reviewed.get("round") == 2, "sealed review round differs")
    review = reviewed.get("review", {})
    require(review.get("status") == "done", "sealed Fresh Review is not done")
    reason = str(review.get("reason", ""))
    for fragment in (
        "69/69 omission regressions",
        "all four runtime checksum-companion omissions",
        "relocated post-review readiness",
        "no authority, intent, marker, namespace, or execution state exists",
    ):
        require(fragment in reason, f"sealed Fresh Review lacks ruling: {fragment}")

    local_artifacts = {
        "contract": CONTRACT,
        "package_audit": PACKAGE_AUDIT,
        "manifest": MANIFEST,
        "self_test": SELF_TEST,
        "acceptance": ACCEPTANCE,
        "stage_audit": STAGE_AUDIT,
    }
    for label, path in local_artifacts.items():
        require(path.is_file(), f"S5 artifact is absent: {path}")
        require(sha256(path) == EXPECTED[label], f"S5 artifact hash differs: {label}")
        require(companion_matches(path), f"S5 checksum companion differs: {label}")

    package = load(PACKAGE_AUDIT)
    stage = load(STAGE_AUDIT)
    acceptance = load(ACCEPTANCE)
    manifest = load(MANIFEST)
    self_test = load(SELF_TEST)
    require(package.get("status") == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S5_PACKAGE", "S5 package audit status differs")
    require(package.get("check_count") == 92 and not package.get("failed_checks"), "S5 package audit count differs")
    require(stage.get("status") == "PASS_FINAL_POSTREVIEW_MARKER_FREE_S5_TRANSITIVE_STAGE_CLOSURE", "S5 final stage status differs")
    require(stage.get("closure_file_count") == 69, "S5 closure count differs")
    require(stage.get("omission_regression_count") == 69, "S5 omission count differs")
    require(len(stage.get("runtime_sidecar_omission_rejections", {})) == 4, "S5 runtime sidecar omission count differs")
    require(stage.get("relocated_readiness", {}).get("status") == "PASS_POSTREVIEW_RELOCATED_AUTHORITY_READY_NO_AUTHORITY", "S5 relocated readiness differs")
    require(acceptance.get("status") == "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S5_PACKAGE_NO_ATTEMPT", "S5 L2 status differs")
    require(acceptance.get("reviewer_status") == "done" and acceptance.get("accepted") is True, "S5 L2 acceptance differs")
    require(acceptance.get("mission_id") == MISSION_ID, "S5 L2 mission binding differs")
    require(manifest.get("tree_sha256") == "54602351773e85fc28e0347aacbee988cc787157306f01913bd74ff190f79a0a", "S5 execution tree differs")
    require(self_test.get("status") == "PASS" and self_test.get("check_count") == 40, "S5 self-test differs")
    for fragment in (
        EXPECTED["acceptance"],
        EXPECTED["package_audit"],
        EXPECTED["stage_audit"],
        "PASS_POSTREVIEW_RELOCATED_AUTHORITY_READY_NO_AUTHORITY",
        "All 30 stage symlinks are relative and resolve",
    ):
        require(fragment in checkpoint, f"sealed checkpoint lacks binding: {fragment}")

    for path in (
        AUTHORITY,
        RUN_ROOT,
        SUBMISSION / "backend-submission-intent.json",
        SUBMISSION / "backend-submission-result.json",
        SUBMISSION / "backend-submission.raw.txt",
    ):
        require(not path.exists(), f"S5 consuming state exists: {path}")

    reviewed_at = datetime.fromtimestamp(float(reviewed["created_at"]), UTC).isoformat().replace("+00:00", "Z")
    return {
        "accepted_evidence": {
            "contract": artifact(CONTRACT),
            "package_audit": artifact(PACKAGE_AUDIT, status=package["status"], check_count=package["check_count"]),
            "execution_manifest": artifact(MANIFEST, tree_sha256=manifest["tree_sha256"]),
            "execution_self_test": artifact(SELF_TEST, status=self_test["status"], check_count=self_test["check_count"]),
            "fresh_l2_acceptance": artifact(ACCEPTANCE, status=acceptance["status"]),
            "postreview_stage_audit": artifact(
                STAGE_AUDIT,
                status=stage["status"],
                closure_file_count=stage["closure_file_count"],
                omission_regression_count=stage["omission_regression_count"],
                runtime_sidecar_omission_count=len(stage["runtime_sidecar_omission_rejections"]),
                relocated_readiness_status=stage["relocated_readiness"]["status"],
            ),
            "package_identity_sha256": package["package_identity_sha256"],
        },
        "claim_boundary": "Project-local provenance mirror of the sealed post-review S5 package disposition. It grants no attempt authority and makes no model, evaluator, checkpoint, quality, W4A8, RTL, PPA, FPGA, U280, or completion claim.",
        "kind": "project_local_provenance_mirror_of_sealed_reviewer_handoff",
        "mission_id": MISSION_ID,
        "producer_role": "reviewer",
        "review": {
            "reason": reason,
            "reviewed_at_utc": reviewed_at,
            "status": "done",
        },
        "round": 2,
        "schema_version": 1,
        "sealed_checkpoint_sha256": EXPECTED["checkpoint"],
        "sealed_mission_sha256": EXPECTED["mission"],
        "sealed_round_sha256": EXPECTED["round"],
        "state": {
            "attempt_authority_exists": False,
            "backend_experiment_or_job_exists": False,
            "evaluator_execution": False,
            "model_execution": False,
            "official_namespace_exists": False,
            "quality_conclusion": None,
            "raw_submission_output_exists": False,
            "submission_intent_exists": False,
            "submission_result_exists": False,
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )


def main() -> int:
    expected = expected_record()
    if OUTPUT.exists():
        require(companion_matches(OUTPUT), "S5 project-local review mirror sidecar differs")
        require(load(OUTPUT) == expected, "S5 project-local review mirror content differs")
        print("ACE2_S5_POSTREVIEW_BINDING_CHECK_PASS")
        return 0
    write_json(OUTPUT, expected)
    print(f"ACE2_S5_POSTREVIEW_BINDING_CREATED sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
