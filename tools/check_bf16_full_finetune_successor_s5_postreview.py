#!/usr/bin/env python3
"""Rerun S5 post-review closure checks without rewriting accepted evidence."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/execution/qwen25_bf16_full_finetune_successor_s5_20260808"
sys.path.insert(0, str(HERE))
audit_stage = importlib.import_module("audit_stage")
stage_builder = importlib.import_module("prepare_stage")

EVIDENCE = ROOT / "build/bf16-full-finetune-successor-s5/stage-closure-audit.json"
REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-postreview-fresh-review-20260808.json"


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def stable_hygiene(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "observed_at_utc"}


def main() -> int:
    require(load(ROOT / "research/PIPELINE_STATE.json").get("current_stage") == "specification", "pipeline stage differs")
    require(stage_builder.companion_matches(EVIDENCE), "accepted S5 stage audit sidecar differs")
    require(stage_builder.companion_matches(REVIEW), "S5 Fresh Review mirror sidecar differs")
    accepted = load(EVIDENCE)
    review = load(REVIEW)
    require(review.get("review", {}).get("status") == "done", "S5 Fresh Review mirror is not done")
    require(accepted.get("status") == "PASS_FINAL_POSTREVIEW_MARKER_FREE_S5_TRANSITIVE_STAGE_CLOSURE", "accepted S5 stage status differs")

    contract = stage_builder.load(stage_builder.CONTRACT)
    freeze = stage_builder.load(stage_builder.FREEZE)
    manifest = stage_builder.load(stage_builder.MANIFEST)
    bindings = stage_builder.required_repo_bindings(contract, freeze, manifest)
    repo = audit_stage.STAGE / "repo"
    stage_builder.verify_repo_bindings(repo, bindings)
    required_sidecars = stage_builder.required_runtime_companions(contract)
    require(required_sidecars <= bindings.keys(), "runtime sidecars are absent from closure")

    hygiene = audit_stage.scan_hygiene(repo, bindings)
    require(hygiene.get("status") == "PASS_COMPLETE_CLOSURE_HYGIENE", "S5 closure hygiene failed")
    require(stable_hygiene(hygiene) == stable_hygiene(accepted["hygiene"]), "S5 closure hygiene evidence differs")

    omissions: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ace2-s5-check-omission-") as directory:
        omission_repo = Path(directory) / "repo"
        for relative in bindings:
            audit_stage.copy_file(repo / relative, omission_repo / relative)
        stage_builder.verify_repo_bindings(omission_repo, bindings)
        for relative in bindings:
            omissions[relative] = audit_stage.capture_missing(omission_repo, bindings, relative)
        stage_builder.verify_repo_bindings(omission_repo, bindings)
    require(omissions == accepted["omission_rejections"], "S5 closure omission evidence differs")

    with tempfile.TemporaryDirectory(prefix="ace2-s5-check-relocated-") as directory:
        relocated = Path(directory) / "assembled-stage"
        shutil.copytree(audit_stage.STAGE, relocated, symlinks=True, copy_function=shutil.copy2)
        relocated_repo = relocated / "repo"
        require(not relocated_repo.resolve().is_relative_to(ROOT.resolve()), "relocated S5 root remains inside repository")
        process, readiness = audit_stage.run_relocated_check(relocated, contract, postreview=True)
        require(process.returncode == 0, f"S5 relocated readiness failed: {process.stdout}")
        require(readiness.get("status") == "PASS_POSTREVIEW_RELOCATED_AUTHORITY_READY_NO_AUTHORITY", "S5 relocated readiness status differs")
        require(readiness.get("first_failed_gate") is None, "S5 relocated readiness reports a failed gate")
        require(Path(readiness.get("audited_root", "")).resolve() == relocated_repo.resolve(), "S5 readiness audited wrong root")
        require(readiness.get("closure_file_count") == len(bindings), "S5 relocated closure count differs")

        sidecar_omissions: dict[str, dict[str, Any]] = {}
        for relative in sorted(required_sidecars):
            target = relocated_repo / relative
            target.unlink()
            negative_process, negative = audit_stage.run_relocated_check(relocated, contract, postreview=True)
            require(negative_process.returncode == 2, f"S5 sidecar omission unexpectedly passed: {relative}")
            errors = json.dumps(negative.get("errors", {}), sort_keys=True)
            require(relative in errors, f"S5 sidecar omission did not name path: {relative}")
            sidecar_omissions[relative] = {
                "errors": negative.get("errors", {}),
                "first_failed_gate": negative.get("first_failed_gate"),
                "status": negative.get("status"),
            }
            shutil.copy2(repo / relative, target)
            audit_stage.clear_readiness(relocated_repo)
    require(sidecar_omissions == accepted["runtime_sidecar_omission_rejections"], "S5 runtime-sidecar omission evidence differs")

    for path in (
        audit_stage.ROOT / audit_stage.AUTHORITY_RELATIVE,
        audit_stage.ROOT / audit_stage.RUN_ROOT_RELATIVE,
        ROOT / "build/bf16-full-finetune-successor-s5/backend-submission-intent.json",
        ROOT / "build/bf16-full-finetune-successor-s5/backend-submission-result.json",
        ROOT / "build/bf16-full-finetune-successor-s5/backend-submission.raw.txt",
    ):
        require(not path.exists(), f"S5 consuming state exists: {path}")

    print(json.dumps({
        "closure_file_count": len(bindings),
        "fresh_review_status": review["review"]["status"],
        "omission_regression_count": len(omissions),
        "relocated_readiness_status": readiness["status"],
        "runtime_sidecar_omission_count": len(sidecar_omissions),
        "status": "ACE2_S5_POSTREVIEW_NONCONSUMING_CHECK_PASS",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
