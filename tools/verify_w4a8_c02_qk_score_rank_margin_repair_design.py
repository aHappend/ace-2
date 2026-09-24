#!/usr/bin/env python3
"""Fail-closed static preflight for the additive c02 repair-design task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "design/W4A8_C02_QK_SCORE_RANK_MARGIN_REPAIR_DESIGN_V1_TASK.json"
DEFAULT_OUTPUT = (
    ROOT
    / "evidence/diagnostics/w4a8-c02-qk-score-rank-margin-repair-design-v1-preflight"
)
EXPECTED_ALTERNATIVES = [
    "qk_s4_residual_cross_term_scale32_v1",
    "qk_group8_diagonal_scale32_rebalance_v1",
]
EXPECTED_INVARIANT = "QK_SCORE_RANK_MARGIN_PRESERVATION_BEFORE_SOFTMAX"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def resolve_bound_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def iter_bindings(value: Any) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256"}:
            bindings.append({"path": str(value["path"]), "sha256": str(value["sha256"])})
        for child in value.values():
            bindings.extend(iter_bindings(child))
    elif isinstance(value, list):
        for child in value:
            bindings.extend(iter_bindings(child))
    return bindings


def validate_task(task: dict[str, Any]) -> list[dict[str, str]]:
    require(task.get("schema_version") == 1, "task schema differs")
    require(task.get("stage") == "rtl", "task stage differs")
    require(task.get("scope") == "bounded", "task scope differs")
    require(
        task.get("status") == "ready_for_additive_non_scoring_repair_design",
        "task status differs",
    )
    require(
        task["shared_runtime_invariant"]["id"] == EXPECTED_INVARIANT,
        "shared runtime invariant differs",
    )
    require(
        [item["alternative_id"] for item in task["alternatives"]]
        == EXPECTED_ALTERNATIVES,
        "alternative set or order differs",
    )
    require(
        task["claim_boundary"]["candidate_package_created"] is False,
        "task claims a candidate package",
    )
    require(
        task["claim_boundary"]["continuation_text_generated"] is False,
        "task claims continuation generation",
    )
    require(
        task["claim_boundary"]["stage1_complete"] is False,
        "task claims Stage 1 completion",
    )
    isolation = task["artifact_isolation"]
    output_parts = Path(isolation["allowed_output_root"]).parts
    require(
        not {"candidate", "candidates", "attempt", "attempts"}.intersection(output_parts),
        "repair-design output root enters a candidate/attempt namespace",
    )
    require(isolation["output_root_must_be_absent_at_start"] is True, "output isolation is open")
    require(
        isolation["immutable_input_verification_required_before_model_load"] is True,
        "pre-load hash verification is not mandatory",
    )
    fail_closed = task["acceptance"]["fail_closed_outcomes"]
    require(
        set(fail_closed.values())
        == {
            "PRE_CANDIDATE_NO_GO",
            "PASS_PRE_CANDIDATE_REPAIR_DESIGN",
            "PRE_CANDIDATE_NO_EXECUTION",
        },
        "fail-closed outcome set differs",
    )
    prohibited = "\n".join(task["prohibited_work"])
    for phrase in (
        "modify or replay c01",
        "create, execute, score",
        "generate continuation text",
        "modify RTL",
        "research/PIPELINE_STATE.json",
        "run synthesis",
        "advance the Manager-owned stage",
    ):
        require(phrase in prohibited, f"missing prohibition: {phrase}")
    return iter_bindings(task["frozen_inputs"])


def verify_binding(binding: dict[str, str]) -> dict[str, Any]:
    path = resolve_bound_path(binding["path"])
    require(path.is_file(), f"bound input missing: {path}")
    observed = sha256_file(path)
    require(observed == binding["sha256"], f"bound input drifted: {path}")
    return {
        "path": binding["path"],
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def verify_semantics(task: dict[str, Any]) -> dict[str, Any]:
    pipeline = load_json(ROOT / "research/PIPELINE_STATE.json")
    require(pipeline.get("current_stage") == "rtl", "Manager-owned current stage is not rtl")

    fresh = load_json(
        resolve_bound_path(task["frozen_inputs"]["fresh_l2_acceptance"]["path"])
    )
    review = fresh.get("review", {})
    require(review.get("status") == "done", "Fresh-L2 did not return done")
    reason = str(review.get("reason", ""))
    require("Fresh L2 accepted" in reason, "Fresh-L2 acceptance text differs")
    require("byte-identical" in reason, "Fresh-L2 did not confirm byte identity")

    c02 = load_json(
        ROOT / task["frozen_inputs"]["c02_attention_substage"]["aggregate_result"]["path"]
    )
    shared = c02.get("shared_classification", {})
    require(shared.get("status") == "PASS_SHARED_PRIMITIVE_LOCALIZED", "c02 shared localization differs")
    require(shared.get("shared_repair_invariant") == EXPECTED_INVARIANT, "c02 invariant differs")
    require(shared.get("score_perturbation_is_material_for_both_models") is True, "score perturbation is not material for both models")
    require(shared.get("softmax_realization_is_material_for_both_models") is False, "fixed softmax became material")

    for alias in ("base", "checkpoint-176"):
        model = c02["models"][alias]
        require(model["status"] == "PASS_ATTENTION_SUBSTAGE_LOCALIZED_NON_SCORING", f"c02 model status differs: {alias}")

    return {
        "current_stage": pipeline["current_stage"],
        "fresh_l2_status": review["status"],
        "shared_repair_invariant": shared["shared_repair_invariant"],
    }


def build_result(task: dict[str, Any]) -> dict[str, Any]:
    bindings = validate_task(task)
    verified = [verify_binding(binding) for binding in bindings]
    semantics = verify_semantics(task)
    return {
        "artifact_kind": "w4a8_c02_qk_score_rank_margin_repair_design_static_preflight",
        "bindings": verified,
        "claim_boundary": {
            "candidate_package_created": False,
            "continuation_generation_performed": False,
            "model_execution_performed": False,
            "rtl_or_spec_changed": False,
            "score_or_quality_metric_executed": False,
            "stage_advanced": False,
        },
        "schema_version": 1,
        "semantics": semantics,
        "status": "PASS_REPAIR_DESIGN_STATIC_PREFLIGHT",
        "task": {
            "path": TASK_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(TASK_PATH),
        },
    }


def write_result(output_root: Path, result: dict[str, Any]) -> None:
    require(ROOT.resolve() in output_root.resolve().parents, "output must be repository-local")
    require(not output_root.exists(), f"preflight output already exists: {output_root}")
    output_root.mkdir(parents=True)
    result_path = output_root / "preflight.json"
    result_path.write_bytes(canonical_bytes(result))
    sums = f"{sha256_file(result_path)}  preflight.json\n"
    (output_root / "SHA256SUMS").write_text(sums, encoding="ascii")


def verify_existing(output_root: Path) -> None:
    sums = output_root / "SHA256SUMS"
    require(sums.is_file(), "preflight SHA256SUMS is missing")
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        member = output_root / name
        require(member.is_file(), f"preflight member missing: {name}")
        require(sha256_file(member) == digest, f"preflight member drifted: {name}")
    stored = load_json(output_root / "preflight.json")
    current = build_result(load_json(TASK_PATH))
    require(stored == current, "stored preflight no longer matches live bound inputs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if args.verify_existing:
        verify_existing(output_root)
        print("PASS_REPAIR_DESIGN_STATIC_PREFLIGHT verified_existing=true")
        return 0
    result = build_result(load_json(TASK_PATH))
    write_result(output_root, result)
    print(
        "PASS_REPAIR_DESIGN_STATIC_PREFLIGHT "
        f"bindings={len(result['bindings'])} task_sha256={result['task']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
