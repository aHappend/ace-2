#!/usr/bin/env python3
"""Seal the bounded ACE-2 numerical-repair lane comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("wikitext2", "c4_en_512")
MODEL_SOURCE = ROOT / "tools" / "ace2_full_model_fixed_point.py"
LOCALIZATION_CLASSIFICATION = (
    "diagnostic_focused_layer0_paired_localization_not_acceptance_evidence"
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
    }


def require_artifact(binding: dict[str, Any], name: str) -> dict[str, Any]:
    path = ROOT / binding["path"]
    observed = artifact(path)
    if binding != observed:
        raise RuntimeError(f"{name} artifact binding is stale")
    return observed


def source_artifact(result: dict[str, Any], relative: str) -> dict[str, Any]:
    matches = [
        item
        for item in result.get("artifacts", {}).get("sources", [])
        if item.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"result does not uniquely bind source {relative}")
    return matches[0]


def verify_hash_manifest(path: Path) -> list[dict[str, Any]]:
    verified = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, separator, name = line.partition("  ")
        if not separator or len(expected) != 64 or Path(name).name != name:
            raise RuntimeError(f"malformed localization hash manifest line: {line!r}")
        target = path.parent / name
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"localization hash manifest is stale for {name}")
        verified.append(artifact(target))
    if {item["path"] for item in verified} != {
        str((path.parent / "results.json").relative_to(ROOT)),
        str((path.parent / "run_contract.json").relative_to(ROOT)),
    }:
        raise RuntimeError("localization hash manifest has an unexpected file set")
    return verified


def verify_localization_bindings(
    localization: dict[str, Any],
    baseline: dict[str, Any],
    lane_a: dict[str, Any],
    lane_b: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:
    if localization.get("classification") != LOCALIZATION_CLASSIFICATION:
        raise RuntimeError("localization classification is not focused layer-0 paired evidence")
    if localization.get("input_observations") != baseline["input_observations"]:
        raise RuntimeError("localization did not reuse the frozen paired input")
    bindings = localization.get("bindings", {})
    if bindings.get("frozen_inputs") != baseline["input_observations"]:
        raise RuntimeError("localization frozen-input binding is stale")
    if bindings.get("frozen_baseline") != artifact(baseline_path):
        raise RuntimeError("localization baseline artifact binding is stale")

    accepted_rtl_hash = baseline["artifacts"]["accepted_rtl"]["candidate_rtl_hash"]
    if bindings.get("accepted_rtl_hash") != accepted_rtl_hash:
        raise RuntimeError("localization accepted RTL binding is stale")
    localization_rtl = localization.get("artifacts", {}).get("accepted_rtl", {})
    if localization_rtl.get("candidate_rtl_hash") != accepted_rtl_hash:
        raise RuntimeError("localization accepted RTL artifact is stale")

    current_model = artifact(MODEL_SOURCE)
    if bindings.get("fixed_point_model") != current_model:
        raise RuntimeError("localization fixed-point model binding is stale")
    for name, lane in (("lane_a", lane_a), ("lane_b", lane_b)):
        if source_artifact(lane, current_model["path"]) != current_model:
            raise RuntimeError(f"{name} fixed-point model binding is stale")
    if localization.get("model") != lane_a.get("model") or localization.get(
        "model"
    ) != lane_b.get("model"):
        raise RuntimeError("localization model identity differs from the lane evidence")

    scope = localization.get("scope", {})
    if scope.get("layer") != 0 or tuple(scope.get("datasets", ())) != DATASETS:
        raise RuntimeError("localization scope is not the frozen paired layer-0 scope")
    boundary_order = tuple(scope.get("boundary_order", ()))
    if not boundary_order or boundary_order[-1] != "model.layers.0.score":
        raise RuntimeError("localization does not terminate at the layer-0 score")

    run_contract = localization.get("artifacts", {}).get("run_contract")
    if not isinstance(run_contract, dict):
        raise RuntimeError("localization run contract is not bound")
    require_artifact(run_contract, "localization run contract")
    for source in localization.get("artifacts", {}).get("sources", []):
        require_artifact(source, f"localization source {source.get('path', '<missing>')}")
    return {
        "accepted_rtl_hash": accepted_rtl_hash,
        "fixed_point_model": current_model,
        "frozen_inputs": baseline["input_observations"],
    }


def verify_stale_localization_rejection(
    localization: dict[str, Any],
    baseline: dict[str, Any],
    lane_a: dict[str, Any],
    lane_b: dict[str, Any],
    baseline_path: Path,
) -> dict[str, bool]:
    cases = (
        (
            "stale_candidate_binding_rejected",
            lambda value: value["bindings"].__setitem__(
                "accepted_rtl_hash", "0" * 64
            ),
            "accepted RTL binding is stale",
        ),
        (
            "stale_model_binding_rejected",
            lambda value: value["bindings"]["fixed_point_model"].__setitem__(
                "sha256", "0" * 64
            ),
            "fixed-point model binding is stale",
        ),
    )
    checks: dict[str, bool] = {}
    for name, mutate, expected in cases:
        stale = copy.deepcopy(localization)
        mutate(stale)
        try:
            verify_localization_bindings(
                stale,
                baseline,
                lane_a,
                lane_b,
                baseline_path,
            )
        except RuntimeError as exc:
            if expected not in str(exc):
                raise
            checks[name] = True
        else:
            raise RuntimeError(f"comparator accepted {name}")
    return checks


def verify_hash_list(path: Path) -> list[dict[str, str]]:
    verified = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        source = ROOT / relative
        actual = sha256(source)
        if actual != expected:
            raise RuntimeError(f"bound source changed: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return verified


def ratios(result: dict[str, Any]) -> dict[str, float]:
    return {name: float(result["metrics"][name]["ratio"]) for name in DATASETS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lane-a", type=Path, required=True)
    parser.add_argument("--lane-b", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--source-hashes", type=Path, required=True)
    parser.add_argument("--constraint-hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "baseline": ROOT / args.baseline,
        "lane_a": ROOT / args.lane_a,
        "lane_b": ROOT / args.lane_b,
        "localization": ROOT / args.localization,
        "source_hashes": ROOT / args.source_hashes,
        "constraint_hashes": ROOT / args.constraint_hashes,
    }
    baseline = load(paths["baseline"])
    lane_a = load(paths["lane_a"])
    lane_b = load(paths["lane_b"])
    localization = load(paths["localization"])

    for name, lane in (("lane_a", lane_a), ("lane_b", lane_b)):
        if lane["input_observations"] != baseline["input_observations"]:
            raise RuntimeError(f"{name} did not reuse the frozen paired input")
        if (
            lane["artifacts"]["accepted_rtl"]["candidate_rtl_hash"]
            != baseline["artifacts"]["accepted_rtl"]["candidate_rtl_hash"]
        ):
            raise RuntimeError(f"{name} changed the accepted RTL binding")
        for dataset in DATASETS:
            if lane["metrics"][dataset]["bf16"] != baseline["metrics"][dataset]["bf16"]:
                raise RuntimeError(f"{name} changed the {dataset} BF16 baseline")

    if lane_a["diagnostic_numerical_repair"]["lane"] != (
        "A_layer0_attention_score_requantization"
    ):
        raise RuntimeError("lane A identity is not bound")
    if lane_b["diagnostic_numerical_repair"]["lane"] != (
        "B_layer0_input_rmsnorm_requantization"
    ):
        raise RuntimeError("lane B identity is not bound")

    baseline_ratios = ratios(baseline)
    lane_ratios = {"lane_a": ratios(lane_a), "lane_b": ratios(lane_b)}
    comparisons: dict[str, Any] = {}
    for lane_name, observed in lane_ratios.items():
        improvement = {
            dataset: 100.0
            * (baseline_ratios[dataset] - observed[dataset])
            / baseline_ratios[dataset]
            for dataset in DATASETS
        }
        comparisons[lane_name] = {
            "perplexity_ratios": observed,
            "improvement_percent_vs_baseline": improvement,
            "joint_end_to_end_improvement": all(
                observed[dataset] < baseline_ratios[dataset] for dataset in DATASETS
            ),
        }

    localization_bindings = verify_localization_bindings(
        localization,
        baseline,
        lane_a,
        lane_b,
        paths["baseline"],
    )
    stale_binding_checks = verify_stale_localization_rejection(
        localization,
        baseline,
        lane_a,
        lane_b,
        paths["baseline"],
    )
    localization_manifest = paths["localization"].parent / "SHA256SUMS"
    localization_manifest_files = verify_hash_manifest(localization_manifest)

    first_divergence = localization["first_material_divergence"]
    if set(first_divergence.values()) != {"model.layers.0.score"}:
        raise RuntimeError("frozen localization no longer identifies layer-0 score")

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": "diagnostic_two_lane_no_winner_not_acceptance_evidence",
        "status": "stop_no_changed_candidate_admitted",
        "accepted_frontier_modified": False,
        "baseline_perplexity_ratios": baseline_ratios,
        "comparisons": comparisons,
        "winner": None,
        "localization_binding_guard": {
            "fresh_localization_accepted": True,
            **stale_binding_checks,
        },
        "losing_lanes_terminated": ["lane_a", "lane_b"],
        "expensive_validation": {
            "complete_shell_regression_run": False,
            "sky130_ppa_run": False,
            "reason": "neither isolated lane improved both frozen paired-smoke datasets; RTL and constraint hashes are unchanged",
        },
        "first_divergent_tensor": first_divergence,
        "next_concrete_arithmetic_contract_change": {
            "proposal_only_operator_approval_required": True,
            "change": "replace scalar Q/K activation scales with static per-attention-head signed-int8 Q scales and per-KV-head signed-int8 K scales, carry those scales through RoPE, and derive per-query-head score multipliers/right shifts",
            "reason": "a scalar post-dot score correction had no end-to-end effect, while RMSNorm rescaling traded C4 improvement for a severe WikiText-2 regression; the remaining first divergence is the layer-0 score tensor and requires finer Q/K quantization granularity rather than another scalar requantization",
        },
        "bindings": {
            "frozen_inputs": baseline["input_observations"],
            "accepted_rtl_hash": baseline["artifacts"]["accepted_rtl"]["candidate_rtl_hash"],
            "focused_localization": localization_bindings,
            "focused_localization_manifest_files": localization_manifest_files,
            "rtl_sources_current": verify_hash_list(paths["source_hashes"]),
            "constraints_and_flow_current": verify_hash_list(paths["constraint_hashes"]),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "claim_boundary": [
            "No official 14-item suite was run.",
            "No complete shell regression, synthesis, STA, or PPA was run.",
            "This diagnostic does not alter or republish the accepted frontier.",
        ],
    }
    result["artifacts"]["localization_manifest"] = artifact(localization_manifest)
    result["artifacts"]["comparator"] = artifact(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "ACE2_FAST_NUMERICAL_REPAIR "
        "status=stop_no_changed_candidate_admitted winner=none"
    )


if __name__ == "__main__":
    main()
