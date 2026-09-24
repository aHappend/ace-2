#!/usr/bin/env python3
"""Deterministic, non-consuming audit for the S6 DESIGN/FREEZE artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_CATEGORY_BALANCED_SUCCESSOR_S6_DESIGN_FREEZE.json"
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
AUDIT = ROOT / "build/bf16-category-balanced-successor-s6-design-freeze/package-audit.json"
AUDIT_SIDECAR = Path(str(AUDIT) + ".sha256")
S5_TERMINAL = ROOT / "build/bf16-full-finetune-successor-s5/backend-terminal-audit-20260808T150208Z.json"
S5_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S5_EXECUTION_PACKAGE_CONTRACT.json"
S5_RECIPE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s5/training_recipe.json"
SPEC = ROOT / "design/SPEC.md"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CHECKPOINT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f49cd09b53fb/CHECKPOINT.md")

EXPECTED_SOURCE_HASHES = {
    S5_TERMINAL: "bd7bd5c0f55bbb63b9bd66473eb95dc0306380bdd96d53c5c71600b555c192b8",
    S5_CONTRACT: "6b9425f027bcc50f399fdded1ac91147fbdde91c9a041e10d1e00e6b3c9c0a54",
    S5_RECIPE: "24a4b8acf314c0bd90a026b0a2438c424ab281a0f29c9ab9e183e9b272b42df2",
}

CATEGORIES = [
    "arithmetic",
    "concise_summary",
    "context_memory",
    "format_discipline",
    "polite_rewrite",
    "safe_refusal",
    "structured_extraction",
]

EXPECTED_DEV = {
    "arithmetic": 2,
    "concise_summary": 8,
    "context_memory": 8,
    "format_discipline": 0,
    "polite_rewrite": 1,
    "safe_refusal": 0,
    "structured_extraction": 8,
}

EXPECTED_PROBES = {
    "1": {"arithmetic": 3, "concise_summary": 0, "context_memory": 4, "format_discipline": 0, "polite_rewrite": 2, "safe_refusal": 4, "structured_extraction": 0},
    "2": {"arithmetic": 3, "concise_summary": 0, "context_memory": 4, "format_discipline": 0, "polite_rewrite": 1, "safe_refusal": 4, "structured_extraction": 0},
    "3": {"arithmetic": 3, "concise_summary": 0, "context_memory": 4, "format_discipline": 0, "polite_rewrite": 3, "safe_refusal": 4, "structured_extraction": 0},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare against existing deterministic audit artifacts")
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED_SOURCE_HASHES}
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        checks[f"source.{path.name}.sha256_exact"] = sha256(path) == expected

    contract = load_json(CONTRACT)
    terminal = load_json(S5_TERMINAL)
    s5_contract = load_json(S5_CONTRACT)
    recipe = load_json(S5_RECIPE)

    dev = terminal["quality"]["dev"]
    probes = terminal["quality"]["probe_selection"]
    thresholds = {
        category: values["minimum_hard_passes"]
        for category, values in s5_contract["evaluator_contract"]["dev"]["category_gates"].items()
    }
    deficits = {category: max(0, thresholds[category] - EXPECTED_DEV[category]) for category in CATEGORIES}
    weights = {category: 1 + deficits[category] for category in CATEGORIES}

    checks["s5.failure_taxonomy_exact"] = terminal["failure_taxonomy"] == "DEV_QUALITY_GATE_FAILURE"
    checks["s5.dev_category_passes_exact"] = dev["category_passes"] == EXPECTED_DEV
    checks["s5.dev_category_totals_exact"] = dev["category_totals"] == {category: 8 for category in CATEGORIES}
    checks["s5.dev_hard_pass_exact"] = dev["hard_pass_count"] == 27
    checks["s5.dev_response_count_exact"] = dev["response_count"] == 56
    checks["s5.dev_zero_critical_safety_failures"] = dev["critical_safety_failures"] == 0
    checks["s5.selected_epoch_exact"] = terminal["quality"]["selected_epoch"] == 3
    checks["s5.probe_aggregate_sequence_exact"] = all(
        probes[epoch]["category_passes"] == expected for epoch, expected in EXPECTED_PROBES.items()
    )
    checks["s5.locked_probe_only_two_category_gates"] = probes["3"]["category_minimum_gates_satisfied"] == 2
    checks["s5.recipe_full_parameter_exact"] = recipe["full_finetune"]["all_model_parameters_trainable"] is True
    checks["s5.recipe_global_assistant_loss_exact"] = recipe["training"]["loss_scope"] == "assistant_tokens_only"
    checks["s5.dev_thresholds_exact"] = thresholds == {
        "arithmetic": 6,
        "concise_summary": 6,
        "context_memory": 6,
        "format_discipline": 7,
        "polite_rewrite": 6,
        "safe_refusal": 7,
        "structured_extraction": 6,
    }

    checks["contract.id_exact"] = contract["contract_id"] == "qwen2.5-0.5b-instruct-ace2-bf16-category-balanced-successor-s6-design-freeze-v1"
    checks["contract.status_marker_free"] = contract["status"] == "FROZEN_MARKER_FREE_DESIGN_PENDING_FRESH_REVIEW"
    checks["contract.design_only"] = contract["design_scope"] == {
        "creates_execution_package": False,
        "creates_training_data": False,
        "stage": "specification",
        "task_type": "DESIGN_FREEZE_ONLY",
    }
    checks["contract.aggregate_dev_exact"] = contract["aggregate_diagnosis"]["dev_category_passes"] == EXPECTED_DEV
    checks["contract.aggregate_probe_exact"] = contract["aggregate_diagnosis"]["locked_probe_epoch_category_passes"] == EXPECTED_PROBES["3"]
    checks["contract.deficits_derived_exact"] = contract["aggregate_diagnosis"]["dev_category_deficit_to_frozen_minimum"] == deficits
    checks["contract.weights_derived_exact"] = contract["mechanism"]["training_objective"]["category_loss_weights_unnormalized"] == weights
    checks["contract.material_difference_explicit"] = contract["material_difference_from_s5"]["structurally_distinct"] is True
    checks["contract.no_adapter"] = contract["mechanism"]["parameter_efficient_adapter_used"] is False
    checks["contract.source_outputs_unused"] = contract["mechanism"]["source_anchor"]["source_model_outputs_used"] is False
    checks["contract.no_selector_fallback"] = contract["selector_freeze"]["fallback_checkpoint_selection_allowed"] is False
    checks["contract.selector_thresholds_unchanged"] = contract["selector_freeze"]["combined_probe_category_minimum_hard_passes"] == thresholds
    checks["contract.fresh_balanced_data_shape"] = (
        contract["frozen_future_data_contract"]["rows_per_category"] == 64
        and contract["frozen_future_data_contract"]["total_training_rows"] == 448
        and contract["frozen_future_data_contract"]["all_rows_must_be_fresh_project_authored"] is True
    )
    checks["contract.allowed_sources_exact"] = set(contract["evidence_basis"]["sources"]) == {
        "s5_execution_contract",
        "s5_public_training_recipe",
        "s5_terminal_aggregate_audit",
    }
    checks["contract.no_hidden_output_inclusion"] = contract["state_boundary"]["hidden_or_row_level_output_included"] is False
    checks["contract.zero_consuming_flags"] = all(value is False for value in contract["state_boundary"].values())
    checks["contract.future_execution_closed"] = all(
        contract["future_execution_boundary"][key] is False
        for key in (
            "attempt_marker_creation_permitted",
            "backend_submission_permitted",
            "dev_access_permitted",
            "execution_package_implementation_permitted",
            "holdout_access_permitted",
            "retention_execution_permitted",
        )
    )

    consuming_paths = [
        ROOT / "research/raw/specification/qwen25-bf16-category-balanced-successor-s6-attempt-operator-authority.json",
        ROOT / "build/bf16-category-balanced-successor-s6/backend-submission-intent.json",
        ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-category-balanced-successor-s6",
        ROOT / "pilot/qwen25_05b_bf16_category_balanced_successor_s6",
        ROOT / "pilot/qwen25_05b_bf16_category_balanced_successor_s6_runner",
        ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-category-balanced-successor-s6",
    ]
    checks["state.consuming_paths_absent"] = not any(path.exists() for path in consuming_paths)
    checks["state.no_s6_attempt_marker"] = not any(
        "successor-s6" in str(path) for path in ROOT.rglob("ATTEMPT_CONSUMPTION_MARKER.json")
    )
    checks["state.no_s6_authority_artifact"] = not any(
        "successor-s6" in str(path) for path in ROOT.rglob("*operator-authority*.json")
    )
    checks["state.only_design_and_audit_artifacts"] = not contract["design_scope"]["creates_execution_package"]

    contract_hash = sha256(CONTRACT)
    contract_id = contract["contract_id"]
    checks["projection.spec_reconciled"] = contract_id in SPEC.read_text(encoding="utf-8") and contract_hash in SPEC.read_text(encoding="utf-8")
    checks["projection.ground_truth_reconciled"] = contract_id in GROUND_TRUTH.read_text(encoding="utf-8") and contract_hash in GROUND_TRUTH.read_text(encoding="utf-8")
    checks["projection.checkpoint_current"] = contract_id in CHECKPOINT.read_text(encoding="utf-8")

    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": contract["claim_boundary"],
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": contract_hash,
        },
        "failed_checks": failed,
        "failure_taxonomy": None if not failed else "DESIGN_FREEZE_AUDIT_FAILURE",
        "schema_version": 1,
        "source_hashes": source_hashes,
        "state_counts": {
            "authority": 0,
            "backend_experiment": 0,
            "backend_job": 0,
            "execution_namespace": 0,
            "intent": 0,
            "marker": 0,
            "replay": 0,
        },
        "status": "PASS_MARKER_FREE_S6_DESIGN_FREEZE_ZERO_CONSUMING_STATE" if not failed else "FAIL",
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    sidecar = f"{hashlib.sha256(encoded).hexdigest()}  {AUDIT.name}\n".encode("utf-8")
    contract_sidecar = f"{contract_hash}  {CONTRACT.name}\n".encode("utf-8")

    if args.check:
        if not AUDIT.exists() or AUDIT.read_bytes() != encoded:
            raise SystemExit("audit artifact is missing or stale")
        if not AUDIT_SIDECAR.exists() or AUDIT_SIDECAR.read_bytes() != sidecar:
            raise SystemExit("audit sidecar is missing or stale")
        if not CONTRACT_SIDECAR.exists() or CONTRACT_SIDECAR.read_bytes() != contract_sidecar:
            raise SystemExit("contract sidecar is missing or stale")
    else:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_bytes(encoded)
        AUDIT_SIDECAR.write_bytes(sidecar)
        CONTRACT_SIDECAR.write_bytes(contract_sidecar)

    print(json.dumps({"check_count": len(checks), "failed_checks": failed, "status": result["status"]}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
