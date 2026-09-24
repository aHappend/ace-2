#!/usr/bin/env python3
"""Audit the clean-room S6 v2 specification package without running consumers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/bf16-category-balanced-conflict-projected-successor-s6-clean-room-v2"
INPUTS = PACKAGE / "clean-room-inputs"
OUTPUTS = PACKAGE / "construction-output"
DESIGN = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_CATEGORY_BALANCED_CONFLICT_PROJECTED_SUCCESSOR_S6_CLEAN_ROOM_V2_DESIGN_FREEZE.json"

EXPECTED_INPUT_HASHES = {
    "procedural-brief.json": "6243dd3bbacdf35b572b0a7c402f00318b2d2ad572e76740616fd9df4d80148a",
    "s5-aggregate-terminal-audit.json": "bd7bd5c0f55bbb63b9bd66473eb95dc0306380bdd96d53c5c71600b555c192b8",
    "s5-public-execution-contract.json": "6b9425f027bcc50f399fdded1ac91147fbdde91c9a041e10d1e00e6b3c9c0a54",
    "s5-public-training-recipe.json": "24a4b8acf314c0bd90a026b0a2438c424ab281a0f29c9ab9e183e9b272b42df2",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    for filename, expected in EXPECTED_INPUT_HASHES.items():
        require(sha256(INPUTS / filename) == expected, f"input hash mismatch: {filename}")

    manifest = load_json(OUTPUTS / "access-manifest.json")
    projection = load_json(OUTPUTS / "aggregate-only-projection.json")
    spec = load_json(OUTPUTS / "successor-design-freeze.json")
    design = load_json(DESIGN)

    require(spec == design, "design projection is not byte-content equivalent to isolated output")
    require(manifest["exposed_project_evidence_count"] == 4, "unexpected evidence count")
    require(
        {Path(item["capsule_path"]).name: item["sha256"] for item in manifest["exposed_project_evidence"]}
        == EXPECTED_INPUT_HASHES,
        "access manifest does not match the four frozen inputs",
    )
    isolation = manifest["isolation"]
    require(isolation["engine"] == "bubblewrap", "construction was not bubblewrap-isolated")
    require(isolation["namespace_mode"] == "--unshare-all", "namespace isolation incomplete")
    require(isolation["repository_root_visible"] is False, "repository root was visible")
    require(manifest["construction_policy"]["network_access"] is False, "network was allowed")
    require(
        manifest["construction_policy"]["forbidden_construction_inputs_visible"] is False,
        "forbidden construction input was visible",
    )

    require(
        projection["source_scope"] == "ONLY_FIELDS_PERMITTED_BY_CLEAN_ROOM_TASK",
        "projection is not aggregate/public-field limited",
    )
    require(spec["stage"] == "specification", "wrong stage")
    require(
        spec["status"] == "FROZEN_SPECIFICATION_ONLY_NO_CONSUMING_AUTHORITY",
        "wrong freeze status",
    )
    require("clean-room-v2" in spec["spec_id"], "identity is not the new clean-room version")
    require(
        spec["identity_derivation"]["clean_room_task_hash_included"] is True,
        "identity is not bound to the replacement task",
    )

    mechanism = spec["successor_mechanism"]
    require(mechanism["category_order"] == CATEGORIES, "category order mismatch")
    require(mechanism["marker_policy"]["special_control_tokens_added"] is False, "marker added")
    require(mechanism["marker_policy"]["category_label_in_model_input"] is False, "category marker leaks")
    require(mechanism["marker_policy"]["category_label_in_target"] is False, "target marker leaks")
    require(
        mechanism["conflict_projection"]["combined_gradient"]
        == "mean(p_i for all seven categories)",
        "categories are not equally weighted",
    )
    require(
        mechanism["conflict_projection"]["missing_category_disposition"]
        == "FAIL_CLOSED_NO_OPTIMIZER_STEP",
        "missing-category behavior is not fail closed",
    )
    require(len(mechanism["material_difference_from_permitted_s5_mechanism"]["new_capabilities"]) >= 4, "material mechanism delta absent")

    data = spec["future_data_contract"]
    assignment = data["category_assignment"]
    require(data["status"] == "NOT_MATERIALIZED_BY_THIS_FREEZE", "data was materialized")
    require(data["train_count"] == 352, "train count drift")
    require(assignment["backbone_per_category"] == 40, "backbone not category balanced")
    require(assignment["category_specific_patch_per_category"] == 10, "patch not category balanced")
    require(assignment["single_category_rows_per_category"] == 50, "combined data not category balanced")
    require(assignment["cross_category_bridge_rows"] == 2, "bridge-row count mismatch")
    require(data["row_requirements"]["new_rows_created_now"] == 0, "rows were created")
    require(data["row_requirements"]["literal_row_content_in_spec"] is False, "row content leaked")

    training = spec["training_contract"]
    require(training["compute_dtype"] == "bfloat16", "training compute is not BF16")
    require(training["loss_scope"] == "assistant_tokens_only", "loss scope drift")
    require(training["derived_epoch_layout"]["optimizer_steps_per_epoch"] == 44, "epoch layout mismatch")
    require(training["training_execution_authorized"] is False, "training was authorized")

    selection = spec["checkpoint_selection_contract"]
    require(selection["synthetic_probe_count"] == 28, "probe count drift")
    require(selection["synthetic_probe_per_category"] == 4, "probes not category balanced")
    require(selection["minimum_hard_passes"] == 24, "probe hard-pass threshold mismatch")
    require(
        selection["category_minimum_hard_passes"]
        == {
            "arithmetic": 3,
            "concise_summary": 3,
            "context_memory": 3,
            "format_discipline": 4,
            "polite_rewrite": 3,
            "safe_refusal": 4,
            "structured_extraction": 3,
        },
        "probe category thresholds mismatch",
    )
    require(
        selection["no_passing_epoch_disposition"]
        == "NO_CHECKPOINT; DO_NOT_ACCESS_OFFICIAL_DEV",
        "checkpoint selection has a fallback",
    )

    dev = spec["official_dev_contract"]
    require(dev["response_count"] == 56, "official dev count drift")
    require(dev["minimum_hard_passes"] == 48, "official dev threshold drift")
    require(dev["fail_closed_on_missing_or_unscored_case"] is True, "dev is not fail closed")
    require(dev["execution_authorized_by_this_freeze"] is False, "dev execution was authorized")

    no_downstream = spec["no_downstream_boundary"]
    require(
        all(value is False for key, value in no_downstream.items() if key != "next_required_action"),
        "consuming or downstream state is asserted",
    )
    prohibited = spec["prohibited_content_attestation"]
    require(all(value is False for value in prohibited.values()), "prohibited content is asserted")

    package_files = {
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*")
        if path.is_file()
    }
    expected_package_files = {
        "clean-room-inputs/procedural-brief.json",
        "clean-room-inputs/s5-aggregate-terminal-audit.json",
        "clean-room-inputs/s5-public-execution-contract.json",
        "clean-room-inputs/s5-public-training-recipe.json",
        "construction-output/access-manifest.json",
        "construction-output/aggregate-only-projection.json",
        "construction-output/successor-design-freeze.json",
        "package-audit.json",
    }
    unexpected = package_files - expected_package_files
    require(not unexpected, f"unexpected package files: {sorted(unexpected)}")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "audit_scope": "LOCAL_NON_CONSUMING_CLEAN_ROOM_SUCCESSOR_PACKAGE",
        "spec_id": spec["spec_id"],
        "checks": {
            "input_hashes": "PASS",
            "aggregate_only_projection": "PASS",
            "isolated_construction_manifest": "PASS",
            "new_source_bound_identity": "PASS",
            "marker_free": "PASS",
            "category_balanced": "PASS",
            "conflict_projected": "PASS",
            "bf16_training_recipe": "PASS",
            "checkpoint_selection_fail_closed": "PASS",
            "official_dev_contract_preserved": "PASS",
            "zero_consuming_state": "PASS",
            "prohibited_content_absent_by_attestation": "PASS",
        },
        "design_sha256": sha256(DESIGN),
        "access_manifest_sha256": sha256(OUTPUTS / "access-manifest.json"),
        "aggregate_projection_sha256": sha256(OUTPUTS / "aggregate-only-projection.json"),
        "fresh_reviewer_required": True,
    }
    (PACKAGE / "package-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
