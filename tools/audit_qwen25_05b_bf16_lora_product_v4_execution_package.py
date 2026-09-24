#!/usr/bin/env python3
"""Audit the non-consuming V4 execution package without opening dev or holdout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v4_runner"
RUNTIME_PATH = PACKAGE / "runtime.py"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_SUM = ROOT / "research/PIPELINE_STATE.sha256"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def import_runtime() -> Any:
    sys.path.insert(0, str(PACKAGE))
    spec = importlib.util.spec_from_file_location("ace2_v4_execution_runtime_audit", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import V4 execution runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ordered(source: str, fragments: list[str]) -> bool:
    position = -1
    for fragment in fragments:
        position = source.find(fragment, position + 1)
        if position < 0:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = import_runtime()
    pipeline = load_json(PIPELINE)
    frozen = runtime.load_frozen()
    package_manifest = runtime.execution_package_manifest()
    self_test = load_json(runtime.SELF_TEST_RESULT) if runtime.SELF_TEST_RESULT.is_file() else {}
    persisted_manifest = load_json(runtime.EXECUTION_MANIFEST) if runtime.EXECUTION_MANIFEST.is_file() else {}
    train_source = (PACKAGE / "train.py").read_text(encoding="utf-8")
    dev_source = (PACKAGE / "evaluate_dev.py").read_text(encoding="utf-8")
    holdout_source = (PACKAGE / "evaluate_holdout.py").read_text(encoding="utf-8")
    run_source = (PACKAGE / "run_once.sh").read_text(encoding="utf-8")
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")

    approval = pipeline.get("approved_next_stage", {})
    stages = pipeline.get("stages", {})
    checks = {
        "active_stage_exact": pipeline.get("current_stage") == "v4_execution_package",
        "active_stage_status_in_progress": stages.get("v4_execution_package", {}).get("status") == "in_progress",
        "specification_closed": stages.get("specification", {}).get("status") == "closed_next_stage_approved",
        "manager_approval_active": approval.get("next_stage") == "v4_execution_package" and approval.get("status") == "active",
        "manager_claim_boundary_no_attempt": "no model namespace" in approval.get("claim_boundary", "") and "training" in approval.get("claim_boundary", ""),
        "pipeline_sidecar_exact": runtime.companion_matches(PIPELINE, PIPELINE_SUM),
        "frozen_inputs_load_and_hash": bool(frozen),
        "frozen_static_package_still_nonexecuting": not any(
            (ROOT / "pilot/qwen25_05b_bf16_lora_product_v4" / name).exists()
            for name in ("train.py", "run_once.sh", "evaluate_dev.py", "evaluate_holdout.py", "merge.py")
        ),
        "runner_file_set_complete": set(package_manifest["files"]) == set(runtime.EXECUTION_FILES),
        "persisted_manifest_matches_runner": persisted_manifest == package_manifest,
        "persisted_manifest_sidecar_exact": runtime.companion_matches(runtime.EXECUTION_MANIFEST, runtime.EXECUTION_MANIFEST_SUM),
        "self_test_pass": self_test.get("status") == "PASS",
        "self_test_sidecar_exact": runtime.companion_matches(runtime.SELF_TEST_RESULT, runtime.SELF_TEST_RESULT_SUM),
        "self_test_tree_binding": self_test.get("execution_tree_sha256") == package_manifest["tree_sha256"],
        "self_test_manifest_binding": self_test.get("execution_manifest_sha256") == runtime.sha256_file(runtime.EXECUTION_MANIFEST),
        "real_model_self_test_present": isinstance(self_test.get("real_model_metrics"), dict),
        "official_namespace_absent": not runtime.RUN_ROOT.exists(),
        "attempt_authority_absent": not runtime.ATTEMPT_AUTHORITY_PATH.exists(),
        "execution_l2_pending": not runtime.EXECUTION_L2_PATH.exists(),
        "attempt_markers_absent": not list(runtime.RUN_ROOT.glob("attempt-*/ATTEMPT_CONSUMPTION_MARKER.json")),
        "marker_created_only_after_all_interlocks": ordered(
            train_source,
            [
                "verify_execution_self_test()",
                "verify_execution_l2()",
                "verify_attempt_authority()",
                "verify_preflight_ready()",
                "exclusive_write_json(marker_path, marker)",
                "Trainer(",
            ],
        ),
        "probe_callback_evaluates_each_saved_epoch": all(
            fragment in train_source
            for fragment in (
                "class ProbeLockCallback(TrainerCallback)",
                "def on_save",
                "evaluate_rows(",
                "control.should_training_stop = True",
            )
        ),
        "probe_lock_precedes_dev": ordered(
            train_source,
            ["select_probe_checkpoint(callback.candidates)", "lock_path = ATTEMPT_DIR / \"checkpoint_lock.json\"", "training_result.json"],
        ) and ordered(
            dev_source,
            ["lock_path = ATTEMPT_DIR / \"checkpoint_lock.json\"", "ACCESS_MARKER.json", "split_path(frozen, \"dev\")"],
        ),
        "dev_is_single_locked_checkpoint_only": all(
            fragment in dev_source
            for fragment in (
                "official_dev_checkpoint_count\": 1",
                "official_dev_reselection_allowed\": False",
                "checkpoint = verify_directory_manifest(checkpoint_identity)",
            )
        ),
        "holdout_marker_precedes_holdout_open": ordered(
            holdout_source,
            [
                "exclusive_write_json(\n        marker_path",
                "observed_holdout_sha256 = sha256_file(split_path(frozen, \"holdout\"))",
                "rows = read_jsonl(split_path(frozen, \"holdout\"))",
            ],
        ),
        "run_order_fail_closed": ordered(
            run_source,
            [
                "preflight.py",
                "train.py",
                "evaluate_dev.py",
                "merge.py",
                "retention.py",
                "evaluate_holdout.py",
            ],
        ),
        "seed_conflict_recorded": "training_recipe.json` sets `determinism.PYTHONHASHSEED`" in ground_truth,
        "seed_resolution_explicit_in_contract": frozen["contract"]["environment_resolution"]["pythonhashseed_conflict"]["selected_value"] == "26080704",
    }
    failed = [name for name, passed in checks.items() if not passed]
    status = "PASS_IMPLEMENTED_PENDING_INDEPENDENT_L2" if not failed else "FAIL"
    result = {
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": "Non-consuming V4 execution-package audit only; no attempt marker, training, dev, retention, holdout, merge, quantization, RTL, synthesis/PPA, or U280 execution.",
        "execution_contract_sha256": runtime.sha256_file(runtime.CONTRACT_PATH),
        "execution_manifest_sha256": runtime.sha256_file(runtime.EXECUTION_MANIFEST) if runtime.EXECUTION_MANIFEST.is_file() else None,
        "execution_tree_sha256": package_manifest["tree_sha256"],
        "failed_checks": failed,
        "freeze_manifest_sha256": runtime.sha256_file(runtime.FREEZE_MANIFEST_PATH),
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "official_namespace_exists": runtime.RUN_ROOT.exists(),
        "schema_version": 1,
        "status": status,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{runtime.sha256_file(output)}  {output.name}\n",
            encoding="ascii",
        )
    print(payload, end="")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
