#!/usr/bin/env python3
"""Audit V4 execution-package closure without consuming attempt authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_SUM = ROOT / "research/PIPELINE_STATE.sha256"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V4_EXECUTION_PACKAGE_CONTRACT.json"
ACCEPTANCE = (
    ROOT
    / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4/"
    "execution-package-l2-acceptance.json"
)
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v4_runner"
RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4"
ATTEMPT_AUTHORITY = (
    ROOT
    / "research/raw/specification/"
    "qwen25-05b-instruct-bf16-lora-product-v4-attempt-operator-authority.json"
)
READINESS = ACCEPTANCE.parent / "v4-launch-readiness.json"
RUNNER_FILES = (
    "common.py",
    "runtime.py",
    "train.py",
    "evaluate_dev.py",
    "merge.py",
    "retention.py",
    "evaluate_holdout.py",
    "preflight.py",
    "self_test.py",
    "run_once.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def project_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def companion_matches(path: Path, companion: Path | None = None) -> bool:
    companion = companion or path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not companion.is_file():
        return False
    fields = companion.read_text(encoding="ascii").split()
    return len(fields) == 2 and fields[0] == sha256(path) and fields[1] == path.name


def all_checks_true(record: dict[str, Any]) -> bool:
    checks = record.get("checks")
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def bindings_reproduce(record: dict[str, Any]) -> bool:
    bindings = record.get("artifact_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return False
    for binding in bindings.values():
        if not isinstance(binding, dict):
            return False
        path = project_path(binding.get("path", ""))
        if not path.is_file() or sha256(path) != binding.get("sha256"):
            return False
    return True


def runner_manifest() -> dict[str, Any]:
    manifest_path = ACCEPTANCE.parent / "v4-execution-package-manifest.json"
    persisted = load_json(manifest_path)
    files = persisted.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("execution package manifest has no files")
    digest = hashlib.sha256()
    observed_files: dict[str, dict[str, Any]] = {}
    if set(files) != set(RUNNER_FILES):
        raise RuntimeError("execution package manifest file set differs")
    for name in RUNNER_FILES:
        path = PACKAGE / name
        payload = path.read_bytes()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        observed_files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    return {
        "files": observed_files,
        "tree_sha256": digest.hexdigest(),
    }


def frozen_artifacts_reproduce(freeze_manifest_path: Path) -> bool:
    freeze = load_json(freeze_manifest_path)
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for relative, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            return False
        path = ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != metadata.get("bytes")
            or sha256(path) != metadata.get("sha256")
        ):
            return False
    return True


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pipeline = load_json(PIPELINE)
    contract = load_json(CONTRACT)
    acceptance = load_json(ACCEPTANCE)
    bindings = acceptance.get("artifact_bindings", {})
    manifest_binding = bindings.get("execution_package_manifest", {})
    self_test_binding = bindings.get("execution_package_self_test", {})
    package_audit_binding = bindings.get("execution_package_audit", {})
    freeze_binding = bindings.get("freeze_manifest", {})
    manifest_path = project_path(manifest_binding.get("path", ""))
    self_test_path = project_path(self_test_binding.get("path", ""))
    package_audit_path = project_path(package_audit_binding.get("path", ""))
    freeze_manifest_path = project_path(freeze_binding.get("path", ""))
    persisted_manifest = load_json(manifest_path)
    self_test = load_json(self_test_path)
    package_audit = load_json(package_audit_path)
    observed_runner = runner_manifest()
    stages = pipeline.get("stages", {})
    approval = pipeline.get("approved_next_stage", {})
    ground_truth = GROUND_TRUTH.read_text(encoding="utf-8")

    checks = {
        "acceptance_all_checks_pass": all_checks_true(acceptance),
        "acceptance_artifact_bindings_reproduce": bindings_reproduce(acceptance),
        "acceptance_companion_exact": companion_matches(ACCEPTANCE),
        "acceptance_is_independent_l2": acceptance.get("independent_from_constructor") is True
        and acceptance.get("reviewer_role") == "independent_l2",
        "acceptance_status_exact": acceptance.get("accepted") is True
        and acceptance.get("status") == "ACCEPTED_V4_EXECUTION_PACKAGE_NO_ATTEMPT",
        "accepted_seed_resolution_exact": acceptance.get("checks", {}).get(
            "pythonhashseed_recipe_precedence_accepted_without_frozen_mutation"
        )
        is True,
        "attempt_authority_absent": not ATTEMPT_AUTHORITY.exists(),
        "attempt_markers_absent": not list(RUN_ROOT.glob("attempt-*/ATTEMPT_CONSUMPTION_MARKER.json")),
        "contract_companion_exact": companion_matches(CONTRACT),
        "contract_requires_separate_attempt_authority": contract.get("attempt_authority", {}).get(
            "separate_fresh_operator_authority_required"
        )
        is True,
        "freeze_artifacts_reproduce": frozen_artifacts_reproduce(freeze_manifest_path),
        "freeze_manifest_companion_exact": companion_matches(freeze_manifest_path),
        "resolved_l2_unknown_removed": "whether an independent L2 Reviewer accepts" not in ground_truth,
        "manager_approval_remains_no_attempt": approval.get("next_stage") == "v4_execution_package"
        and approval.get("status") == "active"
        and approval.get("attempt_namespace_must_remain_absent") is True,
        "official_namespace_absent": not RUN_ROOT.exists(),
        "package_audit_input_exact": package_audit.get("status")
        == "PASS_IMPLEMENTED_PENDING_INDEPENDENT_L2"
        and package_audit.get("check_count") == 28
        and all_checks_true(package_audit),
        "package_evidence_hashes_exact": sha256(manifest_path) == manifest_binding.get("sha256")
        and sha256(self_test_path) == self_test_binding.get("sha256")
        and sha256(package_audit_path) == package_audit_binding.get("sha256"),
        "pipeline_companion_exact": companion_matches(PIPELINE, PIPELINE_SUM),
        "pipeline_stage_exact": pipeline.get("current_stage") == "v4_execution_package"
        and stages.get("v4_execution_package", {}).get("status") == "in_progress"
        and stages.get("specification", {}).get("status") == "closed_next_stage_approved",
        "readiness_not_created": not READINESS.exists(),
        "runner_files_exact": observed_runner["files"] == persisted_manifest.get("files"),
        "runner_tree_exact": observed_runner["tree_sha256"]
        == persisted_manifest.get("tree_sha256")
        == acceptance.get("execution_tree_sha256"),
        "self_test_input_exact": self_test.get("status") == "PASS"
        and self_test.get("check_count") == 29
        and all_checks_true(self_test),
        "stage_grants_no_attempt_or_training": acceptance.get("attempt_authority_granted") is False
        and acceptance.get("attempt_marker_creation_authorized") is False
        and acceptance.get("training_authority_granted") is False
        and acceptance.get("official_model_namespace_created") is False
        and acceptance.get("dev_holdout_content_accessed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    status = "PASS_READY_FOR_MANAGER_STAGE_CLOSURE_NO_ATTEMPT" if not failed else "FAIL"
    result = {
        "acceptance_sha256": sha256(ACCEPTANCE),
        "check_count": len(checks),
        "checks": dict(sorted(checks.items())),
        "claim_boundary": (
            "Read-only V4 execution-package closure audit. No readiness, attempt authority, "
            "attempt marker, training, dev, retention, holdout, merge, quantization, RTL, "
            "synthesis/PPA, or U280 execution was created."
        ),
        "execution_manifest_sha256": sha256(manifest_path),
        "execution_tree_sha256": persisted_manifest.get("tree_sha256"),
        "failed_checks": failed,
        "manager_action_required": (
            "Close or advance the Manager-owned v4_execution_package stage. A separate fresh "
            "operator authority binding this accepted package is still required before an "
            "Engineer may create the sole V4 attempt marker."
        ),
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "official_namespace_exists": RUN_ROOT.exists(),
        "schema_version": 1,
        "status": status,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        atomic_write(output, payload)
        atomic_write(
            output.with_suffix(output.suffix + ".sha256"),
            f"{sha256(output)}  {output.name}\n",
        )
    print(payload, end="")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
