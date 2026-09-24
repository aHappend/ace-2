#!/usr/bin/env python3
"""Validate the construction-only position-01 causal-state package."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_RESULT_SHA256 = "73dd8c2e54cd5275ab0a28e11279fb1b03986e5f82ad7ad6c54e42a575ff1deb"
EXPECTED_CHECKPOINT_SHA256 = "2ce126478f2fbe7af50a316947df78f01cfecc625232a1e3cf94c86e3556c011"
EXPECTED_LAYER23_OUTPUT_SHA256 = "6646ca8322ec0c4271505683eeff751f1a48a6be9d85d19735e877e722fd3187"
EXPECTED_TOKEN_ID = 106339
ZERO_ACTIVITY = {
    "authority_consumed": 0,
    "authority_created": 0,
    "checkpoint_mutation": 0,
    "executor_invocation": 0,
    "generated_tokens": 0,
    "grant_consumed": 0,
    "manager_grant_created": 0,
    "model_execution": 0,
    "ppa_execution": 0,
    "protected_state_delta": 0,
    "reference_execution": 0,
    "rtl_execution": 0,
    "stage1_closure": 0,
    "stage2_execution": 0,
    "u280_execution": 0,
    "unit_execution": 0,
    "workload_execution": 0,
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def checkpoint_canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValidationError(f"nonfinite JSON value: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: Any) -> Path:
    require(isinstance(value, str) and value, "non-empty relative path required")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe path: {value}")
    require(path.as_posix() == value, f"non-canonical path: {value}")
    return path


def regular_path(root: Path, relative: Any) -> Path:
    rel = safe_relative(relative)
    current = root
    for part in rel.parts:
        current = current / part
        require(not current.is_symlink(), f"symlink forbidden: {rel.as_posix()}")
    require(current.is_file(), f"regular file absent: {rel.as_posix()}")
    return current


def validate_record(root: Path, record: Any, label: str) -> Path:
    require(
        isinstance(record, dict)
        and set(record) == {"bytes", "path", "sha256"}
        and isinstance(record["bytes"], int)
        and record["bytes"] >= 0
        and isinstance(record["sha256"], str)
        and len(record["sha256"]) == 64,
        f"invalid file record: {label}",
    )
    path = regular_path(root, record["path"])
    require(path.stat().st_size == record["bytes"], f"byte count changed: {label}")
    require(sha256_file(path) == record["sha256"], f"SHA-256 changed: {label}")
    return path


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "invalid SHA256SUMS line")
        relative = safe_relative(parts[1]).as_posix()
        require(relative not in records, f"duplicate sealed member: {relative}")
        records[relative] = parts[0]
    require(records, "empty SHA256SUMS")
    return records


def validate_sealed_tree(root: Path, record: Any, label: str) -> None:
    require(
        isinstance(record, dict)
        and set(record) == {"path", "tree_root_sha256"}
        and isinstance(record["tree_root_sha256"], str)
        and len(record["tree_root_sha256"]) == 64,
        f"invalid sealed-tree record: {label}",
    )
    relative = safe_relative(record["path"])
    tree = root / relative
    require(tree.is_dir() and not tree.is_symlink(), f"sealed tree absent: {label}")
    sums_path = regular_path(root, (relative / "SHA256SUMS").as_posix())
    root_path = regular_path(root, (relative / "TREE_ROOT.sha256").as_posix())
    require(sha256_file(sums_path) == record["tree_root_sha256"], f"tree root changed: {label}")
    sidecar = root_path.read_text(encoding="ascii").split()
    require(
        sidecar == [record["tree_root_sha256"], "SHA256SUMS"],
        f"tree-root sidecar changed: {label}",
    )
    records = parse_sums(sums_path)
    observed: set[str] = set()
    for path in tree.rglob("*"):
        require(not path.is_symlink(), f"symlink in sealed tree: {label}")
        if path.is_file():
            rel = path.relative_to(tree).as_posix()
            if rel not in {"SHA256SUMS", "TREE_ROOT.sha256"}:
                observed.add(rel)
    require(observed == set(records), f"sealed member set changed: {label}")
    for rel, digest in records.items():
        require(sha256_file(regular_path(tree, rel)) == digest, f"sealed member changed: {label}/{rel}")


def validate_package_tree(package: Path) -> tuple[dict[str, Any], str]:
    require(package.is_dir() and not package.is_symlink(), "package directory absent")
    sums_path = regular_path(package, "SHA256SUMS")
    tree_path = regular_path(package, "TREE_ROOT.sha256")
    tree_root = sha256_file(sums_path)
    require(tree_path.read_text(encoding="ascii").split() == [tree_root, "SHA256SUMS"], "package tree root changed")
    records = parse_sums(sums_path)
    observed = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.relative_to(package).as_posix() not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "package member set changed")
    for rel, digest in records.items():
        require(sha256_file(regular_path(package, rel)) == digest, f"package member changed: {rel}")
    manifest = load_json(regular_path(package, "manifest.json"))
    require(regular_path(package, "manifest.json").read_bytes() == canonical_bytes(manifest), "manifest is not canonical JSON")
    return manifest, tree_root


def validate(package: Path, artifact_root: Path) -> dict[str, Any]:
    manifest, tree_root = validate_package_tree(package)
    require(
        manifest.get("schema") == "ace2-position01-causal-state-successor-package-v1"
        and manifest.get("status") == "CONSTRUCTION_ONLY_READY_NO_EXECUTION_AUTHORITY"
        and manifest.get("activity") == ZERO_ACTIVITY,
        "package identity, status, or zero-activity contract changed",
    )

    sources = manifest.get("sources")
    require(isinstance(sources, dict), "source bindings absent")
    validate_sealed_tree(artifact_root, sources.get("grant0005_package"), "grant0005 package")
    validate_sealed_tree(artifact_root, sources.get("layer23_package"), "Layer-23 package")
    paths = {
        name: validate_record(artifact_root, record, name)
        for name, record in sources.items()
        if isinstance(record, dict) and set(record) == {"bytes", "path", "sha256"}
    }

    result = load_json(paths["selected_token_result"])
    terminal = load_json(paths["selected_token_terminal"])
    consumed = load_json(paths["grant0005_consumed"])
    grant = load_json(paths["grant0005_grant"])
    prerequisite = load_json(paths["grant0005_prerequisite"])
    checkpoint = load_json(paths["layer23_checkpoint"])
    layer23_result = load_json(paths["layer23_result"])

    require(sources["selected_token_result"]["sha256"] == EXPECTED_RESULT_SHA256, "selected-token digest changed")
    require(
        result.get("schema") == "ace2-stage1-post-transformer-selected-token-publication-v1"
        and result.get("selected_token_id") == EXPECTED_TOKEN_ID
        and result.get("result", {}).get("selected_token_id") == EXPECTED_TOKEN_ID
        and result.get("result", {}).get("agreement", {}).get("projection_selected_token_id") == EXPECTED_TOKEN_ID
        and result.get("result", {}).get("agreement", {}).get("rank_guard_selected_token_id") == EXPECTED_TOKEN_ID
        and result.get("result", {}).get("agreement", {}).get("reference_selected_token_id") == EXPECTED_TOKEN_ID
        and result.get("result", {}).get("layer23_output_sha256") == EXPECTED_LAYER23_OUTPUT_SHA256,
        "selected-token result semantics changed",
    )
    require(
        terminal.get("status") == "PASS_EXACT_POST_TRANSFORMER_TOKEN_SELECTION_COMPLETE"
        and terminal.get("result_sha256") == EXPECTED_RESULT_SHA256
        and terminal.get("selected_token_cardinality") == 1,
        "selected-token terminal linkage changed",
    )
    require(
        consumed.get("status") == "CONSUMED_BEFORE_WORKLOAD"
        and consumed.get("grant_sha256") == sources["grant0005_grant"]["sha256"]
        and consumed.get("package_tree_root_sha256") == sources["grant0005_package"]["tree_root_sha256"]
        and grant.get("subject", {}).get("package_tree_root_sha256")
        == sources["grant0005_package"]["tree_root_sha256"],
        "grant0005 consumption lineage changed",
    )
    require(
        sources["grant0005_prerequisite"]["sha256"] == consumed.get("prerequisite_manifest", {}).get("sha256")
        and prerequisite.get("authority_identity") == consumed.get("authority_identity"),
        "grant0005 prerequisite lineage changed",
    )

    frontier = prerequisite.get("layer23_terminal_frontier", {})
    frontier_records = frontier.get("records", {})
    require(
        frontier.get("package", {}).get("tree_root_sha256") == sources["layer23_package"]["tree_root_sha256"]
        and frontier_records.get(sources["layer23_checkpoint"]["path"]) == EXPECTED_CHECKPOINT_SHA256
        and frontier_records.get(sources["layer23_output"]["path"]) == EXPECTED_LAYER23_OUTPUT_SHA256
        and frontier_records.get(sources["layer23_result"]["path"]) == sources["layer23_result"]["sha256"],
        "Layer-23 prerequisite frontier changed",
    )
    require(
        sources["layer23_checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
        and sources["layer23_output"]["sha256"] == EXPECTED_LAYER23_OUTPUT_SHA256
        and checkpoint.get("status") == "PUBLISHED_AUTHENTICATED_THROUGH_LAYER23_END_OF_TRANSFORMER_STACK"
        and checkpoint.get("completed_unit_keys") == [f"position-00/layer-{index:02d}" for index in range(24)]
        and checkpoint.get("next_incomplete_unit") is None
        and layer23_result.get("key") == "position-00/layer-23",
        "Layer-23 checkpoint lineage changed",
    )

    units = checkpoint.get("units")
    resume = checkpoint.get("resume_state")
    kv_bindings = manifest.get("causal_kv_publications")
    require(
        isinstance(units, list)
        and len(units) == 24
        and isinstance(resume, dict)
        and isinstance(resume.get("kv_layers"), list)
        and len(resume["kv_layers"]) == 24
        and isinstance(kv_bindings, list)
        and len(kv_bindings) == 24,
        "24-layer causal state is incomplete",
    )
    require(
        hashlib.sha256(checkpoint_canonical_bytes(resume["kv_layers"])).hexdigest()
        == resume.get("kv_state_sha256"),
        "checkpoint K/V state digest changed",
    )
    previous_output: str | None = None
    for index, (unit, checkpoint_kv, binding) in enumerate(zip(units, resume["kv_layers"], kv_bindings, strict=True)):
        key = f"position-00/layer-{index:02d}"
        require(
            unit.get("key") == key
            and unit.get("ordinal") == index
            and unit.get("validation_status") == "PASS_AUTHENTICATED"
            and unit.get("input_sha256") == (unit.get("input_sha256") if index == 0 else previous_output)
            and unit.get("dependency", {}).get("key") == (None if index == 0 else f"position-00/layer-{index - 1:02d}")
            and binding.get("key") == key
            and binding.get("ordinal") == index,
            f"causal unit lineage changed: {key}",
        )
        if index:
            require(unit.get("dependency", {}).get("output_sha256") == previous_output, f"dependency hash changed: {key}")
        previous_output = unit.get("output", {}).get("sha256")
        require(isinstance(previous_output, str) and len(previous_output) == 64, f"unit output absent: {key}")
        for name in ("k_s8", "v_s8"):
            require(
                checkpoint_kv.get(name, {}).get("sha256") == binding.get(name, {}).get("sha256"),
                f"checkpoint K/V binding changed: {key}/{name}",
            )
            validate_record(artifact_root, binding.get(name), f"{key}/{name}")

    inventory = load_json(paths["attempt0012_inventory"])
    inventory_records = {
        (f"{root_record['root']}/{item['path']}", item["bytes"], item["sha256"])
        for root_record in inventory.get("roots", [])
        for item in root_record.get("files", [])
    }
    for binding in kv_bindings[:11]:
        require(binding.get("source") == "ATTEMPT_0012_AUTHENTICATED_RUNTIME", "attempt-0012 prefix label changed")
        for name in ("k_s8", "v_s8"):
            record = binding[name]
            require(
                (record["path"], record["bytes"], record["sha256"]) in inventory_records,
                f"attempt-0012 inventory binding absent: {binding['key']}/{name}",
            )
    require(
        all(item.get("source") == "APPEND_ONLY_AUTHENTICATED_SUCCESSOR_PUBLICATION" for item in kv_bindings[11:]),
        "successor publication labels changed",
    )

    position01 = manifest.get("position01_contract")
    expected_schedule = [f"position-01/layer-{index:02d}" for index in range(24)] + [
        "position-01/post-transformer-token-selection"
    ]
    require(
        isinstance(position01, dict)
        and position01.get("absolute_position") == 1
        and position01.get("input_token_id") == EXPECTED_TOKEN_ID
        and position01.get("causal_kv_state_sha256") == resume.get("kv_state_sha256")
        and position01.get("ordered_command_schedule") == expected_schedule
        and position01.get("execution_state") == "UNSUPPORTED_AWAITING_SEPARATE_RUNTIME_AUTHORITY"
        and position01.get("runtime_pass", {}).get("fail_closed") is True
        and position01.get("runtime_pass", {}).get("authority_cardinality") == 0,
        "position-01 input or fail-closed schedule changed",
    )
    output_namespace = artifact_root / safe_relative(position01.get("output_namespace"))
    require(not os.path.lexists(output_namespace), "position-01 output namespace must remain absent")
    guard = position01.get("resource_guard")
    require(
        isinstance(guard, dict)
        and guard.get("required_available_bytes")
        == guard.get("runtime_output_max_bytes")
        + guard.get("temporary_write_max_bytes")
        + guard.get("package_max_bytes")
        + guard.get("safety_margin_bytes"),
        "resource-guard arithmetic changed",
    )
    require(
        guard.get("construction_live_capacity_claim") == "NOT_MADE"
        and guard.get("enforcement") == "RECHECK_IMMEDIATELY_BEFORE_ANY_OUTPUT_OR_WORKLOAD"
        and position01.get("runtime_pass", {}).get("resource_guard_required_before_output_or_workload")
        is True,
        "resource-guard enforcement phase changed",
    )
    require(
        manifest.get("stage_status")
        == {
            "rtl": "in_progress",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "stage1_completion_claimed": False,
            "stage2_completion_claimed": False,
        },
        "stage status changed",
    )
    return {
        "activity": ZERO_ACTIVITY,
        "causal_kv_layer_count": 24,
        "execution_performed": False,
        "package_tree_root_sha256": tree_root,
        "schema": "ace2-position01-causal-state-package-validation-v1",
        "selected_token_id": EXPECTED_TOKEN_ID,
        "status": "PASS_CONSTRUCTION_ONLY_POSITION01_CAUSAL_STATE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate(args.package.resolve(), args.artifact_root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
