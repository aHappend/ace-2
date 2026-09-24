#!/usr/bin/env python3
"""Verify sealed V17 evidence after an in-place task-id hygiene redaction.

The official V17 campaign is immutable.  This verifier never rewrites it.  It
normalizes only redacted ``task_id`` fields in four explicitly allowlisted JSON
files, using the public Engineer-task identifier, and then checks the original
sealed hashes plus the artifact records that bind the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "option_b_source_relative_v2_per16_symmetric_scale32_w4a8_v17"
OFFICIAL_ROOT = ROOT / "build/stage1-option-b-source-relative-v2-per16-symmetric-scale32-w4a8-v17"
QUALITY_ROOT = OFFICIAL_ROOT / "quality-campaign-0001"
TASK_PATH = ROOT / "design/OPTION_B_SOURCE_RELATIVE_V2_PER16_SYMMETRIC_SCALE32_W4A8_V17_ENGINEER_TASK.json"

SEALED_JSON_HASHES = {
    OFFICIAL_ROOT / "fresh_reviewer_submission.json": "a6476fe7526a19d4089bda973e5501785b651d73bfffdfaa675898f6a062ea53",
    OFFICIAL_ROOT / "construction_started.json": "a6845cdb111007fd68244969afc941537210172fe2c567cefceb74c1c371da01",
    OFFICIAL_ROOT / "predeclared_contract.json": "de8d158fba7ce4df1e65063063519d98b2d94b2b932fc91e4af18e8b2c026816",
    QUALITY_ROOT / "execution_started.json": "449978f8567f96a633fbb7275586a7fb76e2294113b3fa4c294a62f8461240ab",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path.relative_to(ROOT)}")
    return value


def normalize_task_id(value: Any, task_id: str, *, key: str | None = None) -> tuple[Any, int]:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        count = 0
        for child_key, child_value in value.items():
            normalized_value, child_count = normalize_task_id(child_value, task_id, key=child_key)
            normalized[child_key] = normalized_value
            count += child_count
        return normalized, count
    if isinstance(value, list):
        normalized_items = []
        count = 0
        for item in value:
            normalized_item, child_count = normalize_task_id(item, task_id)
            normalized_items.append(normalized_item)
            count += child_count
        return normalized_items, count
    if isinstance(value, str) and "<REDACTED:" in value:
        require(key == "task_id", "redaction marker outside an allowlisted task_id field")
        return task_id, 1
    return value, 0


def sealed_file(path: Path, task_id: str) -> dict[str, Any]:
    require(path.is_file(), f"sealed artifact missing: {path.relative_to(ROOT)}")
    raw = path.read_bytes()
    raw_sha256 = sha256_bytes(raw)
    if path not in SEALED_JSON_HASHES:
        return {
            "raw_bytes": len(raw),
            "raw_sha256": raw_sha256,
            "sealed_bytes": len(raw),
            "sealed_sha256": raw_sha256,
            "normalization_applied": False,
            "normalized_task_id_field_count": 0,
        }

    value = json.loads(raw)
    require(isinstance(value, dict), f"allowlisted JSON object required: {path.relative_to(ROOT)}")
    require(canonical_bytes(value) == raw, f"non-canonical JSON differs: {path.relative_to(ROOT)}")
    normalized, replacement_count = normalize_task_id(value, task_id)
    sealed_raw = canonical_bytes(normalized)
    sealed_sha256 = sha256_bytes(sealed_raw)
    expected = SEALED_JSON_HASHES[path]
    require(sealed_sha256 == expected, f"sealed hash differs after task-id normalization: {path.relative_to(ROOT)}")
    require(raw_sha256 == expected or replacement_count > 0, f"unexplained allowlisted byte change: {path.relative_to(ROOT)}")
    return {
        "raw_bytes": len(raw),
        "raw_sha256": raw_sha256,
        "sealed_bytes": len(sealed_raw),
        "sealed_sha256": sealed_sha256,
        "normalization_applied": raw_sha256 != sealed_sha256,
        "normalized_task_id_field_count": replacement_count,
        "value": normalized,
    }


def verify_file_record(record: dict[str, Any], task_id: str) -> dict[str, Any]:
    require(set(record) == {"bytes", "path", "sha256"}, "artifact record shape differs")
    path = ROOT / record["path"]
    observed = sealed_file(path, task_id)
    require(observed["sealed_bytes"] == int(record["bytes"]), f"sealed byte count differs: {record['path']}")
    require(observed["sealed_sha256"] == record["sha256"], f"sealed artifact hash differs: {record['path']}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    task = load_json(TASK_PATH)
    require(task["candidate_id"] == CANDIDATE_ID, "Engineer task candidate differs")
    task_id = task["task_id"]
    require(isinstance(task_id, str) and task_id.startswith("task-") and "<REDACTED:" not in task_id, "public Engineer task id differs")

    normalized_records: dict[str, dict[str, Any]] = {}
    for path in SEALED_JSON_HASHES:
        observed = sealed_file(path, task_id)
        normalized_records[path.relative_to(ROOT).as_posix()] = {
            key: value for key, value in observed.items() if key != "value"
        }

    submission_path = OFFICIAL_ROOT / "fresh_reviewer_submission.json"
    submission = sealed_file(submission_path, task_id)["value"]
    require(submission["candidate_id"] == CANDIDATE_ID, "reviewer submission candidate differs")
    require(submission["status"] == "READY_FOR_FRESH_REVIEW", "reviewer submission status differs")
    require(submission["selected_policy_id_before_review"] is None, "V17 selected a policy before review")
    for key, value in submission.items():
        if isinstance(value, dict) and set(value) == {"bytes", "path", "sha256"}:
            verify_file_record(value, task_id)

    contract_path = OFFICIAL_ROOT / "predeclared_contract.json"
    wrapper = sealed_file(contract_path, task_id)["value"]
    require(wrapper["contract"]["candidate_id"] == CANDIDATE_ID, "predeclared contract candidate differs")
    require(wrapper["contract_sha256"] == sha256_bytes(canonical_bytes(wrapper["contract"])), "predeclared contract body hash differs")
    for record in wrapper["contract"]["artifacts"].values():
        verify_file_record(record, task_id)

    sums_path = QUALITY_ROOT / "SHA256SUMS"
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in sums, f"duplicate quality checksum entry: {name}")
        sums[name] = digest
    for name, expected in sums.items():
        observed = sealed_file(QUALITY_ROOT / name, task_id)
        require(observed["sealed_sha256"] == expected, f"quality checksum differs after normalization: {name}")

    result = load_json(QUALITY_ROOT / "RESULT.json")
    rubric = load_json(QUALITY_ROOT / "source_relative_noninferiority.json")
    require(result["status"] == "SOURCE_RELATIVE_NONINFERIOR_NO_GO", "terminal V17 result differs")
    require(result["selected_policy_id"] is None, "terminal V17 result selected a policy")
    require(rubric["source_relative_noninferiority_status"] == "NO_GO", "terminal V17 rubric differs")
    require(len(rubric["source_action_pass_regressions"]) == 2, "V17 source-regression count differs")

    output = {
        "schema_version": 1,
        "status": "PASS_REDACTION_AWARE_SEALED_V17_PROVENANCE",
        "candidate_id": CANDIDATE_ID,
        "quality_status": result["status"],
        "selected_policy_id": None,
        "source_action_pass_regression_count": 2,
        "normalization_scope": "four allowlisted canonical JSON files; task_id fields only; no files rewritten",
        "normalized_files": normalized_records,
        "quality_checksum_entry_count": len(sums),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output_path = args.output.resolve()
        require(output_path.is_relative_to(ROOT), "verification output must remain inside the repository")
        require(not output_path.is_relative_to(OFFICIAL_ROOT), "verification output may not enter the immutable V17 namespace")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
