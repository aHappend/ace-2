#!/usr/bin/env python3
"""Freeze and verify the S5 AMLT submission-client provenance deviation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/bf16-full-finetune-successor-s5"
PROVENANCE = BUILD / "client-provenance-20260808T151648Z"
OUTPUT = BUILD / "submission-client-provenance-audit-20260808T151648Z.json"
SUBMITTED_ARCHIVE = PROVENANCE / "submitted-amlt-launcher"
CANONICAL_ARCHIVE = PROVENANCE / "canonical-amlt-launcher"
SUBMITTED_EXTERNAL = Path("/tmp/ace2-amlt-11.17.0-7292ba255b7a")
CANONICAL_EXTERNAL = Path("/home/argustest/argus-ir01-sol-engine/.venv-amlt-11.17/bin/amlt")
SUBMITTED_SITE = Path("/tmp/ace2-amlt-11.17.0-site-7292ba255b7a")
CANONICAL_SITE = Path("/home/argustest/argus-ir01-sol-engine/.venv-amlt-11.17/lib/python3.13/site-packages")
SUBMITTED_MODULE = SUBMITTED_SITE / "amlt/amlt.py"
CANONICAL_MODULE = CANONICAL_SITE / "amlt/amlt.py"
SUBMITTED_METADATA = SUBMITTED_SITE / "amlt-11.17.0.dist-info/METADATA"
CANONICAL_METADATA = CANONICAL_SITE / "amlt-11.17.0.dist-info/METADATA"
SUBMITTED_RECORD = SUBMITTED_SITE / "amlt-11.17.0.dist-info/RECORD"
CANONICAL_RECORD = CANONICAL_SITE / "amlt-11.17.0.dist-info/RECORD"
AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-attempt-operator-authority.json"
PRE_CANONICAL = BUILD / "preexecution-20260808T144545Z/before-rerun/package-audit.json"
PRE_SUBMITTED = BUILD / "package-audit.json"
RAW = BUILD / "backend-submission.raw.txt"
RESULT = BUILD / "backend-submission-result.json"
TERMINAL_AUDIT = BUILD / "backend-terminal-audit-20260808T150208Z.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"

SUBMITTED_SHA256 = "91791a9eac3b6dda79221a0157a0e9f4296ef7137ebda9a3ffa5b391358c1d76"
CANONICAL_SHA256 = "29c4b23b6ac10971abce369910189567976ae1638393a8f03d6c9b2715814260"
MODULE_SHA256 = "7176b4947e5c7d801ca2c669fc0a4d9ef36811528e8fa6a037bc3d1ea9c70d8b"
METADATA_SHA256 = "902e85139902db83925b3367b36c02063d66f879990ca8b00b865756f523e39e"


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


def record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def version(path: Path) -> str:
    result = subprocess.run(
        [str(path), "--version"],
        cwd="/home/argustest/argus-ir01-sol-engine",
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (result.stdout + result.stderr).strip()


def metadata_identity(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Name: "):
            fields["name"] = line.removeprefix("Name: ")
        elif line.startswith("Version: "):
            fields["version"] = line.removeprefix("Version: ")
        if len(fields) == 2:
            break
    return fields


def build_audit() -> dict[str, Any]:
    required = (
        SUBMITTED_ARCHIVE, CANONICAL_ARCHIVE, SUBMITTED_EXTERNAL, CANONICAL_EXTERNAL,
        SUBMITTED_MODULE, CANONICAL_MODULE, SUBMITTED_METADATA, CANONICAL_METADATA,
        SUBMITTED_RECORD, CANONICAL_RECORD, AUTHORITY, PRE_CANONICAL, PRE_SUBMITTED,
        RAW, RESULT, TERMINAL_AUDIT, PIPELINE,
    )
    for path in required:
        require(path.is_file(), f"required provenance input is absent: {path}")

    authority = load(AUTHORITY)
    canonical_audit = load(PRE_CANONICAL)
    submitted_audit = load(PRE_SUBMITTED)
    submission = load(RESULT)
    terminal = load(TERMINAL_AUDIT)
    pipeline = load(PIPELINE)

    checks = {
        "archive.submitted_launcher_exact": sha256(SUBMITTED_ARCHIVE) == SUBMITTED_SHA256,
        "archive.canonical_launcher_exact": sha256(CANONICAL_ARCHIVE) == CANONICAL_SHA256,
        "external.submitted_launcher_matches_archive": sha256(SUBMITTED_EXTERNAL) == sha256(SUBMITTED_ARCHIVE),
        "external.canonical_launcher_matches_archive": sha256(CANONICAL_EXTERNAL) == sha256(CANONICAL_ARCHIVE),
        "launcher.hashes_distinct": SUBMITTED_SHA256 != CANONICAL_SHA256,
        "launcher.submitted_uses_temp_site_injection": str(SUBMITTED_SITE) in SUBMITTED_ARCHIVE.read_text(encoding="utf-8"),
        "launcher.canonical_uses_project_local_interpreter": CANONICAL_ARCHIVE.read_text(encoding="utf-8").splitlines()[0] == "#!/home/argustest/argus-ir01-sol-engine/.venv-amlt-11.17/bin/python",
        "distribution.submitted_version_11_17_0": metadata_identity(SUBMITTED_METADATA) == {"name": "amlt", "version": "11.17.0"},
        "distribution.canonical_version_11_17_0": metadata_identity(CANONICAL_METADATA) == {"name": "amlt", "version": "11.17.0"},
        "distribution.metadata_bytes_equal": sha256(SUBMITTED_METADATA) == sha256(CANONICAL_METADATA) == METADATA_SHA256,
        "distribution.entry_module_bytes_equal": sha256(SUBMITTED_MODULE) == sha256(CANONICAL_MODULE) == MODULE_SHA256,
        "distribution.record_files_differ": sha256(SUBMITTED_RECORD) != sha256(CANONICAL_RECORD),
        "version.submitted_reports_11_17_0": version(SUBMITTED_EXTERNAL) == "amulet version 11.17.0",
        "version.canonical_reports_11_17_0": version(CANONICAL_EXTERNAL) == "amulet version 11.17.0",
        "preflight.canonical_project_dump_passed": canonical_audit.get("status") == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S5_PACKAGE" and canonical_audit.get("check_count") == 92 and canonical_audit.get("failed_checks") == [] and all(canonical_audit.get("checks", {}).values()) and canonical_audit.get("amlt", {}).get("executable_sha256") == CANONICAL_SHA256 and canonical_audit.get("amlt", {}).get("dump_output_sha256") == "07361d5858ed5aeeee015ebfb492ad6cbd7d98444b0d28582a47e421e4efb3c0",
        "preflight.submitted_client_rerun_passed": submitted_audit.get("status") == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S5_PACKAGE" and submitted_audit.get("check_count") == 92 and submitted_audit.get("failed_checks") == [] and all(submitted_audit.get("checks", {}).values()) and submitted_audit.get("amlt", {}).get("executable_sha256") == SUBMITTED_SHA256 and submitted_audit.get("amlt", {}).get("dump_output_sha256") == "07361d5858ed5aeeee015ebfb492ad6cbd7d98444b0d28582a47e421e4efb3c0",
        "authority.binds_submitted_client_hash": authority.get("preexecution_evidence", {}).get("amlt_executable_sha256") == SUBMITTED_SHA256,
        "authority.does_not_claim_canonical_hash": authority.get("preexecution_evidence", {}).get("amlt_executable_sha256") != CANONICAL_SHA256,
        "submission.raw_exact": sha256(RAW) == "d5b900ea90aa42359ea276490f987e0d4edfeb4e468ff395c11b3757b7f1d7c5",
        "submission.result_binds_raw": submission.get("raw_output_sha256") == sha256(RAW),
        "terminal.audit_complete_no_go": terminal.get("status") == "TERMINAL_S5_DEV_QUALITY_NO_GO_EVIDENCE_COMPLETE" and terminal.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE" and terminal.get("quality", {}).get("dev", {}).get("hard_pass_count") == 27,
        "pipeline.current_stage_specification": pipeline.get("current_stage") == "specification",
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    require(not failed, f"S5 client provenance audit failed checks: {failed}")

    return {
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "PASS_BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION",
        "classification": "BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION",
        "claim_boundary": "The sole S5 submission used the archived temporary AMLT launcher, not the explicitly directed canonical project-local launcher. Both report AMLT 11.17.0 and share identical amlt.amlt module and distribution METADATA bytes, but their launcher hashes, shebang environments, installation locations, and RECORD files differ. Transport equivalence is not claimed. This provenance result does not alter the terminal 27/56 DEV_QUALITY_GATE_FAILURE and grants no retry or downstream authority.",
        "check_count": len(checks),
        "failed_check_count": 0,
        "checks": dict(sorted(checks.items())),
        "classification_limits": {
            "canonical_executable_used_for_submission": False,
            "transport_equivalence_claimed": False,
            "same_reported_version": True,
            "same_entry_module_bytes": True,
            "same_distribution_metadata_bytes": True,
            "same_installation_record": False,
            "quality_pass_claimed": False,
            "retry_or_resubmission_permitted": False,
        },
        "submitted_client": {
            "original_path": str(SUBMITTED_EXTERNAL),
            "archive": record(SUBMITTED_ARCHIVE),
            "sha256": SUBMITTED_SHA256,
            "shebang": SUBMITTED_ARCHIVE.read_text(encoding="utf-8").splitlines()[0],
            "resolved_interpreter": "/home/argustest/miniconda3/bin/python3.13",
            "distribution_path": str(SUBMITTED_SITE),
            "distribution": metadata_identity(SUBMITTED_METADATA),
            "entry_module_sha256": sha256(SUBMITTED_MODULE),
            "metadata_sha256": sha256(SUBMITTED_METADATA),
            "record_sha256": sha256(SUBMITTED_RECORD),
            "version_output": version(SUBMITTED_EXTERNAL),
        },
        "canonical_client": {
            "original_path": str(CANONICAL_EXTERNAL),
            "archive": record(CANONICAL_ARCHIVE),
            "sha256": CANONICAL_SHA256,
            "shebang": CANONICAL_ARCHIVE.read_text(encoding="utf-8").splitlines()[0],
            "resolved_interpreter": "/home/argustest/miniconda3/bin/python3.13",
            "distribution_path": str(CANONICAL_SITE),
            "distribution": metadata_identity(CANONICAL_METADATA),
            "entry_module_sha256": sha256(CANONICAL_MODULE),
            "metadata_sha256": sha256(CANONICAL_METADATA),
            "record_sha256": sha256(CANONICAL_RECORD),
            "version_output": version(CANONICAL_EXTERNAL),
        },
        "bound_evidence": {
            "canonical_preexecution_project_dump": record(PRE_CANONICAL),
            "submitted_client_preexecution_rerun": record(PRE_SUBMITTED),
            "authority": record(AUTHORITY),
            "submission_raw": record(RAW),
            "submission_result": record(RESULT),
            "terminal_audit": record(TERMINAL_AUDIT),
            "pipeline_state": record(PIPELINE),
        },
        "regeneration_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_bf16_full_finetune_successor_s5_client_provenance.py --check",
    }


def write(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists() and not path.with_suffix(path.suffix + ".sha256").exists(), f"refusing to overwrite immutable provenance audit: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    observed = build_audit()
    if args.check:
        require(companion_matches(OUTPUT), "S5 client provenance audit companion differs")
        existing = load(OUTPUT)
        for key in ("status", "classification", "check_count", "failed_check_count", "checks", "classification_limits", "submitted_client", "canonical_client", "bound_evidence"):
            require(existing.get(key) == observed.get(key), f"S5 client provenance audit field differs: {key}")
        print(f"ACE2_S5_CLIENT_PROVENANCE_CHECK_PASS checks={observed['check_count']}")
        return 0
    write(OUTPUT, observed)
    print(f"ACE2_S5_CLIENT_PROVENANCE_WRITTEN checks={observed['check_count']} sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
