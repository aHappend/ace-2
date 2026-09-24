#!/usr/bin/env python3
"""Fail closed before any S7 data/probe or execution-package construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = Path("design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S7_DORA_SFT_DPO_SPECIFICATION_PACKAGE.json")
MANIFEST = Path("design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S7_PACKAGE_MANIFEST.json")
REVIEW = Path("research/raw/specification/qwen25-bf16-successor-s7-specification-fresh-review-binding-20260809.json")

FROZEN_S7_HASHES = {
    SPEC: "c144781eda57946960453ac671501b2facb2a0dac8657b85efdfb5e65a86457a",
    MANIFEST: "40965c7bab3d381f1dabbbb102d6e40e78796f73503780597395c3274a4ce22d",
    REVIEW: "6c2102886465e7dbc9ee012f205ee14ca7115ef3f70ae193c6619f3b425ccb62",
}

AUTHORITY_SEARCH_ROOT = Path("research/raw/specification")
OPERATIONAL_PARENT_ROOTS = (
    Path("build"),
    Path("build/offline-preflight"),
    Path("pilot"),
    Path("research/execution"),
    Path("research/training"),
)
ALLOWED_S7_OPERATIONAL_PATHS = {
    Path("build/bf16-successor-s7-specification-package"),
}
ATTEMPT_MARKER_NAME = "ATTEMPT_CONSUMPTION_MARKER.json"
S7_IDENTIFIERS = ("successor-s7", "successor_s7", "bf16-successor-s7")
EXECUTION_TOKENS = ("amlt", "launch", "train.py", "torchrun", "evaluate", "worker")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def checksum_companion_matches(relative_path: Path, expected: str) -> bool:
    source = ROOT / relative_path
    companion = source.with_suffix(source.suffix + ".sha256")
    if not companion.is_file():
        return False
    fields = companion.read_text(encoding="utf-8").strip().split()
    return len(fields) == 2 and fields[0] == expected and fields[1] == source.name


def authority_candidates() -> list[str]:
    root = ROOT / AUTHORITY_SEARCH_ROOT
    if not root.is_dir():
        return []
    candidates = []
    for path in root.iterdir():
        name = path.name.casefold()
        if (
            path.is_file()
            and path.suffix == ".json"
            and any(identifier in name for identifier in S7_IDENTIFIERS)
            and ("authority" in name or "authorization" in name)
        ):
            candidates.append(relative(path))
    return sorted(candidates)


def is_s7_path(path: Path) -> bool:
    folded = path.as_posix().casefold()
    return any(identifier in folded for identifier in S7_IDENTIFIERS)


def s7_operational_paths() -> list[Path]:
    found: set[Path] = set()
    for parent_root in OPERATIONAL_PARENT_ROOTS:
        absolute_root = ROOT / parent_root
        if not absolute_root.is_dir():
            continue
        for child in absolute_root.iterdir():
            relative_child = child.relative_to(ROOT)
            if is_s7_path(relative_child):
                found.add(relative_child)
    return sorted(found)


def unexpected_s7_operational_paths() -> list[str]:
    return [
        path.as_posix()
        for path in s7_operational_paths()
        if path not in ALLOWED_S7_OPERATIONAL_PATHS
    ]


def s7_attempt_markers() -> list[str]:
    markers: set[str] = set()
    for operational_path in s7_operational_paths():
        absolute_root = ROOT / operational_path
        if not os.path.lexists(absolute_root):
            continue
        if not absolute_root.is_dir():
            if absolute_root.name == ATTEMPT_MARKER_NAME:
                markers.add(relative(absolute_root))
            continue
        for directory, directory_names, file_names in os.walk(absolute_root, followlinks=False):
            entries = [*directory_names, *file_names]
            for entry in entries:
                if entry != ATTEMPT_MARKER_NAME:
                    continue
                candidate = Path(directory) / entry
                if is_s7_path(candidate.relative_to(ROOT)) and os.path.lexists(candidate):
                    markers.add(relative(candidate))
    return sorted(markers)


def ancestor_pids() -> set[int]:
    ancestors = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            stat_fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()
            pid = int(stat_fields[3])
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            break
    return ancestors


def active_s7_execution_pids() -> list[int]:
    excluded = ancestor_pids()
    matches = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").casefold()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(identifier in command for identifier in S7_IDENTIFIERS) and any(
            token in command for token in EXECUTION_TOKENS
        ):
            matches.append(pid)
    return sorted(matches)


def audit() -> tuple[dict[str, Any], bool]:
    hash_results = {
        path.as_posix(): {
            "expected_sha256": expected,
            "observed_sha256": sha256(ROOT / path) if (ROOT / path).is_file() else None,
            "checksum_companion_matches": checksum_companion_matches(path, expected),
        }
        for path, expected in FROZEN_S7_HASHES.items()
    }
    hashes_match = all(
        result["observed_sha256"] == result["expected_sha256"] and result["checksum_companion_matches"]
        for result in hash_results.values()
    )

    review = load_json(ROOT / REVIEW) if hashes_match else {}
    review_binding_matches = bool(
        review.get("accepted_package", {}).get("specification_sha256") == FROZEN_S7_HASHES[SPEC]
        and review.get("accepted_package", {}).get("manifest_sha256") == FROZEN_S7_HASHES[MANIFEST]
        and review.get("review", {}).get("producer_role") == "reviewer"
        and review.get("review", {}).get("status") == "done"
        and "does not authorize data creation" in review.get("acceptance_effect", "")
    )

    candidates = authority_candidates()
    operational_paths = unexpected_s7_operational_paths()
    markers = s7_attempt_markers()
    execution_pids = active_s7_execution_pids()
    zero_state = not operational_paths and not markers and not execution_pids
    clean_blocked_state = hashes_match and review_binding_matches and not candidates and zero_state

    if not hashes_match or not review_binding_matches or not zero_state:
        status = "FAIL_PREAUTHORITY_INTEGRITY_OR_STATE"
    elif candidates:
        status = "STOP_AUTHORITY_CANDIDATE_REQUIRES_INDEPENDENT_PROVENANCE_REVIEW"
    else:
        status = "BLOCKED_FRESH_EXTERNAL_AUTHORITY_ABSENT"

    report = {
        "schema_version": 1,
        "audit_scope": "READ_ONLY_S7_PREAUTHORITY_GATE",
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "frozen_s7_hashes": hash_results,
        "checks": {
            "frozen_hashes_and_companions_match": hashes_match,
            "fresh_reviewer_binding_matches_and_grants_no_materialization_authority": review_binding_matches,
            "fresh_external_authority_candidate_count": len(candidates),
            "unexpected_s7_operational_path_count": len(operational_paths),
            "s7_attempt_marker_count": len(markers),
            "active_local_s7_execution_process_count": len(execution_pids),
        },
        "authority_candidate_paths": candidates,
        "unexpected_s7_operational_paths": operational_paths,
        "s7_attempt_marker_paths": markers,
        "active_local_s7_execution_pids": execution_pids,
        "claim_boundary": (
            "Read-only pre-authority measurement only. It creates no data/probe package, execution tree, namespace, "
            "job, retry, resume, attempt marker, model/evaluator execution, checkpoint, quantization, RTL, PPA, "
            "FPGA, U280, review acceptance, or execution authority."
        ),
    }
    return report, clean_blocked_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-blocked",
        action="store_true",
        help="Return success only for the clean, zero-state, authority-absent condition.",
    )
    args = parser.parse_args()

    report, clean_blocked_state = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.expect_blocked:
        return 0 if clean_blocked_state else 1
    if report["status"] == "BLOCKED_FRESH_EXTERNAL_AUTHORITY_ABSENT":
        return 2
    if report["status"] == "STOP_AUTHORITY_CANDIDATE_REQUIRES_INDEPENDENT_PROVENANCE_REVIEW":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
