#!/usr/bin/env python3
"""Fail closed when protected ACE-2 publication companions are stale."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
CERTIFICATE = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.json"
CANONICAL_RUNNER_ROOT = ROOT / "evidence/canonical_sky130"

# These runners were already consumed and sealed before the publication guard
# existed.  Their exact hashes are historical evidence and must not be edited.
SEALED_PRE_GUARD_RUNNERS = {
    "evidence/canonical_sky130/rtl-critical-cone-repaired-tree-canonical-sky130-ppa-v1/run_once.sh":
        "31ca66dfdc447bb4df704544b3237b908567e90681c318446bfadeadb727d6a7",
    "evidence/canonical_sky130/rtl-final-sumsq-repaired-tree-canonical-sky130-ppa-v1/run_once.sh":
        "bf6cb06050d923b38a880b0e1654e0424ec28e63acce705f3f4428883840c175",
    "evidence/canonical_sky130/rtl-full-qwen-final-tree-canonical-sky130-ppa-v1/run_once.sh":
        "db2cbebd78546596f11e612289ec7379d36cbe24ed5fec3378217c185a9699f4",
    "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1/run_once.sh":
        "17e73ddd6dc2bfbc05fd2163d29a324d39d619018cdc1dd422146fe26bd2701f",
    "evidence/canonical_sky130/rtl-rmsnorm-repaired-tree-canonical-sky130-ppa-v1/run_once.sh":
        "cace30b16e86a7fa559a464058d2c828744480fd9b3bcfc6b15934e4a4602a61",
}
SEALED_MARKER_PATHS = (
    "authorization_consumed/CONSUMPTION_MARKER.json",
    "authorization/AUTHORIZATION_CONSUMED.txt",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_public_hash(value: dict[str, Any]) -> str:
    candidate = copy.deepcopy(value)
    candidate.setdefault("integrity", {})["canonical_sha256"] = None
    return hashlib.sha256(encode_json(candidate)).hexdigest()


def verify_companion(target: Path, companion: Path) -> None:
    require(target.is_file(), f"missing protected target: {target.relative_to(ROOT)}")
    require(companion.is_file(), f"missing protected companion: {companion.relative_to(ROOT)}")
    expected = f"{sha256_file(target)}  {target.name}\n"
    require(
        companion.read_text(encoding="utf-8") == expected,
        f"stale protected companion: {companion.relative_to(ROOT)}",
    )


def is_integrity_guard_line(line: str) -> bool:
    return (
        "make publication-authorization-preflight" in line
        or "make publication-integrity-check" in line
        or (
            "tools/check_publication_integrity.py" in line
            and re.search(r"(?:^|\s)--check(?:\s|$)", line) is not None
        )
    )


def is_marker_creation_line(line: str) -> bool:
    if "$MARKER_DIR" in line and re.search(r"(?:^|\s)mkdir(?:\s|$)", line):
        return True
    if "$MARKER" not in line:
        return False
    return (
        ">" in line
        or re.search(r"(?:^|\s)(?:touch|install|cp|mv)(?:\s|$)", line) is not None
    )


def verify_runner_ordering_text(text: str, label: str) -> None:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    guard_lines = [index for index, line in enumerate(lines) if is_integrity_guard_line(line)]
    marker_lines = [index for index, line in enumerate(lines) if is_marker_creation_line(line)]
    require(marker_lines, f"{label} has no detectable authorization marker creation")
    require(guard_lines, f"{label} lacks a publication-integrity pre-marker guard")
    require(
        min(guard_lines) < min(marker_lines),
        f"{label} runs publication integrity only after authorization marker creation",
    )


def verify_exactly_once_runner_ordering() -> dict[str, int]:
    sealed = 0
    guarded = 0
    runners = sorted(CANONICAL_RUNNER_ROOT.glob("*/run_once.sh"))
    require(runners, "no canonical exactly-once runners found")
    for runner in runners:
        relative = runner.relative_to(ROOT).as_posix()
        expected_sealed_hash = SEALED_PRE_GUARD_RUNNERS.get(relative)
        if expected_sealed_hash is not None:
            require(sha256_file(runner) == expected_sealed_hash, f"sealed historical runner changed: {relative}")
            require(
                any((runner.parent / marker).is_file() for marker in SEALED_MARKER_PATHS),
                f"sealed historical runner lacks its consumption marker: {relative}",
            )
            sealed += 1
            continue
        verify_runner_ordering_text(runner.read_text(encoding="utf-8"), relative)
        guarded += 1
    return {"guarded_future_runners": guarded, "sealed_historical_runners": sealed}


def verify_integrity_guard_artifacts(public: dict[str, Any]) -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    require(isinstance(certificate, dict), "FINAL_PRODUCT_CERTIFICATE.json is not a JSON object")
    guard = certificate.get("publication_integrity_guard")
    require(isinstance(guard, dict), "final certificate lacks publication_integrity_guard")
    require(guard.get("status") == "PASS", "publication_integrity_guard is not PASS")
    require(
        guard.get("ordering") == "integrity_check_before_exactly_once_authorization_marker_creation",
        "publication_integrity_guard ordering contract differs",
    )
    require(public.get("publication_integrity_guard") == guard, "PUBLIC_STATUS guard differs from certificate")
    for key in ("makefile", "checker", "ordering_regression"):
        record = guard.get(key)
        require(isinstance(record, dict), f"publication_integrity_guard lacks {key}")
        relative = record.get("path")
        require(isinstance(relative, str), f"publication_integrity_guard {key} path is invalid")
        path = ROOT / relative
        require(path.is_file(), f"publication_integrity_guard artifact missing: {relative}")
        require(record.get("bytes") == path.stat().st_size, f"guard artifact byte count differs: {relative}")
        require(record.get("sha256") == sha256_file(path), f"guard artifact hash differs: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.parse_args()

    pairs = [
        (ROOT / "design/RTL_MANIFEST.json", ROOT / "design/RTL_MANIFEST.sha256"),
        (ROOT / "design/RTL_TRACEABILITY.md", ROOT / "design/RTL_TRACEABILITY.sha256"),
        (ROOT / "research/PIPELINE_STATE.json", ROOT / "research/PIPELINE_STATE.sha256"),
    ]
    certificate = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.json"
    certificate_companion = ROOT / "research/FINAL_PRODUCT_CERTIFICATE.sha256"
    if certificate.exists() or certificate_companion.exists():
        pairs.append((certificate, certificate_companion))

    for target, companion in pairs:
        verify_companion(target, companion)

    public = json.loads(PUBLIC_STATUS.read_text(encoding="utf-8"))
    require(isinstance(public, dict), "PUBLIC_STATUS.json is not a JSON object")
    runner_status = verify_exactly_once_runner_ordering()
    verify_integrity_guard_artifacts(public)
    require(
        public.get("integrity", {}).get("canonical_sha256") == canonical_public_hash(public),
        "PUBLIC_STATUS canonical integrity is stale",
    )

    records = {record.get("path"): record for record in public.get("artifact_hashes", [])}
    for target, companion in pairs:
        for path in (target, companion):
            relative = path.relative_to(ROOT).as_posix()
            record = records.get(relative)
            require(isinstance(record, dict), f"PUBLIC_STATUS lacks protected artifact record: {relative}")
            require(record.get("bytes") == path.stat().st_size, f"artifact byte count differs: {relative}")
            require(record.get("sha256") == sha256_file(path), f"artifact hash differs: {relative}")

    print(json.dumps({
        **runner_status,
        "protected_companions": len(pairs),
        "public_integrity": "PASS",
        "status": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
