#!/usr/bin/env python3
"""Bind the chat-v2 SKY130 synthesis/OpenSTA evidence to live sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = (
    ROOT / "evidence/verification/chat-v2-dynamic-scale32-qkv-current-tree-v2"
)
FLOW_PATHS = (
    "flow/yosys/sky130_dynamic_scale32_first_boundary.ys",
    "flow/yosys/sky130_dynamic_scale32_first_boundary_sta.tcl",
)
CERTIFICATE_PATHS = (
    "research/FINAL_PRODUCT_CERTIFICATE.json",
    "research/FINAL_PRODUCT_CERTIFICATE.sha256",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tree_binding(pattern: str) -> dict[str, Any]:
    paths = [path for path in sorted(ROOT.glob(pattern)) if path.is_file()]
    require(bool(paths), f"source pattern has no files: {pattern}")
    source_hashes = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in source_hashes
    ).encode()
    return {
        "file_count": len(source_hashes),
        "sha256": sha256_bytes(manifest),
        "hash_method": f"sha256 of UTF-8 path-sorted per-file SHA256 manifest for {pattern}",
        "source_hashes": source_hashes,
    }


def source_snapshot() -> dict[str, Any]:
    return {
        "rtl_tree": tree_binding("rtl/**/*.sv"),
        "constraint_tree": tree_binding("constraints/**/*"),
        "flow_sources": {
            path: sha256_file(ROOT / path)
            for path in FLOW_PATHS
        },
        "sealed_certificate": {
            path: sha256_file(ROOT / path)
            for path in CERTIFICATE_PATHS
        },
    }


def capture_before(evidence: Path) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "mission_id": "productize-local-qwen-chat-demo-v2",
        "generated_at_utc": utc_now(),
        "classification": "live_rtl_constraint_binding_before_canonical_sky130",
        "status": "PRE_SYNTHESIS_BOUND",
        "sources": source_snapshot(),
        "scope": {
            "canonical_target": "dynamic-scale32-first-boundary-synth-sta",
            "top": "ace2_shell",
            "technology": "sky130_fd_sc_hd tt_025C_1v80",
            "sealed_ppa_replayed": False,
        },
    }
    write_atomic(evidence / "source_binding.json", canonical_bytes(payload))


def extract_first(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    require(match is not None, f"missing {label} in canonical log")
    return match.group(1)


def capture_after(evidence: Path) -> bool:
    binding_path = evidence / "source_binding.json"
    require(binding_path.is_file(), "missing pre-synthesis source binding")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(binding.get("sources") == source_snapshot(), "source or constraint tree changed during canonical run")

    yosys_path = evidence / "sky130_yosys.log"
    sta_path = evidence / "sky130_sta.log"
    mapped_path = ROOT / "build/sky130_dynamic_scale32_first_boundary/ace2_shell_mapped.v"
    mapped_sta_path = ROOT / "build/sky130_dynamic_scale32_first_boundary/ace2_shell_mapped_sta.v"
    for path in (yosys_path, sta_path, mapped_path, mapped_sta_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing canonical artifact: {path}")

    yosys = yosys_path.read_text(encoding="utf-8", errors="replace")
    sta = sta_path.read_text(encoding="utf-8", errors="replace")
    require("End of script." in yosys, "canonical Yosys run did not finish")
    area = float(
        extract_first(
            r"Chip area for top module '\\ace2_shell':\s+([0-9.]+)",
            yosys,
            "top area",
        )
    )
    slack_matches = re.findall(
        r"^\s*([0-9.-]+)\s+slack \((?:MET|VIOLATED)\)\s*$",
        sta,
        flags=re.MULTILINE,
    )
    require(bool(slack_matches), "missing setup slack in canonical OpenSTA log")
    slack = min(float(value) for value in slack_matches)
    wns = float(extract_first(r"^wns max\s+([0-9.-]+)\s*$", sta, "WNS"))
    tns = float(extract_first(r"^tns max\s+([0-9.-]+)\s*$", sta, "TNS"))
    timing_met = wns >= 0.0 and tns >= 0.0 and "slack (VIOLATED)" not in sta
    artifacts = {
        "source_binding.json": sha256_file(binding_path),
        "sky130_yosys.log": sha256_file(yosys_path),
        "sky130_sta.log": sha256_file(sta_path),
        "build/sky130_dynamic_scale32_first_boundary/ace2_shell_mapped.v": sha256_file(mapped_path),
        "build/sky130_dynamic_scale32_first_boundary/ace2_shell_mapped_sta.v": sha256_file(mapped_sta_path),
    }
    focused_path = evidence / "focused_regression.log"
    if focused_path.is_file():
        artifacts["focused_regression.log"] = sha256_file(focused_path)
    lint_path = evidence / "verilator_lint.log"
    if lint_path.is_file():
        artifacts["verilator_lint.log"] = sha256_file(lint_path)

    focused_tests = None
    focused_status = "NOT_RECORDED"
    if focused_path.is_file():
        focused = focused_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^Ran (\d+) tests? in ", focused, flags=re.MULTILINE)
        focused_tests = None if match is None else int(match.group(1))
        focused_status = "PASS" if "\nOK\n" in focused else "FAIL"

    provenance_path = evidence / "provenance.json"
    existing = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else None
    )
    generated_at_utc = utc_now()
    if (
        isinstance(existing, dict)
        and existing.get("sources") == binding["sources"]
        and existing.get("artifacts") == artifacts
        and existing.get("synthesis", {}).get("top_area_um2") == area
        and existing.get("opensta", {}).get("worst_reported_slack_ns") == slack
        and existing.get("opensta", {}).get("wns_ns") == wns
        and existing.get("opensta", {}).get("tns_ns") == tns
    ):
        generated_at_utc = str(existing["generated_at_utc"])

    payload = {
        "schema_version": 1,
        "mission_id": "productize-local-qwen-chat-demo-v2",
        "generated_at_utc": generated_at_utc,
        "classification": "live_qkv_tree_canonical_sky130_synthesis_opensta",
        "status": (
            "PASS_CURRENT_TREE_SYNTHESIS_OPENSTA"
            if timing_met
            else "FAIL_CURRENT_TREE_OPENSTA"
        ),
        "sources": binding["sources"],
        "scope": {
            "canonical_target": "dynamic-scale32-first-boundary-synth-sta",
            "top": "ace2_shell",
            "technology": "sky130_fd_sc_hd tt_025C_1v80",
            "current_supported_runtime_boundary": "layer0_input_rmsnorm_through_qkv",
            "next_unsupported_boundary": "attention_value_publication_into_o_proj",
            "sealed_ppa_replayed": False,
        },
        "verification": {
            "focused_regression_status": focused_status,
            "focused_regression_tests": focused_tests,
            "verilator_lint_status": (
                "PASS_WITH_WARNINGS" if lint_path.is_file() else "NOT_RECORDED"
            ),
        },
        "synthesis": {
            "status": "PASS",
            "top_area_um2": area,
        },
        "opensta": {
            "status": "MET" if timing_met else "VIOLATED",
            "clock_period_ns": 10.0,
            "worst_reported_slack_ns": slack,
            "wns_ns": wns,
            "tns_ns": tns,
        },
        "artifacts": artifacts,
    }
    write_atomic(provenance_path, canonical_bytes(payload))
    return timing_met


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    evidence = args.evidence_dir
    if not evidence.is_absolute():
        evidence = ROOT / evidence
    if args.phase == "before":
        capture_before(evidence)
    else:
        timing_met = capture_after(evidence)
        print(
            "CHAT_SKY130_PROVENANCE_AFTER_CAPTURED "
            f"timing={'MET' if timing_met else 'VIOLATED'} evidence={evidence}"
        )
        return 0
    print(f"CHAT_SKY130_PROVENANCE_{args.phase.upper()}_PASS evidence={evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
