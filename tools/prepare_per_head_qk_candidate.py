#!/usr/bin/env python3
"""Create the hash-bound candidate packet used by the bounded Q/K smoke gate."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "per_head_qk_repair" / "latest"
ARCHIVE = ROOT / "evidence" / "per_head_qk_repair" / "archive"
RTL_MANIFEST = ROOT / "design" / "RTL_MANIFEST.json"
NUMERICAL_RTL = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    source_lines = "".join(
        f"{sha256_file(ROOT / relative)}  {relative}\n"
        for relative in NUMERICAL_RTL
    )
    source_list = EVIDENCE / "source_hashes.before"
    source_list.write_text(source_lines, encoding="utf-8")
    rtl_hash = sha256_file(source_list)
    packet = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "per_head_qk_candidate_ready_for_frozen_paired_smoke",
        "candidate_id": f"per_head_qk_{rtl_hash[:16]}",
        "source_binding": {
            "source_hash_list": source_list.relative_to(ROOT).as_posix(),
            "ordered_source_hash_list_sha256": rtl_hash,
        },
        "accepted_frontier_preservation": {
            "rtl_manifest_sha256": sha256_file(RTL_MANIFEST),
            "pipeline_stage_unchanged": "rtl",
            "stage_closing": False,
        },
        "claim_boundaries": [
            "Candidate packet only; it is not verification, PPA, benchmark, or signoff evidence.",
            "The frozen two-dataset improvement gate must pass before a full-shell or PPA rerun.",
        ],
    }
    output = EVIDENCE / "candidate_evidence.json"
    payload = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    packet_sha256 = sha256_bytes(payload.encode("utf-8"))
    archive = ARCHIVE / f"candidate_evidence.{packet_sha256}.json"
    if archive.exists() and archive.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"candidate archive collision: {archive}")
    archive.write_text(payload, encoding="utf-8")
    output.write_text(payload, encoding="utf-8")
    print(
        "ACE2_PER_HEAD_QK_CANDIDATE_PREP_PASS "
        f"candidate_id={packet['candidate_id']} rtl_hash={rtl_hash} "
        f"packet_sha256={packet_sha256} archive={archive.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
