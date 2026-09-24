#!/usr/bin/env python3
"""Validate and seal the frozen paired BF16/W4A8 scale-contract smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = (
    ROOT
    / "benchmark"
    / "raw"
    / "quality"
    / "reviewer-rmsnorm-contract-l2-round3-smoke"
)
QUALITY_CONFIG = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
QUALITY_BINDING = ROOT / "benchmark" / "quality" / "RTL_BINDING.json"
OUTPUT = (
    ROOT
    / "evidence"
    / "rmsnorm_scale_contract"
    / "latest"
    / "paired_smoke_binding.json"
)
RAW_NAMES = [
    "derived_scales.json",
    "input_observations.json",
    "perplexity_bf16_raw.json",
    "perplexity_w4a8_raw.json",
    "results.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing smoke artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT.resolve() not in run_dir.parents:
        raise RuntimeError("smoke run directory must be inside the repository")
    if run_dir == REFERENCE_RUN.resolve():
        raise RuntimeError("refusing to overwrite or reseal the frozen reference smoke")

    results = load_json(run_dir / "results.json")
    reference = load_json(REFERENCE_RUN / "results.json")
    config = load_json(QUALITY_CONFIG)
    binding = load_json(QUALITY_BINDING)
    if results.get("mode") != "smoke":
        raise RuntimeError("paired run is not smoke mode")
    if results.get("classification") != "smoke_not_acceptance_evidence":
        raise RuntimeError("paired run used a diagnostic or official mode")
    if results.get("gate_passed") is not False:
        raise RuntimeError("smoke classification must retain gate_passed=false")
    if results.get("thresholds") != reference.get("thresholds"):
        raise RuntimeError("paired smoke thresholds differ from the frozen smoke")
    expected_thresholds = config["acceptance_thresholds"]
    if results["thresholds"] != expected_thresholds:
        raise RuntimeError("paired smoke thresholds differ from QUALITY_CONFIG")
    if results.get("seeds") != reference.get("seeds"):
        raise RuntimeError("paired smoke seeds differ from the frozen smoke")
    if results.get("input_observations") != reference.get("input_observations"):
        raise RuntimeError("paired smoke inputs differ from the frozen smoke")
    if results.get("model") != reference.get("model"):
        raise RuntimeError("paired smoke model revision differs from the frozen smoke")
    accepted_rtl = results["artifacts"]["accepted_rtl"]
    if accepted_rtl.get("candidate_rtl_hash") != binding["candidate_rtl_hash"]:
        raise RuntimeError("paired smoke candidate differs from live RTL binding")
    if accepted_rtl["binding"]["sha256"] != sha256_file(QUALITY_BINDING):
        raise RuntimeError("paired smoke did not record the live RTL binding hash")
    manifest_path = ROOT / binding["manifest"]["path"]
    if sha256_file(manifest_path) != binding["manifest"]["sha256"]:
        raise RuntimeError("live RTL manifest hash differs from RTL_BINDING")
    if any((run_dir / name).exists() for name in ("lm_eval_bf16_raw.json", "lm_eval_w4a8_raw.json")):
        raise RuntimeError("official lm-eval outputs are prohibited in this smoke")

    artifacts = {name: artifact(run_dir / name) for name in RAW_NAMES}
    sums_path = run_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{artifacts[name]['sha256']}  {name}\n" for name in sorted(artifacts)),
        encoding="utf-8",
    )
    artifacts["SHA256SUMS"] = artifact(sums_path)
    thresholds_met = all(results["checks"].values())
    packet = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": (
            "frozen_paired_smoke_thresholds_met"
            if thresholds_met
            else "frozen_paired_smoke_threshold_miss"
        ),
        "classification": results["classification"],
        "candidate_rtl_hash": binding["candidate_rtl_hash"],
        "rtl_binding": artifact(QUALITY_BINDING),
        "rtl_manifest": artifact(manifest_path),
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "artifacts": artifacts,
        "seeds": results["seeds"],
        "input_observations": results["input_observations"],
        "thresholds": results["thresholds"],
        "checks": results["checks"],
        "thresholds_met": thresholds_met,
        "observed_ratios": {
            "c4_en_512": results["metrics"]["c4_en_512"]["ratio"],
            "wikitext2": results["metrics"]["wikitext2"]["ratio"],
        },
        "official_14_item_evaluation_run": False,
        "official_14_item_evaluation_prohibited": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ACE2_RMSNORM_SCALE_CONTRACT_PAIRED_SMOKE_SEAL_PASS "
        f"thresholds_met={str(thresholds_met).lower()} "
        f"c4_ratio={packet['observed_ratios']['c4_en_512']:.9f} "
        f"wikitext2_ratio={packet['observed_ratios']['wikitext2']:.9f}"
    )


if __name__ == "__main__":
    main()
