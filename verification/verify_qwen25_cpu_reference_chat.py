#!/usr/bin/env python3
"""Verify CPU_SOFTWARE_REFERENCE_ONLY_NOT_RTL_COMPLETION chat evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABEL = "CPU_SOFTWARE_REFERENCE_ONLY_NOT_RTL_COMPLETION"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "build/cpu_reference_chat/latest/manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(payload["artifact_label"] == LABEL, "manifest label differs")
    require(payload["status"] == "PASS", "reference run did not pass")
    require(payload["execution"]["device"] == "cpu", "execution device is not CPU")
    require(payload["execution"]["parameter_device"] == "cpu", "parameters are not on CPU")
    require(not payload["execution"]["network_access_performed"], "network use was claimed")
    require(not payload["execution"]["gpu_execution_performed"], "GPU use was claimed")
    require(not payload["execution"]["fpga_execution_performed"], "FPGA use was claimed")
    require(not any(payload["claims"].values()), "a prohibited completion/hardware claim is true")
    require(payload["exact_command"], "exact command is absent")
    require(payload["model"]["immutable_revision"], "immutable model revision is absent")
    model_hash = payload["model"]["files"]["model.safetensors"]["sha256"]
    require(bool(re.fullmatch(r"[0-9a-f]{64}", model_hash)), "model SHA-256 is malformed")
    require(payload["runtime"]["torch"], "PyTorch version is absent")
    require(payload["runtime"]["transformers"], "Transformers version is absent")
    require(payload["memory"]["peak_rss_bytes"] > 0, "peak RSS was not measured")

    runs = payload["runs"]
    require(len(runs) >= 3, "expected two deterministic prompts and one operator prompt")
    require(
        sum(run["prompt_origin"] == "deterministic_smoke" for run in runs) >= 2,
        "deterministic prompt set is incomplete",
    )
    require(
        sum(run["prompt_origin"] == "operator" for run in runs) >= 1,
        "operator prompt invocation is absent",
    )
    for index, run in enumerate(runs):
        require(run["artifact_label"] == LABEL, f"run {index} label differs")
        require(bool(run["prompt_token_ids"]), f"run {index} prompt token IDs are absent")
        require(run["generated_token_count"] > 1, f"run {index} generated too few tokens")
        require(
            len(run["generated_token_ids"]) == run["generated_token_count"],
            f"run {index} generated-token count differs",
        )
        require(run["decoded_nonempty"], f"run {index} decoded text is empty")
        require(run["decoded_has_alphanumeric"], f"run {index} decoded text is not readable")
        require(run["kv_cache"]["use_cache_requested"], f"run {index} did not request cache")
        require(run["kv_cache"]["path_exercised"], f"run {index} did not reuse KV cache")
        require(run["kv_cache"]["cache_reuse_calls"] >= 1, f"run {index} cache was not reused")
        require(
            all(count == 1 for count in run["kv_cache"]["cache_reuse_input_token_counts"]),
            f"run {index} decode did not use one-token cached steps",
        )
        require(run["generation"]["forward_calls"] >= 2, f"run {index} lacks multiple forwards")

    stdout = payload["stdout"]
    stdout_lines = stdout.splitlines()
    require(bool(stdout_lines), "captured stdout is empty")
    require(all(line.startswith(LABEL + " ") for line in stdout_lines), "stdout is not fully labeled")
    stdout_path = Path(payload["artifacts"]["stdout_log"])
    require(stdout_path.read_text(encoding="utf-8") == stdout, "raw stdout log differs")
    require(payload["smoke_test"]["status"] == "PASS", "embedded smoke status differs")

    return {
        "artifact_label": LABEL,
        "schema_version": 1,
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_run_count": len(runs),
        "criteria": {
            "real_prompt_token_ids_present": True,
            "multiple_generated_tokens": True,
            "decoded_readable_text": True,
            "manual_model_forward_loop": True,
            "single_token_kv_cache_reuse": True,
            "cpu_only_claim_boundary": True,
        },
        "claim_boundary": payload["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else manifest_path.parent / "smoke_verification.json"
    )
    try:
        result = verify(manifest_path)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{LABEL} smoke_verification_status=\"PASS\"")
        print(f"{LABEL} verified_run_count={result['verified_run_count']}")
        print(f"{LABEL} verification_artifact={json.dumps(str(output_path))}")
        return 0
    except Exception as exc:
        failure = {
            "artifact_label": LABEL,
            "status": "FAIL",
            "manifest_path": str(manifest_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{LABEL} smoke_verification_status=\"FAIL\"")
        print(f"{LABEL} error={json.dumps(str(exc))}")
        print(f"{LABEL} verification_artifact={json.dumps(str(output_path))}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
