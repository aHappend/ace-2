#!/usr/bin/env python3
"""Run a one-layer RTL/full-head retained-payload reproduction."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts import verify_full_chain_independent_oracle as oracle
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    verification_root = backend.ROOT / "reports" / "verification"
    backend.require(
        output.is_relative_to(verification_root),
        "--output must be under reports/verification",
    )
    backend.require(not output.exists(), f"output already exists: {backend.public_path(output)}")
    output.mkdir(parents=True)
    started = time.monotonic()

    snapshot = generation.resolve_snapshot()
    snapshot_record = backend.configure_snapshot(snapshot)
    cache = backend.empty_layer_cache()
    token_id = 9707
    with torch.no_grad(), safe_open(
        backend.canonical.MODEL,
        framework="pt",
        device="cpu",
    ) as weights, safe_open(
        backend.canonical.ADAPTER,
        framework="pt",
        device="cpu",
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        shared_weights = backend.derive_lm_head_weights(
            output / "shared",
            embedding,
            backend.MODEL_OUTPUT_DOMAIN,
        )
        state = backend.embedding_state(weights, token_id, 0)
        derived, next_state, _template = backend.derive_layer_token(
            0,
            state,
            cache,
            None,
            weights,
            adapter,
        )
        layer = backend.run_layer_rtl(output / "layer-00", derived)
        head = backend.run_final_head_rtl(
            output / "head-00",
            next_state,
            norm_gain,
            embedding,
            shared_weights,
            backend.MODEL_OUTPUT_DOMAIN,
        )

    summary = {
        "token_executions": [
            {
                "absolute_position": 0,
                "input_token_id": token_id,
                "layers": [
                    {
                        "layer_id": 0,
                        "rtl_execution": backend.file_record(
                            output / "layer-00/rtl_execution.json"
                        ),
                    }
                ],
            }
        ],
        "head_steps": [
            {
                "generation_index": 0,
                "source_absolute_position": 0,
                "head_execution": backend.file_record(
                    output / "head-00/head_execution.json"
                ),
            }
        ],
    }
    backend.write_json(output / "run_summary.json", summary)
    strict_check = oracle.verify_retained_comparison_payloads(summary)

    control_dir = output / "missing-payload-control"
    control_dir.mkdir()
    control_layer = copy.deepcopy(
        load_json(output / "layer-00/rtl_execution.json")
    )
    missing_record = control_layer["tensors"]["layer_output_s8.bin"]
    missing_record["path"] = backend.public_path(
        control_dir / "intentionally-absent-layer-output-s8.bin"
    )
    backend.write_json(control_dir / "rtl_execution.json", control_layer)
    control_summary = {
        "token_executions": [
            {
                "layers": [
                    {
                        "rtl_execution": backend.file_record(
                            control_dir / "rtl_execution.json"
                        )
                    }
                ]
            }
        ],
        "head_steps": [],
    }
    try:
        oracle.verify_retained_comparison_payloads(control_summary)
    except oracle.sealed_verifier.VerificationError as error:
        missing_control = {
            "status": "PASS_FAIL_CLOSED",
            "diagnostic": str(error),
            "missing_path": missing_record["path"],
        }
    else:
        raise backend.GenerationError(
            "strict retained-payload checker accepted a missing payload"
        )

    shared_persistence = backend.prune_transient_execution_artifacts(
        output / "shared"
    )
    result = {
        "schema": "ace2-oracle-payload-retention-representative-v1",
        "status": "PASS_REPRESENTATIVE_RTL_ORACLE_PAYLOAD_RETENTION",
        "classification": (
            "computer-local one-layer real-Icarus retention reproduction with "
            "one full-vocabulary RTL head; not a full-chain numerical result"
        ),
        "snapshot": snapshot_record,
        "input_token_id": token_id,
        "layer_status": layer["status"],
        "head_status": head["status"],
        "full_vocabulary_outputs": head["output_count"],
        "software_transformer_or_logits_fallback": False,
        "strict_retained_payload_check": strict_check,
        "missing_payload_control": missing_control,
        "layer_execution": backend.file_record(
            output / "layer-00/rtl_execution.json"
        ),
        "head_execution": backend.file_record(
            output / "head-00/head_execution.json"
        ),
        "run_summary": backend.file_record(output / "run_summary.json"),
        "shared_persistence": shared_persistence,
        "elapsed_wall_seconds": time.monotonic() - started,
        "limitations": [
            "This bounded run covers one transformer layer and one full-vocabulary head.",
            "It validates retained payload availability and record hashes, not full-chain model agreement.",
            "No FPGA, synthesis, PPA, deployed-hardware, or silicon claim is made.",
        ],
    }
    backend.write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(
        f"{result['status']} "
        f"records={result['strict_retained_payload_check']['records_checked']} "
        f"bytes={result['strict_retained_payload_check']['bytes_checked']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
