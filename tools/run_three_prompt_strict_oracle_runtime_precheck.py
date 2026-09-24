#!/usr/bin/env python3
"""Run the bounded precheck for a future three-prompt RTL/oracle suite."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_full_chain_independent_oracle as oracle  # noqa: E402
from tools import rtl_arbitrary_text_generation_backend as backend  # noqa: E402


PAYLOADS = {
    "layer_output_s8.bin": bytes([0x80, 0x00, 0x7F]),
    "layer_output_scale_f64le.bin": bytes.fromhex("000000000000f03f"),
    "final_rmsnorm_s8.bin": bytes([0x81, 0x01, 0x7E]),
    "final_rmsnorm_scale_f64le.bin": bytes.fromhex("0000000000000040"),
    "lm_head_output_s8.bin": bytes(index & 0xFF for index in range(oracle.EXPECTED_MODEL_OUTPUTS)),
    "lm_head_output_scale_f64le.bin": bytes.fromhex("0000000000000840"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_fresh_output(path: Path) -> None:
    require(
        not os.path.lexists(path),
        f"output namespace already exists: {backend.public_path(path)}",
    )


def write_payloads(directory: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    tensors = directory / "tensors"
    tensors.mkdir(parents=True)
    records = {
        name: backend.write_binary(tensors / name, PAYLOADS[name])
        for name in names
    }
    backend.write_binary(tensors / "transient.bin", b"discard")
    backend.write_binary(directory / "vectors/transient.hex", b"00\n")
    return records


def checker_and_retention_check(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    layer_names = (
        "layer_output_s8.bin",
        "layer_output_scale_f64le.bin",
    )
    head_names = (
        "final_rmsnorm_s8.bin",
        "final_rmsnorm_scale_f64le.bin",
        "lm_head_output_s8.bin",
        "lm_head_output_scale_f64le.bin",
    )
    layer_dir = output / "payload-fixture/layer"
    head_dir = output / "payload-fixture/head"
    layer_records = write_payloads(layer_dir, layer_names)
    head_records_with_suffix = write_payloads(head_dir, head_names)
    head_records = {
        name.removesuffix(".bin"): record
        for name, record in head_records_with_suffix.items()
    }
    layer_pruning = backend.prune_transient_execution_artifacts(layer_dir)
    head_pruning = backend.prune_transient_execution_artifacts(head_dir)
    backend.write_json(layer_dir / "rtl_execution.json", {"tensors": layer_records})
    backend.write_json(
        head_dir / "head_execution.json",
        {"lm_head": {"artifacts": head_records}},
    )
    summary = {
        "token_executions": [
            {
                "layers": [
                    {
                        "rtl_execution": backend.file_record(
                            layer_dir / "rtl_execution.json"
                        )
                    }
                ]
            }
        ],
        "head_steps": [
            {
                "head_execution": backend.file_record(
                    head_dir / "head_execution.json"
                )
            }
        ],
    }
    retained = oracle.verify_retained_comparison_payloads(summary)
    require(retained["records_checked"] == len(PAYLOADS), "strict checker skipped a payload")
    require(
        retained["bytes_checked"] == sum(map(len, PAYLOADS.values())),
        "strict checker byte coverage differs",
    )
    require(
        len(PAYLOADS["lm_head_output_s8.bin"]) == oracle.EXPECTED_MODEL_OUTPUTS,
        "logit fixture does not cover the full model output vocabulary",
    )

    comparison_source = inspect.getsource(oracle.compare_reference)
    mismatch_controls: dict[str, str] = {}
    for name, payload in PAYLOADS.items():
        lookup = name if name in layer_records else name.removesuffix(".bin")
        record = layer_records.get(lookup, head_records.get(lookup))
        require(record is not None, f"missing fixture record for {name}")
        require(name.removesuffix(".bin") in comparison_source or name in comparison_source,
                f"full-chain comparison is not bound to {name}")
        mutated = bytes([payload[0] ^ 1]) + payload[1:]
        try:
            oracle.require_payload_matches_record(mutated, record, name)
        except oracle.OracleError as error:
            mismatch_controls[name] = str(error)
        else:
            raise RuntimeError(f"strict byte comparator accepted mutated {name}")

    expected_names = set(PAYLOADS)
    require(
        backend.RETAINED_COMPARISON_TENSOR_FILES == expected_names,
        "runtime retention class set differs from strict checker inputs",
    )
    require(
        set(layer_pruning["retained_tensor_file_names"]) == set(layer_names)
        and set(head_pruning["retained_tensor_file_names"]) == set(head_names),
        "runtime pruning did not retain every required payload class",
    )
    require(
        not (layer_dir / "tensors/transient.bin").exists()
        and not (head_dir / "tensors/transient.bin").exists()
        and not (layer_dir / "vectors").exists()
        and not (head_dir / "vectors").exists(),
        "runtime pruning retained a transient control",
    )
    checker = {
        "status": "PASS",
        "comparison_mode": "strict byte-for-byte equality with byte-count equality",
        "payload_classes": sorted(expected_names),
        "records_checked": retained["records_checked"],
        "bytes_checked": retained["bytes_checked"],
        "full_vocabulary_logit_bytes": len(PAYLOADS["lm_head_output_s8.bin"]),
        "expected_model_outputs": oracle.EXPECTED_MODEL_OUTPUTS,
        "per_class_mutation_controls": mismatch_controls,
        "checker_source": backend.file_record(Path(oracle.__file__)),
    }
    retention = {
        "status": "PASS",
        "required_payload_classes": sorted(expected_names),
        "runtime_retention_classes": sorted(
            backend.RETAINED_COMPARISON_TENSOR_FILES
        ),
        "layer_pruning": layer_pruning,
        "head_pruning": head_pruning,
    }
    return checker, retention


def rss_guard_check(output: Path) -> dict[str, Any]:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    require(iverilog is not None, "iverilog is unavailable")
    require(vvp is not None, "vvp is unavailable")
    child_dir = output / "rss-child"
    child_dir.mkdir()
    source = child_dir / "child.sv"
    source.write_text(
        "module child; initial begin $display(\"ACE2_RSS_CHILD_PASS\"); $finish; end endmodule\n",
        encoding="ascii",
    )
    executable = child_dir / "child.vvp"
    with backend.process_tree_rss_tracking(interval_seconds=0.01) as tracker:
        compile_result, compile_child = backend.tracked_run(
            [iverilog, "-g2012", "-o", str(executable), str(source)],
            cwd=ROOT,
        )
        require(compile_result.returncode == 0, "representative iverilog child failed")
        run_result, run_child = backend.tracked_run([vvp, str(executable)], cwd=ROOT)
        require(run_result.returncode == 0, "representative vvp child failed")
        require("ACE2_RSS_CHILD_PASS" in run_result.stdout, "representative vvp marker missing")
        telemetry = tracker.summary(require_complete=True)
    require(
        telemetry["complete"] is True
        and telemetry["all_icarus_children_terminally_accounted"] is True
        and telemetry["observed_icarus_programs"] == ["iverilog", "vvp"],
        "representative child telemetry is incomplete",
    )

    try:
        with backend.process_tree_rss_tracking(interval_seconds=0.01) as missing_tracker:
            missing_tracker.summary(require_complete=True)
    except backend.BackendError as error:
        missing_child_control = str(error)
    else:
        raise RuntimeError("RSS guard accepted a run with no required Icarus children")

    runtime_source = ROOT / "tools/run_rtl_arbitrary_text_generation.py"
    return {
        "status": "PASS",
        "representative_real_child_telemetry": telemetry,
        "compile_child": compile_child,
        "vvp_child": run_child,
        "missing_required_children_control": missing_child_control,
        "runtime_guard_source": backend.file_record(runtime_source),
        "sampling_interpretation": (
            "child registration and terminal accounting are mandatory; short-lived "
            "per-child sample coverage is reported separately and is not fabricated"
        ),
    }


def run_check(
    name: str,
    operation: Callable[[], dict[str, Any]],
    checks: dict[str, Any],
    log_lines: list[str],
) -> None:
    started = time.monotonic()
    checks[name] = operation()
    checks[name]["elapsed_wall_seconds"] = time.monotonic() - started
    log_lines.append(f"PASS {name}")


def run(output: Path) -> int:
    output = output.resolve()
    verification_root = (ROOT / "reports/verification").resolve()
    require(
        output.is_relative_to(verification_root),
        "--output must be under reports/verification",
    )
    require_fresh_output(output)
    output.mkdir(parents=True)
    checks: dict[str, Any] = {}
    log_lines = ["PASS output_namespace_fresh"]
    checks["output_namespace_fresh"] = {
        "status": "PASS",
        "path": output.relative_to(ROOT).as_posix(),
        "preexisting": False,
        "create_only": True,
    }
    existing_control = output / "existing-namespace-control"
    existing_control.mkdir()
    try:
        require_fresh_output(existing_control)
    except RuntimeError as error:
        checks["output_namespace_fresh"]["existing_namespace_control"] = str(error)
    else:
        raise RuntimeError("fresh-output guard accepted an existing namespace")

    started = time.monotonic()
    try:
        checker, retention = checker_and_retention_check(output)
        checks["strict_checker_activation"] = checker
        checker["elapsed_wall_seconds"] = time.monotonic() - started
        log_lines.append("PASS strict_checker_activation")
        checks["retained_payload_class_binding"] = retention
        log_lines.append("PASS retained_payload_class_binding")
        run_check(
            "rss_resource_guard_enforcement",
            lambda: rss_guard_check(output),
            checks,
            log_lines,
        )
    except Exception as error:
        failed_check = next(
            name
            for name in (
                "strict_checker_activation",
                "retained_payload_class_binding",
                "rss_resource_guard_enforcement",
            )
            if name not in checks
        )
        log_lines.append(f"FAIL {failed_check}: {type(error).__name__}: {error}")
        log_path = output / "precheck.log"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        backend.write_json(
            output / "result.json",
            {
                "schema": "ace2-three-prompt-strict-oracle-runtime-precheck-v1",
                "status": "FAIL_CLOSED",
                "first_failing_check": failed_check,
                "diagnostic": f"{type(error).__name__}: {error}",
                "log": backend.file_record(log_path),
                "checks": checks,
                "sealed_attempts_mutated": False,
                "rtl_completion_claimed": False,
            },
        )
        return 1

    log_path = output / "precheck.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    backend.write_json(
        output / "result.json",
        {
            "schema": "ace2-three-prompt-strict-oracle-runtime-precheck-v1",
            "status": "PASS",
            "classification": (
                "non-claiming host/runtime precheck for a future fresh suite attempt; "
                "not a three-prompt numerical result or RTL completion claim"
            ),
            "checks": checks,
            "log": backend.file_record(log_path),
            "sealed_attempts_mutated": False,
            "software_transformer_or_logits_fallback": False,
            "rtl_completion_claimed": False,
        },
    )
    print(f"PASS {output.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
