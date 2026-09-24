#!/usr/bin/env python3
"""Run the decisive focused verifier for the layer-16 runtime metadata core."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATE = ROOT / (
    "evidence/candidates/w4a8-layer16-gate-up-output-grouped-a8-repair-v1/"
    "candidate-0004"
)
PREQUEUE = ROOT / (
    "evidence/prequeue/w4a8-layer16-post-attention-rmsnorm-output-signed-a8-repair-v1/"
    "runtime-metadata-core-integration-prequeue.json"
)
GENERATED = ROOT / (
    "verification/generated/"
    "ace2_layer16_post_attention_rmsnorm_runtime_metadata"
)
ARITHMETIC_RTL = ROOT / (
    "rtl/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core.sv"
)
RUNTIME_RTL = ROOT / (
    "rtl/ace2_layer16_post_attention_rmsnorm_runtime_metadata_core.sv"
)
TB = ROOT / (
    "verification/tb/ace2_layer16_post_attention_rmsnorm_runtime_metadata_tb.sv"
)
GENERATOR = ROOT / (
    "tools/gen_layer16_post_attention_rmsnorm_runtime_metadata_vectors.py"
)
RUNNER = Path(__file__).resolve()
ARITHMETIC_RESULT = ROOT / (
    "build/layer16-post-attention-rmsnorm-output-grouped-scale32-focused-r1/result.json"
)
ARITHMETIC_VECTORS = ROOT / (
    "verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/"
    "SHA256SUMS"
)
FRESH_L2 = ROOT / "build/l2-review-layer16-gate-up-output-grouped-scale32-search-r1/result.json"
RTL_TOP = "ace2_layer16_post_attention_rmsnorm_runtime_metadata_core"


EXPECTED_HASHES = {
    ARITHMETIC_RTL: "1a82dd96c0b48652191b4892c3429f1a1d485240473a8f2e155d118e05f59e1b",
    ARITHMETIC_RESULT: "3b24137edb66ffec8a5d0256278d894a6689df6584d124e4863b06da48f65a10",
    ARITHMETIC_VECTORS: "7113d595fc906488cbc903fcb079231c9a16477d722d98a23d32dcf779a2fff3",
    FRESH_L2: "90acdf066f51fdb00cf0d955bfd1f95a759ccb5f889a42244704a4f87f682bbb",
    SOURCE_CANDIDATE / "result.json": "f734ae5e80bafed36fd434d65d1caec1794d6fa28ec6756c88092227726e6a18",
}


PASS_PATTERN = re.compile(
    r"ACE2_LAYER16_POST_ATTENTION_RMSNORM_RUNTIME_METADATA_RTL_PASS "
    r"samples=(?P<samples>\d+) positions=(?P<positions>\d+) "
    r"groups=(?P<groups>\d+) nonzero=(?P<nonzero>\d+) "
    r"saturations=(?P<saturations>\d+) input_stalls=(?P<input_stalls>\d+) "
    r"output_stalls=(?P<output_stalls>\d+) cycles=(?P<cycles>\d+) "
    r"xz_clean=1 reset_recovery=1 clear_recovery=1 invalid_scale_warning=1 "
    r"backpressure_stable=1 metadata_live_derived=1"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def run(command: list[str], log: Path, *, cwd: Path = ROOT) -> str:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    require(
        completed.returncode == 0,
        f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}",
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--verilator", default="verilator")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(ROOT.resolve() in output.parents, "output must be repository-relative")
    require(not output.exists(), "output already exists")
    output.mkdir(parents=True)

    binding_log = output / "immutable_binding_check.log"
    vector_log = output / "vector_regeneration_check.log"
    compile_log = output / "iverilog_compile.log"
    simulation_log = output / "simulation.log"
    lint_log = output / "verilator_lint.log"
    tool_log = output / "tool_versions.log"

    binding_lines: list[str] = []
    for path, expected_hash in EXPECTED_HASHES.items():
        observed = sha256(path)
        require(observed == expected_hash, f"immutable hash differs: {path}")
        binding_lines.append(f"{observed}  {path.relative_to(ROOT).as_posix()}")
    candidate_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=SOURCE_CANDIDATE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(candidate_check.returncode == 0, candidate_check.stdout)
    binding_lines.append(candidate_check.stdout.rstrip())
    binding_log.write_text("\n".join(binding_lines) + "\n", encoding="utf-8")

    run([sys.executable, str(GENERATOR), "--check"], vector_log)

    with tempfile.TemporaryDirectory(
        prefix="ace2-layer16-runtime-rmsnorm-", dir=ROOT / "build"
    ) as temporary:
        executable = Path(temporary) / "tb.vvp"
        compile_stdout = run(
            [
                args.iverilog,
                "-g2012",
                "-Wall",
                "-Wno-timescale",
                "-Iverification/generated",
                "-s",
                "ace2_layer16_post_attention_rmsnorm_runtime_metadata_tb",
                "-o",
                str(executable),
                str(ARITHMETIC_RTL),
                str(RUNTIME_RTL),
                str(TB),
            ],
            compile_log,
        )
        require(not compile_stdout.strip(), "Icarus emitted warning or diagnostic output")
        simulation_stdout = run([args.vvp, str(executable)], simulation_log)

    manifest = json.loads((GENERATED / "manifest.json").read_text(encoding="utf-8"))
    match = PASS_PATTERN.search(simulation_stdout)
    require(match is not None, "runtime metadata RTL PASS marker is missing")
    metrics = {name: int(value) for name, value in match.groupdict().items()}
    require(metrics["samples"] == 30 * 896, "model sample count differs")
    require(metrics["positions"] == 30, "model position count differs")
    require(metrics["groups"] == 30 * 7, "runtime Scale32 group count differs")
    require(
        metrics["nonzero"] == manifest["observed"]["nonzero_outputs"],
        "runtime nonzero output count differs",
    )
    require(metrics["nonzero"] > 0, "runtime RMSNorm output remains zero")

    lint_stdout = run(
        [
            args.verilator,
            "--lint-only",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            RTL_TOP,
            str(ARITHMETIC_RTL),
            str(RUNTIME_RTL),
        ],
        lint_log,
    )
    require(not lint_stdout.strip(), "Verilator emitted warning or diagnostic output")

    versions: list[str] = []
    for command in (
        [args.iverilog, "-V"],
        [args.verilator, "--version"],
        [sys.executable, "--version"],
    ):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        require(completed.returncode == 0, f"tool version failed: {' '.join(command)}")
        versions.append(completed.stdout.strip())
    tool_log.write_text("\n".join(versions) + "\n", encoding="utf-8")

    generated_members = sorted(path for path in GENERATED.iterdir() if path.is_file())
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "PASS_FOCUSED_LAYER16_POST_ATTENTION_RMSNORM_RUNTIME_METADATA_RTL_REFERENCE_AGREEMENT_INCREMENT",
        "classification": "bounded_runtime_metadata_core_increment_no_official_attempt",
        "boundary": "model.layers.16.post_attention_layernorm.output_to_signed_a8",
        "executed_contract": manifest["executed_contract"],
        "claim_boundary": (
            "Cycle-accurate raw-A8/Scale32 plus immutable integer/Scale32 gain to "
            "runtime-derived RMSNorm A8/Scale32 agreement only. Candidate rank, token-9707 "
            "restoration, exact accepted-W4 gate/up accumulator vectors, canonical synthesis, "
            "timing, PPA, independent acceptance, official replay, U280, and Stage 2 remain unclaimed."
        ),
        "numeric_contract": manifest["numeric_contract"],
        "checks": {
            "preserved_arithmetic_rtl_sha256_unchanged": True,
            "preserved_arithmetic_result_sha256_unchanged": True,
            "preserved_arithmetic_vectors_sha256_unchanged": True,
            "preserved_candidate_sha256_unchanged": True,
            "fresh_l2_sha256_unchanged": True,
            "all_26880_model_outputs_match": True,
            "runtime_metadata_derived_without_sidecars": True,
            "nonzero_gate_up_input_payload_produced": metrics["nonzero"] > 0,
            "exact_accepted_w4_gate_up_accumulator_vectors_produced": False,
            "ties_to_even_alignment_mean_gain_and_output": True,
            "signed_a8_saturation_path_reused_from_preserved_core": True,
            "invalid_scale_warning": True,
            "input_backpressure": True,
            "output_backpressure_and_stability": True,
            "asynchronous_reset_recovery": True,
            "synchronous_clear_recovery": True,
            "xz_clean_every_checked_cycle": True,
            "iverilog_warning_clean": True,
            "verilator_warning_clean": True,
            "model_vector_regeneration_byte_exact": True,
        },
        "repair_attempts": [
            {
                "attempt": 1,
                "failure_taxonomy": "numeric_scale_derivation",
                "root_cause": (
                    "A signed/unsigned expression-width ambiguity in the stored aligned-activation "
                    "absolute-value multiply inflated live group magnitudes and selected output "
                    "Scale32 exponents 3-4 bits too large."
                ),
                "regression": (
                    "All 26,880 outputs now compare aligned activation, Q7.8 gain, group Scale32, "
                    "sumsq, mean-square, ceil RMS, reciprocal, and final A8 against the integer reference."
                ),
            },
            {
                "attempt": 2,
                "failure_taxonomy": "lint_width_contract",
                "root_cause": (
                    "The first decisive verifier rejected implicit parameter-width, signed-exponent, "
                    "group-offset, and serial-divider truncations even though Icarus simulation passed."
                ),
                "regression": (
                    "The runtime RTL now uses explicit 48-bit hidden-size arithmetic, 32-bit signed "
                    "exponent extensions, explicit group-offset bits, and width-exact divider state."
                ),
                "preserved_log": (
                    "build/layer16-post-attention-rmsnorm-runtime-metadata-focused-r1/"
                    "verilator_lint.log"
                ),
            },
            {
                "attempt": 3,
                "failure_taxonomy": "lint_residual_truncation",
                "root_cause": (
                    "A second lint pass still saw two intentional narrowing operations without "
                    "explicit casts in the rounded-alignment and divider-remainder paths."
                ),
                "regression": "Both narrowing points are now explicit width casts.",
                "preserved_log": (
                    "build/layer16-post-attention-rmsnorm-runtime-metadata-focused-r2/"
                    "verilator_lint.log"
                ),
            },
        ],
        "metrics": metrics,
        "bindings": {
            "sealed_prequeue": artifact(PREQUEUE),
            "preserved_arithmetic_rtl": artifact(ARITHMETIC_RTL),
            "preserved_arithmetic_result": artifact(ARITHMETIC_RESULT),
            "preserved_arithmetic_vectors": artifact(ARITHMETIC_VECTORS),
            "preserved_candidate_result": artifact(SOURCE_CANDIDATE / "result.json"),
            "fresh_l2_result": artifact(FRESH_L2),
            "runtime_rtl": artifact(RUNTIME_RTL),
            "testbench": artifact(TB),
            "generator_reference": artifact(GENERATOR),
            "runner": artifact(RUNNER),
            "generated_vectors": [artifact(path) for path in generated_members],
            "logs": {
                "immutable_binding_check": artifact(binding_log),
                "vector_regeneration_check": artifact(vector_log),
                "iverilog_compile": artifact(compile_log),
                "simulation": artifact(simulation_log),
                "verilator_lint": artifact(lint_log),
                "tool_versions": artifact(tool_log),
            },
        },
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "gate_up_w4_mutated": False,
            "runtime_bf16_sidecar": False,
            "runtime_precomputed_metadata_sidecar": False,
            "carried_activation_wider_than_a8": False,
            "rank_selection_completed": False,
            "canonical_synthesis_or_timing_claimed": False,
            "u280_or_stage2_entered": False,
            "independent_review_still_required": True,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in members
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )
    print(
        "ACE2_LAYER16_POST_ATTENTION_RMSNORM_RUNTIME_METADATA_RTL_RESULT "
        f"status={result['status']} samples={metrics['samples']} "
        f"nonzero={metrics['nonzero']} cycles={metrics['cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
