#!/usr/bin/env python3
"""Audit whether the frozen ACE-2 benchmark contract is executable."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_gemmini_subset import run_subset
from run_quality_gate import run_gate


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "benchmark" / "raw" / "latest"
RTL_HASH = "a8954ae9eaac4cd89e8c1b07384c3eefba8de1eedc9f57d472f6eb38c3efba07"
PPA_RESULTS_HASH = "b8974d6ea7bbdcefe508d28091269c82285362d29d5a369f95a565dc8655d86a"
PPA_MAKEFILE_HASH = "a31d1719782a521fcd4016133c44881ba3bd10f958fe23c3f0e4e473a1123611"
PROTOTYPE_MAKEFILE_HASH = "cc31d044bc5c6db542f977ea46f5bc37f2a94e74872e9cf9d78139a245a540ed"
PREFILL_LENGTHS = [16, 128, 512, 2048, 8192, 32768]
DECODE_CONTEXTS = [1, 128, 1024, 4096, 8192, 32768]
DECODE_RUNS = [1, 16, 128]
BANDWIDTHS = [1, 2, 4, 8, 16]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    return {
        "path": relative,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reconstruct_makefiles(current: str) -> tuple[str, str]:
    benchmark_phrase = " prototype-stage benchmark-stage rtl-fast-loop"
    benchmark_block = "\nbenchmark-stage:\n\t$(PYTHON) tools/run_benchmark_stage.py\n"
    if benchmark_phrase not in current or benchmark_block not in current:
        raise RuntimeError("current Makefile does not contain the expected benchmark-only delta")
    prototype = current.replace(
        benchmark_phrase,
        " prototype-stage rtl-fast-loop",
        1,
    ).replace(benchmark_block, "", 1)

    prototype_phrase = " ppa-stage prototype-stage rtl-fast-loop"
    prototype_block = "\nprototype-stage:\n\t$(PYTHON) tools/run_prototype_stage.py\n"
    if prototype_phrase not in prototype or prototype_block not in prototype:
        raise RuntimeError("prototype-era Makefile cannot be reconstructed from the current file")
    ppa = prototype.replace(
        prototype_phrase,
        " ppa-stage rtl-fast-loop",
        1,
    ).replace(prototype_block, "", 1)
    return prototype, ppa


def audit_makefile(generated_at: str) -> dict[str, Any]:
    current_path = ROOT / "Makefile"
    current = current_path.read_text(encoding="utf-8")
    prototype, ppa = reconstruct_makefiles(current)
    prototype_hash = sha256_bytes(prototype.encode())
    ppa_hash = sha256_bytes(ppa.encode())

    snapshot_path = RAW_DIR / "Makefile.ppa-stage"
    diff_path = RAW_DIR / "Makefile.ppa-to-benchmark.diff"
    snapshot_path.write_text(ppa, encoding="utf-8")
    diff_path.write_text(
        "".join(
            difflib.unified_diff(
                ppa.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile="Makefile.ppa-stage",
                tofile="Makefile.current-benchmark",
            )
        ),
        encoding="utf-8",
    )

    ppa_results = load_json(ROOT / "ppa" / "RESULTS.json")
    ppa_record = next(
        item
        for item in ppa_results["source_hashes"]["constraint_and_protocol_files"]
        if item["path"] == "Makefile"
    )
    prototype_results = load_json(ROOT / "prototype" / "RESULTS.json")
    prototype_record = next(
        item
        for item in prototype_results["source_hashes"]["input_artifacts"]
        if item["path"] == "Makefile"
    )
    ppa_matches = ppa_hash == ppa_record["sha256"] == PPA_MAKEFILE_HASH
    prototype_matches = (
        prototype_hash == prototype_record["sha256"] == PROTOTYPE_MAKEFILE_HASH
    )
    proof = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": (
            "pass_exact_reconstruction"
            if ppa_matches and prototype_matches
            else "blocked_makefile_provenance_mismatch"
        ),
        "method": (
            "Reverse the two later stage-orchestration additions from the current Makefile, "
            "then require byte counts and SHA-256 values to match the immutable prototype and "
            "PPA packet records exactly."
        ),
        "ppa_record": ppa_record,
        "prototype_record": prototype_record,
        "current": artifact(current_path),
        "reconstructed_ppa": artifact(snapshot_path),
        "ppa_to_current_diff": artifact(diff_path),
        "reconstructed_hashes": {
            "ppa": {
                "sha256": ppa_hash,
                "matches_record": ppa_matches,
            },
            "prototype": {
                "sha256": prototype_hash,
                "matches_record": prototype_matches,
            },
        },
        "verified_hashes": {
            "ppa": ppa_matches,
            "prototype": prototype_matches,
        },
        "delta_classification": {
            "synthesizable_rtl_changed": False,
            "constraints_changed": False,
            "ppa_flow_recipe_changed": False,
            "provenance_mismatch": not (ppa_matches and prototype_matches),
            "changes": [
                "add prototype-stage phony entry and Python binder target",
                "add benchmark-stage phony entry and Python binder target",
            ],
        },
    }
    write_json(RAW_DIR / "makefile_provenance.json", proof)
    return proof


def audit_trace_capability(generated_at: str) -> dict[str, Any]:
    shell_path = ROOT / "rtl" / "ace2_shell.sv"
    traceability_path = ROOT / "design" / "RTL_TRACEABILITY.md"
    verification_path = ROOT / "verification" / "RESULTS.json"
    shell = shell_path.read_text(encoding="utf-8")
    context_match = re.search(r"ATTN_CONTEXT_MAX\s*=\s*(\d+)", shell)
    if context_match is None:
        raise RuntimeError("cannot locate ATTN_CONTEXT_MAX in rtl/ace2_shell.sv")
    context_max = int(context_match.group(1))
    descriptor_checks = {
        opcode: bool(
            re.search(
                rf"wire {signal}.*?cmd_n_buf_q <= ATTN_CONTEXT_MAX_16",
                shell,
                flags=re.DOTALL,
            )
        )
        for opcode, signal in {
            "attention_score": "attn_score_descriptor_valid_w",
            "softmax": "softmax_descriptor_valid_w",
            "attention_value": "attn_value_descriptor_valid_w",
        }.items()
    }
    required_rows = [
        {
            "trace_class": "prefill",
            "prompt_tokens": prompt,
            "bandwidth_bytes_per_cycle": bandwidth,
        }
        for prompt in PREFILL_LENGTHS
        for bandwidth in BANDWIDTHS
    ] + [
        {
            "trace_class": "decode",
            "context_tokens": context,
            "generated_tokens": generated,
            "bandwidth_bytes_per_cycle": bandwidth,
        }
        for context in DECODE_CONTEXTS
        for generated in DECODE_RUNS
        for bandwidth in BANDWIDTHS
    ]
    present_trace_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (RAW_DIR / "full_shape_traces").glob("*.json")
    )
    verification = load_json(verification_path)
    rtl_hash_sources = {
        "ppa_results": load_json(ROOT / "ppa" / "RESULTS.json")["frontier"]["rtl_hash"],
        "rtl_manifest": load_json(ROOT / "design" / "RTL_MANIFEST.json")["rtl_hash"],
    }
    unique_rtl_hashes = sorted(set(rtl_hash_sources.values()))
    rtl_hash_agreement = len(unique_rtl_hashes) == 1

    executable_contexts = [context for context in DECODE_CONTEXTS if context <= context_max]
    result = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": (
            "blocked_frozen_rtl_context_tile_is_not_composable_to_full_shape"
            if rtl_hash_agreement
            else "blocked_rtl_hash_mismatch_and_full_shape_trace_unmet"
        ),
        "accepted_rtl_hash": unique_rtl_hashes[0] if rtl_hash_agreement else None,
        "rtl_hash_agreement": rtl_hash_agreement,
        "rtl_hash_sources": rtl_hash_sources,
        "frozen_contract": {
            "prefill_lengths": PREFILL_LENGTHS,
            "decode_context_lengths": DECODE_CONTEXTS,
            "decode_generated_token_runs": DECODE_RUNS,
            "bandwidth_bytes_per_cycle": BANDWIDTHS,
            "required_trace_rows": len(required_rows),
            "required_metrics": [
                "reference_outputs",
                "rtl_outputs",
                "cycles",
                "external_memory_bytes",
                "logical_sram_accesses",
                "request_stall_cycles",
                "write_data_stall_cycles",
                "read_data_stall_cycles",
                "response_stall_cycles",
            ],
        },
        "rtl_capability": {
            "attention_context_max_per_descriptor": context_max,
            "descriptor_checks_present": descriptor_checks,
            "frozen_decode_contexts_directly_accepted": executable_contexts,
            "frozen_decode_contexts_rejected": [
                context for context in DECODE_CONTEXTS if context > context_max
            ],
            "cross_tile_softmax_merge_command_or_state": "absent",
            "reason": (
                "ATTN_SCORE, SOFTMAX, and ATTN_VALUE all reject n > 8. No command, "
                "architectural state, or verified online-softmax merge path composes those "
                "independent eight-token probability vectors into exact attention over a "
                "longer context."
            ),
        },
        "evidence": [
            artifact(shell_path),
            artifact(traceability_path),
            artifact(verification_path),
            artifact(ROOT / "evidence" / "frontier" / "latest" / "rtl_shell_sim.log"),
        ],
        "full_shape_trace_files": present_trace_files,
        "full_shape_trace_rows_present": 0,
        "full_shape_trace_rows_required": len(required_rows),
        "model_estimates_are_not_accepted_as_trace_substitutes": True,
        "required_resolution": (
            "Resolve the RTL/PPA manifest hash disagreement, then reopen RTL/architecture "
            "to implement and verify exact long-context attention composition before "
            "cycle-accurate full-shape traces can be produced."
            if not rtl_hash_agreement
            else (
                "Reopen RTL/architecture to implement and verify exact long-context attention "
                "composition, then rerun canonical PPA before cycle-accurate full-shape traces "
                "can be produced. This benchmark audit does not change RTL or PPA."
            )
        ),
    }
    write_json(RAW_DIR / "trace_capability_audit.json", result)
    return result


def audit_quality(generated_at: str) -> dict[str, Any]:
    oracle_manifest = load_json(ROOT / "reference" / "ORACLE_MANIFEST.json")
    prompt_manifest = ROOT / "benchmark" / "quality" / "PROMPT_MANIFEST.json"
    quality_config = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
    quality_results = RAW_DIR / "quality_results.json"
    quality_preflight = RAW_DIR / "quality_gate_preflight.json"
    gate_runner = ROOT / "tools" / "run_quality_gate.py"
    full_model_runner = ROOT / "tools" / "ace2_full_model_fixed_point.py"
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in [
            prompt_manifest,
            quality_config,
            quality_preflight,
            quality_results,
            full_model_runner,
        ]
        if not path.exists()
    ]
    gate_result = load_json(quality_results) if quality_results.exists() else {}
    gate_preflight = load_json(quality_preflight) if quality_preflight.exists() else {}
    status = gate_result.get(
        "classification",
        gate_result.get(
            "status",
            gate_preflight.get("status", "blocked_precondition_failure"),
        ),
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": status,
        "gate_passed": bool(gate_result.get("gate_passed", False)),
        "required": {
            "model": "Qwen/Qwen2.5-0.5B",
            "bf16_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "wikitext2_perplexity_ratio_max": 1.05,
            "c4_en_512_prompt_perplexity_ratio_max": 1.05,
            "lm_eval_average_normalized_accuracy_drop_percentage_points_max": 2.0,
            "lm_eval_tasks": [
                "piqa",
                "hellaswag",
                "winogrande",
                "arc_easy",
                "arc_challenge",
            ],
        },
        "preflight": {
            "prompt_manifest_frozen": prompt_manifest.exists(),
            "quality_config_with_scale_derivation_frozen": quality_config.exists(),
            "gate_runner_executed": quality_results.exists(),
            "gate_runner": artifact(gate_runner),
            "gate_result": artifact(quality_results),
            "gate_preflight": artifact(quality_preflight),
            "full_model_bf16_runner_present": full_model_runner.exists(),
            "full_model_w4a8_fixed_point_runner_present": full_model_runner.exists(),
            "quality_results_present": quality_results.exists(),
            "missing_artifacts": missing,
            "quantization_scale_derivation_frozen": quality_config.exists(),
            "reason": (
                gate_result.get("reason")
                or gate_preflight.get("required_resolution")
                or "The official quality measurement has unresolved preconditions."
            ),
        },
        "operator_oracle_manifest": {
            "path": "reference/ORACLE_MANIFEST.json",
            "sha256": sha256_file(ROOT / "reference" / "ORACLE_MANIFEST.json"),
            "entry_count": len(oracle_manifest.get("entries", oracle_manifest.get("oracles", []))),
        },
        "required_resolution": (
            gate_result.get("required_resolution")
            or gate_preflight.get("required_resolution")
            or "Resolve every missing quality prerequisite before starting official measurement."
        ),
    }
    return result


def docker_image_digest() -> str | None:
    command = [
        "docker",
        "image",
        "inspect",
        "sbtscala/scala-sbt:eclipse-temurin-17.0.13_11_1.10.7_2.13.15",
        "--format",
        "{{index .RepoDigests 0}}",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    value = completed.stdout.strip()
    return value or None


def audit_gemmini(generated_at: str) -> dict[str, Any]:
    source_root = ROOT / "build" / "gemmini-baseline"
    log_path = RAW_DIR / "gemmini_transposer_subset.log"
    if not source_root.exists() or not log_path.exists():
        raise RuntimeError("Gemmini source checkout and passing subset log are required")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if revision != "8c3f9923a44a2fe2c7930587be297d6d4f8c09ca":
        raise RuntimeError(f"unexpected Gemmini revision: {revision}")
    log = log_path.read_text(encoding="utf-8")
    passed = (
        "Tests: succeeded 2, failed 0" in log
        and "NaiveTransposer" in log
        and "PipelinedTransposer" in log
    )
    sources = [
        source_root / "src" / "main" / "scala" / "gemmini" / "Transposer.scala",
        source_root / "src" / "test" / "scala" / "gemmini" / "TestUtils.scala",
        source_root / "src" / "test" / "scala" / "gemmini" / "TransposerUnitTest.scala",
        ROOT / "benchmark" / "baselines" / "gemmini-transposer" / "build.sbt",
        ROOT / "benchmark" / "baselines" / "gemmini-transposer" / "project" / "build.properties",
        ROOT
        / "benchmark"
        / "baselines"
        / "gemmini-transposer"
        / "src"
        / "main"
        / "scala"
        / "gemmini"
        / "Util.scala",
        log_path,
    ]
    result = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": "executed_pass_explicitly_incompatible_subset" if passed else "failed",
        "repository": "https://github.com/ucb-bar/gemmini.git",
        "revision": revision,
        "command": "python3 tools/run_gemmini_subset.py",
        "test": "gemmini.TransposerUnitTest",
        "results": {"tests_run": 2, "passed": 2 if passed else 0, "failed": 0 if passed else 2},
        "docker_image": docker_image_digest(),
        "source_and_log_artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sources
        ],
        "compatibility": {
            "same_qwen_workload": False,
            "same_w4a8_numeric_contract": False,
            "same_memory_boundary": False,
            "same_host_boundary": False,
            "same_technology_or_area_budget": False,
            "executed_scope": (
                "Pinned Gemmini NaiveTransposer and PipelinedTransposer Chisel tests with "
                "8-bit data, aligned to the final compatible chisel-iotesters dependency."
            ),
            "build_shim": (
                "A local wrapping-counter helper replaces Gemmini Util.scala because the "
                "full utility file imports arithmetic/HardFloat dependencies outside this "
                "bounded transposer subset."
            ),
        },
        "comparison_claim": "engineering_context_only_excluded_from_win_loss",
        "full_project_preflight": artifact(RAW_DIR / "gemmini_full_project_preflight.log"),
    }
    write_json(RAW_DIR / "gemmini_subset_manifest.json", result)
    return result


def audit_packet_integrity(generated_at: str) -> dict[str, Any]:
    ppa_path = ROOT / "ppa" / "RESULTS.json"
    ppa = load_json(ppa_path)
    embedded = ppa["raw_reports"]["results"]
    current = artifact(ppa_path)
    result = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "ppa_results": {
            "current_external_record": current,
            "embedded_legacy_self_record": embedded,
            "embedded_record_matches": (
                embedded.get("sha256") == current["sha256"]
                and embedded.get("bytes") == current["bytes"]
            ),
            "classification": (
                "legacy_non_verifiable_self_record; canonical PPA is preserved and the "
                "benchmark packet binds its externally computed current hash"
            ),
        },
    }
    if current["sha256"] != PPA_RESULTS_HASH:
        raise RuntimeError("canonical PPA packet changed during benchmark audit")
    write_json(RAW_DIR / "packet_integrity_audit.json", result)
    return result


def run_audit() -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    run_subset()
    run_gate()
    generated_at = utc_now()
    makefile = audit_makefile(generated_at)
    trace = audit_trace_capability(generated_at)
    quality = audit_quality(generated_at)
    gemmini = audit_gemmini(generated_at)
    integrity = audit_packet_integrity(generated_at)
    summary = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "stage": "benchmark",
        "status": "blocked",
        "stage_must_remain_benchmark": True,
        "accepted_rtl_hash": trace.get("accepted_rtl_hash"),
        "rtl_hash_agreement": trace.get("rtl_hash_agreement"),
        "rtl_hash_sources": trace.get("rtl_hash_sources"),
        "canonical_ppa_results_sha256": PPA_RESULTS_HASH,
        "trace_status": trace["status"],
        "quality_status": quality["status"],
        "gemmini_status": gemmini["status"],
        "makefile_provenance_status": makefile["status"],
        "packet_integrity_status": integrity["ppa_results"]["classification"],
    }
    write_json(RAW_DIR / "benchmark_contract_audit.json", summary)
    return summary


def main() -> None:
    summary = run_audit()
    print(
        "ACE2_BENCHMARK_CONTRACT_AUDIT "
        f"status={summary['status']} "
        f"trace={summary['trace_status']} "
        f"quality={summary['quality_status']} "
        f"gemmini={summary['gemmini_status']}"
    )


if __name__ == "__main__":
    main()
