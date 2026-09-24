#!/usr/bin/env python3
"""Revalidate the accepted V74 chat trace and run a fresh current-tree RTL smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "reports/ace2-stage1-chat-attempt-0002"
REVIEW = ROOT / "reports/rtl-stage-closure-v74-attempt002-reviewer-verdict.json"
TRANSITION = ROOT / "reports/rtl-stage-closure-v74-attempt002-transition-0001.json"
EXPECTED_TREE_ROOT = "82f56d1a23429ce72a7f387b658d6f25113db2afd41dbec7a0bc9d4c4fbbfdbc"
EXPECTED_ARTIFACT_HASHES = {
    "attempt-result.json": "5ec4b355a2fd3ad69c45ecc6accedd26da1528a383cde13f8b558e3f4f13e8d8",
    "worker-result.json": "c6891d42e24ded0935ac6be7306e0e47bb38aa676c824d79ba6e07e7693709e4",
    "independent-reference-comparison.json": (
        "40803354b455534bc099913a8531a8552621645d1846c4921733b49ca6d90e9e"
    ),
}
LM_HEAD_MARKER = re.compile(
    r"^ACE2_GENERATION_LM_HEAD_PASS position=(\d+) outputs=(\d+) "
    r"top_token=(\d+) top_logit_s8=(-?\d+) cycles=(\d+)$"
)


class RegressionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def project_path(value: str) -> Path:
    relative = Path(value)
    require(not relative.is_absolute(), f"artifact path is not project-relative: {value}")
    require(".." not in relative.parts, f"artifact path escapes the project: {value}")
    return ROOT / relative


def verify_file_record(record: dict[str, Any]) -> Path:
    path = project_path(str(record["path"]))
    require(path.is_file(), f"recorded file is missing: {record['path']}")
    require(path.stat().st_size == int(record["bytes"]), f"size drift: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"hash drift: {record['path']}")
    return path


def verify_sealed_attempt() -> dict[str, Any]:
    manifest_path = ATTEMPT / "SHA256SUMS"
    manifest_raw = manifest_path.read_bytes()
    tree_record = (ATTEMPT / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
    require(len(tree_record) == 2 and tree_record[1] == "SHA256SUMS", "invalid tree root")
    require(tree_record[0] == EXPECTED_TREE_ROOT, "unexpected accepted tree root")
    require(hashlib.sha256(manifest_raw).hexdigest() == tree_record[0], "tree root mismatch")

    entries = 0
    total_bytes = 0
    writable_members = 0
    for raw_line in manifest_raw.decode("utf-8").splitlines():
        digest, separator, relative_name = raw_line.partition("  ")
        require(separator == "  " and len(digest) == 64, "invalid SHA256SUMS line")
        path = project_path(f"{ATTEMPT.relative_to(ROOT).as_posix()}/{relative_name}")
        require(path.is_file(), f"sealed member is missing: {relative_name}")
        require(sha256_file(path) == digest, f"sealed member hash mismatch: {relative_name}")
        total_bytes += path.stat().st_size
        writable_members += bool(stat.S_IMODE(path.stat().st_mode) & 0o222)
        entries += 1

    require(entries == 43_056, f"unexpected sealed member count: {entries}")
    require(writable_members == 0, f"{writable_members} sealed members are writable")
    for name, digest in EXPECTED_ARTIFACT_HASHES.items():
        require(sha256_file(ATTEMPT / name) == digest, f"accepted artifact drift: {name}")
    return {
        "path": ATTEMPT.relative_to(ROOT).as_posix(),
        "tree_root_sha256": EXPECTED_TREE_ROOT,
        "sha256_manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "entries_checked": entries,
            "bytes_checked": total_bytes,
        },
        "writable_member_files": writable_members,
        "fixed_artifact_hashes": EXPECTED_ARTIFACT_HASHES,
    }


def verify_current_bindings(manifest: dict[str, Any]) -> dict[str, Any]:
    source_binding = manifest["preflight"]["source_and_rtl"]
    records = source_binding["files"]
    for record in records:
        verify_file_record(record)
    aggregate = hashlib.sha256(
        b"".join(
            record["path"].encode("utf-8")
            + b"\0"
            + record["sha256"].encode("ascii")
            + b"\n"
            for record in records
        )
    ).hexdigest()
    require(aggregate == source_binding["sha256"], "current source/RTL aggregate drift")

    rtl_records = [record for record in records if record["path"].startswith("rtl/")]
    rtl_aggregate = hashlib.sha256(
        b"".join(
            record["path"].encode("utf-8")
            + b"\0"
            + record["sha256"].encode("ascii")
            + b"\n"
            for record in rtl_records
        )
    ).hexdigest()

    reused = {}
    for key in ("model", "model_config", "adapter", "prompt"):
        record = manifest[key]
        verify_file_record(record)
        reused[key] = record
    tokenizer_files = manifest["tokenization"]["source_files"]
    for record in tokenizer_files.values():
        verify_file_record(record)
    reused["tokenizer"] = {
        "repository": manifest["tokenization"]["repository"],
        "revision": manifest["tokenization"]["revision"],
        "source_files": tokenizer_files,
        "source_files_sha256": manifest["tokenization"]["source_files_sha256"],
    }
    return {
        "accepted_source_and_rtl_sha256": source_binding["sha256"],
        "source_and_rtl_files_checked": len(records),
        "current_rtl_subset_sha256": rtl_aggregate,
        "current_rtl_files_checked": len(rtl_records),
        "reused_inputs": reused,
    }


def verify_accepted_review() -> dict[str, Any]:
    review = load_json(REVIEW)
    transition = load_json(TRANSITION)
    require(review["decision"] == "PASS_CERTIFIED", "Reviewer verdict is not PASS_CERTIFIED")
    require(review["rtl_stage_certified"] is True, "Reviewer did not certify RTL stage")
    require(
        transition["decision"] == "PASS_STAGE1_RTL_LOCAL_DEMO_EVIDENCE_CLOSED",
        "V74 transition does not close Stage-1 evidence",
    )
    accepted = transition["v74_attempt002_fresh_reviewer_acceptance"]
    require(accepted["decision"] == "PASS", "transition lacks fresh Reviewer PASS")
    require(
        accepted["sealed_attempt_tree_root_sha256"] == EXPECTED_TREE_ROOT,
        "transition binds a different attempt tree",
    )
    return {
        "reviewer_verdict": {
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(REVIEW),
            "decision": review["decision"],
        },
        "transition": {
            "path": TRANSITION.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(TRANSITION),
            "decision": transition["decision"],
        },
    }


def verify_decode(
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    snapshot = project_path(manifest["tokenization"]["snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    prompt = (ATTEMPT / "prompt.utf8").read_text(encoding="utf-8")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    require(prompt_ids == summary["prompt_token_ids"], "fresh prompt tokenization differs")
    generated = summary["generated_token_ids"]
    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    require(decoded == summary["decoded_text"], "fresh tokenizer decode differs")
    require(
        decoded == "".join(step["decoded_piece"] for step in summary["head_steps"]),
        "per-step decode pieces do not compose to the readable output",
    )
    return {
        "prompt_token_count": len(prompt_ids),
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "tokenizer_repository": manifest["tokenization"]["repository"],
        "tokenizer_revision": manifest["tokenization"]["revision"],
    }


def verify_oracle_surfaces() -> dict[str, Any]:
    summary_path = ATTEMPT / "runtime-output/run_summary.json"
    summary = load_json(summary_path)
    comparison = load_json(ATTEMPT / "independent-reference-comparison.json")
    product = load_json(ATTEMPT / "runtime-output/product_result.json")
    manifest = load_json(ATTEMPT / "attempt-manifest.json")

    run_summary_record = product["run_summary"]
    if "bytes" in run_summary_record:
        require(
            summary_path.stat().st_size == int(run_summary_record["bytes"]),
            "run summary size differs from product result",
        )
    require(
        sha256_file(summary_path) == run_summary_record["sha256"],
        "run summary hash differs from product result",
    )
    require(comparison["status"] == "PASS", "independent comparison is not PASS")
    require(
        comparison["oracle_independence"]
        == {
            "reference_derived_before_rtl_execution": True,
            "rtl_outputs_used_only_as_comparison_observations": True,
            "host_reference_consumed_rtl_as_input": False,
        },
        "oracle independence failed",
    )
    require(all(comparison["checks"].values()), "sealed comparison check failed")
    require(summary["software_transformer_or_logits_fallback"] is False, "fallback detected")

    generated = summary["generated_token_ids"]
    prompt_ids = summary["prompt_token_ids"]
    executions = summary["token_executions"]
    require(len(generated) == 2, "accepted trace is not a two-token generation")
    require(len(executions) == len(prompt_ids) + len(generated) - 1, "position count differs")

    mismatch_totals: dict[str, int] = {}
    derived_positions = []
    kv_files_checked = 0
    rtl_records_checked = 0
    for position, execution in enumerate(executions):
        expected_input = prompt_ids[position] if position < len(prompt_ids) else generated[0]
        require(execution["absolute_position"] == position, "non-contiguous token position")
        require(execution["input_token_id"] == expected_input, "causal token input differs")
        expected_phase = "prefill" if position < len(prompt_ids) else "decode"
        require(execution["phase"] == expected_phase, "prefill/decode phase differs")
        require(len(execution["layers"]) == 24, "position does not cover 24 layers")

        derived_layers = []
        for layer_id, layer in enumerate(execution["layers"]):
            require(layer["layer_id"] == layer_id, "layer order differs")
            require(layer["cache_length_before"] == position, "KV length-before differs")
            require(layer["cache_length_after"] == position + 1, "KV length-after differs")
            require(layer["rtl_cache_append"]["host_cache_append_replaced"], "host K/V retained")
            require(layer["rtl_cache_append"]["layer_id"] == layer_id, "K/V source layer differs")
            for surface, mismatches in layer["integer_boundary_mismatches"].items():
                mismatch_totals[surface] = mismatch_totals.get(surface, 0) + int(mismatches)

            execution_path = verify_file_record(layer["rtl_execution"])
            rtl_record = load_json(execution_path)
            require(
                rtl_record["integer_boundary_mismatches"]
                == layer["integer_boundary_mismatches"],
                "summary/RTL mismatch record differs",
            )
            require(rtl_record["status"] == "PASS_SINGLE_TOKEN_ALL_BOUNDARIES_RTL", "RTL failure")
            kv = rtl_record["kv_cache"]
            require(kv["status"] == "PASS_RTL_KV_APPEND_EXACT", "K/V oracle failure")
            require(kv["cache_length_before"] == position, "RTL K/V length-before differs")
            require(kv["cache_length_after"] == position + 1, "RTL K/V length-after differs")
            for name, summary_key in (
                ("rtl_observed_k", "rtl_observed_k_sha256"),
                ("rtl_observed_v", "rtl_observed_v_sha256"),
            ):
                verify_file_record(kv[name])
                require(
                    kv[name]["sha256"] == layer["rtl_cache_append"][summary_key],
                    "retained K/V bytes differ from summary",
                )
                kv_files_checked += 1
            rtl_records_checked += 1
            derived_layers.append(
                {
                    "layer_id": layer["layer_id"],
                    "cache_length_before": layer["cache_length_before"],
                    "cache_length_after": layer["cache_length_after"],
                    "integer_boundary_mismatches": layer["integer_boundary_mismatches"],
                    "rtl_cache_append": layer["rtl_cache_append"],
                }
            )
        derived_positions.append(
            {
                "absolute_position": execution["absolute_position"],
                "phase": execution["phase"],
                "input_token_id": execution["input_token_id"],
                "layers": derived_layers,
            }
        )

    require(all(value == 0 for value in mismatch_totals.values()), "nonzero oracle mismatch")
    require(comparison["positions"] == derived_positions, "independent position oracle differs")
    require(comparison["head_steps"] == summary["head_steps"], "head oracle records differ")
    require(comparison["generated_token_ids"] == generated, "generated token records differ")
    require(comparison["total_integer_mismatches"] == 0, "oracle mismatch total is nonzero")

    head_results = []
    for index, step in enumerate(summary["head_steps"]):
        require(step["generation_index"] == index, "head step order differs")
        require(step["rtl_selected_token_agreement"] is True, "head token disagreement")
        head_path = verify_file_record(step["head_execution"])
        head = load_json(head_path)
        lm_head = head["lm_head"]
        require(head["status"] == "PASS_FINAL_RMSNORM_FULL_LM_HEAD_RTL", "head RTL failed")
        require(lm_head["integer_mismatches"] == 0, "full LM-head has integer mismatches")
        require(lm_head["selected_token_agreement"] is True, "LM-head token differs")
        require(lm_head["selected_logit_agreement"] is True, "LM-head logit differs")
        require(lm_head["top_token"] == generated[index], "LM-head argmax token differs")
        require(lm_head["top_token"] == step["selected_token_id"], "head summary token differs")
        require(lm_head["top_logit_s8"] == step["selected_logit_s8"], "head logit differs")
        stdout_path = verify_file_record(lm_head["rtl"]["stdout"])
        marker = LM_HEAD_MARKER.fullmatch(stdout_path.read_text(encoding="utf-8").strip())
        require(marker is not None, "retained LM-head RTL marker is malformed")
        require(
            tuple(map(int, marker.groups())) == (
                step["source_absolute_position"],
                step["full_vocabulary_outputs"],
                step["selected_token_id"],
                step["selected_logit_s8"],
                lm_head["rtl_marker"]["cycles"],
            ),
            "retained LM-head RTL marker differs from the oracle",
        )
        head_results.append(
            {
                "generation_index": index,
                "source_absolute_position": step["source_absolute_position"],
                "output_rows": step["full_vocabulary_outputs"],
                "selected_token_id": step["selected_token_id"],
                "selected_logit_s8": step["selected_logit_s8"],
                "integer_mismatches": lm_head["integer_mismatches"],
                "rtl_stdout_sha256": lm_head["rtl"]["stdout"]["sha256"],
            }
        )

    decode = verify_decode(manifest, summary)
    require(product["decoded_text"] == decode["decoded_text"], "product decode differs")
    require(product["readability"]["accepted"] is True, "readability policy rejected output")
    return {
        "manifest": manifest,
        "summary": summary,
        "result": {
            "oracle": "independent quantized reference recorded before RTL execution",
            "positions_checked": len(executions),
            "rtl_layer_records_checked": rtl_records_checked,
            "kv_observed_files_checked": kv_files_checked,
            "integer_mismatch_totals": mismatch_totals,
            "head_steps": head_results,
            "decode": decode,
            "software_transformer_or_logits_fallback": False,
        },
    }


def tool_version(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip().splitlines()
    require(completed.returncode == 0 and output, f"tool probe failed: {argv[0]}")
    return output[0]


def makefile_rtl_sources() -> list[str]:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^RTL_SOURCES := (.+)$", makefile, flags=re.MULTILINE)
    require(match is not None, "Makefile RTL_SOURCES is missing")
    sources = match.group(1).split()
    require(sources, "Makefile RTL_SOURCES is empty")
    for source in sources:
        require(project_path(source).is_file(), f"RTL source is missing: {source}")
    return sources


def run_current_tree_smoke(output: Path) -> dict[str, Any]:
    sources = makefile_rtl_sources()
    testbench = "verification/tb/ace2_shell_tb.sv"
    binary = output / "current-tree-shell-smoke.vvp"
    compile_command = [
        "iverilog",
        "-g2012",
        "-Irtl",
        "-Iverification/tb",
        "-o",
        binary.relative_to(ROOT).as_posix(),
        *sources,
        testbench,
    ]
    simulate_command = ["vvp", binary.relative_to(ROOT).as_posix(), "+SMOKE_OPCODE=01"]

    compile_started = time.perf_counter()
    compile_result = subprocess.run(
        compile_command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    compile_seconds = time.perf_counter() - compile_started
    (output / "iverilog.stdout.log").write_text(compile_result.stdout, encoding="utf-8")
    (output / "iverilog.stderr.log").write_text(compile_result.stderr, encoding="utf-8")
    require(compile_result.returncode == 0, "fresh current-tree RTL compilation failed")

    simulate_started = time.perf_counter()
    simulate_result = subprocess.run(
        simulate_command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    simulate_seconds = time.perf_counter() - simulate_started
    (output / "vvp.stdout.log").write_text(simulate_result.stdout, encoding="utf-8")
    (output / "vvp.stderr.log").write_text(simulate_result.stderr, encoding="utf-8")
    require(simulate_result.returncode == 0, "fresh current-tree RTL simulation failed")
    require(
        re.search(
            r"^ACE2_SHELL_SMOKE_TB_PASS opcode=00000001$",
            simulate_result.stdout,
            flags=re.MULTILINE,
        )
        is not None,
        "RTL PASS marker absent",
    )
    simulator_output = simulate_result.stdout + simulate_result.stderr
    require(
        re.search(r"(?:MISMATCH|RTL_FAIL|FATAL)", simulator_output, re.IGNORECASE)
        is None,
        "fresh RTL smoke emitted a failure marker",
    )

    source_records = []
    for relative in [*sources, testbench]:
        path = project_path(relative)
        source_records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    aggregate = hashlib.sha256(
        b"".join(
            record["path"].encode("utf-8")
            + b"\0"
            + record["sha256"].encode("ascii")
            + b"\n"
            for record in source_records
        )
    ).hexdigest()
    return {
        "status": "PASS",
        "scope": "fresh current-tree Icarus shell W4A8 projection-opcode smoke",
        "compile_command": compile_command,
        "simulate_command": simulate_command,
        "source_manifest": source_records,
        "source_manifest_sha256": aggregate,
        "binary_sha256": sha256_file(binary),
        "compile_wall_seconds": compile_seconds,
        "simulation_wall_seconds": simulate_seconds,
        "latency_classification": "computer-local compile/simulation wall time only",
        "stdout_sha256": sha256_file(output / "vvp.stdout.log"),
        "stderr_sha256": sha256_file(output / "vvp.stderr.log"),
    }


def write_output_manifest(output: Path) -> None:
    members = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.name}\n"
        for path in members
    ]
    (output / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    require(output.is_relative_to(ROOT / "reports"), "--output must be under reports/")
    require(not output.exists(), f"output already exists: {output.relative_to(ROOT)}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    sealed_attempt = verify_sealed_attempt()
    accepted_review = verify_accepted_review()
    oracle = verify_oracle_surfaces()
    current_bindings = verify_current_bindings(oracle["manifest"])
    smoke = run_current_tree_smoke(output)
    summary = oracle["summary"]
    result = {
        "schema": "ace2-rtl-chat-independent-oracle-regression-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "reproduction_command": (
            "make rtl-chat-oracle-regression "
            f"ACE2_CHAT_ORACLE_REGRESSION_OUTPUT={output.relative_to(ROOT).as_posix()}"
        ),
        "accepted_v74_attempt002": sealed_attempt,
        "accepted_review": accepted_review,
        "current_bindings": current_bindings,
        "fresh_current_tree_rtl_smoke": smoke,
        "oracle_comparison": oracle["result"],
        "readable_chat_output": {
            "generated_token_ids": summary["generated_token_ids"],
            "decoded_text": summary["decoded_text"],
            "readability_policy_accepted": True,
        },
        "latency": {
            "accepted_full_chat": {
                **summary["latency"],
                "classification": (
                    "historical accepted V74 attempt002 host orchestration and "
                    "Icarus simulation wall time; not hardware latency"
                ),
            },
            "fresh_regression": {
                "total_wall_seconds": time.perf_counter() - started,
                "rtl_compile_wall_seconds": smoke["compile_wall_seconds"],
                "rtl_simulation_wall_seconds": smoke["simulation_wall_seconds"],
                "classification": "computer-local verification wall time only",
            },
        },
        "constraints": {
            "status": "NOT_RUN_OPERATOR_CANCELLED_SYNTHESIS_PPA_SCOPE",
            "reason": (
                "This verification-only regression adds no RTL capability; the operator "
                "explicitly prohibited synthesis/PPA work for this mission."
            ),
        },
        "limitations": [
            (
                "The fresh command does not regenerate the 14.5-hour, 840-layer chat "
                "attempt. It verifies its immutable 43,056-member seal, all retained "
                "per-layer K/V records, both full-head records, and a fresh tokenizer "
                "decode, then compiles and simulates a current-tree shell smoke."
            ),
            (
                "Transient per-step vectors and simulator binaries were intentionally "
                "pruned by attempt002, so full 151,936-row logits and all 840 RTL "
                "executions are not freshly replayed."
            ),
            (
                "The fresh smoke is an Icarus W4A8 shell projection check, not FPGA, "
                "synthesis, PPA, bitstream, deployed-hardware, or silicon evidence."
            ),
        ],
    }
    (output / "result.json").write_bytes(canonical_bytes(result))
    write_output_manifest(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rtl-chat-independent-oracle-regression-0006"),
    )
    args = parser.parse_args()
    result = run(args.output)
    print(f"Assistant:{result['readable_chat_output']['decoded_text']}")
    print(
        "ACE2_RTL_CHAT_ORACLE_REGRESSION "
        f"status={result['status']} "
        f"positions={result['oracle_comparison']['positions_checked']} "
        f"layers={result['oracle_comparison']['rtl_layer_records_checked']} "
        f"kv_files={result['oracle_comparison']['kv_observed_files_checked']} "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
