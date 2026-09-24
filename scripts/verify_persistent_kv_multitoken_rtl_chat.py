#!/usr/bin/env python3
"""Verify a sealed multi-token RTL chat attempt and write a durable report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LAYERS = 24
EXPECTED_MODEL_OUTPUTS = 151_936
EXPECTED_PRODUCT_STATUS = "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT"
REQUIRED_GENERATED_TOKENS = 4
MIN_DECODE_TRANSITIONS = 3
MIN_GENERATED_TOKENS = 4
MAX_GENERATED_TOKENS = 8
REQUIRED_INTEGER_BOUNDARY_SURFACES = frozenset(
    (
        "attention_compose",
        "attention_score",
        "kv_cache",
        "projection",
        "residual",
        "rmsnorm",
        "rope",
        "silu",
        "softmax",
    )
)
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_expected_generated_tokens(value: object) -> int:
    require(
        type(value) is int
        and MIN_GENERATED_TOKENS <= value <= MAX_GENERATED_TOKENS,
        "expected generated-token count must be an integer from 4 through 8",
    )
    return value


def verify_generation_contract(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    expected_generated_tokens: int,
) -> dict[str, int]:
    expected = validate_expected_generated_tokens(expected_generated_tokens)
    prompt_ids = summary["prompt_token_ids"]
    generated = summary["generated_token_ids"]
    executions = summary["token_executions"]
    require(isinstance(prompt_ids, list) and prompt_ids, "prompt token sequence is empty")
    require(isinstance(generated, list), "generated token sequence is not a list")
    require(len(generated) == expected, "runtime generated-token count differs from request")

    bounds = manifest["tokenization"]["generation_bounds"]
    context = summary["context_contract"]
    prompt_count = len(prompt_ids)
    total_context = prompt_count + expected
    processed_positions = total_context - 1
    require(bounds["max_new_tokens"] == expected, "manifest generated-token request differs")
    require(bounds["prompt_token_count"] == prompt_count, "manifest prompt-token count differs")
    require(bounds["total_context_tokens"] == total_context, "manifest total context differs")
    require(bounds["processed_positions"] == processed_positions, "manifest processed-position count differs")
    require(
        bounds["max_prompt_tokens"] == bounds["rtl_context_bound"] - expected,
        "manifest prompt limit is inconsistent with generated-token request",
    )
    require(total_context <= bounds["rtl_context_bound"], "manifest context exceeds RTL bound")
    require(summary["max_new_tokens"] == expected, "runtime generated-token request differs")
    require(context["max_new_tokens"] == expected, "runtime context token request differs")
    require(context["prompt_token_count"] == prompt_count, "runtime prompt-token count differs")
    require(context["total_context_tokens"] == total_context, "runtime total context differs")
    require(context["processed_positions"] == processed_positions, "runtime processed-position count differs")
    require(len(executions) == processed_positions, "runtime execution count is not causal")
    require(total_context <= context["backend_context_bound"], "runtime context exceeds backend bound")

    command = manifest["command"]
    argv = command["argv"]
    require(isinstance(argv, list), "manifest command argv is not a list")
    require(
        command["argv_sha256"] == hashlib.sha256(canonical_bytes(argv)).hexdigest(),
        "manifest command argv hash mismatch",
    )
    flag_indices = [index for index, argument in enumerate(argv) if argument == "--max-new-tokens"]
    require(len(flag_indices) == 1, "manifest command must contain one generated-token request")
    flag_index = flag_indices[0]
    require(flag_index + 1 < len(argv), "manifest generated-token request has no value")
    require(
        argv[flag_index + 1] == str(expected),
        "manifest command generated-token request differs",
    )
    return {
        "max_new_tokens": expected,
        "prompt_token_count": prompt_count,
        "total_context_tokens": total_context,
        "processed_positions": processed_positions,
        "decode_transitions": expected - 1,
    }


def project_path(value: str) -> Path:
    relative = Path(value)
    require(not relative.is_absolute(), f"artifact path is not project-relative: {value}")
    require(".." not in relative.parts, f"artifact path escapes project: {value}")
    return ROOT / relative


def verify_file_record(record: dict[str, Any]) -> Path:
    path = project_path(str(record["path"]))
    require(path.is_file(), f"recorded artifact is missing: {record['path']}")
    require(path.stat().st_size == int(record["bytes"]), f"size mismatch: {record['path']}")
    require(sha256_file(path) == record["sha256"], f"hash mismatch: {record['path']}")
    return path


def verify_seal(attempt: Path) -> dict[str, Any]:
    sums_path = attempt / "SHA256SUMS"
    root_path = attempt / "TREE_ROOT.sha256"
    raw = sums_path.read_bytes()
    root_fields = root_path.read_text(encoding="ascii").split()
    require(
        len(root_fields) == 2 and root_fields[1] == "SHA256SUMS",
        "invalid TREE_ROOT.sha256",
    )
    root_digest = hashlib.sha256(raw).hexdigest()
    require(root_digest == root_fields[0], "SHA256SUMS tree root mismatch")
    entries = 0
    bytes_checked = 0
    for line in raw.decode("utf-8").splitlines():
        digest, separator, relative_name = line.partition("  ")
        require(separator == "  " and HEX_DIGEST.fullmatch(digest) is not None, "invalid SHA256SUMS line")
        member = attempt / relative_name
        require(member.is_file(), f"sealed member is missing: {relative_name}")
        require(sha256_file(member) == digest, f"sealed member hash mismatch: {relative_name}")
        entries += 1
        bytes_checked += member.stat().st_size
    return {
        "tree_root_sha256": root_digest,
        "entries_checked": entries,
        "bytes_checked": bytes_checked,
    }


def verify_source_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    binding = manifest["source_and_rtl"]
    records = binding["files"]
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
    require(aggregate == binding["sha256"], "source/RTL aggregate hash mismatch")
    return {
        "sha256": aggregate,
        "files_checked": len(records),
        "rtl_files_checked": sum(
            str(record["path"]).startswith("rtl/") for record in records
        ),
    }


def verify_inputs(manifest: dict[str, Any], attempt: Path) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for key in ("model", "model_config", "adapter"):
        verify_file_record(manifest[key])
        checked[key] = manifest[key]
    prompt = manifest["prompt"]
    prompt_path = attempt / "prompt.utf8"
    require(
        prompt_path.stat().st_size == int(prompt["bytes"])
        and sha256_file(prompt_path) == prompt["sha256"],
        "prompt hash or size mismatch",
    )
    checked["prompt"] = prompt
    tokenizer_files = manifest["tokenization"]["source_files"]
    for record in tokenizer_files.values():
        verify_file_record(record)
    checked["tokenizer"] = {
        "repository": manifest["tokenization"]["repository"],
        "revision": manifest["tokenization"]["revision"],
        "source_files": tokenizer_files,
        "source_files_sha256": manifest["tokenization"]["source_files_sha256"],
    }
    return checked


def verify_decode(
    manifest: dict[str, Any],
    attempt: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        project_path(manifest["tokenization"]["snapshot"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    prompt = (attempt / "prompt.utf8").read_text(encoding="utf-8")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    require(prompt_ids == summary["prompt_token_ids"], "fresh prompt tokenization mismatch")
    generated = [int(token) for token in summary["generated_token_ids"]]
    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    require(decoded == summary["decoded_text"], "fresh whole-sequence decode mismatch")
    require(decoded == "".join(step["decoded_piece"] for step in summary["head_steps"]), "head pieces do not compose to decoded output")
    require(bool(decoded.strip()), "decoded continuation is blank")
    require("\ufffd" not in decoded, "decoded continuation contains replacement characters")
    require(any(character.isalpha() for character in decoded), "decoded continuation has no letters")
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "decoded_text": decoded,
        "prompt_token_count": len(prompt_ids),
        "whole_sequence_decode_agreement": True,
        "readable_text_observed": True,
    }


def verify_oracle_surfaces(
    summary: dict[str, Any],
    comparison: dict[str, Any],
    *,
    required_generated_tokens: int = REQUIRED_GENERATED_TOKENS,
    min_decode_transitions: int = MIN_DECODE_TRANSITIONS,
) -> dict[str, Any]:
    validate_expected_generated_tokens(required_generated_tokens)
    prompt_ids = [int(token) for token in summary["prompt_token_ids"]]
    generated = [int(token) for token in summary["generated_token_ids"]]
    require(
        min_decode_transitions == required_generated_tokens - 1,
        "decode-transition requirement is not causal",
    )
    require(
        len(generated) == required_generated_tokens,
        (
            "continuation does not contain exactly "
            f"{required_generated_tokens} generated tokens"
        ),
    )
    require(summary["software_transformer_or_logits_fallback"] is False, "software fallback detected")
    require(comparison["status"] == "PASS", "independent comparison did not pass")
    require(comparison["total_integer_mismatches"] == 0, "independent comparison has mismatches")
    require(comparison["generated_token_ids"] == generated, "comparison token sequence mismatch")
    require(comparison["head_steps"] == summary["head_steps"], "comparison head records mismatch")
    require(
        len(summary["head_steps"]) == required_generated_tokens,
        "head-step count differs from generated-token request",
    )

    executions = summary["token_executions"]
    require(
        len(executions) == len(prompt_ids) + len(generated) - 1,
        "token execution count is not causal",
    )
    mismatch_totals: dict[str, int] = {}
    kv_files_checked = 0
    decode_layers_checked = 0
    decode_transitions_checked = 0
    derived_positions = []
    for position, execution in enumerate(executions):
        expected_input = (
            prompt_ids[position]
            if position < len(prompt_ids)
            else generated[position - len(prompt_ids)]
        )
        require(execution["absolute_position"] == position, "non-contiguous token position")
        require(execution["input_token_id"] == expected_input, "generated token feedback mismatch")
        expected_phase = "prefill" if position < len(prompt_ids) else "decode"
        require(execution["phase"] == expected_phase, "prefill/decode phase mismatch")
        decode_transitions_checked += expected_phase == "decode"
        require(len(execution["layers"]) == EXPECTED_LAYERS, "position does not cover 24 layers")
        derived_layers = []
        for layer_id, layer in enumerate(execution["layers"]):
            require(layer["layer_id"] == layer_id, "layer order mismatch")
            require(layer["cache_length_before"] == position, "K/V length-before mismatch")
            require(layer["cache_length_after"] == position + 1, "K/V length-after mismatch")
            append = layer["rtl_cache_append"]
            require(append["host_cache_append_replaced"] is True, "host K/V append was retained")
            require(append["absolute_position"] == position, "K/V source position mismatch")
            require(append["layer_id"] == layer_id, "K/V source layer mismatch")
            rtl_path = verify_file_record(layer["rtl_execution"])
            rtl = load_json(rtl_path)
            require(rtl["status"] == "PASS_SINGLE_TOKEN_ALL_BOUNDARIES_RTL", "RTL layer status failed")
            require(rtl["integer_boundary_mismatches"] == layer["integer_boundary_mismatches"], "RTL/summary mismatch record differs")
            kv = rtl["kv_cache"]
            require(kv["status"] == "PASS_RTL_KV_APPEND_EXACT", "K/V oracle status failed")
            require(kv["cache_length_before"] == position, "RTL K/V length-before mismatch")
            require(kv["cache_length_after"] == position + 1, "RTL K/V length-after mismatch")
            for key, digest_key in (
                ("rtl_observed_k", "rtl_observed_k_sha256"),
                ("rtl_observed_v", "rtl_observed_v_sha256"),
            ):
                require(int(kv[key]["bytes"]) > 0, "K/V comparison payload is empty")
                verify_file_record(kv[key])
                require(kv[key]["sha256"] == append[digest_key], "K/V digest differs from summary")
                kv_files_checked += 1
            require(
                set(layer["integer_boundary_mismatches"])
                == REQUIRED_INTEGER_BOUNDARY_SURFACES,
                "integer-boundary comparison coverage is incomplete",
            )
            for surface, mismatches in layer["integer_boundary_mismatches"].items():
                mismatch_totals[surface] = mismatch_totals.get(surface, 0) + int(mismatches)
            decode_layers_checked += expected_phase == "decode"
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
    require(all(value == 0 for value in mismatch_totals.values()), "nonzero integer mismatch")
    require(
        set(mismatch_totals) == REQUIRED_INTEGER_BOUNDARY_SURFACES,
        "integer-boundary comparison totals are incomplete",
    )
    require(comparison["positions"] == derived_positions, "independent position comparison differs")
    require(
        decode_transitions_checked >= min_decode_transitions,
        (
            "fewer than "
            f"{min_decode_transitions} persistent-K/V decode transitions"
        ),
    )

    head_results = []
    for index, step in enumerate(summary["head_steps"]):
        require(step["generation_index"] == index, "head step order mismatch")
        require(step["full_vocabulary_outputs"] == EXPECTED_MODEL_OUTPUTS, "head did not cover full vocabulary")
        require(step["rtl_selected_token_agreement"] is True, "RTL/head token disagreement")
        require(step["selected_token_id"] == generated[index], "head token differs from decode")
        head = load_json(verify_file_record(step["head_execution"]))
        lm_head = head["lm_head"]
        require(head["status"] == "PASS_FINAL_RMSNORM_FULL_LM_HEAD_RTL", "head RTL status failed")
        require(lm_head["integer_mismatches"] == 0, "head has integer mismatches")
        require(lm_head["selected_token_agreement"] is True, "head token oracle mismatch")
        require(lm_head["selected_logit_agreement"] is True, "head logit oracle mismatch")
        require(lm_head["top_token"] == step["selected_token_id"], "head selected token mismatch")
        require(lm_head["top_logit_s8"] == step["selected_logit_s8"], "head selected logit mismatch")
        head_results.append(
            {
                "generation_index": index,
                "source_absolute_position": step["source_absolute_position"],
                "selected_token_id": step["selected_token_id"],
                "selected_logit_s8": step["selected_logit_s8"],
                "output_rows": step["full_vocabulary_outputs"],
                "integer_mismatches": lm_head["integer_mismatches"],
            }
        )
    return {
        "positions_checked": len(executions),
        "rtl_layer_records_checked": len(executions) * EXPECTED_LAYERS,
        "decode_layers_checked": decode_layers_checked,
        "decode_transitions_checked": decode_transitions_checked,
        "kv_observed_files_checked": kv_files_checked,
        "integer_mismatch_totals": mismatch_totals,
        "integer_boundary_comparisons_checked": (
            len(executions) * EXPECTED_LAYERS * len(REQUIRED_INTEGER_BOUNDARY_SURFACES)
        ),
        "head_steps": head_results,
        "persistent_kv_across_decode_steps": (
            decode_layers_checked >= EXPECTED_LAYERS * min_decode_transitions
        ),
        "oracle_scope": {
            "per_position_quantized_oracle_agreement": True,
            "causal_rtl_kv_feedback": True,
            "independent_whole_sequence_reference_cache": False,
            "erratum": (
                "The sealed comparison metadata incorrectly characterizes RTL outputs "
                "as comparison-only. Later positions consume RTL-observed K/V after "
                "host_cache_append_replaced=true; the zero-mismatch result is therefore "
                "per-position oracle agreement with causal RTL K/V feedback, not an "
                "independently maintained whole-sequence reference cache."
            ),
        },
        "software_transformer_or_logits_fallback": False,
    }


def write_report(output: Path, result: dict[str, Any]) -> None:
    decoded = result["decode"]["decoded_text"].replace("`", "\\`")
    lines = [
        "# Persistent K/V multi-token RTL chat verification",
        "",
        f"**Status:** {result['status']}",
        "",
        "The fresh computer-local run generated "
        f"{result['decode']['generated_token_count']} RTL-selected tokens and decoded "
        f"them as `{decoded}`. The per-position quantized oracle matched every checked "
        "integer boundary, K/V append, selected logit, token ID, and whole-sequence decode.",
        "",
        "## Reproduction",
        "",
        "```bash",
        result["reproduction_command"],
        "```",
        "",
        "Supply another natural-language prompt with "
        "`ACE2_CHAT_MULTITOKEN_ARGS=\"--prompt-file <utf8-file>\"` and fresh output paths.",
        "",
        "## Evidence",
        "",
        f"- Attempt tree root: `{result['attempt']['tree_root_sha256']}`",
        f"- Source/RTL aggregate: `{result['source_binding']['sha256']}`",
        f"- Prompt SHA-256: `{result['reused_inputs']['prompt']['sha256']}`",
        f"- Model SHA-256: `{result['reused_inputs']['model']['sha256']}`",
        f"- Model config SHA-256: `{result['reused_inputs']['model_config']['sha256']}`",
        f"- Adapter SHA-256: `{result['reused_inputs']['adapter']['sha256']}`",
        f"- Tokenizer source aggregate: `{result['reused_inputs']['tokenizer']['source_files_sha256']}`",
        f"- Positions / RTL layer records: {result['oracle']['positions_checked']} / {result['oracle']['rtl_layer_records_checked']}",
        f"- Decode-layer checks: {result['oracle']['decode_layers_checked']}",
        f"- Retained RTL K/V files checked: {result['oracle']['kv_observed_files_checked']}",
        f"- Full-vocabulary head steps checked: {len(result['oracle']['head_steps'])}",
        f"- Integer mismatch totals: `{json.dumps(result['oracle']['integer_mismatch_totals'], sort_keys=True)}`",
        "",
        "## Oracle scope and erratum",
        "",
        result["oracle"]["oracle_scope"]["erratum"],
        "",
        "The sealed attempt remains unchanged. Its comparison metadata fields claiming "
        "comparison-only RTL observations and no host-reference consumption of RTL input "
        "are not used to establish this report's result.",
        "",
        "## Simulation/host-only timing",
        "",
        f"- Total wall: {result['latency']['total_wall_seconds']:.6f} s",
        f"- Icarus compile: {result['latency']['compile_wall_seconds']:.6f} s",
        f"- Icarus simulation: {result['latency']['simulation_wall_seconds']:.6f} s",
        f"- Host model/orchestration: {result['latency']['model_wall_seconds']:.6f} s",
        "",
        "These are computer-local host and RTL-simulation measurements, not hardware latency.",
        "",
        "## Limitations",
        "",
        "- This evidence covers the checked prompt and greedy continuation. `aside crystal sow/ic` passes the predeclared syntactic readability checks but does not establish semantic dialogue quality.",
        "- Per-position zero mismatch with causal RTL K/V feedback is not proof of independently maintained whole-sequence software-cache parity.",
        "- No FPGA, synthesis, STA, PPA, bitstream, deployed-hardware, or silicon claim is made.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    attempt: Path,
    output: Path,
    *,
    expected_generated_tokens: int = REQUIRED_GENERATED_TOKENS,
) -> dict[str, Any]:
    attempt = attempt.resolve()
    output = output.resolve()
    require(attempt.is_relative_to(ROOT / "reports" / "verification"), "--attempt must be under reports/verification")
    require(output.is_relative_to(ROOT / "reports" / "verification"), "--output must be under reports/verification")
    require(not output.exists(), f"verification output already exists: {output}")
    expected_generated_tokens = validate_expected_generated_tokens(
        expected_generated_tokens
    )

    seal = verify_seal(attempt)
    manifest = load_json(attempt / "attempt-manifest.json")
    attempt_result = load_json(attempt / "attempt-result.json")
    worker = load_json(attempt / "worker-result.json")
    comparison = load_json(attempt / "independent-reference-comparison.json")
    summary = load_json(attempt / "runtime-output/run_summary.json")
    product = load_json(attempt / "runtime-output/product_result.json")
    timing = load_json(attempt / "timing.json")

    require(attempt_result["status"] == "PASS", "attempt result is not PASS")
    require(worker["status"] == EXPECTED_PRODUCT_STATUS, "worker result is not coherent PASS")
    require(product["status"] == EXPECTED_PRODUCT_STATUS, "product result is not coherent PASS")
    require(worker["readability"]["accepted"] is True, "worker readability gate failed")
    require(product["readability"]["accepted"] is True, "product readability gate failed")
    require(summary["generated_token_ids"] == attempt_result["generated_token_ids"], "attempt token record mismatch")
    require(summary["generated_token_ids"] == worker["generated_token_ids"], "worker token record mismatch")
    require(summary["generated_token_ids"] == product["generated_token_ids"], "product token record mismatch")

    generation_contract = verify_generation_contract(
        manifest,
        summary,
        expected_generated_tokens,
    )
    source_binding = verify_source_binding(manifest)
    reused_inputs = verify_inputs(manifest, attempt)
    decode = verify_decode(manifest, attempt, summary)
    oracle = verify_oracle_surfaces(
        summary,
        comparison,
        required_generated_tokens=expected_generated_tokens,
        min_decode_transitions=expected_generated_tokens - 1,
    )
    require(
        oracle["persistent_kv_across_decode_steps"],
        "persistent-K/V decode-transition coverage is incomplete",
    )

    latency = timing["backend_latency"]
    require(isinstance(latency, dict), "backend latency record is absent")
    result = {
        "schema": "ace2-persistent-kv-multitoken-rtl-chat-verification-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "reproduction_command": (
            "make persistent-kv-multitoken-rtl-chat "
            f"ACE2_CHAT_MULTITOKEN_OUTPUT={attempt.relative_to(ROOT).as_posix()} "
            "ACE2_CHAT_MULTITOKEN_ARGS=\"--prompt-file <utf8-file>\" && "
            "make persistent-kv-multitoken-rtl-chat-verify "
            f"ACE2_CHAT_MULTITOKEN_OUTPUT={attempt.relative_to(ROOT).as_posix()} "
            f"ACE2_CHAT_MULTITOKEN_VERIFICATION_OUTPUT={output.relative_to(ROOT).as_posix()}"
        ),
        "attempt": {
            "path": attempt.relative_to(ROOT).as_posix(),
            **seal,
        },
        "source_binding": source_binding,
        "reused_inputs": reused_inputs,
        "generation_contract": generation_contract,
        "decode": decode,
        "oracle": oracle,
        "latency": {
            **latency,
            "classification": "computer-local host/orchestration and Icarus simulation wall time only",
        },
        "constraints": {
            "max_new_tokens": manifest["tokenization"]["generation_bounds"]["max_new_tokens"],
            "context_contract": summary["context_contract"],
            "timeout_seconds": manifest["timeout_seconds"],
            "hardware_flow": "NOT_RUN_OPERATOR_CANCELLED",
        },
        "limitations": [
            "Checked one prompt and greedy continuation; arbitrary UTF-8 prompts remain accepted by the command interface.",
            "Syntactic readability does not establish semantic dialogue quality.",
            "Per-position zero mismatch uses causal RTL K/V feedback and is not independently maintained whole-sequence software-cache parity.",
            "No FPGA, synthesis, STA, PPA, bitstream, deployed-hardware, or silicon claim.",
        ],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_bytes(canonical_bytes(result))
    write_report(output, result)
    members = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-generated-tokens",
        type=int,
        default=REQUIRED_GENERATED_TOKENS,
    )
    args = parser.parse_args()
    result = run(
        args.attempt,
        args.output,
        expected_generated_tokens=args.expected_generated_tokens,
    )
    print(
        "PERSISTENT_KV_MULTITOKEN_RTL_CHAT_PASS "
        f"tokens={result['decode']['generated_token_count']} "
        f"positions={result['oracle']['positions_checked']} "
        f"output={Path(args.output)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PERSISTENT_KV_MULTITOKEN_RTL_CHAT_FAIL detail={error}", file=os.sys.stderr)
        raise SystemExit(1)
