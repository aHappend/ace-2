#!/usr/bin/env python3
"""Reconstruct a sealed RTL chat with a host-only persistent-K/V oracle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import resource
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "tools"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import verify_persistent_kv_multitoken_rtl_chat as sealed_verifier  # noqa: E402
import ace2_layer23_v_rank1_integer_correction_reference as rank1_reference  # noqa: E402
import rtl_arbitrary_text_generation_backend as backend  # noqa: E402


EXPECTED_LAYERS = 24
EXPECTED_MODEL_OUTPUTS = 151_936
BACKEND_SOURCE = "tools/rtl_arbitrary_text_generation_backend.py"
RECOVERABLE_RSS_FAILURE = (
    "ACE2_STAGE1_CHAT_ATTEMPT_SETUP_FAIL "
    "detail=process-tree RSS missed an iverilog/vvp child"
)


class OracleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def validate_expected_generated_tokens(value: object) -> int:
    return sealed_verifier.validate_expected_generated_tokens(value)


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), f"artifact is outside project: {path}")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_checksum_manifest(directory: Path) -> dict[str, Any]:
    sums_path = directory / "SHA256SUMS"
    require(sums_path.is_file(), f"missing checksum manifest: {sums_path}")
    entries = 0
    for line in sums_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative_name = line.partition("  ")
        member_name = Path(relative_name)
        require(
            separator == "  "
            and not member_name.is_absolute()
            and ".." not in member_name.parts,
            f"invalid checksum member: {relative_name}",
        )
        member = directory / member_name
        require(member.is_file(), f"missing checksum member: {member}")
        require(sha256_file(member) == digest, f"checksum mismatch: {member}")
        entries += 1
    require(entries > 0, f"empty checksum manifest: {sums_path}")
    return {
        "path": sums_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(sums_path),
        "entries_checked": entries,
    }


def ast_without_rss_tracker(payload: bytes) -> str:
    module = ast.parse(payload.decode("utf-8"))
    tracker_count = sum(
        isinstance(node, ast.ClassDef) and node.name == "ProcessTreeRssTracker"
        for node in module.body
    )
    require(tracker_count == 1, "backend must define exactly one RSS tracker class")
    module.body = [
        node
        for node in module.body
        if not (
            isinstance(node, ast.ClassDef)
            and node.name == "ProcessTreeRssTracker"
        )
    ]
    return ast.dump(module, include_attributes=False)


def verify_source_binding(
    manifest: dict[str, Any],
    *,
    allow_rss_tracker_repair: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = manifest["source_and_rtl"]
    records = binding["files"]
    mismatches: list[tuple[dict[str, Any], Path, str]] = []
    current_records: list[tuple[str, str]] = []
    for record in records:
        path = sealed_verifier.project_path(str(record["path"]))
        require(path.is_file(), f"recorded artifact is missing: {record['path']}")
        current_digest = sha256_file(path)
        current_records.append((str(record["path"]), current_digest))
        if (
            path.stat().st_size != int(record["bytes"])
            or current_digest != record["sha256"]
        ):
            mismatches.append((record, path, current_digest))

    expected_aggregate = hashlib.sha256(
        b"".join(
            record["path"].encode("utf-8")
            + b"\0"
            + record["sha256"].encode("ascii")
            + b"\n"
            for record in records
        )
    ).hexdigest()
    require(expected_aggregate == binding["sha256"], "source/RTL aggregate hash mismatch")

    if not mismatches:
        return (
            {
                "sha256": expected_aggregate,
                "files_checked": len(records),
                "rtl_files_checked": sum(
                    str(record["path"]).startswith("rtl/") for record in records
                ),
            },
            {
                "status": "EXACT_CURRENT_SOURCE_MATCH",
                "changed_paths": [],
            },
        )

    require(
        allow_rss_tracker_repair,
        "current source differs from the sealed attempt binding",
    )
    require(
        len(mismatches) == 1 and mismatches[0][0]["path"] == BACKEND_SOURCE,
        "recovery source delta is not confined to the RSS backend",
    )
    record, current_path, current_digest = mismatches[0]
    indexed = subprocess.run(
        ["git", "show", f":{BACKEND_SOURCE}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    require(indexed.returncode == 0, "sealed backend source is unavailable from git index")
    require(
        len(indexed.stdout) == int(record["bytes"])
        and sha256_bytes(indexed.stdout) == record["sha256"],
        "git-indexed backend does not match the sealed attempt binding",
    )
    require(
        ast_without_rss_tracker(indexed.stdout)
        == ast_without_rss_tracker(current_path.read_bytes()),
        "backend changes extend beyond ProcessTreeRssTracker",
    )
    current_aggregate = hashlib.sha256(
        b"".join(
            path.encode("utf-8")
            + b"\0"
            + digest.encode("ascii")
            + b"\n"
            for path, digest in current_records
        )
    ).hexdigest()
    return (
        {
            "sha256": expected_aggregate,
            "files_checked": len(records),
            "rtl_files_checked": sum(
                str(record["path"]).startswith("rtl/") for record in records
            ),
        },
        {
            "status": "COMPATIBLE_RSS_TRACKER_ONLY_REPAIR",
            "changed_paths": [BACKEND_SOURCE],
            "sealed_backend_sha256": record["sha256"],
            "current_backend_sha256": current_digest,
            "current_source_aggregate_sha256": current_aggregate,
            "proof": (
                "the git-indexed backend matches the sealed manifest byte-for-byte; "
                "the current backend AST differs only in ProcessTreeRssTracker"
            ),
        },
    )


def verify_attempt_admission(
    attempt: Path,
    attempt_result: dict[str, Any],
    summary: dict[str, Any],
    *,
    recover_failed_attempt: bool,
) -> dict[str, Any]:
    if attempt_result.get("status") == "PASS":
        require(
            not recover_failed_attempt,
            "recovery mode requires a failed attempt",
        )
        return {
            "mode": "normal_passed_attempt",
            "attempt_status": "PASS",
            "telemetry_status": "AS_RECORDED_BY_ATTEMPT",
        }

    require(
        recover_failed_attempt,
        "attempt result is not PASS; use explicit recovery only for an authenticated complete trace",
    )
    require(attempt_result.get("status") == "FAIL", "recovery attempt status is not FAIL")
    require(attempt_result.get("exit_code") == 3, "recovery attempt exit code differs")
    require(attempt_result.get("timed_out") is False, "recovery attempt timed out")
    stderr_path = attempt / "stderr.raw.log"
    require(
        stderr_path.read_text(encoding="utf-8").strip() == RECOVERABLE_RSS_FAILURE,
        "recovery failure is not the exact process-tree RSS classification",
    )
    generated = summary.get("generated_token_ids")
    executions = summary.get("token_executions")
    termination = summary.get("termination")
    context = summary.get("context_contract")
    require(
        summary.get("status") == "PASS_STAGE1_PRODUCT_RTL_GENERATION_UNCHECKED",
        "retained backend summary is not numerically complete",
    )
    require(
        isinstance(generated, list) and len(generated) == 4,
        "retained backend summary does not contain four generated tokens",
    )
    require(
        isinstance(executions, list)
        and isinstance(context, dict)
        and len(executions) == int(context.get("processed_positions", -1)),
        "retained backend summary has an incomplete position trace",
    )
    require(
        isinstance(termination, dict)
        and termination.get("reason") == "max_new_tokens"
        and termination.get("generated_token_count") == 4,
        "retained backend summary lacks complete terminal evidence",
    )
    return {
        "mode": "explicit_sealed_failed_attempt_numerical_recovery",
        "attempt_status": "FAIL",
        "attempt_result": file_record(attempt / "attempt-result.json"),
        "failure_log": file_record(stderr_path),
        "failure_classification": "PROCESS_TREE_RSS_TELEMETRY",
        "numerical_execution_complete": True,
        "telemetry_status": "UNRESOLVED_HISTORICAL_PROCESS_TREE_RSS",
        "original_attempt_remains_failed": True,
    }


def retained_backend_timing(
    timing_record: dict[str, Any],
    summary: dict[str, Any],
    *,
    recover_failed_attempt: bool,
) -> tuple[dict[str, Any], str]:
    backend_latency = timing_record.get("backend_latency")
    if isinstance(backend_latency, dict):
        return backend_latency, "attempt_timing_backend_latency"
    require(
        recover_failed_attempt and backend_latency is None,
        "backend latency record is absent",
    )
    summary_latency = summary.get("latency")
    required_fields = {
        "compile_wall_seconds",
        "model_wall_seconds",
        "simulation_wall_seconds",
        "total_wall_seconds",
    }
    require(
        isinstance(summary_latency, dict)
        and required_fields.issubset(summary_latency)
        and all(
            isinstance(summary_latency[field], (int, float))
            for field in required_fields
        ),
        "retained run summary lacks measured backend phase timing",
    )
    return (
        dict(summary_latency),
        "sealed_runtime_output_run_summary_latency_due_to_null_wrapper_record",
    )


def run_representative_rss_reproduction() -> dict[str, Any]:
    commands = (["iverilog", "-V"], ["vvp", "-V"])
    records = []
    with backend.process_tree_rss_tracking(interval_seconds=0.001) as tracker:
        for command in commands:
            completed, child_record = backend.tracked_run(command, cwd=ROOT)
            require(
                completed.returncode == 0,
                f"representative RSS child failed: {command[0]}",
            )
            require(child_record is not None, f"RSS child was not registered: {command[0]}")
            records.append(child_record)
    summary = tracker.summary(require_complete=True)
    return {
        "status": "PASS",
        "scope": "fresh_representative_children_only_not_historical_case_rss",
        "commands": [backend.display_command(command) for command in commands],
        "registered_children": records,
        "process_tree_rss": summary,
    }


def s8_bytes(value: torch.Tensor | list[int] | tuple[int, ...]) -> bytes:
    if isinstance(value, torch.Tensor):
        values = value.to(torch.int64).reshape(-1).tolist()
    else:
        values = value
    return bytes(int(item) & 0xFF for item in values)


class SoftwareRank1Reference:
    """Apply the accepted rank-1 correction without invoking or observing RTL."""

    def __init__(self) -> None:
        self.config = rank1_reference.load_frozen_config(check_review=False)
        self.application_count = 0

    def apply(
        self,
        _working: Path,
        activation_q: torch.Tensor,
        baseline_projection: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        activation = activation_q.to(torch.int64).reshape(-1).tolist()
        baseline = (
            baseline_projection["output_q"].to(torch.int64).reshape(-1).tolist()
        )
        result = rank1_reference.apply_rank1_correction(
            activation,
            baseline,
            self.config,
        )
        corrected = torch.tensor(result.corrected_v_s8, dtype=torch.int8)
        self.application_count += 1
        return corrected, {
            "status": "PASS_INDEPENDENT_SOFTWARE_RANK1_REFERENCE",
            "v_source": "host_only_frozen_rank1_integer_reference",
            "baseline_v_sha256": sha256_bytes(s8_bytes(baseline)),
            "corrected_v_sha256": sha256_bytes(s8_bytes(result.corrected_v_s8)),
            "corrected_v_bytes": len(result.corrected_v_s8),
            "rank_accumulator_s32": result.rank_accumulator_s32,
            "rank_rounded_s32": result.rank_rounded_s32,
            "rank_intermediate_s8": result.rank_intermediate_s8,
            "rank_saturation": result.rank_saturation,
            "correction_saturation_count": sum(result.correction_saturation),
            "add_saturation_count": sum(result.add_saturation),
        }


def derive_head_weights(
    embedding: torch.Tensor,
    output_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    require(
        output_count == EXPECTED_MODEL_OUTPUTS,
        "oracle output domain differs from the accepted full vocabulary",
    )
    qweight = torch.empty((output_count, backend.HIDDEN), dtype=torch.int8)
    weight_scale = torch.empty((output_count,), dtype=torch.float64)
    for start in range(0, output_count, 2048):
        stop = min(output_count, start + 2048)
        rows = embedding[start:stop].to(torch.float64)
        scale = rows.abs().amax(dim=1) / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight[start:stop] = (
            torch.round(rows / scale[:, None]).clamp(-8, 7).to(torch.int8)
        )
        weight_scale[start:stop] = scale
    return qweight, weight_scale


def derive_head(
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> dict[str, Any]:
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    float_outputs = []
    for start in range(0, EXPECTED_MODEL_OUTPUTS, 4096):
        stop = min(EXPECTED_MODEL_OUTPUTS, start + 4096)
        float_outputs.append(
            torch.mv(
                embedding[start:stop].to(torch.float32),
                item["float_norm"],
            )
        )
    float_output = torch.cat(float_outputs).contiguous()
    output_scale = backend.canonical.scale_for(float_output)
    multiplier, right_shift = backend.canonical.derive_multiplier(
        item["final_scale"] * weight_scale / output_scale
    )
    accumulator = torch.empty((EXPECTED_MODEL_OUTPUTS,), dtype=torch.int64)
    activation = item["final_q"].to(torch.int32)
    for start in range(0, EXPECTED_MODEL_OUTPUTS, 4096):
        stop = min(EXPECTED_MODEL_OUTPUTS, start + 4096)
        accumulator[start:stop] = (
            qweight[start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    require(bool(torch.all(accumulator >= -(1 << 31))), "LM-head underflow")
    require(bool(torch.all(accumulator < (1 << 31))), "LM-head overflow")
    rounded = backend.lm_head.round_outputs(
        accumulator,
        multiplier,
        right_shift,
    )
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = (rounded < -128) | (rounded > 127)
    top_token, top_logit = backend.lm_head.top_token(output_q)
    return {
        "final_rmsnorm": s8_bytes(item["final_q"]),
        "final_rmsnorm_scale": struct.pack("<d", float(item["final_scale"])),
        "logits": s8_bytes(output_q),
        "logit_scale": struct.pack("<d", float(output_scale)),
        "selected_token_id": int(top_token),
        "selected_logit_s8": int(top_logit),
        "saturation_count": int(saturation.sum().item()),
        "accumulator_min": int(accumulator.min().item()),
        "accumulator_max": int(accumulator.max().item()),
    }


def tokenize_prompt(
    manifest: dict[str, Any],
    attempt: Path,
) -> tuple[Any, list[int], str, float]:
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        sealed_verifier.project_path(manifest["tokenization"]["snapshot"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    prompt = (attempt / "prompt.utf8").read_text(encoding="utf-8")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    require(
        isinstance(prompt_ids, list) and all(isinstance(item, int) for item in prompt_ids),
        "tokenizer did not return an integer token sequence",
    )
    return tokenizer, prompt_ids, prompt, time.monotonic() - started


def reconstruct(
    manifest: dict[str, Any],
    attempt: Path,
    *,
    tokenizer: Any | None = None,
    prompt_ids: list[int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = time.process_time()
    supplied_prompt_ids = tokenizer is not None and prompt_ids is not None
    if tokenizer is None or prompt_ids is None:
        require(
            tokenizer is None and prompt_ids is None,
            "tokenizer and prompt IDs must be supplied together",
        )
        tokenizer, prompt_ids, _prompt, tokenization_seconds = tokenize_prompt(
            manifest,
            attempt,
        )
    else:
        require(
            bool(prompt_ids) and all(type(item) is int for item in prompt_ids),
            "supplied prompt IDs are invalid",
        )
        prompt_ids = list(prompt_ids)
        tokenization_seconds = 0.0
    generation = manifest["tokenization"]["generation_bounds"]
    if supplied_prompt_ids:
        max_new_tokens = generation["max_new_tokens"]
        require(
            type(max_new_tokens) is int and max_new_tokens > 0,
            "supplied reconstruction token count is invalid",
        )
    else:
        max_new_tokens = validate_expected_generated_tokens(
            generation["max_new_tokens"]
        )
    require(
        len(prompt_ids) + max_new_tokens <= backend.MAX_CONTEXT_TOKENS,
        "oracle request exceeds the RTL context contract",
    )

    model_path = sealed_verifier.project_path(manifest["model"]["path"])
    adapter_path = sealed_verifier.project_path(manifest["adapter"]["path"])
    backend.configure_snapshot(model_path.parent)
    caches = [backend.empty_layer_cache() for _ in range(EXPECTED_LAYERS)]
    templates: list[dict[str, Any] | None] = [None] * EXPECTED_LAYERS
    rank1 = SoftwareRank1Reference()
    positions: list[dict[str, Any]] = []
    heads: list[dict[str, Any]] = []
    generated: list[int] = []
    model_started = time.monotonic()

    with (
        torch.no_grad(),
        safe_open(model_path, framework="pt", device="cpu") as weights,
        safe_open(adapter_path, framework="pt", device="cpu") as adapter,
    ):
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head_qweight, head_weight_scale = derive_head_weights(
            embedding,
            EXPECTED_MODEL_OUTPUTS,
        )
        position = 0
        while True:
            if position < len(prompt_ids):
                token_id = prompt_ids[position]
            else:
                token_id = generated[position - len(prompt_ids)]
            state = backend.embedding_state(weights, token_id, position)
            layer_records = []
            for layer_id in range(EXPECTED_LAYERS):
                derived, state, templates[layer_id] = backend.derive_layer_token(
                    layer_id,
                    state,
                    caches[layer_id],
                    templates[layer_id],
                    weights,
                    adapter,
                    rank1_sidecar=rank1 if layer_id == EXPECTED_LAYERS - 1 else None,
                    sidecar_working=(
                        ROOT
                        / "reports/verification"
                        / ".unused-host-rank1"
                        / f"position-{position:02d}"
                    ),
                )
                cache = caches[layer_id]
                layer_records.append(
                    {
                        "layer_id": layer_id,
                        "k_append": s8_bytes(cache["k"][-1]),
                        "v_append": s8_bytes(cache["v"][-1]),
                        "full_k_cache": s8_bytes(
                            [value for row in cache["k"] for value in row]
                        ),
                        "full_v_cache": s8_bytes(
                            [value for row in cache["v"] for value in row]
                        ),
                        "layer_output": s8_bytes(state["fixed_q"]),
                        "layer_output_scale": struct.pack(
                            "<d",
                            float(state["fixed_scale"]),
                        ),
                        "rank1": derived["positions"][0]["projections"]["v"].get(
                            "rank1_sidecar"
                        ),
                    }
                )
            positions.append(
                {
                    "absolute_position": position,
                    "phase": "prefill" if position < len(prompt_ids) else "decode",
                    "input_token_id": token_id,
                    "layers": layer_records,
                }
            )

            if position >= len(prompt_ids) - 1:
                head = derive_head(
                    state,
                    norm_gain,
                    embedding,
                    head_qweight,
                    head_weight_scale,
                )
                selected = int(head["selected_token_id"])
                piece = tokenizer.decode(
                    [selected],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                require(bool(piece), "oracle selected an empty token piece")
                head["generation_index"] = len(generated)
                head["source_absolute_position"] = position
                head["decoded_piece"] = piece
                heads.append(head)
                generated.append(selected)
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if selected == eos_token_id or len(generated) == max_new_tokens:
                    break
            position += 1
            print(
                "FULL_CHAIN_ORACLE_PROGRESS "
                f"position={position - 1} generated={len(generated)}",
                flush=True,
            )

    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "positions": positions,
        "heads": heads,
        "rank1_application_count": rank1.application_count,
        "timing": {
            "total_host_wall_seconds": time.monotonic() - started,
            "tokenization_wall_seconds": tokenization_seconds,
            "model_and_head_wall_seconds": time.monotonic() - model_started,
            "host_process_cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def exact_artifact_bytes(record: dict[str, Any]) -> bytes:
    return sealed_verifier.verify_file_record(record).read_bytes()


def require_payload_matches_record(
    payload: bytes,
    record: dict[str, Any],
    label: str,
) -> int:
    retained = exact_artifact_bytes(record)
    require(len(payload) == len(retained), f"{label} byte count differs")
    require(payload == retained, f"{label} retained bytes differ")
    return len(retained)


def verify_retained_comparison_payloads(summary: dict[str, Any]) -> dict[str, int]:
    records_checked = 0
    bytes_checked = 0
    for position in summary["token_executions"]:
        for layer in position["layers"]:
            rtl = sealed_verifier.load_json(
                sealed_verifier.verify_file_record(layer["rtl_execution"])
            )
            for name in ("layer_output_s8.bin", "layer_output_scale_f64le.bin"):
                bytes_checked += len(exact_artifact_bytes(rtl["tensors"][name]))
                records_checked += 1
    for step in summary["head_steps"]:
        head = sealed_verifier.load_json(
            sealed_verifier.verify_file_record(step["head_execution"])
        )
        artifacts = head["lm_head"]["artifacts"]
        for name in (
            "final_rmsnorm_s8",
            "final_rmsnorm_scale_f64le",
            "lm_head_output_s8",
            "lm_head_output_scale_f64le",
        ):
            bytes_checked += len(exact_artifact_bytes(artifacts[name]))
            records_checked += 1
    return {
        "records_checked": records_checked,
        "bytes_checked": bytes_checked,
    }


def compare_reference(
    reference: dict[str, Any],
    summary: dict[str, Any],
    *,
    initial_cache_prefixes: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    require(
        reference["prompt_token_ids"] == summary["prompt_token_ids"],
        "independent prompt tokenization differs",
    )
    require(
        reference["generated_token_ids"] == summary["generated_token_ids"],
        "independent greedy token sequence differs",
    )
    require(
        reference["decoded_text"] == summary["decoded_text"],
        "independent decoded continuation differs",
    )
    require(
        len(reference["positions"]) == len(summary["token_executions"]),
        "independent position count differs",
    )

    append_bytes = 0
    full_cache_bytes = 0
    layer_output_bytes = 0
    layer_output_scale_bytes = 0
    rank1_checks = 0
    if initial_cache_prefixes is None:
        initial_cache_prefixes = [(b"", b"") for _ in range(EXPECTED_LAYERS)]
    require(
        len(initial_cache_prefixes) == EXPECTED_LAYERS
        and all(
            isinstance(k_prefix, bytes) and isinstance(v_prefix, bytes)
            for k_prefix, v_prefix in initial_cache_prefixes
        ),
        "initial cache prefixes are invalid",
    )
    rtl_full_k_by_layer = [
        bytearray(k_prefix) for k_prefix, _ in initial_cache_prefixes
    ]
    rtl_full_v_by_layer = [
        bytearray(v_prefix) for _, v_prefix in initial_cache_prefixes
    ]
    for actual, expected in zip(
        reference["positions"],
        summary["token_executions"],
        strict=True,
    ):
        require(
            actual["absolute_position"] == expected["absolute_position"]
            and actual["phase"] == expected["phase"]
            and actual["input_token_id"] == expected["input_token_id"],
            "independent causal token record differs",
        )
        for actual_layer, expected_layer in zip(
            actual["layers"],
            expected["layers"],
            strict=True,
        ):
            require(
                actual_layer["layer_id"] == expected_layer["layer_id"],
                "independent layer order differs",
            )
            rtl = sealed_verifier.load_json(
                sealed_verifier.verify_file_record(expected_layer["rtl_execution"])
            )
            kv = rtl["kv_cache"]
            expected_k = exact_artifact_bytes(kv["rtl_observed_k"])
            expected_v = exact_artifact_bytes(kv["rtl_observed_v"])
            require(actual_layer["k_append"] == expected_k, "independent K append differs")
            require(actual_layer["v_append"] == expected_v, "independent V append differs")
            append_bytes += len(expected_k) + len(expected_v)

            layer_id = actual_layer["layer_id"]
            rtl_full_k_by_layer[layer_id].extend(expected_k)
            rtl_full_v_by_layer[layer_id].extend(expected_v)
            expected_full_k = bytes(rtl_full_k_by_layer[layer_id])
            expected_full_v = bytes(rtl_full_v_by_layer[layer_id])
            tensors = rtl["tensors"]
            full_k_record = tensors["full_k_cache_s8.bin"]
            full_v_record = tensors["full_v_cache_s8.bin"]
            require(
                len(expected_full_k) == full_k_record["bytes"]
                and sha256_bytes(expected_full_k) == full_k_record["sha256"]
                and sha256_bytes(expected_full_k) == kv["full_k_cache_sha256"],
                "reconstructed RTL full K cache differs from sealed metadata",
            )
            require(
                len(expected_full_v) == full_v_record["bytes"]
                and sha256_bytes(expected_full_v) == full_v_record["sha256"]
                and sha256_bytes(expected_full_v) == kv["full_v_cache_sha256"],
                "reconstructed RTL full V cache differs from sealed metadata",
            )
            require(
                actual_layer["full_k_cache"] == expected_full_k,
                "independent full K cache differs",
            )
            require(
                actual_layer["full_v_cache"] == expected_full_v,
                "independent full V cache differs",
            )
            full_cache_bytes += len(expected_full_k) + len(expected_full_v)

            output_record = tensors["layer_output_s8.bin"]
            scale_record = tensors["layer_output_scale_f64le.bin"]
            layer_output_bytes += require_payload_matches_record(
                actual_layer["layer_output"],
                output_record,
                "independent quantized layer output",
            )
            layer_output_scale_bytes += require_payload_matches_record(
                actual_layer["layer_output_scale"],
                scale_record,
                "independent layer output scale",
            )
            require(
                output_record["sha256"]
                == rtl["final_residual"]["output_sha256"],
                "sealed layer output record does not bind the RTL observation",
            )
            actual_rank1 = actual_layer["rank1"]
            expected_rank1 = rtl.get("rank1_sidecar")
            require(
                (actual_rank1 is None) == (expected_rank1 is None),
                "rank-1 execution coverage differs",
            )
            if actual_rank1 is not None:
                for key in (
                    "baseline_v_sha256",
                    "corrected_v_sha256",
                    "rank_accumulator_s32",
                    "rank_rounded_s32",
                    "rank_intermediate_s8",
                ):
                    require(
                        actual_rank1[key] == expected_rank1[key],
                        f"independent rank-1 field differs: {key}",
                    )
                rank1_checks += 1

    require(
        len(reference["heads"]) == len(summary["head_steps"]),
        "independent head count differs",
    )
    logit_bytes = 0
    logit_scale_bytes = 0
    final_rmsnorm_bytes = 0
    final_rmsnorm_scale_bytes = 0
    for actual, expected in zip(
        reference["heads"],
        summary["head_steps"],
        strict=True,
    ):
        require(
            actual["generation_index"] == expected["generation_index"]
            and actual["source_absolute_position"]
            == expected["source_absolute_position"],
            "independent head position differs",
        )
        head = sealed_verifier.load_json(
            sealed_verifier.verify_file_record(expected["head_execution"])
        )
        final_norm = head["final_rmsnorm"]
        lm_head = head["lm_head"]
        final_rmsnorm_bytes += require_payload_matches_record(
            actual["final_rmsnorm"],
            lm_head["artifacts"]["final_rmsnorm_s8"],
            "independent final RMSNorm",
        )
        final_rmsnorm_scale_bytes += require_payload_matches_record(
            actual["final_rmsnorm_scale"],
            lm_head["artifacts"]["final_rmsnorm_scale_f64le"],
            "independent final RMSNorm scale",
        )
        require(
            sha256_bytes(actual["final_rmsnorm"])
            == final_norm["output_sha256"],
            "independent final RMSNorm hash differs",
        )
        logits_record = lm_head["artifacts"]["lm_head_output_s8"]
        logit_bytes += require_payload_matches_record(
            actual["logits"],
            logits_record,
            "independent full logits",
        )
        logit_scale_bytes += require_payload_matches_record(
            actual["logit_scale"],
            lm_head["artifacts"]["lm_head_output_scale_f64le"],
            "independent logit scale",
        )
        require(
            logits_record["sha256"] == lm_head["output_sha256"],
            "sealed logit record does not bind the RTL observation",
        )
        require(
            actual["selected_token_id"] == expected["selected_token_id"]
            and actual["selected_token_id"] == lm_head["top_token"],
            "independent greedy token differs",
        )
        require(
            actual["selected_logit_s8"] == expected["selected_logit_s8"]
            and actual["selected_logit_s8"] == lm_head["top_logit_s8"],
            "independent selected logit differs",
        )
        require(
            actual["saturation_count"] == lm_head["saturation_count"]
            and actual["accumulator_min"] == lm_head["accumulator_min"]
            and actual["accumulator_max"] == lm_head["accumulator_max"],
            "independent LM-head numeric summary differs",
        )
    return {
        "prompt_token_ids_sha256": sha256_bytes(
            canonical_bytes(reference["prompt_token_ids"])
        ),
        "generated_token_ids": reference["generated_token_ids"],
        "decoded_text": reference["decoded_text"],
        "positions_checked": len(reference["positions"]),
        "layers_checked": len(reference["positions"]) * EXPECTED_LAYERS,
        "kv_append_bytes_compared": append_bytes,
        "full_cache_bytes_compared": full_cache_bytes,
        "quantized_layer_output_bytes_compared": layer_output_bytes,
        "quantized_layer_output_scale_bytes_compared": layer_output_scale_bytes,
        "rank1_positions_checked": rank1_checks,
        "final_rmsnorm_bytes_compared": final_rmsnorm_bytes,
        "final_rmsnorm_scale_bytes_compared": final_rmsnorm_scale_bytes,
        "full_vocabulary_head_steps_checked": len(reference["heads"]),
        "full_vocabulary_logit_bytes_compared": logit_bytes,
        "full_vocabulary_logit_scale_bytes_compared": logit_scale_bytes,
        "retained_payload_records_compared": (
            len(reference["positions"]) * EXPECTED_LAYERS * 2
            + len(reference["heads"]) * 4
        ),
        "retained_payload_bytes_compared": (
            layer_output_bytes
            + layer_output_scale_bytes
            + final_rmsnorm_bytes
            + final_rmsnorm_scale_bytes
            + logit_bytes
            + logit_scale_bytes
        ),
        "integer_byte_mismatches": 0,
        "selected_token_mismatches": 0,
    }


def write_report(output: Path, result: dict[str, Any]) -> None:
    agreement = result["agreement"]
    host = result["timing"]["independent_oracle_host"]
    rtl = result["timing"]["reused_rtl_attempt"]
    decoded = agreement["decoded_text"].replace("`", "\\`")
    generated_token_count = validate_expected_generated_tokens(
        result["constraints"]["max_new_tokens"]
    )
    generated_token_label = {
        4: "four",
        8: "eight",
    }.get(generated_token_count, str(generated_token_count))
    lines = [
        "# Full-chain independent oracle / RTL chat comparison",
        "",
        f"**Status:** {result['status']}",
        "",
        "A host-only W4A8 oracle independently rebuilt tokenization, all persistent "
        "K/V state, quantized layer outputs, full-vocabulary logits, and greedy "
        f"selection for the sealed {generated_token_label}-token run. It selected `{agreement['generated_token_ids']}` "
        f"and decoded `{decoded}`, exactly matching the retained computer-local RTL simulation.",
        "",
        "## Reproduction",
        "",
        "```bash",
        result["reproduction_command"],
        "```",
        "",
        "The command reuses the hash-verified sealed RTL attempt and does not replay or mutate it.",
        "",
        "## Admission and telemetry",
        "",
        f"- Admission mode: `{result['admission']['mode']}`",
        f"- Original attempt status: `{result['admission']['attempt_status']}`",
        f"- Numerical status: `{result['numerical_status']}`",
        f"- Historical process-tree RSS status: `{result['telemetry_status']}`",
        f"- Source compatibility: `{result['source_binding_compatibility']['status']}`",
        "",
        "A fresh representative `iverilog`/`vvp` child run validates the repaired "
        "terminal-accounting path. It is not substituted for missing historical case RSS.",
        "",
        "## Exact reused inputs and manifests",
        "",
        f"- Attempt tree root: `{result['attempt']['tree_root_sha256']}`",
        f"- Attempt manifest: `{result['manifests']['attempt_manifest']['sha256']}`",
        f"- RTL run summary: `{result['manifests']['run_summary']['sha256']}`",
        f"- Prompt: `{result['reused_inputs']['prompt']['sha256']}`",
        f"- Model: `{result['reused_inputs']['model']['sha256']}`",
        f"- Model config: `{result['reused_inputs']['model_config']['sha256']}`",
        f"- Adapter: `{result['reused_inputs']['adapter']['sha256']}`",
        f"- Tokenizer sources: `{result['reused_inputs']['tokenizer']['source_files_sha256']}`",
        f"- Source/RTL aggregate: `{result['source_binding']['sha256']}`",
        f"- Independent oracle implementation: `{result['oracle_implementation']['aggregate_sha256']}`",
        "",
        "## Agreement",
        "",
        f"- Positions / layers: {agreement['positions_checked']} / {agreement['layers_checked']}",
        f"- Exact K/V append bytes: {agreement['kv_append_bytes_compared']}",
        f"- Exact persistent-cache prefix bytes: {agreement['full_cache_bytes_compared']}",
        "- Every RTL cache prefix was reconstructed from retained, hash-verified "
        "append artifacts and matched both sealed full-cache hashes.",
        f"- Byte-equal quantized layer-output / scale bytes: {agreement['quantized_layer_output_bytes_compared']} / {agreement['quantized_layer_output_scale_bytes_compared']}",
        f"- Software rank-1 correction positions: {agreement['rank1_positions_checked']}",
        f"- Byte-equal final RMSNorm / scale bytes: {agreement['final_rmsnorm_bytes_compared']} / {agreement['final_rmsnorm_scale_bytes_compared']}",
        f"- Full-vocabulary head steps / byte-equal logit / scale bytes: {agreement['full_vocabulary_head_steps_checked']} / {agreement['full_vocabulary_logit_bytes_compared']} / {agreement['full_vocabulary_logit_scale_bytes_compared']}",
        f"- Direct retained payload records / bytes compared: {agreement['retained_payload_records_compared']} / {agreement['retained_payload_bytes_compared']}",
        "- Layer-output, RMSNorm, scale, and full-vocabulary logit payloads were "
        "read from authenticated retained artifacts and compared byte-for-byte.",
        "- Integer-byte mismatches: 0",
        "- Selected-token mismatches: 0",
        "",
        "## Whole-chain independence erratum",
        "",
        f"Closed for this sealed {generated_token_label}-token trace. The new oracle derives the complete "
        "sequence before reading any RTL result values. Its K/V caches are appended only "
        "from host fixed-point results; layer 23 uses the frozen rank-1 integer reference "
        "rather than the RTL sidecar. RTL artifacts are comparison-only inputs after "
        "reference completion.",
        "",
        "The oracle shares the project's accepted fixed-point primitive definitions and "
        "model/adapter inputs with vector generation. This result establishes independent "
        "whole-sequence state and control flow, not an implementation-diverse second "
        "specification of every arithmetic primitive.",
        "",
        "## Simulation/host-only timing",
        "",
        f"- Reused RTL-attempt total wall: {rtl['total_wall_seconds']:.6f} s",
        f"- Reused Icarus compile: {rtl['compile_wall_seconds']:.6f} s",
        f"- Reused Icarus simulation: {rtl['simulation_wall_seconds']:.6f} s",
        f"- Reused generation host/orchestration: {rtl['model_wall_seconds']:.6f} s",
        f"- Fresh independent-oracle host wall: {host['total_host_wall_seconds']:.6f} s",
        f"- Fresh independent-oracle process CPU: {host['host_process_cpu_seconds']:.6f} s",
        "",
        "The fresh command launched no simulator. RTL timings are retained measurements "
        "from the hash-verified computer-local Icarus attempt; the independent oracle "
        "timing is host-only. The separate representative telemetry reproduction launches "
        "only `iverilog -V` and `vvp -V`.",
        "",
        "## Limitations",
        "",
        f"- This closes the causal RTL-feedback erratum only for the exact checked prompt and {generated_token_label}-token greedy continuation.",
        "- Readable tokenization and exact numerical agreement do not establish semantic dialogue quality.",
        "- No FPGA, synthesis, STA, PPA, bitstream, deployed-hardware, or silicon claim is made.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    attempt: Path,
    output: Path,
    *,
    recover_failed_attempt: bool = False,
    expected_generated_tokens: int = sealed_verifier.REQUIRED_GENERATED_TOKENS,
) -> dict[str, Any]:
    expected_generated_tokens = validate_expected_generated_tokens(
        expected_generated_tokens
    )
    attempt = attempt.resolve()
    output = output.resolve()
    verification_root = ROOT / "reports" / "verification"
    require(
        attempt.is_relative_to(verification_root),
        "--attempt must be under reports/verification",
    )
    require(
        output.is_relative_to(verification_root),
        "--output must be under reports/verification",
    )
    require(not output.exists(), f"output already exists: {output}")
    require(
        not recover_failed_attempt
        or expected_generated_tokens == sealed_verifier.REQUIRED_GENERATED_TOKENS,
        "failed-attempt recovery supports only the sealed four-token contract",
    )

    seal = sealed_verifier.verify_seal(attempt)
    manifest = sealed_verifier.load_json(attempt / "attempt-manifest.json")
    require(
        validate_expected_generated_tokens(
            manifest["tokenization"]["generation_bounds"]["max_new_tokens"]
        )
        == expected_generated_tokens,
        "attempt generated-token contract differs from the requested oracle contract",
    )
    summary_path = attempt / "runtime-output/run_summary.json"
    summary = sealed_verifier.load_json(summary_path)
    attempt_result = sealed_verifier.load_json(attempt / "attempt-result.json")
    admission = verify_attempt_admission(
        attempt,
        attempt_result,
        summary,
        recover_failed_attempt=recover_failed_attempt,
    )
    source_binding, source_compatibility = verify_source_binding(
        manifest,
        allow_rss_tracker_repair=recover_failed_attempt,
    )
    reused_inputs = sealed_verifier.verify_inputs(manifest, attempt)
    retained_payload_preflight = verify_retained_comparison_payloads(summary)

    reference = reconstruct(manifest, attempt)

    comparison = sealed_verifier.load_json(
        attempt / "independent-reference-comparison.json"
    )
    decode = sealed_verifier.verify_decode(manifest, attempt, summary)
    prior_oracle = sealed_verifier.verify_oracle_surfaces(
        summary,
        comparison,
        required_generated_tokens=expected_generated_tokens,
        min_decode_transitions=expected_generated_tokens - 1,
    )
    agreement = compare_reference(reference, summary)
    require(
        decode["generated_token_ids"] == agreement["generated_token_ids"],
        "tokenizer verification and full-chain oracle differ",
    )
    require(
        reference["rank1_application_count"]
        == agreement["rank1_positions_checked"],
        "software rank-1 application count differs",
    )

    timing, timing_source = retained_backend_timing(
        sealed_verifier.load_json(attempt / "timing.json"),
        summary,
        recover_failed_attempt=recover_failed_attempt,
    )
    rss_reproduction = (
        run_representative_rss_reproduction()
        if recover_failed_attempt
        else None
    )
    implementation_paths = [
        Path(__file__),
        ROOT / "scripts/verify_persistent_kv_multitoken_rtl_chat.py",
        ROOT / "tools/rtl_arbitrary_text_generation_backend.py",
        ROOT / "tools/ace2_layer23_v_rank1_integer_correction_reference.py",
    ]
    implementation_records = [file_record(path) for path in implementation_paths]
    implementation_aggregate = hashlib.sha256(
        b"".join(
            record["path"].encode("utf-8")
            + b"\0"
            + record["sha256"].encode("ascii")
            + b"\n"
            for record in implementation_records
        )
    ).hexdigest()
    result = {
        "schema": (
            "ace2-full-chain-independent-oracle-failed-attempt-recovery-v1"
            if recover_failed_attempt
            else "ace2-full-chain-independent-oracle-rtl-chat-v1"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_NUMERICAL_RECOVERY" if recover_failed_attempt else "PASS",
        "numerical_status": "PASS",
        "telemetry_status": admission["telemetry_status"],
        "reproduction_command": (
            (
                "python3 -B scripts/verify_full_chain_independent_oracle.py "
                f"--attempt {attempt.relative_to(ROOT).as_posix()} "
                f"--output {output.relative_to(ROOT).as_posix()} "
                "--recover-failed-attempt"
            )
            if recover_failed_attempt
            else (
                "make persistent-kv-multitoken-full-chain-oracle "
                f"ACE2_CHAT_MULTITOKEN_OUTPUT={attempt.relative_to(ROOT).as_posix()} "
                "ACE2_CHAT_FULL_CHAIN_ORACLE_OUTPUT="
                f"{output.relative_to(ROOT).as_posix()}"
            )
            if expected_generated_tokens == sealed_verifier.REQUIRED_GENERATED_TOKENS
            else (
                "python3 -B scripts/verify_full_chain_independent_oracle.py "
                f"--attempt {attempt.relative_to(ROOT).as_posix()} "
                f"--output {output.relative_to(ROOT).as_posix()} "
                f"--expected-generated-tokens {expected_generated_tokens}"
            )
        ),
        "admission": admission,
        "attempt": {
            "path": attempt.relative_to(ROOT).as_posix(),
            **seal,
        },
        "manifests": {
            "attempt_manifest": file_record(attempt / "attempt-manifest.json"),
            "sha256sums": file_record(attempt / "SHA256SUMS"),
            "run_summary": file_record(summary_path),
            "prior_comparison": file_record(
                attempt / "independent-reference-comparison.json"
            ),
        },
        "source_binding": source_binding,
        "source_binding_compatibility": source_compatibility,
        "oracle_implementation": {
            "aggregate_sha256": implementation_aggregate,
            "files": implementation_records,
            "independence": {
                "reference_completed_before_rtl_value_comparison": True,
                "host_kv_cache_consumed_rtl_values": False,
                "rtl_cache_comparison": (
                    "hash-verified RTL append artifacts concatenated and checked "
                    "against sealed full-cache metadata"
                ),
                "retained_rtl_payload_comparison": (
                    "independently reconstructed payloads compared byte-for-byte "
                    "against authenticated retained execution artifacts"
                ),
                "retained_payload_preflight": retained_payload_preflight,
                "rank1_correction_source": "frozen_host_integer_reference",
                "sampling": "deterministic_full_domain_greedy_argmax",
            },
        },
        "reused_inputs": reused_inputs,
        "tokenization": {
            "prompt_token_count": len(reference["prompt_token_ids"]),
            "prompt_token_ids_sha256": agreement["prompt_token_ids_sha256"],
            "generated_token_ids": agreement["generated_token_ids"],
            "decoded_text": agreement["decoded_text"],
        },
        "agreement": agreement,
        "prior_per_position_verification": {
            "positions_checked": prior_oracle["positions_checked"],
            "rtl_layer_records_checked": prior_oracle[
                "rtl_layer_records_checked"
            ],
            "integer_mismatch_totals": prior_oracle["integer_mismatch_totals"],
        },
        "erratum_resolution": {
            "status": (
                "CLOSED_FOR_EXACT_SEALED_"
                f"{expected_generated_tokens}_TOKEN_TRACE"
            ),
            "closed_issue": (
                "later positions no longer depend on RTL-observed K/V in the "
                "comparison oracle"
            ),
            "remaining_scope": (
                "accepted fixed-point primitive implementations are shared; "
                "implementation-diverse arithmetic is not claimed"
            ),
        },
        "timing": {
            "reused_rtl_attempt": timing,
            "reused_rtl_attempt_source": timing_source,
            "independent_oracle_host": reference["timing"],
            "classification": (
                "reused computer-local Icarus measurements plus fresh host-only "
                "oracle measurements; no hardware timing"
            ),
        },
        "representative_rss_reproduction": rss_reproduction,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "constraints": {
            "max_new_tokens": expected_generated_tokens,
            "rtl_context_bound": backend.MAX_CONTEXT_TOKENS,
            "model_output_domain": EXPECTED_MODEL_OUTPUTS,
            "hardware_flow": "NOT_RUN_OPERATOR_CANCELLED",
        },
        "limitations": [
            (
                "Exact checked prompt and "
                f"{expected_generated_tokens}-token greedy continuation only."
            ),
            "Shared accepted fixed-point primitives are not an implementation-diverse arithmetic specification.",
            "No semantic dialogue-quality claim.",
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


def historical_case_rss(attempt: Path) -> dict[str, Any]:
    for name in ("attempt-result.json", "worker-result.json", "attempt-manifest.json"):
        path = attempt / name
        if not path.is_file():
            continue
        record = sealed_verifier.load_json(path).get("process_tree_rss")
        if isinstance(record, dict) and record.get("complete") is True:
            return {
                "status": "COMPLETE",
                "source": file_record(path),
                "process_tree_rss": record,
            }
    return {
        "status": "UNRESOLVED_HISTORICAL_PROCESS_TREE_RSS",
        "source": None,
        "process_tree_rss": None,
    }


def validate_reviewed_reuse_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    attempt = sealed_verifier.project_path(case["attempt"]["path"]).resolve()
    oracle_output = sealed_verifier.project_path(case["oracle_output"]["path"]).resolve()
    seal = sealed_verifier.verify_seal(attempt)
    require(
        seal["tree_root_sha256"] == case["attempt"]["tree_root_sha256"],
        f"case {case['id']} attempt tree root differs",
    )
    oracle_checksums = verify_checksum_manifest(oracle_output)
    oracle_result = sealed_verifier.load_json(oracle_output / "result.json")
    require(oracle_result.get("status") == "PASS", f"case {case['id']} oracle is not PASS")
    require(
        sha256_file(oracle_output / "result.json")
        == case["oracle_output"]["result"]["sha256"],
        f"case {case['id']} oracle result differs",
    )
    manifest = sealed_verifier.load_json(attempt / "attempt-manifest.json")
    source_binding, compatibility = verify_source_binding(
        manifest,
        allow_rss_tracker_repair=True,
    )
    inputs = sealed_verifier.verify_inputs(manifest, attempt)
    require(
        source_binding["sha256"] == case["source_binding"]["sha256"],
        f"case {case['id']} source binding differs",
    )
    require(
        inputs["model"]["sha256"] == case["reused_inputs"]["model"]["sha256"]
        and inputs["adapter"]["sha256"] == case["reused_inputs"]["adapter"]["sha256"]
        and inputs["tokenizer"]["source_files_sha256"]
        == case["reused_inputs"]["tokenizer"]["source_files_sha256"],
        f"case {case['id']} model, adapter, or tokenizer binding differs",
    )
    agreement = oracle_result["agreement"]
    require(
        len(agreement["generated_token_ids"]) == 4
        and agreement["integer_byte_mismatches"] == 0
        and agreement["selected_token_mismatches"] == 0
        and agreement["full_vocabulary_head_steps_checked"] == 4,
        f"case {case['id']} reviewed numerical agreement is incomplete",
    )
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "origin": "reviewed_reuse_authenticated",
        "attempt": case["attempt"],
        "oracle_output": {
            **case["oracle_output"],
            "checksums": oracle_checksums,
        },
        "source_binding": source_binding,
        "source_binding_compatibility": compatibility,
        "reused_inputs": inputs,
        "tokenization": oracle_result["tokenization"],
        "agreement": agreement,
        "timing": oracle_result["timing"],
        "process_tree_rss": historical_case_rss(attempt),
    }


def write_recovery_suite_report(output: Path, result: dict[str, Any]) -> None:
    fresh_kv_case = result["suite_mode"] == "fresh_passed_kv_attempt"
    lines = [
        (
            "# Deterministic three-prompt RTL/oracle regression"
            if fresh_kv_case
            else "# Deterministic three-prompt RTL/oracle recovery"
        ),
        "",
        f"**Status:** {result['status']}",
        f"**Numerical status:** {result['numerical_status']}",
        f"**Strict per-case RSS status:** {result['telemetry_status']}",
        "",
        (
            "The original failed suite and failed kv-cache attempt remain unchanged. "
            "Two reviewed cases are reused by authenticated references; the kv-cache "
            "case uses a fresh passing RTL attempt and a separate fresh host-only "
            "full-chain oracle."
            if fresh_kv_case
            else
            "The original suite and failed kv-cache attempt remain unchanged. Two reviewed "
            "cases are reused by authenticated references; the kv-cache case uses a fresh "
            "host-only full-chain oracle over its sealed retained RTL trace."
        ),
        "",
        "| Case | Origin | Generated IDs | Decoded text | Integer / token mismatches | RSS |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for case in result["cases"]:
        agreement = case["agreement"]
        lines.append(
            f"| `{case['id']}` | {case['origin']} | "
            f"`{agreement['generated_token_ids']}` | "
            f"`{agreement['decoded_text'].replace('`', '\\`')}` | "
            f"{agreement['integer_byte_mismatches']} / "
            f"{agreement['selected_token_mismatches']} | "
            f"{case['process_tree_rss']['status']} |"
        )
    coverage = result["coverage"]
    lines.extend(
        [
            "",
            "## Numerical coverage",
            "",
            f"- Prompt cases: {coverage['prompt_cases']}",
            f"- Generated tokens: {coverage['generated_tokens']}",
            f"- Persistent K/V bytes compared: {coverage['kv_append_bytes_compared']}",
            f"- Full-cache bytes compared: {coverage['full_cache_bytes_compared']}",
            f"- Full-vocabulary head steps: {coverage['full_vocabulary_head_steps_checked']}",
            f"- Full-vocabulary logit bytes: {coverage['full_vocabulary_logit_bytes_compared']}",
            f"- Integer-byte mismatches: {coverage['integer_byte_mismatches']}",
            f"- Selected-token mismatches: {coverage['selected_token_mismatches']}",
            "",
            "## Telemetry adjudication",
            "",
            (
                "Every referenced case has complete strict process-tree RSS telemetry; "
                "the fresh kv-cache attempt records all registered Icarus children as "
                "terminally accounted and reports sampling completeness separately."
                if result["telemetry_status"] == "PASS"
                else
                "The fresh representative child reproduction is PASS, but it does not "
                "reconstruct any missing historical case peak or sample. Therefore this "
                "recovery is not an all-requirements suite PASS."
            ),
            "",
            "## Simulation/host-only timing",
            "",
        ]
    )
    for case in result["cases"]:
        rtl = case["timing"]["reused_rtl_attempt"]
        host = case["timing"]["independent_oracle_host"]
        lines.append(
            f"- `{case['id']}`: retained RTL-simulation/host wall "
            f"{rtl['total_wall_seconds']:.6f} s; Icarus compile "
            f"{rtl['compile_wall_seconds']:.6f} s; Icarus simulation "
            f"{rtl['simulation_wall_seconds']:.6f} s; host/orchestration "
            f"{rtl['model_wall_seconds']:.6f} s; independent-oracle host "
            f"{host['total_host_wall_seconds']:.6f} s."
        )
    lines.extend(
        [
            "",
            "All timings are computer-local simulation or host measurements. No hardware, "
            "FPGA, synthesis, STA, PPA, bitstream, deployed-hardware, or silicon claim is made.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_recovery_suite(
    source_suite: Path,
    output: Path,
    recovered_output: Path,
    recovered_result: dict[str, Any],
) -> dict[str, Any]:
    source_suite = source_suite.resolve()
    output = output.resolve()
    verification_root = ROOT / "reports" / "verification"
    require(source_suite.is_relative_to(verification_root), "source suite is outside verification reports")
    require(output.is_relative_to(verification_root), "suite output is outside verification reports")
    require(not output.exists(), f"suite output already exists: {output}")
    source_checksums = verify_checksum_manifest(source_suite)
    source_result = sealed_verifier.load_json(source_suite / "result.json")
    source_config = sealed_verifier.load_json(source_suite / "suite-config.json")
    require(source_result.get("status") == "FAIL", "source suite is not the retained FAIL")
    require(len(source_result.get("cases", [])) == 2, "source suite does not contain two reviewed cases")
    cases = [
        validate_reviewed_reuse_case(case)
        for case in source_result["cases"]
    ]
    kv_config = next(
        (
            case
            for case in source_config["cases"]
            if case.get("id") == "kv-cache-question"
        ),
        None,
    )
    require(kv_config is not None, "source suite lacks kv-cache-question")
    require(
        recovered_result["status"] in {"PASS", "PASS_NUMERICAL_RECOVERY"},
        "kv-cache oracle result is not PASS",
    )
    fresh_kv_case = recovered_result["status"] == "PASS"
    require(
        recovered_result["reused_inputs"]["prompt"]["sha256"]
        == sha256_bytes(kv_config["prompt"].encode("utf-8")),
        "kv-cache prompt differs from suite config",
    )
    cases.append(
        {
            "id": kv_config["id"],
            "prompt": kv_config["prompt"],
            "origin": (
                "fresh_rtl_and_oracle"
                if fresh_kv_case
                else "sealed_failed_attempt_numerical_recovery"
            ),
            "attempt": recovered_result["attempt"],
            "oracle_output": {
                "path": recovered_output.resolve().relative_to(ROOT).as_posix(),
                "result": file_record(recovered_output.resolve() / "result.json"),
                "checksums": verify_checksum_manifest(recovered_output.resolve()),
            },
            "source_binding": recovered_result["source_binding"],
            "source_binding_compatibility": recovered_result[
                "source_binding_compatibility"
            ],
            "reused_inputs": recovered_result["reused_inputs"],
            "tokenization": recovered_result["tokenization"],
            "agreement": recovered_result["agreement"],
            "timing": recovered_result["timing"],
            "process_tree_rss": historical_case_rss(
                sealed_verifier.project_path(recovered_result["attempt"]["path"])
            ),
        }
    )
    unresolved = [
        case["id"]
        for case in cases
        if case["process_tree_rss"]["status"] != "COMPLETE"
    ]
    coverage_fields = (
        "kv_append_bytes_compared",
        "full_cache_bytes_compared",
        "full_vocabulary_head_steps_checked",
        "full_vocabulary_logit_bytes_compared",
        "integer_byte_mismatches",
        "selected_token_mismatches",
    )
    coverage = {
        "prompt_cases": len(cases),
        "generated_tokens": sum(
            len(case["agreement"]["generated_token_ids"]) for case in cases
        ),
        **{
            field: sum(int(case["agreement"][field]) for case in cases)
            for field in coverage_fields
        },
    }
    require(
        coverage["prompt_cases"] == 3
        and coverage["generated_tokens"] == 12
        and coverage["integer_byte_mismatches"] == 0
        and coverage["selected_token_mismatches"] == 0,
        "recovery suite numerical coverage is incomplete",
    )
    result = {
        "schema": (
            "ace2-prompt-suite-rtl-oracle-regression-v1"
            if fresh_kv_case
            else "ace2-prompt-suite-rtl-oracle-numerical-recovery-v1"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "PASS"
            if not unresolved
            else "NUMERICAL_PASS_TELEMETRY_UNRESOLVED"
        ),
        "numerical_status": "PASS",
        "telemetry_status": "PASS" if not unresolved else "UNRESOLVED",
        "suite_mode": (
            "fresh_passed_kv_attempt"
            if fresh_kv_case
            else "failed_attempt_numerical_recovery"
        ),
        "source_suite": {
            "path": source_suite.relative_to(ROOT).as_posix(),
            "checksums": source_checksums,
            "result": file_record(source_suite / "result.json"),
        },
        "cases": cases,
        "coverage": coverage,
        "telemetry": {
            "strict_case_rss_complete": not unresolved,
            "unresolved_case_ids": unresolved,
            "representative_reproduction": recovered_result[
                "representative_rss_reproduction"
            ],
            "representative_reproduction_is_not_case_telemetry": True,
        },
        "bindings": {
            "source_and_rtl_sha256": recovered_result["source_binding"]["sha256"],
            "model_sha256": recovered_result["reused_inputs"]["model"]["sha256"],
            "adapter_sha256": recovered_result["reused_inputs"]["adapter"]["sha256"],
            "tokenizer_source_sha256": recovered_result["reused_inputs"]["tokenizer"][
                "source_files_sha256"
            ],
        },
        "measurement_scope": "computer-local RTL simulation and host only",
    }
    output.mkdir(parents=True)
    (output / "suite-config.json").write_bytes(canonical_bytes(source_config))
    (output / "result.json").write_bytes(canonical_bytes(result))
    write_recovery_suite_report(output, result)
    members = [output / "result.json", output / "REPORT.md", output / "suite-config.json"]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-failed-attempt", action="store_true")
    parser.add_argument(
        "--expected-generated-tokens",
        type=int,
        default=sealed_verifier.REQUIRED_GENERATED_TOKENS,
    )
    parser.add_argument("--recovery-suite-source", type=Path)
    parser.add_argument("--recovery-suite-output", type=Path)
    args = parser.parse_args()
    require(
        (args.recovery_suite_source is None)
        == (args.recovery_suite_output is None),
        "recovery suite source and output must be provided together",
    )
    result = run(
        args.attempt,
        args.output,
        recover_failed_attempt=args.recover_failed_attempt,
        expected_generated_tokens=args.expected_generated_tokens,
    )
    suite_result = None
    if args.recovery_suite_source is not None:
        suite_result = build_recovery_suite(
            args.recovery_suite_source,
            args.recovery_suite_output,
            args.output,
            result,
        )
    print(
        "FULL_CHAIN_INDEPENDENT_ORACLE_NUMERICAL_PASS "
        f"tokens={len(result['agreement']['generated_token_ids'])} "
        f"positions={result['agreement']['positions_checked']} "
        f"output={args.output} "
        f"suite_status={suite_result['status'] if suite_result else 'NOT_REQUESTED'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"FULL_CHAIN_INDEPENDENT_ORACLE_FAIL detail={error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
