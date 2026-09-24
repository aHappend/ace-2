#!/usr/bin/env python3
"""Search bounded dual-Scale32 layer-16 residual-sum output contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_checkpoint176_layer16_down_proj_accumulator_grouped_scale32_search as accumulator
from tools import ace2_checkpoint176_layer16_down_proj_output_grouped_scale32_search as downoutput
from tools import ace2_checkpoint176_layer16_source_grouped_scale32_search as layer16
from tools import ace2_checkpoint176_layer17_q_output_grouped_scale32_search as qoutput
from tools import ace2_checkpoint176_layer17_rmsnorm_input_grouped_scale32_search as rmsinput
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools.ace2_quality_contracts import scale32_ratio
from tools.ace2_rmsnorm_reference import derive_scaled_gains_q8


MISSION = "w4a8-layer16-residual-sum-dual-scale-repair-v1"
BOUNDARY = "model.layers.16.mlp.down_proj.output_to_residual_sum"
NEXT_BOUNDARY = "model.layers.16.self_attn.output_to_post_attention_residual_sum.scale_alignment"
ATTENTION_SOURCE_GROUP_SIZE = 1
DOWN_OUTPUT_GROUP_SIZE = 4
Q_OUTPUT_GROUP_SIZE = 128
OUTPUT_GROUP_SIZES = (896, 32, 16, 8, 4, 1)

ACCEPTED_ACCUMULATOR = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-accumulator-requantization-repair-v1/"
    "candidate-0002"
)
LOCALIZATION = (
    ROOT
    / "evidence/candidates/w4a8-layer16-down-proj-group4-remaining-error-localization-v1/"
    "candidate-0001"
)
LOCALIZATION_REVIEW = ROOT / "build/l2-review-layer16-down-proj-group4-localization-r1/result.json"
LOCALIZATION_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-layer16-down-proj-group4-remaining-error-localization-v1/round-0001.json"
)
RTL = ROOT / "rtl/ace2_layer16_residual_sum_dual_scale_core.sv"
RTL_TOP = "ace2_layer16_residual_sum_dual_scale_core"
PROTECTED_HASHES = dict(downoutput.PROTECTED_HASHES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _scale32_float(record: int) -> float:
    numerator, denominator = scale32_ratio(record)
    return numerator / denominator


def _verify_manifest(directory: Path) -> int:
    completed = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=directory,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    localizer.require(completed.returncode == 0, f"manifest verification failed: {directory}")
    return len([line for line in completed.stdout.splitlines() if line.endswith(": OK")])


def _rtl_preflight() -> dict[str, Any]:
    results = {}
    with tempfile.TemporaryDirectory(prefix="ace2-layer16-residual-sum-", dir=ROOT / "build") as temporary:
        executable = Path(temporary) / "preflight.vvp"
        commands = {
            "iverilog": [
                "iverilog", "-g2012", "-Wall", "-Wno-timescale", "-s", RTL_TOP,
                "-o", str(executable), str(RTL),
            ],
            "verilator": [
                "verilator", "--lint-only", "-Wall", "-Wno-fatal",
                "--top-module", RTL_TOP, str(RTL),
            ],
        }
        for name, command in commands.items():
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            localizer.require(
                completed.returncode == 0,
                f"{name} RTL preflight failed: {completed.stdout}",
            )
            localizer.require(not completed.stdout.strip(), f"{name} RTL preflight warned")
            results[name] = {"command": command, "returncode": completed.returncode}
    return {"schema_version": 1, "top": RTL_TOP, "rtl": localizer.file_record(RTL), "checks": results}


def _tool_versions() -> dict[str, str]:
    commands = {
        "python": [sys.executable, "--version"],
        "iverilog": ["iverilog", "-V"],
        "verilator": ["verilator", "--version"],
    }
    versions = {"torch": torch.__version__}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        localizer.require(completed.returncode == 0, f"tool version failed: {name}")
        versions[name] = completed.stdout.strip().splitlines()[0]
    return versions


class ResidualSumSourceGuard(accumulator.DownProjAccumulatorSourceGuard):
    """Replace only the residual-add alignment and sum-to-A8 requantization."""

    def __init__(
        self,
        cache: accumulator.GroupedDownProjAccumulatorScale32Cache,
        weights: Any,
        output_group_size: int,
    ) -> None:
        super().__init__(cache, weights, DOWN_OUTPUT_GROUP_SIZE)
        self.output_group_size = output_group_size

    def _replace_sum(self, position: int) -> None:
        record = self.records[position]
        residual_scales = tuple(
            int(value) for value in record["attention_group_scale32"].tolist()
        )
        down_scales = tuple(
            int(value) for value in record["down_output_group_scale32"].tolist()
        )
        values = downoutput._sum_reals(
            record["attention_group_s8"],
            record["down_output_s8"],
            residual_scales,
            down_scales,
            DOWN_OUTPUT_GROUP_SIZE,
        )
        output_scales = rmsinput._group_scales(values, self.output_group_size)
        sum_s8, saturation = rmsinput._quantize_groups(
            values, output_scales, self.output_group_size
        )
        record.update(
            {
                "residual_group_s8": record["attention_group_s8"],
                "residual_group_scale32": record["attention_group_scale32"],
                "sum_output_group_size": self.output_group_size,
                "sum_output_group_scale32": torch.tensor(output_scales, dtype=torch.int64),
                "sum_s8": sum_s8,
                "sum_saturation": saturation,
            }
        )
        if self.output_group_size == backend.HIDDEN:
            localizer.require(
                output_scales == (int(record["sum_scale32"]),),
                "control output Scale32 differs from the accepted tensor-wide sum scale",
            )
            return

        q_input_scale32 = int(record["q_input_scale32"])
        gain = self.weights.get_tensor(
            f"model.layers.{layer16.TARGET_LAYER}.input_layernorm.weight"
        ).contiguous()
        gains = derive_scaled_gains_q8(
            gain.to(torch.float64).tolist(), _scale32_float(q_input_scale32)
        )
        rms = rmsinput._grouped_rmsnorm(
            sum_s8, output_scales, self.output_group_size, gains
        )
        record.update(
            {
                "rms_input_aligned_s64": rms["aligned"],
                "rms_input_common_exponent": rms["common_exponent"],
                "rms_output_s8": rms["outputs"],
                "rms_output_saturation": rms["saturation"],
                "rms_sumsq": rms["sumsq"],
                "rms_inv_q30": rms["inv_rms_q30"],
                "rms_saturation": bool(rms["saturation"].any()),
            }
        )

    def install(self) -> None:
        super().install()
        derive_with_group4 = backend.derive_layer_token

        def derive(
            layer_id: int,
            state: dict[str, Any],
            cache: dict[str, Any],
            template: dict[str, Any] | None,
            weights: Any,
            adapter: Any,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            result = derive_with_group4(layer_id, state, cache, template, weights, adapter)
            if layer_id == layer16.SOURCE_LAYER:
                self._replace_sum(int(state["position"]))
            return result

        backend.derive_layer_token = derive


def _artifacts(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    artifacts = {
        name: value for name, value in record.items() if isinstance(value, torch.Tensor)
    }
    for name in (
        "attention_base_scale32",
        "down_input_scale32",
        "down_tensor_output_scale32",
        "q_input_scale32",
        "q_tensor_output_scale32",
        "rms_sumsq",
        "rms_inv_q30",
    ):
        artifacts[name] = torch.tensor([int(record[name])], dtype=torch.int64)
    if "rms_input_common_exponent" in record:
        artifacts["rms_input_common_exponent"] = torch.tensor(
            [int(record["rms_input_common_exponent"])], dtype=torch.int64
        )
    artifacts["rms_saturation"] = torch.tensor(
        [int(record["rms_saturation"])], dtype=torch.uint8
    )
    return artifacts


def run_contract(
    output_group_size: int,
    cache: accumulator.GroupedDownProjAccumulatorScale32Cache,
    weights: Any,
    adapter: Any,
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cache.configure_down(DOWN_OUTPUT_GROUP_SIZE)
    source_guard = ResidualSumSourceGuard(cache, weights, output_group_size)
    score_guard = qoutput.QHeadScaleScoreGuard(cache)
    cache.install()
    score_guard.install()
    source_guard.install()
    try:
        result, _ = localizer.run_cut(
            None, [], weights, adapter, norm_gain, embedding, head, tokenizer
        )
    finally:
        source_guard.restore()
        score_guard.restore()
        cache.restore()

    expected_scores = backend.Q_HEADS * (2 * len(localizer.TOKEN_IDS) - 1)
    localizer.require(
        score_guard.replacements == expected_scores,
        "accepted layer-17 grouped Q scale did not reach every score",
    )
    exact = source_guard.records[len(localizer.TOKEN_IDS) - 1]
    output_scales = exact["sum_output_group_scale32"]
    q_scales = exact["q_output_group_scale32"]
    result.update(
        {
            "contract": "control" if output_group_size == 896 else f"group-{output_group_size}",
            "control_contract": output_group_size == 896,
            "attention_source_group_size": ATTENTION_SOURCE_GROUP_SIZE,
            "down_output_group_size": DOWN_OUTPUT_GROUP_SIZE,
            "sum_output_group_size": output_group_size,
            "sum_output_group_count": backend.HIDDEN // output_group_size,
            "sum_output_group_scale32_min": f"0x{int(output_scales.min()):08x}",
            "sum_output_group_scale32_max": f"0x{int(output_scales.max()):08x}",
            "residual_s8_sha256": localizer.tensor_sha256(exact["residual_group_s8"]),
            "residual_scale32_sha256": localizer.tensor_sha256(exact["residual_group_scale32"]),
            "down_s8_sha256": localizer.tensor_sha256(exact["down_output_s8"]),
            "down_scale32_sha256": localizer.tensor_sha256(exact["down_output_group_scale32"]),
            "sum_s8_sha256": localizer.tensor_sha256(exact["sum_s8"]),
            "sum_saturation_count": int(exact["sum_saturation"].sum()),
            "rms_output_s8_sha256": localizer.tensor_sha256(exact["rms_output_s8"]),
            "rms_saturation": bool(exact["rms_saturation"]),
            "q_output_group_size": Q_OUTPUT_GROUP_SIZE,
            "q_output_group_scale32_min": f"0x{int(q_scales.min()):08x}",
            "q_output_group_scale32_max": f"0x{int(q_scales.max()):08x}",
            "q_output_s8_sha256": localizer.tensor_sha256(exact["q_output_s8"]),
            "q_saturation_count": int(exact["q_saturation"].sum()),
            "layer17_score_scale_replacements": score_guard.replacements,
        }
    )
    return result, _artifacts(exact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    localizer.require(ROOT.resolve() in output.parents, "output must be repository-relative")
    localizer.require(not output.exists(), "candidate output already exists")

    for relative, expected in PROTECTED_HASHES.items():
        localizer.require(_sha256(ROOT / relative) == expected, f"protected hash differs: {relative}")

    accepted_result_path = ACCEPTED_ACCUMULATOR / "result.json"
    accepted_freeze_path = ACCEPTED_ACCUMULATOR / "candidate-freeze.json"
    accepted_result = json.loads(accepted_result_path.read_text(encoding="utf-8"))
    accepted_freeze = json.loads(accepted_freeze_path.read_text(encoding="utf-8"))
    localization_result_path = LOCALIZATION / "result.json"
    localization_result = json.loads(localization_result_path.read_text(encoding="utf-8"))
    localization_review = json.loads(LOCALIZATION_REVIEW.read_text(encoding="utf-8"))
    localization_handoff = json.loads(LOCALIZATION_HANDOFF.read_text(encoding="utf-8"))

    accepted_members = _verify_manifest(ACCEPTED_ACCUMULATOR)
    localization_members = _verify_manifest(LOCALIZATION)
    localizer.require(
        accepted_result["selected_down_output_group_size"] == DOWN_OUTPUT_GROUP_SIZE
        and accepted_result["selected_reference_rank"] == 31879
        and accepted_result["selected_top_token_id"] == 37865,
        "accepted group-4 predecessor differs",
    )
    for record, label in ((localization_result, "candidate"), (localization_review, "Fresh-L2")):
        localizer.require(
            record["control"]["reference_rank"] == 31879
            and record["control"]["top_token_id"] == 37865,
            f"{label} localization control differs",
        )
        localizer.require(
            record["confirmation"]["adjacent_repeat"]["reference_rank"] == 36580
            and record["confirmation"]["selected_repeat"]["reference_rank"] == 27814,
            f"{label} localization repeats differ",
        )
    localizer.require(
        localization_handoff["producer_role"] == "reviewer"
        and localization_handoff["review"]["status"] == "done"
        and "31,879" in localization_handoff["review"]["reason"]
        and "27,814" in localization_handoff["review"]["reason"],
        "sealed Fresh-L2 localization acceptance is absent",
    )
    localizer.require(
        accepted_freeze["baseline"]["layer16_attention_source_group_size"] == 1
        and accepted_freeze["baseline"]["layer17_q_output_group_size"] == 128,
        "accepted attention/Q dependencies differ",
    )

    output.mkdir(parents=True)
    freeze = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mission": MISSION,
        "classification": "candidate_only_no_official_attempt_created",
        "checkpoint": accepted_freeze["checkpoint"],
        "prompt": accepted_freeze["prompt"],
        "reference_step0_token_id": localizer.REFERENCE_TOKEN,
        "accepted_control": {
            "top_token_id": 37865,
            "reference_rank": 31879,
            "attention_source_group_size": 1,
            "down_output_group_size": 4,
            "q_output_group_size": 128,
        },
        "localization_gate": {
            "adjacent_requantized_output_rank": 36580,
            "residual_sum_diagnostic_rank": 27814,
            "candidate_and_fresh_l2_exact_repeats": True,
        },
        "ordered_output_group_sizes": list(OUTPUT_GROUP_SIZES),
        "score_policy": (
            "execute every frozen contract; control must exactly reproduce rank 31879/top 37865; "
            "select the first token-9707 restoration in order, otherwise the lowest non-control "
            "rank strictly below 31879, ties by declared order"
        ),
        "numeric_contract": {
            "boundary": BOUNDARY,
            "residual_operand": "accepted layer-16 attention/source group-1 signed A8 plus Scale32",
            "down_operand": "accepted layer-16 down_proj group-4 signed A8 plus Scale32",
            "alignment": "multiply each signed A8 by its independent Scale32 significand, shift to the minimum of both source and output exponents, then perform one widened signed add",
            "requantization": "one exact integer divide by output Scale32, ties-to-even, then signed-A8 saturation",
            "output_group_scale": "smallest legal Scale32 not below exact group max-abs/127",
            "accepted_w4_bytes_and_scales_unchanged": True,
            "bf16_runtime_sidecar": False,
            "wider_carried_activation_payload": False,
            "token_or_logit_override": False,
        },
        "rtl_public_contract": {
            "top": RTL_TOP,
            "latency": "ten arithmetic cycles for valid descriptors; ready/valid output holds under backpressure",
            "reset": "asynchronous active-low reset; synchronous clear",
            "ports": [
                "clk_i", "rst_ni", "clear_i", "in_valid_i", "in_ready_o",
                "residual_s8_i:s8", "residual_scale32_i:u32", "down_s8_i:s8",
                "down_scale32_i:u32", "output_scale32_i:u32", "channel_i:u10", "last_i",
                "out_valid_o", "out_ready_i", "sum_s8_o:s8", "output_scale32_o:u32",
                "channel_o:u10", "last_o", "saturation_o", "descriptor_error_o",
                "numeric_overflow_o", "common_exponent_s8_o:s8", "latency_cycles_u5_o:u5",
            ],
        },
        "tool_versions": _tool_versions(),
        "protected_hashes": PROTECTED_HASHES,
        "scope_guards": {
            "official_run_launched": False,
            "official_attempt_consumed": False,
            "protected_candidate_mutated": False,
            "u280_or_stage2_entered": False,
        },
        "bindings": {
            "accepted_accumulator_result": localizer.file_record(accepted_result_path),
            "accepted_accumulator_freeze": localizer.file_record(accepted_freeze_path),
            "accepted_accumulator_sha256s": localizer.file_record(ACCEPTED_ACCUMULATOR / "SHA256SUMS"),
            "accepted_accumulator_manifest_members_verified": accepted_members,
            "localization_result": localizer.file_record(localization_result_path),
            "localization_sha256s": localizer.file_record(LOCALIZATION / "SHA256SUMS"),
            "localization_manifest_members_verified": localization_members,
            "fresh_l2_localization": localizer.file_record(LOCALIZATION_REVIEW),
            "fresh_l2_handoff": _external_record(LOCALIZATION_HANDOFF),
            "runner": localizer.file_record(Path(__file__).resolve()),
            "rtl": localizer.file_record(RTL),
        },
    }
    localizer.write_json(output / "candidate-freeze.json", freeze)
    localizer.write_json(output / "rtl-interface-preflight.json", _rtl_preflight())

    started = time.monotonic()
    snapshot = generation_runner.resolve_snapshot()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    localizer.require(
        generation_runner.canonical_chat_token_ids(tokenizer, "Hello") == localizer.TOKEN_IDS,
        "canonical tokenizer prefix differs",
    )

    trace = []
    failures = []
    with safe_open(backend.canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        backend.canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        head = localizer.derive_lm_head_weights(embedding)
        cache = accumulator.GroupedDownProjAccumulatorScale32Cache(weights, adapter)
        for group_size in OUTPUT_GROUP_SIZES:
            group_dir = output / f"contracts/group-{group_size:03d}"
            try:
                record, tensors = run_contract(
                    group_size, cache, weights, adapter, norm_gain, embedding, head, tokenizer
                )
                if group_size == 896:
                    localizer.require(
                        record["top_token_id"] == 37865 and record["reference_rank"] == 31879,
                        "exact control outcome differs",
                    )
            except Exception as error:
                failure = {
                    "sum_output_group_size": group_size,
                    "classification": "CANDIDATE_EXECUTION_FAILURE",
                    "detail": str(error),
                    "root_cause_hypothesis": "The frozen dual-scale output-group contract could not be consumed by the unchanged downstream evaluator.",
                    "regression": "Preserve this failed contract and continue the remaining frozen execution order without changing the evaluator or expected values.",
                }
                localizer.write_json(group_dir / "failure.json", failure)
                failures.append(failure)
                print(f"ACE2_LAYER16_RESIDUAL_SUM_CANDIDATE_FAIL group={group_size} detail={error}", flush=True)
                if group_size == backend.HIDDEN:
                    raise
                continue
            artifacts = {
                name: localizer.write_tensor(group_dir / f"{name}.bin", value)
                for name, value in sorted(tensors.items())
            }
            localizer.write_json(group_dir / "result.json", {"metrics": record, "artifacts": artifacts})
            trace.append(record)
            print(
                f"ACE2_LAYER16_RESIDUAL_SUM_CANDIDATE group={group_size} "
                f"top={record['top_token_id']} rank={record['reference_rank']} "
                f"sat={record['sum_saturation_count']}",
                flush=True,
            )

    localizer.require(trace and trace[0]["control_contract"], "exact control did not complete first")
    control = trace[0]
    non_control = [record for record in trace if not record["control_contract"]]
    localizer.require(non_control, "no repair contract completed")
    order = {group: index for index, group in enumerate(OUTPUT_GROUP_SIZES)}
    restored = [record for record in non_control if record["top_token_id"] == localizer.REFERENCE_TOKEN]
    improved = [record for record in non_control if int(record["reference_rank"]) < 31879]
    best_non_control = min(
        non_control,
        key=lambda record: (
            int(record["reference_rank"]), order[int(record["sum_output_group_size"])]
        ),
    )
    if restored:
        selected = restored[0]
        selection_reason = "restored_token_9707_in_declared_execution_order"
    elif improved:
        selected = min(
            improved,
            key=lambda record: (
                int(record["reference_rank"]), order[int(record["sum_output_group_size"])]
            ),
        )
        selection_reason = "best_independent_rank_improvement_over_31879"
    else:
        selected = None
        selection_reason = "no_non_control_contract_improved_rank_31879_or_restored_token_9707"

    validation = selected if selected is not None else best_non_control
    restored_token = selected is not None and selected["top_token_id"] == localizer.REFERENCE_TOKEN
    result = {
        "schema_version": 1,
        "status": (
            "PASS_LAYER16_RESIDUAL_SUM_DUAL_SCALE_TOKEN_9707_RESTORED"
            if restored_token
            else "PARTIAL_LAYER16_RESIDUAL_SUM_DUAL_SCALE_RANK_IMPROVED"
            if selected is not None
            else "PARTIAL_LAYER16_RESIDUAL_SUM_DUAL_SCALE_NO_IMPROVEMENT"
        ),
        "classification": "candidate_only_no_official_attempt_created",
        "boundary": BOUNDARY,
        "baseline_reference_rank": 31879,
        "baseline_top_token_id": 37865,
        "ordered_output_group_sizes": list(OUTPUT_GROUP_SIZES),
        "control": control,
        "trace": trace,
        "failures": failures,
        "selected_output_group_size": int(selected["sum_output_group_size"]) if selected else None,
        "selected_reference_rank": int(selected["reference_rank"]) if selected else 31879,
        "selected_top_token_id": int(selected["top_token_id"]) if selected else 37865,
        "selection_reason": selection_reason,
        "best_non_control_output_group_size": int(best_non_control["sum_output_group_size"]),
        "best_non_control_reference_rank": int(best_non_control["reference_rank"]),
        "best_non_control_top_token_id": int(best_non_control["top_token_id"]),
        "rtl_validation_output_group_size": int(validation["sum_output_group_size"]),
        "first_token_restored": bool(restored_token),
        "next_exact_upstream_scale_alignment_boundary": None if selected else NEXT_BOUNDARY,
        "prequeue_required": selected is None,
        "official_run_launched": False,
        "official_attempt_consumed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "bindings": {
            "freeze": localizer.file_record(output / "candidate-freeze.json"),
            "rtl_preflight": localizer.file_record(output / "rtl-interface-preflight.json"),
            "accepted_accumulator_sha256s": localizer.file_record(ACCEPTED_ACCUMULATOR / "SHA256SUMS"),
            "localization_sha256s": localizer.file_record(LOCALIZATION / "SHA256SUMS"),
        },
    }
    localizer.write_json(output / "result.json", result)
    localizer.write_sums(output)
    print(
        f"ACE2_LAYER16_RESIDUAL_SUM_RESULT status={result['status']} "
        f"selected={result['selected_output_group_size']} "
        f"best_non_control={result['best_non_control_output_group_size']} "
        f"rank={result['best_non_control_reference_rank']} restored={result['first_token_restored']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ACE2_LAYER16_RESIDUAL_SUM_FAIL detail={error}", file=sys.stderr, flush=True)
        raise
