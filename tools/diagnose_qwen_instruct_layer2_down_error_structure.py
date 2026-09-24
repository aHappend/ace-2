#!/usr/bin/env python3
"""Measure exact layer-2 down-projection row/lane causality.

The official run is diagnostic-only.  It binds the frozen nine-prompt matrix,
uses every archived BF16 teacher-forced decode prefix, and keeps the model BF16
except for the unchanged per-output-channel W4 rule at
``model.layers.2.mlp.down_proj``.  At each BF16/W4 winner disagreement it
records every output-row error and every exact single-row BF16-restoration
effect on the fixed BF16-winner minus W4-winner logit margin.  A bounded,
predeclared relevant-row set is then decomposed into all 4,864 exact input-lane
weight-quantization products at the causal position that emits the measured
logits.

No result selects or recommends a weight policy, mutates RTL, or advances a
project stage.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from discriminate_qwen_instruct_w4_weight_policies import PROMPTS
from localize_qwen_instruct_w4_weight_error import quantized_w4_chunk
from qwen_instruct_option_b import (
    ROOT,
    SNAPSHOT,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    verify_versions,
)
from qwen_instruct_w4a8_oracle import DEFAULT_SYSTEM


CONTRACT_ID = "qwen-instruct-layer2-down-row-lane-causality-v1"
DOWN_OPERATOR = "model.layers.2.mlp.down_proj"
MATRIX_SHA256 = "276fc96611ecc2d5205024d0190460cce2847b766e1f436a9982361d4ae6ceaf"
MATRIX = (
    ROOT
    / "build/option-b-v2-weight-policy-20260806/"
    "full_sequence_discrimination.json"
)
CAUSALITY_RESULT = (
    ROOT
    / "build/option-b-four-operator-teacher-forced-causality-v2-20260806/"
    "results.json"
)
PROMPT_SOURCE = ROOT / "tools/discriminate_qwen_instruct_w4_weight_policies.py"
ROW_COUNT = 896
LANE_COUNT = 4864
ROW_BATCH_SIZE = 16
TOP_POSITIVE_RESTORATION_ROWS = 32
TOP_NEGATIVE_RESTORATION_ROWS = 32
TOP_ABSOLUTE_ERROR_ROWS = 32
TOP_LANES_PER_DIRECTION = 16
SERIAL_PREFLIGHT_ROWS = (0, 447, 895)
ROW_ARRAY_COLUMNS = (
    "output_row",
    "bf16_output",
    "w4_output",
    "measured_w4_minus_bf16_error",
    "lane_sum_w4_minus_bf16_error",
    "lane_sum_closure_absolute_error",
    "full_w4_bf16_winner_logit",
    "full_w4_w4_winner_logit",
    "full_w4_margin_bf16_minus_w4",
    "restored_bf16_winner_logit",
    "restored_w4_winner_logit",
    "restored_margin_bf16_minus_w4",
    "restoration_margin_effect",
)


def environment() -> dict[str, Any]:
    expected = ROOT / ".venv/bin/python"
    observed = Path(sys.executable)
    require(expected.is_file(), f"project Python is missing: {expected}")
    require(os.path.samefile(observed, expected), f"wrong Python executable: {observed}")
    return {
        "bound_entrypoint": "./.venv/bin/python",
        "resolved_executable": str(expected.resolve()),
        "sys_executable": str(observed),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": verify_versions(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
    }


def prompt_by_id() -> dict[str, Any]:
    return {case.case_id: case for case in PROMPTS}


def singleton_subset(prompt_record: dict[str, Any]) -> dict[str, Any]:
    return next(
        subset
        for subset in prompt_record["subsets"]
        if subset["w4_operators"] == [DOWN_OPERATOR]
    )


def frozen_prompt_records() -> list[dict[str, Any]]:
    causality = load_json(CAUSALITY_RESULT)
    archived = {record["case_id"]: record for record in causality["prompts"]}
    records: list[dict[str, Any]] = []
    for case in PROMPTS:
        record = archived[case.case_id]
        singleton = singleton_subset(record)
        steps = []
        for bf16_token, step in zip(
            record["bf16_generated_token_ids"], singleton["steps"], strict=True
        ):
            steps.append(
                {
                    "step_index": step["step_index"],
                    "bf16_token_id": bf16_token,
                    "w4_token_id": step["candidate_token_id"],
                    "winner_disagrees": not step["token_match"],
                }
            )
        records.append(
            {
                "case_id": case.case_id,
                "prompt_sha256": sha256_bytes(case.prompt.encode()),
                "bf16_generated_token_ids": record["bf16_generated_token_ids"],
                "steps": steps,
            }
        )
    return records


def predeclared_contract() -> dict[str, Any]:
    prompt_records = frozen_prompt_records()
    disagreement_count = sum(
        step["winner_disagrees"]
        for record in prompt_records
        for step in record["steps"]
    )
    eligible_step_count = sum(len(record["steps"]) for record in prompt_records)
    affected_prompt_count = sum(
        any(step["winner_disagrees"] for step in record["steps"])
        for record in prompt_records
    )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "frozen_inputs": {
            "matrix": file_record(MATRIX),
            "required_matrix_sha256": MATRIX_SHA256,
            "four_operator_causality_result": file_record(CAUSALITY_RESULT),
            "prompt_source": file_record(PROMPT_SOURCE),
            "system_prompt_sha256": sha256_bytes(DEFAULT_SYSTEM.encode()),
            "prompts": prompt_records,
            "prompt_count": len(prompt_records),
            "eligible_decode_step_count": eligible_step_count,
            "winner_disagreement_step_count": disagreement_count,
            "affected_prompt_count": affected_prompt_count,
        },
        "model_and_quantization": {
            "model_snapshot": str(SNAPSHOT),
            "dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "use_cache": False,
            "device": "cpu",
            "operator": DOWN_OPERATOR,
            "weight_bits": 4,
            "signed_integer_range": [-8, 7],
            "w4_rule": "per-output-channel absmax/7, torch.round, clamp[-8,7], dequantize to BF16",
            "all_other_weights": "BF16",
            "activations_and_non_linear_operators": "BF16",
            "greedy_winner_rule": "torch.argmax; exact ties select the lowest token index",
        },
        "measurement": {
            "teacher_forcing": "all archived BF16-generated token prefixes for all nine prompts",
            "row_error_position": "the layer-2 down-projection causal position whose hidden state emits the measured next-token logits",
            "row_error_direction": "W4 output minus BF16 output",
            "single_row_restoration": "start from full per-output-channel W4 down-projection output at every causal position and restore exactly one complete output row to its BF16 output at every causal position",
            "fixed_margin": "BF16-winner logit minus singleton-W4-winner logit",
            "restoration_effect": "single-row-restored fixed margin minus full-W4 fixed margin",
            "all_output_rows_measured": ROW_COUNT,
            "row_batch_size": ROW_BATCH_SIZE,
            "batch_preflight": {
                "serial_rows": list(SERIAL_PREFLIGHT_ROWS),
                "semantics": "use batched row restoration only if full-vocabulary logits for the control and all serial preflight rows are torch.equal to batch-1 execution; otherwise use batch-1 for every row in that prompt",
            },
            "relevant_row_rule": {
                "top_positive_restoration_effect_count": TOP_POSITIVE_RESTORATION_ROWS,
                "top_negative_restoration_effect_count": TOP_NEGATIVE_RESTORATION_ROWS,
                "top_absolute_measured_row_error_count": TOP_ABSOLUTE_ERROR_ROWS,
                "include_every_row_changing_the_full_w4_argmax": True,
                "include_every_row_restoring_the_bf16_argmax": True,
                "union_is_deduplicated_and_sorted_by_output_row": True,
            },
            "lane_decomposition": "for every relevant row, preserve all 4,864 products input_bf16[lane] * (weight_w4[row,lane] - weight_bf16[row,lane]) in float64",
            "lane_summary_limit_each_direction": TOP_LANES_PER_DIRECTION,
            "raw_storage": "deterministic NumPy .npy arrays plus canonical JSON and SHA-256 records",
        },
        "predeclared_tolerances": {
            "lane_sum_float64_internal_absolute": 1e-12,
            "lane_sum_float64_internal_relative": 1e-12,
            "measured_bf16_row_error_absolute_floor": 0.001953125,
            "measured_bf16_row_error_relative_to_output_magnitudes": 0.015625,
            "batch_and_serial_full_vocabulary_logits": "torch.equal",
            "archived_control_winner_tokens": "exact integer equality",
            "aggregate_recalculation": "exact from immutable raw arrays except the declared float64 sum tolerances",
        },
        "pass_criteria": {
            "matrix_sha256_matches_required_value": True,
            "matrix_remains_zero_eligible_and_unselected": True,
            "all_nine_prompts_and_all_eligible_steps_control_match": True,
            "all_archived_winner_disagreements_reproduced_exactly": True,
            "every_output_row_restoration_measured_at_each_disagreement": True,
            "every_relevant_row_has_all_lane_contributions": True,
            "all_lane_sums_close_within_predeclared_tolerances": True,
            "all_raw_artifact_hashes_verify": True,
            "selected_policy_id_remains_null": True,
        },
        "reporting_boundary": {
            "permitted": "diagnostic evidence for Fresh Reviewer failure-taxonomy adjudication",
            "selected_policy_id": None,
            "weight_policy_selection_or_recommendation": False,
            "candidate_or_image_freeze": False,
            "rtl_or_ppa_execution": False,
            "stage_transition_or_closure": False,
        },
        "test_source": file_record(Path(__file__)),
    }


def write_contract(path: Path) -> None:
    require(not path.exists(), f"refusing to overwrite predeclared contract: {path}")
    contract = predeclared_contract()
    wrapper = {
        "contract": contract,
        "contract_sha256": sha256_bytes(canonical_bytes(contract)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(wrapper))


def load_bound_contract(path: Path) -> tuple[dict[str, Any], str]:
    wrapper = load_json(path)
    require(set(wrapper) == {"contract", "contract_sha256"}, "contract envelope differs")
    contract = wrapper["contract"]
    digest = sha256_bytes(canonical_bytes(contract))
    require(digest == wrapper["contract_sha256"], "contract digest differs")
    require(contract == predeclared_contract(), "source or frozen input differs from contract")
    return contract, digest


def verify_frozen_evidence(contract: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(MATRIX) == MATRIX_SHA256, "matrix SHA-256 differs")
    matrix = load_json(MATRIX)
    require(matrix["status"] == "BLOCKED_NO_ELIGIBLE_WEIGHT_POLICY", "matrix status differs")
    selection = matrix["selection"]
    require(selection["selected_policy_id"] is None, "matrix selected a policy")
    require(selection["eligible_policy_ids"] == [], "matrix has an eligible policy")
    require(not selection["unique_selected_policy"], "matrix reports a unique policy")
    causality = load_json(CAUSALITY_RESULT)
    require(
        causality["status"]
        == "PASS_EXACT_FOUR_OPERATOR_TEACHER_FORCED_CAUSALITY_V2_RECORDED",
        "causality result status differs",
    )
    require(contract["frozen_inputs"]["prompt_count"] == 9, "prompt count differs")
    require(
        contract["frozen_inputs"]["eligible_decode_step_count"] == 41,
        "eligible step count differs",
    )
    require(
        contract["frozen_inputs"]["winner_disagreement_step_count"] == 10,
        "disagreement count differs",
    )
    return {
        "matrix_status": matrix["status"],
        "matrix_sha256": sha256_file(MATRIX),
        "selected_policy_id": selection["selected_policy_id"],
        "eligible_policy_ids": selection["eligible_policy_ids"],
        "causality_status": causality["status"],
    }


def chat_input(tokenizer: Any, prompt: str, frozen_tokens: list[int]) -> tuple[Tensor, int]:
    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    require(isinstance(prompt_ids, Tensor) and prompt_ids.ndim == 2, "tokenization differs")
    continuation = torch.tensor([frozen_tokens[:-1]], dtype=prompt_ids.dtype)
    return torch.cat((prompt_ids, continuation), dim=1), int(prompt_ids.shape[1])


def prediction_logits(logits: Tensor, prompt_count: int, step_count: int) -> Tensor:
    start = prompt_count - 1
    stop = start + step_count
    require(stop <= logits.shape[1], "prediction positions exceed logits")
    return logits[:, start:stop].detach().cpu()


def tensor_sha256(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def capture_hook(storage: dict[str, Tensor]) -> Callable[[nn.Module, tuple[Any, ...], Tensor], Tensor]:
    def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> Tensor:
        require(isinstance(output, Tensor), "down-projection output is not a tensor")
        storage["input"] = inputs[0].detach().cpu().clone()
        storage["output"] = output.detach().cpu().clone()
        return output

    return hook


def w4_capture_hook(
    weight: Tensor, storage: dict[str, Tensor]
) -> Callable[[nn.Module, tuple[Any, ...], Tensor], Tensor]:
    def hook(module: nn.Module, inputs: tuple[Any, ...], _output: Tensor) -> Tensor:
        require(isinstance(module, nn.Linear), "down projection is not linear")
        value = F.linear(inputs[0], weight, module.bias)
        storage["input"] = inputs[0].detach().cpu().clone()
        storage["output"] = value.detach().cpu().clone()
        return value

    return hook


def fixed_pair(logits: Tensor, bf16_token: int, w4_token: int) -> tuple[float, float, float]:
    values = logits.to(torch.float64)
    left = float(values[bf16_token])
    right = float(values[w4_token])
    return left, right, left - right


def restored_output_hook(
    w4_output: Tensor,
    bf16_output: Tensor,
    restored_rows: list[int | None],
) -> Callable[[nn.Module, tuple[Any, ...], Tensor], Tensor]:
    def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Tensor) -> Tensor:
        batch = inputs[0].shape[0]
        require(batch == len(restored_rows), "restoration batch size differs")
        value = w4_output.expand(batch, -1, -1).clone()
        for sample, row in enumerate(restored_rows):
            if row is not None:
                value[sample, :, row] = bf16_output[0, :, row]
        return value

    return hook


def run_restoration_batch(
    model: nn.Module,
    down: nn.Linear,
    input_ids: Tensor,
    prompt_count: int,
    step_count: int,
    w4_output: Tensor,
    bf16_output: Tensor,
    rows: list[int | None],
) -> Tensor:
    hook = down.register_forward_hook(restored_output_hook(w4_output, bf16_output, rows))
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids.expand(len(rows), -1), use_cache=False).logits
        return prediction_logits(logits, prompt_count, step_count)
    finally:
        hook.remove()


def exact_batch_mode(
    model: nn.Module,
    down: nn.Linear,
    input_ids: Tensor,
    prompt_count: int,
    step_count: int,
    w4_output: Tensor,
    bf16_output: Tensor,
    full_w4_logits: Tensor,
) -> tuple[bool, dict[int, Tensor]]:
    serial: dict[int, Tensor] = {}
    for row in SERIAL_PREFLIGHT_ROWS:
        serial[row] = run_restoration_batch(
            model,
            down,
            input_ids,
            prompt_count,
            step_count,
            w4_output,
            bf16_output,
            [row],
        )[0]
    rows: list[int | None] = [None, *SERIAL_PREFLIGHT_ROWS]
    batched = run_restoration_batch(
        model,
        down,
        input_ids,
        prompt_count,
        step_count,
        w4_output,
        bf16_output,
        rows,
    )
    exact = torch.equal(batched[0], full_w4_logits[0]) and all(
        torch.equal(batched[index], serial[row])
        for index, row in enumerate(SERIAL_PREFLIGHT_ROWS, start=1)
    )
    return exact, serial


def all_row_restorations(
    *,
    model: nn.Module,
    down: nn.Linear,
    input_ids: Tensor,
    prompt_count: int,
    step_count: int,
    w4_output: Tensor,
    bf16_output: Tensor,
    full_w4_logits: Tensor,
    case_id: str,
) -> tuple[Tensor, str, dict[str, Any]]:
    batch_exact, serial_preflight = exact_batch_mode(
        model,
        down,
        input_ids,
        prompt_count,
        step_count,
        w4_output,
        bf16_output,
        full_w4_logits,
    )
    mode = "batch16_exact_preflight" if batch_exact else "serial_batch1_fallback"
    restored = torch.empty((ROW_COUNT, step_count, full_w4_logits.shape[-1]), dtype=full_w4_logits.dtype)
    chunk_size = ROW_BATCH_SIZE if batch_exact else 1
    for start in range(0, ROW_COUNT, chunk_size):
        stop = min(start + chunk_size, ROW_COUNT)
        rows = list(range(start, stop))
        batch_rows: list[int | None] = [None, *rows] if batch_exact else rows
        logits = run_restoration_batch(
            model,
            down,
            input_ids,
            prompt_count,
            step_count,
            w4_output,
            bf16_output,
            batch_rows,
        )
        if batch_exact:
            require(torch.equal(logits[0], full_w4_logits[0]), "batch control logits differ")
            logits = logits[1:]
        restored[start:stop] = logits
        if start == 0 or stop == ROW_COUNT or stop % 128 == 0:
            print(
                "ACE2_LAYER2_ROW_PROGRESS "
                f"case={case_id} rows={stop}/{ROW_COUNT} mode={mode}",
                flush=True,
            )
    serial_exact = all(
        torch.equal(restored[row], logits)
        for row, logits in serial_preflight.items()
    )
    require(serial_exact, "stored row logits differ from serial preflight")
    return restored, mode, {
        "batch_preflight_exact": batch_exact,
        "serial_preflight_rows": list(SERIAL_PREFLIGHT_ROWS),
        "stored_serial_preflight_exact": serial_exact,
    }


def deterministic_top(values: np.ndarray, count: int, mode: str) -> list[int]:
    if mode == "positive":
        return sorted(range(len(values)), key=lambda i: (-float(values[i]), i))[:count]
    if mode == "negative":
        return sorted(range(len(values)), key=lambda i: (float(values[i]), i))[:count]
    if mode == "absolute":
        return sorted(range(len(values)), key=lambda i: (-abs(float(values[i])), i))[:count]
    raise ValueError(mode)


def relevant_rows(
    row_errors: np.ndarray,
    restoration_effects: np.ndarray,
    restored_candidates: np.ndarray,
    full_w4_candidate: int,
    bf16_candidate: int,
) -> tuple[list[int], dict[int, list[str]]]:
    reasons: dict[int, set[str]] = {}

    def add(rows: list[int], reason: str) -> None:
        for row in rows:
            reasons.setdefault(row, set()).add(reason)

    add(
        deterministic_top(restoration_effects, TOP_POSITIVE_RESTORATION_ROWS, "positive"),
        "top_positive_restoration_effect",
    )
    add(
        deterministic_top(restoration_effects, TOP_NEGATIVE_RESTORATION_ROWS, "negative"),
        "top_negative_restoration_effect",
    )
    add(
        deterministic_top(row_errors, TOP_ABSOLUTE_ERROR_ROWS, "absolute"),
        "top_absolute_measured_row_error",
    )
    add(
        [int(row) for row in np.flatnonzero(restored_candidates != full_w4_candidate)],
        "changes_full_w4_argmax",
    )
    add(
        [int(row) for row in np.flatnonzero(restored_candidates == bf16_candidate)],
        "restores_bf16_argmax",
    )
    rows = sorted(reasons)
    return rows, {row: sorted(reasons[row]) for row in rows}


def measured_row_tolerance(bf16_value: float, w4_value: float) -> float:
    return 0.001953125 + 0.015625 * (abs(bf16_value) + abs(w4_value))


def write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.lib.format.write_array(handle, np.ascontiguousarray(value), allow_pickle=False)


def npy_record(path: Path, value: np.ndarray) -> dict[str, Any]:
    return {
        **file_record(path),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def lane_extremes(values: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    negative = sorted(range(len(values)), key=lambda i: (float(values[i]), i))[
        :TOP_LANES_PER_DIRECTION
    ]
    positive = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))[
        :TOP_LANES_PER_DIRECTION
    ]
    return {
        "most_negative": [
            {"input_lane": lane, "contribution": float(values[lane])}
            for lane in negative
        ],
        "most_positive": [
            {"input_lane": lane, "contribution": float(values[lane])}
            for lane in positive
        ],
    }


def concentration(values: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(values)
    total = float(absolute.sum(dtype=np.float64))
    ordered = np.sort(absolute)[::-1]
    return {
        "absolute_mass": total,
        "top1_absolute_mass_fraction": None if total == 0.0 else float(ordered[:1].sum() / total),
        "top4_absolute_mass_fraction": None if total == 0.0 else float(ordered[:4].sum() / total),
        "top16_absolute_mass_fraction": None if total == 0.0 else float(ordered[:16].sum() / total),
        "nonzero_count": int(np.count_nonzero(values)),
    }


def run(contract_path: Path, output_dir: Path) -> int:
    require(not output_dir.exists(), f"refusing to overwrite official attempt: {output_dir}")
    contract, contract_digest = load_bound_contract(contract_path)
    env = environment()
    source_contract = verify_source_contract()
    source_identity = verify_source_snapshot()
    frozen_evidence = verify_frozen_evidence(contract)
    output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    down = modules.get(DOWN_OPERATOR)
    require(isinstance(down, nn.Linear), "layer-2 down projection is missing")
    require(down.out_features == ROW_COUNT, "output row count differs")
    require(down.in_features == LANE_COUNT, "input lane count differs")
    bf16_weight = down.weight.detach().cpu().clone()
    w4_weight = quantized_w4_chunk(bf16_weight, None)
    delta_weight64 = w4_weight.to(torch.float64) - bf16_weight.to(torch.float64)

    prompts = prompt_by_id()
    prompt_summaries: list[dict[str, Any]] = []
    disagreement_summaries: list[dict[str, Any]] = []
    all_controls_pass = True
    total_measured_rows = 0
    total_relevant_rows = 0
    all_lane_closures_pass = True

    for prompt_index, frozen in enumerate(contract["frozen_inputs"]["prompts"], start=1):
        case = prompts[frozen["case_id"]]
        require(sha256_bytes(case.prompt.encode()) == frozen["prompt_sha256"], "prompt hash differs")
        input_ids, prompt_count = chat_input(
            tokenizer, case.prompt, frozen["bf16_generated_token_ids"]
        )
        step_count = len(frozen["steps"])

        bf16_capture: dict[str, Tensor] = {}
        hook = down.register_forward_hook(capture_hook(bf16_capture))
        try:
            with torch.inference_mode():
                bf16_all = model(input_ids=input_ids, use_cache=False).logits
        finally:
            hook.remove()
        bf16_logits = prediction_logits(bf16_all, prompt_count, step_count)

        w4_capture: dict[str, Tensor] = {}
        hook = down.register_forward_hook(w4_capture_hook(w4_weight, w4_capture))
        try:
            with torch.inference_mode():
                w4_all = model(input_ids=input_ids, use_cache=False).logits
        finally:
            hook.remove()
        w4_logits = prediction_logits(w4_all, prompt_count, step_count)
        require(torch.equal(bf16_capture["input"], w4_capture["input"]), "down inputs differ")

        actual_steps = []
        disagreement_indices: list[int] = []
        for step in frozen["steps"]:
            index = step["step_index"]
            bf16_candidate = int(bf16_logits[0, index].argmax())
            w4_candidate = int(w4_logits[0, index].argmax())
            archived_match = (
                bf16_candidate == step["bf16_token_id"]
                and w4_candidate == step["w4_token_id"]
                and (bf16_candidate != w4_candidate) == step["winner_disagrees"]
            )
            all_controls_pass &= archived_match
            if bf16_candidate != w4_candidate:
                disagreement_indices.append(index)
            actual_steps.append(
                {
                    "step_index": index,
                    "bf16_candidate_token_id": bf16_candidate,
                    "w4_candidate_token_id": w4_candidate,
                    "winner_disagrees": bf16_candidate != w4_candidate,
                    "archived_control_match": archived_match,
                }
            )

        restored_logits: Tensor | None = None
        execution_mode = None
        preflight: dict[str, Any] | None = None
        if disagreement_indices:
            restored_logits, execution_mode, preflight = all_row_restorations(
                model=model,
                down=down,
                input_ids=input_ids,
                prompt_count=prompt_count,
                step_count=step_count,
                w4_output=w4_capture["output"],
                bf16_output=bf16_capture["output"],
                full_w4_logits=w4_logits,
                case_id=frozen["case_id"],
            )

        prompt_summaries.append(
            {
                "case_id": frozen["case_id"],
                "prompt_sha256": frozen["prompt_sha256"],
                "chat_template_token_count": prompt_count,
                "eligible_decode_step_count": step_count,
                "winner_disagreement_step_indices": disagreement_indices,
                "steps": actual_steps,
                "bf16_prediction_logits_sha256": tensor_sha256(bf16_logits),
                "w4_prediction_logits_sha256": tensor_sha256(w4_logits),
                "row_restoration_execution_mode": execution_mode,
                "row_restoration_preflight": preflight,
            }
        )

        if restored_logits is None:
            print(
                "ACE2_LAYER2_PROMPT_CONTROL "
                f"prompt={prompt_index}/9 case={frozen['case_id']} disagreements=0",
                flush=True,
            )
            continue

        for step_index in disagreement_indices:
            step = frozen["steps"][step_index]
            bf16_token = step["bf16_token_id"]
            w4_token = step["w4_token_id"]
            target_position = prompt_count - 1 + step_index
            bf16_output = bf16_capture["output"][0, target_position].to(torch.float64)
            w4_output = w4_capture["output"][0, target_position].to(torch.float64)
            measured_error = (w4_output - bf16_output).numpy()
            input64 = bf16_capture["input"][0, target_position].to(torch.float64)
            lane_sum_all_rows = (delta_weight64 * input64.unsqueeze(0)).sum(dim=1).numpy()
            closure = np.abs(lane_sum_all_rows - measured_error)
            tolerances = np.array(
                [
                    measured_row_tolerance(float(bf16_output[row]), float(w4_output[row]))
                    for row in range(ROW_COUNT)
                ],
                dtype=np.float64,
            )

            full_left, full_right, full_margin = fixed_pair(
                w4_logits[0, step_index], bf16_token, w4_token
            )
            restored_left = restored_logits[:, step_index, bf16_token].to(torch.float64).numpy()
            restored_right = restored_logits[:, step_index, w4_token].to(torch.float64).numpy()
            restored_margins = restored_left - restored_right
            restoration_effects = restored_margins - full_margin
            restored_candidates = restored_logits[:, step_index].argmax(dim=1).to(torch.int64).numpy()

            row_table = np.column_stack(
                (
                    np.arange(ROW_COUNT, dtype=np.float64),
                    bf16_output.numpy(),
                    w4_output.numpy(),
                    measured_error,
                    lane_sum_all_rows,
                    closure,
                    np.full(ROW_COUNT, full_left, dtype=np.float64),
                    np.full(ROW_COUNT, full_right, dtype=np.float64),
                    np.full(ROW_COUNT, full_margin, dtype=np.float64),
                    restored_left,
                    restored_right,
                    restored_margins,
                    restoration_effects,
                )
            )
            selected_rows, selected_reasons = relevant_rows(
                measured_error,
                restoration_effects,
                restored_candidates,
                int(w4_logits[0, step_index].argmax()),
                int(bf16_logits[0, step_index].argmax()),
            )
            lane_values = (
                delta_weight64[selected_rows] * input64.unsqueeze(0)
            ).numpy()
            lane_sums = lane_values.sum(axis=1, dtype=np.float64)
            lane_internal_closure = np.abs(lane_sums - lane_sum_all_rows[selected_rows])
            internal_scale = np.maximum(
                1.0,
                np.maximum(np.abs(lane_sums), np.abs(lane_sum_all_rows[selected_rows])),
            )
            internal_ok = lane_internal_closure <= 1e-12 * internal_scale
            measured_ok = closure[selected_rows] <= tolerances[selected_rows]
            all_lane_closures_pass &= bool(np.all(internal_ok) and np.all(measured_ok))

            relative_dir = Path("raw") / frozen["case_id"] / f"step-{step_index:02d}"
            row_path = output_dir / relative_dir / "row_measurements.npy"
            candidate_path = output_dir / relative_dir / "restored_candidate_token_ids.npy"
            relevant_path = output_dir / relative_dir / "relevant_output_rows.npy"
            lane_path = output_dir / relative_dir / "lane_contributions.npy"
            write_npy(row_path, row_table.astype(np.float64, copy=False))
            write_npy(candidate_path, restored_candidates.astype(np.int64, copy=False))
            write_npy(relevant_path, np.asarray(selected_rows, dtype=np.int64))
            write_npy(lane_path, lane_values.astype(np.float64, copy=False))

            relevant_records = []
            for local_index, row in enumerate(selected_rows):
                relevant_records.append(
                    {
                        "output_row": row,
                        "selection_reasons": selected_reasons[row],
                        "measured_w4_minus_bf16_error": float(measured_error[row]),
                        "lane_sum_w4_minus_bf16_error": float(lane_sums[local_index]),
                        "lane_sum_internal_absolute_error": float(
                            lane_internal_closure[local_index]
                        ),
                        "lane_sum_to_measured_absolute_error": float(closure[row]),
                        "lane_sum_to_measured_tolerance": float(tolerances[row]),
                        "restoration_margin_effect": float(restoration_effects[row]),
                        "restored_candidate_token_id": int(restored_candidates[row]),
                        "lane_concentration": concentration(lane_values[local_index]),
                        "reported_lane_extremes": lane_extremes(lane_values[local_index]),
                    }
                )

            total_measured_rows += ROW_COUNT
            total_relevant_rows += len(selected_rows)
            disagreement_summaries.append(
                {
                    "case_id": frozen["case_id"],
                    "step_index": step_index,
                    "target_position": target_position,
                    "bf16_winner_token_id": bf16_token,
                    "w4_winner_token_id": w4_token,
                    "bf16_control": {
                        "candidate_token_id": int(bf16_logits[0, step_index].argmax()),
                        "fixed_margin_bf16_minus_w4": fixed_pair(
                            bf16_logits[0, step_index], bf16_token, w4_token
                        )[2],
                    },
                    "full_w4_control": {
                        "candidate_token_id": int(w4_logits[0, step_index].argmax()),
                        "bf16_winner_logit": full_left,
                        "w4_winner_logit": full_right,
                        "fixed_margin_bf16_minus_w4": full_margin,
                    },
                    "row_measurement_count": ROW_COUNT,
                    "relevant_row_count": len(selected_rows),
                    "single_row_candidate_change_count": int(
                        np.count_nonzero(
                            restored_candidates != int(w4_logits[0, step_index].argmax())
                        )
                    ),
                    "single_row_bf16_winner_restore_count": int(
                        np.count_nonzero(
                            restored_candidates == int(bf16_logits[0, step_index].argmax())
                        )
                    ),
                    "row_error_concentration": concentration(measured_error),
                    "restoration_effect_concentration": concentration(restoration_effects),
                    "maximum_row_lane_sum_to_measured_absolute_error": float(
                        closure.max()
                    ),
                    "all_relevant_lane_internal_sums_close": bool(np.all(internal_ok)),
                    "all_relevant_lane_sums_close_to_measured_row_error": bool(
                        np.all(measured_ok)
                    ),
                    "raw_artifacts": {
                        "row_measurements": {
                            **npy_record(row_path, row_table),
                            "columns": list(ROW_ARRAY_COLUMNS),
                        },
                        "restored_candidate_token_ids": npy_record(
                            candidate_path, restored_candidates
                        ),
                        "relevant_output_rows": npy_record(
                            relevant_path, np.asarray(selected_rows, dtype=np.int64)
                        ),
                        "lane_contributions": npy_record(lane_path, lane_values),
                    },
                    "relevant_rows": relevant_records,
                }
            )
        print(
            "ACE2_LAYER2_PROMPT_CONTROL "
            f"prompt={prompt_index}/9 case={frozen['case_id']} "
            f"disagreements={len(disagreement_indices)} mode={execution_mode}",
            flush=True,
        )

    checks = {
        "matrix_sha256_matches_required_value": frozen_evidence["matrix_sha256"]
        == MATRIX_SHA256,
        "matrix_remains_zero_eligible_and_unselected": (
            frozen_evidence["selected_policy_id"] is None
            and frozen_evidence["eligible_policy_ids"] == []
        ),
        "all_nine_prompts_evaluated": len(prompt_summaries) == 9,
        "all_41_eligible_decode_steps_evaluated": sum(
            prompt["eligible_decode_step_count"] for prompt in prompt_summaries
        )
        == 41,
        "all_archived_bf16_and_w4_control_winners_match": all_controls_pass,
        "all_10_winner_disagreements_reproduced": len(disagreement_summaries) == 10,
        "all_896_rows_measured_per_disagreement": total_measured_rows == 10 * ROW_COUNT,
        "all_relevant_rows_have_4864_lane_contributions": total_relevant_rows
        == sum(item["relevant_row_count"] for item in disagreement_summaries),
        "all_lane_sums_close": all_lane_closures_pass,
        "selected_policy_id_remains_null": frozen_evidence["selected_policy_id"] is None,
        "no_stage_or_policy_action_taken": True,
    }
    passed = all(checks.values())
    status = (
        "PASS_BOUND_LAYER2_DOWN_ROW_LANE_CAUSALITY_RECORDED"
        if passed
        else "FAIL_BOUND_LAYER2_DOWN_ROW_LANE_CAUSALITY"
    )
    result = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_layer2_down_row_lane_causality",
        "status": status,
        "predeclared_contract": {
            "artifact": file_record(contract_path),
            "contract_sha256": contract_digest,
        },
        "environment": env,
        "model": source_identity,
        "source_contract_status": source_contract["status"],
        "frozen_evidence": frozen_evidence,
        "checks": checks,
        "coverage": {
            "prompt_count": len(prompt_summaries),
            "eligible_decode_step_count": 41,
            "winner_disagreement_step_count": len(disagreement_summaries),
            "affected_prompt_count": sum(
                bool(prompt["winner_disagreement_step_indices"])
                for prompt in prompt_summaries
            ),
            "row_measurement_count": total_measured_rows,
            "relevant_row_lane_decomposition_count": total_relevant_rows,
            "lane_contribution_value_count": total_relevant_rows * LANE_COUNT,
        },
        "prompts": prompt_summaries,
        "winner_disagreements": disagreement_summaries,
        "reporting_limits": contract["measurement"]["relevant_row_rule"],
        "predeclared_tolerances": contract["predeclared_tolerances"],
        "taxonomy_observables_only": {
            "per_disagreement": [
                {
                    "case_id": item["case_id"],
                    "step_index": item["step_index"],
                    "single_row_candidate_change_count": item[
                        "single_row_candidate_change_count"
                    ],
                    "single_row_bf16_winner_restore_count": item[
                        "single_row_bf16_winner_restore_count"
                    ],
                    "row_error_concentration": item["row_error_concentration"],
                    "restoration_effect_concentration": item[
                        "restoration_effect_concentration"
                    ],
                    "relevant_row_count": item["relevant_row_count"],
                }
                for item in disagreement_summaries
            ],
            "taxonomy_label": None,
            "selected_policy_id": None,
            "policy_recommendation": None,
        },
        "scope_guards": {
            "accelerator_executed": False,
            "candidate_frozen": False,
            "grouped_policy_selected": False,
            "network_access_performed": False,
            "ppa_executed": False,
            "rtl_mutated": False,
            "stage_transition_claimed": False,
        },
        "artifacts": {
            "matrix": file_record(MATRIX),
            "four_operator_causality_result": file_record(CAUSALITY_RESULT),
            "prompt_source": file_record(PROMPT_SOURCE),
            "source_contract": file_record(
                ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
            ),
            "test_source": file_record(Path(__file__)),
        },
        "claim_boundary": "Diagnostic measurements for Fresh Reviewer failure-taxonomy adjudication only. No weight policy is selected or recommended, and no RTL-stage or project-stage closure is claimed.",
    }
    result_path = output_dir / "results.json"
    result_path.write_bytes(canonical_bytes(result))
    sums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(
        "ACE2_LAYER2_DOWN_ROW_LANE_CAUSALITY "
        f"status={status} prompts=9 steps=41 disagreements={len(disagreement_summaries)} "
        f"rows={total_measured_rows} output={output_dir.relative_to(ROOT)}",
        flush=True,
    )
    return 0 if passed else 3


def verify_output(contract_path: Path, output_dir: Path) -> int:
    contract, contract_digest = load_bound_contract(contract_path)
    result_path = output_dir / "results.json"
    result = load_json(result_path)
    require(
        result["predeclared_contract"]["contract_sha256"] == contract_digest,
        "result contract digest differs",
    )
    require(result["status"] == "PASS_BOUND_LAYER2_DOWN_ROW_LANE_CAUSALITY_RECORDED", "result did not pass")
    require(result["coverage"]["prompt_count"] == 9, "prompt coverage differs")
    require(result["coverage"]["eligible_decode_step_count"] == 41, "step coverage differs")
    require(result["coverage"]["winner_disagreement_step_count"] == 10, "disagreement coverage differs")
    require(result["coverage"]["row_measurement_count"] == 10 * ROW_COUNT, "row coverage differs")
    require(result["taxonomy_observables_only"]["selected_policy_id"] is None, "policy selected")
    require(result["taxonomy_observables_only"]["taxonomy_label"] is None, "engineer taxonomy label set")
    require(not result["scope_guards"]["stage_transition_claimed"], "stage transition claimed")

    for item in result["winner_disagreements"]:
        artifacts = item["raw_artifacts"]
        row_path = ROOT / artifacts["row_measurements"]["path"]
        candidate_path = ROOT / artifacts["restored_candidate_token_ids"]["path"]
        relevant_path = ROOT / artifacts["relevant_output_rows"]["path"]
        lane_path = ROOT / artifacts["lane_contributions"]["path"]
        for name, path in (
            ("row_measurements", row_path),
            ("restored_candidate_token_ids", candidate_path),
            ("relevant_output_rows", relevant_path),
            ("lane_contributions", lane_path),
        ):
            require(path.is_file(), f"raw artifact missing: {name}")
            require(sha256_file(path) == artifacts[name]["sha256"], f"raw hash differs: {name}")
        rows = np.load(row_path, allow_pickle=False)
        candidates = np.load(candidate_path, allow_pickle=False)
        relevant = np.load(relevant_path, allow_pickle=False)
        lanes = np.load(lane_path, allow_pickle=False)
        require(rows.shape == (ROW_COUNT, len(ROW_ARRAY_COLUMNS)), "row array shape differs")
        require(candidates.shape == (ROW_COUNT,), "candidate array shape differs")
        require(relevant.shape == (item["relevant_row_count"],), "relevant row shape differs")
        require(lanes.shape == (item["relevant_row_count"], LANE_COUNT), "lane array shape differs")
        require(np.array_equal(rows[:, 0], np.arange(ROW_COUNT, dtype=np.float64)), "row indices differ")
        require(np.allclose(rows[:, 11] - rows[:, 8], rows[:, 12], rtol=0.0, atol=0.0), "restoration effects do not recalculate")
        recalculated = lanes.sum(axis=1, dtype=np.float64)
        declared = rows[relevant, 4]
        scale = np.maximum(1.0, np.maximum(np.abs(recalculated), np.abs(declared)))
        require(np.all(np.abs(recalculated - declared) <= 1e-12 * scale), "lane internal closure differs")
        measured = rows[relevant, 3]
        tolerance = 0.001953125 + 0.015625 * (
            np.abs(rows[relevant, 1]) + np.abs(rows[relevant, 2])
        )
        require(np.all(np.abs(recalculated - measured) <= tolerance), "lane measured closure differs")
        expected_relevant, _ = relevant_rows(
            rows[:, 3],
            rows[:, 12],
            candidates,
            item["full_w4_control"]["candidate_token_id"],
            item["bf16_control"]["candidate_token_id"],
        )
        require(np.array_equal(relevant, np.asarray(expected_relevant)), "relevant row selection differs")

    expected_sums = (output_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in expected_sums:
        digest, relative = line.split("  ", 1)
        path = output_dir / relative
        require(path.is_file() and sha256_file(path) == digest, f"SHA256SUMS differs: {relative}")
    require(contract["reporting_boundary"]["selected_policy_id"] is None, "contract selected policy")
    print(
        "ACE2_LAYER2_DOWN_ROW_LANE_CAUSALITY_VERIFY "
        f"status=PASS prompts=9 steps=41 disagreements=10 rows={10 * ROW_COUNT} "
        f"raw_files={len(result['winner_disagreements']) * 4}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--predeclare", type=Path)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.predeclare is not None:
        require(args.contract is None and args.output_dir is None, "predeclare mode is exclusive")
        path = args.predeclare.resolve()
        write_contract(path)
        print(
            "ACE2_LAYER2_DOWN_ROW_LANE_CAUSALITY_PREDECLARED "
            f"output={path.relative_to(ROOT)}",
            flush=True,
        )
        return 0
    require(args.contract is not None and args.output_dir is not None, "contract and output-dir required")
    if args.run:
        return run(args.contract.resolve(), args.output_dir.resolve())
    return verify_output(args.contract.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
