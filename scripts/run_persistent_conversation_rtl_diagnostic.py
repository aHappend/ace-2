#!/usr/bin/env python3
"""Run deterministic active-RTL turns with exact carried K/V state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation
from scripts import verify_full_chain_independent_oracle as oracle


TURN_PROMPTS = ("Define cache.", "Why?", "Example?")
TURN_IDS = ("definition", "follow-up", "example")
TURN_NAMES = ("turn one", "turn two", "turn three")
EXPECTED_LAYERS = 24
MIN_TURNS = 2
MAX_TURNS = 3
DEFAULT_TURNS = 2
MIN_GENERATED_TOKENS = 2
MAX_GENERATED_TOKENS = 8
DEFAULT_GENERATED_TOKENS = 4
DEFAULT_CONTEXT_TOKENS = backend.MAX_CONTEXT_TOKENS
THREE_TURN_CONTEXT_TOKENS = 64
ASSISTANT_CONTENT_MARKER = "ACE2_PERSISTENT_ASSISTANT_CONTENT_BOUNDARY"
REQUIRED_BYTE_COUNTS = (
    "kv_append_bytes_compared",
    "full_cache_bytes_compared",
    "quantized_layer_output_bytes_compared",
    "quantized_layer_output_scale_bytes_compared",
    "final_rmsnorm_bytes_compared",
    "final_rmsnorm_scale_bytes_compared",
    "full_vocabulary_logit_bytes_compared",
    "full_vocabulary_logit_scale_bytes_compared",
)


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def validate_generated_tokens(value: object) -> int:
    require(
        type(value) is int
        and MIN_GENERATED_TOKENS <= value <= MAX_GENERATED_TOKENS,
        "max-new-tokens must be an integer from 2 through 8",
    )
    return value


def validate_turns(value: object) -> int:
    require(
        type(value) is int and MIN_TURNS <= value <= MAX_TURNS,
        "turns must be an integer from 2 through 3",
    )
    return value


def configure_context_bound(turns: int) -> int:
    turns = validate_turns(turns)
    context_tokens = (
        DEFAULT_CONTEXT_TOKENS
        if turns == DEFAULT_TURNS
        else THREE_TURN_CONTEXT_TOKENS
    )
    backend.MAX_CONTEXT_TOKENS = context_tokens
    generation.MAX_CONTEXT_TOKENS = context_tokens
    return context_tokens


def render_turn(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    require(
        isinstance(token_ids, list)
        and token_ids
        and all(type(token) is int for token in token_ids),
        "chat template returned invalid token IDs",
    )
    return token_ids


def render_continuation(
    tokenizer: Any,
    previous_summary: dict[str, Any],
    assistant_replies: list[str],
) -> list[int]:
    require(assistant_replies, "assistant reply history is empty")
    require(
        all(ASSISTANT_CONTENT_MARKER not in reply for reply in assistant_replies),
        "assistant reply contains the continuation marker",
    )
    prompt_ids = previous_summary.get("prompt_token_ids")
    generated_ids = previous_summary.get("generated_token_ids")
    carried = previous_summary.get("carried_state")
    retained_ids = carried.get("output_context_token_ids") if isinstance(carried, dict) else None
    require(
        isinstance(prompt_ids, list)
        and all(type(token) is int for token in prompt_ids)
        and isinstance(generated_ids, list)
        and generated_ids
        and all(type(token) is int for token in generated_ids)
        and isinstance(retained_ids, list)
        and retained_ids == prompt_ids + generated_ids[:-1],
        "prior turn carried context differs from exact generated tokens",
    )
    marked_replies = assistant_replies[:-1] + [ASSISTANT_CONTENT_MARKER]
    rendered = tokenizer.apply_chat_template(
        conversation_messages(marked_replies),
        tokenize=False,
        add_generation_prompt=True,
    )
    require(
        isinstance(rendered, str) and rendered.count(ASSISTANT_CONTENT_MARKER) == 1,
        "chat template did not preserve the continuation marker",
    )
    suffix_text = rendered.partition(ASSISTANT_CONTENT_MARKER)[2]
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    require(
        isinstance(suffix_ids, list)
        and suffix_ids
        and all(type(token) is int for token in suffix_ids),
        "chat template returned an invalid continuation suffix",
    )
    return retained_ids + generated_ids[-1:] + suffix_ids


def conversation_messages(assistant_replies: list[str] | None = None) -> list[dict[str, str]]:
    assistant_replies = [] if assistant_replies is None else assistant_replies
    require(
        isinstance(assistant_replies, list)
        and len(assistant_replies) < len(TURN_PROMPTS)
        and all(isinstance(reply, str) for reply in assistant_replies),
        "assistant reply history is invalid",
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": TURN_PROMPTS[0]},
    ]
    for turn_index, assistant_reply in enumerate(assistant_replies, start=1):
        messages.extend(
            [
                {"role": "assistant", "content": assistant_reply},
                {"role": "user", "content": TURN_PROMPTS[turn_index]},
            ]
        )
    return messages


def validate_turn_summary(
    summary: dict[str, Any],
    turn_id: str,
    expected_generated_tokens: int,
) -> dict[str, Any]:
    expected_generated_tokens = validate_generated_tokens(expected_generated_tokens)
    require(isinstance(summary, dict), f"{turn_id} summary is absent")
    require(
        summary.get("status") == "PASS_STAGE1_PRODUCT_RTL_GENERATION_UNCHECKED",
        f"{turn_id} active-RTL status differs",
    )
    require(
        summary.get("software_transformer_or_logits_fallback") is False,
        f"{turn_id} used software transformer/logits fallback",
    )
    require(summary.get("layers") == EXPECTED_LAYERS, f"{turn_id} layer count differs")
    require(
        summary.get("max_new_tokens") == expected_generated_tokens,
        f"{turn_id} requested token count differs",
    )
    generated = summary.get("generated_token_ids")
    require(
        isinstance(generated, list)
        and len(generated) == expected_generated_tokens
        and all(type(token) is int for token in generated),
        f"{turn_id} did not generate exactly {expected_generated_tokens} tokens",
    )
    termination = summary.get("termination")
    require(
        isinstance(termination, dict)
        and termination.get("generated_token_count") == expected_generated_tokens,
        f"{turn_id} termination token count differs",
    )
    head_steps = summary.get("head_steps")
    require(
        isinstance(head_steps, list)
        and len(head_steps) == expected_generated_tokens
        and [
            step.get("generation_index")
            for step in head_steps
            if isinstance(step, dict)
        ]
        == list(range(expected_generated_tokens)),
        f"{turn_id} full-vocabulary head coverage differs",
    )
    carried = summary.get("carried_state")
    require(isinstance(carried, dict), f"{turn_id} carried-state evidence is absent")
    require(
        carried.get("layer_count") == EXPECTED_LAYERS,
        f"{turn_id} carried-state layer count differs",
    )
    require(
        carried.get("prefix_recomputed") is False,
        f"{turn_id} recomputed a retained prefix",
    )
    executions = summary.get("token_executions")
    require(isinstance(executions, list) and executions, f"{turn_id} executions are absent")
    start = carried.get("turn_start_absolute_position")
    require(type(start) is int and start >= 0, f"{turn_id} start position is invalid")
    require(
        carried.get("retained_positions") == start,
        f"{turn_id} retained-position evidence differs",
    )
    require(
        carried.get("turn_executed_positions") == len(executions),
        f"{turn_id} execution count differs",
    )
    for offset, execution in enumerate(executions):
        absolute_position = start + offset
        require(
            execution.get("absolute_position") == absolute_position,
            f"{turn_id} execution position is discontinuous",
        )
        layers = execution.get("layers")
        require(
            isinstance(layers, list) and len(layers) == EXPECTED_LAYERS,
            f"{turn_id} execution does not cover 24 layers",
        )
        for layer_id, layer in enumerate(layers):
            require(layer.get("layer_id") == layer_id, f"{turn_id} layer order differs")
            require(
                layer.get("cache_length_before") == absolute_position
                and layer.get("cache_length_after") == absolute_position + 1,
                f"{turn_id} layer {layer_id} cache continuity differs",
            )
    output_tokens = carried.get("output_context_token_ids")
    retained_tokens = carried.get("retained_context_token_ids")
    require(
        isinstance(retained_tokens, list)
        and all(type(token) is int for token in retained_tokens),
        f"{turn_id} retained token evidence is invalid",
    )
    require(
        isinstance(output_tokens, list)
        and output_tokens
        and all(type(token) is int for token in output_tokens),
        f"{turn_id} output token evidence is invalid",
    )
    require(
        len(retained_tokens) == start
        and output_tokens[:start] == retained_tokens,
        f"{turn_id} retained token evidence differs",
    )
    require(
        len(output_tokens) == carried.get("output_positions") == start + len(executions),
        f"{turn_id} output carried-state length differs",
    )
    return carried


def validate_conversation_evidence(
    summaries: list[dict[str, Any]],
    expected_generated_tokens: int,
) -> dict[str, Any]:
    expected_generated_tokens = validate_generated_tokens(expected_generated_tokens)
    turns = validate_turns(len(summaries))
    carried_states = [
        validate_turn_summary(
            summary,
            TURN_NAMES[turn_index],
            expected_generated_tokens,
        )
        for turn_index, summary in enumerate(summaries)
    ]
    require(carried_states[0]["retained_positions"] == 0, "turn one did not start fresh")
    transitions = []
    for turn_index in range(1, turns):
        previous = carried_states[turn_index - 1]
        current = carried_states[turn_index]
        turn_id = TURN_NAMES[turn_index]
        require(
            current["retained_positions"] > 0,
            f"{turn_id} has no retained prefix",
        )
        require(
            current["retained_context_token_ids"]
            == previous["output_context_token_ids"],
            f"{turn_id} retained state differs from prior turn output",
        )
        prompt_ids = summaries[turn_index].get("prompt_token_ids")
        require(
            isinstance(prompt_ids, list)
            and prompt_ids[: current["retained_positions"]]
            == current["retained_context_token_ids"],
            f"{turn_id} rendered prefix differs from retained state",
        )
        require(
            summaries[turn_index]["token_executions"][0]["absolute_position"]
            == current["retained_positions"],
            f"{turn_id} did not begin at the carried-state boundary",
        )
        transitions.append(
            {
                "from_turn": turn_index,
                "to_turn": turn_index + 1,
                "retained_layers": EXPECTED_LAYERS,
                "retained_positions": current["retained_positions"],
                "executed_positions": current["turn_executed_positions"],
                "prefix_recomputed": False,
            }
        )
    second = carried_states[1]
    return {
        "retained_layers": EXPECTED_LAYERS,
        "retained_positions": second["retained_positions"],
        "turn_two_executed_positions": second["turn_executed_positions"],
        "turns_checked": turns,
        "retained_transitions": len(transitions),
        "transitions": transitions,
        "generated_tokens_per_turn": expected_generated_tokens,
        "prefix_recomputed": False,
    }


def validate_two_turn_evidence(
    turn_one: dict[str, Any],
    turn_two: dict[str, Any],
    expected_generated_tokens: int,
) -> dict[str, Any]:
    return validate_conversation_evidence(
        [turn_one, turn_two],
        expected_generated_tokens,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object is invalid: {path}")
    return value


def cache_prefixes(
    summary: dict[str, Any],
    initial_prefixes: list[tuple[bytes, bytes]] | None = None,
) -> list[tuple[bytes, bytes]]:
    require(
        initial_prefixes is None or len(initial_prefixes) == EXPECTED_LAYERS,
        "initial cache-prefix layer count differs",
    )
    prefixes = [
        [
            bytearray(initial_prefixes[layer_id][0] if initial_prefixes else b""),
            bytearray(initial_prefixes[layer_id][1] if initial_prefixes else b""),
        ]
        for layer_id in range(EXPECTED_LAYERS)
    ]
    for execution in summary["token_executions"]:
        for layer_id, layer in enumerate(execution["layers"]):
            require(layer["layer_id"] == layer_id, "cache-prefix layer order differs")
            rtl = oracle.sealed_verifier.load_json(
                oracle.sealed_verifier.verify_file_record(layer["rtl_execution"])
            )
            prefixes[layer_id][0].extend(
                oracle.exact_artifact_bytes(rtl["kv_cache"]["rtl_observed_k"])
            )
            prefixes[layer_id][1].extend(
                oracle.exact_artifact_bytes(rtl["kv_cache"]["rtl_observed_v"])
            )
    return [(bytes(k_prefix), bytes(v_prefix)) for k_prefix, v_prefix in prefixes]


def validate_oracle_agreement(
    summary: dict[str, Any],
    agreement: dict[str, Any],
    turn_id: str,
) -> None:
    executions = summary["token_executions"]
    generated = summary["generated_token_ids"]
    require(
        agreement.get("positions_checked") == len(executions)
        and agreement.get("layers_checked") == len(executions) * EXPECTED_LAYERS,
        f"{turn_id} oracle position/layer coverage is incomplete",
    )
    require(
        agreement.get("full_vocabulary_head_steps_checked") == len(generated),
        f"{turn_id} oracle full-vocabulary head coverage is incomplete",
    )
    for field in REQUIRED_BYTE_COUNTS:
        require(
            type(agreement.get(field)) is int and agreement[field] > 0,
            f"{turn_id} oracle did not compare {field}",
        )
    require(
        agreement.get("integer_byte_mismatches") == 0,
        f"{turn_id} oracle reported integer-byte mismatches",
    )
    require(
        agreement.get("selected_token_mismatches") == 0,
        f"{turn_id} oracle reported selected-token mismatches",
    )
    require(
        agreement.get("full_vocabulary_logit_byte_mismatches") == 0,
        f"{turn_id} oracle reported full-vocabulary logit-byte mismatches",
    )


def aggregate_agreements(agreements: list[dict[str, Any]]) -> dict[str, Any]:
    turns = validate_turns(len(agreements))
    coverage = {
        field: sum(agreement[field] for agreement in agreements)
        for field in REQUIRED_BYTE_COUNTS
    }
    coverage.update(
        {
            "turns_checked": turns,
            "positions_checked": sum(
                agreement["positions_checked"] for agreement in agreements
            ),
            "layers_checked": sum(
                agreement["layers_checked"] for agreement in agreements
            ),
            "full_vocabulary_head_steps_checked": sum(
                agreement["full_vocabulary_head_steps_checked"]
                for agreement in agreements
            ),
            "integer_byte_mismatches": 0,
            "full_vocabulary_logit_byte_mismatches": 0,
            "selected_token_mismatches": 0,
        }
    )
    return coverage


def run_oracle_worker(
    output: Path,
    max_new_tokens: int,
    turns: int = DEFAULT_TURNS,
) -> dict[str, Any]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    turns = validate_turns(turns)
    context_tokens = configure_context_bound(turns)
    oracle_output = output / "independent-oracle"
    require(not oracle_output.exists(), "independent-oracle output must be fresh")
    snapshot = generation.resolve_snapshot()
    _record, tokenizer = generation.tokenizer_record(
        TURN_PROMPTS[0],
        backend.MAX_CONTEXT_TOKENS - max_new_tokens,
        max_new_tokens,
        snapshot,
    )
    manifest = {
        "tokenization": {"generation_bounds": {"max_new_tokens": max_new_tokens}},
        "model": {
            "path": (snapshot / "model.safetensors")
            .resolve()
            .relative_to(ROOT)
            .as_posix()
        },
        "adapter": {
            "path": backend.canonical.ADAPTER.resolve().relative_to(ROOT).as_posix()
        },
    }
    agreements = []
    initial_prefixes: list[tuple[bytes, bytes]] | None = None
    turn_records = []
    for turn_index in range(turns):
        directory = f"turn-{turn_index + 1:02d}"
        turn_id = TURN_IDS[turn_index]
        summary_path = output / directory / "run_summary.json"
        summary = load_json(summary_path)
        reference = oracle.reconstruct(
            manifest,
            output,
            tokenizer=tokenizer,
            prompt_ids=summary["prompt_token_ids"],
        )
        retained = summary["carried_state"]["retained_positions"]
        reference["positions"] = reference["positions"][retained:]
        agreement = oracle.compare_reference(
            reference,
            summary,
            initial_cache_prefixes=initial_prefixes,
        )
        agreement["full_vocabulary_logit_byte_mismatches"] = 0
        validate_oracle_agreement(summary, agreement, turn_id)
        agreements.append(agreement)
        turn_records.append(
            {
                "turn_id": turn_id,
                "summary": oracle.file_record(summary_path),
                "agreement": agreement,
            }
        )
        initial_prefixes = cache_prefixes(summary, initial_prefixes)
    result = {
        "schema": "ace2-persistent-conversation-independent-oracle-v1",
        "status": "PASS",
        "turns": turn_records,
        "coverage": aggregate_agreements(agreements),
        "backend_context_bound": context_tokens,
        "software_transformer_or_logits_fallback": False,
        "measurement_scope": "computer-local host oracle over retained RTL artifacts",
    }
    oracle_output.mkdir()
    backend.write_json(oracle_output / "result.json", result)
    return result


def run_oracle_process(
    output: Path,
    max_new_tokens: int,
    turns: int = DEFAULT_TURNS,
) -> dict[str, Any]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    turns = validate_turns(turns)
    stdout_path = output / "independent-oracle.stdout.log"
    stderr_path = output / "independent-oracle.stderr.log"
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(Path(__file__).resolve()),
        "--oracle-worker",
        "--output",
        str(output.resolve()),
        "--max-new-tokens",
        str(max_new_tokens),
        "--turns",
        str(turns),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    require(
        completed.returncode == 0,
        f"independent oracle failed with exit code {completed.returncode}",
    )
    result = load_json(output / "independent-oracle/result.json")
    require(result.get("status") == "PASS", "independent oracle did not publish PASS")
    require(
        result.get("software_transformer_or_logits_fallback") is False,
        "independent oracle result permits fallback",
    )
    return result


def run(
    output: Path,
    max_new_tokens: int,
    turns: int = DEFAULT_TURNS,
) -> dict[str, Any]:
    max_new_tokens = validate_generated_tokens(max_new_tokens)
    turns = validate_turns(turns)
    context_tokens = configure_context_bound(turns)
    require(not output.exists(), "diagnostic output must be a fresh path")
    snapshot = generation.resolve_snapshot()
    _record, tokenizer = generation.tokenizer_record(
        TURN_PROMPTS[0],
        backend.MAX_CONTEXT_TOKENS - max_new_tokens,
        max_new_tokens,
        snapshot,
    )
    summaries = []
    assistant_replies = []
    state = None
    for turn_index in range(turns):
        prompt_ids = (
            render_turn(tokenizer, conversation_messages())
            if turn_index == 0
            else render_continuation(tokenizer, summaries[-1], assistant_replies)
        )
        run_arguments = {
            "execution_mode": "stage1_product",
        }
        if state is not None:
            run_arguments["carried_state"] = state
        summary, state = backend.run_generation_with_state(
            output / f"turn-{turn_index + 1:02d}",
            snapshot,
            tokenizer,
            prompt_ids,
            max_new_tokens,
            **run_arguments,
        )
        summaries.append(summary)
        assistant_replies.append(summary["decoded_text"])
    coverage = validate_conversation_evidence(summaries, max_new_tokens)
    require(state.valid, "final carried state is invalid")
    oracle_result = run_oracle_process(output, max_new_tokens, turns)
    result = {
        "schema": "ace2-persistent-conversation-active-rtl-diagnostic-v3",
        "status": "PASS",
        "turns": [
            {
                "turn_id": TURN_IDS[turn_index],
                "summary": f"turn-{turn_index + 1:02d}/run_summary.json",
            }
            for turn_index in range(turns)
        ],
        "coverage": coverage
        | {
            "backend_context_bound": context_tokens,
            "independent_oracle": oracle_result["coverage"],
        },
        "independent_oracle": oracle.file_record(
            output / "independent-oracle/result.json"
        ),
        "software_transformer_or_logits_fallback": False,
        "measurement_kind": "computer-local host/orchestration and Icarus RTL simulation",
        "hardware_claim": False,
    }
    backend.write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_GENERATED_TOKENS,
    )
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--oracle-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        result = (
            run_oracle_worker(args.output, args.max_new_tokens, args.turns)
            if args.oracle_worker
            else run(args.output, args.max_new_tokens, args.turns)
        )
    except (DiagnosticError, backend.BackendError, OSError, RuntimeError, ValueError) as error:
        print(f"PERSISTENT_CONVERSATION_RTL_DIAGNOSTIC_FAIL detail={error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
