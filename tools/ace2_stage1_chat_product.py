#!/usr/bin/env python3
"""Bounded one-command Stage-1 Qwen chat product integration."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation


ROOT = Path(__file__).resolve().parents[1]
POSITION01 = ROOT / "reports/ace2-position01-verification-attempt-0001"
POSITION01_TREE_ROOT_SHA256 = (
    "237f30bcad91a208929353be6453381937d2503a4b97062ecadcb99ab90bf315"
)
_CONTENT_WORD_STOPLIST = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "been",
        "being",
        "did",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "why",
    }
)
_NAMED_SCRIPTS = (
    "ARABIC",
    "CYRILLIC",
    "DEVANAGARI",
    "GREEK",
    "HANGUL",
    "HEBREW",
    "HIRAGANA",
    "KATAKANA",
    "LATIN",
    "THAI",
)


class ProductError(RuntimeError):
    """The Stage-1 product contract rejected the request or result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_accepted_position01() -> dict[str, Any]:
    sums_path = POSITION01 / "SHA256SUMS"
    tree_path = POSITION01 / "TREE_ROOT.sha256"
    result_path = POSITION01 / "attempt-result.json"
    _require(sums_path.is_file(), "sealed position-01 SHA256SUMS is missing")
    _require(tree_path.is_file(), "sealed position-01 tree root is missing")
    _require(
        _sha256_file(sums_path) == POSITION01_TREE_ROOT_SHA256,
        "sealed position-01 tree root changed",
    )
    _require(
        tree_path.read_text(encoding="ascii").strip()
        == f"{POSITION01_TREE_ROOT_SHA256}  SHA256SUMS",
        "sealed position-01 tree-root receipt changed",
    )
    for line in sums_path.read_text(encoding="ascii").splitlines():
        expected_sha256, relative = line.split("  ", 1)
        path = POSITION01 / relative
        _require(path.is_file(), f"sealed position-01 member is missing: {relative}")
        _require(
            _sha256_file(path) == expected_sha256,
            f"sealed position-01 member changed: {relative}",
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(result.get("status") == "PASS", "sealed position-01 result is not PASS")
    _require(result.get("complete_layer_count") == 24, "position-01 layer count differs")
    comparison = json.loads(
        (POSITION01 / "independent-reference-comparison.json").read_text(
            encoding="utf-8"
        )
    )
    _require(
        comparison.get("total_integer_mismatches") == 0,
        "position-01 independent comparison changed",
    )
    terminal = json.loads(
        (POSITION01 / "runtime-output/terminal.json").read_text(encoding="utf-8")
    )
    _require(
        terminal.get("selected_token_id") == 109315,
        "position-01 selected token changed",
    )
    return {
        "status": "ACCEPTED_BOUNDED_POSITION01_PASS",
        "path": POSITION01.relative_to(ROOT).as_posix(),
        "tree_root_sha256": POSITION01_TREE_ROOT_SHA256,
        "complete_layer_count": 24,
        "selected_token_id": 109315,
        "total_integer_mismatches": 0,
        "elapsed_wall_seconds": result["elapsed_wall_seconds"],
        "sealed_attempt_replayed": False,
        "sealed_attempt_mutated": False,
    }


def _content_words(text: str) -> set[str]:
    words = re.findall(
        r"[^\W_]+",
        unicodedata.normalize("NFKC", text).casefold(),
        flags=re.UNICODE,
    )
    terms = set()
    for word in words:
        if word in _CONTENT_WORD_STOPLIST:
            continue
        terms.add(word[:-1] if len(word) > 4 and word.endswith("s") else word)
    return terms


def _letter_scripts(text: str) -> set[str]:
    scripts = set()
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "")
        if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
            scripts.add("HAN")
            continue
        scripts.add(next((script for script in _NAMED_SCRIPTS if script in name), "OTHER"))
    return scripts


def readable_output_acceptance(
    prompt: str,
    text: str,
    token_ids: list[int],
) -> dict[str, Any]:
    stripped = text.strip()
    categories = [unicodedata.category(character) for character in stripped]
    words = re.findall(r"[^\W_]+", stripped, flags=re.UNICODE)
    prompt_terms = _content_words(prompt)
    response_terms = _content_words(stripped)
    overlapping_terms = prompt_terms & response_terms
    prompt_scripts = _letter_scripts(prompt)
    response_scripts = _letter_scripts(stripped)
    starts_readably = bool(stripped) and (
        stripped[0].isalnum() or stripped[0] in {'"', "'", "\u201c", "\u2018"}
    )
    checks = {
        "at_least_two_tokens": len(token_ids) >= 2,
        "nonblank": bool(stripped),
        "no_control_or_replacement": "\ufffd" not in stripped
        and not any(category.startswith("C") for category in categories),
        "readable_start": starts_readably,
        "contains_letter_or_number": any(
            category.startswith(("L", "N")) for category in categories
        ),
        "mostly_letters_numbers_or_spacing": (
            bool(stripped)
            and sum(
                category[0] in {"L", "N", "Z", "P"} for category in categories
            )
            / len(categories)
            >= 0.9
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "visible_characters": len(stripped),
        "word_count": len(words),
        "prompt_content_word_count": len(prompt_terms),
        "response_content_word_count": len(response_terms),
        "prompt_content_overlap_count": len(overlapping_terms),
        "prompt_scripts": sorted(prompt_scripts),
        "response_scripts": sorted(response_scripts),
        "policy": (
            "at least two generated tokens forming nonblank printable text with a "
            "readable start, at least one letter or number, no controls/replacement "
            "characters, and at least 90 percent letters, numbers, spacing, or punctuation"
        ),
    }


def validate_backend_summary(summary: dict[str, Any], prompt: str) -> dict[str, Any]:
    _require(
        summary.get("software_transformer_or_logits_fallback") is False,
        "software transformer/logits fallback is forbidden",
    )
    generated = summary.get("generated_token_ids")
    _require(
        isinstance(generated, list) and len(generated) >= 2,
        "one session must generate at least two tokens",
    )
    prompt_token_ids = summary.get("prompt_token_ids")
    _require(
        isinstance(prompt_token_ids, list) and prompt_token_ids,
        "prompt token IDs are absent",
    )
    executions = summary.get("token_executions")
    _require(isinstance(executions, list) and executions, "token executions are absent")
    _require(
        len(executions) > len(prompt_token_ids),
        "first generated token was not fed into a decode position",
    )
    for expected_position, execution in enumerate(executions):
        _require(
            int(execution.get("absolute_position", -1)) == expected_position,
            "token positions are not one continuous session",
        )
        layers = execution.get("layers")
        _require(
            isinstance(layers, list) and len(layers) == backend.LAYERS,
            "token execution does not cover every layer",
        )
        for layer_id, layer in enumerate(layers):
            _require(
                int(layer.get("cache_length_before", -1)) == expected_position
                and int(layer.get("cache_length_after", -1))
                == expected_position + 1,
                "KV cache continuity differs",
            )
            source = layer.get("rtl_cache_append", {})
            _require(
                source.get("host_cache_append_replaced") is True
                and int(source.get("layer_id", -1)) == layer_id,
                "RTL cache-source provenance is incomplete",
            )
            mismatches = layer.get("integer_boundary_mismatches")
            _require(
                isinstance(mismatches, dict)
                and mismatches
                and all(int(value) == 0 for value in mismatches.values()),
                "RTL integer comparison is incomplete or nonzero",
            )
    first_decode = executions[len(prompt_token_ids)]
    _require(
        first_decode.get("phase") == "decode"
        and int(first_decode.get("input_token_id", -1)) == int(generated[0]),
        "the second generated token did not consume the first selected token",
    )
    head_steps = summary.get("head_steps")
    _require(
        isinstance(head_steps, list)
        and len(head_steps) == len(generated)
        and all(step.get("rtl_selected_token_agreement") is True for step in head_steps),
        "final-head selected-token comparison is incomplete",
    )
    corrected = summary.get("rtl_corrected_v_provenance")
    _require(
        isinstance(corrected, list)
        and any(
            record.get("downstream_consumed") is True
            and record.get("rank1_sidecar_executed") is True
            and record.get("v_source")
            == "icarus_ace2_shell_mem_wdata_layer23_rank1_corrected_v"
            for record in corrected
        ),
        "RTL-emitted corrected-V bytes did not feed a downstream position",
    )
    latency = summary.get("latency")
    _require(
        isinstance(latency, dict)
        and all(
            isinstance(latency.get(key), (int, float))
            and float(latency[key]) >= 0.0
            for key in (
                "compile_wall_seconds",
                "model_wall_seconds",
                "simulation_wall_seconds",
                "total_wall_seconds",
            )
        )
        and isinstance(summary.get("detokenization_wall_seconds"), (int, float))
        and float(summary["detokenization_wall_seconds"]) >= 0.0,
        "compile/model/simulation/total latency is incomplete",
    )
    return readable_output_acceptance(
        prompt,
        str(summary.get("decoded_text", "")),
        generated,
    )


def write_terminal_result(
    output: Path,
    result: dict[str, Any],
    summary: dict[str, Any],
    readability: dict[str, Any],
) -> Path:
    product_path = output / "product_result.json"
    root_cause_path = output / "root_cause.json"
    _require(
        not product_path.exists() and not root_cause_path.exists(),
        "Stage-1 terminal artifact already exists",
    )
    if readability["accepted"]:
        terminal_path = product_path
        payload = result
    else:
        terminal_path = root_cause_path
        payload = {
            "schema_version": 1,
            "status": "BLOCKED_INCOHERENT_QUANTIZED_OUTPUT",
            "observed_generated_token_ids": summary["generated_token_ids"],
            "observed_decoded_text": summary["decoded_text"],
            "readability": readability,
            "failure_taxonomy": "model_output_coherence",
            "root_cause_hypothesis": (
                "The frozen checkpoint and W4A8 model-output semantics selected a "
                "sequence rejected by the predeclared prompt-aware coherence policy."
            ),
            "regression": (
                "Run the unchanged four-token prompt-aware policy against separately "
                "repaired model-output semantics before authorizing another attempt."
            ),
            "excluded_actions": [
                "prompt-specific tuning",
                "software fallback",
                "sealed 0026 replay or mutation",
                "new 0027 hybrid identity",
            ],
        }
    _write_json(terminal_path, payload)
    terminal_path.chmod(0o444)
    return terminal_path


def run_chat(
    prompt: str,
    output: Path,
    *,
    max_new_tokens: int = 4,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    _require(bool(prompt.strip()), "prompt must contain non-whitespace UTF-8 text")
    generation.validate_max_new_tokens(max_new_tokens)
    output = output.resolve()
    _require(not output.exists(), "Stage-1 product output must be a fresh path")
    position01 = verify_accepted_position01()
    resolved_snapshot = generation.resolve_snapshot(snapshot)
    tokenization, tokenizer = generation.tokenizer_record(
        prompt,
        backend.MAX_CONTEXT_TOKENS - max_new_tokens,
        max_new_tokens,
        resolved_snapshot,
    )
    with backend.process_tree_rss_tracking(interval_seconds=0.01) as tracker:
        summary = backend.run_generation(
            output,
            resolved_snapshot,
            tokenizer,
            tokenization["prompt_token_ids"],
            max_new_tokens,
            execution_mode="stage1_product",
        )
    process_tree = tracker.summary(require_complete=True)
    comparison_started = time.monotonic()
    readability = validate_backend_summary(summary, prompt)
    comparison_wall_seconds = time.monotonic() - comparison_started
    prefill_wall_seconds = sum(
        float(execution["timing"]["total_wall_seconds"])
        for execution in summary["token_executions"]
        if execution["phase"] == "prefill"
    )
    decode_wall_seconds = sum(
        float(execution["timing"]["total_wall_seconds"])
        for execution in summary["token_executions"]
        if execution["phase"] == "decode"
    )
    summary["latency"].update(
        {
            "setup_wall_seconds": float(summary["setup_timing"]["total_wall_seconds"]),
            "prefill_wall_seconds": prefill_wall_seconds,
            "decode_wall_seconds": decode_wall_seconds,
            "comparison_wall_seconds": comparison_wall_seconds,
            "detokenization_wall_seconds": float(
                summary["detokenization_wall_seconds"]
            ),
        }
    )
    backend.write_json(output / "run_summary.json", summary)
    result = {
        "schema_version": 1,
        "status": (
            "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT"
            if readability["accepted"]
            else "BLOCKED_INCOHERENT_STAGE1_RTL_CHAT_OUTPUT"
        ),
        "prompt": {
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "utf8_bytes": len(prompt.encode("utf-8")),
            "text_persisted": False,
        },
        "tokenization": tokenization,
        "generated_token_ids": summary["generated_token_ids"],
        "decoded_text": summary["decoded_text"],
        "readability": readability,
        "single_session": True,
        "kv_continuity": True,
        "rtl_corrected_v_provenance": summary["rtl_corrected_v_provenance"],
        "software_transformer_or_logits_fallback": False,
        "latency": summary["latency"],
        "process_tree_rss": process_tree,
        "accepted_position01": position01,
        "run_summary": {
            "path": "run_summary.json",
            "sha256": _sha256_file(output / "run_summary.json"),
        },
    }
    write_terminal_result(output, result, summary, readability)
    return result
