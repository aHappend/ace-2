#!/usr/bin/env python3
"""Extract only aggregate fields from the immutable V1 dev-selection artifact.

The source artifact embeds generated responses. This parser decodes selected
root/candidate aggregate fields and skips every response value as raw JSON
bytes, so no prompt, answer, response, scored row, or per-case semantics are
emitted or retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v1"
    / "attempt-0001/dev_selection.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Cursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.index = 0

    def whitespace(self) -> None:
        while self.index < len(self.payload) and self.payload[self.index] in b" \t\r\n":
            self.index += 1

    def expect(self, token: int) -> None:
        self.whitespace()
        require(self.index < len(self.payload) and self.payload[self.index] == token, "malformed JSON structure")
        self.index += 1

    def string_span(self) -> tuple[int, int]:
        self.whitespace()
        require(self.index < len(self.payload) and self.payload[self.index] == ord('"'), "expected JSON string")
        start = self.index
        self.index += 1
        while self.index < len(self.payload):
            byte = self.payload[self.index]
            if byte == ord('"'):
                self.index += 1
                return start, self.index
            if byte == ord("\\"):
                self.index += 1
                require(self.index < len(self.payload), "unterminated JSON escape")
                if self.payload[self.index] == ord("u"):
                    end = self.index + 5
                    require(
                        end <= len(self.payload)
                        and re.fullmatch(rb"[0-9a-fA-F]{4}", self.payload[self.index + 1 : end]),
                        "malformed JSON unicode escape",
                    )
                    self.index = end
                    continue
            self.index += 1
        raise RuntimeError("unterminated JSON string")

    def decode_string(self) -> str:
        start, end = self.string_span()
        value = json.loads(self.payload[start:end].decode("utf-8"))
        require(isinstance(value, str), "decoded JSON string differs")
        return value

    def skip_value(self) -> None:
        self.whitespace()
        require(self.index < len(self.payload), "missing JSON value")
        token = self.payload[self.index]
        if token == ord('"'):
            self.string_span()
            return
        if token == ord("{"):
            self.index += 1
            self.whitespace()
            if self.index < len(self.payload) and self.payload[self.index] == ord("}"):
                self.index += 1
                return
            while True:
                self.string_span()
                self.expect(ord(":"))
                self.skip_value()
                self.whitespace()
                if self.index < len(self.payload) and self.payload[self.index] == ord("}"):
                    self.index += 1
                    return
                self.expect(ord(","))
        if token == ord("["):
            self.index += 1
            self.whitespace()
            if self.index < len(self.payload) and self.payload[self.index] == ord("]"):
                self.index += 1
                return
            while True:
                self.skip_value()
                self.whitespace()
                if self.index < len(self.payload) and self.payload[self.index] == ord("]"):
                    self.index += 1
                    return
                self.expect(ord(","))
        for literal in (b"true", b"false", b"null"):
            if self.payload.startswith(literal, self.index):
                self.index += len(literal)
                return
        self.number()

    def number(self) -> int | float:
        self.whitespace()
        end = self.index
        while end < len(self.payload) and self.payload[end] not in b" \t\r\n,]}":
            end += 1
        encoded = self.payload[self.index:end]
        require(
            bool(re.fullmatch(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", encoded)),
            "malformed JSON number",
        )
        self.index = end
        text = encoded.decode("ascii")
        return float(text) if any(character in text for character in ".eE") else int(text)

    def scalar(self) -> Any:
        self.whitespace()
        token = self.payload[self.index]
        if token == ord('"'):
            return self.decode_string()
        for encoded, value in ((b"true", True), (b"false", False), (b"null", None)):
            if self.payload.startswith(encoded, self.index):
                self.index += len(encoded)
                return value
        return self.number()


def numeric_object(cursor: Cursor) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    cursor.expect(ord("{"))
    cursor.whitespace()
    if cursor.index < len(cursor.payload) and cursor.payload[cursor.index] == ord("}"):
        cursor.index += 1
        return values
    while True:
        key = cursor.decode_string()
        cursor.expect(ord(":"))
        value = cursor.number()
        values[key] = value
        cursor.whitespace()
        if cursor.payload[cursor.index] == ord("}"):
            cursor.index += 1
            return values
        cursor.expect(ord(","))


AGGREGATE_SCALARS = {
    "critical_safety_failures",
    "epoch",
    "hard_pass_count",
    "minimum_category_pass_rate",
    "response_count",
    "validation_assistant_token_loss",
}


def aggregate_object(cursor: Cursor) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    cursor.expect(ord("{"))
    cursor.whitespace()
    while cursor.payload[cursor.index] != ord("}"):
        key = cursor.decode_string()
        cursor.expect(ord(":"))
        if key in {"category_pass_rates", "category_passes"}:
            aggregate[key] = numeric_object(cursor)
        elif key in AGGREGATE_SCALARS:
            aggregate[key] = cursor.scalar()
        else:
            cursor.skip_value()
        cursor.whitespace()
        if cursor.payload[cursor.index] == ord("}"):
            break
        cursor.expect(ord(","))
        cursor.whitespace()
    cursor.expect(ord("}"))
    required = {
        "category_pass_rates",
        "category_passes",
        "critical_safety_failures",
        "hard_pass_count",
        "minimum_category_pass_rate",
        "response_count",
        "validation_assistant_token_loss",
    }
    require(required <= set(aggregate), f"aggregate fields missing: {sorted(required - set(aggregate))}")
    return aggregate


def candidates_object(cursor: Cursor) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    cursor.expect(ord("{"))
    cursor.whitespace()
    while cursor.payload[cursor.index] != ord("}"):
        epoch = cursor.decode_string()
        cursor.expect(ord(":"))
        candidate = aggregate_object(cursor)
        require(int(candidate["epoch"]) == int(epoch), f"candidate epoch differs: {epoch}")
        candidates[epoch] = candidate
        cursor.whitespace()
        if cursor.payload[cursor.index] == ord("}"):
            break
        cursor.expect(ord(","))
        cursor.whitespace()
    cursor.expect(ord("}"))
    return candidates


def extract_source() -> dict[str, Any]:
    payload = SOURCE.read_bytes()
    cursor = Cursor(payload)
    root: dict[str, Any] = {}
    cursor.expect(ord("{"))
    cursor.whitespace()
    while cursor.payload[cursor.index] != ord("}"):
        key = cursor.decode_string()
        cursor.expect(ord(":"))
        if key == "candidates":
            root[key] = candidates_object(cursor)
        elif key == "source":
            root[key] = aggregate_object(cursor)
        elif key in {"selected_epoch", "status"}:
            root[key] = cursor.scalar()
        else:
            cursor.skip_value()
        cursor.whitespace()
        if cursor.payload[cursor.index] == ord("}"):
            break
        cursor.expect(ord(","))
        cursor.whitespace()
    cursor.expect(ord("}"))
    cursor.whitespace()
    require(cursor.index == len(payload), "trailing JSON data")
    require(set(root["candidates"]) == {"1", "2", "3"}, "V1 candidate epochs differ")
    return root


def build_report() -> dict[str, Any]:
    extracted = extract_source()
    best_epoch, best = min(
        extracted["candidates"].items(),
        key=lambda item: (-int(item[1]["hard_pass_count"]), int(item[0])),
    )
    return {
        "baseline_id": "qwen25-lora-v1-aggregate-dev-baseline-v1",
        "best_candidate": {
            "category_passes": best["category_passes"],
            "critical_safety_failures": best["critical_safety_failures"],
            "epoch": int(best_epoch),
            "hard_pass_count": best["hard_pass_count"],
            "response_count": best["response_count"],
        },
        "candidates": extracted["candidates"],
        "evidence_boundary": {
            "aggregate_fields_decoded": True,
            "per_case_fields_decoded": False,
            "raw_prompts_answers_or_responses_emitted": False,
            "responses_value_skipped_as_raw_json_bytes": True,
        },
        "schema_version": 1,
        "selection": {
            "selected_epoch": extracted["selected_epoch"],
            "status": extracted["status"],
        },
        "source": extracted["source"],
        "source_artifact": {
            "bytes": SOURCE.stat().st_size,
            "path": relative(SOURCE),
            "sha256": sha256_file(SOURCE),
        },
        "tool_binding": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_report(path: Path) -> None:
    payload = canonical_bytes(build_report())
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write(path, payload)
    atomic_write(sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    print(f"WROTE {relative(path)} sha256={digest}")


def verify_report(path: Path) -> None:
    expected = canonical_bytes(build_report())
    observed = path.read_bytes()
    require(observed == expected, "V1 aggregate baseline differs from selective recomputation")
    digest = hashlib.sha256(observed).hexdigest()
    require(sidecar_path(path).read_text(encoding="ascii").split() == [digest, path.name], "V1 aggregate sidecar differs")
    report = json.loads(observed)
    require(report["best_candidate"]["hard_pass_count"] == 40, "V1 aggregate best is not 40/56")
    require(report["best_candidate"]["response_count"] == 56, "V1 aggregate response count differs")
    print(f"QWEN25_LORA_V1_AGGREGATE_BASELINE_VERIFIED sha256={digest}")


def main() -> None:
    args = parse_args()
    if args.output is not None:
        write_report(args.output.resolve())
    else:
        verify_report(args.verify.resolve())


if __name__ == "__main__":
    main()
