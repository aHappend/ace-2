#!/usr/bin/env python3
"""Run the exactly-once blinded V2 train/dev/holdout prompt separation scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V2_CONTRACT.json"
MANIFEST = (
    ROOT
    / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2"
    / "dataset_manifest.json"
)
TRAIN = (
    ROOT
    / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2"
    / "train.jsonl"
)
DEV = (
    ROOT
    / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1"
    / "dev.jsonl"
)
HOLDOUT = (
    ROOT
    / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1"
    / "holdout.jsonl"
)
SIDECAR = (
    ROOT
    / "research/raw/specification"
    / "qwen25-05b-instruct-bf16-lora-product-v2-cross-split-blinded-sidecar.json"
)
ATTESTATION = (
    ROOT
    / "research/raw/specification"
    / "qwen25-05b-instruct-bf16-lora-product-v2-cross-split-blinded-attestation.json"
)
OPERATOR_APPROVAL_REFERENCE = "Manager operator-answer decision 2026-08-07T20:18Z"
SCANNER_ID = "ace2-blinded-cross-split-scanner"
SCANNER_VERSION = "1.0.0"
METHOD = {
    "answer_fields_accessed": False,
    "human_exposure_to_evaluation_text": False,
    "method_id": "ace2-blinded-cross-split-leakage-v1",
    "near_duplicate_metric": "normalized-token-five-gram-jaccard",
    "normalization": "NFKC-casefold-collapse-whitespace",
    "prompt_only_access": True,
    "raw_prompts_or_fingerprints_emitted": False,
    "template_family_method": "normalized-lexical-skeleton-v1",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class SelectiveJsonCursor:
    """Decode only selected JSON strings; skip all other values as raw bytes."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.index = 0

    def whitespace(self) -> None:
        while self.index < len(self.payload) and self.payload[self.index] in b" \t\r\n":
            self.index += 1

    def expect(self, token: int) -> None:
        self.whitespace()
        if self.index >= len(self.payload) or self.payload[self.index] != token:
            raise RuntimeError("malformed JSON structure")
        self.index += 1

    def string_span(self) -> tuple[int, int]:
        self.whitespace()
        if self.index >= len(self.payload) or self.payload[self.index] != ord('"'):
            raise RuntimeError("expected JSON string")
        start = self.index
        self.index += 1
        while self.index < len(self.payload):
            byte = self.payload[self.index]
            if byte == ord('"'):
                self.index += 1
                return start, self.index
            if byte == ord("\\"):
                self.index += 1
                if self.index >= len(self.payload):
                    break
                escape = self.payload[self.index]
                if escape == ord("u"):
                    end = self.index + 5
                    if end > len(self.payload) or not re.fullmatch(
                        rb"[0-9a-fA-F]{4}", self.payload[self.index + 1 : end]
                    ):
                        raise RuntimeError("malformed JSON unicode escape")
                    self.index = end
                    continue
                if escape not in b'"\\/bfnrt':
                    raise RuntimeError("malformed JSON escape")
            elif byte < 0x20:
                raise RuntimeError("unescaped control byte in JSON string")
            self.index += 1
        raise RuntimeError("unterminated JSON string")

    def decode_span(self, span: tuple[int, int]) -> str:
        value = json.loads(self.payload[span[0] : span[1]].decode("utf-8"))
        if not isinstance(value, str):
            raise RuntimeError("expected decoded JSON string")
        return value

    def decode_string(self) -> str:
        return self.decode_span(self.string_span())

    def skip_value(self) -> None:
        self.whitespace()
        if self.index >= len(self.payload):
            raise RuntimeError("missing JSON value")
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
        end = self.index
        while end < len(self.payload) and self.payload[end] not in b" \t\r\n,]}":
            end += 1
        number = self.payload[self.index : end]
        if not re.fullmatch(
            rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", number
        ):
            raise RuntimeError("malformed JSON scalar")
        self.index = end

    def messages(self, evaluation_input: bool) -> list[dict[str, str]]:
        decoded: list[dict[str, str]] = []
        all_roles: list[str] = []
        self.expect(ord("["))
        self.whitespace()
        if self.index < len(self.payload) and self.payload[self.index] == ord("]"):
            raise RuntimeError("empty prompt message list")
        while True:
            self.expect(ord("{"))
            role_span: tuple[int, int] | None = None
            content_span: tuple[int, int] | None = None
            seen: set[str] = set()
            self.whitespace()
            while self.index >= len(self.payload) or self.payload[self.index] != ord("}"):
                key = self.decode_string()
                if key in seen:
                    raise RuntimeError("duplicate message field")
                seen.add(key)
                self.expect(ord(":"))
                if key == "role":
                    role_span = self.string_span()
                elif key == "content":
                    content_span = self.string_span()
                else:
                    self.skip_value()
                self.whitespace()
                if self.index < len(self.payload) and self.payload[self.index] == ord("}"):
                    break
                self.expect(ord(","))
            self.expect(ord("}"))
            if role_span is None or content_span is None:
                raise RuntimeError("prompt message lacks role or content")
            role = self.decode_span(role_span)
            if role not in {"system", "user", "assistant"}:
                raise RuntimeError("unexpected prompt role")
            all_roles.append(role)
            if evaluation_input or role != "assistant":
                decoded.append(
                    {
                        "content": normalized_text(self.decode_span(content_span)),
                        "role": role,
                    }
                )
            self.whitespace()
            if self.index < len(self.payload) and self.payload[self.index] == ord("]"):
                self.index += 1
                break
            self.expect(ord(","))
        if evaluation_input:
            if all_roles[0] != "system" or all_roles[-1] != "user":
                raise RuntimeError("evaluation input role boundary differs")
        elif all_roles != ["system", "user", "assistant"]:
            raise RuntimeError("training prompt role boundary differs")
        return decoded


def selective_record(payload: bytes, split: str) -> tuple[list[dict[str, str]], str]:
    cursor = SelectiveJsonCursor(payload)
    cursor.expect(ord("{"))
    prompt_messages: list[dict[str, str]] | None = None
    template_identifier: str | None = None
    seen: set[str] = set()
    cursor.whitespace()
    while cursor.index >= len(payload) or payload[cursor.index] != ord("}"):
        key = cursor.decode_string()
        if key in seen:
            raise RuntimeError("duplicate top-level field")
        seen.add(key)
        cursor.expect(ord(":"))
        if split == "train" and key == "messages":
            prompt_messages = cursor.messages(evaluation_input=False)
        elif split != "train" and key == "input_messages":
            prompt_messages = cursor.messages(evaluation_input=True)
        elif split == "train" and key == "template_id":
            template_identifier = normalized_text(cursor.decode_string())
        elif split != "train" and key == "template_family":
            template_identifier = normalized_text(cursor.decode_string())
        else:
            cursor.skip_value()
        cursor.whitespace()
        if cursor.index < len(payload) and payload[cursor.index] == ord("}"):
            break
        cursor.expect(ord(","))
    cursor.expect(ord("}"))
    cursor.whitespace()
    if cursor.index != len(payload):
        raise RuntimeError("trailing JSON data")
    if prompt_messages is None or template_identifier is None:
        raise RuntimeError("required blinded prompt field is absent")
    return prompt_messages, template_identifier


def digest(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + b"\x00" + payload).digest()


def prompt_digest(messages: list[dict[str, str]]) -> bytes:
    return digest(b"ace2/prompt/v1", canonical_json(messages))


def prompt_words(messages: list[dict[str, str]]) -> list[str]:
    text = " ".join(
        item["content"] for item in messages if item["role"] != "system"
    )
    return re.findall(r"[\w']+", normalized_text(text), flags=re.UNICODE)


def five_gram_digests(messages: list[dict[str, str]]) -> frozenset[bytes]:
    words = prompt_words(messages)
    grams: Iterable[tuple[str, ...]]
    if len(words) < 5:
        grams = [tuple(words)]
    else:
        grams = (tuple(words[index : index + 5]) for index in range(len(words) - 4))
    return frozenset(
        digest(b"ace2/five-gram/v1", "\x1f".join(gram).encode("utf-8"))
        for gram in grams
    )


def identifier_template_digest(identifier: str) -> bytes:
    skeleton = re.sub(r"[0-9]+", "<number>", normalized_text(identifier))
    skeleton = re.sub(r"[^\w<>]+", "-", skeleton, flags=re.UNICODE).strip("-")
    return digest(b"ace2/template-identifier/v1", skeleton.encode("utf-8"))


def lexical_template_digest(messages: list[dict[str, str]]) -> bytes:
    text = " ".join(
        item["content"] for item in messages if item["role"] != "system"
    )
    tokens = re.findall(r"[\w']+|[^\w\s]", normalized_text(text), flags=re.UNICODE)
    skeleton: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[+-]?[0-9]+(?:[.,][0-9]+)*", token):
            skeleton.append("<number>")
        elif any(character.isdigit() for character in token):
            skeleton.append("<alphanumeric>")
        else:
            skeleton.append(token)
    return digest(b"ace2/lexical-template/v1", " ".join(skeleton).encode("utf-8"))


@dataclass(frozen=True)
class BlindedRecord:
    prompt: bytes
    five_grams: frozenset[bytes]
    identifier_template: bytes
    lexical_template: bytes


def scan_split(path: Path, split: str, expected_count: int) -> list[BlindedRecord]:
    records: list[BlindedRecord] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.rstrip(b"\r\n")
            if not line:
                raise RuntimeError("blank JSONL record")
            messages, template_identifier = selective_record(line, split)
            records.append(
                BlindedRecord(
                    prompt=prompt_digest(messages),
                    five_grams=five_gram_digests(messages),
                    identifier_template=identifier_template_digest(template_identifier),
                    lexical_template=lexical_template_digest(messages),
                )
            )
    if len(records) != expected_count:
        raise RuntimeError(f"{split} record count differs")
    return records


def set_commitment(domain: bytes, values: Iterable[bytes]) -> dict[str, Any]:
    unique = sorted(set(values))
    framed = b"".join(len(value).to_bytes(4, "big") + value for value in unique)
    return {
        "count": len(unique),
        "set_sha256": sha256_bytes(domain + b"\x00" + framed),
    }


def split_commitments(records: list[BlindedRecord]) -> dict[str, Any]:
    return {
        "five_gram_fingerprints": set_commitment(
            b"ace2/five-gram-set/v1",
            (fingerprint for record in records for fingerprint in record.five_grams),
        ),
        "lexical_template_fingerprints": set_commitment(
            b"ace2/lexical-template-set/v1",
            (record.lexical_template for record in records),
        ),
        "normalized_prompt_fingerprints": set_commitment(
            b"ace2/prompt-set/v1", (record.prompt for record in records)
        ),
        "record_count": len(records),
        "template_identifier_fingerprints": set_commitment(
            b"ace2/template-identifier-set/v1",
            (record.identifier_template for record in records),
        ),
    }


def exact_collision_count(
    train: list[BlindedRecord], evaluation: list[BlindedRecord]
) -> int:
    train_counts = Counter(record.prompt for record in train)
    evaluation_counts = Counter(record.prompt for record in evaluation)
    return sum(
        count * evaluation_counts[fingerprint]
        for fingerprint, count in train_counts.items()
    )


def jaccard(left: frozenset[bytes], right: frozenset[bytes]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def near_collision_result(
    train: list[BlindedRecord],
    evaluation: list[BlindedRecord],
    threshold: float,
) -> tuple[int, float]:
    collisions = 0
    maximum = 0.0
    for left in train:
        for right in evaluation:
            score = jaccard(left.five_grams, right.five_grams)
            maximum = max(maximum, score)
            if score > threshold:
                collisions += 1
    return collisions, round(maximum, 12)


def template_collision_count(
    train: list[BlindedRecord], evaluation: list[BlindedRecord]
) -> int:
    return sum(
        1
        for left in train
        for right in evaluation
        if left.identifier_template == right.identifier_template
        or left.lexical_template == right.lexical_template
    )


def json_payload(value: dict[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def write_companion(path: Path) -> None:
    companion = path.with_suffix(".sha256")
    write_new(companion, f"{sha256_file(path)}  {path.name}\n".encode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator-approval-reference",
        required=True,
        choices=[OPERATOR_APPROVAL_REFERENCE],
    )
    args = parser.parse_args()

    artifacts = [
        SIDECAR,
        SIDECAR.with_suffix(".sha256"),
        ATTESTATION,
        ATTESTATION.with_suffix(".sha256"),
    ]
    if any(path.exists() for path in artifacts):
        raise RuntimeError("blinded scanner artifacts already exist; refusing a repeat scan")

    contract = load_json(CONTRACT)
    manifest = load_json(MANIFEST)
    leakage = contract["leakage_qualification"]
    if leakage != manifest["cross_split_leakage_qualification"]:
        raise RuntimeError("contract and manifest leakage gates differ")
    if leakage["required_method"] != METHOD:
        raise RuntimeError("frozen blinded method differs")
    expected_hashes = leakage["required_input_bindings"]
    source_paths = {"dev": DEV, "holdout": HOLDOUT, "train": TRAIN}
    source_counts = {"dev": 56, "holdout": 56, "train": 216}
    source_bindings: dict[str, dict[str, Any]] = {}
    for split, path in source_paths.items():
        observed_hash = sha256_file(path)
        if observed_hash != expected_hashes[f"{split}_sha256"]:
            raise RuntimeError(f"{split} source hash differs")
        source_bindings[split] = {
            "count": source_counts[split],
            "path": str(path.relative_to(ROOT)),
            "sha256": observed_hash,
        }

    train = scan_split(TRAIN, "train", source_counts["train"])
    dev = scan_split(DEV, "dev", source_counts["dev"])
    holdout = scan_split(HOLDOUT, "holdout", source_counts["holdout"])
    evaluation = dev + holdout
    threshold = float(
        leakage["required_results"][
            "maximum_train_to_evaluation_five_gram_jaccard"
        ]
    )
    exact_collisions = exact_collision_count(train, evaluation)
    near_collisions, maximum_jaccard = near_collision_result(
        train, evaluation, threshold
    )
    template_collisions = template_collision_count(train, evaluation)
    status = (
        "PASS"
        if exact_collisions == 0
        and near_collisions == 0
        and template_collisions == 0
        else "NO-GO"
    )
    scanner_path = Path(__file__).resolve()
    scanner_binding = {
        "path": str(scanner_path.relative_to(ROOT)),
        "sha256": sha256_file(scanner_path),
    }
    collision_counts = {
        "exact_normalized_prompt_collision_count": exact_collisions,
        "near_duplicate_collision_count": near_collisions,
        "template_family_collision_count": template_collisions,
    }
    sidecar = {
        "algorithm": {
            "field_decoder": "selective-json-field-decoder-v1",
            "method": METHOD,
            "near_duplicate_threshold_strictly_greater_than": threshold,
            "salt_policy": {
                "mode": "domain-separated-unsalted-sha256-set-commitments",
                "per_record_fingerprints_emitted": False,
                "secret_or_random_salt": False,
            },
            "scanner_id": SCANNER_ID,
            "version": SCANNER_VERSION,
        },
        "aggregate_collision_counts": collision_counts,
        "fingerprint_sets": {
            "dev": split_commitments(dev),
            "holdout": split_commitments(holdout),
            "train": split_commitments(train),
        },
        "maximum_train_to_evaluation_five_gram_jaccard": maximum_jaccard,
        "scanner": scanner_binding,
        "schema_version": 1,
        "source_bindings": source_bindings,
        "status": status,
    }
    write_new(SIDECAR, json_payload(sidecar))
    write_companion(SIDECAR)

    results = {
        **collision_counts,
        "maximum_train_to_evaluation_five_gram_jaccard": maximum_jaccard,
        "raw_prompts_or_fingerprints_included": False,
        "status": status,
    }
    attestation = {
        "input_bindings": {
            split: {"count": binding["count"], "sha256": binding["sha256"]}
            for split, binding in source_bindings.items()
        },
        "method": METHOD,
        "producer": {
            "fresh_isolated_non_training_session": True,
            "network_access_used": False,
            "operator_approval_reference": args.operator_approval_reference,
            "role": "independent_blinded_split_auditor",
            "training_executed": False,
        },
        "results": results,
        "scanner": scanner_binding,
        "schema_version": 1,
        "sidecar": {
            "path": str(SIDECAR.relative_to(ROOT)),
            "sha256": sha256_file(SIDECAR),
        },
    }
    write_new(ATTESTATION, json_payload(attestation))
    write_companion(ATTESTATION)
    print(
        json.dumps(
            {
                "attestation_sha256": sha256_file(ATTESTATION),
                "collision_counts": collision_counts,
                "maximum_train_to_evaluation_five_gram_jaccard": maximum_jaccard,
                "sidecar_sha256": sha256_file(SIDECAR),
                "status": status,
            },
            sort_keys=True,
        )
    )
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
