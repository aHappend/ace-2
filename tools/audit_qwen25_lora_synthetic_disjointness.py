#!/usr/bin/env python3
"""Blindly prove synthetic diagnostics are disjoint from sealed eval prompts.

The evaluation JSONL files are handled by the existing selective decoder. Only
normalized prompt fingerprints are retained in memory, and this tool emits no
raw prompt, answer, response, per-record fingerprint, or collision pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from run_qwen25_05b_bf16_lora_product_v2_blinded_cross_split_scan import (
    BlindedRecord,
    METHOD,
    exact_collision_count,
    five_gram_digests,
    identifier_template_digest,
    lexical_template_digest,
    near_collision_result,
    prompt_digest,
    scan_split,
    split_commitments,
    template_collision_count,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_fixtures.json"
DEV = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dev.jsonl"
HOLDOUT = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/holdout.jsonl"
SOURCE_MANIFEST = (
    ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dataset_manifest.json"
)
BLINDED_SCANNER = ROOT / "tools/run_qwen25_05b_bf16_lora_product_v2_blinded_cross_split_scan.py"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {relative(path)}")
    return value


def fixture_records(fixture: dict[str, Any]) -> list[BlindedRecord]:
    cases = fixture["cases"]
    require(isinstance(cases, list), "fixture cases are not a list")
    records: list[BlindedRecord] = []
    identifiers: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), "fixture case is not an object")
        identifier = case["id"]
        require(identifier not in identifiers, f"duplicate fixture id: {identifier}")
        identifiers.add(identifier)
        messages = case["input_messages"]
        require(
            isinstance(messages, list)
            and messages
            and messages[0].get("role") == "system"
            and messages[-1].get("role") == "user",
            f"fixture role boundary differs: {identifier}",
        )
        normalized_messages = [
            {"content": str(item["content"]), "role": str(item["role"])}
            for item in messages
        ]
        records.append(
            BlindedRecord(
                prompt=prompt_digest(normalized_messages),
                five_grams=five_gram_digests(normalized_messages),
                identifier_template=identifier_template_digest(case["template_family"]),
                lexical_template=lexical_template_digest(normalized_messages),
            )
        )
    expected = int(fixture["policy"]["expected_case_count"])
    require(len(records) == expected, "fixture case count differs")
    return records


def build_report() -> dict[str, Any]:
    fixture = load_json(FIXTURE)
    manifest = load_json(SOURCE_MANIFEST)
    expected_files = manifest["files"]
    source_bindings: dict[str, Any] = {}
    evaluation: list[BlindedRecord] = []
    for split, path in (("dev", DEV), ("holdout", HOLDOUT)):
        expected = expected_files[f"{split}.jsonl"]
        observed_hash = sha256_file(path)
        require(observed_hash == expected["sha256"], f"{split} bytes differ")
        records = scan_split(path, split, int(manifest["counts"][split]))
        source_bindings[split] = {
            "bytes": path.stat().st_size,
            "count": len(records),
            "sha256": observed_hash,
        }
        evaluation.extend(records)

    probes = fixture_records(fixture)
    policy = fixture["policy"]
    threshold = float(policy["maximum_eval_five_gram_jaccard"])
    exact = exact_collision_count(probes, evaluation)
    near, maximum = near_collision_result(probes, evaluation, threshold)
    templates = template_collision_count(probes, evaluation)
    status = (
        "PASS"
        if exact == int(policy["required_eval_exact_prompt_collisions"])
        and near == 0
        and templates == int(policy["required_eval_template_collisions"])
        and maximum <= threshold
        else "NO_GO"
    )
    return {
        "attestation_id": "qwen25-lora-v3-synthetic-disjointness-v1",
        "evidence_boundary": {
            "answer_fields_accessed": False,
            "human_exposure_to_evaluation_text": False,
            "prompt_only_selective_decoder": True,
            "raw_prompts_or_fingerprints_emitted": False,
        },
        "fixture": {
            "case_count": len(probes),
            "path": relative(FIXTURE),
            "prompt_fingerprint_commitments": split_commitments(probes),
            "sha256": sha256_file(FIXTURE),
        },
        "method": {
            **METHOD,
            "comparison_direction": "synthetic-fixture-to-dev-and-holdout",
            "near_duplicate_threshold_strictly_greater_than": threshold,
        },
        "results": {
            "exact_normalized_prompt_collision_count": exact,
            "maximum_fixture_to_evaluation_five_gram_jaccard": maximum,
            "near_duplicate_collision_count": near,
            "status": status,
            "template_family_collision_count": templates,
        },
        "schema_version": 1,
        "source_bindings": source_bindings,
        "tool_bindings": {
            relative(BLINDED_SCANNER): sha256_file(BLINDED_SCANNER),
            relative(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
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
    require(observed == expected, "disjointness attestation differs from blinded recomputation")
    digest = hashlib.sha256(observed).hexdigest()
    require(
        sidecar_path(path).read_text(encoding="ascii").split() == [digest, path.name],
        "disjointness sidecar differs",
    )
    require(json.loads(observed)["results"]["status"] == "PASS", "synthetic fixtures are not disjoint")
    print(f"QWEN25_LORA_SYNTHETIC_DISJOINTNESS_VERIFIED sha256={digest}")


def main() -> None:
    args = parse_args()
    if args.output is not None:
        write_report(args.output.resolve())
    else:
        verify_report(args.verify.resolve())


if __name__ == "__main__":
    main()
