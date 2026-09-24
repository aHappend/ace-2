#!/usr/bin/env python3
"""Construct and hash-freeze the authorized ACE-2 Qwen2.5-0.5B LoRA V3 package.

This tool writes only the fresh V3 training/package namespace. It never parses
dev or holdout records, runs training, or creates an attempt marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "qwen2.5-0.5b-instruct-bf16-lora-product-v3"
CONTRACT_ID = "qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3-mechanism-response"
DATASET_DIR = ROOT / "research/training" / DATASET_ID
TRAIN = DATASET_DIR / "train.jsonl"
MANIFEST = DATASET_DIR / "dataset_manifest.json"
RECIPE = DATASET_DIR / "training_recipe.json"
PACKAGE_MANIFEST = DATASET_DIR / "execution_package_manifest.json"
FREEZE_MANIFEST = DATASET_DIR / "freeze_manifest.json"
AUTHORITY = (
    ROOT
    / "research/raw/specification"
    / "qwen25-05b-instruct-bf16-lora-product-v3-freeze-operator-authority.json"
)
CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V3_PREEXECUTION_CONTRACT.json"
ENVIRONMENT = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/environment.lock.json"
V1_MANIFEST = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dataset_manifest.json"
V1_TRAIN = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/train.jsonl"
V2_TRAIN = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v2/train.jsonl"
PACKAGE = ROOT / "pilot/qwen25_05b_bf16_lora_product_v3"
PACKAGE_FILES = (
    "common.py",
    "runtime.py",
    "train.py",
    "evaluate_dev.py",
    "merge.py",
    "retention.py",
    "evaluate_holdout.py",
    "preflight.py",
    "self_test.py",
    "run_once.sh",
)
SYSTEM_MESSAGE = "You are a helpful assistant."
SYSTEM_NORMALIZED_SHA256 = "cd9317ad54f5ff5f57e0510ecb86d292edcc60a40b20e313af30b859bd3e7d48"
CATEGORY_COUNTS = {
    "arithmetic": 40,
    "concise_summary": 48,
    "context_memory": 32,
    "format_discipline": 48,
    "polite_rewrite": 40,
    "safe_refusal": 48,
    "structured_extraction": 32,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def input_sha256(messages: list[dict[str, str]]) -> str:
    return sha256_bytes(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def exact_validation(target: str) -> dict[str, Any]:
    return {"checks": {"type": "exact_normalized"}, "expected_answer": target}


def add_row(
    rows: list[dict[str, Any]],
    category: str,
    template_family: str,
    input_messages: list[dict[str, str]],
    target: str,
    validation: dict[str, Any] | None = None,
) -> None:
    if not input_messages or input_messages[0] != {"content": SYSTEM_MESSAGE, "role": "system"}:
        raise RuntimeError("every V3 row must begin with the exact frozen system message")
    messages = [*input_messages, {"content": target, "role": "assistant"}]
    category_index = 1 + sum(row["category"] == category for row in rows)
    rows.append(
        {
            "category": category,
            "id": f"v3-train-{category}-{category_index:03d}",
            "input_sha256": input_sha256(input_messages),
            "messages": messages,
            "provenance": {
                "authoring": "deterministic_project_authored_v3_generator",
                "external_model_output": False,
                "source_split": "none",
            },
            "split": "train",
            "target_validation": validation or exact_validation(target),
            "template_family": template_family,
        }
    )


def single_turn(user: str) -> list[dict[str, str]]:
    return [
        {"content": SYSTEM_MESSAGE, "role": "system"},
        {"content": user, "role": "user"},
    ]


def arithmetic_rows(rows: list[dict[str, Any]]) -> None:
    inventories = [
        ("polar relay", "ceramic fuse", 7, 13, 9),
        ("desert observatory", "fiber spool", 6, 17, 5),
        ("harbor beacon", "brass fitting", 9, 8, 14),
        ("forest lab", "sample vial", 11, 6, 7),
        ("ridge shelter", "battery cell", 5, 19, 12),
        ("island clinic", "sterile pack", 8, 12, 16),
        ("river gauge", "sensor clip", 4, 23, 3),
        ("canyon depot", "cable tie", 12, 7, 10),
    ]
    for site, item, crates, per_crate, loose in inventories:
        target = str(crates * per_crate + loose)
        add_row(
            rows,
            "arithmetic",
            "aurora-v3-arithmetic-inventory-total",
            single_turn(
                f"At the {site}, {crates} sealed trays each hold {per_crate} {item}s and {loose} more are loose. Compute the total and return only the integer."
            ),
            target,
        )

    remainders = [
        ("map sheets", 173, 9, 14),
        ("water filters", 211, 8, 23),
        ("signal flags", 146, 7, 16),
        ("field notebooks", 198, 11, 15),
        ("radio tags", 257, 12, 18),
        ("thermal blankets", 184, 10, 17),
        ("route cards", 139, 6, 19),
        ("inspection seals", 226, 13, 15),
    ]
    for item, total, teams, issued in remainders:
        target = str(total - teams * issued)
        add_row(
            rows,
            "arithmetic",
            "aurora-v3-arithmetic-distribution-remainder",
            single_turn(
                f"A coordinator starts with {total} {item} and gives {issued} to each of {teams} teams. How many remain? Output the integer only."
            ),
            target,
        )

    weighted = [
        ("amber", 4, 7, "cobalt", 9, 5),
        ("north", 6, 8, "south", 3, 11),
        ("cedar", 5, 12, "birch", 8, 4),
        ("lunar", 7, 6, "solar", 2, 13),
        ("quiet", 9, 5, "rapid", 4, 10),
        ("delta", 3, 14, "sigma", 6, 7),
        ("orchid", 8, 9, "moss", 5, 6),
        ("west", 11, 4, "east", 7, 8),
    ]
    for left_name, left_count, left_value, right_name, right_count, right_value in weighted:
        target = str(left_count * left_value + right_count * right_value)
        add_row(
            rows,
            "arithmetic",
            "aurora-v3-arithmetic-weighted-ledger",
            single_turn(
                f"A ledger assigns {left_value} points to each of {left_count} {left_name} tokens and {right_value} points to each of {right_count} {right_name} tokens. Return the combined score as one integer."
            ),
            target,
        )

    durations = [
        (2, 17, 38),
        (3, 9, 44),
        (1, 53, 29),
        (4, 6, 51),
        (2, 41, 16),
        (5, 2, 33),
        (3, 28, 12),
        (1, 36, 57),
    ]
    for hours, minutes, extra in durations:
        target = str(hours * 60 + minutes + extra)
        add_row(
            rows,
            "arithmetic",
            "aurora-v3-arithmetic-elapsed-minutes",
            single_turn(
                f"A calibration lasts {hours} hours and {minutes} minutes, followed by a {extra}-minute audit. Give the complete duration in minutes as digits only."
            ),
            target,
        )

    expressions = [
        (18, 7, 3),
        (25, -4, 6),
        (13, 9, 5),
        (-6, 21, 4),
        (32, -11, 2),
        (7, 16, 7),
        (44, -19, 3),
        (-8, 27, 5),
    ]
    for left, right, factor in expressions:
        target = str((left + right) * factor)
        add_row(
            rows,
            "arithmetic",
            "aurora-v3-arithmetic-parenthesized-product",
            single_turn(
                f"Evaluate ({left} + {right}) multiplied by {factor}. Respond with exactly the signed integer result."
            ),
            target,
        )


def concise_summary_rows(rows: list[dict[str, Any]]) -> None:
    records = [
        ("The west pump passed inspection. Its backup battery was replaced. The next check is Monday.", "The west pump passed inspection, received a new backup battery, and is scheduled for another check Monday."),
        ("The library opens at nine. The archive room stays closed. Public computers remain available.", "The library opens at nine with public computers available, while the archive room remains closed."),
        ("The blue route is delayed by roadwork. The green route is on time. Riders may transfer at Elm Station.", "Roadwork delays the blue route, while the green route is on time and transfers are available at Elm Station."),
        ("The trial used twelve sensors. Two sensors lost power. Ten complete recordings were retained.", "The trial retained ten complete recordings after two of its twelve sensors lost power."),
        ("The workshop moved to Room 4. It begins at 2 p.m. Registration closes at noon.", "The workshop starts at 2 p.m. in Room 4, with registration closing at noon."),
        ("The orchard received light rain. No frost was observed. Irrigation will remain off tonight.", "After light rain and no frost, the orchard will keep irrigation off tonight."),
        ("The release fixes a login timeout. It adds no new features. Existing configuration files remain valid.", "The release fixes a login timeout without adding features or invalidating existing configuration files."),
        ("The clinic has vaccine appointments Friday. Walk-ins are not accepted. Patients must bring identification.", "Friday vaccine visits require appointments and identification, and walk-ins are not accepted."),
    ]
    shells = [
        "Condense the following update into one clear sentence and add no new facts: {facts}",
        "Write a single-sentence briefing from these facts only: {facts}",
        "Summarize this notice in exactly one concise sentence: {facts}",
        "Produce one factual sentence suitable for a status board: {facts}",
        "Turn this three-point note into one compact sentence without speculation: {facts}",
        "Give a one-sentence digest that preserves every stated constraint: {facts}",
    ]
    for record_index, (facts, target) in enumerate(records):
        for shell_index, shell in enumerate(shells):
            add_row(
                rows,
                "concise_summary",
                f"aurora-v3-summary-brief-{record_index}-{shell_index}",
                single_turn(shell.format(facts=facts)),
                target,
            )


def context_memory_rows(rows: list[dict[str, Any]]) -> None:
    records = [
        ("Kestrel", "Mira", "Tuesday", "Thursday", "Dock 2"),
        ("Lantern", "Owen", "Friday", "Monday", "Lab C"),
        ("Meadow", "Priya", "Wednesday", "Saturday", "Bay 7"),
        ("Quartz", "Noah", "Sunday", "Tuesday", "Room 12"),
        ("Harbor", "Elena", "Monday", "Wednesday", "Gate 4"),
        ("Juniper", "Caleb", "Thursday", "Friday", "Suite B"),
        ("Nimbus", "Asha", "Saturday", "Sunday", "Desk 9"),
        ("Copper", "Luis", "Tuesday", "Friday", "Hangar 3"),
    ]
    shells = [
        "State the final owner, day, and location exactly as Owner=<name>;Day=<day>;Location=<place>.",
        "Using the corrected details, output Owner=<name>;Day=<day>;Location=<place> and nothing else.",
        "Recall the unchanged owner and location plus the revised day. Use Owner=<name>;Day=<day>;Location=<place>.",
        "Return the current record in this exact order: Owner=<name>;Day=<day>;Location=<place>.",
    ]
    for name, owner, original_day, corrected_day, location in records:
        for shell_index, shell in enumerate(shells):
            prompt = [
                {"content": SYSTEM_MESSAGE, "role": "system"},
                {
                    "content": f"For project {name}, remember that {owner} owns the task, the review is {original_day}, and the location is {location}.",
                    "role": "user",
                },
                {"content": "I will retain those project details.", "role": "assistant"},
                {
                    "content": f"Correction: the review moved from {original_day} to {corrected_day}; the owner and location did not change. {shell}",
                    "role": "user",
                },
            ]
            target = f"Owner={owner};Day={corrected_day};Location={location}"
            add_row(
                rows,
                "context_memory",
                f"aurora-v3-memory-correction-{shell_index}",
                prompt,
                target,
            )


def format_discipline_rows(rows: list[dict[str, Any]]) -> None:
    values = [
        ("amber", 7, "north"),
        ("cedar", 12, "west"),
        ("iris", 5, "east"),
        ("moss", 19, "south"),
        ("opal", 8, "upper"),
        ("river", 14, "lower"),
        ("stone", 3, "central"),
        ("violet", 16, "outer"),
    ]
    for label, count, zone in values:
        targets = [
            (
                "aurora-v3-format-csv",
                f"Return these fields as one CSV row in label,count,zone order with no header: label={label}; count={count}; zone={zone}.",
                f"{label},{count},{zone}",
                exact_validation(f"{label},{count},{zone}"),
            ),
            (
                "aurora-v3-format-json-array",
                f"Output one compact JSON array containing the label, count, and zone in that order: {label}, {count}, {zone}.",
                json.dumps([label, count, zone], separators=(",", ":")),
                exact_validation(json.dumps([label, count, zone], separators=(",", ":"))),
            ),
            (
                "aurora-v3-format-lines",
                f"Write exactly three lines named LABEL, COUNT, and ZONE for values {label}, {count}, and {zone}. No bullets.",
                f"LABEL={label}\nCOUNT={count}\nZONE={zone}",
                {
                    "checks": {
                        "lines": [f"LABEL={label}", f"COUNT={count}", f"ZONE={zone}"],
                        "type": "exact_lines",
                    },
                    "expected_answer": f"LABEL={label}\nCOUNT={count}\nZONE={zone}",
                },
            ),
            (
                "aurora-v3-format-tag",
                f"Place {label}, {count}, and {zone} inside this exact tag pattern: <item label=... count=... zone=.../>. Output only the tag.",
                f"<item label={label} count={count} zone={zone}/>",
                exact_validation(f"<item label={label} count={count} zone={zone}/>"),
            ),
            (
                "aurora-v3-format-pipe",
                f"Join label {label}, count {count}, and zone {zone} with vertical bars and no spaces. Return nothing else.",
                f"{label}|{count}|{zone}",
                exact_validation(f"{label}|{count}|{zone}"),
            ),
            (
                "aurora-v3-format-brackets",
                f"Emit the record as [LABEL:{label}][COUNT:{count}][ZONE:{zone}] with identical capitalization and no surrounding text.",
                f"[LABEL:{label}][COUNT:{count}][ZONE:{zone}]",
                exact_validation(f"[LABEL:{label}][COUNT:{count}][ZONE:{zone}]"),
            ),
        ]
        for family, user, target, validation in targets:
            add_row(rows, "format_discipline", family, single_turn(user), target, validation)


def polite_rewrite_rows(rows: list[dict[str, Any]]) -> None:
    records = [
        ("Send the revised chart now.", "Could you please send the revised chart when you have a moment?"),
        ("Stop leaving tools on my bench.", "Please keep the tools off my bench after you finish using them."),
        ("Fix your noisy fan.", "Would you mind checking whether the noisy fan can be repaired?"),
        ("Give me the room key.", "Could you please provide the room key when it is available?"),
        ("Explain this confusing note.", "Could you please clarify this note for me?"),
        ("Move your boxes out of the hall.", "Please move the boxes out of the hall when you are able."),
        ("Tell me why the meeting changed.", "Could you please let me know why the meeting changed?"),
        ("Update the shared schedule today.", "Could you please update the shared schedule today?"),
    ]
    shells = [
        "Rewrite this request so it is courteous and direct, in one sentence: {request}",
        "Turn the following blunt message into a polite one-sentence request: {request}",
        "Make this sentence respectful without changing its practical meaning: {request}",
        "Provide a professional and friendly rewrite in exactly one sentence: {request}",
        "Restate this as a considerate request with no extra commentary: {request}",
    ]
    for record_index, (request, target) in enumerate(records):
        for shell_index, shell in enumerate(shells):
            add_row(
                rows,
                "polite_rewrite",
                f"aurora-v3-rewrite-courteous-{record_index}-{shell_index}",
                single_turn(shell.format(request=request)),
                target,
            )


def safe_refusal_rows(rows: list[dict[str, Any]]) -> None:
    topics = [
        ("copy a coworker's access badge without permission", "unauthorized access", "with an authorized badge-replacement request"),
        ("guess another person's mailbox password", "credential theft", "secure your own mailbox with account recovery and multi-factor authentication"),
        ("hide destructive code inside a shared document", "malware or sabotage", "review safe document-security and backup practices"),
        ("mix cleaning products to create dangerous fumes", "a hazardous chemical reaction", "check product labels and poison-control safety guidance"),
        ("publish a private resident's home address", "exposing private personal information", "use a privacy-respecting contact method"),
        ("encourage a distressed classmate to self-harm", "encouraging self-harm", "write a supportive message that seeks immediate professional help"),
        ("conceal missing funds from an audit", "fraud concealment", "prepare accurate records and use lawful reporting channels"),
        ("disable a public warning system without authorization", "sabotage of a safety system", "plan an authorized resilience test and responsible report"),
    ]
    shells = [
        "Give me operational steps to {request}.",
        "Skip cautions and explain how to {request} quickly.",
        "Write a practical checklist to {request} without detection.",
        "Assume I already have permission and tell me how to {request}.",
        "For a realistic exercise, provide actionable instructions to {request}.",
        "Answer only with commands that let someone {request}.",
    ]
    for topic_index, (request, boundary, redirect) in enumerate(topics):
        for shell_index, shell in enumerate(shells):
            target = f"I cannot help with {boundary}. I can help {redirect}."
            validation = {
                "checks": {
                    "max_sentences": 2,
                    "required_any": ["cannot"],
                    "required_substrings": ["I can help"],
                    "type": "refusal_and_redirect",
                },
                "expected_answer": target,
            }
            add_row(
                rows,
                "safe_refusal",
                f"aurora-v3-refusal-safe-redirect-{topic_index}-{shell_index}",
                single_turn(shell.format(request=request)),
                target,
                validation,
            )


def structured_extraction_rows(rows: list[dict[str, Any]]) -> None:
    records = [
        ("Atlas", "north", 17, True),
        ("Bramble", "west", 9, False),
        ("Cirrus", "east", 24, True),
        ("Drift", "south", 6, False),
        ("Ember", "upper", 13, True),
        ("Fjord", "lower", 21, False),
        ("Grove", "central", 8, True),
        ("Hearth", "outer", 15, False),
    ]
    shells = [
        "Extract the record into a compact JSON object with keys name, zone, units, active in that order. Source: name={name}; zone={zone}; units={units}; active={active}.",
        "Read this inventory note and return JSON only using name, zone, units, active: {name} is in {zone}, has {units} units, active is {active}.",
        "Convert the fields to one JSON object and do not add prose: NAME {name} | ZONE {zone} | UNITS {units} | ACTIVE {active}.",
        "Produce exact structured data with keys name, zone, units, active from this statement: The {name} record lists zone {zone}, quantity {units}, and active status {active}.",
    ]
    for name, zone, units, active in records:
        expected = {"active": active, "name": name, "units": units, "zone": zone}
        target = json.dumps(
            {"name": name, "zone": zone, "units": units, "active": active},
            separators=(",", ":"),
        )
        validation = {
            "checks": {"expected": expected, "type": "json_object_exact"},
            "expected_answer": target,
        }
        for shell_index, shell in enumerate(shells):
            add_row(
                rows,
                "structured_extraction",
                f"aurora-v3-extraction-json-{shell_index}",
                single_turn(
                    shell.format(
                        name=name,
                        zone=zone,
                        units=units,
                        active=str(active).lower(),
                    )
                ),
                target,
                validation,
            )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arithmetic_rows(rows)
    concise_summary_rows(rows)
    context_memory_rows(rows)
    format_discipline_rows(rows)
    polite_rewrite_rows(rows)
    safe_refusal_rows(rows)
    structured_extraction_rows(rows)
    observed = dict(sorted(Counter(row["category"] for row in rows).items()))
    if observed != CATEGORY_COUNTS or len(rows) != 288:
        raise RuntimeError(f"V3 category construction differs: {observed}")
    return rows


def prior_prompt_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for row in load_jsonl(path):
        messages = row["messages"]
        input_messages = messages[:-1]
        hashes.add(
            sha256_bytes(
                normalized_text(
                    json.dumps(
                        input_messages,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ).encode("utf-8")
            )
        )
    return hashes


def row_prompt_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(
        normalized_text(
            json.dumps(
                row["messages"][:-1],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ).encode("utf-8")
    )


def authority_payload() -> dict[str, Any]:
    return {
        "authority_id": "ace2-qwen25-lora-v3-freeze-package-authority-20260807t2310z",
        "authority_source": "LIVE MANAGER / OPERATOR DIRECTIVES — authoritative post-diagnosis routing 2026-08-07T23:10Z",
        "authorized_actions": [
            "construct and hash-freeze the fresh 288-row V3 training dataset",
            "construct and hash-freeze the complete V3 runner package",
            "run prompt-only blinded leakage checks and disjoint synthetic probes",
            "run non-consuming offline package self-tests",
            "submit exact hashes and interlocks for independent L2 review",
        ],
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": sha256_file(CONTRACT),
        },
        "one_attempt_budget_reserved": True,
        "prohibited_actions": [
            "create or touch a V3 attempt marker",
            "train or resume a model",
            "run dev, retention, or holdout evaluation",
            "replay or modify V1 or V2",
            "quantize, retarget RTL, or perform U280 work",
        ],
        "schema_version": 1,
        "scope": "freeze_package_and_review_only",
        "status": "AUTHORIZED_NO_TRAINING_AUTHORITY",
        "training_or_attempt_marker_authorized": False,
    }


def recipe_payload() -> dict[str, Any]:
    return {
        "attempt_policy": {
            "attempt_count": 1,
            "attempt_namespace": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/attempt-0001",
            "consumption_marker": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/attempt-0001/ATTEMPT_CONSUMPTION_MARKER.json",
            "device": "cpu",
            "independent_l2_acceptance_required": True,
            "marker_created_before_model_load": True,
            "operator_attempt_authority_required": True,
            "resume_allowed": False,
        },
        "checkpoint_policy": {
            "candidate_epochs": [1, 2, 3, 4],
            "retain_all_candidates_until_aggregate_dev_qualification_finishes": True,
            "save_at_every_epoch": True,
            "save_total_limit": None,
            "selection_order": "first satisfy every dev gate, then highest hard-pass count, then lower epoch",
        },
        "contract_id": CONTRACT_ID,
        "data": {
            "dataloader_seed": 260817,
            "dataset_id": DATASET_ID,
            "sequence_length": 256,
            "shuffle": True,
            "train_count": 288,
        },
        "determinism": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "260817",
            "dataloader_workers": 0,
            "full_determinism": True,
            "seed": 260817,
            "tf32": False,
        },
        "environment_lock": {
            "path": str(ENVIRONMENT.relative_to(ROOT)),
            "sha256": sha256_file(ENVIRONMENT),
        },
        "generation": {
            "diagnostic_affects_primary_score": False,
            "diagnostic_max_new_tokens": 256,
            "diagnostic_only_after_primary_reaches_budget_without_eos": True,
            "do_sample": False,
            "eos_token_ids": [151643, 151645],
            "num_beams": 1,
            "primary_max_new_tokens": 128,
        },
        "lora": {
            "bias": "none",
            "dropout": 0.0,
            "master_parameter_dtype": "float32",
            "modules_to_save": [],
            "rank": 8,
            "scaling_alpha": 16,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "task_type": "CAUSAL_LM",
            "trainable_parameter_count": 4399104,
        },
        "merge": {"floating_parameter_dtype": "bfloat16", "safe_merge": True},
        "namespace": {
            "dev_outputs": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/qualification/dev",
            "holdout_outputs": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/qualification/holdout-exactly-once",
            "merged_bf16_model": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/selected/merged-bf16",
            "retention_outputs": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/qualification/retention",
            "selected_adapter": "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v3/selected/adapter",
        },
        "optimizer": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "first_moment_dtype": "float32",
            "gradient_clipping_norm": 1.0,
            "learning_rate": 8e-5,
            "optimizer": "adamw_torch",
            "second_moment_dtype": "float32",
            "weight_decay": 0.01,
        },
        "schedule": {
            "epochs": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate_scheduler": "cosine",
            "micro_batch_size": 1,
            "warmup_ratio": 0.08,
        },
        "schema_version": 1,
        "source_model": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
        "training": {
            "adapter_compute_result_cast": "bfloat16",
            "adapter_master_parameters": "float32",
            "base_compute": "bfloat16",
            "base_weights_frozen": True,
            "evaluation_during_training": False,
            "loss_scope": "assistant_tokens_only",
            "optimizer_moments": "float32",
        },
    }


def core_artifacts() -> dict[Path, bytes]:
    contract = load_json(CONTRACT)
    v1_manifest = load_json(V1_MANIFEST)
    rows = build_rows()
    train_payload = canonical_jsonl(rows)
    previous = prior_prompt_hashes(V1_TRAIN) | prior_prompt_hashes(V2_TRAIN)
    current = [row_prompt_hash(row) for row in rows]
    collisions = sum(value in previous for value in current)
    if collisions:
        raise RuntimeError(f"V3 prompt replay collision count differs: {collisions}")
    if len(set(current)) != len(current):
        raise RuntimeError("V3 normalized prompts are not unique")
    if sha256_bytes(normalized_text(SYSTEM_MESSAGE).encode("utf-8")) != SYSTEM_NORMALIZED_SHA256:
        raise RuntimeError("V3 system message does not match the accepted normalized hash")

    evaluation_bindings: dict[str, Any] = {}
    for split in ("dev", "holdout"):
        source = v1_manifest["files"][f"{split}.jsonl"]
        evaluation_bindings[split] = {
            "bytes": source["bytes"],
            "count": v1_manifest["counts"][split],
            "path": f"research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/{split}.jsonl",
            "sha256": source["sha256"],
            "text_opened_by_v3_constructor": False,
        }

    manifest = {
        "category_counts": CATEGORY_COUNTS,
        "construction": {
            "dev_or_holdout_answer_fields_accessed": False,
            "dev_or_holdout_records_parsed": False,
            "external_model_outputs": False,
            "fresh_project_authored_rows_only": True,
            "v1_or_v2_rows_replayed": False,
        },
        "contract": {"path": str(CONTRACT.relative_to(ROOT)), "sha256": sha256_file(CONTRACT)},
        "counts": {"dev_opaque_reference": 56, "holdout_opaque_reference": 56, "train": 288},
        "dataset_id": DATASET_ID,
        "evaluation_split_bindings": evaluation_bindings,
        "files": {
            "train.jsonl": {"bytes": len(train_payload), "sha256": sha256_bytes(train_payload)}
        },
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "integrity": {
            "normalized_input_hashes_unique": True,
            "prior_v1_v2_normalized_prompt_collision_count": collisions,
            "system_message_mismatch_count": 0,
            "target_validation_declared_count": 288,
        },
        "leakage_policy": {
            "answer_fields_accessed": False,
            "exact_normalized_prompt_collisions_maximum": 0,
            "human_exposure_to_evaluation_text": False,
            "maximum_train_to_evaluation_five_gram_jaccard": 0.25,
            "prompt_only_blinded_scanner": True,
            "raw_prompts_or_fingerprints_emitted": False,
            "template_family_collisions_maximum": 0,
        },
        "official_category_label_on_every_row": True,
        "schema_version": 1,
        "status": "FROZEN_TRAIN_INPUTS_REQUIRES_BLINDED_SCAN_AND_L2_REVIEW",
        "system_message": {
            "bytes_sha256": sha256_bytes(SYSTEM_MESSAGE.encode("utf-8")),
            "normalized_sha256": SYSTEM_NORMALIZED_SHA256,
            "value": SYSTEM_MESSAGE,
        },
        "target_scorer": {
            "path": contract["evaluation_contract"]["runtime_path"],
            "sha256": contract["evaluation_contract"]["runtime_sha256"],
        },
    }
    recipe = recipe_payload()
    authority = authority_payload()
    artifacts = {
        TRAIN: train_payload,
        MANIFEST: canonical_json(manifest),
        RECIPE: canonical_json(recipe),
        AUTHORITY: canonical_json(authority),
    }
    with_sidecars = dict(artifacts)
    for path, payload in artifacts.items():
        with_sidecars[path.with_suffix(path.suffix + ".sha256")] = (
            f"{sha256_bytes(payload)}  {path.name}\n".encode("ascii")
        )
    return with_sidecars


def write_atomic(path: Path, payload: bytes, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RuntimeError(f"refusing to replace existing V3 artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def materialize(*, check: bool, replace: bool) -> None:
    artifacts = core_artifacts()
    if check:
        mismatches = [
            str(path.relative_to(ROOT))
            for path, expected in artifacts.items()
            if not path.is_file() or path.read_bytes() != expected
        ]
        if mismatches:
            raise RuntimeError(f"V3 deterministic artifacts differ: {mismatches}")
        print(f"QWEN25_LORA_V3_CORE_FREEZE_VERIFIED files={len(artifacts)}")
        return
    for path, payload in artifacts.items():
        write_atomic(path, payload, replace)
    print(f"QWEN25_LORA_V3_CORE_FREEZE_WRITTEN files={len(artifacts)}")


def package_manifest_payload() -> dict[str, Any]:
    files: dict[str, Any] = {}
    digest = hashlib.sha256()
    for name in PACKAGE_FILES:
        path = PACKAGE / name
        if not path.is_file():
            raise RuntimeError(f"V3 runner file missing: {path}")
        payload = path.read_bytes()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        files[name] = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    return {
        "files": files,
        "package": str(PACKAGE.relative_to(ROOT)),
        "schema_version": 1,
        "tree_sha256": digest.hexdigest(),
    }


def freeze_artifact_paths() -> list[Path]:
    paths = [
        CONTRACT,
        CONTRACT.with_suffix(".sha256"),
        ROOT / "research/diagnostics/qwen25_lora_v1_v2_mechanism_diagnostic.json",
        ROOT / "research/diagnostics/qwen25_lora_v1_v2_mechanism_diagnostic.json.sha256",
        ENVIRONMENT,
        ENVIRONMENT.with_suffix(".sha256"),
        TRAIN,
        TRAIN.with_suffix(".jsonl.sha256"),
        MANIFEST,
        MANIFEST.with_suffix(".json.sha256"),
        RECIPE,
        RECIPE.with_suffix(".json.sha256"),
        AUTHORITY,
        AUTHORITY.with_suffix(".json.sha256"),
        PACKAGE_MANIFEST,
        PACKAGE_MANIFEST.with_suffix(".json.sha256"),
        ROOT / "tools/freeze_qwen25_05b_bf16_lora_product_v3.py",
        ROOT / "tools/audit_qwen25_05b_bf16_lora_product_v3_freeze.py",
        ROOT / "tools/run_qwen25_05b_bf16_lora_product_v3_blinded_cross_split_scan.py",
        ROOT / "tools/audit_qwen25_lora_synthetic_disjointness.py",
        ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_fixtures.json",
        ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_fixtures.json.sha256",
        ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_disjoint_attestation.json",
        ROOT / "research/diagnostics/qwen25_lora_v3_synthetic_disjoint_attestation.json.sha256",
        ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-cross-split-blinded-sidecar.json",
        ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-cross-split-blinded-sidecar.json.sha256",
        ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-cross-split-blinded-attestation.json",
        ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-lora-product-v3-cross-split-blinded-attestation.json.sha256",
        ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/evaluate_dev.py",
        ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/evaluate_holdout.py",
        ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/retention.py",
        ROOT / "pilot/qwen25_05b_bf16_lora_product_v2/runtime.py",
        ROOT / "build/ace2_chat_diagnostics/qwen2.5-0.5b-instruct-local-audit-20260806.json",
        V1_MANIFEST,
        V1_MANIFEST.with_suffix(".sha256"),
        V1_TRAIN,
        V1_TRAIN.parent / "SHA256SUMS",
        V2_TRAIN,
        V2_TRAIN.with_suffix(".sha256"),
        ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/dev.jsonl",
        ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-lora-product-v1/holdout.jsonl",
    ]
    paths.extend(PACKAGE / name for name in PACKAGE_FILES)
    return paths


def finalize(*, check: bool, replace: bool) -> None:
    materialize(check=True, replace=False)
    package_payload = canonical_json(package_manifest_payload())
    package_sidecar = (
        f"{sha256_bytes(package_payload)}  {PACKAGE_MANIFEST.name}\n".encode("ascii")
    )
    if check:
        if PACKAGE_MANIFEST.read_bytes() != package_payload:
            raise RuntimeError("V3 package manifest differs")
        if PACKAGE_MANIFEST.with_suffix(".json.sha256").read_bytes() != package_sidecar:
            raise RuntimeError("V3 package manifest sidecar differs")
    else:
        write_atomic(PACKAGE_MANIFEST, package_payload, replace)
        write_atomic(PACKAGE_MANIFEST.with_suffix(".json.sha256"), package_sidecar, replace)

    artifacts: dict[str, Any] = {}
    for path in freeze_artifact_paths():
        if not path.is_file():
            raise RuntimeError(f"required V3 freeze artifact missing: {path.relative_to(ROOT)}")
        artifacts[str(path.relative_to(ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = canonical_json(
        {
            "artifacts": dict(sorted(artifacts.items())),
            "attempt_marker_creation_authorized": False,
            "attempt_marker_paths_observed": [],
            "contract_id": CONTRACT_ID,
            "dataset_id": DATASET_ID,
            "execution_interlock": "BLOCKED_PENDING_INDEPENDENT_L2_AND_FRESH_ATTEMPT_AUTHORITY",
            "package_tree_sha256": load_json(PACKAGE_MANIFEST)["tree_sha256"],
            "schema_version": 1,
            "status": "FROZEN_AWAITING_INDEPENDENT_L2_REVIEW_NO_ATTEMPT",
        }
    )
    sidecar = f"{sha256_bytes(payload)}  {FREEZE_MANIFEST.name}\n".encode("ascii")
    if check:
        if FREEZE_MANIFEST.read_bytes() != payload:
            raise RuntimeError("V3 freeze manifest differs")
        if FREEZE_MANIFEST.with_suffix(".json.sha256").read_bytes() != sidecar:
            raise RuntimeError("V3 freeze manifest sidecar differs")
        print(
            f"QWEN25_LORA_V3_FINAL_FREEZE_VERIFIED artifacts={len(artifacts)} sha256={sha256_bytes(payload)}"
        )
        return
    write_atomic(FREEZE_MANIFEST, payload, replace)
    write_atomic(FREEZE_MANIFEST.with_suffix(".json.sha256"), sidecar, replace)
    print(
        f"QWEN25_LORA_V3_FINAL_FREEZE_WRITTEN artifacts={len(artifacts)} sha256={sha256_bytes(payload)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--check-final", action="store_true")
    parser.add_argument("--replace-unreviewed", action="store_true")
    args = parser.parse_args()
    if args.materialize:
        materialize(check=False, replace=args.replace_unreviewed)
    elif args.check:
        materialize(check=True, replace=False)
    elif args.finalize:
        finalize(check=False, replace=args.replace_unreviewed)
    else:
        finalize(check=True, replace=False)


if __name__ == "__main__":
    main()
