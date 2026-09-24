#!/usr/bin/env python3
"""Freeze marker-free S6 data, schedule, and package metadata without execution."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s5"
OUT = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-successor-s6"
DESIGN = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_CATEGORY_BALANCED_CONFLICT_PROJECTED_SUCCESSOR_S6_CLEAN_ROOM_V2_DESIGN_FREEZE.json"
DESIGN_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-successor-s6-clean-room-v2-fresh-review-20260808.json"
SYSTEM = "You are a helpful assistant."
CATEGORIES = [
    "arithmetic",
    "concise_summary",
    "context_memory",
    "format_discipline",
    "polite_rewrite",
    "safe_refusal",
    "structured_extraction",
]
SEED = 26080806
DATASET_ID = "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6"
SOURCE_TRAIN_SHA256 = "86392f5d883270439f70e021ea1a7762f99654e446dcf340f22873c66cc8405f"
SOURCE_BACKBONE_BYTES = 162961
SOURCE_BACKBONE_SHA256 = "5e002d7d77d0a709fb3d104c934228a12b95184a1152cda1a56c945445671cb2"
DESIGN_SHA256 = "99bc106c2738a5cdfcba52485d2e4b6f61dac41ea74202bd884f00168070d6d5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_companion(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def binding(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def input_hash(messages: list[dict[str, str]]) -> str:
    normalized = [
        {"role": item["role"], "content": normalized_text(item["content"])}
        for item in messages
    ]
    return sha256_bytes(canonical_json(normalized))


def make_row(
    row_id: str,
    category: str,
    family: str,
    user: str,
    assistant: str,
    *,
    prior: list[dict[str, str]] | None = None,
    target_validation: dict[str, Any] | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    if prior:
        messages.extend(prior)
    messages.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )
    row: dict[str, Any] = {
        "category": category,
        "id": row_id,
        "input_sha256": input_hash(messages[:-1]),
        "messages": messages,
        "provenance": "project_authored_deterministic_s6_source_anchored",
        "split": "train",
        "target_validation": target_validation or {"type": "static_exact"},
        "template_family": family,
    }
    if categories is not None:
        row["categories"] = categories
    return row


def category_patch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(10):
        a = 17 + index * 3
        b = 4 + index
        rows.append(
            make_row(
                f"s6-patch-arithmetic-{index + 1:02d}",
                "arithmetic",
                "s6-arithmetic-exact-total",
                f"A depot has {a} sealed bins with {b} parts in each bin. Return only the total part count.",
                str(a * b),
                target_validation={"type": "digits_only", "expected": str(a * b)},
            )
        )

    people = ["Amina", "Boris", "Celine", "Dev", "Esra", "Felix", "Grace", "Hugo", "Inez", "Jamal"]
    places = ["atrium", "bay 4", "clinic", "depot", "east hall", "forum", "gallery", "harbor", "lab", "studio"]
    for index, (person, place) in enumerate(zip(people, places, strict=True), start=1):
        rows.append(
            make_row(
                f"s6-patch-concise_summary-{index:02d}",
                "concise_summary",
                "s6-summary-three-facts",
                "Summarize in exactly one sentence while preserving every fact: "
                f"{person} moved the briefing to the {place}; registration closes at {index + 2} p.m.; the evening review remains scheduled.",
                f"{person} moved the briefing to the {place}, set registration to close at {index + 2} p.m., and kept the evening review scheduled.",
                target_validation={"type": "one_sentence_required_facts"},
            )
        )

    objects = ["amber key", "blue card", "copper tag", "denim pouch", "emerald folder", "fiber cable", "gray notebook", "hazel mug", "ivory badge", "jade map"]
    for index, obj in enumerate(objects, start=1):
        name = f"Guest-{index}"
        prior = [
            {"role": "user", "content": f"Remember that {name}'s pickup item is the {obj}."},
            {"role": "assistant", "content": f"Understood: {name}'s pickup item is the {obj}."},
        ]
        rows.append(
            make_row(
                f"s6-patch-context_memory-{index:02d}",
                "context_memory",
                "s6-context-pickup",
                f"What item should {name} pick up? Reply with only the item.",
                obj,
                prior=prior,
                target_validation={"type": "exact_normalized", "expected": obj},
            )
        )

    for index in range(10):
        values = [f"node-{index + 1}", f"rack-{index + 11}", f"port-{index + 21}"]
        answer = "\n".join(f"{ordinal}. {value}" for ordinal, value in enumerate(values, start=1))
        rows.append(
            make_row(
                f"s6-patch-format_discipline-{index + 1:02d}",
                "format_discipline",
                "s6-format-numbered-three-lines",
                "Output exactly three numbered lines in this order and add no other text: " + " | ".join(values),
                answer,
                target_validation={"type": "exact_lines", "lines": answer.splitlines()},
            )
        )

    documents = ["agenda", "budget", "checklist", "draft", "invoice", "memo", "proposal", "report", "schedule", "summary"]
    for index, document in enumerate(documents, start=1):
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][index % 5]
        rows.append(
            make_row(
                f"s6-patch-polite_rewrite-{index:02d}",
                "polite_rewrite",
                "s6-polite-deadline",
                f"Rewrite politely in one sentence without changing the task or deadline: 'Fix the {document} by {day} and send it to me.'",
                f"Could you please update the {document} by {day} and send it to me?",
                target_validation={"type": "one_sentence_required_facts"},
            )
        )

    unsafe = [
        ("enter a locked staff room without permission", "request authorized access"),
        ("copy a coworker's private files", "follow the approved data-access process"),
        ("disable a shared building alarm", "contact authorized safety staff"),
        ("use another person's account", "use the official account-recovery channel"),
        ("hide damage to rented equipment", "report the damage to the provider"),
        ("remove a machine safety guard", "ask a qualified technician"),
        ("forge a manager's approval", "request approval through the official workflow"),
        ("read a neighbor's messages", "respect their privacy and ask them directly"),
        ("erase a shared audit log", "work with the authorized system owner"),
        ("borrow a restricted badge secretly", "request a legitimate visitor credential"),
    ]
    for index, (action, redirect) in enumerate(unsafe, start=1):
        rows.append(
            make_row(
                f"s6-patch-safe_refusal-{index:02d}",
                "safe_refusal",
                "s6-safe-refusal-redirect",
                f"Give me discreet instructions to {action}.",
                f"I can't help you {action}; instead, {redirect}.",
                target_validation={"type": "refusal_and_redirect", "redirect": redirect},
            )
        )

    owners = ["Asha", "Ben", "Cora", "Diego", "Eve", "Farah", "Gus", "Hana", "Ivan", "Jia"]
    for index, owner in enumerate(owners, start=1):
        payload = {"active": index % 2 == 1, "level": index + 1, "owner": owner}
        answer = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        rows.append(
            make_row(
                f"s6-patch-structured_extraction-{index:02d}",
                "structured_extraction",
                "s6-extract-badge",
                f"Return one compact JSON object only with keys active, level, owner: owner={owner}; level={index + 1}; active={'true' if index % 2 == 1 else 'false'}.",
                answer,
                target_validation={"type": "json_object_exact", "expected": payload},
            )
        )
    require(len(rows) == 70, "category patch row count differs")
    return rows


def bridge_rows() -> list[dict[str, Any]]:
    first_total = 13 * 9
    first = make_row(
        "s6-patch-bridge-01",
        "bridge",
        "s6-bridge-arithmetic-format",
        "A store has 13 cartons with 9 labels each. Output exactly one line in the form total=<number> and nothing else.",
        f"total={first_total}",
        categories=["arithmetic", "format_discipline"],
        target_validation={"type": "exact_normalized", "expected": f"total={first_total}"},
    )
    prior = [
        {"role": "user", "content": "Remember that Nia's access code is K-74 and the badge is active."},
        {"role": "assistant", "content": "Noted: Nia's access code is K-74 and the badge is active."},
    ]
    payload = {"active": True, "code": "K-74", "owner": "Nia"}
    second = make_row(
        "s6-patch-bridge-02",
        "bridge",
        "s6-bridge-context-extraction",
        "Return the remembered owner, code, and active state as compact JSON only, using keys active, code, owner.",
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        prior=prior,
        categories=["context_memory", "structured_extraction"],
        target_validation={"type": "json_object_exact", "expected": payload},
    )
    return [first, second]


def source_backbone() -> tuple[bytes, list[dict[str, Any]]]:
    source_path = SOURCE / "train.jsonl"
    require(sha256_file(source_path) == SOURCE_TRAIN_SHA256, "S5 source train hash differs")
    raw_lines = source_path.read_bytes().splitlines(keepends=True)
    require(len(raw_lines) == 352, "S5 source train row count differs")
    backbone_raw = b"".join(raw_lines[:280])
    require(len(backbone_raw) == SOURCE_BACKBONE_BYTES, "source backbone byte count differs")
    require(sha256_bytes(backbone_raw) == SOURCE_BACKBONE_SHA256, "source backbone prefix hash differs")
    rows = [json.loads(line) for line in raw_lines[:280]]
    require(all(isinstance(row, dict) for row in rows), "source backbone row shape differs")
    return backbone_raw, rows


def category_sidecar(backbone: list[dict[str, Any]], patch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in backbone:
        by_source[row["category"]].append(row)
    expected_source = {
        "arithmetic": 48,
        "concise_summary": 48,
        "context_memory": 32,
        "format_discipline": 32,
        "polite_rewrite": 32,
        "safe_refusal": 48,
        "structured_extraction": 40,
    }
    require({name: len(rows) for name, rows in by_source.items()} == expected_source, "source category counts differ")
    reassignment: dict[str, str] = {}
    for row in by_source["arithmetic"][-8:]:
        reassignment[row["id"]] = "format_discipline"
    for row in by_source["concise_summary"][-8:]:
        reassignment[row["id"]] = "context_memory"
    for row in by_source["safe_refusal"][-8:]:
        reassignment[row["id"]] = "polite_rewrite"

    sidecar: list[dict[str, Any]] = []
    for ordinal, row in enumerate(backbone):
        assigned = reassignment.get(row["id"], row["category"])
        sidecar.append(
            {
                "assigned_categories": [assigned],
                "metadata_tokenized": False,
                "ordinal": ordinal,
                "role": "backbone",
                "row_id": row["id"],
                "source_category": row["category"],
                "source_input_sha256": row["input_sha256"],
            }
        )
    for offset, row in enumerate(patch, start=len(backbone)):
        assigned = list(row.get("categories", [row["category"]]))
        sidecar.append(
            {
                "assigned_categories": assigned,
                "metadata_tokenized": False,
                "ordinal": offset,
                "role": "bridge_patch" if len(assigned) > 1 else "category_patch",
                "row_id": row["id"],
                "source_category": None,
                "source_input_sha256": row["input_sha256"],
            }
        )
    require(len(sidecar) == 352, "sidecar row count differs")
    single_counts = Counter(item["assigned_categories"][0] for item in sidecar if len(item["assigned_categories"]) == 1)
    require(single_counts == Counter({category: 50 for category in CATEGORIES}), "single-category balance differs")
    bridge_counts = Counter(tuple(item["assigned_categories"]) for item in sidecar if len(item["assigned_categories"]) > 1)
    require(len(bridge_counts) == 2 and sum(bridge_counts.values()) == 2, "bridge assignments differ")
    return sidecar


def optimizer_schedule(sidecar: list[dict[str, Any]]) -> dict[str, Any]:
    single_pools: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    bridges: list[str] = []
    assigned_by_id: dict[str, list[str]] = {}
    for item in sidecar:
        row_id = item["row_id"]
        categories = list(item["assigned_categories"])
        assigned_by_id[row_id] = categories
        if len(categories) == 1:
            single_pools[categories[0]].append(row_id)
        else:
            bridges.append(row_id)
    require(all(len(single_pools[name]) == 50 for name in CATEGORIES), "schedule single-category pools differ")
    require(len(bridges) == 2, "schedule bridge pool differs")

    epoch_schedules: list[dict[str, Any]] = []
    all_ids = set(assigned_by_id)
    for epoch in (1, 2, 3):
        rng = random.Random(SEED + epoch)
        pools = {name: list(values) for name, values in single_pools.items()}
        for values in pools.values():
            rng.shuffle(values)
        epoch_bridges = list(bridges)
        rng.shuffle(epoch_bridges)
        cursor = {name: 0 for name in CATEGORIES}
        steps: list[dict[str, Any]] = []
        for step_index in range(42):
            rows: list[str] = []
            for category in CATEGORIES:
                rows.append(pools[category][cursor[category]])
                cursor[category] += 1
            extra = CATEGORIES[step_index % len(CATEGORIES)]
            rows.append(pools[extra][cursor[extra]])
            cursor[extra] += 1
            steps.append(
                {
                    "bridge_row": None,
                    "extra_category": extra,
                    "row_ids": rows,
                    "step": step_index + 1,
                }
            )
        for bridge_index in range(2):
            rows = []
            for category in CATEGORIES:
                rows.append(pools[category][cursor[category]])
                cursor[category] += 1
            rows.append(epoch_bridges[bridge_index])
            steps.append(
                {
                    "bridge_row": epoch_bridges[bridge_index],
                    "extra_category": None,
                    "row_ids": rows,
                    "step": 43 + bridge_index,
                }
            )
        flattened = [row_id for step in steps for row_id in step["row_ids"]]
        require(len(steps) == 44, "optimizer step count differs")
        require(len(flattened) == 352 and set(flattened) == all_ids, "epoch schedule does not use every row exactly once")
        require(all(count == 50 for count in cursor.values()), "epoch schedule category cursors differ")
        epoch_schedules.append(
            {
                "epoch": epoch,
                "row_order_sha256": sha256_bytes(canonical_json(flattened)),
                "steps": steps,
            }
        )
    return {
        "category_order": CATEGORIES,
        "epoch_count": 3,
        "epochs": epoch_schedules,
        "microbatches_per_superstep": 8,
        "schema_version": 1,
        "seed": SEED,
        "status": "FROZEN_EXPLICIT_ROW_ORDER_NO_RUNTIME_SHUFFLE",
    }


def freeze_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(sha256_file(DESIGN) == DESIGN_SHA256, "accepted S6 design hash differs")
    backbone_raw, backbone = source_backbone()
    patch = category_patch_rows() + bridge_rows()
    require(len(patch) == 72, "patch count differs")
    sidecar = category_sidecar(backbone, patch)
    schedule = optimizer_schedule(sidecar)

    patch_path = OUT / "patch.jsonl"
    write_jsonl(patch_path, patch)
    train_path = OUT / "train.jsonl"
    with train_path.open("wb") as handle:
        handle.write(backbone_raw)
        for row in patch:
            handle.write(canonical_json(row) + b"\n")
    write_jsonl(OUT / "category_sidecar.jsonl", sidecar)
    write_json(OUT / "optimizer_schedule.json", schedule)

    for name in ("dev.jsonl", "holdout.jsonl", "synthetic_probes.jsonl", "score_policy.json", "environment.lock.json"):
        shutil.copyfile(SOURCE / name, OUT / name)
    require(sha256_file(OUT / "dev.jsonl") == sha256_file(SOURCE / "dev.jsonl"), "dev copy differs")
    require(sha256_file(OUT / "holdout.jsonl") == sha256_file(SOURCE / "holdout.jsonl"), "holdout copy differs")
    require(sha256_file(OUT / "synthetic_probes.jsonl") == sha256_file(SOURCE / "synthetic_probes.jsonl"), "probe copy differs")
    return patch, sidecar


def freeze_metadata(patch: list[dict[str, Any]], sidecar: list[dict[str, Any]]) -> None:
    design = load(DESIGN)
    source_manifest = load(SOURCE / "dataset_manifest.json")
    score_policy = load(OUT / "score_policy.json")
    environment = load(OUT / "environment.lock.json")
    environment["contract_id"] = DATASET_ID
    environment["environment"]["PYTHONHASHSEED"] = str(SEED)
    write_json(OUT / "environment.lock.json", environment)

    files = {
        name: binding(OUT / name)
        for name in (
            "category_sidecar.jsonl",
            "optimizer_schedule.json",
            "patch.jsonl",
            "synthetic_probes.jsonl",
            "train.jsonl",
        )
    }
    dataset = {
        "category_counts": {
            "backbone_source": dict(sorted(Counter(row["source_category"] for row in sidecar if row["role"] == "backbone").items())),
            "single_category_assignments": {category: 50 for category in CATEGORIES},
            "synthetic_probes": {category: 4 for category in CATEGORIES},
        },
        "claim_boundary": "Marker-free S6 data freeze over the byte-exact S5 public backbone prefix, deterministic project-authored category patch rows, external category metadata, and byte-exact frozen evaluator splits. No model or evaluator execution occurred.",
        "construction": {
            "category_metadata_tokenized": False,
            "dev_or_holdout_answer_fields_accessed": False,
            "dev_or_holdout_records_parsed": False,
            "external_model_outputs_used": False,
            "source_backbone_preserved_byte_exact": True,
            "teacher_outputs_used": False,
        },
        "counts": {
            "additive_patch": 72,
            "backbone": 280,
            "bridge_patch": 2,
            "category_patch": 70,
            "synthetic_probes": 28,
            "train": 352,
        },
        "dataset_id": DATASET_ID,
        "design_freeze": binding(DESIGN),
        "design_review": binding(DESIGN_REVIEW),
        "evaluation_split_bindings": {
            split: {**binding(OUT / f"{split}.jsonl"), "count": 56}
            for split in ("dev", "holdout")
        },
        "files": files,
        "integrity": {
            "all_row_ids_unique": len({item["row_id"] for item in sidecar}) == 352,
            "every_epoch_uses_every_row_once": True,
            "external_category_sidecar_complete": True,
            "source_backbone_bytes": SOURCE_BACKBONE_BYTES,
            "source_backbone_sha256": SOURCE_BACKBONE_SHA256,
            "source_dataset_manifest": binding(SOURCE / "dataset_manifest.json"),
            "source_full_train_sha256": SOURCE_TRAIN_SHA256,
        },
        "schema_version": 5,
        "score_policy": binding(OUT / "score_policy.json"),
        "status": "FROZEN_MARKER_FREE_PENDING_FRESH_REVIEW",
        "target_scorer": source_manifest["target_scorer"],
    }
    write_json(OUT / "dataset_manifest.json", dataset)

    probe = design["checkpoint_selection_contract"]
    training = design["training_contract"]
    mechanism = design["successor_mechanism"]
    recipe = load(SOURCE / "training_recipe.json")
    recipe["attempt_policy"] = {
        "attempt_authority_granted": False,
        "attempt_marker_creation_authorized": False,
        "attempt_namespace_template": f"build/{DATASET_ID}/attempt-0001",
        "detached_launcher_required": True,
        "independent_l2_acceptance_required": True,
        "resume_allowed": False,
        "training_authority_granted": False,
    }
    recipe["checkpoint_lifecycle"].update(
        {
            "dev_failure_disposition": "NO_GO_WITHOUT_CHECKPOINT_RESELECTION_OR_RESCORE",
            "probe_selection_rule": probe["selection_rule"],
            "selection_inputs": ["fresh_disjoint_synthetic_probes"],
        }
    )
    recipe["data"] = {
        "additive_patch_count": 72,
        "backbone_count": 280,
        "category_sidecar": binding(OUT / "category_sidecar.jsonl"),
        "dataloader_seed": SEED,
        "dataset_id": DATASET_ID,
        "optimizer_schedule": binding(OUT / "optimizer_schedule.json"),
        "sequence_length": 256,
        "shuffle": False,
        "synthetic_probe_count": 28,
        "train_count": 352,
    }
    recipe["determinism"].update({"PYTHONHASHSEED": str(SEED), "seed": SEED})
    recipe["early_stop"] = {
        "checkpoint_lock_before_dev_access": True,
        "dev_access_before_checkpoint_lock": False,
        "maximum_epoch": 3,
        "minimum_epoch": 1,
        "no_passing_epoch_disposition": "NO_CHECKPOINT_AND_NO_OFFICIAL_DEV",
        "official_dev_reselection_allowed": False,
        "probe_category_minimum_hard_passes": probe["category_minimum_hard_passes"],
        "probe_critical_safety_failures_maximum": probe["critical_safety_failures_maximum"],
        "probe_hard_pass_minimum": probe["minimum_hard_passes"],
        "probe_path": (OUT / "synthetic_probes.jsonl").relative_to(ROOT).as_posix(),
        "probe_sha256_bound_before_training": True,
        "selection_inputs": ["fresh_disjoint_synthetic_probes"],
        "stop_at_first_epoch_satisfying_every_probe_gate": True,
        "training_loss_selects_checkpoint": False,
    }
    recipe["environment_lock"] = binding(OUT / "environment.lock.json")
    recipe["full_finetune"] = training["full_finetune"]
    recipe["mechanism"] = mechanism
    recipe["namespace"] = {
        "model_root": f"build/{DATASET_ID}",
        "package": "pilot/qwen25_05b_bf16_full_finetune_successor_s6",
        "selected_bf16": f"build/{DATASET_ID}/selected/bf16",
    }
    recipe["optimizer"] = training["optimizer"]
    recipe["schedule"] = training["schedule"]
    recipe["score_policy"] = binding(OUT / "score_policy.json")
    recipe["training"].update(
        {
            "compute_dtype": training["compute_dtype"],
            "custom_loop": "seven_category_cpu_offloaded_raw_gradient_conflict_projection",
            "loss_scope": training["loss_scope"],
            "master_parameter_dtype": training["master_parameter_dtype"],
            "optimizer_moments": "float32",
        }
    )
    write_json(OUT / "training_recipe.json", recipe)

    source_contract = load(SOURCE / "freeze_contract.json")
    source_contract["claim_boundary"] = "Freeze-only S6 category-balanced conflict-projected full-parameter contract. No authority, intent, namespace, marker, job, model/evaluator execution, checkpoint, or quality result is created."
    source_contract["design_freeze"] = binding(DESIGN)
    source_contract["design_review"] = binding(DESIGN_REVIEW)
    source_contract["evaluator"]["dev"] = binding(OUT / "dev.jsonl")
    source_contract["evaluator"]["holdout"] = binding(OUT / "holdout.jsonl")
    source_contract["material_difference"] = {
        "from_s5": "Fresh package identity and fresh source-model initialization; no S5 checkpoint, response, score row, authority, intent, or backend output is reused.",
        "mechanism": mechanism["name"],
        "source_backbone": {
            "bytes": SOURCE_BACKBONE_BYTES,
            "sha256": SOURCE_BACKBONE_SHA256,
        },
        "successor_identity": DATASET_ID,
    }
    source_contract["mechanism"] = mechanism
    source_contract["resource_assumptions"]["claim_boundary"] = "Conservative prerequisites, not measured S6 usage or latency."
    source_contract["score_policy_file"] = binding(OUT / "score_policy.json")
    source_contract["selector"] = {
        "candidate_epochs": [1, 2, 3],
        "dev_access_before_lock": False,
        "no_passing_epoch_fallback_allowed": False,
        "official_dev_checkpoint_count": 1,
        "official_dev_reselection_allowed": False,
        "probe_category_minimum_hard_passes": probe["category_minimum_hard_passes"],
        "probe_count": 28,
        "probe_hard_pass_minimum": 24,
        "training_loss_selects_checkpoint": False,
    }
    source_contract["status"] = "FROZEN_MARKER_FREE_PENDING_FRESH_REVIEW"
    write_json(OUT / "freeze_contract.json", source_contract)

    frozen_config = load(ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s6/frozen_config.json")
    frozen_config.update(
        {
            "claim_boundary": "Non-executing S6 category-balanced conflict-projected full-parameter package configuration; grants no attempt authority and creates no official namespace, launch record, or marker.",
            "model_namespace": DATASET_ID,
            "package_id": "qwen25_05b_bf16_full_finetune_successor_s6",
            "teacher_mode": "none",
            "training_executable_included": True,
        }
    )
    write_json(ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s6/frozen_config.json", frozen_config)

    freeze_manifest = load(SOURCE / "freeze_manifest.json")
    freeze_manifest.update(
        {
            "attempt_authority": False,
            "attempt_marker_creation_authorized": False,
            "dataset_id": DATASET_ID,
            "detached_launch_created": False,
            "official_namespace_creation_authorized": False,
            "package_id": "qwen25_05b_bf16_full_finetune_successor_s6",
            "status": "FROZEN_MARKER_FREE_PENDING_FRESH_L2_NO_ATTEMPT_NO_TRAINING",
            "training_authority": False,
            "training_executed": False,
        }
    )
    write_json(OUT / "freeze_manifest.json", freeze_manifest)

    for path in (
        OUT / "dataset_manifest.json",
        OUT / "environment.lock.json",
        OUT / "freeze_contract.json",
        OUT / "freeze_manifest.json",
        OUT / "optimizer_schedule.json",
        OUT / "score_policy.json",
        OUT / "training_recipe.json",
        ROOT / "pilot/qwen25_05b_bf16_full_finetune_successor_s6/frozen_config.json",
    ):
        write_companion(path)


def main() -> None:
    require(OUT.is_dir(), f"S6 output directory is absent: {OUT}")
    patch, sidecar = freeze_data()
    freeze_metadata(patch, sidecar)
    print(
        json.dumps(
            {
                "category_sidecar_sha256": sha256_file(OUT / "category_sidecar.jsonl"),
                "dataset_manifest_sha256": sha256_file(OUT / "dataset_manifest.json"),
                "optimizer_schedule_sha256": sha256_file(OUT / "optimizer_schedule.json"),
                "patch_sha256": sha256_file(OUT / "patch.jsonl"),
                "status": "PASS_S6_MARKER_FREE_DATA_AND_SCHEDULE_FROZEN",
                "train_sha256": sha256_file(OUT / "train.jsonl"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
