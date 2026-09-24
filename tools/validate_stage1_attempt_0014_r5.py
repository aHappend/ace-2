#!/usr/bin/env python3
"""Validate the unauthorized Stage-1 lifecycle-repair r5 successor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
SOURCE_PACKAGE = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-preparation-r4"
)
SOURCE_STATE = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-authority-state"
)
SOURCE_OUTPUT = ROOT / (
    "build/ace2_chat_demo/"
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "runtime-output-0013-a62e0c91"
)
EXCLUDED = {"SHA256SUMS", "TREE_ROOT.sha256"}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat(follow_symlinks=False).st_size,
            "mode": stat.S_IMODE(item.stat(follow_symlinks=False).st_mode),
            "sha256": digest(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum line: {line!r}")
        assert match is not None
        require(match[2] not in records, f"duplicate checksum path: {match[2]}")
        records[match[2]] = match[1]
    return records


def validate_closure() -> None:
    sums = parse_sums(PACKAGE / "SHA256SUMS")
    actual = {
        item.relative_to(PACKAGE).as_posix()
        for item in PACKAGE.rglob("*")
        if item.is_file()
        and item.relative_to(PACKAGE).as_posix() not in EXCLUDED
        and "__pycache__" not in item.parts
    }
    require(actual == set(sums), "candidate package file set changed")
    for relative, expected in sums.items():
        path = PACKAGE / relative
        require(not path.is_symlink(), f"candidate symlink: {relative}")
        require(digest(path) == expected, f"candidate member changed: {relative}")
        expected_mode = 0o555 if path.suffix in {".py", ".sh"} else 0o444
        require(
            stat.S_IMODE(path.stat().st_mode) == expected_mode,
            f"candidate mode changed: {relative}",
        )
    root = digest(PACKAGE / "SHA256SUMS")
    require(
        (PACKAGE / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{root}  SHA256SUMS",
        "candidate package root changed",
    )


def validate() -> None:
    candidate = load(PACKAGE / "package.json")
    commands = load(PACKAGE / "commands.json")
    preservation = load(PACKAGE / "attempt-0013-terminal-seal.json")
    regression = load(PACKAGE / "lifecycle-regression.json")
    mechanism = load(PACKAGE / "failure-mechanism.json")
    review = load(PACKAGE / "review-contract.json")
    require(candidate.get("attempt") == "0014", "successor attempt changed")
    require(
        candidate.get("status") == "PREPARED_UNAUTHORIZED_UNCONSUMED"
        and candidate.get("current_authority_cardinality") == 0
        and candidate.get("future_authority_cardinality") == 0
        and candidate.get("execution_occurred") is False,
        "successor is not unauthorized and unconsumed",
    )
    require(
        commands.get("submitted") is False
        and commands.get("run_id_assigned") is None
        and commands.get("authorization_state") == "ABSENT",
        "successor command state is consumed or authorized",
    )
    for key in (
        "future_authority_state",
        "future_output",
        "future_review_receipt_namespace",
    ):
        require(
            not os.path.lexists(ROOT / candidate[key]),
            f"successor namespace is not fresh: {candidate[key]}",
        )
    require(
        preservation.get("status") == "AUTHENTICATED_SEALED_IMMUTABLE"
        and preservation.get("before_after_byte_exact") is True,
        "attempt-0013 seal changed",
    )
    require(
        manifest(SOURCE_PACKAGE) == preservation["package"]["files"],
        "attempt-0013 package changed",
    )
    require(
        manifest(SOURCE_STATE) == preservation["authority_state"]["files"],
        "attempt-0013 authority state changed",
    )
    require(
        manifest(SOURCE_OUTPUT) == preservation["runtime_output"]["files"],
        "attempt-0013 runtime output changed",
    )
    terminal = load(SOURCE_STATE / "terminal-status.json")
    require(
        terminal.get("status") == "SEALED_TERMINAL_NO_RETRY"
        and terminal.get("classification") == "MODEL_EXCEPTION_FileNotFoundError"
        and terminal.get("natural_terminal") is False
        and terminal.get("second_model_invocation_permitted") is False
        and terminal["child_lifecycle"].get("process_group_empty") is True
        and terminal["child_lifecycle"].get("capture_drains_finished") is True,
        "attempt-0013 terminal contract changed",
    )
    expected_layers = [f"layer-{layer:02d}" for layer in range(11)]
    require(
        preservation["runtime_output"]["position_00_layers"] == expected_layers,
        "attempt-0013 11-layer extent changed",
    )
    for receipt in preservation["durable_receipts"].values():
        path = ROOT / receipt["path"]
        require(
            path.is_file()
            and path.stat().st_size == receipt["bytes"]
            and digest(path) == receipt["sha256"],
            f"attempt-0013 durable receipt changed: {receipt['path']}",
        )
    require(
        mechanism.get("status") == "DETERMINISTICALLY_REPRODUCED_NO_EXECUTION"
        and mechanism.get("original_exact_missing_path")
        == "NOT_DURABLY_CAPTURED_BY_ATTEMPT_0013"
        and mechanism["reproduction"]["exception_type"] == "FileNotFoundError"
        and mechanism["reproduction"]["phase"] == "runtime_output_size_check",
        "missing-path mechanism record changed",
    )
    require(
        regression.get("status") == "PASS"
        and regression.get("official") is False
        and regression.get("model_executed") is False
        and regression.get("rtl_executed") is False
        and regression.get("layer_count") == 24
        and regression.get("layer_ids") == list(range(24))
        and regression.get("create_use_prune_order_exact") is True,
        "24-layer inert lifecycle regression changed",
    )
    negatives = regression["negative_controls"]
    require(
        negatives["premature_prune"]["status"] == "REJECTED"
        and negatives["missing_generated_input"]["status"] == "REJECTED"
        and negatives["exception_record_publication_failure"]["status"]
        == "PASS_FALLBACK_DURABLE_BEFORE_TERMINAL",
        "lifecycle negative controls changed",
    )
    require(
        review.get("status") == "PENDING_INDEPENDENT_NO_EXECUTION_REVIEW"
        and review.get("authority_granted_by_review") is False
        and review.get("execution_permitted_during_review") is False,
        "review gate changed",
    )
    validate_closure()


if __name__ == "__main__":
    validate()
    print("ACE2_STAGE1_ATTEMPT_0014_R5_VALID")

