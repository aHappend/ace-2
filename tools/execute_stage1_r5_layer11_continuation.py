#!/usr/bin/env python3
"""Create and consume the bounded position-00/layer-11 continuation authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
PACKAGE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T193501Z-"
    "r5-checkpoint-resume-preparation-0003"
)
REVIEW_ROOT = BUILD / (
    "stage1-attempt-0013-r5-checkpoint-resume-fresh-review-0004"
)
REVIEW = REVIEW_ROOT / "fresh-review.json"
AUTHORITY_ROOT = BUILD / (
    "stage1-r5-position00-layer11-continuation-0001-authority-state"
)
CONSUMPTION_ROOT = BUILD / (
    "stage1-r5-position00-layer11-continuation-0001-consumption-state"
)
RUNTIME_ROOT = BUILD / (
    "stage1-r5-position00-layer11-continuation-0001-runtime-output"
)
TERMINAL_ROOT = BUILD / (
    "stage1-r5-position00-layer11-continuation-0001-terminal-evidence"
)

PACKAGE_IDENTITY = (
    "stage1-official-rtl-backed-chat-demo-20260831T193501Z-"
    "r5-checkpoint-resume-preparation-0003"
)
AUTHORITY_IDENTITY = "ace2:stage1:r5:position-00:layer-11:continuation-0001"
UNIT_KEY = "position-00/layer-11"
NEXT_UNIT_KEY = "position-00/layer-12"
AUTHENTICATED_PREFIX = [
    f"position-00/layer-{ordinal:02d}" for ordinal in range(11)
]
COMPLETED_PREFIX = AUTHENTICATED_PREFIX + [UNIT_KEY]

PACKAGE_ROOT_SHA256 = (
    "3e083acc2b1ddfb67575610d4fa10eab9493d5aec3132dbef3f1152712b1c31e"
)
CHECKPOINT_SEED_SHA256 = (
    "cc490bd64c4e3545ce9707ffd7d19acf14e07dfd1851341e34d9366584816c01"
)
CHECKPOINT_SOURCE_SHA256 = (
    "0dbd7756f21696123108163bfdddd09fcbd19abbd6f0f922315433e8f9b35c63"
)
EXECUTOR_SOURCE_SHA256 = (
    "b27220f6beecb717ae51f1a05d2447ff4cbd0f3fd39061fc155058409ca2f287"
)
REVIEW_ROOT_SHA256 = (
    "1abcf6cc1ec7438128c715c40ae199fc6ae63a8c99f63007d80ad4d7ae1cce9b"
)
REVIEW_SHA256 = (
    "62a226f8d92e666a4d66d1ff85fef3678f582af76a7934d6a7c0bb1e4235cca8"
)
PREREQUISITE_ROOT_SHA256 = (
    "f851fd60d63372ebc73d3760b7aefd9879c60b08dcd4903cedfbc3da6c079777"
)
SEAL_FILES = {"SHA256SUMS", "TREE_ROOT.sha256"}


class ContinuationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuationError(f"JSON is unreadable: {path}") from error
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def file_record(path: Path, *, base: Path = ROOT) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    require(not os.path.lexists(temporary), f"stale temporary file: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def durable_json(path: Path, value: Any) -> None:
    durable_bytes(path, canonical_bytes(value))


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        require(
            separator == "  "
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name
            and name not in records,
            f"invalid checksum record: {line!r}",
        )
        records[name] = digest
    return records


def validate_sealed_tree(path: Path, expected_root: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"sealed tree absent: {path}")
    sums_path = path / "SHA256SUMS"
    require(
        sha256_file(sums_path) == expected_root,
        f"sealed tree root changed: {path}",
    )
    sidecar = (path / "TREE_ROOT.sha256").read_text(encoding="ascii").split()
    require(
        sidecar in ([expected_root], [expected_root, "SHA256SUMS"]),
        f"tree-root sidecar changed: {path}",
    )
    records = parse_sums(sums_path)
    observed: set[str] = set()
    for directory, names, filenames in os.walk(path, followlinks=False):
        root = Path(directory)
        for name in names:
            require(
                not (root / name).is_symlink(),
                f"directory symlink rejected: {root / name}",
            )
        for name in filenames:
            member = root / name
            require(
                member.is_file() and not member.is_symlink(),
                f"non-regular member rejected: {member}",
            )
            relative_name = member.relative_to(path).as_posix()
            if relative_name not in SEAL_FILES:
                observed.add(relative_name)
    require(observed == set(records), f"sealed member set changed: {path}")
    for name, expected in records.items():
        require(
            sha256_file(path / name) == expected,
            f"sealed member changed: {path / name}",
        )


def seal_existing_tree(path: Path) -> str:
    members = sorted(
        member
        for member in path.rglob("*")
        if member.is_file() and member.name not in SEAL_FILES
    )
    sums = "".join(
        f"{sha256_file(member)}  {member.relative_to(path).as_posix()}\n"
        for member in members
    ).encode("ascii")
    durable_bytes(path / "SHA256SUMS", sums)
    tree_root = sha256_bytes(sums)
    durable_bytes(
        path / "TREE_ROOT.sha256",
        f"{tree_root}  SHA256SUMS\n".encode("ascii"),
    )
    for member in sorted(path.rglob("*"), reverse=True):
        member.chmod(0o555 if member.is_dir() else 0o444)
    path.chmod(0o555)
    fsync_directory(path.parent)
    return tree_root


def write_sealed_tree(destination: Path, files: dict[str, bytes]) -> str:
    require(
        not os.path.lexists(destination),
        f"refusing to replace sealed tree: {destination}",
    )
    temporary = destination.with_name(destination.name + ".preparing")
    require(not os.path.lexists(temporary), f"stale preparation path: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for name, value in files.items():
            durable_bytes(temporary / name, value)
        tree_root = seal_existing_tree(temporary)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
        return tree_root
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_inputs() -> dict[str, Any]:
    validate_sealed_tree(PACKAGE, PACKAGE_ROOT_SHA256)
    validate_sealed_tree(REVIEW_ROOT, REVIEW_ROOT_SHA256)
    require(
        sha256_file(PACKAGE / "checkpoint-manifest.json")
        == CHECKPOINT_SEED_SHA256
        and sha256_file(PACKAGE / "checkpoint_resume.py")
        == CHECKPOINT_SOURCE_SHA256
        and sha256_file(PACKAGE / "production_unit_executor.py")
        == EXECUTOR_SOURCE_SHA256
        and sha256_file(REVIEW) == REVIEW_SHA256,
        "reviewed package source binding changed",
    )
    checkpoint = load(PACKAGE / "checkpoint-manifest.json")
    require(
        checkpoint.get("package_identity") == PACKAGE_IDENTITY
        and checkpoint.get("completed_unit_keys") == AUTHENTICATED_PREFIX
        and checkpoint.get("next_incomplete_unit") == UNIT_KEY
        and checkpoint.get("authenticated_prerequisite", {}).get("root_sha256")
        == PREREQUISITE_ROOT_SHA256
        and checkpoint.get("resume_state", {})
        .get("decode_state", {})
        .get("generated_token_ids")
        == [],
        "reviewed checkpoint boundary changed",
    )
    review = load(REVIEW)
    subject = review.get("subject", {})
    require(
        review.get("decision") == "PASS"
        and review.get("execution_disposition", {}).get("continuation_execution")
        == "RECOMMEND_SEPARATE_LAYER_11_CONTINUATION_AUTHORITY"
        and review.get("salvage_boundary", {}).get("authenticated_unit_keys")
        == AUTHENTICATED_PREFIX
        and review.get("salvage_boundary", {}).get("first_incomplete_unit")
        == UNIT_KEY
        and review.get("salvage_boundary", {}).get(
            "layers_00_10_execution_eligible"
        )
        is False
        and subject.get("package_tree_root_sha256") == PACKAGE_ROOT_SHA256
        and subject.get("exact_package_path") == str(PACKAGE),
        "Fresh Reviewer PASS does not bind this layer-11 continuation package",
    )
    return {
        "package": {
            "identity": PACKAGE_IDENTITY,
            "path": relative(PACKAGE),
            "tree_root_sha256": PACKAGE_ROOT_SHA256,
            "checkpoint_seed_sha256": CHECKPOINT_SEED_SHA256,
            "checkpoint_source_sha256": CHECKPOINT_SOURCE_SHA256,
            "production_executor_sha256": EXECUTOR_SOURCE_SHA256,
        },
        "fresh_review": {
            "path": relative(REVIEW),
            "sha256": REVIEW_SHA256,
            "tree_root_sha256": REVIEW_ROOT_SHA256,
            "decision": "PASS",
        },
        "prerequisite_root_sha256": PREREQUISITE_ROOT_SHA256,
    }


def validate_authority() -> dict[str, Any]:
    expected_root = (
        AUTHORITY_ROOT / "TREE_ROOT.sha256"
    ).read_text(encoding="ascii").split()[0]
    validate_sealed_tree(AUTHORITY_ROOT, expected_root)
    authority = load(AUTHORITY_ROOT / "authority.json")
    require(
        authority.get("schema") == "ace2-stage1-r5-single-unit-authority-v1"
        and authority.get("authority_identity") == AUTHORITY_IDENTITY
        and authority.get("status") == "SEALED_GRANTED_UNCONSUMED"
        and authority.get("consumable") is True
        and authority.get("grant", {}).get("permitted_unit_keys") == [UNIT_KEY]
        and authority.get("grant", {}).get("execution_cardinality") == 1
        and authority.get("authenticated_prefix") == AUTHENTICATED_PREFIX
        and authority.get("bindings", {})
        .get("package", {})
        .get("tree_root_sha256")
        == PACKAGE_ROOT_SHA256
        and authority.get("bindings", {})
        .get("fresh_review", {})
        .get("sha256")
        == REVIEW_SHA256
        and authority.get("prohibitions", {}).get("first_forbidden_unit")
        == NEXT_UNIT_KEY
        and authority.get("prohibitions", {}).get("token_generation")
        == "FORBIDDEN",
        "single-unit authority changed",
    )
    require(
        authority.get("launcher_sha256") == sha256_file(HERE),
        "authority-bound launcher source changed",
    )
    return authority


def create_authority() -> None:
    require(
        all(
            not os.path.lexists(path)
            for path in (
                AUTHORITY_ROOT,
                CONSUMPTION_ROOT,
                RUNTIME_ROOT,
                TERMINAL_ROOT,
            )
        ),
        "one or more execution namespaces already exist",
    )
    bindings = validate_inputs()
    authority = {
        "schema": "ace2-stage1-r5-single-unit-authority-v1",
        "authority_identity": AUTHORITY_IDENTITY,
        "status": "SEALED_GRANTED_UNCONSUMED",
        "consumable": True,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "authority_cardinality": 1,
        "consumption_cardinality": 0,
        "grant": {
            "scope": "EXACTLY_ONE_POSITION_00_LAYER_11_EXECUTION",
            "permitted_unit_keys": [UNIT_KEY],
            "execution_cardinality": 1,
        },
        "authenticated_prefix": AUTHENTICATED_PREFIX,
        "bindings": bindings,
        "launcher_sha256": sha256_file(HERE),
        "namespaces": {
            "authority": relative(AUTHORITY_ROOT),
            "consumption": relative(CONSUMPTION_ROOT),
            "runtime": relative(RUNTIME_ROOT),
            "terminal": relative(TERMINAL_ROOT),
        },
        "prohibitions": {
            "authenticated_prefix_recomputation": "FORBIDDEN",
            "first_forbidden_unit": NEXT_UNIT_KEY,
            "position_01_plus": "FORBIDDEN",
            "token_generation": "FORBIDDEN",
            "stage2_u280": "FORBIDDEN",
            "ppa_replay": "FORBIDDEN",
        },
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    tree_root = write_sealed_tree(
        AUTHORITY_ROOT,
        {
            "authority.json": canonical_bytes(authority),
            "command.txt": (
                "python3 -B tools/execute_stage1_r5_layer11_continuation.py "
                "--create-authority\n"
            ).encode("ascii"),
        },
    )
    validate_authority()
    print(
        "ACE2_LAYER11_AUTHORITY_CREATED "
        f"identity={AUTHORITY_IDENTITY} authority_root={tree_root}"
    )


def load_bound_module(path: Path, expected: str, name: str) -> dict[str, Any]:
    require(
        path.is_file()
        and not path.is_symlink()
        and sha256_file(path) == expected
        and stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
        f"bound source changed: {path}",
    )
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(path),
        "__name__": name,
        "__package__": None,
    }
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), namespace)
    return namespace


def prepare_runtime() -> Path:
    require(not os.path.lexists(RUNTIME_ROOT), "runtime namespace is not fresh")
    temporary = RUNTIME_ROOT.with_name(RUNTIME_ROOT.name + ".preparing")
    require(not os.path.lexists(temporary), f"stale runtime preparation: {temporary}")
    temporary.mkdir(parents=True)
    checkpoint = temporary / "checkpoint-manifest.json"
    shutil.copyfile(PACKAGE / "checkpoint-manifest.json", checkpoint)
    with checkpoint.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, RUNTIME_ROOT)
    fsync_directory(RUNTIME_ROOT.parent)
    return RUNTIME_ROOT / "checkpoint-manifest.json"


def consume_authority(authority: dict[str, Any]) -> dict[str, Any]:
    require(
        not os.path.lexists(CONSUMPTION_ROOT),
        "execution authority was already consumed",
    )
    CONSUMPTION_ROOT.mkdir(parents=True)
    fsync_directory(CONSUMPTION_ROOT.parent)
    consumed = {
        "schema": "ace2-stage1-r5-single-unit-consumption-v1",
        "authority_identity": AUTHORITY_IDENTITY,
        "authority_sha256": sha256_file(AUTHORITY_ROOT / "authority.json"),
        "consumed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "consumption_cardinality": 1,
        "permitted_unit_keys": authority["grant"]["permitted_unit_keys"],
        "package_tree_root_sha256": PACKAGE_ROOT_SHA256,
    }
    durable_json(CONSUMPTION_ROOT / "consumed.json", consumed)
    return consumed


def execute_one_unit(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], float, float]:
    runner = load_bound_module(
        PACKAGE / "checkpoint_resume.py",
        CHECKPOINT_SOURCE_SHA256,
        "ace2_r5_layer11_checkpoint",
    )
    executor_module = load_bound_module(
        PACKAGE / "production_unit_executor.py",
        EXECUTOR_SOURCE_SHA256,
        "ace2_r5_layer11_executor",
    )
    total_started = time.monotonic()
    executor = executor_module["make_production_unit_executor"](RUNTIME_ROOT)
    with runner["RunnerLock"](RUNTIME_ROOT):
        checkpoint = runner["load_json"](checkpoint_path)
        runner["validate_checkpoint"](checkpoint, RUNTIME_ROOT)
        require(
            checkpoint["completed_unit_keys"] == AUTHENTICATED_PREFIX
            and checkpoint["next_incomplete_unit"] == UNIT_KEY,
            "layer-11 is not the first incomplete authenticated unit",
        )
        ordinal = 11
        unit = checkpoint["units"][ordinal]
        publication = RUNTIME_ROOT / "published" / UNIT_KEY
        work = RUNTIME_ROOT / "work" / UNIT_KEY
        require(
            not publication.exists() and not work.exists(),
            "layer-11 execution namespace is not fresh",
        )
        dependency = runner["_dependency_bytes"](checkpoint, RUNTIME_ROOT, ordinal)
        generated = work / "generated-input.bin"
        runner["durable_bytes"](generated, dependency)
        unit_started = time.monotonic()
        execution = executor(UNIT_KEY, generated)
        unit_elapsed = time.monotonic() - unit_started
        runner["_publish_unit"](
            RUNTIME_ROOT,
            unit,
            execution,
            runner["sha256_bytes"](dependency),
            unit_elapsed,
        )
        runner["_adopt_publication"](
            checkpoint_path,
            checkpoint,
            ordinal,
        )
        runner["_prune_work_directory"](work, RUNTIME_ROOT)
        result = runner["load_json"](checkpoint_path)
        runner["validate_checkpoint"](result, RUNTIME_ROOT)
        require(
            result["completed_unit_keys"] == COMPLETED_PREFIX
            and result["next_incomplete_unit"] == NEXT_UNIT_KEY
            and result["units"][11]["validation_status"] == "PASS_AUTHENTICATED"
            and result["units"][11]["rtl_reference_agreement"] is True
            and result["units"][12]["validation_status"] == "INCOMPLETE"
            and not (RUNTIME_ROOT / "published" / NEXT_UNIT_KEY).exists()
            and not (RUNTIME_ROOT / "work" / NEXT_UNIT_KEY).exists()
            and result["resume_state"]["decode_state"]["generated_token_ids"] == [],
            "single-unit stop boundary was not preserved",
        )
    total_elapsed = time.monotonic() - total_started
    return result, unit_elapsed, total_elapsed


def publish_terminal(
    result: dict[str, Any],
    consumed: dict[str, Any],
    unit_elapsed: float,
    total_elapsed: float,
) -> str:
    require(not os.path.lexists(TERMINAL_ROOT), "terminal evidence already exists")
    validate_inputs()
    checkpoint_bytes = canonical_bytes(result)
    unit = result["units"][11]
    result_path = RUNTIME_ROOT / "published" / UNIT_KEY / "result.json"
    terminal = {
        "schema": "ace2-stage1-r5-single-unit-terminal-v1",
        "status": "PASS_SINGLE_UNIT_LAYER_11_CONTINUATION",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "authority": {
            "identity": AUTHORITY_IDENTITY,
            "consumed": True,
            "consumption_cardinality": 1,
            "consumed_record": file_record(CONSUMPTION_ROOT / "consumed.json"),
        },
        "lineage": {
            "package_identity": PACKAGE_IDENTITY,
            "package_tree_root_sha256": PACKAGE_ROOT_SHA256,
            "fresh_review_sha256": REVIEW_SHA256,
            "prerequisite_root_sha256": PREREQUISITE_ROOT_SHA256,
            "preserved_without_recomputation": AUTHENTICATED_PREFIX,
        },
        "execution": {
            "executed_unit_keys": [UNIT_KEY],
            "unit_execution_cardinality": 1,
            "reference_execution_cardinality": 1,
            "rtl_simulation_execution_cardinality": 1,
            "rtl_reference_agreement": True,
            "rtl_status": unit["evidence"]["status"],
            "integer_boundary_mismatches": unit["evidence"][
                "integer_boundary_mismatches"
            ],
            "rtl_execution_sha256": unit["evidence"]["rtl_execution_sha256"],
            "generated_token_count": 0,
            "layer_12_executed": False,
            "token_generation_executed": False,
            "ppa_executed": False,
            "u280_executed": False,
        },
        "checkpoint": {
            "immutable_snapshot": {
                "path": "checkpoint-manifest.json",
                "bytes": len(checkpoint_bytes),
                "sha256": sha256_bytes(checkpoint_bytes),
            },
            "active_checkpoint": file_record(
                RUNTIME_ROOT / "checkpoint-manifest.json"
            ),
            "completed_unit_keys": result["completed_unit_keys"],
            "next_incomplete_unit": result["next_incomplete_unit"],
            "kv_state_sha256": result["resume_state"]["kv_state_sha256"],
            "decode_state_sha256": result["resume_state"]["decode_state_sha256"],
            "validation_status": "PASS",
        },
        "unit_result": file_record(result_path),
        "timing": {
            **result["timing"],
            "layer_11_executor_wall_seconds": unit_elapsed,
            "launcher_through_checkpoint_wall_seconds": total_elapsed,
        },
        "activity": {
            "authority_consumption_cardinality": 1,
            "unit_execution_cardinality": 1,
            "rtl_execution_cardinality": 1,
            "generated_token_cardinality": 0,
            "stage1_launch_cardinality": 0,
            "ppa_cardinality": 0,
            "u280_cardinality": 0,
        },
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    terminal_root = write_sealed_tree(
        TERMINAL_ROOT,
        {
            "checkpoint-manifest.json": checkpoint_bytes,
            "consumed.json": canonical_bytes(consumed),
            "terminal.json": canonical_bytes(terminal),
            "command.txt": (
                "python3 -B tools/execute_stage1_r5_layer11_continuation.py "
                "--execute\n"
            ).encode("ascii"),
        },
    )
    durable_json(
        CONSUMPTION_ROOT / "terminal-reference.json",
        {
            "schema": "ace2-stage1-r5-single-unit-terminal-reference-v1",
            "status": terminal["status"],
            "terminal_path": relative(TERMINAL_ROOT / "terminal.json"),
            "terminal_sha256": sha256_file(TERMINAL_ROOT / "terminal.json"),
            "terminal_tree_root_sha256": terminal_root,
        },
    )
    seal_existing_tree(CONSUMPTION_ROOT)
    return terminal_root


def publish_failure(
    error: BaseException,
    consumed: dict[str, Any],
    started: float,
) -> None:
    if os.path.lexists(TERMINAL_ROOT):
        return
    failure = {
        "schema": "ace2-stage1-r5-single-unit-terminal-v1",
        "status": "FAILED_SINGLE_UNIT_LAYER_11_CONTINUATION",
        "failure_taxonomy": "RTL_REFERENCE_SINGLE_UNIT_EXECUTION_FAILURE",
        "root_cause_hypothesis": "UNRESOLVED_RECORDED_EXCEPTION",
        "regression": "POSITION_00_LAYER_11_REFERENCE_RTL_AGREEMENT",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "authority_consumed": True,
        "consumed_record": consumed,
        "elapsed_wall_seconds": time.monotonic() - started,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(
                type(error), error, error.__traceback__
            ),
        },
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    write_sealed_tree(
        TERMINAL_ROOT,
        {"terminal.json": canonical_bytes(failure)},
    )


def execute() -> None:
    require(
        not os.path.lexists(CONSUMPTION_ROOT)
        and not os.path.lexists(RUNTIME_ROOT)
        and not os.path.lexists(TERMINAL_ROOT),
        "single-unit authority was already consumed or execution state exists",
    )
    validate_inputs()
    authority = validate_authority()
    consumed = consume_authority(authority)
    started = time.monotonic()
    try:
        checkpoint_path = prepare_runtime()
        result, unit_elapsed, total_elapsed = execute_one_unit(checkpoint_path)
        terminal_root = publish_terminal(
            result,
            consumed,
            unit_elapsed,
            total_elapsed,
        )
    except BaseException as error:
        publish_failure(error, consumed, started)
        raise
    print(
        "ACE2_LAYER11_CONTINUATION_PASS "
        f"unit={UNIT_KEY} next={NEXT_UNIT_KEY} "
        f"layer_wall_seconds={unit_elapsed:.9f} "
        f"cumulative_wall_seconds="
        f"{result['timing']['cumulative_campaign_wall_seconds']:.9f} "
        f"terminal_root={terminal_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-authority", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.create_authority:
        create_authority()
    else:
        execute()


if __name__ == "__main__":
    try:
        main()
    except ContinuationError as error:
        print(f"ACE2_LAYER11_CONTINUATION_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
