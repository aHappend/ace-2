#!/usr/bin/env python3
"""Build the no-execution repair-0006 production continuation package."""

from __future__ import annotations

import hashlib
import ast
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
STEM = "stage1-official-rtl-backed-chat-demo-20260831T134355Z"
IDENTITY = (
    f"{STEM}-attempt-0014-preparation-r5-continuation-"
    "repair-0006-production-executor"
)
CANDIDATE_REVISION = "reviewer-repair-0001"
DESTINATION = BUILD / f"{IDENTITY}-{CANDIDATE_REVISION}"
VALIDATION = BUILD / f"{IDENTITY}-{CANDIDATE_REVISION}-validation"
PRIOR_CANDIDATE = BUILD / IDENTITY
PRIOR_CANDIDATE_ROOT = (
    "cbb9ba899cd8e589c317752d1199d8be91e5c04d1871009f8c497eaf1d027e2f"
)
PRIOR_CANDIDATE_MEMBER_COUNT = 33
PARENT_IDENTITY = f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0005"
PARENT = BUILD / PARENT_IDENTITY
PARENT_ROOT = "9b7c8042354911f218962a41362ea24b3a678a520a3c70e0b88d3529eff3c159"
PARENT_MEMBER_COUNT = 30
PARENT_REVIEW = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-authority-review/"
    "fresh-review.json"
)
PARENT_REVIEW_SHA256 = (
    "bac75391f5dabdab1ab926c5df23b9950515b991431fe0552f567ed8fef6ca77"
)
SOURCE_BINDINGS = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-preparation-r4/source-bindings.sha256"
)
SOURCE_BINDINGS_SHA256 = (
    "7d68a6c4800bbe01504969dcdeee84f3cc37bcd9dac2bc72dbac381970d1754b"
)
SCALE_SOURCE = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "runtime-output-0012-6f2c8d4e/tokens/position-00/layer-10/"
    "tensors/layer_output_scale_f64le.bin"
)
SCALE_SHA256 = "c8cf7401ef2ec1c3fc8c9f7314cc2546e1675cf2af26722febb05fcdc257b325"
EXECUTOR_SOURCE = ROOT / "tools/stage1_repair_0006_production_executor.py"
LAUNCHER_SOURCE = ROOT / "tools/stage1_repair_0006_atomic_launcher.py"
TEST_SOURCE = ROOT / "tools/test_attempt_0014_repair_0006_production_executor.py"
REVIEW_NAMESPACE = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-"
    "repair-0006-production-executor-authority-review"
)
AUTHORITY_NAMESPACE = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-"
    "repair-0006-production-executor-authority-state"
)
CONSUMPTION_NAMESPACE = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-"
    "repair-0006-production-executor-consumption-state"
)
RUNTIME_NAMESPACE = BUILD / (
    f"{STEM}-runtime-output-0014-r5-continuation-"
    "repair-0006-production-executor"
)
SEAL_FILES = {"SHA256SUMS", "TREE_ROOT.sha256"}
PASS_UNITS = [f"position-00/layer-{index:02d}" for index in range(11)]
PERMITTED_UNITS = [f"position-00/layer-{index:02d}" for index in range(11, 24)]
PRODUCTION_ENTRY_SOURCES = [
    "tools/rtl_arbitrary_text_generation_backend.py",
    "rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv",
    "verification/tb/ace2_layer23_v_rank1_hybrid_shell_tb.sv",
]


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any, mode: int = 0o444) -> None:
    path.write_bytes(canonical_bytes(value))
    path.chmod(mode)


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        require(
            separator == "  "
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and relative
            and relative not in records,
            f"invalid checksum record: {line!r}",
        )
        records[relative] = digest
    return records


def authenticate_tree(path: Path, expected_root: str, expected_count: int) -> None:
    require(path.is_dir() and not path.is_symlink(), f"sealed tree absent: {path}")
    records = parse_sums(path / "SHA256SUMS")
    require(len(records) == expected_count, f"sealed member count changed: {path}")
    require(
        sha256_file(path / "SHA256SUMS") == expected_root,
        f"sealed tree root changed: {path}",
    )
    require(
        (path / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{expected_root}  SHA256SUMS",
        f"sealed tree sidecar changed: {path}",
    )
    for relative, expected in records.items():
        member = path / relative
        require(
            member.is_file()
            and not member.is_symlink()
            and sha256_file(member) == expected,
            f"sealed member changed: {member}",
        )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def production_source_records() -> dict[str, str]:
    records = parse_sums(SOURCE_BINDINGS)
    pending = ["rtl_arbitrary_text_generation_backend"]
    visited: set[str] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = ROOT / "tools" / f"{module}.py"
        require(path.is_file(), f"local production module is absent: {module}")
        relative_path = path.relative_to(ROOT).as_posix()
        records[relative_path] = sha256_file(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        pending.extend(
            name
            for name in imported
            if (ROOT / "tools" / f"{name}.py").is_file()
        )
    for relative_path in PRODUCTION_ENTRY_SOURCES:
        path = ROOT / relative_path
        require(path.is_file(), f"production source is absent: {relative_path}")
        records[relative_path] = sha256_file(path)
    for relative_path, expected in records.items():
        path = ROOT / relative_path
        require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == expected,
            f"production source differs from bound bytes: {relative_path}",
        )
    return dict(sorted(records.items()))


def authenticate_parent() -> dict[str, Any]:
    authenticate_tree(PARENT, PARENT_ROOT, PARENT_MEMBER_COUNT)
    require(
        PARENT_REVIEW.is_file()
        and not PARENT_REVIEW.is_symlink()
        and sha256_file(PARENT_REVIEW) == PARENT_REVIEW_SHA256,
        "repair-0005 Fresh Review receipt changed",
    )
    review = load(PARENT_REVIEW)
    require(
        review["decision"] == "PASS"
        and review["independent"] is True
        and review["review_execution"]["model_executed"] is False
        and review["review_execution"]["rtl_executed"] is False,
        "repair-0005 reviewed parent semantics changed",
    )
    checkpoint = load(PARENT / "checkpoint-manifest.json")
    layer10 = checkpoint["units"][10]
    result_path = ROOT / layer10["result"]["path"]
    require(
        layer10["key"] == PASS_UNITS[-1]
        and layer10["validation_status"] == "PASS_AUTHENTICATED"
        and result_path.is_file()
        and sha256_file(result_path) == layer10["result"]["sha256"],
        "repair-0005 layer-10 evidence chain changed",
    )
    result = load(result_path)
    require(
        result["tensors"]["layer_output_scale_f64le.bin"]["sha256"] == SCALE_SHA256
        and SCALE_SOURCE.is_file()
        and not SCALE_SOURCE.is_symlink()
        and sha256_file(SCALE_SOURCE) == SCALE_SHA256,
        "authenticated layer-10 scale recovery failed",
    )
    require(
        SOURCE_BINDINGS.is_file()
        and sha256_file(SOURCE_BINDINGS) == SOURCE_BINDINGS_SHA256,
        "reviewed production source-binding index changed",
    )
    return {
        "identity": PARENT_IDENTITY,
        "path": relative(PARENT),
        "sealed_member_count": PARENT_MEMBER_COUNT,
        "tree_root_sha256": PARENT_ROOT,
        "fresh_review": {
            **file_record(PARENT_REVIEW),
            "decision": "PASS",
            "execution_performed": False,
        },
    }


def patched_runner() -> bytes:
    source = (PARENT / "continuation_runner.py").read_text(encoding="utf-8")
    require(
        source.count("continuation-repair-0005") == 1
        and source.count(
            "0baed352fa192aff3578d8e5f8b7eac39b2ad42726fb0f41fbfbc032f54a03c9"
        )
        == 1,
        "reviewed runner identity constants changed",
    )
    source = source.replace(
        "continuation-repair-0005",
        "continuation-repair-0006-production-executor",
    ).replace(
        "0baed352fa192aff3578d8e5f8b7eac39b2ad42726fb0f41fbfbc032f54a03c9",
        PARENT_ROOT,
    )
    geometry_gate = '''    _require(
        geometry.get("positions") == 1
        and geometry.get("layers_per_position") == 24
        and len(units) == 24,
        "checkpoint geometry is not the bounded 24-layer position",
    )
'''
    boundary_gate = geometry_gate + '''    _require(
        [unit.get("key") for unit in units[:11]]
        == [f"position-00/layer-{index:02d}" for index in range(11)]
        and all(
            unit.get("validation_status") == "PASS_AUTHENTICATED"
            for unit in units[:11]
        ),
        "authenticated layers 00 through 10 changed",
    )
'''
    require(geometry_gate in source, "reviewed runner geometry gate changed")
    source = source.replace(geometry_gate, boundary_gate, 1)
    execution_line = "                execution = executor(key, generated)\n"
    guarded_execution = (
        "                _require(\n"
        "                    key in {f\"position-00/layer-{index:02d}\" for index in range(11, 24)},\n"
        "                    f\"executor selection escaped layers 11 through 23: {key}\",\n"
        "                )\n"
        + execution_line
    )
    require(execution_line in source, "reviewed runner execution call changed")
    source = source.replace(execution_line, guarded_execution, 1)
    return source.encode("utf-8")


def seal_package(package: Path) -> str:
    records = []
    for path in sorted(package.rglob("*")):
        require(not path.is_symlink(), f"package symlink: {path}")
        if path.is_file() and path.name not in SEAL_FILES:
            records.append(
                f"{sha256_file(path)}  {path.relative_to(package).as_posix()}"
            )
    sums = package / "SHA256SUMS"
    sums.write_text("\n".join(records) + "\n", encoding="ascii")
    sums.chmod(0o444)
    root = sha256_file(sums)
    sidecar = package / "TREE_ROOT.sha256"
    sidecar.write_text(f"{root}  SHA256SUMS\n", encoding="ascii")
    sidecar.chmod(0o444)
    return root


def validate_package() -> dict[str, Any]:
    authenticate_parent()
    authenticate_tree(
        PRIOR_CANDIDATE,
        PRIOR_CANDIDATE_ROOT,
        PRIOR_CANDIDATE_MEMBER_COUNT,
    )
    records = parse_sums(DESTINATION / "SHA256SUMS")
    actual = {
        path.relative_to(DESTINATION).as_posix()
        for path in DESTINATION.rglob("*")
        if path.is_file() and path.name not in SEAL_FILES
    }
    require(actual == set(records), "repair-0006 sealed member set changed")
    for relative_path, expected in records.items():
        require(
            sha256_file(DESTINATION / relative_path) == expected,
            f"repair-0006 sealed member changed: {relative_path}",
        )
    root = sha256_file(DESTINATION / "SHA256SUMS")
    manifest = load(DESTINATION / "successor-manifest.json")
    authority = load(DESTINATION / "authority.json")
    checkpoint = load(DESTINATION / "checkpoint-manifest.json")
    require(
        manifest["identity"] == IDENTITY
        and manifest["parent_evidence"]["tree_root_sha256"] == PARENT_ROOT
        and manifest["selectable_unit_executor"]["name"]
        == "ProductionUnitExecutor"
        and manifest["permitted_unit_keys"] == PERMITTED_UNITS
        and manifest["first_incomplete_unit"] == PERMITTED_UNITS[0]
        and manifest["future_namespaces"]["consumption"]
        == relative(CONSUMPTION_NAMESPACE)
        and manifest["future_namespaces"]["runtime"]
        == relative(RUNTIME_NAMESPACE),
        "repair-0006 executor or boundary binding changed",
    )
    require(
        checkpoint["completed_unit_keys"] == PASS_UNITS
        and checkpoint["next_incomplete_unit"] == PERMITTED_UNITS[0]
        and checkpoint["resume_state"]["hidden_scale"]["sha256"] == SCALE_SHA256
        and checkpoint["continuation_identity"]["package"] == IDENTITY
        and checkpoint["continuation_identity"][
            "sealed_repair_predecessor_tree_root"
        ]
        == PARENT_ROOT,
        "repair-0006 checkpoint boundary changed",
    )
    require(
        authority["status"] == "PENDING_FRESH_REVIEW"
        and authority["consumable"] is False
        and set(authority["activity"].values()) == {0},
        "repair-0006 no-execution authority gate changed",
    )
    for path in (
        REVIEW_NAMESPACE,
        AUTHORITY_NAMESPACE,
        CONSUMPTION_NAMESPACE,
        RUNTIME_NAMESPACE,
    ):
        require(not os.path.lexists(path), f"future namespace already exists: {path}")
    command = [
        sys.executable,
        "-B",
        str(DESTINATION / "atomic_launcher.py"),
        "--verify-only",
    ]
    verified = subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(
        verified.stderr == ""
        and "authority=0 consumption=0 submission=0 model=0 rtl=0 output=0"
        in verified.stdout,
        "atomic launcher no-execution verification failed",
    )
    tested = subprocess.run(
        [sys.executable, "-B", str(DESTINATION / "no_execution_tests.py"), "-q"],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require("OK" in tested.stderr, "focused no-execution tests did not pass")
    require(
        sha256_file(PARENT / "SHA256SUMS") == PARENT_ROOT,
        "repair-0005 changed during repair-0006 validation",
    )
    return {
        "activity": authority["activity"],
        "atomic_launcher": "PASS",
        "candidate_authority_root_sha256": root,
        "external_executor_substitution": "FAIL_CLOSED",
        "focused_test_count": 6,
        "namespace_authority_binding": "PASS_FAIL_CLOSED",
        "parent_authentication": "PASS_IMMUTABLE_REVIEWED",
        "permitted_unit_keys": PERMITTED_UNITS,
        "preserved_authenticated_unit_keys": PASS_UNITS,
        "production_unit_executor": "BOUND_NON_NULL",
        "status": "PASS",
    }


def build() -> None:
    require(not DESTINATION.exists(), f"refusing to overwrite: {DESTINATION}")
    require(not VALIDATION.exists(), f"refusing to overwrite: {VALIDATION}")
    authenticate_tree(
        PRIOR_CANDIDATE,
        PRIOR_CANDIDATE_ROOT,
        PRIOR_CANDIDATE_MEMBER_COUNT,
    )
    for path in (
        REVIEW_NAMESPACE,
        AUTHORITY_NAMESPACE,
        CONSUMPTION_NAMESPACE,
        RUNTIME_NAMESPACE,
    ):
        require(not os.path.lexists(path), f"future namespace is not fresh: {path}")
    parent = authenticate_parent()
    temporary = DESTINATION.with_name(DESTINATION.name + ".building")
    require(not temporary.exists(), f"stale construction path: {temporary}")
    try:
        temporary.mkdir(parents=True)
        shutil.copytree(PARENT / "salvage-state", temporary / "salvage-state")
        shutil.copyfile(SCALE_SOURCE, temporary / "salvage-state/layer-10-scale-f64le.bin")
        source_records = production_source_records()
        (temporary / "production-source-bindings.sha256").write_text(
            "".join(
                f"{digest}  {relative_path}\n"
                for relative_path, digest in source_records.items()
            ),
            encoding="ascii",
        )
        (temporary / "continuation_runner.py").write_bytes(patched_runner())
        shutil.copyfile(EXECUTOR_SOURCE, temporary / "production_unit_executor.py")
        shutil.copyfile(LAUNCHER_SOURCE, temporary / "atomic_launcher.py")
        shutil.copyfile(TEST_SOURCE, temporary / "no_execution_tests.py")
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        (temporary / "atomic_launcher.py").chmod(0o555)

        checkpoint = load(PARENT / "checkpoint-manifest.json")
        checkpoint["continuation_identity"] = {
            "package": IDENTITY,
            "sealed_repair_predecessor_tree_root": PARENT_ROOT,
        }
        checkpoint["resume_state"]["hidden_scale"] = {
            "bytes": 8,
            "path": "salvage-state/layer-10-scale-f64le.bin",
            "sha256": SCALE_SHA256,
            "authentication": (
                "HASH_BOUND_BY_REPAIR_0005_LAYER10_RESULT_AND_MATCHED_"
                "IMMUTABLE_RETAINED_BYTES"
            ),
        }
        write_json(temporary / "checkpoint-manifest.json", checkpoint)
        members = {
            "atomic_launcher": {
                "path": "atomic_launcher.py",
                "sha256": sha256_file(temporary / "atomic_launcher.py"),
            },
            "continuation_runner": {
                "path": "continuation_runner.py",
                "sha256": sha256_file(temporary / "continuation_runner.py"),
            },
            "production_unit_executor": {
                "path": "production_unit_executor.py",
                "sha256": sha256_file(temporary / "production_unit_executor.py"),
            },
        }
        manifest = {
            "schema": "ace2-stage1-repair-0006-production-executor-package-v1",
            "identity": IDENTITY,
            "status": "SEALED_PENDING_FRESH_REVIEW",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "parent_evidence": parent,
            "authenticated_completed_unit_keys": PASS_UNITS,
            "first_incomplete_unit": PERMITTED_UNITS[0],
            "permitted_unit_keys": PERMITTED_UNITS,
            "prohibited_unit_keys": PASS_UNITS,
            "selectable_unit_executor": {
                "name": "ProductionUnitExecutor",
                "factory": "make_production_unit_executor",
                **members["production_unit_executor"],
                "external_substitution": "FORBIDDEN",
            },
            "execution_members": members,
            "production_source_closure": {
                "path": "production-source-bindings.sha256",
                "sha256": sha256_file(
                    temporary / "production-source-bindings.sha256"
                ),
                "required_members": list(source_records),
                "validation": "EXACT_SHA256_BEFORE_AUTHORITY_CONSUMPTION",
            },
            "unit_descriptor_contract": {
                "layer_11_input": "AUTHENTICATED_RAW_S8_896_PLUS_BOUND_F64_SCALE",
                "layers_12_through_23_input": "ace2-stage1-unit-state-v1",
                "output": "ace2-stage1-unit-state-v1",
                "position": 0,
                "first_layer": 11,
                "last_layer": 23,
            },
            "atomic_consumption": {
                "transition": "PREPARED_FILE_FSYNC_THEN_EXCLUSIVE_HARD_LINK",
                "duplicate_start": "REFUSED",
                "interruption_recovery": (
                    "SAME_AUTHORITY_AND_CONSUMPTION_RECORD_ONLY"
                ),
                "runner_semantics": "PER_UNIT_ATOMIC_PUBLICATION_AND_ADOPTION",
                "authority_bound_consumption_namespace": relative(
                    CONSUMPTION_NAMESPACE
                ),
                "authority_bound_runtime_namespace": relative(RUNTIME_NAMESPACE),
            },
            "future_namespaces": {
                "fresh_review": relative(REVIEW_NAMESPACE),
                "execution_authority": relative(AUTHORITY_NAMESPACE),
                "consumption": relative(CONSUMPTION_NAMESPACE),
                "runtime": relative(RUNTIME_NAMESPACE),
            },
        }
        write_json(temporary / "successor-manifest.json", manifest)
        authority = {
            "schema": "ace2-stage1-repair-0006-authority-candidate-v1",
            "status": "PENDING_FRESH_REVIEW",
            "consumable": False,
            "unit_execution_authority": "NOT_GRANTED",
            "separate_fresh_review_required": True,
            "separate_execution_authority_required": True,
            "activity": {
                "authority_cardinality": 0,
                "consumption_cardinality": 0,
                "model_execution_cardinality": 0,
                "output_cardinality": 0,
                "rtl_execution_cardinality": 0,
                "submission_cardinality": 0,
            },
            "first_incomplete_unit": PERMITTED_UNITS[0],
            "permitted_unit_keys": PERMITTED_UNITS,
            "authenticated_completed_unit_keys": PASS_UNITS,
        }
        write_json(temporary / "authority.json", authority)
        write_json(
            temporary / "review-contract.json",
            {
                "schema": "ace2-stage1-repair-0006-fresh-review-request-v1",
                "status": "PENDING_FRESH_REVIEW",
                "execution_permitted_during_review": False,
                "fresh_review_namespace": relative(REVIEW_NAMESPACE),
                "checks": [
                    "authenticate repair-0005 tree and independent PASS receipt",
                    "verify production executor and atomic launcher exact hashes",
                    "verify only position-00/layers-11-23 can be selected",
                    "verify layers-00-10 checkpoint records remain unchanged",
                    "verify atomic consumption, interruption recovery, and duplicate refusal",
                    "verify authority binds the sole canonical consumption and runtime namespaces",
                    "verify alternate namespace reuse fails before authority consumption",
                    "verify external executor substitution fails closed",
                    "verify all real activity cardinalities remain zero",
                ],
            },
        )
        seal_package(temporary)
        os.replace(temporary, DESTINATION)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    result = validate_package()
    VALIDATION.mkdir(parents=True)
    report = {
        **result,
        "schema": "ace2-stage1-repair-0006-no-execution-validation-v1",
        "candidate_path": relative(DESTINATION),
        "validation_mode": "FOCUSED_FAKE_EXECUTOR_NO_MODEL_OR_RTL",
        "execution_performed": False,
        "model_executed": False,
        "rtl_executed": False,
        "authority_consumed": False,
        "validator_sha256": sha256_file(Path(__file__)),
    }
    write_json(VALIDATION / "validation.json", report)
    digest = sha256_file(VALIDATION / "validation.json")
    (VALIDATION / "validation.json.sha256").write_text(
        f"{digest}  validation.json\n", encoding="ascii"
    )
    (VALIDATION / "validation.json.sha256").chmod(0o444)
    print(
        "ACE2_REPAIR_0006_PRODUCTION_EXECUTOR_PASS "
        f"authority_root={result['candidate_authority_root_sha256']} "
        "first_incomplete=position-00/layer-11 "
        "authority=0 consumption=0 submission=0 model=0 rtl=0 output=0 "
        "consumable=false review=PENDING_FRESH_REVIEW"
    )


def check() -> None:
    result = validate_package()
    print(
        "ACE2_REPAIR_0006_PRODUCTION_EXECUTOR_PASS "
        f"authority_root={result['candidate_authority_root_sha256']} "
        "first_incomplete=position-00/layer-11 "
        "authority=0 consumption=0 submission=0 model=0 rtl=0 output=0 "
        "consumable=false review=PENDING_FRESH_REVIEW"
    )


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
    elif not sys.argv[1:]:
        build()
    else:
        raise BuildError(
            "usage: build_attempt_0014_repair_0006_production_executor.py [--check]"
        )


if __name__ == "__main__":
    main()
