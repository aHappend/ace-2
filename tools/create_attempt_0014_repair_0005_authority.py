#!/usr/bin/env python3
"""Create or validate the bounded repair-0005 continuation authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent if HERE.parent.name == "tools" else HERE.parent.parents[2]
BUILD = ROOT / "build/ace2_chat_demo"
STEM = "stage1-official-rtl-backed-chat-demo-20260831T134355Z"
CANDIDATE_IDENTITY = (
    f"{STEM}-attempt-0014-preparation-r5-continuation-repair-0005"
)
CANDIDATE = BUILD / CANDIDATE_IDENTITY
REVIEW = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-authority-review/"
    "fresh-review.json"
)
AUTHORITY = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-authority-state"
)
CONSUMPTION = BUILD / (
    f"{STEM}-attempt-0014-r5-continuation-repair-0005-consumption-state"
)
RUNTIME = BUILD / (
    f"{STEM}-runtime-output-0014-r5-continuation-repair-0005"
)
ATTEMPT_0012_TERMINAL = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260829T175516Z-"
    "attempt-0012-6f2c8d4e-authority-state/terminal-status.json"
)
ATTEMPT_0013_TERMINAL = BUILD / (
    "stage1-official-rtl-backed-chat-demo-20260831T122000Z-"
    "attempt-0013-a62e0c91-authority-state/terminal-status.json"
)

AUTHORITY_IDENTITY = (
    "ace2:stage1:attempt-0014:r5:repair-0005:"
    "position-00:layers-11-23:continuation-authority-0001"
)
CANDIDATE_ROOT = (
    "9b7c8042354911f218962a41362ea24b3a678a520a3c70e0b88d3529eff3c159"
)
CANDIDATE_MEMBER_COUNT = 30
REVIEW_SHA256 = (
    "bac75391f5dabdab1ab926c5df23b9950515b991431fe0552f567ed8fef6ca77"
)
SUCCESSOR_MANIFEST_SHA256 = (
    "479e777a646d34aa8bfd2d04dbc01e94fd2351bc7aab3ab0cfdbe55ed2ae6def"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "9ff84bd2e6c1b0bc1c754ff58486a5a27835865cc1320c6904db73668977b214"
)
RUNTIME_CLOSURE_SHA256 = (
    "0cce16a7a3b975f86c13ad3fa42ca72c302033c3ce65d53b4b12fd6f49a27e5a"
)
SOURCE_LAUNCH_SHA256 = (
    "42a2c7640e47ca25cda5bc644ce417f9fcf9aa90cf493de5750110be441be128"
)
CONTINUATION_RUNNER_SHA256 = (
    "b6a888cbcb73678699361cc81ce820e513c0cc4ab4255a5e01e8f94e835388b2"
)
ATTEMPT_0012_TERMINAL_SHA256 = (
    "41943596b1065c887b1fbb638f413b639f13271b9618206151a5d438394e110a"
)
ATTEMPT_0013_TERMINAL_SHA256 = (
    "1ae9c3988259155394eab35471f9234f44558309c894d8aa628dedb11a228d25"
)
SEAL_FILES = {"SHA256SUMS", "TREE_ROOT.sha256"}
PASS_UNITS = [f"position-00/layer-{index:02d}" for index in range(11)]
PERMITTED_UNITS = [f"position-00/layer-{index:02d}" for index in range(11, 24)]
ACTIVITY = {
    "authority_creation_cardinality": 1,
    "authorized_unit_execution_cardinality": 0,
    "consumption_cardinality": 0,
    "continuation_entrypoint_invocation_cardinality": 0,
    "model_execution_cardinality": 0,
    "output_cardinality": 0,
    "rtl_execution_cardinality": 0,
    "submission_cardinality": 0,
}


class AuthorityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


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


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum record: {line!r}")
        assert match is not None
        require(match[2] not in records, f"duplicate checksum member: {match[2]}")
        records[match[2]] = match[1]
    return records


def regular_file_set(path: Path) -> set[str]:
    files: set[str] = set()
    for directory, names, filenames in os.walk(path, followlinks=False):
        root = Path(directory)
        for name in names:
            child = root / name
            require(not child.is_symlink(), f"directory symlink rejected: {child}")
        for name in filenames:
            child = root / name
            require(
                child.is_file() and not child.is_symlink(),
                f"non-regular member rejected: {child}",
            )
            files.add(child.relative_to(path).as_posix())
    return files


def validate_sealed_tree(
    path: Path, expected_root: str, expected_members: int
) -> None:
    require(path.is_dir() and not path.is_symlink(), f"sealed tree absent: {path}")
    records = parse_sums(path / "SHA256SUMS")
    require(
        len(records) == expected_members,
        f"sealed member count changed: {path}",
    )
    require(
        sha256_file(path / "SHA256SUMS") == expected_root,
        f"sealed tree root changed: {path}",
    )
    require(
        (path / "TREE_ROOT.sha256").read_text(encoding="ascii").strip()
        == f"{expected_root}  SHA256SUMS",
        f"sealed tree root sidecar changed: {path}",
    )
    actual = regular_file_set(path) - SEAL_FILES
    require(actual == set(records), f"sealed tree closure changed: {path}")
    for name, expected in records.items():
        require(
            sha256_file(path / name) == expected,
            f"sealed member changed: {path / name}",
        )


def authenticate_prerequisites(*, require_pristine_runtime: bool) -> None:
    validate_sealed_tree(CANDIDATE, CANDIDATE_ROOT, CANDIDATE_MEMBER_COUNT)
    exact_files = {
        "successor-manifest.json": SUCCESSOR_MANIFEST_SHA256,
        "checkpoint-manifest.json": CHECKPOINT_MANIFEST_SHA256,
        "runtime-closure.json": RUNTIME_CLOSURE_SHA256,
        "source_launch.py": SOURCE_LAUNCH_SHA256,
        "continuation_runner.py": CONTINUATION_RUNNER_SHA256,
    }
    for name, expected in exact_files.items():
        require(
            sha256_file(CANDIDATE / name) == expected,
            f"candidate prerequisite changed: {name}",
        )

    successor = load(CANDIDATE / "successor-manifest.json")
    checkpoint = load(CANDIDATE / "checkpoint-manifest.json")
    candidate_authority = load(CANDIDATE / "authority.json")
    require(
        successor["identity"] == CANDIDATE_IDENTITY
        and successor["authenticated_pass_count"] == 11
        and successor["first_incomplete_unit"] == PERMITTED_UNITS[0]
        and successor["stage1"] == "OPEN"
        and successor["stage2"] == "FORBIDDEN",
        "successor identity or salvage boundary changed",
    )
    require(
        checkpoint["completed_unit_keys"] == PASS_UNITS
        and checkpoint["next_incomplete_unit"] == PERMITTED_UNITS[0]
        and checkpoint["validation"]["status"] == "PASS"
        and checkpoint["validation"]["salvage_boundary"] == PASS_UNITS[-1]
        and checkpoint["validation"][
            "authenticated_completed_units_must_not_be_recomputed"
        ]
        is True,
        "checkpoint authenticated prefix changed",
    )
    require(
        candidate_authority["status"] == "PENDING_FRESH_REVIEW"
        and candidate_authority["consumable"] is False
        and set(candidate_authority["activity"].values()) == {0},
        "reviewed candidate state changed",
    )

    require(
        REVIEW.is_file()
        and not REVIEW.is_symlink()
        and sha256_file(REVIEW) == REVIEW_SHA256,
        "Fresh Reviewer receipt changed",
    )
    review = load(REVIEW)
    require(
        review["decision"] == "PASS"
        and review["decision_identity"] == "FRESH_REVIEWER_DECISION"
        and review["independent"] is True
        and review["review_execution"]["mode"]
        == "FRESH_INDEPENDENT_NO_EXECUTION"
        and review["review_execution"]["candidate_code_invoked"] is False
        and review["review_execution"]["model_executed"] is False
        and review["review_execution"]["rtl_executed"] is False
        and review["decision_binding"]["candidate_authority_root_sha256"]
        == CANDIDATE_ROOT
        and review["decision_binding"]["successor_manifest_sha256"]
        == SUCCESSOR_MANIFEST_SHA256
        and review["salvage_boundary"]["authenticated_unit_keys"] == PASS_UNITS
        and review["salvage_boundary"]["first_incomplete_unit"]
        == PERMITTED_UNITS[0]
        and review["salvage_boundary"]["incomplete_count"] == 13
        and review["later_continuation_authority_task"]["start_unit"]
        == PERMITTED_UNITS[0]
        and review["later_continuation_authority_task"][
            "must_not_recompute_authenticated_units"
        ]
        == PASS_UNITS
        and set(review["activity"].values()) == {False, 0},
        "Fresh Reviewer PASS scope or zero-activity receipt changed",
    )

    for path, expected in (
        (ATTEMPT_0012_TERMINAL, ATTEMPT_0012_TERMINAL_SHA256),
        (ATTEMPT_0013_TERMINAL, ATTEMPT_0013_TERMINAL_SHA256),
    ):
        require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == expected
            and load(path)["status"] == "SEALED_TERMINAL_NO_RETRY",
            f"terminal attempt receipt changed: {path}",
        )
    if require_pristine_runtime:
        for path in (CONSUMPTION, RUNTIME):
            require(
                not os.path.lexists(path),
                f"zero-activity namespace is not pristine: {path}",
            )


def interface_record() -> dict[str, Any]:
    successor = load(CANDIDATE / "successor-manifest.json")
    runtime = load(CANDIDATE / "runtime-closure.json")
    return {
        "authority_identity": AUTHORITY_IDENTITY,
        "authorized_callable": {
            "call_signature": (
                "run_continuation(checkpoint_path, production_unit_executor)"
            ),
            "module_path": relative(CANDIDATE / "continuation_runner.py"),
            "module_sha256": CONTINUATION_RUNNER_SHA256,
            "name": "run_continuation",
        },
        "candidate_source_gate": {
            "load_policy": "READ_BYTES_COMPILE_EXEC_EXACT_BOUND_SOURCE",
            "path": relative(CANDIDATE / "source_launch.py"),
            "sha256": SOURCE_LAUNCH_SHA256,
        },
        "checkpoint_seed": {
            "path": relative(CANDIDATE / "checkpoint-manifest.json"),
            "sha256": CHECKPOINT_MANIFEST_SHA256,
        },
        "consumption_state_namespace": relative(CONSUMPTION),
        "entrypoint_invoked_during_authority_creation": False,
        "executor_contract": {
            "generated_input_must_be_consumed": True,
            "required_evidence": {"rtl_reference_agreement": True},
            "required_validation_status": "PASS_RTL_REFERENCE_AUTHENTICATED",
            "selectable_implementation_in_candidate": False,
        },
        "first_call_unit": PERMITTED_UNITS[0],
        "interpreter": runtime["interpreter"],
        "launch_flags": successor["launch"]["argv"][1:6],
        "permitted_unit_keys": PERMITTED_UNITS,
        "recovery": (
            "SAME_AUTHORITY_IDENTITY_SAME_CONSUMPTION_IDENTITY_"
            "FIRST_INCOMPLETE_AUTHENTICATED_UNIT"
        ),
        "runtime_output_namespace": relative(RUNTIME),
        "schema": "ace2-stage1-r5-repair-0005-continuation-interface-v1",
    }


def authority_record(
    activity_sha256: str, interface_sha256: str
) -> dict[str, Any]:
    return {
        "activity": {
            "path": "activity.json",
            "sha256": activity_sha256,
            "state": "AUTHORITY_CREATION_ONLY",
        },
        "authority_cardinality": 1,
        "authority_identity": AUTHORITY_IDENTITY,
        "authority_provenance": (
            "CURRENT_OPERATOR_OBJECTIVE_AUTHORIZE_REPAIR_0005_LAYER11_CONTINUATION"
        ),
        "bindings": {
            "candidate": {
                "identity": CANDIDATE_IDENTITY,
                "member_count": CANDIDATE_MEMBER_COUNT,
                "path": relative(CANDIDATE),
                "tree_root_sha256": CANDIDATE_ROOT,
            },
            "checkpoint_manifest": {
                "path": relative(CANDIDATE / "checkpoint-manifest.json"),
                "sha256": CHECKPOINT_MANIFEST_SHA256,
            },
            "continuation_interface": {
                "path": "continuation-interface.json",
                "sha256": interface_sha256,
            },
            "continuation_runner": {
                "path": relative(CANDIDATE / "continuation_runner.py"),
                "sha256": CONTINUATION_RUNNER_SHA256,
            },
            "executable_closure": {
                "runtime_closure_path": relative(
                    CANDIDATE / "runtime-closure.json"
                ),
                "runtime_closure_sha256": RUNTIME_CLOSURE_SHA256,
                "source_launch_path": relative(CANDIDATE / "source_launch.py"),
                "source_launch_sha256": SOURCE_LAUNCH_SHA256,
            },
            "fresh_reviewer_pass_receipt": {
                "authentication": "EXACT_SHA256_AND_REQUIRED_PASS_FIELDS",
                "decision": "PASS",
                "decision_identity": "FRESH_REVIEWER_DECISION",
                "path": relative(REVIEW),
                "sha256": REVIEW_SHA256,
            },
            "successor_manifest": {
                "path": relative(CANDIDATE / "successor-manifest.json"),
                "sha256": SUCCESSOR_MANIFEST_SHA256,
            },
            "terminal_attempts": {
                "attempt_0012": {
                    "path": relative(ATTEMPT_0012_TERMINAL),
                    "sha256": ATTEMPT_0012_TERMINAL_SHA256,
                    "status": "SEALED_TERMINAL_NO_RETRY",
                },
                "attempt_0013": {
                    "path": relative(ATTEMPT_0013_TERMINAL),
                    "sha256": ATTEMPT_0013_TERMINAL_SHA256,
                    "status": "SEALED_TERMINAL_NO_RETRY",
                },
            },
        },
        "consumable": True,
        "consumption_cardinality": 0,
        "exactly_once": {
            "authenticated_completed_unit_reexecution": "FORBIDDEN",
            "concurrent_duplicate_start": "NONBLOCKING_EXCLUSIVE_LOCK_REFUSAL",
            "grant_cardinality": 1,
            "interrupted_recovery_creates_new_consumption": False,
            "interrupted_recovery_creates_new_grant": False,
            "published_unit_adoption": (
                "AUTHENTICATE_AND_ADOPT_WITHOUT_REEXECUTION"
            ),
            "terminal_replay": "FORBIDDEN",
            "unit_semantics": "PER_UNIT_EXACTLY_ONCE_DURABLE_PUBLICATION",
        },
        "grant": {
            "first_unit": PERMITTED_UNITS[0],
            "grant_cardinality": 1,
            "last_unit": PERMITTED_UNITS[-1],
            "permitted_unit_count": len(PERMITTED_UNITS),
            "permitted_unit_keys": PERMITTED_UNITS,
            "position": "position-00",
            "scope": "EXACT_MISSING_LAYERS_11_THROUGH_23_ONLY",
        },
        "interruption_controls": {
            "checkpoint_commit": "ATOMIC_REPLACE_FILE_FSYNC_AND_PARENT_FSYNC",
            "exception_capture": (
                "DURABLE_PRIMARY_WITH_DURABLE_FALLBACK_AND_ORIGINAL_TRACEBACK"
            ),
            "resume_point": "FIRST_INCOMPLETE_AUTHENTICATED_UNIT",
            "stale_work": "PRUNE_BEFORE_EXECUTION",
            "timing": "PRESERVE_CUMULATIVE_AND_PER_UNIT_WALL_SECONDS",
            "unit_publication": "ATOMIC_PENDING_TO_PUBLISHED_DIRECTORY_RENAME",
            "work_prune": "ONLY_AFTER_DURABLE_PUBLICATION_AND_CHECKPOINT_COMMIT",
        },
        "prohibitions": {
            "attempt_0012": "ALL_EXECUTION_REPLAY_RESUME_AND_MUTATION_FORBIDDEN",
            "attempt_0013": "ALL_EXECUTION_REPLAY_RESUME_AND_MUTATION_FORBIDDEN",
            "layers_00_through_10": "RECOMPUTATION_AND_EXECUTION_FORBIDDEN",
            "stage2": "FORBIDDEN",
            "units_outside_permitted_set": "FORBIDDEN",
        },
        "salvage_boundary": {
            "authenticated_pass_count": len(PASS_UNITS),
            "authenticated_unit_keys": PASS_UNITS,
            "first_incomplete_unit": PERMITTED_UNITS[0],
            "recomputation": "FORBIDDEN",
            "salvage_boundary": PASS_UNITS[-1],
            "status": "PASS_AUTHENTICATED",
        },
        "schema": "ace2-stage1-r5-bounded-continuation-authority-v1",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "status": "SEALED_GRANTED_UNCONSUMED",
    }


def open_directory_nofollow(path: Path) -> int:
    require(path.is_absolute(), f"absolute path required: {path}")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def create_file(
    directory_descriptor: int, name: str, content: bytes, mode: int = 0o444
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
        dir_fd=directory_descriptor,
    )
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_authority() -> str:
    authenticate_prerequisites(require_pristine_runtime=True)
    validate_sealed_tree(AUTHORITY, sha256_file(AUTHORITY / "SHA256SUMS"), 4)
    tree_root = sha256_file(AUTHORITY / "SHA256SUMS")
    authority = load(AUTHORITY / "authority.json")
    activity = load(AUTHORITY / "activity.json")
    interface = load(AUTHORITY / "continuation-interface.json")
    require(activity == ACTIVITY, "authority activity record changed")
    require(
        authority["schema"]
        == "ace2-stage1-r5-bounded-continuation-authority-v1"
        and authority["authority_identity"] == AUTHORITY_IDENTITY
        and authority["status"] == "SEALED_GRANTED_UNCONSUMED"
        and authority["authority_cardinality"] == 1
        and authority["consumption_cardinality"] == 0
        and authority["consumable"] is True
        and authority["grant"]["permitted_unit_keys"] == PERMITTED_UNITS
        and authority["grant"]["permitted_unit_count"] == 13
        and authority["salvage_boundary"]["authenticated_unit_keys"]
        == PASS_UNITS
        and authority["salvage_boundary"]["first_incomplete_unit"]
        == PERMITTED_UNITS[0]
        and authority["prohibitions"]["layers_00_through_10"]
        == "RECOMPUTATION_AND_EXECUTION_FORBIDDEN"
        and authority["prohibitions"]["attempt_0012"]
        == "ALL_EXECUTION_REPLAY_RESUME_AND_MUTATION_FORBIDDEN"
        and authority["prohibitions"]["attempt_0013"]
        == "ALL_EXECUTION_REPLAY_RESUME_AND_MUTATION_FORBIDDEN"
        and authority["prohibitions"]["stage2"] == "FORBIDDEN",
        "bounded authority grant changed",
    )
    require(
        authority["bindings"]["candidate"]["tree_root_sha256"] == CANDIDATE_ROOT
        and authority["bindings"]["fresh_reviewer_pass_receipt"]["sha256"]
        == REVIEW_SHA256
        and authority["bindings"]["successor_manifest"]["sha256"]
        == SUCCESSOR_MANIFEST_SHA256
        and authority["bindings"]["checkpoint_manifest"]["sha256"]
        == CHECKPOINT_MANIFEST_SHA256
        and authority["bindings"]["executable_closure"][
            "runtime_closure_sha256"
        ]
        == RUNTIME_CLOSURE_SHA256
        and authority["bindings"]["continuation_runner"]["sha256"]
        == CONTINUATION_RUNNER_SHA256
        and authority["activity"]["sha256"]
        == sha256_file(AUTHORITY / "activity.json")
        and authority["bindings"]["continuation_interface"]["sha256"]
        == sha256_file(AUTHORITY / "continuation-interface.json"),
        "authority prerequisite binding changed",
    )
    require(
        interface["authority_identity"] == AUTHORITY_IDENTITY
        and interface["permitted_unit_keys"] == PERMITTED_UNITS
        and interface["first_call_unit"] == PERMITTED_UNITS[0]
        and interface["entrypoint_invoked_during_authority_creation"] is False
        and interface["authorized_callable"]["module_sha256"]
        == CONTINUATION_RUNNER_SHA256
        and interface["checkpoint_seed"]["sha256"]
        == CHECKPOINT_MANIFEST_SHA256
        and interface["executor_contract"]["required_evidence"]
        == {"rtl_reference_agreement": True},
        "continuation interface changed",
    )
    return tree_root


def create_authority() -> str:
    require(HERE.parent.name == "tools", "creation is restricted to the source tool")
    require(
        not os.path.lexists(AUTHORITY),
        f"refusing to replace authority namespace: {AUTHORITY}",
    )
    authenticate_prerequisites(require_pristine_runtime=True)
    activity_bytes = canonical_bytes(ACTIVITY)
    interface_bytes = canonical_bytes(interface_record())
    authority_bytes = canonical_bytes(
        authority_record(
            sha256_bytes(activity_bytes),
            sha256_bytes(interface_bytes),
        )
    )
    members = {
        "activity.json": activity_bytes,
        "authority.json": authority_bytes,
        "continuation-interface.json": interface_bytes,
        "validate_authority.py": HERE.read_bytes(),
    }
    sums_bytes = (
        "".join(
            f"{sha256_bytes(content)}  {name}\n"
            for name, content in sorted(members.items())
        )
    ).encode("ascii")
    root = sha256_bytes(sums_bytes)
    parent_descriptor = open_directory_nofollow(AUTHORITY.parent)
    directory_descriptor = -1
    try:
        os.mkdir(AUTHORITY.name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        directory_descriptor = os.open(
            AUTHORITY.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        for name, content in sorted(members.items()):
            create_file(directory_descriptor, name, content)
        create_file(directory_descriptor, "SHA256SUMS", sums_bytes)
        create_file(
            directory_descriptor,
            "TREE_ROOT.sha256",
            f"{root}  SHA256SUMS\n".encode("ascii"),
        )
        os.fchmod(directory_descriptor, 0o555)
        os.fsync(directory_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        os.close(parent_descriptor)
    require(validate_authority() == root, "published authority validation failed")
    return root


def main() -> None:
    inside_authority = HERE.parent == AUTHORITY.resolve(strict=False)
    if inside_authority or sys.argv[1:] == ["--check"]:
        root = validate_authority()
    elif not sys.argv[1:]:
        root = create_authority()
    else:
        raise AuthorityError(
            "usage: create_attempt_0014_repair_0005_authority.py [--check]"
        )
    print(
        "ACE2_REPAIR_0005_CONTINUATION_AUTHORITY_PASS "
        f"authority_root={root} permitted_units=13 "
        "first=position-00/layer-11 last=position-00/layer-23 "
        "authority_creation=1 consumption=0 entrypoint=0 model=0 rtl=0 "
        "output=0 submission=0 stage2=FORBIDDEN"
    )


if __name__ == "__main__":
    main()
