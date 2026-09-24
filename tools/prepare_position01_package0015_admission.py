#!/usr/bin/env python3
"""Create or validate the fail-closed Package0015 admission preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
PACKAGE0014 = (
    ROOT / "reports/ace2-position01-integrated-runtime-pass-package-0014"
)
CONSUMPTION0014 = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-consumption-state-0014"
)
DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
TARGET_DIR = ROOT / ".argus/live/package0015-no-execution-admission-v1"
TARGET = TARGET_DIR / "admission-preparation.json"
REQUIRED_AVAILABLE_BYTES = 20_600_000_000

EXPECTED_HASHES = {
    "package_tree_root": (
        "be0a65fea61274adf00f140f276a77018cedf748df09f820701ded27efb9000c"
    ),
    "consumed": (
        "bc055091be449aec6444e9dfe94f5a76557b983501a8a913ef9782bbc196f5d9"
    ),
    "failure": (
        "9ff5ad9baf958f85c2d718ad4ecbfb573b6158b11c2f06b2399ff5dc51aab373"
    ),
    "grant": (
        "e3dc881b59facaa67ab03a63b3613c5a15ab4892cf2c0463501b8bc2b2fd7cde"
    ),
    "gate": (
        "e940777f408944300943712219fd8677895cb2918acb68714e67db6bb0a0a404"
    ),
    "execution_authorization": (
        "5a1dbc670fa80050224333538ab096950598081aa3222b24a7216f0edd1c0c0c"
    ),
    "directive": (
        "3515e5aa065d88363ddb229755087e6f9bf419eca5f84923213fdb13715de965"
    ),
}

EVIDENCE_PATHS = {
    "consumed": (
        CONSUMPTION0014 / "consumed.json"
    ),
    "failure": (
        CONSUMPTION0014 / "failure.json"
    ),
    "grant": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-manager-grant-0014"
        / "grant.json"
    ),
    "gate": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-live-launch-gate-0014"
        / "gate.json"
    ),
    "execution_authorization": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-execution-authorization-0014"
        / "execution-authorization.json"
    ),
}

PROSPECTIVE_PATHS = {
    "package": (
        ROOT / "reports/ace2-position01-integrated-runtime-pass-package-0015"
    ),
    "review": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-package-review-0015"
    ),
    "host_binding": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-package-host-binding-0015"
    ),
    "grant": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-manager-grant-0015"
    ),
    "gate": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-live-launch-gate-0015"
    ),
    "execution_authorization": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-execution-authorization-0015"
    ),
    "consumption": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-consumption-state-0015"
    ),
    "output": ROOT / "reports/ace2-position01-runtime-pass-0008",
    "scratch": ROOT / "reports/ace2-position01-runtime-pass-scratch-0008",
    "continuation": (
        ROOT
        / "reports/ace2-position01-integrated-runtime-pass-continuation-0015"
    ),
    "replay": (
        ROOT / "reports/ace2-position01-integrated-runtime-pass-replay-0015"
    ),
}

ZERO_ACTIVITY = {
    "authority_consumed": 0,
    "checkpoint_mutation": 0,
    "consumption_created": 0,
    "continuation_created": 0,
    "execution_authorization_created": 0,
    "executor_invocation": 0,
    "generated_tokens": 0,
    "grant_consumed": 0,
    "launch_gate_created": 0,
    "manager_grant_created": 0,
    "model_execution": 0,
    "output_created": 0,
    "reference_execution": 0,
    "replay_created": 0,
    "rtl_execution": 0,
    "scratch_created": 0,
    "stage1_closure": 0,
    "stage2_execution": 0,
    "workload_execution": 0,
}


class AdmissionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def validate_package0014_tree() -> int:
    require(
        PACKAGE0014.is_dir()
        and not PACKAGE0014.is_symlink()
        and stat.S_IMODE(PACKAGE0014.stat().st_mode) == 0o555,
        "Package0014 sealed directory changed",
    )
    manifest = PACKAGE0014 / "SHA256SUMS"
    require(
        sha256_file(manifest) == EXPECTED_HASHES["package_tree_root"],
        "Package0014 tree root changed",
    )
    require(
        (PACKAGE0014 / "TREE_ROOT.sha256")
        .read_text(encoding="ascii")
        .split()
        == [EXPECTED_HASHES["package_tree_root"], "SHA256SUMS"],
        "Package0014 tree root record changed",
    )
    records: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in records,
            "Package0014 checksum manifest changed",
        )
        records[name] = digest
    observed = {
        item.relative_to(PACKAGE0014).as_posix()
        for item in PACKAGE0014.rglob("*")
        if item.is_file() and item.name not in {"SHA256SUMS", "TREE_ROOT.sha256"}
    }
    require(observed == set(records), "Package0014 sealed member set changed")
    for name, digest in records.items():
        path = PACKAGE0014 / name
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444
            and sha256_file(path) == digest,
            f"Package0014 sealed member changed: {name}",
        )
    return len(records)


def validate_terminal_evidence() -> dict[str, dict[str, Any]]:
    for label, path in EVIDENCE_PATHS.items():
        require(
            path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == EXPECTED_HASHES[label],
            f"Package0014 {label} evidence changed",
        )
    consumed = load_json(EVIDENCE_PATHS["consumed"])
    failure = load_json(EVIDENCE_PATHS["failure"])
    grant = load_json(EVIDENCE_PATHS["grant"])
    authorization = load_json(EVIDENCE_PATHS["execution_authorization"])
    require(
        consumed.get("status") == "CONSUMED_BEFORE_WORKLOAD"
        and consumed.get("package_tree_root_sha256")
        == EXPECTED_HASHES["package_tree_root"],
        "Package0014 consumption state is not terminally bound",
    )
    require(
        failure.get("status") == "FAIL_PACKAGE0014_CONSUMED_TERMINAL_EXECUTION"
        and failure.get("terminal") is True
        and failure.get("burned") is True
        and failure.get("retry_policy")
        == "PERMANENTLY_FORBIDDEN_FOR_PACKAGE0014_GRANT0014"
        and failure.get("message")
        == "unauthenticated subprocess rejected by quota guard",
        "Package0014 terminal failure state changed",
    )
    require(
        grant.get("authority_consumed") is False
        and grant.get("execution_performed") is False
        and authorization.get("execution_performed") is False,
        "historical immutable grant/authorization fields changed",
    )
    return {
        label: {
            "path": relative(path),
            "sha256": EXPECTED_HASHES[label],
        }
        for label, path in EVIDENCE_PATHS.items()
    }


def validate_quota_rejection_path() -> None:
    quota_source = (PACKAGE0014 / "quota_io.py").read_text(encoding="utf-8")
    runtime_source = (
        PACKAGE0014 / "run_position01_integrated.py"
    ).read_text(encoding="utf-8")
    require(
        'frozenset({"/usr/bin/iverilog", "/usr/bin/vvp"})' in quota_source
        and "if not command or command[0] not in guard.allowed_commands:"
        in quota_source
        and 'QuotaExceeded("unauthenticated subprocess rejected by quota guard")'
        in quota_source,
        "Package0014 quota rejection predicate changed",
    )
    require(
        "executed_command = [str(pinned.proc_path), *command[1:]]"
        in runtime_source
        and "backend.subprocess.Popen(" in runtime_source,
        "Package0014 authenticated simulator hot path changed",
    )


def validate_directive() -> dict[str, Any]:
    require(
        DIRECTIVE.is_file()
        and not DIRECTIVE.is_symlink()
        and sha256_file(DIRECTIVE) == EXPECTED_HASHES["directive"],
        "active Manager V49 directive changed",
    )
    directive = load_json(DIRECTIVE)
    require(
        directive.get("revision") == "14dcff00a4724f7e95b7b4f7bcba4f11"
        and directive.get("source")
        == "manager.autonomous-supervision.ace2-package0014-execution-v49"
        and isinstance(directive.get("text"), str)
        and "PACKAGE0014 EXECUTION AUTHORIZATION PREPARATION"
        in directive["text"]
        and "package0015" not in directive["text"].lower(),
        "active directive is not the sealed Package0014-only V49 directive",
    )
    return {
        "objective_sha256": directive["objective_sha256"],
        "path": str(DIRECTIVE),
        "revision": directive["revision"],
        "sha256": EXPECTED_HASHES["directive"],
        "source": directive["source"],
    }


def available_bytes() -> int:
    observed = os.statvfs(ROOT)
    require(
        observed.f_bavail >= 0 and observed.f_frsize > 0,
        "invalid statvfs capacity observation",
    )
    return observed.f_bavail * observed.f_frsize


def package0015_executor_count() -> int:
    marker = (
        b"ace2-position01-integrated-runtime-pass-package-0015/"
        b"run_position01_integrated.py"
    )
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command:
            count += 1
    return count


def validate_fresh_package0015_namespaces() -> dict[str, dict[str, Any]]:
    result = {}
    for label, path in PROSPECTIVE_PATHS.items():
        require(not os.path.lexists(path), f"Package0015 {label} namespace exists")
        result[label] = {"exists": False, "path": relative(path)}
    return result


def build_record() -> dict[str, Any]:
    member_count = validate_package0014_tree()
    evidence = validate_terminal_evidence()
    validate_quota_rejection_path()
    directive = validate_directive()
    fresh_namespaces = validate_fresh_package0015_namespaces()
    capacity = available_bytes()
    executors = package0015_executor_count()
    require(
        capacity >= REQUIRED_AVAILABLE_BYTES,
        "live capacity below the operator threshold",
    )
    require(executors == 0, "Package0015 executor process exists")
    return {
        "activity": ZERO_ACTIVITY,
        "active_directive": {
            **directive,
            "authorizes_package0015": False,
        },
        "directive_compatibility": {
            "resolution": (
                "ADMISSION_PREPARATION_ONLY; A FRESH PACKAGE0015 MANAGER OR "
                "OPERATOR DIRECTIVE IS REQUIRED BEFORE GRANT OR GATE CREATION"
            ),
            "status": "ambiguous_objective",
            "summary": (
                "The bounded live objective requests Package0015 admission "
                "preparation while active Manager V49 authorizes Package0014 only."
            ),
        },
        "live_observation": {
            "available_bytes_at_preparation": capacity,
            "method": "os.statvfs(ROOT).f_bavail_times_f_frsize",
            "package0015_executor_process_count": executors,
            "required_available_bytes": REQUIRED_AVAILABLE_BYTES,
        },
        "package0014_retirement": {
            "evidence": evidence,
            "historical_execution_namespaces": {
                "output": {
                    "exists": (
                        ROOT / "reports/ace2-position01-runtime-pass-0007"
                    ).exists(),
                    "path": "reports/ace2-position01-runtime-pass-0007",
                },
                "scratch": {
                    "exists": (
                        ROOT / "reports/ace2-position01-runtime-pass-scratch-0007"
                    ).exists(),
                    "path": "reports/ace2-position01-runtime-pass-scratch-0007",
                },
            },
            "package_member_count": member_count,
            "package_tree_root_sha256": EXPECTED_HASHES["package_tree_root"],
            "retry_allowed": False,
            "status": "RETIRED_BURNED_CONSUMED_TERMINAL",
        },
        "package0015": {
            "authorization_state": "NOT_AUTHORIZED",
            "fresh_namespaces": fresh_namespaces,
            "grant_created": False,
            "grant_consumed": False,
            "scope": "NO_EXECUTION_ADMISSION_PREPARATION",
        },
        "quota_rejection": {
            "avoidance_requirement": (
                "A successor must authenticate and admit the actual pinned "
                "simulator executable identity before consumption; it must not "
                "broaden subprocess admission to arbitrary commands."
            ),
            "default_allowed_commands": ["/usr/bin/iverilog", "/usr/bin/vvp"],
            "evidence_message": (
                "unauthenticated subprocess rejected by quota guard"
            ),
            "guard_rejection_predicate": (
                "not command or command[0] not in guard.allowed_commands"
            ),
            "hot_path_transform": (
                "executed_command[0] = str(pinned.proc_path)"
            ),
            "status": (
                "LITERAL_ALLOWLIST_REJECTS_PINNED_PROC_FD_SIMULATOR_IDENTITY"
            ),
        },
        "schema": "ace2-position01-package0015-admission-preparation-v1",
        "status": "PASS_PACKAGE0015_NO_EXECUTION_ADMISSION_PREPARATION",
    }


def validate_record(record: dict[str, Any]) -> None:
    expected = build_record()
    observed_live = record.get("live_observation")
    require(isinstance(observed_live, dict), "recorded live observation absent")
    recorded_capacity = observed_live.get("available_bytes_at_preparation")
    require(
        isinstance(recorded_capacity, int)
        and recorded_capacity >= REQUIRED_AVAILABLE_BYTES,
        "recorded preparation capacity was insufficient",
    )
    expected["live_observation"]["available_bytes_at_preparation"] = (
        recorded_capacity
    )
    require(record == expected, "Package0015 admission preparation changed")


def write_exclusive(record: dict[str, Any]) -> None:
    os.mkdir(TARGET_DIR, mode=0o700)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    descriptor = os.open(
        TARGET,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(TARGET_DIR, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_record(load_json(TARGET))
    else:
        record = build_record()
        write_exclusive(record)
        validate_record(load_json(TARGET))
    print(
        json.dumps(
            {
                "artifact": relative(TARGET),
                "status": (
                    "PASS_PACKAGE0015_NO_EXECUTION_ADMISSION_PREPARATION"
                ),
                "validation": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
