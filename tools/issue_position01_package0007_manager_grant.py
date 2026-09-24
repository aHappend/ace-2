#!/usr/bin/env python3
"""Issue the single Manager grant for the accepted position-01 package0007."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
PACKAGE = ROOT / "reports/ace2-position01-integrated-runtime-pass-package-0007"
VALIDATION = ROOT / "reports/ace2-position01-integrated-runtime-pass-package-validation-0009"
ADJUDICATION = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-0007-adjudication"
    / "terminal-verdict.json"
)
REVIEWER_RECEIPT = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-0007-adjudication"
    / "reviewer-receipt.json"
)
HOST_BINDING = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-0007-adjudication"
    / "host-completion-binding.json"
)
ACTIVE_DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/active_manager_directive.json"
)
GRANT_NAMESPACE = ROOT / "reports/ace2-position01-integrated-runtime-pass-manager-grant-0007"
GRANT = GRANT_NAMESPACE / "grant.json"
CONSUMPTION = ROOT / "reports/ace2-position01-integrated-runtime-pass-consumption-state-0007"
OUTPUT = ROOT / "reports/ace2-position01-runtime-pass-0001"
SCRATCH = ROOT / "reports/ace2-position01-runtime-pass-scratch-0001"
PACKAGE0006_GRANT = (
    ROOT / "reports/ace2-position01-integrated-runtime-pass-manager-grant-0006/grant.json"
)
PACKAGE0006_BURN = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-review-0006"
    / "terminal-burn-record.json"
)
PACKAGE0006_CONSUMPTION = (
    ROOT / "reports/ace2-position01-integrated-runtime-pass-consumption-state-0006"
)

EXPECTED_PACKAGE_ROOT = "23cdd7f8b37dfa65943bc213c711cce1bcab662a0eefcc77a78194ba36c6c038"
EXPECTED_VALIDATION_TREE = "37d3a57320ec3e474be7ea2fbdd3c18ac94a76d3884ae7de54787eddf358e360"
EXPECTED_ADJUDICATION = "2da2edc916e109d152036cefab50c03f4fd5c78cc940896bb76506ec4980881a"
EXPECTED_REVIEWER_RECEIPT = "55cd41c8119a0ec135fe2f2252b73f38d10d7edfe24d15a88362f2abe98d7bd0"
EXPECTED_HOST_BINDING = "9603b7d39ca874c456a6d22b6e56f2272b73af6b45cff2c4e20fb89de6c6166a"
EXPECTED_PACKAGE0006_GRANT = "301f1930b6053068fda658b4435988770c6a1eda0bcc95e9d2a19cc13ee321f0"
EXPECTED_PACKAGE0006_BURN = "5025b874f8409d0f4ad0ed22d21f55dd9a697ce84f001d0d4a59ce6d381b7c29"
EXPECTED_DIRECTIVE_REVISION = "4f62915ba69242b4b9a817b5f4e50aff"
EXPECTED_DIRECTIVE_OBJECTIVE = "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
EXPECTED_DIRECTIVE_SOURCE = "manager.autonomous-supervision.position01-package0007-grant-v25"
EXPECTED_DIRECTIVE_TEXT = (
    "MANAGER DIRECTIVE POSITION-01 PACKAGE0007 GRANT V25: This supersedes package0007 "
    "construction and every Layer18/19 recovery scope. Package0007 is the sole "
    "review-accepted execution candidate: sealed root "
    f"{EXPECTED_PACKAGE_ROOT}; validation0009 canonical tree {EXPECTED_VALIDATION_TREE}; "
    f"terminal L2 PASS {EXPECTED_ADJUDICATION}; reviewer receipt "
    f"{EXPECTED_REVIEWER_RECEIPT}; Host binding {EXPECTED_HOST_BINDING}. Grant0006 "
    "remains permanently NON-CONSUMABLE and package0006 remains burned.\n\n"
    "Authorize creation of exactly one fresh create-exclusive package0007 Manager grant "
    "only. It must authenticate those exact digests, the package-declared absolute "
    "invocation, all live prerequisites/tools, the 20,434,649,088-byte capacity admission "
    "and hard aggregate output/scratch quota contract, honest LM-head wall_seconds/"
    "total-latency receipt, reserved namespaces, durable consume-before-workload record, "
    "terminal/failure publication, and no-retry rule. Set authority_consumed=false and "
    "execution_performed=false. Fail closed and create no grant if any bound identity/path/"
    "hash differs, if live capacity is insufficient, if package0007 grant/consumption "
    "namespaces already exist, or if the future output/scratch namespace is not safely "
    "fresh under the package0007 authority. Grant construction only: zero executor, "
    "consumption, workload, runtime/scratch output, checkpoint mutation, Stage1 closure, "
    "PPA, or Stage2."
)

ZERO_ACTIVITY = {
    "authority_consumed": 0,
    "authority_created": 0,
    "checkpoint_mutation": 0,
    "executor_invocation": 0,
    "generated_tokens": 0,
    "grant_consumed": 0,
    "manager_grant_created": 1,
    "model_execution": 0,
    "ppa_execution": 0,
    "protected_state_delta": 0,
    "reference_execution": 0,
    "rtl_execution": 0,
    "stage1_closure": 0,
    "stage2_execution": 0,
    "u280_execution": 0,
    "unit_execution": 0,
    "workload_execution": 0,
}


class GrantError(RuntimeError):
    """The exact grant preconditions were not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GrantError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("ascii"), object_pairs_hook=no_duplicate_keys)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_exact_file(path: Path, expected_sha256: str, label: str) -> tuple[bytes, dict[str, Any]]:
    status = path.lstat()
    require(
        stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode),
        f"{label} is not a regular non-symlink file",
    )
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"{label} hash changed")
    return payload, load_json_bytes(payload, label)


def canonical_validation_tree(path: Path) -> tuple[str, int, int]:
    root_status = path.lstat()
    require(
        stat.S_ISDIR(root_status.st_mode) and not stat.S_ISLNK(root_status.st_mode),
        "validation0009 is not a regular directory",
    )
    descendants = sorted(
        path.rglob("*"), key=lambda item: item.relative_to(path).as_posix().encode("utf-8")
    )
    entries = [(path, ".")] + [
        (item, item.relative_to(path).as_posix()) for item in descendants
    ]
    manifest = bytearray()
    for item, name in entries:
        status = item.lstat()
        mode = stat.S_IMODE(status.st_mode)
        if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
            manifest.extend(f"d\0{name}\0{mode:04o}\0-\n".encode("utf-8"))
        elif stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode):
            manifest.extend(
                f"f\0{name}\0{mode:04o}\0{status.st_size}\0{sha256_file(item)}\n".encode(
                    "utf-8"
                )
            )
        else:
            raise GrantError(f"validation0009 has non-regular entry: {name}")
    return sha256_bytes(bytes(manifest)), len(entries), len(manifest)


def load_package_authority() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ace2_position01_package0007_authority", PACKAGE / "authority.py"
    )
    require(spec is not None and spec.loader is not None, "cannot load package0007 authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def validate_directive() -> tuple[bytes, dict[str, Any]]:
    status = ACTIVE_DIRECTIVE.lstat()
    require(
        stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode),
        "active Manager directive is not a regular non-symlink file",
    )
    payload = ACTIVE_DIRECTIVE.read_bytes()
    record = load_json_bytes(payload, "active Manager directive")
    require(record.get("revision") == EXPECTED_DIRECTIVE_REVISION, "Manager directive revision changed")
    require(record.get("objective_sha256") == EXPECTED_DIRECTIVE_OBJECTIVE, "Manager directive objective changed")
    require(record.get("source") == EXPECTED_DIRECTIVE_SOURCE, "Manager directive source changed")
    require(record.get("text") == EXPECTED_DIRECTIVE_TEXT, "Manager directive text changed")
    require(record.get("version") == 1, "Manager directive version changed")
    return payload, record


def validate_accepted_review() -> dict[str, Any]:
    _, verdict = load_exact_file(ADJUDICATION, EXPECTED_ADJUDICATION, "L2 adjudication")
    _, receipt = load_exact_file(
        REVIEWER_RECEIPT, EXPECTED_REVIEWER_RECEIPT, "Reviewer receipt"
    )
    _, host = load_exact_file(HOST_BINDING, EXPECTED_HOST_BINDING, "Host binding")
    require(
        verdict.get("terminal") == "PASS"
        and verdict.get("review_level") == "L2"
        and verdict.get("package_root_sha256") == EXPECTED_PACKAGE_ROOT
        and verdict.get("validation_digest", {}).get("value") == EXPECTED_VALIDATION_TREE,
        "terminal L2 adjudication bindings changed",
    )
    require(
        receipt.get("package_root_sha256") == EXPECTED_PACKAGE_ROOT
        and receipt.get("validation_digest", {}).get("value") == EXPECTED_VALIDATION_TREE
        and receipt.get("verdict_digest", {}).get("value") == EXPECTED_ADJUDICATION,
        "Reviewer receipt bindings changed",
    )
    require(
        host.get("package", {}).get("tree_root_sha256") == EXPECTED_PACKAGE_ROOT
        and host.get("validation", {}).get("canonical_tree_sha256")
        == EXPECTED_VALIDATION_TREE
        and host.get("adjudication", {}).get("sha256") == EXPECTED_ADJUDICATION
        and host.get("reviewer_receipt", {}).get("sha256") == EXPECTED_REVIEWER_RECEIPT,
        "Host binding changed",
    )
    return {
        "host_binding": {
            "path": relative(HOST_BINDING),
            "sha256": EXPECTED_HOST_BINDING,
        },
        "l2_adjudication": {
            "decision": "PASS",
            "level": "L2",
            "path": relative(ADJUDICATION),
            "sha256": EXPECTED_ADJUDICATION,
        },
        "reviewer_receipt": {
            "path": relative(REVIEWER_RECEIPT),
            "sha256": EXPECTED_REVIEWER_RECEIPT,
        },
    }


def validate_package0006_burn() -> dict[str, Any]:
    _, burn = load_exact_file(PACKAGE0006_BURN, EXPECTED_PACKAGE0006_BURN, "package0006 burn")
    _, prior_grant = load_exact_file(
        PACKAGE0006_GRANT, EXPECTED_PACKAGE0006_GRANT, "grant0006"
    )
    require(
        burn.get("burned") is True
        and burn.get("terminal") is True
        and burn.get("decision") == "FAIL",
        "package0006 is not terminally burned",
    )
    require(
        prior_grant.get("authority_consumed") is False
        and prior_grant.get("execution_performed") is False
        and prior_grant.get("consumption_state") == "UNCONSUMED",
        "grant0006 state changed",
    )
    require(
        not os.path.lexists(PACKAGE0006_CONSUMPTION),
        "grant0006 consumption namespace exists",
    )
    return {
        "grant0006": {
            "consumable": False,
            "disposition": "PERMANENTLY_NON_CONSUMABLE",
            "path": relative(PACKAGE0006_GRANT),
            "sha256": EXPECTED_PACKAGE0006_GRANT,
        },
        "package0006": {
            "burned": True,
            "burn_record_path": relative(PACKAGE0006_BURN),
            "burn_record_sha256": EXPECTED_PACKAGE0006_BURN,
            "disposition": "IMMUTABLE_TERMINALLY_BURNED",
        },
    }


def validate_fresh_namespaces() -> dict[str, Any]:
    namespaces = {
        "consumption": CONSUMPTION,
        "grant": GRANT_NAMESPACE,
        "output": OUTPUT,
        "scratch": SCRATCH,
    }
    for label, path in namespaces.items():
        require(not os.path.lexists(path), f"reserved package0007 {label} namespace exists")
    return {
        label: {
            "path": relative(path),
            "preexisting": False,
            "semantics": "CREATE_EXCLUSIVE",
        }
        for label, path in namespaces.items()
    }


def build_grant(
    authority_record: dict[str, Any],
    dispatch: dict[str, Any],
    directive_bytes: bytes,
    directive: dict[str, Any],
    review_bindings: dict[str, Any],
    predecessor: dict[str, Any],
    reserved_namespaces: dict[str, Any],
) -> dict[str, Any]:
    future_unit = authority_record["future_execution_unit"]
    output_contract = authority_record["future_output_contract"]
    resource_guard = authority_record["resource_guard"]
    exact_argv = future_unit["exact_argv"]
    require(
        exact_argv[0].startswith("/")
        and exact_argv[2] == str((PACKAGE / "run_position01_integrated.py").resolve()),
        "package0007 invocation is not the exact package-declared absolute invocation",
    )
    require(
        resource_guard["required_available_bytes"] == 20_434_649_088,
        "package0007 capacity admission changed",
    )
    require(
        output_contract["quota_enforcement"]["output_max_bytes"]
        == resource_guard["runtime_output_max_bytes"]
        and output_contract["quota_enforcement"]["scratch_max_bytes"]
        == resource_guard["temporary_write_max_bytes"],
        "package0007 output/scratch quota contract changed",
    )
    latency = output_contract["terminal_latency_receipt"]
    require(
        latency["w4a8_lm_head_binding"]
        == "EXACT_CYCLES_AND_MEASURED_ICARUS_WALL_SECONDS_FROM_EXECUTION_DETAILS"
        and latency["total_latency"]
        == "MONOTONIC_WALL_SECONDS_FROM_POST_CONSUMPTION_TO_DURABLE_TERMINAL",
        "package0007 LM-head or total-latency receipt changed",
    )
    return {
        "activity": ZERO_ACTIVITY,
        "append_only_creation": {
            "namespace": relative(GRANT_NAMESPACE),
            "path": relative(GRANT),
            "semantics": "CREATE_EXCLUSIVE_NEW_PACKAGE0007_MANAGER_GRANT",
            "target_preexisting": False,
        },
        "authority_cardinality": 1,
        "authority_consumed": False,
        "bindings": {
            "active_manager_directive": {
                "objective_sha256": directive["objective_sha256"],
                "path": str(ACTIVE_DIRECTIVE),
                "revision": directive["revision"],
                "sha256": sha256_bytes(directive_bytes),
                "source": directive["source"],
            },
            "package": {
                "authority_path": relative(PACKAGE / "authority.json"),
                "authority_sha256": sha256_file(PACKAGE / "authority.json"),
                "path": relative(PACKAGE),
                "tree_root_sha256": EXPECTED_PACKAGE_ROOT,
            },
            "validation": {
                "canonical_tree_sha256": EXPECTED_VALIDATION_TREE,
                "path": relative(VALIDATION),
            },
            **review_bindings,
        },
        "capacity_and_quota": {
            "available_bytes_at_grant": dispatch["available_bytes"],
            "capacity_source": resource_guard["capacity_source"],
            "enforcement": resource_guard["enforcement"],
            "filesystem_root": str(ROOT),
            "output_namespace": relative(OUTPUT),
            "package_max_bytes": resource_guard["package_max_bytes"],
            "required_available_bytes": resource_guard["required_available_bytes"],
            "runtime_output_max_bytes": resource_guard["runtime_output_max_bytes"],
            "safety_margin_bytes": resource_guard["safety_margin_bytes"],
            "scratch_namespace": relative(SCRATCH),
            "temporary_write_max_bytes": resource_guard["temporary_write_max_bytes"],
        },
        "consumption_protocol": {
            "consume_before_any_output_or_workload": True,
            "consumption_record": relative(CONSUMPTION / "consumed.json"),
            "durability": "CREATE_EXCLUSIVE_FILE_FSYNC_THEN_DIRECTORY_FSYNC",
            "failure_after_consumption": "AUTHORITY_REMAINS_CONSUMED_NO_RETRY",
            "repeat_use": "REJECT",
        },
        "consumption_state": "UNCONSUMED",
        "decision": "AUTHORIZE_EXACTLY_ONE_POSITION01_INTEGRATED_RUNTIME_PASS_PACKAGE0007",
        "exact_invocation": {"argv": exact_argv, "cwd": str(ROOT)},
        "execution_limit": 1,
        "execution_performed": False,
        "failure_recovery": {
            "failure_publication": {
                "durability": "CREATE_EXCLUSIVE_FILE_FSYNC_THEN_DIRECTORY_FSYNC",
                "path": relative(CONSUMPTION / "failure.json"),
                "required_after_any_failed_consumption_or_execution_attempt": True,
            },
            "grant_reuse_after_failure": "FORBIDDEN",
            "no_retry": True,
            "partial_output_disposition": "PRESERVE_FOR_INDEPENDENT_REVIEW",
            "recovery": "NEW_EXPLICIT_MANAGER_AUTHORITY_REQUIRED",
        },
        "live_prerequisites": {
            "authority_bindings": authority_record["bindings"],
            "dispatch_checks": dispatch["checks"],
            "simulator_paths": dispatch["simulator_paths"],
        },
        "origin": "ACTIVE_MANAGER_DIRECTIVE_POSITION01_PACKAGE0007_GRANT_V25",
        "predecessor_preservation": predecessor,
        "producer_role": "manager",
        "prohibited_activity": {
            key: value
            for key, value in ZERO_ACTIVITY.items()
            if key != "manager_grant_created"
        },
        "reserved_namespaces": reserved_namespaces,
        "schema": "ace2-position01-integrated-runtime-pass-package0007-manager-grant-v1",
        "terminal_publication": {
            "execution_details_path": output_contract["execution_details_path"],
            "failure_path": relative(CONSUMPTION / "failure.json"),
            "lm_head_and_total_latency_receipt": latency,
            "selected_token_path": output_contract[
                "exactly_one_selected_token_publication"
            ]["path"],
        },
        "unit_key": "position-01/integrated-decode-runtime-pass",
    }


def create_grant(payload: bytes) -> None:
    os.mkdir(GRANT_NAMESPACE, 0o700)
    directory_fd = os.open(GRANT_NAMESPACE, os.O_RDONLY | os.O_DIRECTORY)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_fd = os.open(GRANT, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(file_fd, view)
                require(written > 0, "grant write made no progress")
                view = view[written:]
            os.fsync(file_fd)
            os.fchmod(file_fd, 0o444)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(directory_fd)
        os.chmod(GRANT_NAMESPACE, 0o555)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    parent_fd = os.open(GRANT_NAMESPACE.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def validate_created_grant(expected: dict[str, Any], payload: bytes, directive_bytes: bytes) -> str:
    require(
        GRANT_NAMESPACE.is_dir()
        and not GRANT_NAMESPACE.is_symlink()
        and stat.S_IMODE(GRANT_NAMESPACE.stat().st_mode) == 0o555,
        "created grant namespace mode or type changed",
    )
    require(
        GRANT.is_file()
        and not GRANT.is_symlink()
        and stat.S_IMODE(GRANT.stat().st_mode) == 0o444,
        "created grant mode or type changed",
    )
    require(GRANT.read_bytes() == payload, "created grant bytes changed")
    require(load_json_bytes(payload, "created grant") == expected, "created grant JSON changed")
    require(ACTIVE_DIRECTIVE.read_bytes() == directive_bytes, "Manager directive changed during grant issue")
    for path in (CONSUMPTION, OUTPUT, SCRATCH):
        require(not os.path.lexists(path), f"prohibited namespace created: {path}")
    return sha256_bytes(payload)


def main() -> int:
    require(Path.cwd() == ROOT, "exact repository cwd required")
    directive_bytes, directive = validate_directive()
    reserved_namespaces = validate_fresh_namespaces()
    validation_digest, validation_entries, validation_bytes = canonical_validation_tree(
        VALIDATION
    )
    require(validation_digest == EXPECTED_VALIDATION_TREE, "validation0009 tree changed")
    require(
        validation_entries == 30 and validation_bytes == 2894,
        "validation0009 tree shape changed",
    )
    review_bindings = validate_accepted_review()
    predecessor = validate_package0006_burn()

    authority = load_package_authority()
    authority_record = authority.load_json(PACKAGE / "authority.json")
    dispatch = authority.validate_dispatch(authority_record, EXPECTED_PACKAGE_ROOT)
    require(dispatch["package_tree_root_sha256"] == EXPECTED_PACKAGE_ROOT, "package0007 root changed")

    require(ACTIVE_DIRECTIVE.read_bytes() == directive_bytes, "Manager directive changed before grant issue")
    validate_fresh_namespaces()
    for path, expected, label in (
        (ADJUDICATION, EXPECTED_ADJUDICATION, "L2 adjudication"),
        (REVIEWER_RECEIPT, EXPECTED_REVIEWER_RECEIPT, "Reviewer receipt"),
        (HOST_BINDING, EXPECTED_HOST_BINDING, "Host binding"),
        (PACKAGE0006_GRANT, EXPECTED_PACKAGE0006_GRANT, "grant0006"),
        (PACKAGE0006_BURN, EXPECTED_PACKAGE0006_BURN, "package0006 burn"),
    ):
        require(sha256_file(path) == expected, f"{label} changed before grant issue")

    grant = build_grant(
        authority_record,
        dispatch,
        directive_bytes,
        directive,
        review_bindings,
        predecessor,
        reserved_namespaces,
    )
    payload = canonical_bytes(grant)
    create_grant(payload)
    grant_sha256 = validate_created_grant(grant, payload, directive_bytes)
    sys.stdout.write(
        json.dumps(
            {
                "authority_consumed": False,
                "available_bytes_at_grant": dispatch["available_bytes"],
                "decision": "PASS_PACKAGE0007_MANAGER_GRANT_CREATED_AND_VALIDATED",
                "execution_performed": False,
                "grant_path": relative(GRANT),
                "grant_sha256": grant_sha256,
                "package_tree_root_sha256": EXPECTED_PACKAGE_ROOT,
                "status": "PASS",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GrantError, OSError, ValueError, KeyError, TypeError) as error:
        sys.stderr.write(f"FAIL_CLOSED: {error}\n")
        raise SystemExit(1)
