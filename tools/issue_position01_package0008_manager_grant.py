#!/usr/bin/env python3
"""Issue the single Manager grant for the accepted position-01 package0008."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
PACKAGE = ROOT / "reports/ace2-position01-integrated-runtime-pass-package-0008"
AUTHORITY = PACKAGE / "authority.json"
VALIDATION_RESULT = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-validation-0010"
    / "004-postseal-validation.stdout"
)
ADJUDICATION = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-review-0008"
    / "reviewer-adjudication.json"
)
REVIEWER_RECEIPT = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-review-0008"
    / "reviewer-receipt.json"
)
HOST_BINDING = (
    ROOT
    / "reports/ace2-position01-integrated-runtime-pass-package-host-binding-0008"
    / "host-completion-binding.json"
)
ACTIVE_DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/active_manager_directive.json"
)
GRANT_NAMESPACE = (
    ROOT / "reports/ace2-position01-integrated-runtime-pass-manager-grant-0008"
)
GRANT = GRANT_NAMESPACE / "grant.json"
CONSUMPTION = (
    ROOT / "reports/ace2-position01-integrated-runtime-pass-consumption-state-0008"
)
OUTPUT = ROOT / "reports/ace2-position01-runtime-pass-0002"
SCRATCH = ROOT / "reports/ace2-position01-runtime-pass-scratch-0002"

EXPECTED_PACKAGE_ROOT = (
    "c5d01820637c37991989910bc822f01771d46db6c4f40f169e78ae5988a86dc3"
)
EXPECTED_AUTHORITY = (
    "594093d7821d0e8de8b198c80b07c4e5b1fc95e37c2141f69ef1958f23a78aa6"
)
EXPECTED_VALIDATION_RESULT = (
    "d30136437d8a7c1a6fc2637250b47e6c49d94eed15d27c67425753d877847b8e"
)
EXPECTED_ADJUDICATION = (
    "95b84cd1b4b2dc22349579a5e7949884f9daf7c7c02d8c890b08b443a6912817"
)
EXPECTED_REVIEWER_RECEIPT = (
    "6540c438f02e25dc12df087c111745e7cc967449ff89061b121d072d8b9da044"
)
EXPECTED_HOST_BINDING = (
    "e1bba5f57fddc8044c9f4da29f437a59f913d217e89d42356f2bccceaaeac36f"
)
EXPECTED_DIRECTIVE_SHA256 = (
    "43a0302d6f8ae2ab2a7f92afc867df567458f66e99e8bd95a50ff1e953cc1abc"
)
EXPECTED_DIRECTIVE_REVISION = "ffb9b216879141e28570c2ae0307cd3b"
EXPECTED_DIRECTIVE_OBJECTIVE = (
    "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
)
EXPECTED_DIRECTIVE_SOURCE = (
    "manager.autonomous-supervision.position01-package0008-grant-v29"
)
EXPECTED_DIRECTIVE_TEXT = (
    "MANAGER DIRECTIVE POSITION-01 PACKAGE0008 GRANT V29: This supersedes all "
    "attempt0012 stewardship, Layer18/19 recovery, and package0007 disposition "
    "tasks. Package0008 is the sole review-accepted candidate: package root "
    f"{EXPECTED_PACKAGE_ROOT}; authority.json SHA256 {EXPECTED_AUTHORITY}; "
    f"validation0010 result {EXPECTED_VALIDATION_RESULT}; L2 adjudication "
    f"{EXPECTED_ADJUDICATION}; reviewer receipt {EXPECTED_REVIEWER_RECEIPT}; "
    f"Host binding {EXPECTED_HOST_BINDING}. Package/grant0007 remains consumed, "
    "failed, burned, and non-retryable.\n\n"
    "Authorize creation of exactly one create-exclusive UNCONSUMED Manager "
    "grant0008 only. Authenticate every listed digest, the complete read-only "
    "four-member snapshot, package-declared exact argv/cwd, live tools/prerequisites, "
    "capacity/quota contract, fresh consumption path "
    "reports/ace2-position01-integrated-runtime-pass-consumption-state-0008/"
    "consumed.json, fresh output reports/ace2-position01-runtime-pass-0002, and "
    "fresh scratch reports/ace2-position01-runtime-pass-scratch-0002. Bind durable "
    "consume-before-workload, failure publication, authority cardinality one, "
    "execution limit one, and permanent no-retry after any future live call. Fail "
    "closed and create no grant if any binding differs or namespace exists. Grant "
    "creation only: authority_consumed=false, execution_performed=false, zero "
    "executor, consumption, workload, output/scratch, checkpoint mutation, Stage1 "
    "closure, PPA, or Stage2."
)
EXPECTED_HISTORICAL_DIRECTIVE = {
    "file_sha256": "8820a5b40042f7772688c38972920c8d4e40f66bc3a12781e94f399865e2f3ef",
    "objective_sha256": EXPECTED_DIRECTIVE_OBJECTIVE,
    "revision": "15d57d19a7164a93abdb493764f1000d",
    "source": "manager.autonomous-supervision.position01-package0008-hashes-v28",
}
EXPECTED_ARGV = [
    "/home/argustest/miniconda3/bin/python3.13",
    "-B",
    (
        "/home/argustest/ace-2/reports/"
        "ace2-position01-integrated-runtime-pass-package-0008/"
        "run_position01_integrated.py"
    ),
    "--causal-manifest",
    "reports/ace2-position01-causal-state-successor-0004/manifest.json",
    "--consumed-record",
    (
        "reports/ace2-position01-integrated-runtime-pass-consumption-state-0008/"
        "consumed.json"
    ),
    "--output",
    "reports/ace2-position01-runtime-pass-0002",
]

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


def load_exact_file(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    status = path.lstat()
    require(
        stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode),
        f"{label} is not a regular non-symlink file",
    )
    if expected_mode is not None:
        require(
            stat.S_IMODE(status.st_mode) == expected_mode,
            f"{label} mode changed",
        )
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"{label} hash changed")
    return payload, load_json_bytes(payload, label)


def file_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    status = path.lstat()
    return {
        "bytes": status.st_size,
        "mode": stat.S_IMODE(status.st_mode),
        "path": relative(path),
        "sha256": expected_sha256,
    }


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_package_authority() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ace2_position01_package0008_authority", PACKAGE / "authority.py"
    )
    require(spec is not None and spec.loader is not None, "cannot load package authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_directive() -> tuple[bytes, dict[str, Any]]:
    payload, directive = load_exact_file(
        ACTIVE_DIRECTIVE,
        EXPECTED_DIRECTIVE_SHA256,
        "active Manager directive",
    )
    require(
        directive.get("version") == 1
        and directive.get("revision") == EXPECTED_DIRECTIVE_REVISION
        and directive.get("objective_sha256") == EXPECTED_DIRECTIVE_OBJECTIVE
        and directive.get("source") == EXPECTED_DIRECTIVE_SOURCE
        and directive.get("text") == EXPECTED_DIRECTIVE_TEXT,
        "active Manager directive identity or text changed",
    )
    return payload, directive


def validate_review_chain(
    authority_module: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _, validation = load_exact_file(
        VALIDATION_RESULT,
        EXPECTED_VALIDATION_RESULT,
        "validation0010 result",
        expected_mode=0o664,
    )
    _, adjudication = load_exact_file(
        ADJUDICATION,
        EXPECTED_ADJUDICATION,
        "L2 adjudication",
        expected_mode=0o444,
    )
    _, receipt = load_exact_file(
        REVIEWER_RECEIPT,
        EXPECTED_REVIEWER_RECEIPT,
        "Reviewer receipt",
        expected_mode=0o444,
    )
    _, host = load_exact_file(
        HOST_BINDING,
        EXPECTED_HOST_BINDING,
        "Host binding",
        expected_mode=0o444,
    )
    require(
        validation.get("status")
        == "PASS_POSITION01_INTEGRATED_RUNTIME_PASS_PACKAGE_0008_ZERO_WORKLOAD"
        and validation.get("execution_performed") is False
        and validation.get("package_tree_root_sha256") == EXPECTED_PACKAGE_ROOT
        and validation.get("activity", {}).get("protected_state_delta") == 0
        and validation.get("activity", {}).get("workload_execution") == 0,
        "validation0010 result no longer accepts the exact zero-workload package",
    )
    require(
        adjudication.get("decision") == "PASS"
        and adjudication.get("review_level") == "L2"
        and adjudication.get("execution_performed") is False
        and adjudication.get("package_tree_root_sha256") == EXPECTED_PACKAGE_ROOT
        and adjudication.get("validation", {}).get("status")
        == validation.get("status"),
        "L2 adjudication bindings changed",
    )
    require(
        receipt.get("decision") == "PASS"
        and receipt.get("execution_performed") is False
        and receipt.get("package", {}).get("tree_root_sha256")
        == EXPECTED_PACKAGE_ROOT
        and receipt.get("validation", {}).get("result_record", {}).get("sha256")
        == EXPECTED_VALIDATION_RESULT
        and receipt.get("verdict", {}).get("adjudication", {}).get("sha256")
        == EXPECTED_ADJUDICATION,
        "Reviewer receipt bindings changed",
    )
    expected_snapshot = authority_module.expected_snapshot_observation()
    host_members = host.get("snapshot", {}).get("members")
    require(
        host.get("status")
        == "PASS_PACKAGE0008_WHOLE_TREE_L2_REVIEW_AND_HOST_BINDING"
        and host.get("framework_l2_acceptance") is True
        and host.get("execution_performed") is False
        and host.get("package", {}).get("tree_root_sha256")
        == EXPECTED_PACKAGE_ROOT
        and host.get("validation", {}).get("result_record", {}).get("sha256")
        == EXPECTED_VALIDATION_RESULT
        and host.get("review", {}).get("adjudication", {}).get("sha256")
        == EXPECTED_ADJUDICATION
        and host.get("review", {}).get("reviewer_receipt", {}).get("sha256")
        == EXPECTED_REVIEWER_RECEIPT
        and host.get("boundaries", {}).get("stage1") == "OPEN"
        and host.get("boundaries", {}).get("stage2") == "FORBIDDEN",
        "Host binding changed",
    )
    require(
        host.get("snapshot", {}).get("revision") == expected_snapshot["revision"]
        and host.get("snapshot", {}).get("directory", {}).get("mode")
        == expected_snapshot["directory"]["mode"]
        and host.get("snapshot", {}).get("directory", {}).get("path")
        == expected_snapshot["directory"]["path"]
        and isinstance(host_members, dict)
        and set(host_members) == set(expected_snapshot["members"]),
        "Host snapshot binding changed",
    )
    for name, expected in expected_snapshot["members"].items():
        observed = host_members[name]
        require(
            observed.get("bytes") == expected["bytes"]
            and observed.get("mode") == expected["mode"]
            and observed.get("sha256") == expected["sha256"],
            f"Host snapshot member binding changed: {name}",
        )
    bindings = {
        "host_binding": file_binding(HOST_BINDING, EXPECTED_HOST_BINDING),
        "l2_adjudication": {
            **file_binding(ADJUDICATION, EXPECTED_ADJUDICATION),
            "decision": "PASS",
            "level": "L2",
        },
        "reviewer_receipt": file_binding(
            REVIEWER_RECEIPT, EXPECTED_REVIEWER_RECEIPT
        ),
        "validation": file_binding(VALIDATION_RESULT, EXPECTED_VALIDATION_RESULT),
    }
    return validation, bindings


def protected_snapshot(authority_module: Any) -> dict[str, Any]:
    return {
        "authority": authority_module.file_record(AUTHORITY),
        "model_snapshot": authority_module.observe_snapshot(),
        "native_sources": authority_module.native_source_records(),
        "package0007_terminal": authority_module.package0007_terminal_binding(),
        "package_root": authority_module.parse_package_tree(
            PACKAGE,
            EXPECTED_PACKAGE_ROOT,
            read_only=True,
        ),
        "prerequisites": authority_module.prerequisite_binding(),
        "review_chain": {
            "adjudication": sha256_file(ADJUDICATION),
            "host_binding": sha256_file(HOST_BINDING),
            "reviewer_receipt": sha256_file(REVIEWER_RECEIPT),
            "validation": sha256_file(VALIDATION_RESULT),
        },
        "toolchain": authority_module.toolchain_records(),
    }


def validate_live_package(
    authority_module: Any,
    authority_record: dict[str, Any],
    before: dict[str, Any],
) -> dict[str, Any]:
    require(before["package_root"] == EXPECTED_PACKAGE_ROOT, "package root changed")
    require(
        before["authority"]["sha256"] == EXPECTED_AUTHORITY,
        "authority.json hash changed",
    )
    require(
        authority_record.get("schema")
        == "ace2-position01-integrated-runtime-pass-authority-v8"
        and authority_record.get("stage1") == "OPEN"
        and authority_record.get("stage2") == "FORBIDDEN",
        "package stage boundary changed",
    )
    historical = authority_record.get("bindings", {}).get(
        "active_manager_directive", {}
    )
    require(
        historical.get("file", {}).get("sha256")
        == EXPECTED_HISTORICAL_DIRECTIVE["file_sha256"]
        and historical.get("objective_sha256")
        == EXPECTED_HISTORICAL_DIRECTIVE["objective_sha256"]
        and historical.get("revision") == EXPECTED_HISTORICAL_DIRECTIVE["revision"]
        and historical.get("source") == EXPECTED_HISTORICAL_DIRECTIVE["source"],
        "sealed package V28 construction-directive binding changed",
    )
    expected_snapshot = authority_module.expected_snapshot_observation()
    require(
        before["model_snapshot"] == expected_snapshot
        and authority_record["bindings"]["model_snapshot"] == expected_snapshot,
        "complete four-member model snapshot changed",
    )
    require(
        authority_record["bindings"]["package0007_terminal"]
        == before["package0007_terminal"],
        "package0007 consumed terminal burn binding changed",
    )
    require(
        authority_record["bindings"]["prerequisites"] == before["prerequisites"],
        "protected prerequisite binding changed",
    )
    for name, record in authority_record["bindings"]["repo_native_sources"].items():
        authority_module.validate_record(record, f"repo-native source {name}")
    require(
        authority_record["bindings"]["repo_native_sources"]
        == before["native_sources"],
        "repo-native source set changed",
    )
    for name, record in authority_record["bindings"]["toolchain"].items():
        authority_module.validate_record(record, f"toolchain member {name}")
    require(
        authority_record["bindings"]["toolchain"] == before["toolchain"],
        "toolchain binding changed",
    )
    authority_module.validate_snapshot()
    authority_module.validate_failure_ancestry()
    authority_module.validate_storage_measurement(
        authority_module.load_json(authority_module.STORAGE_MEASUREMENT)
    )
    prerequisite_checks = authority_module.validate_prerequisites(
        authority_record["bindings"]["prerequisites"]
    )
    simulator_paths = authority_module.authenticated_simulator_paths(authority_record)
    authority_module.validate_resource_arithmetic(authority_record["resource_guard"])
    available = authority_module.check_live_capacity()
    future = authority_record["future_execution_unit"]
    output_contract = authority_record["future_output_contract"]
    quota = output_contract["quota_enforcement"]
    latency = output_contract["terminal_latency_receipt"]
    require(
        future["exact_argv"] == EXPECTED_ARGV
        and authority_module.workload_argv() == EXPECTED_ARGV,
        "package-declared exact future argv changed",
    )
    require(
        authority_module.ROOT == ROOT
        and authority_module.PACKAGE == PACKAGE
        and authority_module.GRANT == GRANT_NAMESPACE
        and authority_module.CONSUMPTION == CONSUMPTION
        and authority_module.OUTPUT == OUTPUT
        and authority_module.SCRATCH == SCRATCH,
        "package-declared cwd or future namespaces changed",
    )
    require(
        authority_record["consumption"]["future_path"]
        == relative(CONSUMPTION / "consumed.json")
        and authority_record["consumption"]["consume_before_any_output_or_workload"]
        is True
        and authority_record["consumption"]["repeat_use"] == "REJECT",
        "consume-before-workload contract changed",
    )
    require(
        output_contract["namespace"] == relative(OUTPUT)
        and quota["scratch_namespace"] == relative(SCRATCH)
        and quota["output_max_bytes"]
        == authority_record["resource_guard"]["runtime_output_max_bytes"]
        and quota["scratch_max_bytes"]
        == authority_record["resource_guard"]["temporary_write_max_bytes"],
        "output/scratch quota contract changed",
    )
    require(
        latency["published_after_terminal_fsync"] is True
        and latency["w4a8_lm_head_binding"]
        == "EXACT_CYCLES_AND_MEASURED_ICARUS_WALL_SECONDS_FROM_EXECUTION_DETAILS"
        and latency["total_latency"]
        == "MONOTONIC_WALL_SECONDS_FROM_POST_CONSUMPTION_TO_DURABLE_TERMINAL",
        "terminal latency publication contract changed",
    )
    require(
        shutil.which("iverilog") == simulator_paths["iverilog"]
        and shutil.which("vvp") == simulator_paths["vvp"],
        "live simulator path changed",
    )
    return {
        "available_bytes": available,
        "checks": {
            "complete_model_snapshot": "PASS_EXACT_READ_ONLY_FOUR_MEMBER_BINDING",
            "historical_package_directive": (
                "PASS_SEALED_V28_CONSTRUCTION_BINDING_RETAINED"
            ),
            "package0007": "PASS_CONSUMED_TERMINALLY_BURNED_NON_RETRYABLE",
            "prerequisites": prerequisite_checks,
            "repo_native_sources": "PASS_EXACT_LIVE_HASH_MODE_BINDINGS",
            "toolchain": "PASS_EXACT_LIVE_HASH_MODE_PATH_BINDINGS",
        },
        "simulator_paths": simulator_paths,
    }


def validate_fresh_namespaces() -> dict[str, Any]:
    namespaces = {
        "consumption": CONSUMPTION,
        "grant": GRANT_NAMESPACE,
        "output": OUTPUT,
        "scratch": SCRATCH,
    }
    for label, path in namespaces.items():
        require(not os.path.lexists(path), f"reserved package0008 {label} namespace exists")
    require(
        not os.path.lexists(CONSUMPTION / "consumed.json"),
        "package0008 consumption record exists",
    )
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
    directive_bytes: bytes,
    directive: dict[str, Any],
    review_bindings: dict[str, dict[str, Any]],
    live: dict[str, Any],
    before: dict[str, Any],
    reserved_namespaces: dict[str, Any],
) -> dict[str, Any]:
    resource_guard = authority_record["resource_guard"]
    output_contract = authority_record["future_output_contract"]
    historical_directive = authority_record["bindings"]["active_manager_directive"]
    return {
        "activity": ZERO_ACTIVITY,
        "append_only_creation": {
            "namespace": relative(GRANT_NAMESPACE),
            "path": relative(GRANT),
            "semantics": "CREATE_EXCLUSIVE_NEW_PACKAGE0008_MANAGER_GRANT",
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
            "host_binding": review_bindings["host_binding"],
            "l2_adjudication": review_bindings["l2_adjudication"],
            "model_snapshot": before["model_snapshot"],
            "package": {
                "authority_path": relative(AUTHORITY),
                "authority_sha256": EXPECTED_AUTHORITY,
                "historical_construction_directive": {
                    "objective_sha256": historical_directive["objective_sha256"],
                    "revision": historical_directive["revision"],
                    "sha256": historical_directive["file"]["sha256"],
                    "source": historical_directive["source"],
                },
                "path": relative(PACKAGE),
                "tree_root_sha256": EXPECTED_PACKAGE_ROOT,
            },
            "reviewer_receipt": review_bindings["reviewer_receipt"],
            "validation": review_bindings["validation"],
        },
        "capacity_and_quota": {
            "available_bytes_at_grant": live["available_bytes"],
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
            "failure_after_any_future_live_call": (
                "AUTHORITY_REMAINS_CONSUMED_PERMANENT_NO_RETRY"
            ),
            "repeat_use": "REJECT",
        },
        "consumption_state": "UNCONSUMED",
        "decision": (
            "AUTHORIZE_EXACTLY_ONE_POSITION01_INTEGRATED_RUNTIME_PASS_PACKAGE0008"
        ),
        "exact_invocation": {"argv": EXPECTED_ARGV, "cwd": str(ROOT)},
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
            "dispatch_checks": live["checks"],
            "simulator_paths": live["simulator_paths"],
        },
        "origin": "ACTIVE_MANAGER_DIRECTIVE_POSITION01_PACKAGE0008_GRANT_V29",
        "predecessor_preservation": before["package0007_terminal"],
        "producer_role": "manager",
        "prohibited_activity": {
            key: value
            for key, value in ZERO_ACTIVITY.items()
            if key != "manager_grant_created"
        },
        "reserved_namespaces": reserved_namespaces,
        "schema": "ace2-position01-integrated-runtime-pass-package0008-manager-grant-v1",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "terminal_publication": {
            "execution_details_path": output_contract["execution_details_path"],
            "failure_path": relative(CONSUMPTION / "failure.json"),
            "lm_head_and_total_latency_receipt": output_contract[
                "terminal_latency_receipt"
            ],
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
            remaining = memoryview(payload)
            while remaining:
                written = os.write(file_fd, remaining)
                require(written > 0, "grant write made no progress")
                remaining = remaining[written:]
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


def validate_created_grant(
    authority_module: Any,
    expected: dict[str, Any],
    payload: bytes,
    directive_bytes: bytes,
    before: dict[str, Any],
) -> str:
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
    require({path.name for path in GRANT_NAMESPACE.iterdir()} == {"grant.json"}, "grant cardinality changed")
    require(GRANT.read_bytes() == payload, "created grant bytes changed")
    require(
        load_json_bytes(payload, "created grant") == expected,
        "created grant JSON changed",
    )
    require(
        ACTIVE_DIRECTIVE.read_bytes() == directive_bytes,
        "Manager directive changed during grant issue",
    )
    for path in (CONSUMPTION, OUTPUT, SCRATCH):
        require(not os.path.lexists(path), f"prohibited namespace created: {path}")
    after = protected_snapshot(authority_module)
    require(before == after, "protected package, evidence, prerequisite, or tool state changed")
    require(expected["activity"]["protected_state_delta"] == 0, "nonzero protected-state delta")
    return sha256_bytes(payload)


def main() -> int:
    require(Path.cwd() == ROOT, "exact repository cwd required")
    directive_bytes, directive = validate_directive()
    reserved_namespaces = validate_fresh_namespaces()
    authority_module = load_package_authority()
    _, authority_record = load_exact_file(
        AUTHORITY,
        EXPECTED_AUTHORITY,
        "package0008 authority",
        expected_mode=0o444,
    )
    before = protected_snapshot(authority_module)
    live = validate_live_package(authority_module, authority_record, before)
    _, review_bindings = validate_review_chain(authority_module)
    require(
        ACTIVE_DIRECTIVE.read_bytes() == directive_bytes,
        "Manager directive changed before grant issue",
    )
    validate_fresh_namespaces()
    grant = build_grant(
        authority_record,
        directive_bytes,
        directive,
        review_bindings,
        live,
        before,
        reserved_namespaces,
    )
    payload = canonical_bytes(grant)
    create_grant(payload)
    grant_sha256 = validate_created_grant(
        authority_module,
        grant,
        payload,
        directive_bytes,
        before,
    )
    sys.stdout.write(
        json.dumps(
            {
                "authority_consumed": False,
                "available_bytes_at_grant": live["available_bytes"],
                "consumption_state": "UNCONSUMED",
                "decision": "PASS_PACKAGE0008_MANAGER_GRANT_CREATED_AND_VALIDATED",
                "execution_performed": False,
                "grant_path": relative(GRANT),
                "grant_sha256": grant_sha256,
                "package_tree_root_sha256": EXPECTED_PACKAGE_ROOT,
                "status": "PASS",
                "workload_execution": 0,
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
