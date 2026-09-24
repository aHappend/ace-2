#!/usr/bin/env python3
"""Build one immutable, static-only V9 execution-authority successor package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
PACKAGE_ROOT = ROOT / "build/v9-single-request-execution-authority-successor-package-v1-candidate-0004"
REVIEW_ROOT = ROOT / "build/v9-single-request-execution-authority-successor-package-v1-candidate-0004-review"
MANIFEST_NAME = "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE.json"
VERIFIER_SOURCE = ROOT / "tools/v9_execution_authority_successor_verify_independent.py"
DISPOSITION = "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE_READY_NO_AUTHORITY_CONSUMED"
PACKAGE_ID = "v9-single-request-execution-authority-successor-package-v1-candidate-0004"
ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
BASE_IDENTITY = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
EXPECTED_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
ACTION_ROOT = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root"
EXPECTED_ARGV = [
    "/home/argustest/miniconda3/bin/python3.13",
    str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v9.py"),
    "--package",
    str(ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_PACKAGE.json"),
    "--acceptance",
    str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
    "--irreversible-action-id",
    ACTION_ID,
]
EXPECTED = {
    "execution_domain": {
        "root": ROOT / "build/v9-kernel-verifiable-execution-domain-package-v3-repair-attempt-0004",
        "manifest": "V9_KERNEL_VERIFIABLE_EXECUTION_DOMAIN_PACKAGE_V3.json",
        "manifest_raw_sha256": "1b7d73ffef214dffa20f0b2a9a70c65ac04f75b152fdcabd378970260b1e071b",
        "package_content_sha256": "9066b2d9a91c94d464be3598a24977fef947adcfa5ae68ba7e222e457e17c9fa",
        "accepted_disposition": "V9_KERNEL_VERIFIABLE_EXECUTION_DOMAIN_PACKAGE_READY_NO_AUTHORITY_CONSUMED",
        "review": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b9v9execdomainpkg01/round-0007.json"),
        "review_raw_sha256": "4f1252d4e8cb17ba8bbb8e70816975b1d90b61f27cd42435a8ae2d501ea5be49",
        "review_round": 7,
    },
    "direct_spawn": {
        "root": ROOT / "build/v9-direct-spawn-bridge-v1",
        "manifest": "V9_DIRECT_SPAWN_BRIDGE_V1_PACKAGE.json",
        "manifest_raw_sha256": "93c352e24120336203afb009301fa3b45f8ffec0ed32e65a80baca50630784ce",
        "package_content_sha256": "7c4923e7862167bd3b2fd52d5e8ee8d58d0288d79a5b6ed209f3cf6278fdb22b",
        "accepted_disposition": "PACKAGE_READY_NO_EXECUTION_AUTHORITY",
        "review": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b9v9directspawn01/round-0003.json"),
        "review_raw_sha256": "7368814cff8a0052e5fd73891551c9604d5778325c8ec7be0256266d6f5ad0b4",
        "review_round": 3,
    },
    "activation": {
        "root": ROOT / "build/v9-one-shot-activation-v2-repair-attempt-0003",
        "manifest": "V9_ONE_SHOT_ACTIVATION_V2_PACKAGE.json",
        "manifest_raw_sha256": "db5a6ecaf9af67eb575cc413067842d39d78796834cddcc2adab34f4e158b1cf",
        "package_content_sha256": "dd18531f1047b0170ca2c1947701e55ca6ad8b2804d77ec5de9c5f7bab5ecbc8",
        "accepted_disposition": "V9_ONE_SHOT_ACTIVATION_V2_PACKAGE_READY_NO_EXECUTION_AUTHORITY",
        "review": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b9v9activationv2fix01/round-0003.json"),
        "review_raw_sha256": "193fdf5099ec2989b63cebbd31e78e04ea9ef0061464eda4c49fc90be6a3fcd7",
        "review_round": 3,
    },
    "broker": {
        "root": ROOT / "build/v9-broker-start-authority-v2-repair-attempt-0002",
        "manifest": "V9_BROKER_START_AUTHORITY_V2_PACKAGE.json",
        "manifest_raw_sha256": "5e0a4798cf710ca7d25ad630db011022eab651dbd7283fb592d08209aed11ccf",
        "package_content_sha256": "3717d6969abf049a162c99ec474d200c91222baf6a18e30daaa06dad59a0ac68",
        "accepted_disposition": "BROKER_START_AUTHORITY_PACKAGE_READY_NO_AUTHORITY_CONSUMED",
        "review": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b9v9brokerauthpkg02/round-0002.json"),
        "review_raw_sha256": "ab58c972f8dfec59fe863693df69f9b383e8b62f079d82ac3df378ba0fd42341",
        "review_round": 2,
    },
}
CANONICAL_EXPECTED = {
    "manifest_raw_sha256": "14fae526ce7c8c65e898eaa5400166855f214fb28dd676693632d3a85178727f",
    "package_content_sha256": "099be430faf60888b92ca5f358a21ee16f860360843a98f0f0944aeace47305f",
    "acceptance_raw_sha256": "8b969b27120243f8ce5787a6bd598a1d2f18da8b3e0728434177fcbbd89a42cb",
    "acceptance_self_sha256": "2a2e82431a399b220f27f581de49dc12104ae0c25572dbcb502073eae5cb10cf",
    "launcher_sha256": "0857368ec09ab6c69ea5b22482c34a214298e6f8bcf41c19a9bcf5b2494de513",
    "transport_sha256": "7e03a35881642bd5a93b7047ece920122534c7afab4bbf773bbc22d9ef69faf7",
    "static_verifier_sha256": "8e687659b4f8eb3f8262adf0e791ae2db662b8f2a9367d456ad48b6ee60552cc",
}
PUBLICATION_FIELDS = [
    "request_id", "run_anchor_sha256", "domain_admission_sha256", "contract_path", "contract_sha256",
    "contract_device", "contract_inode", "contract_mode", "contract_size", "action_id", "supervisor_path",
    "supervisor_sha256", "supervisor_device", "supervisor_inode", "supervisor_mode", "supervisor_size",
    "run_public_key_ed25519", "request_consumption", "request_consumption_count", "request_queue_observation",
    "witness_pid", "witness_pidfd_registered", "witness_terminal_channel", "executor_pidfd_opened",
    "executor_pidfd_dead_at_seal", "supervisor_pid", "supervisor_pidfd_opened", "supervisor_pidfd_dead",
    "domain_admission_status", "receipt_directory_device", "receipt_directory_inode", "receipt_count",
    "receipt_chain_root_sha256", "supervisor_wait_kind", "supervisor_wait_value", "status", "failure_stage",
    "authority_consumed", "production_state_created", "record_sha256", "signature_ed25519",
]
REVIEW_CONTRACT = """# V9 single-request execution-authority successor review contract

This package is static-only. It grants no authority and may not start the activation controller, broker, transport, launcher, evaluator, model, RTL, XRT, U280, cgroup, or PID-namespace path. Candidates 0001 through 0003 and their available review evidence are immutable predecessors; candidate-0003 received a substantive `done` review but its wrapper rejected brittle prose-format requirements.

Fresh L2 must independently rerun `verify_package_independent.py` from an empty environment with `LANG=C`, `LC_ALL=C`, `PYTHONHASHSEED=0`, and `TZ=UTC`. The reviewer must not import candidate code; the package contains only this review contract, a canonical manifest, its sidecar, and the independent verifier.

The review must adversarially inspect these required areas:

- `v9.successor.provenance-chain`: exact accepted execution-domain attempt-0004, direct-spawn, activation, broker, canonical V9, policy, and predecessor identities are bound and live.
- `v9.successor.request-output-descriptors`: the request, outer V9ED publication, and inner canonical V9 first-terminal/result descriptors are independently self-hashed, source/schema cross-checked, and replacement is rejected.
- `v9.successor.one-consumption-terminal-publication`: one request maximum, no replay/retry/replacement, and terminal publication after consumption are explicit.
- `v9.successor.interpreter-input-fd-binding`: exact and dynamic interpreter inputs use retained source FDs, fully sealed memfd snapshots, seccomp ADDFD read-only injection, and required `INPUT_FD_BOUND` receipts.
- `v9.successor.no-production-state`: all authority, credential, claim, request, socket, live, production, consumption, and terminal paths remain absent and target starts remain zero.
- `v9.successor.canonical-historical-verifier-not-run`: the old canonical verifier is preserved byte-for-byte but is not run because its frozen file policy predates the accepted review file and is expected to report `FAIL V9 exact static file set`.
- `v9.successor.sealed-immutable`: the candidate root is mode 0555, its exact four files are mode 0444, and all hashes close.

The outer publication vocabulary is exactly `SUCCEEDED_TERMINAL`, `REJECTED_TERMINAL_CONSUMED`, `CRASH_TERMINAL_CONSUMED`, and `UNKNOWN_TERMINAL_CONSUMED`. The inner canonical first-terminal vocabulary is exactly `SUCCEEDED_TERMINAL`, `FAILED_TERMINAL`, `CONSUMED_ORPHAN`, and `PREFLIGHT_FAILED_TERMINAL`. They are distinct layers and must never be merged or renamed.

The sealed reviewer-verdict record may normalize a substantive independent `done` decision into the exact disposition `V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE_READY_NO_AUTHORITY_CONSUMED`; no exact prose template is required from the reviewer. A done review grants no execution permission. The real G8/G4/G2/G1 sweep requires a separate later execution mission.
"""


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs)
    require(type(value) is dict and not duplicate and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def self_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(value))
    return result


def verify_self_hash(value: dict[str, Any], field: str, expected: str) -> None:
    require(value[field] == expected, f"self hash field mismatch: {field}")
    observed = sha256_bytes(compact_bytes({key: item for key, item in value.items() if key != field}))
    require(observed == expected, f"self hash mismatch: {field}")


def verify_domain_content_hash(value: dict[str, Any], expected: str) -> None:
    require(value["package_content_sha256"] == expected, "domain content hash field")
    payload = {key: item for key, item in value.items() if key != "package_content_sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    require(sha256_bytes(raw) == expected, "domain content hash")


def mode_octal(path: Path) -> str:
    return f"{stat.S_IMODE(os.lstat(path).st_mode):04o}"


def inventory(root: Path) -> dict[str, Any]:
    observed_root = os.lstat(root)
    require(stat.S_ISDIR(observed_root.st_mode) and not stat.S_ISLNK(observed_root.st_mode), f"invalid root: {root}")
    files: dict[str, Any] = {}
    directories: dict[str, Any] = {".": {"mode_octal": mode_octal(root)}}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        observed = os.lstat(path)
        require(not stat.S_ISLNK(observed.st_mode), f"symlink in inventory: {path}")
        if stat.S_ISDIR(observed.st_mode):
            directories[relative] = {"mode_octal": mode_octal(path)}
        else:
            require(stat.S_ISREG(observed.st_mode), f"non-regular entry: {path}")
            files[relative] = {"byte_count": observed.st_size, "mode_octal": mode_octal(path), "sha256": sha256_file(path)}
    return {"directories": directories, "files": files}


def verify_review(path: Path, expected_hash: str, disposition: str, round_index: int) -> None:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_hash, f"review hash mismatch: {path}")
    value = json.loads(raw.decode("utf-8"))
    review = value.get("review", {})
    require(value.get("producer_role") == "reviewer" and value.get("round") == round_index, f"review identity: {path}")
    require(review.get("status") == "done" and disposition in str(review.get("reason", "")), f"review disposition: {path}")


def dependency_binding(label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = EXPECTED[label]
    root = expected["root"]
    manifest_path = root / expected["manifest"]
    manifest, raw = read_canonical(manifest_path)
    require(sha256_bytes(raw) == expected["manifest_raw_sha256"], f"{label} raw manifest hash")
    if label == "execution_domain":
        verify_domain_content_hash(manifest, expected["package_content_sha256"])
    else:
        verify_self_hash(manifest, "package_content_sha256", expected["package_content_sha256"])
    require(manifest.get("acceptance_boundary") == expected["accepted_disposition"], f"{label} acceptance boundary")
    verify_review(expected["review"], expected["review_raw_sha256"], expected["accepted_disposition"], expected["review_round"])
    binding = {
        "accepted_disposition": expected["accepted_disposition"],
        "inventory": inventory(root),
        "manifest_path": str(manifest_path),
        "manifest_raw_sha256": expected["manifest_raw_sha256"],
        "package_content_sha256": expected["package_content_sha256"],
        "review_path": str(expected["review"]),
        "review_raw_sha256": expected["review_raw_sha256"],
        "review_round": expected["review_round"],
        "root": str(root),
    }
    return binding, manifest


def artifact(path: Path, relative: str) -> dict[str, Any]:
    return {"byte_count": path.stat().st_size, "mode_octal": mode_octal(path), "path": relative, "sha256": sha256_file(path)}


def main() -> int:
    require(not PACKAGE_ROOT.exists(), f"refusing to overwrite: {PACKAGE_ROOT}")
    require(not REVIEW_ROOT.exists(), f"review root already exists: {REVIEW_ROOT}")
    require(VERIFIER_SOURCE.is_file(), "independent verifier source is absent")

    provenance: dict[str, Any] = {}
    dependency_manifests: dict[str, dict[str, Any]] = {}
    for label in ("execution_domain", "direct_spawn", "activation", "broker"):
        provenance[label], dependency_manifests[label] = dependency_binding(label)

    canonical_path = ACTION_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_PACKAGE.json"
    canonical, canonical_raw = read_canonical(canonical_path)
    require(sha256_bytes(canonical_raw) == CANONICAL_EXPECTED["manifest_raw_sha256"], "canonical manifest raw hash")
    verify_self_hash(canonical, "package_content_sha256", CANONICAL_EXPECTED["package_content_sha256"])
    acceptance_path = ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
    acceptance, acceptance_raw = read_canonical(acceptance_path)
    require(sha256_bytes(acceptance_raw) == CANONICAL_EXPECTED["acceptance_raw_sha256"], "canonical acceptance raw hash")
    verify_self_hash(acceptance, "acceptance_sha256", CANONICAL_EXPECTED["acceptance_self_sha256"])
    launcher_path = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v9.py"
    transport_path = ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v9.py"
    static_verifier_path = ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree.py"
    require(sha256_file(launcher_path) == CANONICAL_EXPECTED["launcher_sha256"], "launcher hash")
    require(sha256_file(transport_path) == CANONICAL_EXPECTED["transport_sha256"], "transport hash")
    require(sha256_file(static_verifier_path) == CANONICAL_EXPECTED["static_verifier_sha256"], "static verifier hash")
    require(canonical["official_identities"]["model_sha256"] == BASE_IDENTITY, "Base identity")
    require(canonical["future_invocation"]["argv"] == EXPECTED_ARGV, "canonical argv")
    require(canonical["future_invocation"]["environment"] == EXPECTED_ENVIRONMENT, "canonical environment")
    require(canonical["future_invocation"]["invocation_sha256"] == "e329556cae9d7903705c03ded4e3b347ed52007fa6e2c175887af820041e1c83", "transport invocation hash")
    require(canonical["launcher_invocation"]["invocation_sha256"] == "55d685e3ca5420529c131e50fffcee276e15e96ff8726ea5e5dbcf40ee085635", "launcher invocation hash")
    static_policy = canonical["static_file_policy"]
    require("review/FRESH_L2_STATIC_ACCEPTANCE.json" not in static_policy["allowed_relative_files"] and "review" in static_policy["forbidden_directories"], "historical static verifier incompatibility")
    provenance["canonical_v9"] = {
        "acceptance_path": str(acceptance_path),
        "acceptance_raw_sha256": CANONICAL_EXPECTED["acceptance_raw_sha256"],
        "acceptance_self_sha256": CANONICAL_EXPECTED["acceptance_self_sha256"],
        "inventory": inventory(ACTION_ROOT),
        "launcher_path": str(launcher_path),
        "launcher_sha256": CANONICAL_EXPECTED["launcher_sha256"],
        "manifest_path": str(canonical_path),
        "manifest_raw_sha256": CANONICAL_EXPECTED["manifest_raw_sha256"],
        "package_content_sha256": CANONICAL_EXPECTED["package_content_sha256"],
        "root": str(ACTION_ROOT),
        "static_verifier_path": str(static_verifier_path),
        "static_verifier_sha256": CANONICAL_EXPECTED["static_verifier_sha256"],
        "transport_path": str(transport_path),
        "transport_sha256": CANONICAL_EXPECTED["transport_sha256"],
    }

    legacy_manifest_path = ROOT / "build/v9-single-request-execution-authority-package-v1-repair-attempt-0006/V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_PACKAGE.json"
    legacy, legacy_raw = read_canonical(legacy_manifest_path)
    require(sha256_bytes(legacy_raw) == "c940371696dc977e6b7fca4618d1eca71fbbc8b788516f7f7fd6faa7f4f17648", "attempt-0006 raw hash")
    forbidden_paths = set(legacy["forbidden_actual_state"]["paths"])
    forbidden_paths.update(dependency_manifests["execution_domain"]["forbidden_actual_state"])
    for path_text in sorted(forbidden_paths):
        require(not os.path.lexists(path_text), f"forbidden production/live state exists: {path_text}")

    predecessor_roots = sorted({
        path
        for pattern in (
            "v9-single-request-execution-authority-package-v1*",
            "v9-single-request-execution-authority-successor-package-v1-candidate-0001*",
            "v9-single-request-execution-authority-successor-package-v1-candidate-0002*",
            "v9-single-request-execution-authority-successor-package-v1-candidate-0003*",
        )
        for path in (ROOT / "build").glob(pattern)
        if path.is_dir()
    })
    require(predecessor_roots, "no preserved predecessor roots")
    preserved_predecessors = [{"inventory": inventory(path), "root": str(path)} for path in predecessor_roots]

    PACKAGE_ROOT.mkdir(mode=0o700)
    verifier_path = PACKAGE_ROOT / "verify_package_independent.py"
    review_contract_path = PACKAGE_ROOT / "REVIEW_CONTRACT.md"
    shutil.copyfile(VERIFIER_SOURCE, verifier_path)
    review_contract_path.write_text(REVIEW_CONTRACT, encoding="utf-8")
    os.chmod(verifier_path, 0o444)
    os.chmod(review_contract_path, 0o444)

    domain_artifacts = {item["path"]: item["sha256"] for item in dependency_manifests["execution_domain"]["artifact_bindings"]}
    request_descriptor = self_hash({
        "action_id": ACTION_ID,
        "argv": EXPECTED_ARGV,
        "candidate_order": ["G8", "G4", "G2", "G1"],
        "command_representation": "ARGV_VECTOR_ONLY",
        "cwd": str(ACTION_ROOT),
        "environment": EXPECTED_ENVIRONMENT,
        "environment_inheritance_permitted": False,
        "every_hard_gate_required": True,
        "invocation_sha256": "e329556cae9d7903705c03ded4e3b347ed52007fa6e2c175887af820041e1c83",
        "launcher_invocation_sha256": "55d685e3ca5420529c131e50fffcee276e15e96ff8726ea5e5dbcf40ee085635",
        "request_cardinality": 1,
        "selection_policy": "FIRST_ALL_HARD_GATES_PASS_IN_FIXED_ORDER",
        "shell": False,
        "sweep_scope": "BASE_GRANULARITY_ONLY",
    }, "descriptor_sha256")
    execution_domain_publication_descriptor = self_hash({
        "authority_consumed": "false",
        "format": "V9ED-PUBLICATION-V3",
        "publication_fields": PUBLICATION_FIELDS,
        "production_state_created": "false",
        "request_consumption": "CONSUMED_ONCE",
        "request_consumption_count": "1",
        "request_queue_observation": "EMPTY_AFTER_CONSUMPTION",
        "source_path": str(EXPECTED["execution_domain"]["root"] / "verify_run_publication_independent.py"),
        "source_sha256": domain_artifacts["verify_run_publication_independent.py"],
        "status_values": ["CRASH_TERMINAL_CONSUMED", "REJECTED_TERMINAL_CONSUMED", "SUCCEEDED_TERMINAL", "UNKNOWN_TERMINAL_CONSUMED"],
    }, "descriptor_sha256")
    first_terminal_schema = json.loads((ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_FIRST_TERMINAL_SCHEMA.json").read_text(encoding="ascii"))
    canonical_v9_terminal_descriptor = self_hash({
        "first_terminal_create_only": True,
        "first_terminal_path": str(ACTION_ROOT / "live/authority/base/first-terminal.json"),
        "reason_code_values": first_terminal_schema["properties"]["reason_code"]["enum"],
        "replacement_permitted": False,
        "required_fields": first_terminal_schema["required"],
        "result_path": str(ACTION_ROOT / "live/result/base/result.json"),
        "result_terminal_status_values": ["FAILED_TERMINAL", "SUCCEEDED_TERMINAL"],
        "schema_path": str(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_FIRST_TERMINAL_SCHEMA.json"),
        "schema_sha256": "54955aec6fcc9af52bb7c18468f16f1edfc8e2d407783d1d7b409edc2f55fa9b",
        "status_values": first_terminal_schema["properties"]["status"]["enum"],
    }, "descriptor_sha256")
    interpreter_binding = self_hash({
        "dynamic_binding_contract": dependency_manifests["execution_domain"]["execution_domain_contract"]["dynamic_binding"],
        "exact_and_dynamic_bindings_required": True,
        "input_delivery": "SEALED_MEMFD_READ_ONLY_SECCOMP_IOCTL_NOTIF_ADDFD_NO_PATHNAME_CONTINUE",
        "missing_input_fd_bound_receipt_rejects": True,
        "production_contract_sha256": domain_artifacts["production.contract"],
        "replacement_interpreter_input_inodes_consumed": 0,
        "source_or_ast_inference": "PROHIBITED",
        "verify_dynamic_interpreter_fd_binding_sha256": domain_artifacts["verify_dynamic_interpreter_fd_binding_independent.py"],
        "verify_interpreter_fd_binding_sha256": domain_artifacts["verify_interpreter_fd_binding_independent.py"],
        "verify_receipts_sha256": domain_artifacts["verify_receipts_independent.py"],
    }, "binding_sha256")

    manifest = {
        "acceptance_boundary": DISPOSITION,
        "artifact_bindings": [artifact(review_contract_path, "REVIEW_CONTRACT.md"), artifact(verifier_path, "verify_package_independent.py")],
        "artifact_kind": "v9_single_request_execution_authority_successor_static_package",
        "base_identity_sha256": BASE_IDENTITY,
        "canonical_historical_incompatibility": {
            "current_acceptance_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
            "expected_failure": "FAIL V9 exact static file set",
            "old_policy_omits_current_acceptance": True,
            "old_verifier_execution_permitted": False,
            "preservation_policy": "PRESERVE_BYTES_AND_FAILURE_AS_HISTORICAL_EVIDENCE_NO_IN_PLACE_REPAIR",
        },
        "claim_boundary": {
            "activation_invoked": False,
            "authority_consumed": False,
            "authority_materialized": False,
            "broker_invoked": False,
            "cgroup_created": False,
            "claim_materialized": False,
            "credential_materialized": False,
            "launcher_invoked": False,
            "live_state_materialized": False,
            "pid_namespace_created": False,
            "production_state_materialized": False,
            "request_materialized": False,
            "socket_materialized": False,
            "target_process_started": False,
            "terminal_materialized": False,
            "transport_invoked": False,
        },
        "forbidden_actual_state": {
            "paths": sorted(forbidden_paths),
            "target_argv_markers": legacy["forbidden_actual_state"]["target_argv_markers"],
        },
        "execution_domain_publication_descriptor": execution_domain_publication_descriptor,
        "interpreter_input_fd_binding": interpreter_binding,
        "one_consumption_contract": {
            "canonical_v9_post_ledger_exception_terminal": "CONSUMED_ORPHAN",
            "consumption_cardinality_maximum": 1,
            "execution_domain_post_consumption_unknown": "UNKNOWN_TERMINAL_CONSUMED",
            "initial_state": "READY_UNCONSUMED",
            "prohibited_operations": ["RETRY", "REPLAY", "RESUME", "REPAIR", "REPLACEMENT"],
            "request_replacement_permitted": False,
            "terminal_publication_required_after_consumption": True,
        },
        "canonical_v9_terminal_descriptor": canonical_v9_terminal_descriptor,
        "cross_layer_terminal_contract": {
            "execution_domain_and_canonical_vocabularies_distinct": True,
            "mapping_policy": "NO_TOTAL_STATUS_MAPPING_CLAIMED_VALIDATE_LAYERS_INDEPENDENTLY",
            "outer_success_requires_canonical_success_terminal": True,
            "status_renaming_or_unification_permitted": False,
        },
        "package_id": PACKAGE_ID,
        "preserved_predecessors": preserved_predecessors,
        "protected_policy_bindings": {
            "FAST_LOOP_POLICY.json": {"path": str(ROOT / "design/FAST_LOOP_POLICY.json"), "sha256": "18b5b76950e713939e105d2b2df2f65f41e3d27dcb486f5d05791a0a91c984a0"},
            "MISSION.md": {"path": str(ROOT / "MISSION.md"), "sha256": "61b54a93ce4b973d91d7eb96fb6866d23d2dc8308a07c2f091750728f25cfdef"},
        },
        "provenance_chain": provenance,
        "request_descriptor": request_descriptor,
        "review_status": "PENDING_FRESH_L2",
        "schema_version": 1,
        "static_policy": {
            "creates_authority_or_credentials": False,
            "execution_authorized_by_readiness": False,
            "independent_verifier_imports_candidate_code": False,
            "predecessor_mutation_permitted": False,
            "production_or_live_writes_permitted": False,
            "separate_execution_mission_required": True,
            "static_only": True,
        },
        "test_policy": {
            "adversarial_manifest_mutations": 23,
            "old_canonical_verifier_runs": 0,
            "production_paths_must_remain_absent": True,
            "target_process_starts_permitted": 0,
        },
    }
    manifest = self_hash(manifest, "package_content_sha256")
    manifest_path = PACKAGE_ROOT / MANIFEST_NAME
    manifest_path.write_bytes(compact_bytes(manifest))
    sidecar_path = PACKAGE_ROOT / f"{MANIFEST_NAME}.sha256"
    sidecar_path.write_text(f"{sha256_file(manifest_path)}  {MANIFEST_NAME}\n", encoding="ascii")
    os.chmod(manifest_path, 0o444)
    os.chmod(sidecar_path, 0o444)
    os.chmod(PACKAGE_ROOT, 0o555)
    print(json.dumps({
        "disposition": DISPOSITION,
        "manifest_raw_sha256": sha256_file(manifest_path),
        "package_content_sha256": manifest["package_content_sha256"],
        "package_root": str(PACKAGE_ROOT),
        "status": "SEALED_STATIC_SUCCESSOR_PACKAGE",
    }, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
