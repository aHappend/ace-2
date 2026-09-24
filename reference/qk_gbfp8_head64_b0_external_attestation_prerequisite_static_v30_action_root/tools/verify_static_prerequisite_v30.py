#!/usr/bin/env python3
"""Verify the inert V30 external-attestation prerequisite package.

This tool never verifies or accepts a production receipt.  Its cryptographic
exercise uses an in-memory TEST-ONLY key and exists solely to prove that the
production gate rejects test credentials and exact-binding violations.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import stat
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import jsonschema
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


ACTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ACTION_ROOT.parents[1]
PACKAGE_PATH = ACTION_ROOT / "QK_GBFP8_HEAD64_B0_EXTERNAL_ATTESTATION_PREREQUISITE_STATIC_V30_PACKAGE.json"
BUNDLE_SCHEMA_PATH = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_B0_V30_SIGNED_ATTESTATION_BUNDLE_SCHEMA.json"
FRAMEWORK_SCHEMA_PATH = ACTION_ROOT / "reference/QK_GBFP8_HEAD64_B0_V30_EXTERNAL_FRAMEWORK_CONTRACT_SCHEMA.json"

PACKAGE_NAME = PACKAGE_PATH.name
REPORT_KIND = "qk_gbfp8_head64_b0_external_attestation_prerequisite_static_v30_report"
BLOCKED_STATUS = "BUILD_READY_BLOCKED_EXTERNAL_ATTESTATION"
TEST_DOMAIN = "TEST_ONLY_EPHEMERAL"
TEST_ALGORITHM = "TEST_ONLY_ED25519"
TEST_KEY_PREFIX = "TEST-ONLY-"


class ContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}")
        self.code = code
        self.detail = detail


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise ContractError(code, detail)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON_OBJECT_REQUIRED", str(path))
    return value


def tree_manifest(root: Path) -> tuple[str, int]:
    lines: list[str] = []
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in entries:
        relative = path.relative_to(root).as_posix()
        mode = format(stat.S_IMODE(path.lstat().st_mode), "o")
        if path.is_dir():
            lines.append(f"d\t{relative}\t{mode}\t0\t-\n")
        else:
            require(path.is_file() and not path.is_symlink(), "TREE_NON_REGULAR_ENTRY", str(path))
            lines.append(f"f\t{relative}\t{mode}\t{path.stat().st_size}\t{sha256_file(path)}\n")
    return sha256_bytes("".join(lines).encode("utf-8")), len(entries)


def expect_error(case_id: str, code: str, operation: Callable[[], Any], rejected: list[str]) -> None:
    try:
        operation()
    except Exception as error:
        observed = getattr(error, "code", None)
        require(observed == code, "NEGATIVE_CASE_WRONG_CODE", f"{case_id}:{observed}")
        rejected.append(case_id)
    else:
        raise ContractError("NEGATIVE_CASE_ACCEPTED", case_id)


def expect_schema_rejected(case_id: str, schema: dict[str, Any], value: Any, rejected: list[str]) -> None:
    try:
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.ValidationError:
        rejected.append(case_id)
    else:
        raise ContractError("NEGATIVE_SCHEMA_CASE_ACCEPTED", case_id)


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(key)
            keys.update(nested_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(nested_keys(nested))
    return keys


FORBIDDEN_LOCAL_RECEIPT_KEYS = {
    "events_jsonl_path",
    "framework_raw_stream",
    "local_key_path",
    "mutable_log_path",
    "parent_pid",
    "private_key",
    "private_key_pem",
    "process_ancestry",
    "public_key",
    "public_key_pem",
    "raw_stream_path",
    "scheduler_prompt",
    "session_id",
    "thread_id",
    "verification_key",
}


def reject_local_bypass(receipt: dict[str, Any]) -> None:
    overlap = sorted(nested_keys(receipt) & FORBIDDEN_LOCAL_RECEIPT_KEYS)
    require(not overlap, "LOCAL_CORRELATION_OR_EMBEDDED_KEY_REJECTED", ",".join(overlap))


def verify_production_receipt_fail_closed(receipt: dict[str, Any], package: dict[str, Any]) -> None:
    reject_local_bypass(receipt)
    require(receipt.get("trust_domain") == "PRODUCTION_EXTERNAL_FRAMEWORK", "TEST_KEY_REJECTED", str(receipt.get("trust_domain")))
    issuer = receipt.get("issuer")
    require(isinstance(issuer, dict), "ISSUER_REQUIRED")
    require(issuer.get("service_ownership") == "FRAMEWORK_OWNED", "SELF_ISSUED_SIGNATURE_REJECTED")
    signature = receipt.get("signature")
    require(isinstance(signature, str) and bool(signature), "UNSIGNED_JSON_REJECTED")
    profile = package["production_verification_contract"]
    require(profile["status"] != "UNAVAILABLE_EXTERNAL_FRAMEWORK_CONTRACT_REQUIRED", "EXTERNAL_ATTESTATION_CONTRACT_UNAVAILABLE")
    raise ContractError("PRODUCTION_VERIFIER_MUST_BE_FRAMEWORK_OWNED")


def get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        require(isinstance(current, dict) and part in current, "BINDING_FIELD_MISSING", path)
        current = current[part]
    return current


def set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    current: Any = value
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = replacement


def base_payload(actor: str, layer: str, decision: str, nonce: str) -> dict[str, Any]:
    return {
        "action": {
            "action_id": "ace2:qk-gbfp8-base-v30:fixture-action",
            "decision": decision,
            "effect_class": "TEST_ONLY_NO_B0_EFFECT",
        },
        "package": {
            "package_file_sha256": "1" * 64,
            "package_tree_sha256": "2" * 64,
        },
        "candidate_tree": {
            "root_id": "test-only-candidate-tree",
            "tree_sha256": "3" * 64,
        },
        "interpreter": {
            "path": "/test-only/python",
            "sha256": "4" * 64,
            "version": "TEST-ONLY-3.13",
        },
        "invocation": {
            "argv": ["/test-only/python", "-B", "/test-only/entry.py"],
            "argv_sha256": "5" * 64,
            "cwd": "/test-only/cwd",
            "cwd_sha256": "6" * 64,
            "replacement_environment": {"LANG": "C", "TZ": "UTC"},
            "replacement_environment_sha256": "7" * 64,
        },
        "runtime": {
            "namespace_id": "test-only-runtime-never-created",
            "namespace_sha256": "8" * 64,
        },
        "event_frontier": {
            "source_kind": "FRAMEWORK_SIGNED_IMMUTABLE_EVENT_RECEIPTS",
            "frontier_sequence": 10,
            "frontier_sha256": "9" * 64,
            "first_terminal_sequence": None,
            "first_terminal_sha256": None,
        },
        "call": {"call_id": "test-only-call"},
        "mission": {"mission_id": "test-only-mission"},
        "identity": {"actor": actor, "layer": layer, "run_label": f"test-only-{layer}"},
        "freshness": {
            "issued_at_unix_ns": 1_000_000_000,
            "not_before_unix_ns": 900_000_000,
            "expires_at_unix_ns": 2_000_000_000,
        },
        "nonce": {"nonce": nonce, "replay_scope": "test-only-scope"},
    }


def signed_portion(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in receipt.items() if key != "signature"}


def sign_test_receipt(
    private_key: Ed25519PrivateKey,
    public_key_sha256: str,
    payload: dict[str, Any],
    *,
    key_id: str = "TEST-ONLY-ACTIVE-1",
    epoch: int = 1,
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "artifact_kind": "qk_gbfp8_head64_b0_v30_test_only_attestation_receipt",
        "trust_domain": TEST_DOMAIN,
        "issuer": {
            "service_ownership": "TEST_ONLY_IN_MEMORY",
            "algorithm": TEST_ALGORITHM,
            "key_id": key_id,
            "public_key_sha256": public_key_sha256,
            "rotation_epoch": epoch,
        },
        "canonicalization": {
            "identifier": "TEST_ONLY_SORTED_COMPACT_JSON",
            "version": "1",
            "signed_payload_sha256": sha256_bytes(compact_bytes(payload)),
        },
        "payload": deepcopy(payload),
    }
    receipt["signature"] = base64.b64encode(private_key.sign(compact_bytes(signed_portion(receipt)))).decode("ascii")
    return receipt


def verify_test_receipt(
    receipt: dict[str, Any],
    public_key: Ed25519PublicKey,
    expected_public_key_sha256: str,
    expected_payload: dict[str, Any],
    required_paths: list[str],
    replay_seen: set[tuple[str, str]],
    *,
    now_unix_ns: int = 1_500_000_000,
    active_epoch: int = 1,
    active_key_ids: set[str] | None = None,
    retired_key_ids: set[str] | None = None,
) -> None:
    active_key_ids = active_key_ids or {"TEST-ONLY-ACTIVE-1"}
    retired_key_ids = retired_key_ids or set()
    require(receipt.get("trust_domain") == TEST_DOMAIN, "TEST_TRUST_DOMAIN_MISMATCH")
    issuer = receipt["issuer"]
    require(issuer.get("algorithm") == TEST_ALGORITHM, "TEST_ALGORITHM_MISMATCH")
    require(str(issuer.get("key_id", "")).startswith(TEST_KEY_PREFIX), "TEST_KEY_ID_PREFIX_MISMATCH")
    require(issuer.get("key_id") in active_key_ids, "TEST_KEY_ID_NOT_ACTIVE")
    require(issuer.get("key_id") not in retired_key_ids, "TEST_KEY_RETIRED")
    require(issuer.get("rotation_epoch") == active_epoch, "TEST_ROTATION_EPOCH_MISMATCH")
    require(issuer.get("public_key_sha256") == expected_public_key_sha256, "TEST_PUBLIC_KEY_DIGEST_MISMATCH")
    payload = receipt["payload"]
    require(
        receipt["canonicalization"]["signed_payload_sha256"] == sha256_bytes(compact_bytes(payload)),
        "TEST_SIGNED_PAYLOAD_DIGEST_MISMATCH",
    )
    signature = base64.b64decode(receipt["signature"], validate=True)
    try:
        public_key.verify(signature, compact_bytes(signed_portion(receipt)))
    except Exception as error:
        raise ContractError("TEST_SIGNATURE_INVALID", type(error).__name__) from error
    freshness = payload["freshness"]
    require(freshness["not_before_unix_ns"] <= now_unix_ns, "TEST_RECEIPT_NOT_YET_VALID")
    require(freshness["issued_at_unix_ns"] <= now_unix_ns, "TEST_RECEIPT_FUTURE_ISSUED")
    require(now_unix_ns <= freshness["expires_at_unix_ns"], "TEST_RECEIPT_STALE")
    frontier = payload["event_frontier"]
    terminal_sequence = frontier["first_terminal_sequence"]
    terminal_sha256 = frontier["first_terminal_sha256"]
    require((terminal_sequence is None) == (terminal_sha256 is None), "TEST_FIRST_TERMINAL_PAIR_MISMATCH")
    if terminal_sequence is not None:
        require(frontier["frontier_sequence"] <= terminal_sequence, "TEST_RECEIPT_AFTER_FIRST_TERMINAL")
    for path in required_paths:
        require(get_path(payload, path) == get_path(expected_payload, path), "TEST_EXACT_BINDING_MISMATCH", path)
    replay_key = (payload["nonce"]["replay_scope"], payload["nonce"]["nonce"])
    require(replay_key not in replay_seen, "TEST_REPLAY_REJECTED")
    replay_seen.add(replay_key)


def resign_mutation(
    receipt: dict[str, Any], private_key: Ed25519PrivateKey, path: str, replacement: Any
) -> dict[str, Any]:
    changed = deepcopy(receipt)
    set_path(changed["payload"], path, replacement)
    changed["canonicalization"]["signed_payload_sha256"] = sha256_bytes(compact_bytes(changed["payload"]))
    changed["signature"] = base64.b64encode(private_key.sign(compact_bytes(signed_portion(changed)))).decode("ascii")
    return changed


def production_shape_receipt(actor: str, layer: str, decision: str, nonce: str) -> dict[str, Any]:
    payload = base_payload(actor, layer, decision, nonce)
    payload["action"]["effect_class"] = "FUTURE_EXACT_ACTION_EXTERNALLY_BOUND"
    return {
        "schema_version": 1,
        "artifact_kind": "qk_gbfp8_head64_b0_v30_framework_signed_attestation_receipt",
        "trust_domain": "PRODUCTION_EXTERNAL_FRAMEWORK",
        "issuer": {
            "service_ownership": "FRAMEWORK_OWNED",
            "private_key_access": "INACCESSIBLE_TO_ENGINEER_REVIEWER_AND_PROJECT_CODE",
            "signing_endpoint_access": "INACCESSIBLE_TO_ENGINEER_REVIEWER_AND_PROJECT_CODE",
            "algorithm": "EXTERNALLY_SUPPLIED_PLACEHOLDER_NOT_PINNED",
            "key_id": "EXTERNALLY_SUPPLIED_PLACEHOLDER_NOT_PINNED",
            "public_key_sha256": "a" * 64,
            "rotation_epoch": 0,
        },
        "canonicalization": {
            "identifier": "EXTERNALLY_SUPPLIED_PLACEHOLDER_NOT_PINNED",
            "version": "EXTERNALLY_SUPPLIED_PLACEHOLDER_NOT_PINNED",
            "signed_payload_sha256": sha256_bytes(compact_bytes(payload)),
        },
        "payload": payload,
        "signature": "QUJDREVGR0g=",
    }


def verify_package_inventory(package: dict[str, Any]) -> str:
    policy = package["package_file_policy"]
    observed_files = sorted(path.relative_to(ACTION_ROOT).as_posix() for path in ACTION_ROOT.rglob("*") if path.is_file())
    observed_directories = sorted(path.relative_to(ACTION_ROOT).as_posix() for path in ACTION_ROOT.rglob("*") if path.is_dir())
    require(observed_files == sorted(policy["allowed_relative_files"]), "PACKAGE_FILE_INVENTORY_MISMATCH", repr(observed_files))
    require(observed_directories == sorted(policy["allowed_relative_directories"]), "PACKAGE_DIRECTORY_INVENTORY_MISMATCH", repr(observed_directories))
    tree_sha256, _ = tree_manifest(ACTION_ROOT)
    return tree_sha256


def verify_predecessors(package: dict[str, Any]) -> dict[str, Any]:
    preservation = package["predecessor_preservation"]
    report_paths = {
        "additive_0001": PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0001/fresh-l2-sealed-0001/V29_CANDIDATE_REPORT.json",
        "additive_0002": PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0002/fresh-l2-sealed-0002/V29_CANDIDATE_REPORT.json",
        "additive_0003": PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0003/fresh-l2-sealed-0003/V29_CANDIDATE_REPORT.json",
        "additive_0004": PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0004/fresh-l2-sealed-0004/V29_CANDIDATE_REPORT.json",
    }
    verified: dict[str, Any] = {}
    for key in ("additive_0001", "additive_0002", "additive_0003", "additive_0004", "additive_0005"):
        expected = preservation[key]
        root = PROJECT_ROOT / expected["root"]
        require(root.is_dir(), "PREDECESSOR_ROOT_MISSING", key)
        package_file = root / "QK_GBFP8_HEAD64_B0_EXECUTION_BINDING_REPAIR_STATIC_V29_PACKAGE.json"
        require(sha256_file(package_file) == expected["package_file_sha256"], "PREDECESSOR_PACKAGE_DRIFT", key)
        observed_tree, observed_count = tree_manifest(root)
        require(observed_tree == expected["tree_manifest_sha256"], "PREDECESSOR_TREE_DRIFT", key)
        require(observed_count == expected["entry_count"], "PREDECESSOR_ENTRY_COUNT_DRIFT", key)
        if key != "additive_0005":
            require(expected["status"] == "SEALED_REJECTED_EVIDENCE_ONLY", "SEALED_REJECTED_STATUS_DRIFT", key)
            report_path = report_paths[key]
            require(sha256_file(report_path) == expected["candidate_report_file_sha256"], "PREDECESSOR_REPORT_DRIFT", key)
        else:
            require(
                expected["status"] == "TERMINATED_UNSEALED_DRAFT_NEVER_FINISH_SEAL_PROMOTE_ACCEPT_EXECUTE_OR_REUSE",
                "ADDITIVE_0005_STATUS_DRIFT",
            )
            for path in root.rglob("*"):
                mode = stat.S_IMODE(path.lstat().st_mode)
                if path.is_dir():
                    require(mode == 0o755, "ADDITIVE_0005_DIRECTORY_MODE_DRIFT", str(path))
                else:
                    require(mode == 0o644, "ADDITIVE_0005_FILE_MODE_DRIFT", str(path))
            require(not (root / "review/FRESH_L2_STATIC_ACCEPTANCE.json").exists(), "ADDITIVE_0005_ACCEPTANCE_MATERIALIZED")
            require(not (PROJECT_ROOT / expected["build_attempt_root_must_be_absent"]).exists(), "ADDITIVE_0005_BUILD_ATTEMPT_MATERIALIZED")
        verified[key] = {
            "package_file_sha256": expected["package_file_sha256"],
            "tree_manifest_sha256": observed_tree,
            "entry_count": observed_count,
            "status": expected["status"],
        }
    reviewer_report = PROJECT_ROOT / "build/v29-b0-execution-binding-repair-static-0004/fresh-l2-review-0004/V29_REVIEWER_CANDIDATE_REPORT.json"
    require(
        sha256_file(reviewer_report) == preservation["additive_0004"]["reviewer_report_file_sha256"],
        "ADDITIVE_0004_REVIEWER_REPORT_DRIFT",
    )
    return verified


def verify_zero_effects(package: dict[str, Any]) -> dict[str, Any]:
    boundary = package["claim_boundary"]
    expected_false = [
        "admission_created",
        "authority_created",
        "capsule_created",
        "runtime_namespace_created",
        "ledger_created",
        "result_created",
        "terminal_created",
    ]
    for field in expected_false:
        require(boundary[field] is False, "ZERO_EFFECT_BOOLEAN_DRIFT", field)
    for field in ("payload_open_count", "evaluator_call_count", "target_start_count", "b0_effect_count"):
        require(boundary[field] == 0, "ZERO_EFFECT_COUNTER_DRIFT", field)
    forbidden_effect_roots = [
        PROJECT_ROOT / "external/qk_gbfp8_head64_b0_external_attestation_prerequisite_v30",
        PROJECT_ROOT / "runtime/qk_gbfp8_head64_b0_external_attestation_prerequisite_v30",
        ACTION_ROOT / "review",
    ]
    for path in forbidden_effect_roots:
        require(not path.exists(), "V30_LIVE_EFFECT_PATH_MATERIALIZED", str(path))
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_surfaces = (
        "subprocess." + "run",
        "subprocess." + "Popen",
        "os." + "system(",
        "ex" + "ec(",
    )
    for forbidden_call in forbidden_surfaces:
        require(forbidden_call not in source, "V30_VERIFIER_LIVE_CALL_SURFACE", forbidden_call)
    key_files = [
        path.relative_to(ACTION_ROOT).as_posix()
        for path in ACTION_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".key", ".pem", ".der", ".crt"}
    ]
    require(not key_files, "PACKAGE_EMBEDDED_KEY_FILE", repr(key_files))
    return {
        "admission_created": False,
        "authority_created": False,
        "capsule_created": False,
        "runtime_namespace_created": False,
        "ledger_created": False,
        "result_created": False,
        "terminal_created": False,
        "official_payload_open_count": 0,
        "official_evaluator_call_count": 0,
        "official_target_start_count": 0,
        "b0_effect_count": 0,
    }


def run_static_verification() -> dict[str, Any]:
    package = load_json(PACKAGE_PATH)
    bundle_schema = load_json(BUNDLE_SCHEMA_PATH)
    framework_schema = load_json(FRAMEWORK_SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(bundle_schema)
    jsonschema.Draft202012Validator.check_schema(framework_schema)
    require(package["package_state"]["decision"] == BLOCKED_STATUS, "PACKAGE_STATUS_DRIFT")
    require(package["package_state"]["execution_package_acceptance_ready"] is False, "EXECUTION_READINESS_CLAIMED")
    require(package["production_trust_boundary"]["production_receipt_acceptance_enabled"] is False, "PRODUCTION_ACCEPTANCE_ENABLED")
    profile = package["production_verification_contract"]
    for field in (
        "algorithm",
        "key_id",
        "public_key_sha256",
        "canonicalization_identifier",
        "canonicalization_version",
        "signed_payload_field_order",
        "key_rotation_contract_sha256",
        "anti_replay_contract_sha256",
        "first_terminal_contract_sha256",
        "framework_verifier_endpoint",
        "immutable_receipt_source",
    ):
        require(profile[field] is None, "PRODUCTION_PROFILE_LOCALLY_PINNED", field)
    prerequisites = package["external_prerequisites"]
    require(len(prerequisites) == 1, "EXTERNAL_PREREQUISITE_CARDINALITY", str(len(prerequisites)))
    require(
        prerequisites[0]
        == "Argus/provider must expose a framework-owned signed-attestation endpoint or immutable signed event receipts and publish the production verification key/algorithm/key-rotation contract inaccessible to role code.",
        "EXTERNAL_PREREQUISITE_TEXT_DRIFT",
    )

    package_tree_sha256 = verify_package_inventory(package)
    predecessors = verify_predecessors(package)
    zero_effects = verify_zero_effects(package)

    manager = production_shape_receipt("MANAGER", "manager", "ADMIT_EXACT_ACTION_ONCE", "a" * 64)
    operator = production_shape_receipt("HUMAN_OPERATOR", "operator", "AUTHORIZE_EXACT_ACTION_ONCE", "b" * 64)
    reviewer = production_shape_receipt("FRESH_L2_REVIEWER", "reviewer", "REVIEW_EXACT_CANDIDATE_ONCE", "c" * 64)
    production_bundle = {
        "schema_version": 1,
        "artifact_kind": "qk_gbfp8_head64_b0_v30_framework_signed_attestation_bundle",
        "manager_receipt": manager,
        "operator_receipt": operator,
        "reviewer_receipt": reviewer,
    }
    jsonschema.Draft202012Validator(bundle_schema).validate(production_bundle)

    rejected: list[str] = []
    expect_error(
        "production_contract_unavailable_fail_closed",
        "EXTERNAL_ATTESTATION_CONTRACT_UNAVAILABLE",
        lambda: verify_production_receipt_fail_closed(manager, package),
        rejected,
    )
    unsigned = deepcopy(manager)
    unsigned["signature"] = ""
    expect_error("unsigned_json", "UNSIGNED_JSON_REJECTED", lambda: verify_production_receipt_fail_closed(unsigned, package), rejected)
    self_issued = deepcopy(manager)
    self_issued["issuer"]["service_ownership"] = "PROJECT_CODE"
    expect_error(
        "self_issued_signature",
        "SELF_ISSUED_SIGNATURE_REJECTED",
        lambda: verify_production_receipt_fail_closed(self_issued, package),
        rejected,
    )
    for case_id, key in (
        ("package_embedded_public_key", "public_key_pem"),
        ("mutable_log_correlation", "mutable_log_path"),
        ("events_jsonl_correlation", "events_jsonl_path"),
        ("process_ancestry_correlation", "process_ancestry"),
        ("thread_id_correlation", "thread_id"),
        ("session_id_correlation", "session_id"),
        ("scheduler_prompt_correlation", "scheduler_prompt"),
        ("framework_raw_stream_correlation", "framework_raw_stream"),
    ):
        changed = deepcopy(manager)
        changed["payload"][key] = "same-user-forgeable"
        expect_error(
            case_id,
            "LOCAL_CORRELATION_OR_EMBEDDED_KEY_REJECTED",
            lambda changed=changed: verify_production_receipt_fail_closed(changed, package),
            rejected,
        )
    copied_bundle = deepcopy(production_bundle)
    copied_bundle["operator_receipt"] = deepcopy(manager)
    expect_schema_rejected("copied_manager_receipt_as_operator", bundle_schema, copied_bundle, rejected)
    unsigned_schema = deepcopy(production_bundle)
    del unsigned_schema["reviewer_receipt"]["signature"]
    expect_schema_rejected("unsigned_schema_receipt", bundle_schema, unsigned_schema, rejected)
    embedded_schema = deepcopy(production_bundle)
    embedded_schema["manager_receipt"]["issuer"]["public_key_pem"] = "forbidden"
    expect_schema_rejected("embedded_key_schema_receipt", bundle_schema, embedded_schema, rejected)

    test_private = Ed25519PrivateKey.generate()
    test_public = test_private.public_key()
    test_public_raw = test_public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    test_public_sha256 = sha256_bytes(test_public_raw)
    expected_payload = base_payload("MANAGER", "manager", "ADMIT_EXACT_ACTION_ONCE", "d" * 64)
    test_receipt = sign_test_receipt(test_private, test_public_sha256, expected_payload)
    replay_seen: set[tuple[str, str]] = set()
    required_paths = package["canonical_receipt_contract"]["required_semantic_binding_paths"]
    verify_test_receipt(
        test_receipt,
        test_public,
        test_public_sha256,
        expected_payload,
        required_paths,
        replay_seen,
    )
    expect_error(
        "test_only_key_on_production_path",
        "TEST_KEY_REJECTED",
        lambda: verify_production_receipt_fail_closed(test_receipt, package),
        rejected,
    )
    expect_error(
        "test_only_replay",
        "TEST_REPLAY_REJECTED",
        lambda: verify_test_receipt(
            test_receipt,
            test_public,
            test_public_sha256,
            expected_payload,
            required_paths,
            replay_seen,
        ),
        rejected,
    )

    mutation_cases = {
        "wrong_action": ("action.action_id", "wrong-action"),
        "wrong_package": ("package.package_file_sha256", "e" * 64),
        "wrong_candidate_tree": ("candidate_tree.tree_sha256", "e" * 64),
        "wrong_interpreter": ("interpreter.sha256", "e" * 64),
        "wrong_argv": ("invocation.argv", ["/wrong"]),
        "wrong_cwd": ("invocation.cwd", "/wrong"),
        "wrong_replacement_environment": ("invocation.replacement_environment", {"WRONG": "1"}),
        "wrong_runtime": ("runtime.namespace_id", "wrong-runtime"),
        "wrong_event_frontier": ("event_frontier.frontier_sha256", "e" * 64),
        "wrong_call": ("call.call_id", "wrong-call"),
        "wrong_mission": ("mission.mission_id", "wrong-mission"),
        "wrong_actor": ("identity.actor", "HUMAN_OPERATOR"),
        "wrong_layer": ("identity.layer", "operator"),
        "wrong_run_label": ("identity.run_label", "wrong-label"),
        "wrong_payload_effect_class": ("action.effect_class", "WRONG"),
    }
    for case_id, (path, replacement) in mutation_cases.items():
        changed = resign_mutation(test_receipt, test_private, path, replacement)
        expect_error(
            case_id,
            "TEST_EXACT_BINDING_MISMATCH",
            lambda changed=changed: verify_test_receipt(
                changed,
                test_public,
                test_public_sha256,
                expected_payload,
                required_paths,
                set(),
            ),
            rejected,
        )

    stale = resign_mutation(test_receipt, test_private, "freshness.expires_at_unix_ns", 1_400_000_000)
    expect_error(
        "stale_receipt",
        "TEST_RECEIPT_STALE",
        lambda: verify_test_receipt(stale, test_public, test_public_sha256, expected_payload, required_paths, set()),
        rejected,
    )
    future_issued = resign_mutation(test_receipt, test_private, "freshness.issued_at_unix_ns", 1_600_000_000)
    expect_error(
        "future_issued_receipt",
        "TEST_RECEIPT_FUTURE_ISSUED",
        lambda: verify_test_receipt(future_issued, test_public, test_public_sha256, expected_payload, required_paths, set()),
        rejected,
    )
    after_terminal = resign_mutation(test_receipt, test_private, "event_frontier.first_terminal_sequence", 9)
    after_terminal = resign_mutation(after_terminal, test_private, "event_frontier.first_terminal_sha256", "f" * 64)
    expect_error(
        "receipt_after_first_terminal",
        "TEST_RECEIPT_AFTER_FIRST_TERMINAL",
        lambda: verify_test_receipt(after_terminal, test_public, test_public_sha256, expected_payload, required_paths, set()),
        rejected,
    )
    expect_error(
        "wrong_algorithm",
        "TEST_ALGORITHM_MISMATCH",
        lambda: verify_test_receipt(
            {**test_receipt, "issuer": {**test_receipt["issuer"], "algorithm": "WRONG"}},
            test_public,
            test_public_sha256,
            expected_payload,
            required_paths,
            set(),
        ),
        rejected,
    )
    expect_error(
        "wrong_key_id",
        "TEST_KEY_ID_NOT_ACTIVE",
        lambda: verify_test_receipt(
            {**test_receipt, "issuer": {**test_receipt["issuer"], "key_id": "TEST-ONLY-WRONG"}},
            test_public,
            test_public_sha256,
            expected_payload,
            required_paths,
            set(),
        ),
        rejected,
    )
    expect_error(
        "retired_key",
        "TEST_KEY_RETIRED",
        lambda: verify_test_receipt(
            test_receipt,
            test_public,
            test_public_sha256,
            expected_payload,
            required_paths,
            set(),
            active_key_ids={"TEST-ONLY-ACTIVE-1", "TEST-ONLY-NEXT"},
            retired_key_ids={"TEST-ONLY-ACTIVE-1"},
        ),
        rejected,
    )
    expect_error(
        "wrong_rotation_epoch",
        "TEST_ROTATION_EPOCH_MISMATCH",
        lambda: verify_test_receipt(
            test_receipt,
            test_public,
            test_public_sha256,
            expected_payload,
            required_paths,
            set(),
            active_epoch=2,
        ),
        rejected,
    )

    require(len(rejected) >= 35, "NEGATIVE_CASE_COUNT_TOO_SMALL", str(len(rejected)))
    require(len(set(rejected)) == len(rejected), "NEGATIVE_CASE_DUPLICATE")
    return {
        "schema_version": 1,
        "artifact_kind": REPORT_KIND,
        "action_id": package["action_identity"]["action_id"],
        "mission_id": package["action_identity"]["mission_id"],
        "status": BLOCKED_STATUS,
        "claim_boundary": package["claim_boundary"]["claim"],
        "package_file_sha256": sha256_file(PACKAGE_PATH),
        "package_tree_manifest_sha256": package_tree_sha256,
        "schema_checks": {
            "bundle_schema_valid": True,
            "external_framework_contract_schema_valid": True,
            "production_shape_bundle_schema_valid": True,
            "external_framework_contract_instantiated": False,
        },
        "predecessor_preservation": predecessors,
        "production_verification": {
            "external_contract_available": False,
            "algorithm_pinned": False,
            "key_id_pinned": False,
            "public_key_digest_pinned": False,
            "canonicalization_pinned": False,
            "rotation_contract_pinned": False,
            "production_receipt_verifications": 0,
            "production_receipts_accepted": 0,
            "fail_closed_case": "production_contract_unavailable_fail_closed",
        },
        "test_only_verification": {
            "protected_data_free": True,
            "algorithm": TEST_ALGORITHM,
            "ephemeral_private_key_in_memory_only": True,
            "key_persisted": False,
            "positive_receipts_verified": 1,
            "production_path_accepts_test_receipts": False,
        },
        "negative_cases": sorted(rejected),
        "negative_case_count": len(rejected),
        "zero_effects": zero_effects,
        "external_prerequisites": prerequisites,
        "review_decision_requested": BLOCKED_STATUS,
        "accept_static_package_requested": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_static_verification()
    if args.report is not None:
        write_report(args.report, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
