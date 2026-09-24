#!/usr/bin/env python3
"""Verify three framework-signed receipts for one frozen Stage 1 action."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
NONCE_RE = SHA256_RE
SIGNED_FIELDS = [
    "schema_version",
    "artifact_kind",
    "trust_domain",
    "issuer",
    "canonicalization",
    "payload",
]
ROLE_CONTRACTS = {
    "manager": {
        "bundle_field": "manager_receipt",
        "actor": "MANAGER",
        "layer": "manager",
        "decision": "ADMIT_EXACT_ACTION_ONCE",
    },
    "operator": {
        "bundle_field": "operator_receipt",
        "actor": "HUMAN_OPERATOR",
        "layer": "operator",
        "decision": "AUTHORIZE_EXACT_ACTION_ONCE",
    },
    "fresh_l2": {
        "bundle_field": "fresh_l2_receipt",
        "actor": "FRESH_L2_REVIEWER",
        "layer": "reviewer",
        "decision": "REVIEW_EXACT_ACTION_ONCE",
    },
}
COMMON_PAYLOAD_PATHS = (
    "action.action_id",
    "action.effect_class",
    "package",
    "candidate_tree",
    "interpreter",
    "invocation",
    "runtime",
    "event_frontier",
    "call",
    "mission",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_with_exact_keys(
    value: Any, expected: set[str], context: str
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{context} must be an object")
    require(set(value) == expected, f"{context} fields differ from contract")
    return value


def hash_without(value: dict[str, Any], field: str) -> str:
    return sha256_bytes(canonical_bytes({key: item for key, item in value.items() if key != field}))


def get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        require(isinstance(current, dict) and part in current, f"missing payload path: {path}")
        current = current[part]
    return current


def decode_base64(value: Any, context: str) -> bytes:
    require(isinstance(value, str), f"{context} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise VerificationError(f"{context} is not canonical base64") from error


def verify_contract(contract: dict[str, Any]) -> Ed25519PublicKey:
    object_with_exact_keys(
        contract,
        {
            "schema_version",
            "artifact_kind",
            "signature_profile",
            "key_rotation",
            "replay",
            "freshness",
            "contract_sha256",
        },
        "verification contract",
    )
    require(contract["schema_version"] == 1, "verification contract schema version")
    require(
        contract["artifact_kind"] == "ace2_stage1_attestation_verification_contract",
        "verification contract artifact kind",
    )
    require(
        SHA256_RE.fullmatch(str(contract["contract_sha256"])) is not None
        and contract["contract_sha256"] == hash_without(contract, "contract_sha256"),
        "verification contract hash",
    )

    profile = object_with_exact_keys(
        contract["signature_profile"],
        {
            "algorithm",
            "key_id",
            "verification_key_base64",
            "verification_key_sha256",
            "canonicalization_identifier",
            "canonicalization_version",
            "signed_fields",
        },
        "signature profile",
    )
    require(profile["algorithm"] == "Ed25519", "unsupported signature algorithm")
    require(isinstance(profile["key_id"], str) and profile["key_id"], "empty key id")
    require(
        not profile["key_id"].startswith("TEST-ONLY-"),
        "test-only verification key rejected",
    )
    require(
        profile["canonicalization_identifier"]
        == "ACE2_SORTED_COMPACT_JSON_ASCII_LF"
        and profile["canonicalization_version"] == "1",
        "unsupported canonicalization",
    )
    require(profile["signed_fields"] == SIGNED_FIELDS, "signed field contract")
    public_key_raw = decode_base64(
        profile["verification_key_base64"], "verification key"
    )
    require(len(public_key_raw) == 32, "Ed25519 verification key length")
    require(
        SHA256_RE.fullmatch(str(profile["verification_key_sha256"])) is not None
        and sha256_bytes(public_key_raw) == profile["verification_key_sha256"],
        "verification key hash",
    )

    rotation = object_with_exact_keys(
        contract["key_rotation"],
        {
            "active_epoch",
            "active_key_ids",
            "retired_key_ids",
            "revoked_key_ids",
            "contract_sha256",
        },
        "key rotation contract",
    )
    require(
        rotation["contract_sha256"] == hash_without(rotation, "contract_sha256"),
        "key rotation contract hash",
    )
    active = rotation["active_key_ids"]
    retired = rotation["retired_key_ids"]
    revoked = rotation["revoked_key_ids"]
    require(
        isinstance(rotation["active_epoch"], int) and rotation["active_epoch"] >= 0,
        "active rotation epoch",
    )
    require(
        all(
            isinstance(items, list)
            and all(isinstance(item, str) and item for item in items)
            for items in (active, retired, revoked)
        ),
        "rotation key-id sets",
    )
    require(
        len(active) == len(set(active))
        and len(retired) == len(set(retired))
        and len(revoked) == len(set(revoked)),
        "duplicate rotation key id",
    )
    require(
        not (set(active) & (set(retired) | set(revoked))),
        "active key is retired or revoked",
    )
    require(profile["key_id"] in active, "verification key is not active")

    replay = object_with_exact_keys(
        contract["replay"],
        {
            "authority",
            "nonce_uniqueness_scope",
            "consumed_nonce_sha256",
            "snapshot_sequence",
            "snapshot_sha256",
            "contract_sha256",
        },
        "replay contract",
    )
    require(
        replay["contract_sha256"] == hash_without(replay, "contract_sha256"),
        "replay contract hash",
    )
    require(replay["authority"] == "FRAMEWORK_OWNED", "replay authority")
    require(
        isinstance(replay["nonce_uniqueness_scope"], str)
        and replay["nonce_uniqueness_scope"],
        "replay scope",
    )
    consumed = replay["consumed_nonce_sha256"]
    require(
        isinstance(consumed, list)
        and len(consumed) == len(set(consumed))
        and all(SHA256_RE.fullmatch(str(item)) is not None for item in consumed),
        "consumed nonce digest set",
    )
    require(
        isinstance(replay["snapshot_sequence"], int)
        and replay["snapshot_sequence"] >= 0
        and SHA256_RE.fullmatch(str(replay["snapshot_sha256"])) is not None,
        "replay snapshot identity",
    )

    freshness = object_with_exact_keys(
        contract["freshness"], {"max_future_skew_ns"}, "freshness contract"
    )
    require(
        isinstance(freshness["max_future_skew_ns"], int)
        and freshness["max_future_skew_ns"] >= 0,
        "future skew contract",
    )
    return Ed25519PublicKey.from_public_bytes(public_key_raw)


def verify_frozen_action(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    object_with_exact_keys(
        frozen,
        {
            "schema_version",
            "artifact_kind",
            "action_id",
            "mission_id",
            "payloads",
            "action_sha256",
        },
        "frozen action",
    )
    require(frozen["schema_version"] == 1, "frozen action schema version")
    require(
        frozen["artifact_kind"] == "ace2_stage1_frozen_action",
        "frozen action artifact kind",
    )
    require(
        isinstance(frozen["action_id"], str)
        and frozen["action_id"]
        and isinstance(frozen["mission_id"], str)
        and frozen["mission_id"],
        "frozen action identity",
    )
    require(
        frozen["action_sha256"] == hash_without(frozen, "action_sha256"),
        "frozen action hash",
    )
    payloads = object_with_exact_keys(
        frozen["payloads"], set(ROLE_CONTRACTS), "frozen action payloads"
    )
    for role, payload in payloads.items():
        require(isinstance(payload, dict), f"{role} frozen payload")
        role_contract = ROLE_CONTRACTS[role]
        require(
            get_path(payload, "action.action_id") == frozen["action_id"],
            f"{role} action id",
        )
        require(
            get_path(payload, "mission.mission_id") == frozen["mission_id"],
            f"{role} mission id",
        )
        require(
            get_path(payload, "action.decision") == role_contract["decision"],
            f"{role} decision",
        )
        require(
            get_path(payload, "identity.actor") == role_contract["actor"]
            and get_path(payload, "identity.layer") == role_contract["layer"],
            f"{role} identity",
        )
    manager = payloads["manager"]
    for role, payload in payloads.items():
        for path in COMMON_PAYLOAD_PATHS:
            require(
                get_path(payload, path) == get_path(manager, path),
                f"{role} differs at shared action binding {path}",
            )
    return payloads


def verify_receipt(
    receipt: dict[str, Any],
    expected_payload: dict[str, Any],
    role: str,
    contract: dict[str, Any],
    public_key: Ed25519PublicKey,
    now_unix_ns: int,
    observed_replay_keys: set[tuple[str, str]],
) -> str:
    object_with_exact_keys(
        receipt,
        {
            "schema_version",
            "artifact_kind",
            "trust_domain",
            "issuer",
            "canonicalization",
            "payload",
            "signature",
        },
        f"{role} receipt",
    )
    require(receipt["schema_version"] == 1, f"{role} receipt schema version")
    require(
        receipt["artifact_kind"] == "ace2_stage1_framework_signed_receipt",
        f"{role} receipt artifact kind",
    )
    require(
        receipt["trust_domain"] == "PRODUCTION_EXTERNAL_FRAMEWORK",
        f"{role} receipt trust domain",
    )
    profile = contract["signature_profile"]
    rotation = contract["key_rotation"]
    issuer = object_with_exact_keys(
        receipt["issuer"],
        {
            "service_ownership",
            "private_key_access",
            "signing_endpoint_access",
            "algorithm",
            "key_id",
            "public_key_sha256",
            "rotation_epoch",
        },
        f"{role} issuer",
    )
    require(issuer["service_ownership"] == "FRAMEWORK_OWNED", f"{role} issuer ownership")
    require(
        issuer["private_key_access"]
        == "INACCESSIBLE_TO_ENGINEER_REVIEWER_AND_PROJECT_CODE"
        and issuer["signing_endpoint_access"]
        == "INACCESSIBLE_TO_ENGINEER_REVIEWER_AND_PROJECT_CODE",
        f"{role} signer authority exposure",
    )
    require(
        issuer["algorithm"] == profile["algorithm"]
        and issuer["key_id"] == profile["key_id"]
        and issuer["public_key_sha256"] == profile["verification_key_sha256"],
        f"{role} signature profile mismatch",
    )
    require(
        issuer["rotation_epoch"] == rotation["active_epoch"]
        and issuer["key_id"] in rotation["active_key_ids"]
        and issuer["key_id"] not in rotation["retired_key_ids"]
        and issuer["key_id"] not in rotation["revoked_key_ids"],
        f"{role} inactive signing key",
    )

    canonicalization = object_with_exact_keys(
        receipt["canonicalization"],
        {"identifier", "version", "signed_payload_sha256"},
        f"{role} canonicalization",
    )
    require(
        canonicalization["identifier"] == profile["canonicalization_identifier"]
        and canonicalization["version"] == profile["canonicalization_version"],
        f"{role} canonicalization profile mismatch",
    )
    payload = receipt["payload"]
    require(payload == expected_payload, f"{role} receipt does not match frozen payload")
    payload_sha256 = sha256_bytes(canonical_bytes(payload))
    require(
        canonicalization["signed_payload_sha256"] == payload_sha256,
        f"{role} canonical payload hash",
    )

    freshness = get_path(payload, "freshness")
    require(isinstance(freshness, dict), f"{role} freshness")
    issued = freshness.get("issued_at_unix_ns")
    not_before = freshness.get("not_before_unix_ns")
    expires = freshness.get("expires_at_unix_ns")
    require(
        all(isinstance(item, int) for item in (issued, not_before, expires)),
        f"{role} freshness values",
    )
    require(not_before <= issued <= expires, f"{role} freshness ordering")
    require(not_before <= now_unix_ns <= expires, f"{role} receipt outside validity window")
    require(
        issued <= now_unix_ns + contract["freshness"]["max_future_skew_ns"],
        f"{role} receipt issued too far in future",
    )

    nonce_record = get_path(payload, "nonce")
    require(isinstance(nonce_record, dict), f"{role} nonce")
    nonce = nonce_record.get("nonce")
    replay_scope = nonce_record.get("replay_scope")
    require(NONCE_RE.fullmatch(str(nonce)) is not None, f"{role} nonce syntax")
    require(
        replay_scope == contract["replay"]["nonce_uniqueness_scope"],
        f"{role} replay scope",
    )
    replay_key = (replay_scope, nonce)
    require(replay_key not in observed_replay_keys, f"{role} duplicate bundle nonce")
    replay_digest = sha256_bytes(f"{replay_scope}\0{nonce}".encode("utf-8"))
    require(
        replay_digest not in contract["replay"]["consumed_nonce_sha256"],
        f"{role} nonce already consumed",
    )

    frontier = get_path(payload, "event_frontier")
    sequence = frontier.get("frontier_sequence")
    terminal_sequence = frontier.get("first_terminal_sequence")
    terminal_sha256 = frontier.get("first_terminal_sha256")
    require(isinstance(sequence, int) and sequence >= 0, f"{role} event frontier")
    require(
        (terminal_sequence is None) == (terminal_sha256 is None),
        f"{role} first-terminal pair",
    )
    if terminal_sequence is not None:
        require(
            isinstance(terminal_sequence, int)
            and terminal_sequence >= sequence
            and SHA256_RE.fullmatch(str(terminal_sha256)) is not None,
            f"{role} receipt is after first terminal",
        )

    signature = object_with_exact_keys(
        receipt["signature"], {"encoding", "value"}, f"{role} signature"
    )
    require(signature["encoding"] == "base64", f"{role} signature encoding")
    signature_raw = decode_base64(signature["value"], f"{role} signature")
    require(len(signature_raw) == 64, f"{role} Ed25519 signature length")
    signed_portion = {field: receipt[field] for field in profile["signed_fields"]}
    try:
        public_key.verify(signature_raw, canonical_bytes(signed_portion))
    except InvalidSignature as error:
        raise VerificationError(f"{role} signature invalid") from error
    observed_replay_keys.add(replay_key)
    return payload_sha256


def verify_bundle(
    contract: dict[str, Any],
    frozen: dict[str, Any],
    bundle: dict[str, Any],
    now_unix_ns: int,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    require(
        SHA256_RE.fullmatch(expected_contract_sha256) is not None
        and contract.get("contract_sha256") == expected_contract_sha256,
        "out-of-band verification contract hash",
    )
    public_key = verify_contract(contract)
    expected_payloads = verify_frozen_action(frozen)
    object_with_exact_keys(
        bundle,
        {
            "schema_version",
            "artifact_kind",
            "contract_sha256",
            "action_sha256",
            "manager_receipt",
            "operator_receipt",
            "fresh_l2_receipt",
        },
        "receipt bundle",
    )
    require(bundle["schema_version"] == 1, "receipt bundle schema version")
    require(
        bundle["artifact_kind"] == "ace2_stage1_framework_signed_receipt_bundle",
        "receipt bundle artifact kind",
    )
    require(
        bundle["contract_sha256"] == contract["contract_sha256"],
        "bundle verification contract binding",
    )
    require(
        bundle["action_sha256"] == frozen["action_sha256"],
        "bundle frozen action binding",
    )
    observed_replay_keys: set[tuple[str, str]] = set()
    receipt_hashes: dict[str, str] = {}
    for role, role_contract in ROLE_CONTRACTS.items():
        receipt_hashes[role] = verify_receipt(
            bundle[role_contract["bundle_field"]],
            expected_payloads[role],
            role,
            contract,
            public_key,
            now_unix_ns,
            observed_replay_keys,
        )
    return {
        "schema_version": 1,
        "artifact_kind": "ace2_stage1_attestation_verification_result",
        "status": "PASS_STAGE1_THREE_RECEIPTS_VERIFIED_PRE_EXECUTION",
        "contract_sha256": contract["contract_sha256"],
        "action_sha256": frozen["action_sha256"],
        "action_id": frozen["action_id"],
        "mission_id": frozen["mission_id"],
        "algorithm": contract["signature_profile"]["algorithm"],
        "key_id": contract["signature_profile"]["key_id"],
        "rotation_epoch": contract["key_rotation"]["active_epoch"],
        "replay_snapshot_sequence": contract["replay"]["snapshot_sequence"],
        "verified_roles": list(ROLE_CONTRACTS),
        "receipt_payload_sha256": receipt_hashes,
        "protected_effects_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--frozen-action", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--now-unix-ns", type=int)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()
    result = verify_bundle(
        load_json(args.contract),
        load_json(args.frozen_action),
        load_json(args.bundle),
        args.now_unix_ns if args.now_unix_ns is not None else time.time_ns(),
        args.expected_contract_sha256,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
