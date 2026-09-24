from __future__ import annotations

import base64
import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/verify_stage1_attestation_bundle.py"
SPEC = importlib.util.spec_from_file_location("verify_stage1_attestation_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def seal_hash(value: dict[str, Any], field: str) -> None:
    value[field] = verifier.hash_without(value, field)


def payload(role: str, nonce: str) -> dict[str, Any]:
    identities = {
        "manager": ("MANAGER", "manager", "ADMIT_EXACT_ACTION_ONCE"),
        "operator": ("HUMAN_OPERATOR", "operator", "AUTHORIZE_EXACT_ACTION_ONCE"),
        "fresh_l2": ("FRESH_L2_REVIEWER", "reviewer", "REVIEW_EXACT_ACTION_ONCE"),
    }
    actor, layer, decision = identities[role]
    return {
        "action": {
            "action_id": "ace2:stage1:test-only-frozen-action",
            "decision": decision,
            "effect_class": "TEST_ONLY_NO_PROTECTED_EFFECT",
        },
        "package": {"package_sha256": "1" * 64},
        "candidate_tree": {"tree_sha256": "2" * 64},
        "interpreter": {"path": "/test/python", "sha256": "3" * 64},
        "invocation": {
            "argv": ["/test/python", "/test/action.py"],
            "cwd": "/test/action",
            "environment": {"LANG": "C"},
        },
        "runtime": {"namespace_id": "not-created"},
        "event_frontier": {
            "frontier_sequence": 40,
            "frontier_sha256": "4" * 64,
            "first_terminal_sequence": None,
            "first_terminal_sha256": None,
        },
        "call": {"call_id": "test-call"},
        "mission": {"mission_id": "test-mission"},
        "identity": {"actor": actor, "layer": layer, "run_label": f"test-{role}"},
        "freshness": {
            "issued_at_unix_ns": 1_000,
            "not_before_unix_ns": 900,
            "expires_at_unix_ns": 2_000,
        },
        "nonce": {
            "nonce": nonce,
            "replay_scope": "ace2-stage1-test-scope",
        },
    }


def fixture() -> tuple[
    Ed25519PrivateKey, dict[str, Any], dict[str, Any], dict[str, Any]
]:
    private_key = Ed25519PrivateKey.generate()
    public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    contract = {
        "schema_version": 1,
        "artifact_kind": "ace2_stage1_attestation_verification_contract",
        "signature_profile": {
            "algorithm": "Ed25519",
            "key_id": "FRAMEWORK-FIXTURE-ACTIVE-1",
            "verification_key_base64": base64.b64encode(public_raw).decode("ascii"),
            "verification_key_sha256": verifier.sha256_bytes(public_raw),
            "canonicalization_identifier": "ACE2_SORTED_COMPACT_JSON_ASCII_LF",
            "canonicalization_version": "1",
            "signed_fields": verifier.SIGNED_FIELDS,
        },
        "key_rotation": {
            "active_epoch": 7,
            "active_key_ids": ["FRAMEWORK-FIXTURE-ACTIVE-1"],
            "retired_key_ids": [],
            "revoked_key_ids": [],
            "contract_sha256": "",
        },
        "replay": {
            "authority": "FRAMEWORK_OWNED",
            "nonce_uniqueness_scope": "ace2-stage1-test-scope",
            "consumed_nonce_sha256": [],
            "snapshot_sequence": 39,
            "snapshot_sha256": "5" * 64,
            "contract_sha256": "",
        },
        "freshness": {"max_future_skew_ns": 0},
        "contract_sha256": "",
    }
    seal_hash(contract["key_rotation"], "contract_sha256")
    seal_hash(contract["replay"], "contract_sha256")
    seal_hash(contract, "contract_sha256")

    frozen = {
        "schema_version": 1,
        "artifact_kind": "ace2_stage1_frozen_action",
        "action_id": "ace2:stage1:test-only-frozen-action",
        "mission_id": "test-mission",
        "payloads": {
            "manager": payload("manager", "a" * 64),
            "operator": payload("operator", "b" * 64),
            "fresh_l2": payload("fresh_l2", "c" * 64),
        },
        "action_sha256": "",
    }
    seal_hash(frozen, "action_sha256")

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "ace2_stage1_framework_signed_receipt_bundle",
        "contract_sha256": contract["contract_sha256"],
        "action_sha256": frozen["action_sha256"],
    }
    for role, role_contract in verifier.ROLE_CONTRACTS.items():
        receipt = {
            "schema_version": 1,
            "artifact_kind": "ace2_stage1_framework_signed_receipt",
            "trust_domain": "PRODUCTION_EXTERNAL_FRAMEWORK",
            "issuer": {
                "service_ownership": "FRAMEWORK_OWNED",
                "private_key_access": "INACCESSIBLE_TO_ENGINEER_REVIEWER_AND_PROJECT_CODE",
                "signing_endpoint_access": "INACCESSIBLE_TO_ENGINEER_REVIEWER_AND_PROJECT_CODE",
                "algorithm": "Ed25519",
                "key_id": "FRAMEWORK-FIXTURE-ACTIVE-1",
                "public_key_sha256": contract["signature_profile"][
                    "verification_key_sha256"
                ],
                "rotation_epoch": 7,
            },
            "canonicalization": {
                "identifier": "ACE2_SORTED_COMPACT_JSON_ASCII_LF",
                "version": "1",
                "signed_payload_sha256": verifier.sha256_bytes(
                    verifier.canonical_bytes(frozen["payloads"][role])
                ),
            },
            "payload": deepcopy(frozen["payloads"][role]),
        }
        signed = {field: receipt[field] for field in verifier.SIGNED_FIELDS}
        receipt["signature"] = {
            "encoding": "base64",
            "value": base64.b64encode(
                private_key.sign(verifier.canonical_bytes(signed))
            ).decode("ascii"),
        }
        bundle[role_contract["bundle_field"]] = receipt
    return private_key, contract, frozen, bundle


def test_accepts_all_three_receipts_for_one_frozen_action() -> None:
    _, contract, frozen, bundle = fixture()
    result = verifier.verify_bundle(
        contract, frozen, bundle, 1_500, contract["contract_sha256"]
    )
    assert result["status"] == "PASS_STAGE1_THREE_RECEIPTS_VERIFIED_PRE_EXECUTION"
    assert result["verified_roles"] == ["manager", "operator", "fresh_l2"]
    assert result["protected_effects_started"] is False


def test_rejects_role_substitution() -> None:
    _, contract, frozen, bundle = fixture()
    bundle["operator_receipt"] = deepcopy(bundle["manager_receipt"])
    with pytest.raises(verifier.VerificationError, match="operator receipt"):
        verifier.verify_bundle(
            contract, frozen, bundle, 1_500, contract["contract_sha256"]
        )


def test_rejects_invalid_signature() -> None:
    _, contract, frozen, bundle = fixture()
    bundle["fresh_l2_receipt"]["signature"]["value"] = base64.b64encode(
        b"\x00" * 64
    ).decode("ascii")
    with pytest.raises(verifier.VerificationError, match="fresh_l2 signature invalid"):
        verifier.verify_bundle(
            contract, frozen, bundle, 1_500, contract["contract_sha256"]
        )


def test_rejects_replayed_nonce() -> None:
    _, contract, frozen, bundle = fixture()
    nonce = frozen["payloads"]["operator"]["nonce"]
    replay_digest = verifier.sha256_bytes(
        f"{nonce['replay_scope']}\0{nonce['nonce']}".encode("utf-8")
    )
    contract["replay"]["consumed_nonce_sha256"] = [replay_digest]
    seal_hash(contract["replay"], "contract_sha256")
    seal_hash(contract, "contract_sha256")
    bundle["contract_sha256"] = contract["contract_sha256"]
    with pytest.raises(verifier.VerificationError, match="operator nonce already consumed"):
        verifier.verify_bundle(
            contract, frozen, bundle, 1_500, contract["contract_sha256"]
        )


def test_rejects_retired_verification_key() -> None:
    _, contract, frozen, bundle = fixture()
    key_id = contract["signature_profile"]["key_id"]
    contract["key_rotation"]["active_key_ids"] = []
    contract["key_rotation"]["retired_key_ids"] = [key_id]
    seal_hash(contract["key_rotation"], "contract_sha256")
    seal_hash(contract, "contract_sha256")
    bundle["contract_sha256"] = contract["contract_sha256"]
    with pytest.raises(verifier.VerificationError, match="verification key is not active"):
        verifier.verify_bundle(
            contract, frozen, bundle, 1_500, contract["contract_sha256"]
        )


def test_rejects_unpinned_contract() -> None:
    _, contract, frozen, bundle = fixture()
    with pytest.raises(
        verifier.VerificationError, match="out-of-band verification contract hash"
    ):
        verifier.verify_bundle(contract, frozen, bundle, 1_500, "0" * 64)
