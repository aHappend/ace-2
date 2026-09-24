#!/usr/bin/env python3
"""NO_EXECUTION_AUTHORITY static verifier for QK_BFP8_E16_HEAD64_V1."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


AUTHORITY = "NO_EXECUTION_AUTHORITY"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "design/W4A8_C02_SATURATION_FREE_JOINT_QK_REPRESENTATION_PREREQUISITE_V1.json"
CONTRACT_SHA_PATH = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + ".sha256")
READINESS_PATH = ROOT / "reference/QK_BFP8_E16_HEAD64_V1_READINESS.json"
READINESS_SHA_PATH = READINESS_PATH.with_suffix(READINESS_PATH.suffix + ".sha256")
REFERENCE_PATH = ROOT / "reference/qk_bfp8_e16_head64_v1.py"
SOFTWARE_CONTRACT_PATH = ROOT / "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json"
FRESH_L2_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/71ae6f113ca4/round-0001.json"
)
FRESH_L2_MISSION_PATH = FRESH_L2_PATH.with_name("mission.json")

EXPECTED_CONTRACT_SHA256 = "92840df94b6f9bc2b1bfc752fd4e24da58cde64f811b2fd84190462badd1a4ef"
EXPECTED_FRESH_L2_SHA256 = "65c439db0d88dceb19426cd0bfef9656dcf5d55ce8bcf741615dfa992d6632ce"
EXPECTED_FRESH_L2_MISSION_SHA256 = "5dae659157a79764d1d9c283a72e5d0260202dcf03ccc331903697fd316d2094"
EXPECTED_SOFTWARE_CONTRACT_SHA256 = "8f037b438f53bb378c60eded9cf15593a8173b11dd42130b40ea7658939140ef"
EXPECTED_BASE_IDENTITY_SHA256 = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
EXPECTED_CHECKPOINT_IDENTITY_SHA256 = "94eb1fe142aa11925863386e796e1157341351cfb664283388291a4fa9d8bf98"
EXPECTED_ARTIFACT_KIND = "qk_bfp8_e16_head64_v1_implementation_readiness_binding"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_CLAIM_BOUNDARY = {
    "calibration_or_scoring_performed": False,
    "candidate_or_execution_package_created": False,
    "generation_or_repair_replay_performed": False,
    "model_loaded_or_executed": False,
    "rtl_spec_manifest_or_u280_modified": False,
    "stage_transition_authorized": False,
}
READINESS_TOP_LEVEL_KEYS = frozenset(
    {
        "artifact_kind",
        "authority",
        "claim_boundary",
        "normalized_subject",
        "normalized_subject_sha256",
        "readiness_artifacts",
        "schema_version",
        "source_identities",
    }
)
READINESS_ARTIFACT_KEYS = frozenset({"pure_reference", "static_verifier"})
READINESS_ARTIFACT_RECORD_KEYS = frozenset({"authority", "path", "sha256"})


class StaticInvalidity(RuntimeError):
    def __init__(self, taxonomy: str, message: str):
        super().__init__(message)
        self.taxonomy = taxonomy


def require(condition: bool, taxonomy: str, message: str) -> None:
    if not condition:
        raise StaticInvalidity(taxonomy, message)


def require_exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    require(type(value) is dict, "schema_mismatch", f"{label} is not an object")
    actual = frozenset(value)
    require(
        actual == expected,
        "schema_mismatch",
        f"{label} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}",
    )


def sha256_file(path: Path) -> str:
    require(path.is_file(), "missing_binding", f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StaticInvalidity("malformed_artifact", f"cannot load JSON {path}: {error}") from error
    require(isinstance(value, dict), "malformed_artifact", f"JSON root is not an object: {path}")
    return value


def canonical_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_subject_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_file_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_file_bytes(value)).hexdigest()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def verify_named_sidecar(sidecar_path: Path, subject_path: Path) -> None:
    lines = sidecar_path.read_text(encoding="ascii").splitlines()
    require(
        len(lines) == 2 and lines[0] == AUTHORITY,
        "authority_violation",
        f"checksum sidecar lacks the authority marker: {sidecar_path}",
    )
    fields = lines[1].split("  ", 1)
    require(
        len(fields) == 2 and fields[1] == subject_path.name,
        "binding_mismatch",
        f"checksum sidecar names the wrong subject: {sidecar_path}",
    )
    require(
        fields[0] == sha256_file(subject_path),
        "binding_mismatch",
        f"checksum sidecar does not close: {subject_path}",
    )


def iter_path_bindings(value: Any) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            bindings.append({"path": value["path"], "sha256": value["sha256"]})
        for child in value.values():
            bindings.extend(iter_path_bindings(child))
    elif isinstance(value, list):
        for child in value:
            bindings.extend(iter_path_bindings(child))
    return bindings


def verify_contract_bindings(contract: dict[str, Any]) -> int:
    bindings = iter_path_bindings(contract["frozen_bindings"])
    require(len(bindings) == 21, "binding_mismatch", "accepted contract no longer has 21 path bindings")
    for binding in bindings:
        path = resolve_path(binding["path"])
        require(
            sha256_file(path) == binding["sha256"],
            "binding_mismatch",
            f"accepted contract binding drifted: {binding['path']}",
        )
    return len(bindings)


def expected_normalized_subject() -> dict[str, Any]:
    return {
        "arithmetic": {
            "bf16_finite_maximum": "255 * 2^120",
            "bf16_minimum_positive_subnormal": "2^-133",
            "decoder": "signed_mantissa_times_2_pow_signed_exponent",
            "exponent_rule": "least_integer_e_such_that_max_abs_le_127_times_2_pow_e",
            "head_lanes": 64,
            "non_finite_policy": "reject_before_payload_creation",
            "rounding": "round_to_nearest_ties_to_even",
            "saturation_or_clamping_allowed": False,
            "score_rule": "sum_64_exact_signed_mantissa_products_and_add_head_exponents",
            "zero_head_exponent": 0,
        },
        "authority": AUTHORITY,
        "contract_id": "QK_BFP8_E16_HEAD64_V1",
        "cost": {
            "baseline_bytes_per_token_per_layer": 264,
            "contract_bytes_per_token_per_layer": 260,
            "delta_bytes_for_24_layers_and_512_tokens": -49152,
            "delta_bytes_per_token_across_24_layers": -96,
            "delta_bytes_per_token_per_layer": -4,
            "query_transient_bytes_per_token_per_layer": 924,
        },
        "model_bindings": {
            "Base": {
                "model_identity_sha256": EXPECTED_BASE_IDENTITY_SHA256,
                "representation": "QK_BFP8_E16_HEAD64_V1",
            },
            "checkpoint-176": {
                "model_identity_sha256": EXPECTED_CHECKPOINT_IDENTITY_SHA256,
                "representation": "QK_BFP8_E16_HEAD64_V1",
            },
        },
        "packing": {
            "head_record_bytes": 66,
            "head_record_layout": "64_signed_int8_lane_bytes_then_signed_int16_little_endian_exponent",
            "key_record_order": ["layer", "token", "kv_head", "head_record"],
            "padding_bytes": 0,
            "query_persisted_in_kv_cache": False,
        },
        "ranges": {
            "head_exponent": [-139, 122],
            "lane_mantissa": [-127, 127],
            "lane_product": [-16129, 16129],
            "reserved_lane_payload": -128,
            "score_exponent": [-278, 244],
            "score_mantissa": [-1032256, 1032256],
        },
        "topology": {
            "head_dim": 64,
            "hidden_size": 896,
            "maximum_bound_sequence_length": 512,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
        },
    }


def expected_source_identities() -> dict[str, Any]:
    return {
        "accepted_contract": {
            "contract_id": "QK_BFP8_E16_HEAD64_V1",
            "path": "design/W4A8_C02_SATURATION_FREE_JOINT_QK_REPRESENTATION_PREREQUISITE_V1.json",
            "sha256": EXPECTED_CONTRACT_SHA256,
            "verdict": "CONTRACT_ACCEPTED_NO_EXECUTION_AUTHORITY",
        },
        "corrected_fresh_l2_handoff": {
            "mission_id": "71ae6f113ca4",
            "path": str(FRESH_L2_PATH),
            "producer_role": "reviewer",
            "review_status": "done",
            "round": 1,
            "sha256": EXPECTED_FRESH_L2_SHA256,
        },
        "corrected_fresh_l2_mission": {
            "mission_id": "71ae6f113ca4",
            "path": str(FRESH_L2_MISSION_PATH),
            "sha256": EXPECTED_FRESH_L2_MISSION_SHA256,
        },
        "models": {
            "Base": {
                "alias": "base",
                "manifest": {
                    "path": "build/w4a8-full-model-software-contract-v1/base/c01-mse-clip-grid/attempt-0001/manifest.json",
                    "sha256": "130431a0e5f681a5947b5baae7ae9aa0db4d6ce501f23fe6ab4b7ce952b066cf",
                },
                "model_id": "qwen2.5-0.5b-instruct",
                "model_identity_sha256": EXPECTED_BASE_IDENTITY_SHA256,
            },
            "checkpoint-176": {
                "alias": "checkpoint-176",
                "base_model_identity": "qwen2.5-0.5b-instruct",
                "manifest": {
                    "path": "build/w4a8-full-model-software-contract-v1/checkpoint-176/c01-mse-clip-grid/attempt-0001/manifest.json",
                    "sha256": "3b1e8f8cd15d481c40fc9cb7f73c20ebb754a80bcdeebcaf3cf422574c96b2c7",
                },
                "model_id": "ace2-lora-v4-checkpoint-176",
                "model_identity_sha256": EXPECTED_CHECKPOINT_IDENTITY_SHA256,
            },
        },
        "software_contract": {
            "path": "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json",
            "sha256": EXPECTED_SOFTWARE_CONTRACT_SHA256,
        },
    }


def expected_readiness_artifacts() -> dict[str, Any]:
    return {
        "pure_reference": {
            "authority": AUTHORITY,
            "path": "reference/qk_bfp8_e16_head64_v1.py",
            "sha256": sha256_file(REFERENCE_PATH),
        },
        "static_verifier": {
            "authority": AUTHORITY,
            "path": "tools/verify_qk_bfp8_e16_head64_v1_readiness.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def verify_readiness_schema(readiness: dict[str, Any]) -> None:
    require_exact_keys(readiness, READINESS_TOP_LEVEL_KEYS, "readiness")
    require(
        readiness["artifact_kind"] == EXPECTED_ARTIFACT_KIND,
        "schema_mismatch",
        "readiness artifact_kind differs",
    )
    require(
        type(readiness["schema_version"]) is int
        and readiness["schema_version"] == EXPECTED_SCHEMA_VERSION,
        "schema_mismatch",
        "readiness schema_version differs",
    )
    require(type(readiness["authority"]) is str, "schema_mismatch", "readiness authority is not a string")
    require(
        type(readiness["normalized_subject_sha256"]) is str,
        "schema_mismatch",
        "normalized subject checksum is not a string",
    )
    require(type(readiness["normalized_subject"]) is dict, "schema_mismatch", "normalized subject is not an object")
    require(type(readiness["source_identities"]) is dict, "schema_mismatch", "source identities are not an object")

    boundary = readiness["claim_boundary"]
    require_exact_keys(boundary, frozenset(EXPECTED_CLAIM_BOUNDARY), "claim_boundary")
    require(
        boundary == EXPECTED_CLAIM_BOUNDARY,
        "authority_violation",
        "readiness claims prohibited authority or execution",
    )

    artifacts = readiness["readiness_artifacts"]
    require_exact_keys(artifacts, READINESS_ARTIFACT_KEYS, "readiness_artifacts")
    for label in sorted(READINESS_ARTIFACT_KEYS):
        record = artifacts[label]
        require_exact_keys(record, READINESS_ARTIFACT_RECORD_KEYS, f"readiness_artifacts.{label}")
        for field in READINESS_ARTIFACT_RECORD_KEYS:
            require(
                type(record[field]) is str,
                "schema_mismatch",
                f"readiness_artifacts.{label}.{field} is not a string",
            )


def expect_schema_rejection(readiness: dict[str, Any], mutate: Any, label: str) -> None:
    candidate = copy.deepcopy(readiness)
    mutate(candidate)
    try:
        verify_readiness_schema(candidate)
    except StaticInvalidity:
        return
    raise StaticInvalidity("schema_mismatch", f"schema mutation was accepted: {label}")


def verify_schema_mutations(readiness: dict[str, Any]) -> int:
    mutations = (
        (lambda value: value.pop("artifact_kind"), "missing top-level field"),
        (lambda value: value.__setitem__("unexpected", False), "extra top-level field"),
        (
            lambda value: value["claim_boundary"].pop("model_loaded_or_executed"),
            "missing claim-boundary field",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__("unexpected", False),
            "extra claim-boundary field",
        ),
        (
            lambda value: value["readiness_artifacts"].pop("pure_reference"),
            "missing readiness artifact",
        ),
        (
            lambda value: value["readiness_artifacts"].__setitem__(
                "execution_package",
                {"authority": AUTHORITY, "path": "forbidden", "sha256": "0" * 64},
            ),
            "extra readiness artifact",
        ),
        (
            lambda value: value["readiness_artifacts"]["pure_reference"].pop("authority"),
            "missing artifact-record field",
        ),
        (
            lambda value: value["readiness_artifacts"]["pure_reference"].__setitem__(
                "execution_authority", False
            ),
            "extra artifact-record field",
        ),
        (
            lambda value: value.__setitem__("artifact_kind", "unexpected"),
            "wrong artifact_kind",
        ),
        (
            lambda value: value.__setitem__("schema_version", 2),
            "wrong schema_version",
        ),
    )
    for mutate, label in mutations:
        expect_schema_rejection(readiness, mutate, label)
    return len(mutations)


def verify_contract_semantics(contract: dict[str, Any]) -> None:
    require(contract.get("contract_id") == "QK_BFP8_E16_HEAD64_V1", "semantic_mismatch", "contract id differs")
    require(
        contract["static_review"]["verdict"] == "CONTRACT_ACCEPTED_NO_EXECUTION_AUTHORITY",
        "authority_violation",
        "accepted contract verdict differs",
    )
    require(contract["authority_boundary"]["stage1_complete"] is False, "authority_violation", "contract claims Stage 1 completion")
    representation = contract["representation"]
    require("least integer e" in representation["encoder"]["exponent_selection"], "semantic_mismatch", "exponent rule differs")
    require("ties_to_even" in representation["encoder"]["mantissa_quantization"], "semantic_mismatch", "rounding rule differs")
    require("forbidden" in representation["encoder"]["mantissa_quantization"], "semantic_mismatch", "no-clamp rule differs")
    require(representation["integer_fields"]["lane_mantissa"]["canonical_range"] == [-127, 127], "semantic_mismatch", "mantissa range differs")
    require(representation["integer_fields"]["lane_mantissa"]["reserved_values"] == [-128], "semantic_mismatch", "reserved payload differs")
    require(representation["integer_fields"]["head_exponent"]["canonical_nonzero_range"] == [-139, 122], "semantic_mismatch", "head exponent range differs")
    require(representation["integer_fields"]["dot_product_exponent"]["canonical_range"] == [-278, 244], "semantic_mismatch", "score exponent range differs")
    require(representation["integer_fields"]["lane_product"]["exact_range"] == [-16129, 16129], "semantic_mismatch", "lane product range differs")
    require(representation["integer_fields"]["score_mantissa"]["exact_range_for_64_lanes"] == [-1032256, 1032256], "semantic_mismatch", "score mantissa range differs")
    require(representation["packing"]["key_head_record"]["bytes"] == 66, "semantic_mismatch", "packed head byte count differs")
    require("little-endian" in representation["packing"]["key_head_record"]["layout"], "semantic_mismatch", "packed exponent endian differs")
    require(contract["model_topology"] == expected_normalized_subject()["topology"] | {"representation_bindings": {"Base": "QK_BFP8_E16_HEAD64_V1", "checkpoint-176": "QK_BFP8_E16_HEAD64_V1"}}, "semantic_mismatch", "topology or shared model binding differs")
    proof = contract["acceptance_contract"]["required_static_proofs"]
    require(proof["bf16_finite_domain"] == {"maximum_absolute_value": "255 * 2^120", "minimum_positive_subnormal": "2^-133"}, "semantic_mismatch", "BF16 finite-domain proof differs")
    require(proof["dot_product"]["exact_signed_15_product_range"] == [-16129, 16129], "semantic_mismatch", "signed-15 proof differs")
    require(proof["dot_product"]["exact_signed_21_sum_range_for_64_lanes"] == [-1032256, 1032256], "semantic_mismatch", "signed-21 proof differs")
    cost = contract["storage_cost"]
    require(cost["c01_baseline"]["total_key_and_value_bytes_per_token_per_layer"] == 264, "cost_mismatch", "baseline cost differs")
    require(cost["contract"]["total_key_and_value_bytes_per_token_per_layer"] == 260, "cost_mismatch", "contract cost differs")
    require(cost["contract"]["query_transient_bytes_per_token_per_layer"] == 924, "cost_mismatch", "query transient cost differs")
    require(cost["delta_from_c01"]["bytes_for_24_layers_and_512_tokens"] == -49152, "cost_mismatch", "aggregate cost delta differs")


def verify_fresh_l2() -> None:
    require(sha256_file(FRESH_L2_PATH) == EXPECTED_FRESH_L2_SHA256, "predecessor_mismatch", "corrected Fresh-L2 handoff drifted")
    require(sha256_file(FRESH_L2_MISSION_PATH) == EXPECTED_FRESH_L2_MISSION_SHA256, "predecessor_mismatch", "corrected Fresh-L2 mission drifted")
    handoff = load_json(FRESH_L2_PATH)
    mission = load_json(FRESH_L2_MISSION_PATH)
    require(FRESH_L2_PATH.parent.name == "71ae6f113ca4", "predecessor_mismatch", "Fresh-L2 directory label differs")
    require(handoff.get("mission_id") == "71ae6f113ca4", "predecessor_mismatch", "Fresh-L2 mission id differs")
    require(handoff.get("round") == 1, "predecessor_mismatch", "Fresh-L2 round differs")
    require(handoff.get("producer_role") == "reviewer", "predecessor_mismatch", "Fresh-L2 producer is not Reviewer")
    require(handoff.get("review", {}).get("status") == "done", "predecessor_mismatch", "Fresh-L2 status is not done")
    reason = handoff["review"]["reason"]
    for phrase in ("accepted the C02 contract", "21/21 bound hashes", "no-saturation", "No RTL/model flow", "no RTL correctness is claimed"):
        require(phrase in reason, "predecessor_mismatch", f"Fresh-L2 reason lacks: {phrase}")
    require(mission.get("mission_id") == handoff["mission_id"], "predecessor_mismatch", "Fresh-L2 mission file identity differs")
    objective = mission.get("objective", "")
    for phrase in (EXPECTED_CONTRACT_SHA256, "CONTRACT_ACCEPTED_NO_EXECUTION_AUTHORITY", "no-RTL", "no-stage-transition", "No elaboration"):
        require(phrase in objective, "predecessor_mismatch", f"Fresh-L2 mission lacks: {phrase}")


def verify_model_identities(contract: dict[str, Any], readiness: dict[str, Any]) -> None:
    require(sha256_file(SOFTWARE_CONTRACT_PATH) == EXPECTED_SOFTWARE_CONTRACT_SHA256, "model_identity_mismatch", "software contract drifted")
    software = load_json(SOFTWARE_CONTRACT_PATH)
    identities = software["model_identities"]
    base = identities["qwen2.5-0.5b-instruct"]
    checkpoint = identities["ace2_lora_v4_checkpoint176"]
    require(canonical_file_sha256(base) == EXPECTED_BASE_IDENTITY_SHA256, "model_identity_mismatch", "Base normalized identity differs")
    require(canonical_file_sha256(checkpoint) == EXPECTED_CHECKPOINT_IDENTITY_SHA256, "model_identity_mismatch", "checkpoint-176 normalized identity differs")
    require(base["model_id"] == "qwen2.5-0.5b-instruct", "model_identity_mismatch", "Base model id differs")
    require(base["repository"] == "Qwen/Qwen2.5-0.5B-Instruct", "model_identity_mismatch", "Base repository differs")
    require(base["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775", "model_identity_mismatch", "Base revision differs")
    require(base["config"]["head_dim"] == 64 and base["config"]["num_hidden_layers"] == 24, "model_identity_mismatch", "Base topology differs")
    require(checkpoint["model_id"] == "ace2-lora-v4-checkpoint-176", "model_identity_mismatch", "checkpoint model id differs")
    require(checkpoint["base_model_identity"] == base["model_id"], "model_identity_mismatch", "checkpoint Base linkage differs")
    require(checkpoint["adapter"]["checkpoint_tree_sha256"] == "8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee", "model_identity_mismatch", "checkpoint tree identity differs")
    for label, expected_alias, expected_identity in (
        ("Base", "base", EXPECTED_BASE_IDENTITY_SHA256),
        ("checkpoint-176", "checkpoint-176", EXPECTED_CHECKPOINT_IDENTITY_SHA256),
    ):
        record = readiness["source_identities"]["models"][label]
        manifest_path = resolve_path(record["manifest"]["path"])
        require(sha256_file(manifest_path) == record["manifest"]["sha256"], "model_identity_mismatch", f"{label} manifest drifted")
        manifest = load_json(manifest_path)
        require(manifest.get("model_alias") == expected_alias, "model_identity_mismatch", f"{label} manifest alias differs")
        require(manifest.get("model_identity_sha256") == expected_identity, "model_identity_mismatch", f"{label} manifest identity differs")
        require(manifest.get("contract_sha256") == EXPECTED_SOFTWARE_CONTRACT_SHA256, "model_identity_mismatch", f"{label} software contract binding differs")
    require(contract["model_topology"]["representation_bindings"] == {"Base": "QK_BFP8_E16_HEAD64_V1", "checkpoint-176": "QK_BFP8_E16_HEAD64_V1"}, "model_identity_mismatch", "models do not share the representation")


def verify_reference_purity() -> None:
    source = REFERENCE_PATH.read_text(encoding="utf-8")
    require(AUTHORITY in source, "authority_violation", "pure reference lacks authority marker")
    tree = ast.parse(source, filename=str(REFERENCE_PATH))
    forbidden_calls = {
        "eval", "exec", "open", "compile", "input", "__import__",
        "read_text", "read_bytes", "write_text", "write_bytes",
        "mkdir", "unlink", "rename", "replace", "system", "popen",
    }
    for node in ast.walk(tree):
        require(not isinstance(node, (ast.Import, ast.ImportFrom)), "authority_violation", "pure reference imports a module")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                call_name = ""
            require(call_name not in forbidden_calls, "authority_violation", f"pure reference contains forbidden call: {call_name}")


def verify_no_reference_bytecode() -> None:
    bytecode = sorted(
        REFERENCE_PATH.parent.glob(f"__pycache__/{REFERENCE_PATH.stem}.*.pyc")
    )
    require(
        not bytecode,
        "artifact_set_mismatch",
        f"unexpected reference bytecode artifact(s): {[str(path) for path in bytecode]}",
    )


def load_reference() -> Any:
    specification = importlib.util.spec_from_file_location("qk_bfp8_e16_head64_v1", REFERENCE_PATH)
    require(specification is not None and specification.loader is not None, "reference_mismatch", "cannot create reference module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expect_reference_violation(reference: Any, operation: Any, label: str) -> None:
    try:
        operation()
    except reference.ReferenceViolation:
        return
    raise StaticInvalidity("reference_mismatch", f"malformed vector was accepted: {label}")


def verify_reference_vectors(reference: Any) -> int:
    require(reference.AUTHORITY == AUTHORITY, "authority_violation", "reference authority marker differs")
    zero_words = tuple(0x8000 if lane & 1 else 0x0000 for lane in range(64))
    zero_mantissas, zero_exponent = reference.encode_head(zero_words)
    require(zero_mantissas == (0,) * 64 and zero_exponent == 0, "reference_mismatch", "signed-zero vector differs")

    minimum_words = (0x0001,) + (0,) * 63
    minimum_mantissas, minimum_exponent = reference.encode_head(minimum_words)
    require(minimum_exponent == -139 and minimum_mantissas[0] == 64, "reference_mismatch", "minimum subnormal vector differs")
    negative_minimum, negative_minimum_exponent = reference.encode_head((0x8001,) + (0,) * 63)
    require(negative_minimum_exponent == -139 and negative_minimum[0] == -64, "reference_mismatch", "negative minimum subnormal differs")

    tie_words = (0x3F80, 0x3C00, 0x3CC0, 0xBC00, 0xBCC0) + (0,) * 59
    tie_mantissas, tie_exponent = reference.encode_head(tie_words)
    require(tie_exponent == -6, "reference_mismatch", "tie vector exponent differs")
    require(tie_mantissas[:5] == (64, 0, 2, 0, -2), "reference_mismatch", "ties-to-even vector differs")

    maximum_mantissas, maximum_exponent = reference.encode_head((0x7F7F,) + (0,) * 63)
    require(maximum_exponent == 122 and maximum_mantissas[0] == 64, "reference_mismatch", "maximum finite vector differs")
    exact_127, exact_127_exponent = reference.encode_head((0x42FE,) + (0,) * 63)
    require(exact_127_exponent == 0 and exact_127[0] == 127, "reference_mismatch", "exact +127 vector differs")

    packed_mantissas = (-127, -1, 0, 1, 127) + (0,) * 59
    packed = reference.pack_head(packed_mantissas, -139)
    require(len(packed) == 66, "packing_mismatch", "packed byte count differs")
    require(packed[:5] == bytes((0x81, 0xFF, 0x00, 0x01, 0x7F)), "packing_mismatch", "mantissa byte order differs")
    require(packed[-2:] == bytes((0x75, 0xFF)), "packing_mismatch", "signed little-endian exponent differs")
    require(reference.unpack_head(packed) == (packed_mantissas, -139), "packing_mismatch", "pack/unpack round trip differs")
    require(reference.decode_head_dyadics(packed_mantissas, -139)[0] == (-127, -139), "reference_mismatch", "decoder dyadic differs")

    positive = (127,) * 64
    negative = (-127,) * 64
    require(reference.dot_product(positive, 122, positive, 122) == (1032256, 244), "range_mismatch", "maximum dot product differs")
    require(reference.dot_product(negative, -139, positive, -139) == (-1032256, -278), "range_mismatch", "minimum dot product differs")
    require(127 * 127 == 16129 and 64 * 16129 == 1032256, "range_mismatch", "integer range proof differs")
    require(-(1 << 14) <= -16129 and 16129 <= (1 << 14) - 1, "range_mismatch", "lane product does not fit signed 15 bits")
    require(-(1 << 20) <= -1032256 and 1032256 <= (1 << 20) - 1, "range_mismatch", "score sum does not fit signed 21 bits")
    require(-(1 << 9) <= -278 and 244 <= (1 << 9) - 1, "range_mismatch", "score exponent does not fit signed 10 bits")

    expected_cost = {
        "baseline_total_bytes_for_24_layers_and_512_tokens": 3244032,
        "baseline_total_bytes_per_token_per_layer": 264,
        "contract_total_bytes_for_24_layers_and_512_tokens": 3194880,
        "contract_total_bytes_per_token_per_layer": 260,
        "delta_bytes_for_24_layers_and_512_tokens": -49152,
        "delta_bytes_per_token_across_24_layers": -96,
        "delta_bytes_per_token_per_layer": -4,
        "query_transient_bytes_per_token_per_layer": 924,
    }
    require(reference.storage_cost() == expected_cost, "cost_mismatch", "reference storage cost differs")

    expect_reference_violation(reference, lambda: reference.encode_head((0x7F80,) + (0,) * 63), "infinity")
    expect_reference_violation(reference, lambda: reference.encode_head((0x7FC1,) + (0,) * 63), "NaN")
    expect_reference_violation(reference, lambda: reference.unpack_head(bytes(65)), "short payload")
    expect_reference_violation(reference, lambda: reference.unpack_head(bytes((0x80,)) + bytes(63) + bytes(2)), "reserved payload")
    expect_reference_violation(reference, lambda: reference.unpack_head(bytes(64) + (1).to_bytes(2, "little", signed=True)), "zero head nonzero exponent")
    expect_reference_violation(reference, lambda: reference.unpack_head(bytes((1,)) + bytes(63) + (123).to_bytes(2, "little", signed=True)), "high noncanonical exponent")
    expect_reference_violation(reference, lambda: reference.unpack_head(bytes((1,)) + bytes(63) + (-140).to_bytes(2, "little", signed=True)), "low noncanonical exponent")

    require(reference.encode_head(tie_words) == reference.encode_head(tie_words), "determinism_mismatch", "reference result is not deterministic")
    return 19


def verify_readiness() -> dict[str, Any]:
    require(sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256, "binding_mismatch", "accepted contract hash differs")
    contract_sidecar = CONTRACT_SHA_PATH.read_text(encoding="ascii").strip().split()
    require(contract_sidecar == [EXPECTED_CONTRACT_SHA256, CONTRACT_PATH.name], "binding_mismatch", "accepted contract checksum companion differs")
    contract = load_json(CONTRACT_PATH)
    verify_contract_semantics(contract)
    binding_count = verify_contract_bindings(contract)
    verify_fresh_l2()

    readiness = load_json(READINESS_PATH)
    require(READINESS_PATH.read_bytes() == canonical_file_bytes(readiness), "malformed_artifact", "readiness JSON is not canonical")
    verify_named_sidecar(READINESS_SHA_PATH, READINESS_PATH)
    verify_readiness_schema(readiness)
    mutation_count = verify_schema_mutations(readiness)
    require(readiness["authority"] == AUTHORITY, "authority_violation", "readiness authority marker differs")
    require(readiness["source_identities"] == expected_source_identities(), "binding_mismatch", "readiness source identities differ")

    normalized_subject = readiness["normalized_subject"]
    require(normalized_subject == expected_normalized_subject(), "semantic_mismatch", "normalized readiness subject differs")
    subject_sha256 = hashlib.sha256(canonical_subject_bytes(normalized_subject)).hexdigest()
    require(readiness["normalized_subject_sha256"] == subject_sha256, "binding_mismatch", "normalized subject checksum differs")

    require(
        readiness["readiness_artifacts"] == expected_readiness_artifacts(),
        "binding_mismatch",
        "readiness artifact set or binding differs",
    )

    verify_model_identities(contract, readiness)
    verify_no_reference_bytecode()
    verify_reference_purity()
    reference = load_reference()
    vector_count = verify_reference_vectors(reference)
    verify_no_reference_bytecode()
    return {
        "authority": AUTHORITY,
        "bound_contract_files": binding_count,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fresh_l2": "71ae6f113ca4/round-0001",
        "normalized_subject_sha256": subject_sha256,
        "reference_vectors": vector_count,
        "schema_mutations": mutation_count,
        "status": "STATIC_VALID_NO_EXECUTION_AUTHORITY",
    }


def main() -> int:
    try:
        result = verify_readiness()
    except StaticInvalidity as error:
        print(f"STATIC_INVALID_NO_EXECUTION_AUTHORITY taxonomy={error.taxonomy} detail={error}")
        return 1
    print(
        "STATIC_VALID_NO_EXECUTION_AUTHORITY "
        f"contract_sha256={result['contract_sha256']} "
        f"fresh_l2={result['fresh_l2']} "
        f"bindings={result['bound_contract_files']} "
        f"vectors={result['reference_vectors']} "
        f"schema_mutations={result['schema_mutations']} "
        f"subject_sha256={result['normalized_subject_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
