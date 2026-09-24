#!/usr/bin/env python3
"""Fail-closed static verifier for the candidate-free c02 score-path package."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json"
)
PROOF_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-score-path-evaluation-package-v1/proof.json"
)
PACKAGE = ROOT / PACKAGE_REL
PACKAGE_SIDECAR = PACKAGE.with_suffix(PACKAGE.suffix + ".sha256")
PROOF = ROOT / PROOF_REL
PROOF_SIDECAR = PROOF.with_suffix(PROOF.suffix + ".sha256")
VERIFIER = Path(__file__).resolve()

PACKAGE_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE_V1"
EVALUATOR_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATOR_SPEC_V1"
AUTHORITY = "NO_EXECUTION_AUTHORITY"
OUTPUT_ROOT = "build/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-v1"
EXPECTED_EVALUATOR_SPEC_SHA256 = "b719d18f571e664b25c926ecb972f9972f2a13c4b8013d89f8bf970cf8a99756"
EXPECTED_OUTPUT_SCHEMA_SHA256 = "b59be734589d054f3785c2b0b20e6c48b5c20f81ea2ead668cbc71eee1491f62"

EXPECTED_NORMALIZATION_BINDINGS = {
    "contract": {
        "authority": AUTHORITY,
        "byte_count": 11221,
        "path": "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_CONTRACT.json",
        "sha256": "539ff95d8f1ad253880c9999fa59e6ac41840761908b188cf2c4bce09f3c1b01",
    },
    "pure_reference": {
        "authority": AUTHORITY,
        "byte_count": 11499,
        "path": "reference/qk_bfp8_e16_head64_v1_score_normalization.py",
        "sha256": "8b1db0b13ee3a4534848814e75d13cea9946e380cd09b8bec0e76a527df098a3",
    },
    "qualifying_fresh_l2_review": {
        "authority": AUTHORITY,
        "byte_count": 882,
        "path": (
            "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
            "handoffs/7d4f89c0ec7b/round-0002.json"
        ),
        "sha256": "416853a915893be6116c6911940f9055b219d2348ed6e6e02c1afa5063a825eb",
    },
    "readiness": {
        "authority": AUTHORITY,
        "byte_count": 2216,
        "path": "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_READINESS.json",
        "sha256": "3fb8229b5b7940d9667b4220356b2f10a8532898f9e69a6839704e7f1001fea3",
    },
    "static_verifier": {
        "authority": AUTHORITY,
        "byte_count": 26672,
        "path": "tools/verify_qk_bfp8_e16_head64_v1_score_normalization.py",
        "sha256": "1aaddcaa74045c3de83b833af016a25c492c878b1120ed9cf557ec548a89c5af",
    },
}

EXPECTED_REPRESENTATION_BINDINGS = {
    "contract": {
        "authority": AUTHORITY,
        "byte_count": 15489,
        "path": "design/W4A8_C02_SATURATION_FREE_JOINT_QK_REPRESENTATION_PREREQUISITE_V1.json",
        "sha256": "92840df94b6f9bc2b1bfc752fd4e24da58cde64f811b2fd84190462badd1a4ef",
    },
    "pure_reference": {
        "authority": AUTHORITY,
        "byte_count": 7959,
        "path": "reference/qk_bfp8_e16_head64_v1.py",
        "sha256": "bfb8b0d95ef8a8b939694fe8fec710d4acb1abff8b38b4bbbb4a43bdea5bb704",
    },
    "readiness": {
        "authority": AUTHORITY,
        "byte_count": 5475,
        "path": "reference/QK_BFP8_E16_HEAD64_V1_READINESS.json",
        "sha256": "400d4a4a8b627cd71e930f670852b89e5deead83f14be3f528e752f6d1705fa7",
    },
    "static_verifier": {
        "authority": AUTHORITY,
        "byte_count": 32890,
        "path": "tools/verify_qk_bfp8_e16_head64_v1_readiness.py",
        "sha256": "ea40e12f39351fe6240d861a68488a4bc22517dab72434762f05a8bdf88fdd2e",
    },
}

EXPECTED_SHARED_INPUTS = {
    "aggregate_metadata": {
        "byte_count": 4556,
        "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/result.json",
        "sha256": "935b079f3ea5328c38d65c978e94ba2c8dcff6ab384f3943e9869eabb9f59343",
    },
    "software_contract": {
        "byte_count": 22042,
        "path": "design/QWEN25_05B_W4A8_FULL_MODEL_SOFTWARE_CONTRACT_V1.json",
        "sha256": "8f037b438f53bb378c60eded9cf15593a8173b11dd42130b40ea7658939140ef",
    },
}

EXPECTED_LANES = [
    {
        "authoritative_tensors": {
            "bf16_oracle_scores": {
                "dtype": "torch.bfloat16",
                "sha256": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765",
                "shape": [1, 14, 41, 41],
                "tensor_name": "bf16.qk_scaled_scores",
            },
            "realized_key_source": {
                "dtype": "torch.bfloat16",
                "sha256": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a",
                "shape": [1, 2, 41, 64],
                "tensor_name": "bf16.k_rope",
            },
            "realized_query_source": {
                "dtype": "torch.bfloat16",
                "sha256": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429",
                "shape": [1, 14, 41, 64],
                "tensor_name": "bf16.q_rope",
            },
        },
        "generation_id": "c02-score-path-eval-base-v1",
        "input_token_ids_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
        "label": "Base",
        "metadata": {
            "byte_count": 20057,
            "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json",
            "sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
        },
        "model_identity_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
        "namespace_label": "base",
        "output_path": OUTPUT_ROOT + "/base/result.json",
        "tensor_bundle": {
            "byte_count": 1305797,
            "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin",
            "sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
        },
    },
    {
        "authoritative_tensors": {
            "bf16_oracle_scores": {
                "dtype": "torch.bfloat16",
                "sha256": "776be2e58246c8641fa483e331e0d63896f96f14b3756548e15af3870d89c456",
                "shape": [1, 14, 41, 41],
                "tensor_name": "bf16.qk_scaled_scores",
            },
            "realized_key_source": {
                "dtype": "torch.bfloat16",
                "sha256": "2c4b99f5bc97d0f95b886dfebf62242148940d7cc9021e197dfdee0aed9e9902",
                "shape": [1, 2, 41, 64],
                "tensor_name": "bf16.k_rope",
            },
            "realized_query_source": {
                "dtype": "torch.bfloat16",
                "sha256": "18671805abe3835d8f2e23bffb041ced6a05f41b8eac4f698f41d302def73b18",
                "shape": [1, 14, 41, 64],
                "tensor_name": "bf16.q_rope",
            },
        },
        "generation_id": "c02-score-path-eval-checkpoint-176-v1",
        "input_token_ids_sha256": "1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86",
        "label": "checkpoint-176",
        "metadata": {
            "byte_count": 20070,
            "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/checkpoint-176/result.json",
            "sha256": "3f73b5359e021ddb6183ec468ae154b3cda0807365b248773d80466e477b4e86",
        },
        "model_identity_sha256": "94eb1fe142aa11925863386e796e1157341351cfb664283388291a4fa9d8bf98",
        "namespace_label": "checkpoint-176",
        "output_path": OUTPUT_ROOT + "/checkpoint-176/result.json",
        "tensor_bundle": {
            "byte_count": 1305797,
            "path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/checkpoint-176/attention-substage-tensors.bin",
            "sha256": "07743a5df6bfa00d5e977dc622b70e90af9b6789feaeb5ae689448ac803ceb59",
        },
    },
]

CLAIM_BOUNDARY = {
    "authority_consumed": False,
    "calibration_or_generation_performed": False,
    "candidate_or_execution_namespace_created": False,
    "fixed_input_execution_or_score_computation_performed": False,
    "model_loaded_or_executed": False,
    "repair_replay_resume_or_retry_performed": False,
    "rtl_spec_manifest_u280_or_stage_state_modified": False,
}

NEGATIVE_TESTS = [
    "authority_marker_mutation",
    "normalization_hash_mutation",
    "representation_hash_mutation",
    "lane_model_identity_mutation",
    "oracle_tensor_name_mutation",
    "realized_key_shape_mutation",
    "lane_tensor_record_hash_mutation",
    "causal_mask_rule_mutation",
    "cross_lane_output_alias",
    "invocation_hash_mutation",
    "top_key_threshold_relaxation",
    "singleton_rank_margin_rule_mutation",
    "empty_fraction_rule_mutation",
    "nested_input_schema_field_mutation",
    "rank_margin_nullability_mutation",
    "null_fraction_threshold_pass_mutation",
    "terminal_retry_permission",
    "terminal_state_schema_mutation",
    "missing_required_key",
]


class VerificationError(RuntimeError):
    """Static package verification failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _pretty_text(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_compact_bytes(value)).hexdigest()


def _resolve(bound_path: str) -> Path:
    path = Path(bound_path)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path, *, canonical: bool = False) -> Any:
    _require(path.is_file(), f"missing JSON artifact: {path}")
    _require(not path.is_symlink(), f"symlinked JSON artifact: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON artifact: {path}: {exc}") from exc
    if canonical:
        _require(text == _pretty_text(value), f"noncanonical JSON artifact: {path}")
    return value


def _verify_sidecar(path: Path, artifact: Path, artifact_rel: str) -> None:
    _require(path.is_file(), f"missing SHA-256 sidecar: {path}")
    _require(not path.is_symlink(), f"symlinked SHA-256 sidecar: {path}")
    expected = f"{_sha256_file(artifact)}  {artifact_rel}\n"
    _require(path.read_text(encoding="ascii") == expected, f"invalid SHA-256 sidecar: {path}")


def _verify_binding(binding: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    _require(binding == expected, f"{label} binding mismatch")
    path = _resolve(binding["path"])
    _require(path.is_file(), f"{label} missing: {path}")
    _require(not path.is_symlink(), f"{label} is a symlink: {path}")
    stat = path.stat()
    _require(stat.st_size == binding["byte_count"], f"{label} byte count mismatch")
    _require(_sha256_file(path) == binding["sha256"], f"{label} SHA-256 mismatch")


def _recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def _verify_no_execution_surface() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {
        "cocotb",
        "numpy",
        "onnxruntime",
        "subprocess",
        "tensorflow",
        "torch",
        "transformers",
    }
    forbidden_calls = {
        "Popen",
        "call",
        "compile",
        "eval",
        "exec",
        "from_pretrained",
        "generate",
        "popen",
        "run",
        "system",
    }
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    _require(not (imported & forbidden_modules), "verifier imports an execution/model module")
    _require(not (called & forbidden_calls), "verifier contains a prohibited execution call")


def _verify_normalization_semantics() -> None:
    contract = _load_json(_resolve(EXPECTED_NORMALIZATION_BINDINGS["contract"]["path"]))
    readiness = _load_json(_resolve(EXPECTED_NORMALIZATION_BINDINGS["readiness"]["path"]))
    review = _load_json(_resolve(EXPECTED_NORMALIZATION_BINDINGS["qualifying_fresh_l2_review"]["path"]))

    _require(contract.get("contract_id") == "QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_V2", "normalization contract id")
    _require(contract.get("authority") == AUTHORITY, "normalization contract authority")
    _require(contract.get("realization", {}).get("format") == "signed-int64-container Q12.20 attention-softmax logit", "normalization realization format")
    _require(contract.get("normalization", {}).get("argmax_tie") == "Choose the lowest key index among exact equal valid maxima. Every valid score equal to the maximum centers to exact zero.", "normalization tie semantics")
    _require(contract.get("claim_boundary", {}).get("fixed_input_or_model_execution_performed") is False, "normalization execution claim")

    _require(readiness.get("authority") == AUTHORITY, "normalization readiness authority")
    _require(readiness.get("contract", {}).get("sha256") == EXPECTED_NORMALIZATION_BINDINGS["contract"]["sha256"], "readiness contract binding")
    _require(readiness.get("static_artifacts", {}).get("pure_reference", {}).get("sha256") == EXPECTED_NORMALIZATION_BINDINGS["pure_reference"]["sha256"], "readiness reference binding")
    _require(readiness.get("static_artifacts", {}).get("static_verifier", {}).get("sha256") == EXPECTED_NORMALIZATION_BINDINGS["static_verifier"]["sha256"], "readiness verifier binding")
    _require(readiness.get("claim_boundary") == {
        "calibration_or_scoring_performed": False,
        "candidate_or_execution_namespace_created": False,
        "fixed_input_or_model_execution_performed": False,
        "generation_or_repair_replay_performed": False,
        "rtl_spec_manifest_or_u280_modified": False,
        "stage_transition_authorized": False,
    }, "normalization readiness claim boundary")

    _require(review.get("kind") == "round_reviewed_handoff", "Fresh-L2 review kind")
    _require(review.get("mission_id") == "7d4f89c0ec7b", "Fresh-L2 mission id")
    _require(review.get("producer_role") == "reviewer", "Fresh-L2 producer role")
    _require(review.get("round") == 2, "Fresh-L2 round")
    _require(review.get("review", {}).get("status") == "done", "Fresh-L2 status")
    reason = review.get("review", {}).get("reason", "")
    _require("NO_EXECUTION_AUTHORITY" in reason, "Fresh-L2 authority semantics")


def _verify_representation_semantics() -> None:
    contract = _load_json(_resolve(EXPECTED_REPRESENTATION_BINDINGS["contract"]["path"]))
    readiness = _load_json(_resolve(EXPECTED_REPRESENTATION_BINDINGS["readiness"]["path"]))

    _require(contract.get("contract_id") == "QK_BFP8_E16_HEAD64_V1", "representation contract id")
    topology = contract.get("model_topology", {})
    _require(topology.get("head_dim") == 64, "representation head dimension")
    _require(topology.get("num_attention_heads") == 14, "representation query-head count")
    _require(topology.get("num_key_value_heads") == 2, "representation key-head count")
    _require(topology.get("representation_bindings") == {
        "Base": "QK_BFP8_E16_HEAD64_V1",
        "checkpoint-176": "QK_BFP8_E16_HEAD64_V1",
    }, "representation lane binding")
    representation = contract.get("representation", {})
    encoder = representation.get("encoder", {})
    _require(encoder.get("source_domain") == "The finite BF16 post-RoPE Q or K head values defined by the bound c02 fixed software reference; upstream projection, RoPE coefficient, and model weights are not changed by this contract.", "representation source domain")
    _require(encoder.get("non_finite_policy") == "Any NaN or infinity rejects the record before payload creation.", "representation non-finite policy")
    _require(representation.get("joint_score_rule") == "For one query head and its mapped key head, score_mantissa = sum over 64 lanes of q_mantissa[i] * k_mantissa[i], score_exponent = q_exponent + k_exponent, and the exact represented pre-scale dot product is score_mantissa * 2^score_exponent. No intermediate requantization, residual sideband, diagonal gain, saturation, or clamp is permitted.", "representation score rule")

    _require(readiness.get("authority") == AUTHORITY, "representation readiness authority")
    subject = readiness.get("normalized_subject", {})
    _require(subject.get("contract_id") == "QK_BFP8_E16_HEAD64_V1", "representation readiness id")
    _require(subject.get("model_bindings") == {
        "Base": {
            "model_identity_sha256": EXPECTED_LANES[0]["model_identity_sha256"],
            "representation": "QK_BFP8_E16_HEAD64_V1",
        },
        "checkpoint-176": {
            "model_identity_sha256": EXPECTED_LANES[1]["model_identity_sha256"],
            "representation": "QK_BFP8_E16_HEAD64_V1",
        },
    }, "representation readiness models")
    _require(readiness.get("source_identities", {}).get("accepted_contract", {}).get("sha256") == EXPECTED_REPRESENTATION_BINDINGS["contract"]["sha256"], "representation readiness contract hash")
    _require(readiness.get("readiness_artifacts", {}).get("pure_reference", {}).get("sha256") == EXPECTED_REPRESENTATION_BINDINGS["pure_reference"]["sha256"], "representation readiness reference hash")
    _require(readiness.get("readiness_artifacts", {}).get("static_verifier", {}).get("sha256") == EXPECTED_REPRESENTATION_BINDINGS["static_verifier"]["sha256"], "representation readiness verifier hash")


def _verify_input_metadata(lane: dict[str, Any]) -> None:
    metadata = _load_json(_resolve(lane["metadata"]["path"]))
    _require(metadata.get("model_identity_sha256") == lane["model_identity_sha256"], f"{lane['label']} metadata model identity")
    _require(metadata.get("status") == "PASS_ATTENTION_SUBSTAGE_LOCALIZED_NON_SCORING", f"{lane['label']} metadata status")
    tensor_bundle = metadata.get("tensor_bundle", {})
    relative_tensor = Path(lane["tensor_bundle"]["path"]).relative_to(
        "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2"
    )
    _require(tensor_bundle.get("path") == str(relative_tensor), f"{lane['label']} tensor metadata path")
    _require(tensor_bundle.get("sha256") == lane["tensor_bundle"]["sha256"], f"{lane['label']} tensor metadata hash")
    _require(tensor_bundle.get("bytes") == lane["tensor_bundle"]["byte_count"], f"{lane['label']} tensor metadata bytes")
    tensor_records = tensor_bundle.get("tensors", {})
    _require(isinstance(tensor_records, dict), f"{lane['label']} tensor record map")
    for role, binding in lane["authoritative_tensors"].items():
        tensor_name = binding["tensor_name"]
        _require(tensor_name in tensor_records, f"{lane['label']} missing authoritative tensor {role}")
        expected_record = {
            "dtype": binding["dtype"],
            "sha256": binding["sha256"],
            "shape": binding["shape"],
        }
        _require(tensor_records[tensor_name] == expected_record, f"{lane['label']} authoritative tensor {role}")
    token_hashes = _recursive_values(metadata, "input_token_ids_sha256")
    _require(token_hashes == [lane["input_token_ids_sha256"]], f"{lane['label']} input-token identity")


def _verify_evaluation_contract(contract: dict[str, Any]) -> None:
    _require(_object_sha256(contract) == EXPECTED_EVALUATOR_SPEC_SHA256, "evaluation contract hash")
    _require(set(contract) == {
        "authoritative_tensor_mapping",
        "bf16_oracle",
        "comparison_metrics",
        "deterministic_ordering",
        "realized_q12_20",
        "thresholds",
    }, "evaluation contract keys")
    mapping = contract["authoritative_tensor_mapping"]
    _require(mapping.get("batch_index") == 0, "authoritative batch index")
    _require(mapping.get("sequence_length") == 41, "authoritative sequence length")
    _require(mapping.get("causal_validity") == "a key slot is valid if and only if key_index <= query_index; all 41 key slots are supplied to normalization and the authoritative valid-key bitmap marks exactly those valid slots", "causal validity rule")
    _require(mapping.get("key_head_mapping") == "kv_head = floor(query_head / 7) for query_head 0 through 13", "key-head mapping")
    _require(mapping.get("roles") == {
        "bf16_oracle_scores": {
            "dtype": "torch.bfloat16",
            "shape": [1, 14, 41, 41],
            "tensor_name": "bf16.qk_scaled_scores",
        },
        "realized_key_source": {
            "dtype": "torch.bfloat16",
            "shape": [1, 2, 41, 64],
            "tensor_name": "bf16.k_rope",
        },
        "realized_query_source": {
            "dtype": "torch.bfloat16",
            "shape": [1, 14, 41, 64],
            "tensor_name": "bf16.q_rope",
        },
    }, "authoritative tensor roles")
    _require(mapping.get("excluded_tensor_records") == [
        "bf16.qk_centered_scores",
        "w4.score_key_s8",
        "w4.score_key_scale32",
        "w4.score_query_s8",
        "w4.score_query_scale32",
        "w4.scores_q6_9",
        "w4.softmax_input_q6_9",
    ], "excluded tensor records")

    oracle = contract["bf16_oracle"]
    _require(oracle.get("source") == "use only bf16.qk_scaled_scores[0,query_head,query_index,key_index]; do not recompute the oracle from Q/K and do not use bf16.qk_centered_scores", "BF16 oracle source")
    _require(oracle.get("stored_scale") == "bf16.qk_scaled_scores is already post-HEAD64 scale 1/8, so no additional scale is applied", "BF16 oracle scale")
    _require(oracle.get("score_error_sign") == "error_q12_20_lsb = realized_q12_20_lsb - oracle_q12_20_lsb", "score-error sign")
    _require(oracle.get("top_key_tie") == "lowest valid key index among exact equal maxima", "BF16 oracle tie")

    realized = contract["realized_q12_20"]
    _require(realized.get("representation_contract_id") == "QK_BFP8_E16_HEAD64_V1", "realized representation id")
    _require(realized.get("representation_contract_sha256") == EXPECTED_REPRESENTATION_BINDINGS["contract"]["sha256"], "realized representation hash")
    _require(realized.get("normalization_contract_id") == "QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_V2", "realized normalization id")
    _require(realized.get("normalization_contract_sha256") == EXPECTED_NORMALIZATION_BINDINGS["contract"]["sha256"], "realized normalization hash")
    _require(realized.get("source") == "query vector bf16.q_rope[0,query_head,query_index,:] and key vector bf16.k_rope[0,floor(query_head/7),key_index,:]", "realized source mapping")
    _require(realized.get("row_input") == "for each row form 41 exact score pairs in increasing key order, pair them with the exact causal bitmap, and normalize the complete row once", "realized row input")

    metrics = contract["comparison_metrics"]
    _require(set(metrics) == {"invalid_accounting", "rank_margin", "score_error", "top_key"}, "metric keys")
    _require(metrics["score_error"].get("gate_mode") == "TRACKING", "score error gate mode")
    _require(metrics["score_error"].get("acceptance_threshold") is None, "score error threshold must remain tracking-only")
    _require(metrics["score_error"].get("fields") == [
        "valid_value_count",
        "sum_signed_error_q12_20_lsb",
        "sum_absolute_error_q12_20_lsb",
        "sum_squared_error_q40_40_lsb2",
        "maximum_absolute_error_q12_20_lsb",
    ], "score error fields")
    _require(metrics["top_key"].get("fields") == [
        "row_count",
        "matching_row_count",
        "mismatch_count",
        "matching_fraction",
    ], "top-key fields")
    _require(metrics["top_key"].get("empty_set_encoding") == "matching_fraction is null if row_count is zero, and a null actual fraction never passes the minimum threshold", "top-key empty fraction")
    _require(metrics["rank_margin"].get("fields") == [
        "singleton_valid_key_row_count",
        "unique_oracle_top_row_count",
        "preserved_positive_margin_row_count",
        "violation_count",
        "minimum_realized_margin_q12_20_lsb",
        "oracle_tied_row_count",
        "preserved_positive_margin_fraction",
    ], "rank-margin fields")
    _require("singleton row" in metrics["rank_margin"].get("accounting", ""), "singleton rank-margin rule")
    _require(metrics["rank_margin"].get("empty_set_encoding") == "minimum_realized_margin_q12_20_lsb and preserved_positive_margin_fraction are null if unique_oracle_top_row_count is zero; an empty population never passes the preservation-fraction threshold", "rank-margin empty encoding")
    _require(metrics["rank_margin"].get("invariants") == "singleton_valid_key_row_count + unique_oracle_top_row_count + oracle_tied_row_count = top_key.row_count; preserved_positive_margin_row_count + violation_count = unique_oracle_top_row_count; preservation requires realized_margin strictly greater than zero", "rank-margin invariants")
    _require(metrics["invalid_accounting"].get("fields") == [
        "cross_lane_record_count",
        "invalid_or_non_finite_value_count",
        "normalization_rejection_count",
        "positive_centered_realized_score_count",
        "saturation_event_count",
    ], "invalid accounting fields")
    _require(contract["thresholds"] == {
        "cross_lane_record_count_maximum": 0,
        "invalid_or_non_finite_value_count_maximum": 0,
        "normalization_rejection_count_maximum": 0,
        "positive_centered_realized_score_count_maximum": 0,
        "rank_margin_violation_count_maximum": 0,
        "saturation_event_count_maximum": 0,
        "top_key_matching_fraction_minimum": {"denominator": 1, "numerator": 1},
        "top_key_mismatch_count_maximum": 0,
        "unique_oracle_positive_margin_preserved_fraction_minimum": {"denominator": 1, "numerator": 1},
    }, "acceptance thresholds")
    _require(contract["deterministic_ordering"] == {
        "head_lane_order": "increasing head_lane 0 through 63 for every QK_BFP8 head encoding and dot product",
        "key_order": "increasing sealed key index",
        "lane_order": ["Base", "checkpoint-176"],
        "row_order": "within each isolated lane: batch index 0 only, then increasing query_head 0 through 13, then increasing query_index 0 through 40; parallel completion order may not alter publication",
    }, "deterministic ordering")


def _verify_output_schema(schema: dict[str, Any]) -> None:
    _require(_object_sha256(schema) == EXPECTED_OUTPUT_SCHEMA_SHA256, "output schema hash")
    _require(schema.get("schema_id") == "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_RESULT_V1", "output schema id")
    _require(schema.get("serialization") == "UTF-8 canonical JSON: sorted keys, separators comma/colon, ASCII escapes, no NaN or Infinity, one trailing LF", "output serialization")
    _require(schema.get("additional_top_level_fields_permitted") is False, "output top-level closure")
    required = schema.get("required_fields")
    _require(required == [
        "schema_id",
        "package_id",
        "evaluator_spec_sha256",
        "lane_label",
        "namespace_label",
        "generation_id",
        "invocation_sha256",
        "model_identity_sha256",
        "input_bindings",
        "terminal",
        "metrics",
        "threshold_evaluation",
        "result_sha256",
    ], "output required fields")
    _require(schema.get("terminal_status_values") == [
        "SUCCEEDED_TERMINAL",
        "FAILED_TERMINAL",
        "NO_EXECUTION_TERMINAL",
    ], "terminal status schema")
    _require(schema.get("result_sha256_rule") == "SHA-256 of the canonical result object with result_sha256 omitted", "result hash rule")

    fractions = schema.get("fraction_encoding", {})
    _require(fractions.get("empty_population") == "encode as JSON null; denominator-zero objects are forbidden", "empty fraction encoding")
    _require(fractions.get("threshold_on_null_actual") == "pass must be false", "null fraction threshold rule")
    _require(fractions.get("threshold_ordering") == "compare fractions by exact integer cross multiplication without host floating point", "fraction threshold arithmetic")

    nested = schema.get("nested_objects", {})
    _require(set(nested) == {
        "artifact_binding",
        "authoritative_tensor_record_map",
        "fraction",
        "fraction_minimum_check",
        "input_bindings",
        "integer_maximum_check",
        "invalid_accounting_metrics",
        "metrics",
        "rank_margin_metrics",
        "score_error_metrics",
        "tensor_record",
        "terminal",
        "threshold_evaluation",
        "top_key_metrics",
    }, "nested output object set")
    for label, object_schema in nested.items():
        _require(object_schema.get("additional_fields_permitted") is False, f"{label} output object closure")
        _require(isinstance(object_schema.get("fields"), dict), f"{label} output fields")
    _require(nested["input_bindings"]["fields"] == {
        "input_token_ids_sha256": "lowercase_64_hex_string",
        "lane_metadata": "artifact_binding",
        "normalization_contract_sha256": "lowercase_64_hex_string",
        "representation_contract_sha256": "lowercase_64_hex_string",
        "sealed_set_id": "string",
        "tensor_bundle": "artifact_binding",
        "tensor_records": "authoritative_tensor_record_map",
    }, "input-bindings schema")
    _require(nested["rank_margin_metrics"]["fields"] == {
        "minimum_realized_margin_q12_20_lsb": "integer_or_null",
        "oracle_tied_row_count": "nonnegative_integer",
        "preserved_positive_margin_fraction": "fraction_or_null",
        "preserved_positive_margin_row_count": "nonnegative_integer",
        "singleton_valid_key_row_count": "nonnegative_integer",
        "unique_oracle_top_row_count": "nonnegative_integer",
        "violation_count": "nonnegative_integer",
    }, "rank-margin result schema")
    _require(nested["terminal"]["fields"] == {
        "first_record_immutable": "boolean_const_true",
        "invocation_count_performed": "integer_0_or_1",
        "metrics_published": "boolean",
        "reason_code": "terminal_reason_code",
        "retry_replay_resume_repair_permitted": "boolean_const_false",
        "status": "terminal_status",
        "thresholds_evaluated": "boolean",
    }, "terminal result schema")
    _require(nested["threshold_evaluation"]["fields"] == {
        "all_hard_thresholds_pass": "boolean",
        "cross_lane_record_count_maximum": "integer_maximum_check",
        "invalid_or_non_finite_value_count_maximum": "integer_maximum_check",
        "normalization_rejection_count_maximum": "integer_maximum_check",
        "positive_centered_realized_score_count_maximum": "integer_maximum_check",
        "rank_margin_violation_count_maximum": "integer_maximum_check",
        "saturation_event_count_maximum": "integer_maximum_check",
        "top_key_matching_fraction_minimum": "fraction_minimum_check",
        "top_key_mismatch_count_maximum": "integer_maximum_check",
        "unique_oracle_positive_margin_preserved_fraction_minimum": "fraction_minimum_check",
    }, "threshold-evaluation result schema")

    invariants = schema.get("nullability_and_invariants", {})
    _require(invariants.get("rank_margin_empty") == "minimum_realized_margin_q12_20_lsb and preserved_positive_margin_fraction are both null if and only if unique_oracle_top_row_count is zero", "rank-margin nullability")
    _require(invariants.get("top_key_empty") == "matching_fraction is null if and only if row_count is zero", "top-key nullability")
    terminal_rules = schema.get("terminal_state_rules", {})
    _require(terminal_rules.get("NO_EXECUTION_TERMINAL") == {
        "invocation_count_performed": 0,
        "metrics": None,
        "metrics_published": False,
        "reason_code": "NO_EXECUTION_AUTHORITY",
        "threshold_evaluation": None,
        "thresholds_evaluated": False,
    }, "no-execution terminal schema")
    _require(terminal_rules.get("SUCCEEDED_TERMINAL", {}).get("threshold_evaluation") == "threshold_evaluation object with all_hard_thresholds_pass=true", "success terminal threshold schema")
    _require(terminal_rules.get("FAILED_TERMINAL", {}).get("hard_threshold_failure", {}).get("threshold_evaluation") == "threshold_evaluation object with all_hard_thresholds_pass=false", "failed terminal threshold schema")
    _require(terminal_rules.get("FAILED_TERMINAL", {}).get("pre_metric_failure", {}).get("metrics") is None, "pre-metric failure null metrics")

    threshold_rules = schema.get("threshold_rules", {})
    _require(threshold_rules.get("fraction_minimum") == "pass is false for null actual; otherwise pass is actual.numerator * limit.denominator >= limit.numerator * actual.denominator", "fraction pass schema")
    _require(threshold_rules.get("integer_maximum") == "pass is actual <= limit", "integer pass schema")


def _verify_invocations(package: dict[str, Any], spec_sha256: str) -> None:
    invocations = package["invocations"]
    _require(isinstance(invocations, list) and len(invocations) == 2, "invocation count")
    for lane, invocation in zip(EXPECTED_LANES, invocations, strict=True):
        _require(set(invocation) == {"descriptor", "invocation_sha256"}, f"{lane['label']} invocation keys")
        descriptor = invocation["descriptor"]
        expected_descriptor = {
            "authoritative_tensors_sha256": _object_sha256(lane["authoritative_tensors"]),
            "environment": {
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
                "TZ": "UTC",
            },
            "evaluator_id": EVALUATOR_ID,
            "evaluator_spec_sha256": spec_sha256,
            "generation_id": lane["generation_id"],
            "input_metadata_sha256": lane["metadata"]["sha256"],
            "input_tensor_sha256": lane["tensor_bundle"]["sha256"],
            "invocation_cardinality": 1,
            "lane_label": lane["label"],
            "model_identity_sha256": lane["model_identity_sha256"],
            "namespace_label": lane["namespace_label"],
            "normalization_contract_sha256": EXPECTED_NORMALIZATION_BINDINGS["contract"]["sha256"],
            "output_path": lane["output_path"],
            "package_id": PACKAGE_ID,
            "representation_contract_sha256": EXPECTED_REPRESENTATION_BINDINGS["contract"]["sha256"],
            "terminal_no_retry_replay_resume_repair": True,
            "toolchain": {
                "implementation": "CPython",
                "version": "3.13.5",
            },
        }
        _require(descriptor == expected_descriptor, f"{lane['label']} invocation descriptor")
        _require(invocation["invocation_sha256"] == _object_sha256(descriptor), f"{lane['label']} invocation hash")


def _verify_package(package: dict[str, Any]) -> None:
    _require(set(package) == {
        "artifact_kind",
        "authority",
        "claim_boundary",
        "evaluation_contract",
        "evaluator",
        "input_set",
        "invocations",
        "isolation",
        "normalization_bindings",
        "output_schema",
        "package_id",
        "package_verifier",
        "prohibited_actions",
        "representation_bindings",
        "schema_version",
        "terminal_semantics",
    }, "package keys")
    _require(package["artifact_kind"] == "qk_bfp8_e16_head64_v1_c02_score_path_pre_execution_package", "artifact kind")
    _require(package["authority"] == AUTHORITY, "package authority")
    _require(package["claim_boundary"] == CLAIM_BOUNDARY, "package claim boundary")
    _require(package["package_id"] == PACKAGE_ID, "package id")
    _require(package["schema_version"] == 1, "package schema version")
    expected_package_verifier = {
        "authority": AUTHORITY,
        "byte_count": VERIFIER.stat().st_size,
        "path": str(VERIFIER.relative_to(ROOT)),
        "sha256": _sha256_file(VERIFIER),
    }
    _verify_binding(package["package_verifier"], expected_package_verifier, "package verifier")

    _require(package["normalization_bindings"] == EXPECTED_NORMALIZATION_BINDINGS, "normalization binding set")
    for label, binding in EXPECTED_NORMALIZATION_BINDINGS.items():
        _verify_binding(package["normalization_bindings"][label], binding, f"normalization {label}")
    _verify_normalization_semantics()

    _require(package["representation_bindings"] == EXPECTED_REPRESENTATION_BINDINGS, "representation binding set")
    for label, binding in EXPECTED_REPRESENTATION_BINDINGS.items():
        _verify_binding(package["representation_bindings"][label], binding, f"representation {label}")
    _verify_representation_semantics()

    input_set = package["input_set"]
    _require(set(input_set) == {"lanes", "sealed_set_id", "shared"}, "input-set keys")
    _require(input_set["sealed_set_id"] == "w4a8-c02-attention-substage-trace-v2", "sealed input-set id")
    _require(input_set["shared"] == EXPECTED_SHARED_INPUTS, "shared input bindings")
    for label, binding in EXPECTED_SHARED_INPUTS.items():
        _verify_binding(input_set["shared"][label], binding, f"shared input {label}")
    _require(input_set["lanes"] == EXPECTED_LANES, "lane input bindings")
    for lane in input_set["lanes"]:
        _verify_binding(lane["metadata"], lane["metadata"], f"{lane['label']} metadata")
        _verify_binding(lane["tensor_bundle"], lane["tensor_bundle"], f"{lane['label']} tensor bundle")
        _verify_input_metadata(lane)

    _verify_evaluation_contract(package["evaluation_contract"])
    spec_sha256 = _object_sha256(package["evaluation_contract"])
    _require(spec_sha256 == EXPECTED_EVALUATOR_SPEC_SHA256, "evaluator spec identity")
    _require(package["evaluator"] == {
        "evaluator_id": EVALUATOR_ID,
        "implementation_boundary": "The checksum-bound evaluation contract is the evaluator identity; no executable evaluator or execution authority is included in this package.",
        "spec_sha256": spec_sha256,
        "toolchain": {
            "canonical_json_profile": "python-json-sort-keys-compact-ascii-no-nan-v1",
            "implementation": "CPython",
            "version": "3.13.5",
        },
    }, "evaluator identity")
    _require(platform.python_implementation() == "CPython", "verifier Python implementation")
    _require(platform.python_version() == "3.13.5", "verifier Python version")
    _verify_invocations(package, spec_sha256)
    _verify_output_schema(package["output_schema"])

    isolation = package["isolation"]
    _require(isolation == {
        "cross_lane_reads_or_writes_forbidden": True,
        "execution_namespace_must_be_absent_during_static_verification": True,
        "lane_outputs": {
            "Base": OUTPUT_ROOT + "/base/result.json",
            "checkpoint-176": OUTPUT_ROOT + "/checkpoint-176/result.json",
        },
        "output_root": OUTPUT_ROOT,
        "partial_or_cross_lane_result_substitution_forbidden": True,
    }, "isolation contract")
    output_root = ROOT / isolation["output_root"]
    _require(not output_root.exists(), "execution namespace already exists")
    lane_outputs = list(isolation["lane_outputs"].values())
    _require(len(lane_outputs) == len(set(lane_outputs)), "lane outputs alias")
    for lane_output in lane_outputs:
        parts = Path(lane_output).parts
        _require("candidate" not in parts and "attempt" not in parts, "forbidden output path component")
        _require(Path(lane_output).is_relative_to(Path(OUTPUT_ROOT)), "lane output escapes output root")

    _require(package["terminal_semantics"] == {
        "authority_cardinality_in_this_package": 0,
        "first_record_immutability": "the first terminal record for a lane is final and may never be replaced or amended",
        "lane_independence": "Base and checkpoint-176 have distinct generation ids, invocation hashes, namespaces, outputs, and terminal outcomes; neither may satisfy or mask the other",
        "planned_external_invocation_cardinality_per_lane": 1,
        "terminal_outcomes": [
            "SUCCEEDED_TERMINAL",
            "FAILED_TERMINAL",
            "NO_EXECUTION_TERMINAL",
        ],
        "unconditional_prohibitions": [
            "retry",
            "replay",
            "resume",
            "repair",
            "recalibration",
            "regeneration",
            "authority reuse",
            "partial-result substitution",
        ],
    }, "terminal semantics")
    _require(package["prohibited_actions"] == [
        "load or execute a model",
        "execute or deserialize the sealed fixed-input tensor bundles",
        "compute evaluation scores or calibration statistics",
        "create a candidate, attempt, or execution namespace",
        "consume, mint, or reuse execution authority",
        "retry, replay, resume, repair, recalibrate, or regenerate under any outcome",
        "modify RTL, specifications, manifests, U280 configuration, constraints, or stage state",
    ], "prohibited actions")


def _expect_reject(package: dict[str, Any], label: str) -> None:
    try:
        _verify_package(package)
    except (VerificationError, KeyError, TypeError, ValueError):
        return
    raise VerificationError(f"negative mutation was accepted: {label}")


def _run_negative_tests(package: dict[str, Any]) -> None:
    mutations: list[tuple[str, dict[str, Any]]] = []

    mutated = copy.deepcopy(package)
    mutated["authority"] = "EXECUTION_AUTHORITY"
    mutations.append(("authority_marker_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["normalization_bindings"]["contract"]["sha256"] = "0" * 64
    mutations.append(("normalization_hash_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["representation_bindings"]["contract"]["sha256"] = "0" * 64
    mutations.append(("representation_hash_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["input_set"]["lanes"][0]["model_identity_sha256"] = "0" * 64
    mutations.append(("lane_model_identity_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["evaluation_contract"]["authoritative_tensor_mapping"]["roles"]["bf16_oracle_scores"]["tensor_name"] = "bf16.qk_centered_scores"
    mutations.append(("oracle_tensor_name_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["evaluation_contract"]["authoritative_tensor_mapping"]["roles"]["realized_key_source"]["shape"] = [1, 14, 41, 64]
    mutations.append(("realized_key_shape_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["input_set"]["lanes"][0]["authoritative_tensors"]["realized_query_source"]["sha256"] = "0" * 64
    mutations.append(("lane_tensor_record_hash_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["evaluation_contract"]["authoritative_tensor_mapping"]["causal_validity"] = "all key slots are valid"
    mutations.append(("causal_mask_rule_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["isolation"]["lane_outputs"]["checkpoint-176"] = mutated["isolation"]["lane_outputs"]["Base"]
    mutations.append(("cross_lane_output_alias", mutated))

    mutated = copy.deepcopy(package)
    mutated["invocations"][0]["invocation_sha256"] = "0" * 64
    mutations.append(("invocation_hash_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["evaluation_contract"]["thresholds"]["top_key_matching_fraction_minimum"] = {"denominator": 10, "numerator": 9}
    mutations.append(("top_key_threshold_relaxation", mutated))

    mutated = copy.deepcopy(package)
    mutated["evaluation_contract"]["comparison_metrics"]["rank_margin"]["accounting"] = "singleton rows are unique-oracle rows with an infinite realized margin"
    mutations.append(("singleton_rank_margin_rule_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["evaluation_contract"]["comparison_metrics"]["rank_margin"]["empty_set_encoding"] = "empty populations pass vacuously"
    mutations.append(("empty_fraction_rule_mutation", mutated))

    mutated = copy.deepcopy(package)
    del mutated["output_schema"]["nested_objects"]["input_bindings"]["fields"]["representation_contract_sha256"]
    mutations.append(("nested_input_schema_field_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["output_schema"]["nested_objects"]["rank_margin_metrics"]["fields"]["minimum_realized_margin_q12_20_lsb"] = "integer"
    mutations.append(("rank_margin_nullability_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["output_schema"]["fraction_encoding"]["threshold_on_null_actual"] = "pass must be true"
    mutations.append(("null_fraction_threshold_pass_mutation", mutated))

    mutated = copy.deepcopy(package)
    mutated["terminal_semantics"]["unconditional_prohibitions"].remove("retry")
    mutations.append(("terminal_retry_permission", mutated))

    mutated = copy.deepcopy(package)
    mutated["output_schema"]["terminal_state_rules"]["NO_EXECUTION_TERMINAL"]["metrics"] = "metrics object"
    mutations.append(("terminal_state_schema_mutation", mutated))

    mutated = copy.deepcopy(package)
    del mutated["output_schema"]
    mutations.append(("missing_required_key", mutated))

    _require([label for label, _ in mutations] == NEGATIVE_TESTS, "negative-test label drift")

    for label, candidate in mutations:
        _expect_reject(candidate, label)


def _verify_proof(proof: dict[str, Any], package_sha256: str) -> None:
    _require(set(proof) == {
        "artifact_kind",
        "authority",
        "claim_boundary",
        "negative_tests",
        "package",
        "schema_version",
        "static_observations",
        "verifier",
    }, "proof keys")
    _require(proof["artifact_kind"] == "qk_bfp8_e16_head64_v1_c02_score_path_pre_execution_proof", "proof kind")
    _require(proof["authority"] == AUTHORITY, "proof authority")
    _require(proof["claim_boundary"] == CLAIM_BOUNDARY, "proof claim boundary")
    _require(proof["schema_version"] == 1, "proof schema version")
    _require(proof["package"] == {
        "authority": AUTHORITY,
        "path": PACKAGE_REL,
        "sha256": package_sha256,
    }, "proof package binding")
    _require(proof["verifier"] == {
        "authority": AUTHORITY,
        "path": str(VERIFIER.relative_to(ROOT)),
        "sha256": _sha256_file(VERIFIER),
    }, "proof verifier binding")
    _require(proof["negative_tests"] == {
        "count": len(NEGATIVE_TESTS),
        "expected_result": "all mutations rejected before eligibility",
        "labels": NEGATIVE_TESTS,
    }, "proof negative tests")
    _require(proof["static_observations"] == {
        "execution_namespace_present": False,
        "fixed_input_payload_deserialized": False,
        "invocation_count_performed": 0,
        "model_or_evaluator_execution_performed": False,
        "score_count_computed": 0,
        "verifier_operation": "canonical JSON validation, semantic checks, byte counts, SHA-256 streaming, namespace absence checks, and in-memory negative mutations only",
    }, "proof observations")


def main() -> int:
    _verify_no_execution_surface()
    package = _load_json(PACKAGE, canonical=True)
    _verify_sidecar(PACKAGE_SIDECAR, PACKAGE, PACKAGE_REL)
    _verify_package(package)
    _run_negative_tests(package)

    package_sha256 = _sha256_file(PACKAGE)
    proof = _load_json(PROOF, canonical=True)
    _verify_sidecar(PROOF_SIDECAR, PROOF, PROOF_REL)
    _verify_proof(proof, package_sha256)

    print(
        "STATIC_VALID_NO_EXECUTION_AUTHORITY "
        f"package_sha256={package_sha256} "
        f"invocations={len(package['invocations'])} "
        f"negative_rejections={len(NEGATIVE_TESTS)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"STATIC_INVALID_NO_EXECUTION_AUTHORITY: {exc}", file=sys.stderr)
        raise SystemExit(1)
