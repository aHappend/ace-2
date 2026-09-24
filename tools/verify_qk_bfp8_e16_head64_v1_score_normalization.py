#!/usr/bin/env python3
"""NO_EXECUTION_AUTHORITY static verifier for score-pair normalization."""

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
CONTRACT_PATH = ROOT / "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_CONTRACT.json"
CONTRACT_SHA_PATH = CONTRACT_PATH.with_suffix(CONTRACT_PATH.suffix + ".sha256")
REFERENCE_PATH = ROOT / "reference/qk_bfp8_e16_head64_v1_score_normalization.py"
PRODUCER_PATH = ROOT / "reference/qk_bfp8_e16_head64_v1.py"
READINESS_PATH = ROOT / "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_READINESS.json"
READINESS_SHA_PATH = READINESS_PATH.with_suffix(READINESS_PATH.suffix + ".sha256")

PREREQUISITES = {
    "authoritative_softmax_scope": {
        "path": "design/CHIP_SCOPE.json",
        "sha256": "a5d61c40c89988a53fd3d039da48bd3de26d60f5d6d0bf5cce1e60176d4dcb2f",
    },
    "pure_reference": {
        "path": "reference/qk_bfp8_e16_head64_v1.py",
        "sha256": "bfb8b0d95ef8a8b939694fe8fec710d4acb1abff8b38b4bbbb4a43bdea5bb704",
    },
    "qualifying_fresh_l2_review": {
        "path": "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/c4be36371316/round-0002.json",
        "sha256": "c76b51059c853d361d00e72ccbc8af9216175bb1be6471811efe5f74ee3ed046",
    },
    "readiness_package": {
        "path": "reference/QK_BFP8_E16_HEAD64_V1_READINESS.json",
        "sha256": "400d4a4a8b627cd71e930f670852b89e5deead83f14be3f528e752f6d1705fa7",
    },
    "static_verifier": {
        "path": "tools/verify_qk_bfp8_e16_head64_v1_readiness.py",
        "sha256": "ea40e12f39351fe6240d861a68488a4bc22517dab72434762f05a8bdf88fdd2e",
    },
}
EXPECTED_CONTRACT_SHA256 = "539ff95d8f1ad253880c9999fa59e6ac41840761908b188cf2c4bce09f3c1b01"
EXPECTED_ARTIFACT_KIND = "qk_bfp8_e16_head64_v1_score_normalization_prerequisite_contract"
EXPECTED_CONTRACT_ID = "QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_V2"
EXPECTED_CLAIM_BOUNDARY = {
    "calibration_or_scoring_performed": False,
    "candidate_or_execution_namespace_created": False,
    "fixed_input_or_model_execution_performed": False,
    "generation_or_repair_replay_performed": False,
    "rtl_spec_manifest_or_u280_modified": False,
    "stage_transition_authorized": False,
}
CONTRACT_KEYS = frozenset(
    {
        "api",
        "artifact_kind",
        "authority",
        "claim_boundary",
        "comparison",
        "contract_id",
        "input",
        "normalization",
        "objective",
        "packing",
        "prerequisite_bindings",
        "prohibited_actions",
        "proof_widths",
        "realization",
        "schema_version",
    }
)
READINESS_KEYS = frozenset(
    {
        "artifact_kind",
        "authority",
        "claim_boundary",
        "contract",
        "prerequisite_bindings",
        "schema_version",
        "static_artifacts",
    }
)


class StaticInvalidity(RuntimeError):
    def __init__(self, taxonomy: str, message: str):
        super().__init__(message)
        self.taxonomy = taxonomy


def require(condition: bool, taxonomy: str, message: str) -> None:
    if not condition:
        raise StaticInvalidity(taxonomy, message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), "missing_binding", f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StaticInvalidity("malformed_artifact", f"cannot load JSON {path}: {error}") from error
    require(type(value) is dict, "malformed_artifact", f"JSON root is not an object: {path}")
    return value


def canonical_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def require_exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    require(type(value) is dict, "schema_mismatch", f"{label} is not an object")
    actual = frozenset(value)
    require(
        actual == expected,
        "schema_mismatch",
        f"{label} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}",
    )


def verify_named_sidecar(sidecar_path: Path, subject_path: Path) -> None:
    try:
        lines = sidecar_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise StaticInvalidity("missing_binding", f"cannot read sidecar {sidecar_path}: {error}") from error
    require(
        len(lines) == 2 and lines[0] == AUTHORITY,
        "authority_violation",
        f"checksum sidecar lacks exact authority marker: {sidecar_path}",
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


def verify_prerequisites() -> None:
    for label, binding in PREREQUISITES.items():
        path = resolve_path(binding["path"])
        require(
            sha256_file(path) == binding["sha256"],
            "predecessor_mismatch",
            f"{label} prerequisite hash differs",
        )

    chip_scope = load_json(resolve_path(PREREQUISITES["authoritative_softmax_scope"]["path"]))
    numerical_behavior = chip_scope.get("numerical_behavior", {})
    attention_softmax = numerical_behavior.get("attention_softmax", {})
    require(
        attention_softmax.get("logit_format_after_scale") == "signed_int64_container_for_Q12_20",
        "predecessor_mismatch",
        "authoritative attention-softmax logit format differs",
    )
    require(
        attention_softmax.get("mask_value") == "invalid_or_future_keys_do_not_enter_the_recurrence",
        "predecessor_mismatch",
        "authoritative attention-softmax mask semantics differ",
    )
    require(
        attention_softmax.get("score_requantization_metadata")
        == "static_Q_and_K_Scale32_significands_and_exponents_composed_into_one_signed_Q12_20_logit",
        "predecessor_mismatch",
        "authoritative attention-softmax requantization metadata differs",
    )
    historical_q6_9 = numerical_behavior.get("rope", {}).get(
        "projection_shadow_staged_attention_contract", {}
    )
    require(
        historical_q6_9.get("score_format") == "materialized_signed_q6_9_in_s16"
        and historical_q6_9.get("status") == "rejected_focused_discriminator_and_paired_smoke_failed",
        "predecessor_mismatch",
        "historical Q6.9 rejection evidence differs",
    )

    readiness = load_json(resolve_path(PREREQUISITES["readiness_package"]["path"]))
    require(readiness.get("authority") == AUTHORITY, "authority_violation", "predecessor readiness authority differs")
    artifacts = readiness.get("readiness_artifacts", {})
    require(
        artifacts.get("pure_reference", {}).get("sha256") == PREREQUISITES["pure_reference"]["sha256"],
        "predecessor_mismatch",
        "predecessor readiness pure-reference binding differs",
    )
    require(
        artifacts.get("static_verifier", {}).get("sha256") == PREREQUISITES["static_verifier"]["sha256"],
        "predecessor_mismatch",
        "predecessor readiness verifier binding differs",
    )

    review = load_json(resolve_path(PREREQUISITES["qualifying_fresh_l2_review"]["path"]))
    require(review.get("producer_role") == "reviewer", "predecessor_mismatch", "qualifying review producer differs")
    require(review.get("review", {}).get("status") == "done", "predecessor_mismatch", "qualifying review is not done")
    reason = review.get("review", {}).get("reason", "")
    for phrase in (
        "STATIC_VALID_NO_EXECUTION_AUTHORITY",
        "65,280 finite BF16 cases",
        "no RTL correctness is claimed",
    ):
        require(phrase in reason, "predecessor_mismatch", f"qualifying review lacks phrase: {phrase}")


def verify_contract() -> dict[str, Any]:
    require(
        sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256,
        "binding_mismatch",
        "normalization contract hash differs",
    )
    contract = load_json(CONTRACT_PATH)
    require(
        CONTRACT_PATH.read_bytes() == canonical_file_bytes(contract),
        "malformed_artifact",
        "normalization contract JSON is not canonical",
    )
    verify_named_sidecar(CONTRACT_SHA_PATH, CONTRACT_PATH)
    require_exact_keys(contract, CONTRACT_KEYS, "contract")
    require(contract["artifact_kind"] == EXPECTED_ARTIFACT_KIND, "schema_mismatch", "artifact_kind differs")
    require(contract["contract_id"] == EXPECTED_CONTRACT_ID, "semantic_mismatch", "contract_id differs")
    require(contract["schema_version"] == 2, "schema_mismatch", "schema_version differs")
    require(contract["authority"] == AUTHORITY, "authority_violation", "contract authority differs")
    require(contract["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "authority_violation", "claim boundary differs")
    require(contract["prerequisite_bindings"] == PREREQUISITES, "binding_mismatch", "prerequisite set differs")

    score_input = contract["input"]
    require(score_input["minimum_row_length"] == 1, "semantic_mismatch", "minimum row length differs")
    require(score_input["maximum_row_length"] == 512, "semantic_mismatch", "maximum row length differs")
    require(score_input["score_mantissa"] == {
        "canonical_range": [-1032256, 1032256],
        "minimum_exact_signed_width": 21,
        "storage_width": 32,
    }, "semantic_mismatch", "score mantissa contract differs")
    require(score_input["score_exponent"] == {
        "canonical_nonzero_range": [-278, 244],
        "minimum_exact_signed_width": 10,
        "storage_width": 16,
    }, "semantic_mismatch", "score exponent contract differs")
    require("1/sqrt(64)=1/8" in score_input["head_scale"], "semantic_mismatch", "HEAD64 scale differs")

    normalization = contract["normalization"]
    for phrase, field in (
        ("minimum score_exponent among valid nonzero pairs", "common_exponent"),
        ("global maximum across all valid keys", "row_max"),
        ("lowest key index", "argmax_tie"),
        ("signed 544-bit or wider", "row_max_centering"),
        ("validated but excluded", "mask_order"),
        ("Reject", "all_masked_row"),
    ):
        require(phrase in normalization[field], "semantic_mismatch", f"normalization.{field} differs")

    realization = contract["realization"]
    require(
        PREREQUISITES["authoritative_softmax_scope"]["sha256"] in realization["authoritative_binding"]
        and "signed_int64_container_for_Q12_20" in realization["authoritative_binding"],
        "binding_mismatch",
        "authoritative softmax format binding differs",
    )
    require(realization["format"] == "signed-int64-container Q12.20 attention-softmax logit", "semantic_mismatch", "realization format differs")
    require("common_exponent+17" in realization["formula"], "semantic_mismatch", "realization formula differs")
    require(realization["valid_output_range"] == [-(1 << 63), 0], "semantic_mismatch", "realization range differs")
    require("sole rounding boundary" in realization["rounding_boundary"], "semantic_mismatch", "rounding boundary differs")
    require("retained_lsb" in realization["guard_round_sticky"], "semantic_mismatch", "ties-to-even rule differs")
    require("bitmap" in realization["masked_output"], "semantic_mismatch", "mask authority differs")

    packing = contract["packing"]
    require(packing["input_frame"]["header_bytes"] == 16, "packing_mismatch", "input header bytes differ")
    require(packing["output_frame"]["header_bytes"] == 24, "packing_mismatch", "output header bytes differ")
    require(packing["output_frame"]["masked_score"] == 0, "packing_mismatch", "masked score differs")
    require("score_bytes=8" in packing["output_frame"]["layout"], "packing_mismatch", "output score width differs")
    require("QKSPIN01" in packing["input_frame"]["layout"], "packing_mismatch", "input magic differs")
    require("QKSPOU01" in packing["output_frame"]["layout"], "packing_mismatch", "output magic differs")
    require(
        "unused high bits" in packing["common"]["mask_tail"].lower(),
        "packing_mismatch",
        "mask tail rule differs",
    )

    widths = contract["proof_widths"]
    require(widths["score_exponent_span"] == 522, "width_mismatch", "exponent span differs")
    for width, field in ((543, "aligned_score"), (544, "centered_score"), (805, "realization_intermediate")):
        require(str(width) in widths[field], "width_mismatch", f"{field} width differs")
    return contract


def verify_reference_purity() -> None:
    source = REFERENCE_PATH.read_text(encoding="utf-8")
    require(AUTHORITY in source, "authority_violation", "normalization reference lacks authority marker")
    tree = ast.parse(source, filename=str(REFERENCE_PATH))
    forbidden_calls = {
        "compile", "eval", "exec", "input", "open", "read_bytes", "read_text",
        "system", "popen", "write_bytes", "write_text", "__import__",
    }
    for node in ast.walk(tree):
        require(not isinstance(node, (ast.Import, ast.ImportFrom)), "authority_violation", "normalization reference imports a module")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            require(name not in forbidden_calls, "authority_violation", f"normalization reference contains forbidden call: {name}")


def verify_no_reference_bytecode() -> None:
    bytecode = sorted(REFERENCE_PATH.parent.glob(f"__pycache__/{REFERENCE_PATH.stem}.*.pyc"))
    require(not bytecode, "artifact_set_mismatch", f"unexpected normalization reference bytecode: {bytecode}")


def load_reference() -> Any:
    specification = importlib.util.spec_from_file_location("qk_score_normalization", REFERENCE_PATH)
    require(specification is not None and specification.loader is not None, "reference_mismatch", "cannot create reference module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_producer() -> Any:
    specification = importlib.util.spec_from_file_location("qk_score_producer", PRODUCER_PATH)
    require(specification is not None and specification.loader is not None, "reference_mismatch", "cannot create producer module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expect_reference_rejection(reference: Any, operation: Any, label: str) -> None:
    try:
        operation()
    except reference.ReferenceViolation:
        return
    raise StaticInvalidity("reference_mismatch", f"malformed request was accepted: {label}")


def verify_width_proofs() -> None:
    mantissa_bound = 1032256
    exponent_span = 244 - (-278)
    aligned_bound = mantissa_bound << exponent_span
    require(exponent_span == 522, "width_mismatch", "exponent span proof differs")
    require(-(1 << 542) <= -aligned_bound and aligned_bound <= (1 << 542) - 1, "width_mismatch", "aligned bound exceeds signed 543")
    require(aligned_bound > (1 << 541) - 1, "width_mismatch", "signed 542 unexpectedly contains aligned bound")
    centered_conservative = 2 * aligned_bound
    require(-(1 << 543) <= -centered_conservative and centered_conservative <= (1 << 543) - 1, "width_mismatch", "conservative centered bound exceeds signed 544")
    realization_bound = centered_conservative << (244 + 17)
    require(-(1 << 804) <= -realization_bound and realization_bound <= (1 << 804) - 1, "width_mismatch", "realization bound exceeds signed 805")


def verify_reference_vectors(reference: Any, producer: Any) -> tuple[int, int]:
    require(reference.AUTHORITY == AUTHORITY, "authority_violation", "normalization reference authority differs")
    require(reference.CONTRACT_ID == EXPECTED_CONTRACT_ID, "reference_mismatch", "normalization reference contract id differs")
    require(reference.SCHEMA_VERSION == 2, "reference_mismatch", "normalization reference schema differs")

    comparisons = (
        (((0, -12), (0, 244)), 0),
        (((1, 0), (2, -1)), 0),
        (((4, -2), (1, 0)), 0),
        (((-1, 244), (0, 0)), -1),
        (((0, 0), (-1, -278)), 1),
        (((-1, 0), (-2, 0)), 1),
        (((1032256, -278), (1, 244)), -1),
        (((-1032256, -278), (-1, 244)), 1),
    )
    for (left, right), expected in comparisons:
        require(reference.compare_score_pairs(left, right) == expected, "comparison_mismatch", f"comparison differs: {left} versus {right}")

    vectors = (
        (
            ((1, 0), (1, 1), (-1, 0)),
            (True, True, True),
            ((-131072, 0, -393216), 0, 1),
        ),
        (
            ((1, 0), (2, -1), (4, -2)),
            (True, True, True),
            ((0, 0, 0), -2, 0),
        ),
        (
            ((100, 0), (1, 0)),
            (False, True),
            ((0, 0), 0, 1),
        ),
        (
            ((0, 0), (0, 0), (0, 0)),
            (True, True, False),
            ((0, 0, 0), 0, 0),
        ),
        (
            ((0, 0), (-1, -7), (-3, -7), (-5, -7)),
            (True, True, True, True),
            ((0, -1024, -3072, -5120), -7, 0),
        ),
        (
            ((0, 0), (-1032256, 244)),
            (True, True),
            ((0, -(1 << 63)), 244, 0),
        ),
    )
    for score_pairs, valid_mask, expected in vectors:
        require(reference.normalize_score_row(score_pairs, valid_mask) == expected, "normalization_mismatch", f"normalization vector differs: {score_pairs}")

    score_pairs = ((0, 0), (-1, -1), (1, 1))
    valid_mask = (True, False, True)
    expected_input_hex = (
        "514b5350494e30310200000003000600"
        "000000000000ffffffffffff01000000010005"
    )
    packed_input = reference.pack_input_row(score_pairs, valid_mask)
    require(packed_input.hex() == expected_input_hex, "packing_mismatch", "canonical input bytes differ")
    require(reference.unpack_input_row(packed_input) == (score_pairs, valid_mask), "packing_mismatch", "input round trip differs")

    expected_output_hex = (
        "514b53504f55303102000000030008000100020000000000"
        "0000fcffffffffff0000000000000000000000000000000005"
    )
    packed_output = reference.pack_normalized_row(score_pairs, valid_mask)
    require(packed_output.hex() == expected_output_hex, "packing_mismatch", "canonical output bytes differ")
    require(
        reference.unpack_normalized_row(packed_output) == ((-262144, 0, 0), valid_mask, 1, 2),
        "packing_mismatch",
        "output round trip differs",
    )

    one = 0x3F80
    negative_one = 0xBF80
    zero = 0x0000
    q_head = producer.encode_head((one, one) + (zero,) * 62)
    k_head = producer.encode_head((one, negative_one) + (zero,) * 62)
    cancelling_score = producer.dot_product(q_head[0], q_head[1], k_head[0], k_head[1])
    require(cancelling_score == (0, -12), "producer_boundary_mismatch", "cancelling head no longer reaches producer zero with summed exponent")
    require(
        reference.canonicalize_producer_score_pair(cancelling_score) == (0, 0),
        "producer_boundary_mismatch",
        "cancelling producer zero was not canonicalized",
    )
    require(
        reference.normalize_score_row((cancelling_score,), (True,)) == ((0,), 0, 0),
        "producer_boundary_mismatch",
        "cancelling producer zero was rejected by normalization",
    )
    cancelling_frame = reference.pack_input_row((cancelling_score,), (True,))
    require(
        reference.unpack_input_row(cancelling_frame) == (((0, 0),), (True,)),
        "producer_boundary_mismatch",
        "cancelling producer zero was not packed canonically",
    )

    rejections = (
        (lambda: reference.canonicalize_producer_score_pair((0, -279)), "zero exponent outside producer range"),
        (lambda: reference.compare_score_pairs((1032257, 0), (0, 0)), "mantissa overflow"),
        (lambda: reference.compare_score_pairs((1, -279), (0, 0)), "exponent underflow"),
        (lambda: reference.normalize_score_row(((0, 0),), (False,)), "all masked"),
        (lambda: reference.normalize_score_row(((0, -279), (1, 0)), (False, True)), "masked malformed pair"),
        (lambda: reference.normalize_score_row(((True, 0),), (True,)), "boolean mantissa"),
        (lambda: reference.normalize_score_row(((0, 0),), (1,)), "integer mask"),
        (lambda: reference.unpack_input_row(packed_input + b"\x00"), "input trailing byte"),
        (lambda: reference.unpack_input_row(packed_input[:-1] + b"\x85"), "nonzero mask tail"),
        (lambda: reference.unpack_input_row(packed_input[:20] + b"\x01\x00" + packed_input[22:]), "packed noncanonical zero"),
        (lambda: reference.unpack_normalized_row(packed_output + b"\x00"), "output trailing byte"),
        (lambda: reference.unpack_normalized_row(packed_output[:20] + b"\x01\x00\x00\x00" + packed_output[24:]), "nonzero reserved"),
        (lambda: reference.unpack_normalized_row(packed_output[:32] + (1).to_bytes(8, "little", signed=True) + packed_output[40:]), "masked score lacks zero filler"),
    )
    for operation, label in rejections:
        expect_reference_rejection(reference, operation, label)
    return len(comparisons) + len(vectors) + 3, len(rejections)


def expected_readiness_artifacts() -> dict[str, Any]:
    return {
        "pure_reference": {
            "authority": AUTHORITY,
            "path": "reference/qk_bfp8_e16_head64_v1_score_normalization.py",
            "sha256": sha256_file(REFERENCE_PATH),
        },
        "static_verifier": {
            "authority": AUTHORITY,
            "path": "tools/verify_qk_bfp8_e16_head64_v1_score_normalization.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def verify_readiness_schema(readiness: dict[str, Any]) -> None:
    require_exact_keys(readiness, READINESS_KEYS, "readiness")
    require(readiness["artifact_kind"] == "qk_bfp8_e16_head64_v1_score_normalization_readiness", "schema_mismatch", "readiness artifact_kind differs")
    require(readiness["schema_version"] == 2, "schema_mismatch", "readiness schema version differs")
    require(readiness["authority"] == AUTHORITY, "authority_violation", "readiness authority differs")
    require(readiness["claim_boundary"] == EXPECTED_CLAIM_BOUNDARY, "authority_violation", "readiness claim boundary differs")
    require(readiness["contract"] == {
        "authority": AUTHORITY,
        "contract_id": EXPECTED_CONTRACT_ID,
        "path": "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_CONTRACT.json",
        "sha256": EXPECTED_CONTRACT_SHA256,
    }, "binding_mismatch", "readiness contract binding differs")
    require(readiness["prerequisite_bindings"] == PREREQUISITES, "binding_mismatch", "readiness prerequisite set differs")
    require(readiness["static_artifacts"] == expected_readiness_artifacts(), "binding_mismatch", "readiness static artifact set differs")


def expect_schema_rejection(readiness: dict[str, Any], mutate: Any, label: str) -> None:
    candidate = copy.deepcopy(readiness)
    mutate(candidate)
    try:
        verify_readiness_schema(candidate)
    except StaticInvalidity:
        return
    raise StaticInvalidity("schema_mismatch", f"readiness schema mutation was accepted: {label}")


def verify_schema_mutations(readiness: dict[str, Any]) -> int:
    mutations = (
        (lambda value: value.pop("contract"), "missing contract"),
        (lambda value: value.__setitem__("unexpected", False), "extra top-level field"),
        (lambda value: value["claim_boundary"].__setitem__("fixed_input_or_model_execution_performed", True), "execution claim"),
        (lambda value: value["prerequisite_bindings"].pop("qualifying_fresh_l2_review"), "missing prerequisite"),
        (lambda value: value["static_artifacts"].pop("pure_reference"), "missing static artifact"),
        (lambda value: value["static_artifacts"]["static_verifier"].__setitem__("sha256", "0" * 64), "verifier substitution"),
        (lambda value: value["contract"].__setitem__("contract_id", "unexpected"), "wrong contract id"),
        (lambda value: value.__setitem__("schema_version", 3), "wrong schema version"),
    )
    for mutate, label in mutations:
        expect_schema_rejection(readiness, mutate, label)
    return len(mutations)


def verify_readiness() -> int:
    readiness = load_json(READINESS_PATH)
    require(READINESS_PATH.read_bytes() == canonical_file_bytes(readiness), "malformed_artifact", "readiness JSON is not canonical")
    verify_named_sidecar(READINESS_SHA_PATH, READINESS_PATH)
    verify_readiness_schema(readiness)
    return verify_schema_mutations(readiness)


def verify_static_package() -> dict[str, Any]:
    verify_prerequisites()
    verify_contract()
    verify_width_proofs()
    verify_no_reference_bytecode()
    verify_reference_purity()
    reference = load_reference()
    producer = load_producer()
    vector_count, rejection_count = verify_reference_vectors(reference, producer)
    verify_no_reference_bytecode()
    mutation_count = verify_readiness()
    return {
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "rejections": rejection_count,
        "schema_mutations": mutation_count,
        "vectors": vector_count,
    }


def main() -> int:
    try:
        result = verify_static_package()
    except StaticInvalidity as error:
        print(f"STATIC_INVALID_NO_EXECUTION_AUTHORITY taxonomy={error.taxonomy} detail={error}")
        return 1
    print(
        "STATIC_VALID_NO_EXECUTION_AUTHORITY "
        f"contract_sha256={result['contract_sha256']} "
        f"vectors={result['vectors']} "
        f"rejections={result['rejections']} "
        f"schema_mutations={result['schema_mutations']} "
        "prerequisites=5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
