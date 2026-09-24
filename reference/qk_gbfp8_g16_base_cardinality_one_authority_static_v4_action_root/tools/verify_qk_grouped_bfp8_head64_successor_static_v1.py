#!/usr/bin/env python3
"""Decisive static-only verifier for QK grouped-BFP8 successor V1."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "design/QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_CONTRACT.json"
REFERENCE_PATH = ROOT / "reference/qk_grouped_bfp8_head64_successor_static_v1.py"
SCHEMA_PATH = ROOT / "reference/QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_SCHEMA.json"
PACKAGE_PATH = ROOT / "reference/QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_PACKAGE.json"
PACKAGE_SHA_PATH = ROOT / "reference/QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_PACKAGE.json.sha256"
STATIC_MANIFEST_PATH = ROOT / "evidence/verification/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2/SHA256SUMS"

EXPECTED_STATIC_MANIFEST_SHA256 = "cb71a735884550fa6186bcb2623de9d1c8a378c1d4ef728a09ab306a685b4a50"
EXPECTED_ALTERNATIVES = {
    "QK_GBFP8_G16_E16_HEAD64_V1": (16, 4, 72, 19),
    "QK_GBFP8_G8_E16_HEAD64_V1": (8, 8, 80, 18),
}

sys.dont_write_bytecode = True


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_exact_keys(value: object, keys: set[str], context: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        fail(f"{context} exact-key schema mismatch")
    return value


def validate_binding(binding: object, context: str) -> dict:
    value = require_exact_keys(binding, {"byte_count", "path", "sha256"}, context)
    if type(value["byte_count"]) is not int or value["byte_count"] <= 0:
        fail(f"{context} byte_count invalid")
    if type(value["path"]) is not str or not value["path"]:
        fail(f"{context} path invalid")
    digest = value["sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        fail(f"{context} sha256 invalid")
    return value


def validate_package(package: object) -> dict:
    package = require_exact_keys(
        package,
        {
            "artifact_kind",
            "artifacts",
            "authority",
            "claim_boundary",
            "contract_id",
            "package_id",
            "schema_id",
            "schema_version",
        },
        "package",
    )
    if package["artifact_kind"] != "qk_grouped_bfp8_head64_successor_static_pre_execution_package":
        fail("package artifact_kind")
    if package["authority"] != "NO_EXECUTION_AUTHORITY":
        fail("package authority")
    if package["contract_id"] != "QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1":
        fail("package contract_id")
    if package["package_id"] != "QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_PACKAGE":
        fail("package package_id")
    if package["schema_id"] != "QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1_PACKAGE_SCHEMA":
        fail("package schema_id")
    if package["schema_version"] != 1:
        fail("package schema_version")
    artifacts = require_exact_keys(
        package["artifacts"], {"contract", "pure_reference", "schema", "static_verifier"}, "artifacts"
    )
    for name, binding in artifacts.items():
        validate_binding(binding, f"artifacts.{name}")
    claim = require_exact_keys(
        package["claim_boundary"],
        {"execution_performed", "model_or_tensor_accessed", "numerical_improvement_claimed", "rtl_or_stage_modified"},
        "claim_boundary",
    )
    if any(type(value) is not bool or value for value in claim.values()):
        fail("package claim boundary must contain exact false booleans")
    return package


def validate_contract(contract: object) -> dict:
    if type(contract) is not dict:
        fail("contract must be object")
    if contract.get("contract_id") != "QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1":
        fail("contract id")
    if contract.get("schema_version") != 1 or contract.get("authority") != "NO_EXECUTION_AUTHORITY":
        fail("contract schema or authority")
    alternatives = contract.get("alternatives")
    if type(alternatives) is not list or len(alternatives) != 2:
        fail("contract must define exactly two alternatives")
    seen = set()
    for alternative in alternatives:
        if type(alternative) is not dict:
            fail("alternative object")
        alternative_id = alternative.get("alternative_id")
        if alternative_id in seen or alternative_id not in EXPECTED_ALTERNATIVES:
            fail("alternative identity closure")
        seen.add(alternative_id)
        group_size, group_count, record_bytes, partial_width = EXPECTED_ALTERNATIVES[alternative_id]
        encoding = alternative.get("encoding", {})
        dot = alternative.get("groupwise_dot_product", {})
        packing = alternative.get("packing", {})
        if encoding.get("group_size_lanes") != group_size or encoding.get("group_count") != group_count:
            fail("group geometry closure")
        if group_size >= 64:
            fail("successor failed finer-than-64 requirement")
        if packing.get("record_bytes_per_head") != record_bytes:
            fail("record byte closure")
        if dot.get("group_partial_minimum_signed_width") != partial_width:
            fail("partial width closure")
    if seen != set(EXPECTED_ALTERNATIVES):
        fail("alternative set closure")
    widths = contract.get("arithmetic_width_closure", {})
    if widths.get("aligned_key_score", "").split()[1] != "543":
        fail("aligned score width closure")
    if widths.get("centered_key_score", "").split()[1] != "544":
        fail("centered width closure")
    if widths.get("q12_20_realization_intermediate", "").split()[1] != "805":
        fail("realization width closure")
    public_api = contract.get("accepted_boundary", {}).get("public_packing_api")
    if public_api != (
        "pack_normalized_row(query, keys, valid_mask, alternative_id) validates encoded heads and bitmap, "
        "then performs groupwise dot-product normalization internally; callers cannot supply derived scores, "
        "row_common_exponent, or argmax_index."
    ):
        fail("public packing API normalization closure")
    claim = contract.get("claim_boundary")
    if type(claim) is not dict or any(type(value) is not bool or value for value in claim.values()):
        fail("contract no-execution claim boundary")
    evidence = contract.get("accepted_c02_evidence", {})
    if evidence.get("base_first_terminal", {}).get("status") != "FAILED_TERMINAL":
        fail("Base terminal disposition")
    if evidence.get("base_result", {}).get("terminal_reason") != "HARD_THRESHOLD_FAILED":
        fail("Base result disposition")
    if evidence.get("checkpoint_176_c02", {}).get("disposition") != "UNMATERIALIZED_AND_UNAVAILABLE_UNDER_CONSUMED_BASE_GATED_AUTHORITY":
        fail("checkpoint-176 disposition")
    recommendation = contract.get("recommendation", {})
    if recommendation.get("alternative_id") != "QK_GBFP8_G16_E16_HEAD64_V1":
        fail("recommendation closure")
    if recommendation.get("numerical_claim") != "NONE_PRE_EXECUTION":
        fail("numerical claim closure")
    return contract


def expect_reject(function, *args) -> None:
    try:
        function(*args)
    except (AssertionError, ValueError, TypeError):
        return
    fail("mutation unexpectedly accepted")


def import_reference():
    spec = importlib.util.spec_from_file_location("qk_grouped_bfp8_static_ref", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        fail("cannot import pure reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bf16_fraction(bits: int) -> Fraction:
    sign = -1 if bits & 0x8000 else 1
    exponent = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    if exponent == 0xFF:
        raise ValueError("nonfinite")
    if exponent == 0:
        return Fraction(sign * fraction, 1 << 133)
    significand = sign * (128 + fraction)
    power = exponent - 134
    if power >= 0:
        return Fraction(significand << power, 1)
    return Fraction(significand, 1 << (-power))


def fraction_rne(value: Fraction) -> int:
    sign = -1 if value < 0 else 1
    value = abs(value)
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled > value.denominator or (doubled == value.denominator and (quotient & 1)):
        quotient += 1
    return sign * quotient


def canonical_encoded(mantissas: list[int], exponents: list[int]):
    return (tuple(mantissas), tuple(exponents))


def run_reference_vectors(ref) -> tuple[int, int]:
    checks = 0
    semantic_mutation_checks = 0
    for value in range(-255, 256):
        for shift in range(-12, 5):
            actual = ref.round_signed_pow2(value, shift)
            expected = fraction_rne(Fraction(value, 1) * (Fraction(2**shift, 1) if shift >= 0 else Fraction(1, 2 ** (-shift))))
            if actual != expected:
                fail("exhaustive small-domain RNE mismatch")
            checks += 1
    bf16_vectors = {
        0x0000: Fraction(0),
        0x8000: Fraction(0),
        0x0001: Fraction(1, 1 << 133),
        0x007F: Fraction(127, 1 << 133),
        0x0080: Fraction(128, 1 << 133),
        0x3F00: Fraction(1, 2),
        0x3F80: Fraction(1),
        0xBF80: Fraction(-1),
        0x7F7F: Fraction(255 << 120),
        0xFF7F: Fraction(-(255 << 120)),
    }
    for bits, expected in bf16_vectors.items():
        numerator, exponent = ref.decode_bf16_bits(bits)
        actual = Fraction(numerator << exponent, 1) if exponent >= 0 else Fraction(numerator, 1 << (-exponent))
        if actual != expected or actual != bf16_fraction(bits):
            fail("BF16 exact decode vector mismatch")
        checks += 1
    expect_reject(ref.decode_bf16_bits, 0x7F80)
    expect_reject(ref.decode_bf16_bits, 0x7FC1)
    expect_reject(ref.decode_bf16_bits, True)
    checks += 3

    for alternative_id, (group_size, group_count, record_bytes, _) in EXPECTED_ALTERNATIVES.items():
        zero = ref.encode_head_bf16((0x8000,) * 64, alternative_id)
        if zero != ((0,) * 64, (0,) * group_count):
            fail("zero canonical vector")
        minimum = ref.encode_head_bf16((0x0001,) * 64, alternative_id)
        maximum = ref.encode_head_bf16((0x7F7F,) * 64, alternative_id)
        if minimum[0] != (64,) * 64 or minimum[1] != (-139,) * group_count:
            fail("minimum-subnormal boundary vector")
        if maximum[0] != (64,) * 64 or maximum[1] != (122,) * group_count:
            fail("maximum-finite boundary vector")
        packed = ref.pack_head_record(minimum, alternative_id)
        if len(packed) != record_bytes or ref.unpack_head_record(packed, alternative_id) != minimum:
            fail("head packing round trip")
        expect_reject(ref.unpack_head_record, packed[:-1], alternative_id)
        expect_reject(ref.unpack_head_record, packed + b"\x00", alternative_id)
        reserved = bytearray(packed)
        reserved[0] = 0x80
        expect_reject(ref.unpack_head_record, bytes(reserved), alternative_id)
        denormal = canonical_encoded([1] * 64, [0] * group_count)
        expect_reject(ref.validate_encoded_head, denormal, alternative_id)
        bad_zero_exp = canonical_encoded([0] * 64, [1] + [0] * (group_count - 1))
        expect_reject(ref.validate_encoded_head, bad_zero_exp, alternative_id)
        bad_exp = canonical_encoded([64] * group_size + [0] * (64 - group_size), [123] + [0] * (group_count - 1))
        expect_reject(ref.validate_encoded_head, bad_exp, alternative_id)

        query_m = [0] * 64
        query_e = [0] * group_count
        key0_m = [0] * 64
        key1_m = [0] * 64
        key_e = [0] * group_count
        for group in range(group_count):
            lane = group * group_size
            query_m[lane] = 64
            key0_m[lane] = 64
            key1_m[lane] = 64 if group != group_count - 1 else -64
        query = canonical_encoded(query_m, query_e)
        key0 = canonical_encoded(key0_m, key_e)
        key1 = canonical_encoded(key1_m, key_e)
        scores, common, argmax, saturations = ref.normalize_score_row(query, (key0, key1), (True, True), alternative_id)
        if common != 0 or argmax != 0 or scores[0] != 0 or scores[1] >= 0 or saturations != 0:
            fail("adversarial cross-group ordering vector")

        tie_scores, tie_common, tie_argmax, tie_saturations = ref.normalize_score_row(query, (key0, key0), (True, True), alternative_id)
        if tie_scores != (0, 0) or tie_common != 0 or tie_argmax != 0 or tie_saturations != 0:
            fail("exact tie vector")
        masked_scores, _, masked_argmax, _ = ref.normalize_score_row(query, (key1, key0), (False, True), alternative_id)
        if masked_scores != (0, 0) or masked_argmax != 1:
            fail("authoritative mask vector")
        expect_reject(ref.normalize_score_row, query, (key0,), (False,), alternative_id)

        extreme_query_m = [0] * 64
        extreme_query_e = [0] * group_count
        extreme_key0_m = [0] * 64
        extreme_key1_m = [0] * 64
        extreme_key0_e = [0] * group_count
        extreme_key1_e = [0] * group_count
        extreme_query_m[0] = 127
        extreme_query_e[0] = 122
        extreme_key0_m[0] = 127
        extreme_key0_e[0] = 122
        extreme_key1_m[0] = -127
        extreme_key1_e[0] = 122
        extreme_query = canonical_encoded(extreme_query_m, extreme_query_e)
        extreme_key0 = canonical_encoded(extreme_key0_m, extreme_key0_e)
        extreme_key1 = canonical_encoded(extreme_key1_m, extreme_key1_e)
        extreme_scores, extreme_common, extreme_argmax, extreme_saturations = ref.normalize_score_row(
            extreme_query, (extreme_key0, extreme_key1), (True, True), alternative_id
        )
        if extreme_common != 244 or extreme_argmax != 0 or extreme_scores != (0, -(1 << 63)) or extreme_saturations != 1:
            fail("negative saturation boundary vector")

        output = ref.pack_normalized_row(query, (key0, key1), (True, True), alternative_id)
        if ref.unpack_normalized_row(output) != (scores, common, argmax, (True, True)):
            fail("normalized frame round trip")
        expect_reject(ref.pack_normalized_row, scores, common, argmax, (True, True))
        semantic_mutation_checks += 1
        frame_mutations = []
        for offset, value in ((0, ord("X")), (8, 3), (10, 1), (14, 4), (20, 1)):
            mutated = bytearray(output)
            mutated[offset] = value
            frame_mutations.append(bytes(mutated))
        frame_mutations.extend((output[:-1], output + b"\x00"))
        for mutation in frame_mutations:
            expect_reject(ref.unpack_normalized_row, mutation)

        no_centered_zero = bytearray(output)
        no_centered_zero[24:32] = (-1).to_bytes(8, "little", signed=True)
        no_centered_zero[32:40] = (-2).to_bytes(8, "little", signed=True)
        zero_only_off_argmax = bytearray(no_centered_zero)
        zero_only_off_argmax[32:40] = (0).to_bytes(8, "little", signed=True)
        argmax_points_to_negative = bytearray(output)
        argmax_points_to_negative[18:20] = (1).to_bytes(2, "little")
        for mutation in (no_centered_zero, zero_only_off_argmax, argmax_points_to_negative):
            expect_reject(ref.unpack_normalized_row, bytes(mutation))
            semantic_mutation_checks += 1
        checks += 29
    return checks, semantic_mutation_checks


def run_mutation_closure(package: dict, contract: dict) -> int:
    checks = 0
    for field in tuple(package):
        mutation = copy.deepcopy(package)
        mutation.pop(field)
        expect_reject(validate_package, mutation)
        checks += 1
    mutation = copy.deepcopy(package)
    mutation["extra"] = 0
    expect_reject(validate_package, mutation)
    checks += 1
    for field in package["claim_boundary"]:
        mutation = copy.deepcopy(package)
        mutation["claim_boundary"][field] = True
        expect_reject(validate_package, mutation)
        checks += 1
    for artifact_name in package["artifacts"]:
        mutation = copy.deepcopy(package)
        mutation["artifacts"][artifact_name]["sha256"] = "0" * 63
        expect_reject(validate_package, mutation)
        checks += 1
    contract_mutations = []
    mutation = copy.deepcopy(contract)
    mutation["alternatives"] = mutation["alternatives"][:1]
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["alternatives"].append(copy.deepcopy(mutation["alternatives"][0]))
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["alternatives"][0]["encoding"]["group_size_lanes"] = 64
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["claim_boundary"]["c02_evaluator_invoked"] = True
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["accepted_c02_evidence"]["base_first_terminal"]["status"] = "SUCCEEDED_TERMINAL"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["accepted_c02_evidence"]["checkpoint_176_c02"]["disposition"] = "AVAILABLE"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["recommendation"]["alternative_id"] = "QK_GBFP8_G8_E16_HEAD64_V1"
    contract_mutations.append(mutation)
    mutation = copy.deepcopy(contract)
    mutation["accepted_boundary"]["public_packing_api"] = "caller supplies normalized fields"
    contract_mutations.append(mutation)
    for mutation in contract_mutations:
        expect_reject(validate_contract, mutation)
        checks += 1
    return checks


def verify_artifact_bindings(package: dict) -> None:
    for binding in package["artifacts"].values():
        path = ROOT / binding["path"]
        if not path.is_file() or path.stat().st_size != binding["byte_count"] or sha256(path) != binding["sha256"]:
            fail(f"artifact binding mismatch: {binding['path']}")
    sidecar = PACKAGE_SHA_PATH.read_text(encoding="ascii").strip().split()
    if len(sidecar) != 2 or sidecar[0] != sha256(PACKAGE_PATH) or sidecar[1] != PACKAGE_PATH.name:
        fail("package sidecar mismatch")


def verify_accepted_c02_bytes(contract: dict) -> int:
    evidence = contract["accepted_c02_evidence"]
    bindings = [
        evidence["base_authority_ledger"],
        evidence["base_first_terminal"],
        evidence["base_result"],
        evidence["exactly_once_fresh_l2_verdict"],
        evidence["static_v2_fresh_l2_verdict"],
    ]
    for binding in bindings:
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or path.stat().st_size != binding["byte_count"] or sha256(path) != binding["sha256"]:
            fail(f"accepted c02 evidence drift: {path}")
    if sha256(STATIC_MANIFEST_PATH) != EXPECTED_STATIC_MANIFEST_SHA256:
        fail("accepted V2 static checksum manifest drift")
    manifest_entries = []
    for line in STATIC_MANIFEST_PATH.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            fail(f"accepted c02 artifact drift: {relative}")
        manifest_entries.append(relative)
    if len(manifest_entries) != 8:
        fail("accepted V2 manifest cardinality drift")
    terminal = load_json(ROOT / evidence["base_first_terminal"]["path"])
    result = load_json(ROOT / evidence["base_result"]["path"])
    review = load_json(Path(evidence["exactly_once_fresh_l2_verdict"]["path"]))
    if terminal.get("status") != "FAILED_TERMINAL":
        fail("Base terminal JSON no longer records FAILED_TERMINAL")
    if result.get("terminal", {}).get("status") != "FAILED_TERMINAL" or result.get("terminal", {}).get("reason_code") != "HARD_THRESHOLD_FAILED":
        fail("Base result JSON terminal drift")
    if result.get("result_sha256") != evidence["base_result"]["result_self_sha256"]:
        fail("Base result self hash drift")
    if review.get("review", {}).get("status") != "done":
        fail("accepted exactly-once Fresh-L2 verdict drift")
    checkpoint = evidence["checkpoint_176_c02"]
    if (ROOT / checkpoint["live_namespace_must_remain_absent"]).exists():
        fail("checkpoint-176 live namespace materialized")
    if (ROOT / checkpoint["result_namespace_must_remain_absent"]).exists():
        fail("checkpoint-176 result namespace materialized")
    return len(bindings) + len(manifest_entries) + 5


def main() -> int:
    package = validate_package(load_json(PACKAGE_PATH))
    contract = validate_contract(load_json(CONTRACT_PATH))
    schema = load_json(SCHEMA_PATH)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("schema identity")
    verify_artifact_bindings(package)
    mutation_checks = run_mutation_closure(package, contract)
    evidence_checks = verify_accepted_c02_bytes(contract)
    reference_checks, semantic_mutation_checks = run_reference_vectors(import_reference())
    mutation_checks += semantic_mutation_checks
    print("QK_GROUPED_BFP8_HEAD64_SUCCESSOR_STATIC_V1=PASS")
    print("AUTHORITY=NO_EXECUTION_AUTHORITY")
    print("ALTERNATIVES=2")
    print(f"REFERENCE_CHECKS={reference_checks}")
    print(f"MUTATION_CHECKS={mutation_checks}")
    print(f"ACCEPTED_C02_BYTE_CHECKS={evidence_checks}")
    print("MODEL_OR_TENSOR_ACCESS=NONE")
    print("C02_EVALUATOR_INVOCATIONS=0")
    print("NEW_MODEL_SCORES_COMPUTED=0")
    print("RTL_OR_STAGE_EDITS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
