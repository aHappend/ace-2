#!/usr/bin/env python3
"""Payload-inert public/synthetic arithmetic regressions for static V22."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
if ROOT.name != "ace-2":
    raise RuntimeError("V22 regression repository root sentinel")
EVALUATOR = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
RESOLVER = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_v21_binding_resolution.py"
OFFICIAL_PAYLOAD_NAMES = {"attention-substage-tensors.bin", "fixed-input-tensors.bin"}
AUDIT = {"official_payload_open_count": 0, "official_target_process_starts": 0}


class RegressionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0])
        except (TypeError, OSError):
            return
        if path.name in OFFICIAL_PAYLOAD_NAMES:
            AUDIT["official_payload_open_count"] += 1
            raise RegressionError("official payload open prohibited")
    if event == "subprocess.Popen":
        AUDIT["official_target_process_starts"] += 1
        raise RegressionError("subprocess start prohibited in static V22 regressions")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent, 1) if exponent >= 0 else Fraction(1, 1 << (-exponent))


def rne_fraction(value: Fraction) -> int:
    negative = value < 0
    magnitude = -value if negative else value
    retained, remainder = divmod(magnitude.numerator, magnitude.denominator)
    twice = 2 * remainder
    if twice > magnitude.denominator or (twice == magnitude.denominator and (retained & 1)):
        retained += 1
    return -retained if negative else retained


def bf16_components(word: int) -> tuple[bool, int, int]:
    require(type(word) is int and 0 <= word <= 0xFFFF, "BF16 word")
    exponent_field = (word >> 7) & 0xFF
    require(exponent_field != 0xFF, "finite BF16 word")
    fraction = word & 0x7F
    coefficient, exponent = (fraction, -133) if exponent_field == 0 else (128 + fraction, exponent_field - 134)
    return bool(word & 0x8000), coefficient, exponent


def bf16_value(word: int) -> Fraction:
    negative, coefficient, exponent = bf16_components(word)
    value = Fraction(coefficient) * pow2(exponent)
    return -value if negative else value


def independent_group(words: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    components = tuple(bf16_components(word) for word in words)
    nonzero = tuple((coefficient, exponent) for _, coefficient, exponent in components if coefficient)
    if not nonzero:
        return tuple(0 for _ in words), 0
    group_exponent = min(exponent - 8 for coefficient, exponent in nonzero)
    while True:
        magnitudes = tuple(rne_fraction(Fraction(coefficient) * pow2(exponent - group_exponent)) for coefficient, exponent in nonzero)
        if all(magnitude <= 127 for magnitude in magnitudes):
            break
        group_exponent += 1
    mantissas = []
    for negative, coefficient, exponent in components:
        magnitude = rne_fraction(Fraction(coefficient) * pow2(exponent - group_exponent))
        mantissas.append(-magnitude if negative and magnitude else magnitude)
    require(all(-127 <= value <= 127 and value != -128 for value in mantissas), "independent mantissa range")
    return tuple(mantissas), group_exponent


def independent_encode(words: tuple[int, ...], group_size: int) -> dict[str, Any]:
    mantissas: list[int] = []
    exponents: list[int] = []
    for offset in range(0, 64, group_size):
        group_mantissas, exponent = independent_group(words[offset : offset + group_size])
        mantissas.extend(group_mantissas)
        exponents.append(exponent)
    return {"exponents": tuple(exponents), "group_size": group_size, "mantissas": tuple(mantissas)}


def independent_pack(encoded: dict[str, Any]) -> bytes:
    return bytes(value & 0xFF for value in encoded["mantissas"]) + b"".join(
        value.to_bytes(2, "little", signed=True) for value in encoded["exponents"]
    )


def pair_from_fraction(value: Fraction) -> tuple[int, int]:
    if value == 0:
        return (0, 0)
    denominator = value.denominator
    require(denominator & (denominator - 1) == 0, "dyadic denominator")
    exponent = -(denominator.bit_length() - 1)
    coefficient = value.numerator
    while coefficient % 2 == 0:
        coefficient //= 2
        exponent += 1
    return coefficient, exponent


def pair_value(pair: tuple[int, int]) -> Fraction:
    return Fraction(pair[0]) * pow2(pair[1])


def independent_dot(query: dict[str, Any], key: dict[str, Any]) -> tuple[int, int]:
    group_size = query["group_size"]
    total = Fraction(0)
    for lane, (q_mantissa, k_mantissa) in enumerate(zip(query["mantissas"], key["mantissas"])):
        group_index = lane // group_size
        total += Fraction(q_mantissa * k_mantissa) * pow2(query["exponents"][group_index] + key["exponents"][group_index])
    return pair_from_fraction(total)


def independent_source_dot(query_words: tuple[int, ...], key_words: tuple[int, ...]) -> tuple[int, int]:
    return pair_from_fraction(sum((bf16_value(q) * bf16_value(k) for q, k in zip(query_words, key_words)), Fraction(0)))


def independent_normalize(score_pairs: tuple[tuple[int, int], ...], valid_mask: tuple[bool, ...]) -> tuple[tuple[int, ...], int]:
    values = tuple(pair_value(pair) for pair in score_pairs)
    top = next(index for index, valid in enumerate(valid_mask) if valid)
    for index, valid in enumerate(valid_mask):
        if valid and values[index] > values[top]:
            top = index
    realized = tuple(
        0 if not valid else rne_fraction((value - values[top]) * pow2(17))
        for value, valid in zip(values, valid_mask)
    )
    return realized, top


def verify_exhaustive_bf16(evaluator: Any) -> dict[str, Any]:
    finite_words = tuple(word for word in range(1 << 16) if ((word >> 7) & 0xFF) != 0xFF)
    require(len(finite_words) == 65280, "finite BF16 cardinality")
    categories = {"normal": 0, "signed_zero": 0, "subnormal": 0}
    g1_roundtrip_heads = 0
    tie_selected_count = 0
    for offset in range(0, len(finite_words), 64):
        chunk = finite_words[offset : offset + 64]
        padded = chunk + (0,) * (64 - len(chunk))
        production = evaluator.encode_grouped_head(padded, 1)
        independent = independent_encode(padded, 1)
        require(production == independent, f"G1 exhaustive encode chunk {offset // 64}")
        packed = evaluator.pack_grouped_head(production)
        require(packed == independent_pack(independent), "G1 exhaustive pack")
        require(evaluator.unpack_grouped_head(packed, 1) == production, "G1 exhaustive unpack")
        g1_roundtrip_heads += 1
        for lane, word in enumerate(chunk):
            require(evaluator._bf16_components(word) == bf16_components(word), f"BF16 components {word:04x}")
            exponent_field = (word >> 7) & 0xFF
            fraction = word & 0x7F
            if exponent_field == 0 and fraction == 0:
                categories["signed_zero"] += 1
            elif exponent_field == 0:
                categories["subnormal"] += 1
            else:
                categories["normal"] += 1
            negative, coefficient, source_exponent = bf16_components(word)
            if coefficient:
                selected_exponent = production["exponents"][lane]
                scaled = Fraction(coefficient) * pow2(source_exponent - selected_exponent)
                if 2 * (scaled.numerator % scaled.denominator) == scaled.denominator:
                    tie_selected_count += 1
                reconstructed = Fraction(production["mantissas"][lane]) * pow2(selected_exponent)
                expected = Fraction(-1 if negative else 1) * Fraction(rne_fraction(scaled)) * pow2(selected_exponent)
                require(reconstructed == expected, "G1 decoded dyadic")
    require(categories == {"normal": 65024, "signed_zero": 2, "subnormal": 254}, "BF16 categories")
    require(tie_selected_count > 0, "RNE tie coverage")
    return {
        "category_counts": categories,
        "finite_word_count": len(finite_words),
        "g1_pack_unpack_head_count": g1_roundtrip_heads,
        "selected_exponent_rne_tie_count": tie_selected_count,
        "status": "PASS_EXHAUSTIVE_FINITE_BF16_G1",
    }


def public_vectors() -> dict[str, tuple[int, ...]]:
    seeds = {
        "all_zero": [0x0000],
        "signed_zero": [0x0000, 0x8000],
        "subnormal_boundary": [0x0001, 0x007F, 0x8001, 0x807F],
        "unit_signs": [0x3F80, 0xBF80, 0x3F00, 0xBF00],
        "rounding_boundary": [0x3F7F, 0x3F81, 0x3FFF, 0x4001, 0xBF7F, 0xBFFF],
        "extreme_dynamic_range": [0x0001, 0x0080, 0x3F80, 0x7F7F, 0x8001, 0x8080, 0xBF80, 0xFF7F],
    }
    vectors: dict[str, tuple[int, ...]] = {}
    for name, seed in seeds.items():
        vectors[name] = tuple(seed[index % len(seed)] for index in range(64))
    base = vectors["extreme_dynamic_range"]
    for shift in range(4):
        vectors[f"permutation_rotate_{shift}"] = base[shift:] + base[:shift]
    return vectors


def verify_grouped_boundaries(evaluator: Any) -> dict[str, Any]:
    vectors = public_vectors()
    roundtrip_count = 0
    exponent_minimality_checks = 0
    for group_size in (8, 4, 2, 1):
        for name, words in vectors.items():
            production = evaluator.encode_grouped_head(words, group_size)
            independent = independent_encode(words, group_size)
            require(production == independent, f"grouped encode {group_size} {name}")
            packed = evaluator.pack_grouped_head(production)
            require(packed == independent_pack(independent), f"grouped pack {group_size} {name}")
            require(evaluator.unpack_grouped_head(packed, group_size) == production, f"grouped unpack {group_size} {name}")
            require(all(value != -128 and -127 <= value <= 127 for value in production["mantissas"]), "no saturation/reserved mantissa")
            roundtrip_count += 1
            exponent_minimality_checks += len(production["exponents"])
        malformed = bytes([0x80]) + bytes(63 + 2 * (64 // group_size))
        try:
            evaluator.unpack_grouped_head(malformed, group_size)
        except evaluator.EvaluationError:
            pass
        else:
            raise RegressionError(f"reserved -128 accepted for G{group_size}")
    return {
        "exponent_minimality_check_count": exponent_minimality_checks,
        "group_sizes": [8, 4, 2, 1],
        "pack_unpack_case_count": roundtrip_count,
        "reserved_mantissa_negative_case_count": 4,
        "saturation_event_count": 0,
        "vector_count": len(vectors),
        "status": "PASS_GROUPED_BFP8_BOUNDARIES",
    }


def verify_dot_scale_mask_center_ties(evaluator: Any) -> dict[str, Any]:
    vectors = public_vectors()
    dot_cases = [
        ("zero", vectors["all_zero"], vectors["unit_signs"], 8),
        ("cancellation", vectors["unit_signs"], tuple(reversed(vectors["unit_signs"])), 4),
        ("mixed_exponents", vectors["extreme_dynamic_range"], vectors["rounding_boundary"], 2),
        ("per_lane", vectors["rounding_boundary"], vectors["permutation_rotate_3"], 1),
    ]
    observed_dot_cases = []
    for name, q_words, k_words, group_size in dot_cases:
        query = evaluator.encode_grouped_head(q_words, group_size)
        key = evaluator.encode_grouped_head(k_words, group_size)
        production = evaluator.grouped_dot_score_pair(query, key)
        independent = independent_dot(independent_encode(q_words, group_size), independent_encode(k_words, group_size))
        require(production == independent, f"dot oracle {name}")
        observed_dot_cases.append({"group_size": group_size, "name": name, "pair": list(production)})

    realization_pairs = ((1, 0), (3, -18), (5, -18), (-3, -18), (1, -40), (0, 0))
    for pair in realization_pairs:
        require(evaluator.realize_score_pair_q12_20(pair) == rne_fraction(pair_value(pair) * pow2(17)), f"Q12.20 realization {pair}")

    row_cases = [
        (((3, -18), (5, -18), (7, -18), (99, 5)), (True, True, True, False), "causal_mask"),
        (((5, -18), (5, -18), (3, -18)), (True, True, True), "lowest_index_tie"),
        (((1, -40), (3, -40), (-1, -40)), (True, True, True), "underflow_rounding"),
    ]
    row_results = []
    for pairs, mask, name in row_cases:
        production_scores, _, production_top = evaluator.normalize_score_row_q12_20(pairs, mask)
        independent_scores, independent_top = independent_normalize(pairs, mask)
        require(production_scores == independent_scores and production_top == independent_top, f"row normalization {name}")
        require(all(score <= 0 for score, valid in zip(production_scores, mask) if valid), f"row centering {name}")
        if name == "causal_mask":
            require(production_top != 3 and production_scores[3] == 0, "causal mask exclusion")
        if name == "lowest_index_tie":
            require(production_top == 0, "lowest-key-index tie rule")
        row_results.append({"name": name, "scores_q12_20_lsb": list(production_scores), "top_key": production_top})
    return {
        "dot_cases": observed_dot_cases,
        "head64_scale": {"denominator": 8, "q12_20_exponent_offset": 17},
        "realization_case_count": len(realization_pairs),
        "row_cases": row_results,
        "status": "PASS_DOT_SCALE_MASK_CENTER_TIES_Q12_20",
    }


def candidate_row(evaluator: Any, query: tuple[int, ...], keys: tuple[tuple[int, ...], ...], group_size: int) -> tuple[tuple[int, ...], int]:
    encoded_query = evaluator.encode_grouped_head(query, group_size)
    pairs = tuple(evaluator.grouped_dot_score_pair(encoded_query, evaluator.encode_grouped_head(key, group_size)) for key in keys)
    scores, _, top = evaluator.normalize_score_row_q12_20(pairs, tuple(True for _ in keys))
    return scores, top


def source_row(query: tuple[int, ...], keys: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], int]:
    pairs = tuple(independent_source_dot(query, key) for key in keys)
    return independent_normalize(pairs, tuple(True for _ in keys))


def small_candidate_row(query: tuple[int, int], keys: tuple[tuple[int, int], ...], group_size: int) -> tuple[tuple[int, ...], int]:
    require(group_size in (1, 2), "small candidate group size")
    if group_size == 1:
        q_groups = tuple(independent_group((word,)) for word in query)
    else:
        q_groups = (independent_group(query),)
    pairs = []
    for key in keys:
        if group_size == 1:
            k_groups = tuple(independent_group((word,)) for word in key)
        else:
            k_groups = (independent_group(key),)
        total = Fraction(0)
        for group_index, ((q_mantissas, q_exponent), (k_mantissas, k_exponent)) in enumerate(zip(q_groups, k_groups)):
            for q_mantissa, k_mantissa in zip(q_mantissas, k_mantissas):
                total += Fraction(q_mantissa * k_mantissa) * pow2(q_exponent + k_exponent)
        pairs.append(pair_from_fraction(total))
    return independent_normalize(tuple(pairs), tuple(True for _ in keys))


def small_source_row(query: tuple[int, int], keys: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], int]:
    pairs = tuple(pair_from_fraction(sum((bf16_value(q) * bf16_value(k) for q, k in zip(query, key)), Fraction(0))) for key in keys)
    return independent_normalize(pairs, tuple(True for _ in keys))


def fast_round_unsigned(coefficient: int, shift: int) -> int:
    if shift >= 0:
        return coefficient << shift
    divisor = 1 << (-shift)
    retained, remainder = divmod(coefficient, divisor)
    half = divisor >> 1
    return retained + (remainder > half or (remainder == half and (retained & 1)))


def fast_round_signed(value: int, right_shift: int) -> int:
    magnitude = fast_round_unsigned(abs(value), -right_shift)
    return -magnitude if value < 0 else magnitude


def fast_encode(words: tuple[int, ...], group_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    components = tuple(bf16_components(word) for word in words)
    mantissas: list[int] = []
    exponents: list[int] = []
    for offset in range(0, len(words), group_size):
        group = components[offset : offset + group_size]
        nonzero = tuple((coefficient, exponent) for _, coefficient, exponent in group if coefficient)
        if not nonzero:
            group_exponent = 0
        else:
            group_exponent = min(exponent - 8 for coefficient, exponent in nonzero)
            while not all(fast_round_unsigned(coefficient, exponent - group_exponent) <= 127 for coefficient, exponent in nonzero):
                group_exponent += 1
        for negative, coefficient, exponent in group:
            magnitude = fast_round_unsigned(coefficient, exponent - group_exponent)
            mantissas.append(-magnitude if negative and magnitude else magnitude)
        exponents.append(group_exponent)
    return tuple(mantissas), tuple(exponents)


def fast_pair(terms: list[tuple[int, int]]) -> tuple[int, int]:
    terms = [(coefficient, exponent) for coefficient, exponent in terms if coefficient]
    if not terms:
        return (0, 0)
    common = min(exponent for _, exponent in terms)
    coefficient = sum(value << (exponent - common) for value, exponent in terms)
    if coefficient == 0:
        return (0, 0)
    while coefficient % 2 == 0:
        coefficient //= 2
        common += 1
    return coefficient, common


def fast_source_pair(query: tuple[int, ...], key: tuple[int, ...]) -> tuple[int, int]:
    terms = []
    for q_word, k_word in zip(query, key):
        q_negative, q_coefficient, q_exponent = bf16_components(q_word)
        k_negative, k_coefficient, k_exponent = bf16_components(k_word)
        coefficient = q_coefficient * k_coefficient
        if q_negative != k_negative:
            coefficient = -coefficient
        terms.append((coefficient, q_exponent + k_exponent))
    return fast_pair(terms)


def fast_encoded_pair(query: tuple[tuple[int, ...], tuple[int, ...]], key: tuple[tuple[int, ...], tuple[int, ...]], group_size: int) -> tuple[int, int]:
    q_mantissas, q_exponents = query
    k_mantissas, k_exponents = key
    terms = []
    for group_index, (q_exponent, k_exponent) in enumerate(zip(q_exponents, k_exponents)):
        start = group_index * group_size
        stop = start + group_size
        coefficient = sum(q * k for q, k in zip(q_mantissas[start:stop], k_mantissas[start:stop]))
        terms.append((coefficient, q_exponent + k_exponent))
    return fast_pair(terms)


def fast_normalize(pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], int]:
    common = min((exponent for coefficient, exponent in pairs if coefficient), default=0)
    aligned = tuple(0 if coefficient == 0 else coefficient << (exponent - common) for coefficient, exponent in pairs)
    top = 0
    for index in range(1, len(aligned)):
        if aligned[index] > aligned[top]:
            top = index
    shift = common + 17
    scores = tuple(
        (value - aligned[top]) << shift if shift >= 0 else fast_round_signed(value - aligned[top], -shift)
        for value in aligned
    )
    return scores, top


def find_near_tie_witness(evaluator: Any) -> dict[str, Any]:
    rows = {
        "oracle": ((0, 0), (-2, -17), (-100, -17)),
        "G2": ((0, 0), (-1000, -17), (-100, -17)),
        "G1": ((-1, -17), (0, 0), (-100, -17)),
    }
    observed: dict[str, dict[str, Any]] = {}
    for label, pairs in rows.items():
        scores, _, top = evaluator.normalize_score_row_q12_20(pairs, (True, True, True))
        independent_scores, independent_top = independent_normalize(pairs, (True, True, True))
        require((scores, top) == (independent_scores, independent_top), f"near-tie independent row {label}")
        observed[label] = {"scores_q12_20_lsb": list(scores), "top_key": top}
    oracle_scores = tuple(observed["oracle"]["scores_q12_20_lsb"])
    g2_scores = tuple(observed["G2"]["scores_q12_20_lsb"])
    g1_scores = tuple(observed["G1"]["scores_q12_20_lsb"])
    g2_error = sum(abs(actual - expected) for actual, expected in zip(g2_scores, oracle_scores))
    g1_error = sum(abs(actual - expected) for actual, expected in zip(g1_scores, oracle_scores))
    require(observed["oracle"]["top_key"] == observed["G2"]["top_key"] == 0, "near-tie G2 top preservation")
    require(observed["G1"]["top_key"] == 1, "near-tie G1 top regression")
    require(g1_error < g2_error, "near-tie aggregate error ordering")
    return {
        "construction_kind": "SYNTHETIC_Q12_20_SCORE_PAIR_ROW_NOT_V21_CAUSAL_ATTRIBUTION",
        "corpus_sha256": sha256_bytes(compact_bytes({label: [list(pair) for pair in pairs] for label, pairs in rows.items()})),
        "g1_absolute_error_q12_20_lsb": g1_error,
        "g1_top_key": observed["G1"]["top_key"],
        "g2_absolute_error_q12_20_lsb": g2_error,
        "g2_top_key": observed["G2"]["top_key"],
        "oracle_scores_q12_20_lsb": list(oracle_scores),
        "oracle_top_key": observed["oracle"]["top_key"],
    }


def verify_binding_resolution(resolver: Any) -> dict[str, Any]:
    names = tuple(resolver.CANONICAL_TENSOR_NAMES)
    role_records = {
        "k": {"dtype": "torch.bfloat16", "sha256": "1" * 64, "shape": [1, 2, 3, 64], "tensor_name": names[0]},
        "q": {"dtype": "torch.bfloat16", "sha256": "2" * 64, "shape": [1, 14, 3, 64], "tensor_name": names[1]},
        "o": {"dtype": "torch.bfloat16", "sha256": "3" * 64, "shape": [1, 14, 3, 3], "tensor_name": names[2]},
    }
    binding_table = {"record_count": 25, "records": {f"record_{index:02d}": {} for index in range(25)}, "selected_tensor_names": list(names)}
    permutation_digests = []
    for order in permutations(role_records):
        records = {f"source_{role}": role_records[role] for role in order}
        selected = resolver.resolve_production_selection(records, binding_table)
        require(selected.canonical_names == names, "canonical resolver names")
        require(tuple(selected.records_by_name) == names, "canonical resolver order")
        permutation_digests.append(sha256_bytes(compact_bytes({name: selected.source_keys_by_name[name] for name in names})))

    negative_cases: dict[str, str] = {}
    malformed_cases = {
        "duplicate": {"a": role_records["k"], "b": dict(role_records["k"]), "c": role_records["q"]},
        "missing": {"a": role_records["k"], "b": role_records["q"]},
        "unexpected": {"a": role_records["k"], "b": role_records["q"], "c": {**role_records["o"], "tensor_name": "bf16.unexpected"}},
        "malformed": {"a": role_records["k"], "b": role_records["q"], "c": {"dtype": "torch.bfloat16"}},
    }
    for name, records in malformed_cases.items():
        try:
            resolver.resolve_production_selection(records, binding_table)
        except resolver.BindingResolutionError as error:
            negative_cases[name] = error.code
        else:
            raise RegressionError(f"binding negative case accepted: {name}")
    for name, altered in {
        "altered_record_count": {**binding_table, "record_count": 24},
        "altered_selected_names": {**binding_table, "selected_tensor_names": list(reversed(names))},
    }.items():
        try:
            resolver.resolve_production_selection({f"source_{role}": role_records[role] for role in role_records}, altered)
        except resolver.BindingResolutionError as error:
            negative_cases[name] = error.code
        else:
            raise RegressionError(f"binding table negative case accepted: {name}")
    return {
        "canonical_tensor_names": list(names),
        "negative_case_codes": negative_cases,
        "permutation_count": len(permutation_digests),
        "permutation_resolution_digest": sha256_bytes(compact_bytes(sorted(permutation_digests))),
        "status": "PASS_SIX_PERMUTATIONS_AND_FAIL_CLOSED_BINDINGS",
    }


def run_regressions() -> dict[str, Any]:
    AUDIT.update({"official_payload_open_count": 0, "official_target_process_starts": 0})
    evaluator = load_module(EVALUATOR, "v22_static_public_evaluator_core")
    resolver = load_module(RESOLVER, "v22_static_v21_binding_resolver")
    report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_attribution_v22_public_synthetic_regression_report",
        "bf16_decode_and_g1": verify_exhaustive_bf16(evaluator),
        "binding_resolution": verify_binding_resolution(resolver),
        "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
        "dot_scale_mask_center_ties": verify_dot_scale_mask_center_ties(evaluator),
        "grouped_bfp8_boundaries": verify_grouped_boundaries(evaluator),
        "near_tie_argmax": find_near_tie_witness(evaluator),
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V22_PUBLIC_SYNTHETIC_SOURCE_ORACLE_REGRESSIONS",
        "v22_execution_authority_granted": False,
    }
    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "static audit counters")
    report["report_sha256"] = sha256_bytes(compact_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.addaudithook(audit_hook)
    report = run_regressions()
    raw = compact_bytes(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
