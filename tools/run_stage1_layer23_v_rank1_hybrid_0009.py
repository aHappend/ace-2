#!/usr/bin/env python3
"""Pristine-environment preloader for nonofficial hybrid attempt 0009."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
PREDECESSOR_LAUNCHER = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0008.py"
IMPLEMENTATION = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid.py"
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0009"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0009.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0009.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0009.json"
PREDECESSOR_0006_TOKENIZATION = PRIVATE / "exact-tokenization-preflight-0006.json"
PREDECESSOR_0008_TERMINAL = (
    OUTPUT.parent / "nonofficial-hybrid-0008/terminal-record.json"
)
EXPECTED_PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
EXPECTED_PYTHON_SHA256 = (
    "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
)
EXPECTED_PREDECESSOR_0006_TOKENIZATION_SHA256 = (
    "d1b6cfe2e208dfece95e57da31ce0fe4c1df1ed08c1acd9f61ad5847d8d8cb49"
)
EXPECTED_PREDECESSOR_0008_TERMINAL_SHA256 = (
    "566e3bf4fd7a4a4b625b72acdcf5f80e19d5aca144be3c0c8c28d7cf51cb8daa"
)
RETAINED_OUTPUT_SCALE = 0.06586176203930472
MAX_CONTEXT_TOKENS = 40
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/home/argustest",
    "LC_ALL": "C",
    "PATH": (
        "/home/argustest/argustest2/.venv/bin:/home/argustest/bin:"
        "/home/argustest/.bun/bin:/home/argustest/.krew/bin:"
        "/home/argustest/.local/bin:/home/argustest/bin:/home/argustest/.krew/bin:"
        "/home/argustest/.vscode-server/data/User/globalStorage/"
        "github.copilot-chat/debugCommand:"
        "/home/argustest/.vscode-server/data/User/globalStorage/"
        "github.copilot-chat/copilotCli:"
        "/home/argustest/.vscode-server/cli/servers/Stable-"
        "a5b500951314efd502d07465bd138dfbd714a960/server/bin/remote-cli:"
        "/home/argustest/.elan/bin:/home/argustest/.local/bin:"
        "/home/argustest/bin:/home/argustest/Argus_BoXiuLi8T_MOF/code/"
        "Argus_mofgen_uv:/home/argustest/bin:/home/argustest/.bun/bin:"
        "/home/argustest/.krew/bin:/home/argustest/.local/bin:"
        "/home/argustest/bin:/home/argustest/.krew/bin:"
        "/home/argustest/.nvm/versions/node/v22.23.1/bin:"
        "/home/argustest/miniconda3/bin:/home/argustest/miniconda3/condabin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
        "/usr/games:/usr/local/games:/snap/bin"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/home/argustest/ace-2/.venv/lib/python3.13/site-packages",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
REQUIRED_PREIMPORT_ABSENT = (
    "numpy",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "tools",
)
EXPECTED_TOKENIZATION_CHECKS = {
    "model_executed": False,
    "simulator_executed": False,
    "icarus_compile_executed": False,
    "process_not_started": True,
    "output_namespace_absent": True,
    "exact_final_auto_tokenizer_path_executed": True,
    "canonical_chat_token_ids_path_executed": True,
    "resolved_tokenizer_files_bound": True,
    "chat_template_hash_bound": True,
    "exactly_two_new_unseen_prompts": True,
    "generation_count_is_four": True,
    "all_total_context_counts_at_most_40": True,
    "all_maximum_processed_positions_below_40": True,
    "accepted_rtl_context_bound_unchanged": True,
    "predecessor_0008_terminal_and_0006_certificate_seals_immutable": True,
}

# Capture the launch state before any dependency or project import.
PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)


class PreloaderError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreloaderError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid07",
        "attempt_identity": "nonofficial-hybrid-0009",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": _utc_now(),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_nonconsuming_gate",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "regression": "No retry, replay, or resume; preserve the failed gate artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _verify_pre_import_launch() -> None:
    interpreter = Path(sys.executable).resolve()
    _require(interpreter == EXPECTED_PYTHON, "Python interpreter binding changed")
    _require(
        _sha256_file(interpreter) == EXPECTED_PYTHON_SHA256,
        "Python interpreter SHA-256 changed",
    )
    _require(PRE_IMPORT_CWD == str(ROOT), "final-child cwd binding changed")
    _require(
        Path(PRE_IMPORT_ARGV[0]).resolve() == LAUNCHER,
        "launcher argv[0] binding changed",
    )
    _require(
        PRE_IMPORT_ENVIRONMENT == EXPECTED_ENVIRONMENT,
        "pristine pre-import OS environment differs from the frozen launch environment",
    )
    contaminated = sorted(
        name
        for name in PRE_IMPORT_MODULES
        if any(name == root or name.startswith(f"{root}.") for root in REQUIRED_PREIMPORT_ABSENT)
    )
    _require(not contaminated, f"dependency/project modules loaded before snapshot: {contaminated}")
    _require(
        stat.S_ISREG(os.lstat(LAUNCHER).st_mode)
        and stat.S_ISREG(os.lstat(PREDECESSOR_LAUNCHER).st_mode)
        and stat.S_ISREG(os.lstat(IMPLEMENTATION).st_mode),
        "launcher or bound implementation source is not a regular file",
    )


def _round_fraction_even(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("oracle fraction must be finite and nonnegative")
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and (quotient & 1)):
        quotient += 1
    return quotient


def _oracle_derive_multiplier(real_multiplier: float) -> tuple[int, int]:
    if not math.isfinite(real_multiplier) or real_multiplier < 0.0:
        raise ValueError("oracle multiplier input must be finite and nonnegative")
    numerator, denominator = real_multiplier.as_integer_ratio()
    for shift in range(63, -1, -1):
        candidate = _round_fraction_even(numerator << shift, denominator)
        if candidate <= INT32_MAX:
            return candidate, shift
    raise OverflowError("oracle multiplier is not representable")


def _oracle_round_shift_even(value: int, shift: int) -> int:
    if not 0 <= shift <= 63:
        raise ValueError("oracle shift is not u6")
    if shift == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return -quotient if value < 0 else quotient


def _oracle_requantize(
    accumulators: list[int], multipliers: list[int], shifts: list[int]
) -> tuple[list[int], list[bool]]:
    if not (len(accumulators) == len(multipliers) == len(shifts)):
        raise ValueError("oracle vector lengths differ")
    converted: list[int] = []
    saturation: list[bool] = []
    for accumulator, multiplier, shift in zip(
        accumulators, multipliers, shifts, strict=True
    ):
        if not -(1 << 31) <= accumulator < (1 << 31):
            raise OverflowError("oracle accumulator escaped signed int32")
        if not 0 <= multiplier <= INT32_MAX:
            raise OverflowError("oracle multiplier escaped nonnegative signed int32")
        product = accumulator * multiplier
        if not INT64_MIN <= product <= INT64_MAX:
            raise OverflowError("oracle product escaped signed int64")
        rounded = _oracle_round_shift_even(product, shift)
        saturated = rounded < -128 or rounded > 127
        converted.append(max(-128, min(127, rounded)))
        saturation.append(saturated)
    return converted, saturation


def _oracle_rank1(vector: Any, config: Any) -> dict[str, Any]:
    rank_accumulator = sum(
        int(sample) * int(factor)
        for sample, factor in zip(
            vector.input_s8, config.input_to_rank_s8, strict=True
        )
    )
    rank_rounded = _oracle_round_shift_even(
        rank_accumulator * int(config.first_stage_multiplier_s32),
        int(config.first_stage_right_shift_u6),
    )
    rank_s8 = max(-128, min(127, rank_rounded))
    correction_accumulator = [
        rank_s8 * int(factor) for factor in config.rank_to_channel_s8
    ]
    correction_rounded = [
        _oracle_round_shift_even(accumulator * int(multiplier), int(shift))
        for accumulator, multiplier, shift in zip(
            correction_accumulator,
            config.second_stage_multiplier_s32,
            config.second_stage_right_shift_u6,
            strict=True,
        )
    ]
    correction_s8 = [max(-128, min(127, value)) for value in correction_rounded]
    corrected = [
        max(-128, min(127, int(baseline) + correction))
        for baseline, correction in zip(
            vector.baseline_v_s8, correction_s8, strict=True
        )
    ]
    return {
        "rank_accumulator_s32": rank_accumulator,
        "rank_rounded_s32": rank_rounded,
        "rank_intermediate_s8": rank_s8,
        "correction_accumulator_s32": correction_accumulator,
        "correction_rounded_s32": correction_rounded,
        "correction_s8": correction_s8,
        "corrected_v_s8": corrected,
    }


def _signed_bytes(values: list[int] | tuple[int, ...]) -> bytes:
    return bytes(int(value) & 0xFF for value in values)


def _bind_predecessor_module(predecessor: Any) -> None:
    bindings = {
        "LAUNCHER": LAUNCHER,
        "OUTPUT": OUTPUT,
        "TERMINAL": TERMINAL,
        "PRIVATE": PRIVATE,
        "TOKENIZATION_PREFLIGHT": TOKENIZATION_PREFLIGHT,
        "TOKENIZATION_REGRESSION": TOKENIZATION_REGRESSION,
        "PREDECESSOR_0006_TOKENIZATION": PREDECESSOR_0006_TOKENIZATION,
        "EXPECTED_PREDECESSOR_0006_TOKENIZATION_SHA256": (
            EXPECTED_PREDECESSOR_0006_TOKENIZATION_SHA256
        ),
        "EXPECTED_TOKENIZATION_CHECKS": EXPECTED_TOKENIZATION_CHECKS,
        "EXPECTED_ENVIRONMENT": EXPECTED_ENVIRONMENT,
        "MAX_CONTEXT_TOKENS": MAX_CONTEXT_TOKENS,
        "PRE_IMPORT_ENVIRONMENT": PRE_IMPORT_ENVIRONMENT,
        "PRE_IMPORT_SYS_PATH": PRE_IMPORT_SYS_PATH,
        "PRE_IMPORT_SYS_PREFIX": PRE_IMPORT_SYS_PREFIX,
        "PRE_IMPORT_ARGV": PRE_IMPORT_ARGV,
        "PRE_IMPORT_CWD": PRE_IMPORT_CWD,
        "PRE_IMPORT_MODULES": PRE_IMPORT_MODULES,
    }
    for name, value in bindings.items():
        setattr(predecessor, name, value)


def _configure_runner(
    runner: Any,
    predecessor: Any,
    rank1_reference: Any,
    post_import_environment: dict[str, str],
) -> None:
    predecessor._configure_runner(runner, post_import_environment)
    runner.MISSION_ID = "stage1rank1hybrid07"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0009"
    runner.OUTPUT = OUTPUT
    runner.PREFLIGHT = OUTPUT / "preflight.json"
    runner.FREEZE = OUTPUT / "freeze.json"
    runner.PACKAGE = OUTPUT / "execution-package.json"
    runner.AUTHORITY = OUTPUT / "execution-authority.json"
    runner.LIVE = OUTPUT / "live"
    runner.RESULT = runner.LIVE / "result.json"
    runner.FAILURE = runner.LIVE / "failure.json"
    runner.SUMS = runner.LIVE / "SHA256SUMS"
    runner.STATE = OUTPUT / "supervisor-state"
    runner.STDOUT = OUTPUT / "execution.stdout.log"
    runner.STDERR = OUTPUT / "execution.stderr.log"
    runner.TERMINAL = TERMINAL
    runner.DEFAULT_PROMPTS = PRIVATE / "prompts-0009.json"
    runner.TEMPLATE_GATE = PRIVATE / "package-template-invariant-0009.json"
    runner.DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0009.json"
    runner.PREDECESSOR_TERMINAL_SEALS = {
        **runner.PREDECESSOR_TERMINAL_SEALS,
        "nonofficial-hybrid-0008": {
            "sha256": EXPECTED_PREDECESSOR_0008_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_dependency_probe = runner.dependency_probe
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_package_template = runner.package_template_invariant
    prior_preflight = runner.preflight
    prior_freeze = runner.freeze
    prior_child_entry_attestation = runner.child_entry_attestation
    prior_execute = runner.execute
    prior_verify = runner.verify
    prior_compare_runs = runner.compare_runs
    prior_cache_class = runner.Rank1HybridProjectionCache

    def verify_predecessor_seals_0009() -> None:
        prior_verify_predecessor_seals()
        runner.require(
            runner.sha256_file(PREDECESSOR_0008_TERMINAL)
            == EXPECTED_PREDECESSOR_0008_TERMINAL_SHA256,
            "nonofficial-hybrid-0008 terminal seal changed",
        )
        terminal = runner.read_json(PREDECESSOR_0008_TERMINAL)
        runner.require(
            terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("outcome") == "child_nonzero_exit",
            "nonofficial-hybrid-0008 terminal state changed",
        )

    runner.verify_predecessor_seals = verify_predecessor_seals_0009

    def dependency_probe_0009() -> int:
        runner.require(
            not SCALE_REGRESSION.exists(),
            "retained-scale regression already exists before dependency probe",
        )
        result = prior_dependency_probe()
        record = runner.read_json(runner.DEPENDENCY_PROBE)
        record["top_level_import_chain"].extend(
            [
                {
                    "module": "tools.run_stage1_layer23_v_rank1_hybrid_0008",
                    "file": runner.file_record(PREDECESSOR_LAUNCHER),
                },
                {
                    "module": (
                        "tools.ace2_layer23_v_rank1_integer_correction_reference"
                    ),
                    "file": runner.file_record(
                        Path(rank1_reference.__file__).resolve(strict=True)
                    ),
                },
            ]
        )
        record["checks"][
            "predecessor_launcher_source_explicitly_bound"
        ] = True
        record["checks"]["independent_integer_oracle_source_bound"] = True
        record["checks"]["retained_scale_regression_absent"] = True
        record["checks"].pop(
            "predecessor_0007_terminal_and_0006_certificate_seals_immutable",
            None,
        )
        record["checks"][
            "predecessor_0008_terminal_and_0006_certificate_seals_immutable"
        ] = True
        runner.write_json(runner.DEPENDENCY_PROBE, record)
        return result

    runner.dependency_probe = dependency_probe_0009

    def _production_convert(
        values: list[int] | tuple[int, ...], source_scale: float
    ) -> dict[str, Any]:
        real = runner.torch.tensor(
            source_scale / RETAINED_OUTPUT_SCALE, dtype=runner.torch.float64
        )
        multiplier, shift = runner.backend.canonical.derive_multiplier(real)
        multiplier_value = int(multiplier.item())
        shift_value = int(shift.item())
        value_tensor = runner.torch.tensor(values, dtype=runner.torch.int64)
        multipliers = runner.torch.full_like(value_tensor, multiplier_value)
        shifts = runner.torch.full_like(value_tensor, shift_value)
        rounded = runner.localizer.round_shift_even_tensor(
            value_tensor * multipliers, shifts
        )
        converted = rounded.clamp(-128, 127).to(runner.torch.int8).tolist()
        saturation = ((rounded < -128) | (rounded > 127)).tolist()
        return {
            "real_multiplier_f64": float(real.item()),
            "multiplier_s32": multiplier_value,
            "right_shift_u6": shift_value,
            "converted_s8": [int(value) for value in converted],
            "saturation": [bool(value) for value in saturation],
        }

    def _build_scale_regression_payload() -> dict[str, Any]:
        config = rank1_reference.load_frozen_config(check_review=True)
        vectors = rank1_reference.load_frozen_position_vectors(check_review=True)
        frozen_records = []
        for vector in vectors:
            production = _production_convert(
                vector.baseline_v_s8, RETAINED_OUTPUT_SCALE
            )
            oracle_multiplier, oracle_shift = _oracle_derive_multiplier(
                production["real_multiplier_f64"]
            )
            oracle_converted, oracle_saturation = _oracle_requantize(
                [int(value) for value in vector.baseline_v_s8],
                [oracle_multiplier] * len(vector.baseline_v_s8),
                [oracle_shift] * len(vector.baseline_v_s8),
            )
            oracle_rank = _oracle_rank1(vector, config)
            runner.require(
                production["multiplier_s32"] == oracle_multiplier
                and production["right_shift_u6"] == oracle_shift
                and production["converted_s8"] == oracle_converted
                and production["saturation"] == oracle_saturation,
                f"frozen position {vector.position} scale conversion differs from oracle",
            )
            runner.require(
                oracle_converted == list(vector.baseline_v_s8)
                and not any(oracle_saturation),
                f"frozen position {vector.position} retained identity conversion changed bytes",
            )
            expected_rank = {
                "rank_accumulator_s32": vector.expected_rank_accumulator_s32,
                "rank_rounded_s32": vector.expected_rank_rounded_s32,
                "rank_intermediate_s8": vector.expected_rank_intermediate_s8,
                "correction_accumulator_s32": list(
                    vector.expected_correction_accumulator_s32
                ),
                "correction_rounded_s32": list(
                    vector.expected_correction_rounded_s32
                ),
                "correction_s8": list(vector.expected_correction_s8),
                "corrected_v_s8": list(vector.expected_corrected_v_s8),
            }
            runner.require(
                oracle_rank == expected_rank,
                f"frozen position {vector.position} rank-1 result differs from oracle",
            )
            frozen_records.append(
                {
                    "position": vector.position,
                    "source_scale": RETAINED_OUTPUT_SCALE,
                    "retained_scale": RETAINED_OUTPUT_SCALE,
                    "multiplier_s32": oracle_multiplier,
                    "right_shift_u6": oracle_shift,
                    "saturation_count": sum(oracle_saturation),
                    "converted_bytes_s8": oracle_converted,
                    "converted_bytes_sha256": runner.sha256_bytes(
                        _signed_bytes(oracle_converted)
                    ),
                    "rank1_corrected_bytes_sha256": runner.sha256_bytes(
                        _signed_bytes(oracle_rank["corrected_v_s8"])
                    ),
                }
            )

        edge_specs = (
            (
                "positive_negative_half_ties",
                RETAINED_OUTPUT_SCALE * 0.5,
                [0, 1, -1, 3, -3, 5, -5],
            ),
            (
                "positive_negative_saturation",
                RETAINED_OUTPUT_SCALE * 2.0,
                [0, 63, 64, 127, -64, -65, -128],
            ),
            (
                "small_bounded_shift",
                math.ldexp(RETAINED_OUTPUT_SCALE, -32),
                [0, 1, -1, 127, -128],
            ),
        )
        edge_records = []
        for name, source_scale, values in edge_specs:
            production = _production_convert(values, source_scale)
            oracle_multiplier, oracle_shift = _oracle_derive_multiplier(
                production["real_multiplier_f64"]
            )
            converted, saturation = _oracle_requantize(
                values,
                [oracle_multiplier] * len(values),
                [oracle_shift] * len(values),
            )
            runner.require(
                production["multiplier_s32"] == oracle_multiplier
                and production["right_shift_u6"] == oracle_shift
                and production["converted_s8"] == converted
                and production["saturation"] == saturation,
                f"scale edge case differs from independent oracle: {name}",
            )
            edge_records.append(
                {
                    "case": name,
                    "source_scale": source_scale,
                    "retained_scale": RETAINED_OUTPUT_SCALE,
                    "multiplier_s32": oracle_multiplier,
                    "right_shift_u6": oracle_shift,
                    "input_s8": values,
                    "converted_bytes_s8": converted,
                    "saturation": saturation,
                }
            )
        runner.require(
            edge_records[0]["converted_bytes_s8"] == [0, 0, 0, 2, -2, 2, -2],
            "ties-to-even edge expectation changed",
        )
        runner.require(
            edge_records[1]["converted_bytes_s8"]
            == [0, 126, 127, 127, -128, -128, -128]
            and edge_records[1]["saturation"]
            == [False, False, True, True, False, True, True],
            "int8 saturation edge expectation changed",
        )
        return {
            "conversion_contract": {
                "method": (
                    "configure fused-QKV V per-channel signed-int32 multiplier/u6 "
                    "right-shift metadata to emit retained-scale bytes directly"
                ),
                "source_domain": "dynamic projection output_scale",
                "retained_output_scale": RETAINED_OUTPUT_SCALE,
                "rounding": "signed round-to-nearest ties-to-even",
                "product_width": "signed_int64",
                "output": "signed_int8_saturation",
                "descriptor_interface_change_required": False,
                "sidecar_input": "ace2_shell shared_payload_q in retained-scale quanta",
            },
            "frozen_positions": frozen_records,
            "edge_cases": edge_records,
            "sources": {
                "launcher": runner.file_record(LAUNCHER),
                "rank1_reference": runner.file_record(
                    Path(rank1_reference.__file__).resolve(strict=True)
                ),
                "candidate_freeze": runner.file_record(rank1_reference.FREEZE),
                "candidate_result": runner.file_record(rank1_reference.RESULT),
            },
            "checks": {
                "all_34_frozen_positions_match_independent_integer_oracle": True,
                "frozen_rank1_outputs_unchanged": True,
                "positive_and_negative_half_ties_covered": True,
                "positive_and_negative_int8_saturation_covered": True,
                "multiplier_signed_int32_bounded": True,
                "right_shift_u6_bounded": True,
                "signed_int64_products_bounded": True,
                "no_model_execution": True,
                "no_simulator_execution": True,
                "no_icarus_compile_execution": True,
                "process_not_started": True,
            },
        }

    def verify_scale_regression() -> dict[str, Any]:
        runner.require(
            SCALE_REGRESSION.is_file(), "retained-scale regression is absent"
        )
        record = runner.read_json(SCALE_REGRESSION)
        runner.require(
            record.get("mission_id") == runner.MISSION_ID
            and record.get("attempt_identity") == runner.ATTEMPT_ID
            and record.get("status")
            == "PASS_NONCONSUMING_RETAINED_SCALE_INTEGER_ORACLE",
            "retained-scale regression identity/status changed",
        )
        runner.require(
            record.get("proof") == _build_scale_regression_payload(),
            "retained-scale regression proof changed",
        )
        return record

    runner.verify_scale_regression = verify_scale_regression

    def scale_regression_0009(prompts_path: Path) -> int:
        runner.require(
            not SCALE_REGRESSION.exists(),
            "retained-scale regression already exists",
        )
        runner.require(
            not runner.TEMPLATE_GATE.exists(),
            "package-template invariant exists before retained-scale regression",
        )
        runner.require(
            not runner.OUTPUT.exists(),
            f"{runner.ATTEMPT_ID} output namespace exists before retained-scale regression",
        )
        runner.validate_no_execution()
        runner.verify_predecessor_seals()
        runner.verify_tokenization_preflight(prompts_path)
        runner.verify_tokenization_verifier_regression()
        record = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_NONCONSUMING_RETAINED_SCALE_INTEGER_ORACLE",
            "created_at_utc": runner.utc_now(),
            "proof": _build_scale_regression_payload(),
        }
        runner.write_json(SCALE_REGRESSION, record)
        runner.validate_no_execution()
        print(
            "ACE2_HYBRID_RETAINED_SCALE_REGRESSION_PASS "
            f"sha256={runner.sha256_file(SCALE_REGRESSION)} "
            "frozen_positions=34 process_start_count=0",
            flush=True,
        )
        return 0

    runner.scale_regression = scale_regression_0009

    def source_records_0009(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(
            model, tokenizer_json, tokenizer_config, prompts
        )
        for path in (
            SCALE_REGRESSION,
            PREDECESSOR_LAUNCHER,
            Path(rank1_reference.__file__).resolve(strict=True),
        ):
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0009

    def execution_bound_paths_0009(
        prompts_path: Path,
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        *,
        include_frozen_stage_files: bool,
    ) -> list[Path]:
        paths = prior_execution_bound_paths(
            prompts_path,
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=include_frozen_stage_files,
        )
        return sorted(
            {
                *paths,
                SCALE_REGRESSION.resolve(strict=True),
                PREDECESSOR_LAUNCHER.resolve(strict=True),
                Path(rank1_reference.__file__).resolve(strict=True),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0009

    def package_template_0009(prompts_path: Path) -> int:
        runner.validate_no_execution()
        scale = verify_scale_regression()
        result = prior_package_template(prompts_path)
        gate = runner.read_json(runner.TEMPLATE_GATE)
        gate["retained_scale_regression"] = runner.file_record(SCALE_REGRESSION)
        gate["retained_scale_contract"] = scale["proof"]["conversion_contract"]
        gate["checks"]["retained_scale_integer_oracle_passed"] = True
        gate["checks"]["all_frozen_positions_scale_aligned"] = True
        runner.write_json(runner.TEMPLATE_GATE, gate)
        return result

    runner.package_template_invariant = package_template_0009

    def preflight_0009(prompts_path: Path) -> int:
        scale = verify_scale_regression()
        result = prior_preflight(prompts_path)
        record = runner.read_json(runner.PREFLIGHT)
        record["retained_scale_regression"] = runner.file_record(SCALE_REGRESSION)
        record["retained_scale_contract"] = scale["proof"]["conversion_contract"]
        record["checks"]["retained_scale_integer_oracle_passed"] = True
        record["checks"]["projection_metadata_interface_sufficient"] = True
        runner.write_json(runner.PREFLIGHT, record)
        return result

    runner.preflight = preflight_0009

    def freeze_0009(prompts_path: Path) -> int:
        scale = verify_scale_regression()
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["retained_scale_regression"] = runner.file_record(SCALE_REGRESSION)
        frozen["retained_scale_alignment"] = scale["proof"]["conversion_contract"]
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0009

    class RetainedScaleProjectionCache(prior_cache_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._source_v_metadata: dict[str, Any] | None = None

        @staticmethod
        def _is_layer23_v(name: str) -> bool:
            return runner.re.fullmatch(r"layer23_position\d+_v", name) is not None

        def _attach_alignment(self, result: dict[str, Any]) -> dict[str, Any]:
            source = self._source_v_metadata
            runner.require(source is not None, "dynamic source V metadata is absent")
            accumulators = [
                int(value) for value in result["accumulator"].reshape(-1).tolist()
            ]
            multipliers = [
                int(value) for value in result["multiplier"].reshape(-1).tolist()
            ]
            shifts = [
                int(value) for value in result["right_shift"].reshape(-1).tolist()
            ]
            oracle_bytes, oracle_saturation = _oracle_requantize(
                accumulators, multipliers, shifts
            )
            production_bytes = [
                int(value) for value in result["output_q"].reshape(-1).tolist()
            ]
            production_saturation = [
                bool(value) for value in result["saturation"].reshape(-1).tolist()
            ]
            runner.require(
                production_bytes == oracle_bytes
                and production_saturation == oracle_saturation,
                "retained projection bytes differ from independent integer oracle",
            )
            source_bytes, source_saturation = _oracle_requantize(
                accumulators,
                source["multiplier_s32"],
                source["right_shift_u6"],
            )
            source_ratio = float(source["output_scale"]) / RETAINED_OUTPUT_SCALE
            ratio_multiplier, ratio_shift = runner.backend.canonical.derive_multiplier(
                runner.torch.tensor(source_ratio, dtype=runner.torch.float64)
            )
            oracle_ratio_multiplier, oracle_ratio_shift = _oracle_derive_multiplier(
                source_ratio
            )
            runner.require(
                int(ratio_multiplier.item()) == oracle_ratio_multiplier
                and int(ratio_shift.item()) == oracle_ratio_shift,
                "source-to-retained Scale32 metadata differs from oracle",
            )
            result = dict(result)
            result["retained_scale_alignment"] = {
                "method": "direct_projection_output_requantization",
                "source_scale": float(source["output_scale"]),
                "retained_output_scale": RETAINED_OUTPUT_SCALE,
                "source_to_retained_scale32": {
                    "real_multiplier_f64": source_ratio,
                    "multiplier_s32": oracle_ratio_multiplier,
                    "right_shift_u6": oracle_ratio_shift,
                },
                "projection_multiplier_s32": multipliers,
                "projection_right_shift_u6": shifts,
                "source_dynamic_bytes_s8": source_bytes,
                "source_dynamic_saturation": source_saturation,
                "converted_bytes_s8": production_bytes,
                "converted_saturation": production_saturation,
                "converted_saturation_count": sum(production_saturation),
                "independent_integer_oracle_match": True,
                "product_width": "signed_int64",
                "rounding": "ties_to_even",
                "output_saturation": "signed_int8",
            }
            return result

        def _apply(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
            alignment = result.get("retained_scale_alignment")
            converted = super()._apply(name, result)
            if alignment is not None:
                record = self.records[-1]
                alignment_record = dict(alignment)
                alignment_record["converted_bytes_sha256"] = runner.sha256_bytes(
                    _signed_bytes(alignment_record["converted_bytes_s8"])
                )
                alignment_record["software_reference_baseline_sha256"] = record[
                    "baseline_v_sha256"
                ]
                alignment_record["ace2_shell_shared_payload_verified"] = (
                    self.mode == "rtl"
                )
                runner.require(
                    alignment_record["converted_bytes_sha256"]
                    == record["baseline_v_sha256"],
                    "retained conversion bytes differ from sidecar baseline",
                )
                record["retained_scale_alignment"] = alignment_record
            return converted

        def derive_projection(
            self,
            name: str,
            merged: Any,
            input_q: Any,
            input_scale: float,
            float_input: Any,
            source_hashes: dict[str, str],
        ) -> dict[str, Any]:
            if not self._is_layer23_v(name):
                return super().derive_projection(
                    name,
                    merged,
                    input_q,
                    input_scale,
                    float_input,
                    source_hashes,
                )
            source = runner.localizer.FastProjectionCache.derive_projection(
                self,
                name,
                merged,
                input_q,
                input_scale,
                float_input,
                source_hashes,
            )
            weight_scale = source["weight_scale"].to(runner.torch.float64)
            retained_real = (
                float(input_scale) * weight_scale / RETAINED_OUTPUT_SCALE
            )
            multiplier, right_shift = runner.backend.canonical.derive_multiplier(
                retained_real
            )
            self._source_v_metadata = {
                "output_scale": float(source["output_scale"]),
                "multiplier_s32": [
                    int(value)
                    for value in source["multiplier"].reshape(-1).tolist()
                ],
                "right_shift_u6": [
                    int(value)
                    for value in source["right_shift"].reshape(-1).tolist()
                ],
            }
            retained = runner.localizer.FastProjectionCache._fixed(
                name,
                input_q,
                input_scale,
                source["qweight"],
                multiplier,
                right_shift,
                RETAINED_OUTPUT_SCALE,
            )
            retained.update(
                {
                    "weight_scale": source["weight_scale"],
                    "float_output": source["float_output"],
                    "source_hashes": source_hashes,
                }
            )
            return self._apply(name, self._attach_alignment(retained))

        def from_fixed_metadata(
            self,
            name: str,
            input_q: Any,
            input_scale: float,
            qweight: Any,
            multiplier: Any,
            right_shift: Any,
            output_scale: float,
        ) -> dict[str, Any]:
            if not self._is_layer23_v(name):
                return super().from_fixed_metadata(
                    name,
                    input_q,
                    input_scale,
                    qweight,
                    multiplier,
                    right_shift,
                    output_scale,
                )
            runner.require(
                output_scale == RETAINED_OUTPUT_SCALE,
                "fixed V projection output scale is not retained",
            )
            retained = runner.localizer.FastProjectionCache.from_fixed_metadata(
                self,
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            )
            return self._apply(name, self._attach_alignment(retained))

    runner.Rank1HybridProjectionCache = RetainedScaleProjectionCache

    def compare_runs_0009(
        rtl: dict[str, Any], software: dict[str, Any]
    ) -> dict[str, Any]:
        comparison = prior_compare_runs(rtl, software)
        for index, (rtl_v, software_v) in enumerate(
            zip(
                rtl["layer23_v_records"],
                software["layer23_v_records"],
                strict=True,
            )
        ):
            rtl_scale = rtl_v["retained_scale_alignment"]
            software_scale = software_v["retained_scale_alignment"]
            comparable_keys = (
                "source_scale",
                "retained_output_scale",
                "source_to_retained_scale32",
                "projection_multiplier_s32",
                "projection_right_shift_u6",
                "source_dynamic_bytes_s8",
                "source_dynamic_saturation",
                "converted_bytes_s8",
                "converted_saturation",
                "converted_saturation_count",
                "converted_bytes_sha256",
                "software_reference_baseline_sha256",
                "independent_integer_oracle_match",
            )
            runner.require(
                all(rtl_scale[key] == software_scale[key] for key in comparable_keys),
                f"retained-scale RTL/software record differs at step {index}",
            )
            runner.require(
                rtl_scale["ace2_shell_shared_payload_verified"] is True
                and software_scale["ace2_shell_shared_payload_verified"] is False,
                f"retained-scale shell/software source classification differs at step {index}",
            )
            comparison["steps"][index].update(
                {
                    "retained_scale_metadata_equal": True,
                    "retained_baseline_128_bytes_equal": True,
                    "independent_scale_oracle_equal": True,
                    "ace2_shell_shared_payload_retained_scale": True,
                }
            )
        comparison["retained_scale_alignment_all_steps"] = True
        return comparison

    runner.compare_runs = compare_runs_0009

    def child_entry_attestation_0009(
        prompts_path: Path, expected_freeze_sha256: str
    ) -> dict[str, Any]:
        runner.require(
            SCALE_REGRESSION.is_file(),
            "retained-scale regression is absent at child entry",
        )
        attestation = prior_child_entry_attestation(
            prompts_path, expected_freeze_sha256
        )
        frozen = runner.read_json(runner.FREEZE)
        runner.require(
            frozen.get("retained_scale_regression")
            == runner.file_record(SCALE_REGRESSION)
            and frozen.get("retained_scale_alignment")
            == verify_scale_regression()["proof"]["conversion_contract"],
            "frozen retained-scale binding changed at child entry",
        )
        attestation["retained_scale_regression_sha256"] = runner.sha256_file(
            SCALE_REGRESSION
        )
        return attestation

    runner.child_entry_attestation = child_entry_attestation_0009

    def execute_0009(prompts_path: Path, expected_freeze_sha256: str) -> int:
        child_entry_attestation_0009(prompts_path, expected_freeze_sha256)
        return prior_execute(prompts_path, expected_freeze_sha256)

    runner.execute = execute_0009

    def verify_0009() -> int:
        result = prior_verify()
        verify_scale_regression()
        execution_result = runner.read_json(runner.RESULT)
        records = [
            record
            for prompt in execution_result["prompts"]
            for mode in ("rtl_hybrid", "software_integer_reference")
            for record in prompt[mode]["layer23_v_records"]
        ]
        runner.require(
            len(records) == 2 * 2 * runner.STEPS,
            "retained-scale fresh prompt record count changed",
        )
        runner.require(
            all(
                record["retained_scale_alignment"]["retained_output_scale"]
                == RETAINED_OUTPUT_SCALE
                and len(
                    record["retained_scale_alignment"]["converted_bytes_s8"]
                )
                == 128
                for record in records
            ),
            "retained-scale fresh prompt record shape changed",
        )
        print(
            "ACE2_HYBRID_RETAINED_SCALE_DECISIVE_VERIFY_PASS "
            f"regression_sha256={runner.sha256_file(SCALE_REGRESSION)} "
            "fresh_prompt_steps=8",
            flush=True,
        )
        return result

    runner.verify = verify_0009


def main() -> int:
    try:
        _verify_pre_import_launch()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as predecessor

        _bind_predecessor_module(predecessor)
        post_import_environment = dict(os.environ)
        _configure_runner(
            runner,
            predecessor,
            ace2_layer23_v_rank1_integer_correction_reference,
            post_import_environment,
        )
        if "--tokenization-preflight" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--tokenization-preflight", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_preflight(arguments.prompts.resolve())
        if "--check-tokenization-verifier" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-tokenization-verifier", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_verifier_regression(
                arguments.prompts.resolve()
            )
        if "--check-scale-alignment" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-scale-alignment", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.scale_regression(arguments.prompts.resolve())
        return runner.main()
    except BaseException as error:
        _seal_preloader_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
