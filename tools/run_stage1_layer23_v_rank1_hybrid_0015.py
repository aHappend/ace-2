#!/usr/bin/env python3
"""Fixed-input-scale launcher for nonofficial hybrid attempt 0015."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0015"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0015.json"
SELECTION = PRIVATE / "candidate-selection-0015.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0015.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0015.json"
INPUT_SCALE_REGRESSION = PRIVATE / "fixed-input-scale-regression-0015.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0015.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0015.json"
GATE_ORDER_REGRESSION = PRIVATE / "gate-order-regression-0015.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0015.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0015.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0015.json"
FROZEN_INPUT_SCALE = 0.25991058349609375

PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)

SEALED_0014_TERMINAL_SHA256 = (
    "1ed9c0d0ed75566c71bb1fe32c9aaa8e8093ec3b78807a1eb9758cf55eecbdff"
)
SEALED_0014_LAUNCHER_SHA256 = (
    "c985b1782dec9feb21506ffe61fd509055b2dba3eae5a40cfcdba50fdba7528d"
)

CANDIDATES = (
    {"id": "fresh-0015-a", "text": "Why do leaves curl?"},
    {"id": "fresh-0015-b", "text": "How do puddles dry?"},
    {"id": "fresh-0015-c", "text": "Where do moths rest?"},
    {"id": "fresh-0015-d", "text": "Why does clay crack?"},
    {"id": "fresh-0015-e", "text": "How do reeds bend?"},
    {"id": "fresh-0015-f", "text": "What makes bark rough?"},
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sanitized_traceback(error: BaseException) -> dict[str, Any]:
    frames = []
    rendered = ["Traceback (most recent call last):"]
    for frame in traceback.extract_tb(error.__traceback__):
        path = Path(frame.filename)
        try:
            source_file = str(path.resolve().relative_to(ROOT))
        except (OSError, ValueError):
            source_file = f"<external>/{path.name}"
        record = {
            "source_file": source_file,
            "function": frame.name,
            "line": frame.lineno,
            "expression": frame.line or "",
        }
        frames.append(record)
        rendered.append(
            f'  File "{source_file}", line {frame.lineno}, in {frame.name}'
        )
        if frame.line:
            rendered.append(f"    {frame.line}")
    rendered.append(f"{type(error).__name__}: {error}")
    return {
        "exception_type": type(error).__name__,
        "message": str(error),
        "frames": frames,
        "formatted": "\n".join(rendered) + "\n",
    }


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid15",
        "attempt_identity": "nonofficial-hybrid-0015",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_nonconsuming_gate",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "traceback": _sanitized_traceback(error),
            "regression": "No retry, replay, or resume; preserve all 0015 artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _signed_bytes(values: list[int] | tuple[int, ...]) -> bytes:
    return bytes(int(value) & 0xFF for value in values)


def _configure_0014(prior: Any) -> None:
    prior.LAUNCHER = LAUNCHER
    prior.OUTPUT = OUTPUT
    prior.TERMINAL = TERMINAL
    prior.PRIVATE = PRIVATE
    prior.PROMPTS = PROMPTS
    prior.SELECTION = SELECTION
    prior.TOKENIZATION_PREFLIGHT = TOKENIZATION_PREFLIGHT
    prior.TOKENIZATION_REGRESSION = TOKENIZATION_REGRESSION
    prior.SCALE_REGRESSION = SCALE_REGRESSION
    prior.LIFECYCLE_REGRESSION = LIFECYCLE_REGRESSION
    prior.GATE_ORDER_REGRESSION = GATE_ORDER_REGRESSION
    prior.TEMPLATE_GATE = TEMPLATE_GATE
    prior.DEPENDENCY_PROBE = DEPENDENCY_PROBE
    prior.LAUNCH_REGRESSION = LAUNCH_REGRESSION
    prior.CANDIDATES = CANDIDATES
    prior.DIAGNOSTIC_PROMPT_SOURCES = (
        *prior.DIAGNOSTIC_PROMPT_SOURCES,
        PRIVATE / "prompts-0014.json",
    )
    prior.SEALED_BOUNDARY_SHA256 = {
        **prior.SEALED_BOUNDARY_SHA256,
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0014/terminal-record.json"
        ): SEALED_0014_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0014.py": (
            SEALED_0014_LAUNCHER_SHA256
        ),
    }
    prior.PRE_IMPORT_ENVIRONMENT = PRE_IMPORT_ENVIRONMENT
    prior.PRE_IMPORT_SYS_PATH = PRE_IMPORT_SYS_PATH
    prior.PRE_IMPORT_SYS_PREFIX = PRE_IMPORT_SYS_PREFIX
    prior.PRE_IMPORT_ARGV = PRE_IMPORT_ARGV
    prior.PRE_IMPORT_CWD = PRE_IMPORT_CWD
    prior.PRE_IMPORT_MODULES = PRE_IMPORT_MODULES


def _configure_runner(
    runner: Any,
    prior: Any,
    inherited_prior: Any,
    prior_base: Any,
    base: Any,
    rank1_reference: Any,
    input_scale_oracle: Any,
    post_import_environment: dict[str, str],
) -> None:
    prior._configure_runner(
        runner,
        inherited_prior,
        prior_base,
        base,
        rank1_reference,
        post_import_environment,
    )
    runner.MISSION_ID = "stage1rank1hybrid15"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0015"
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
    runner.DEFAULT_PROMPTS = PROMPTS
    runner.TEMPLATE_GATE = TEMPLATE_GATE
    runner.DEPENDENCY_PROBE = DEPENDENCY_PROBE
    runner.PREDECESSOR_TERMINAL_SEALS = {
        **runner.PREDECESSOR_TERMINAL_SEALS,
        "nonofficial-hybrid-0014": {
            "sha256": SEALED_0014_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_dependency_probe = runner.dependency_probe
    prior_tokenization_preflight = runner.tokenization_preflight
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_package_template = runner.package_template_invariant
    prior_preflight = runner.preflight
    prior_freeze = runner.freeze
    prior_authority = runner.grant_authority
    prior_child_entry_attestation = runner.child_entry_attestation
    prior_execute = runner.execute
    prior_compare_runs = runner.compare_runs
    prior_verify = runner.verify
    inherited_cache_class = runner.Rank1HybridProjectionCache

    terminal_0014_path = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0014/terminal-record.json"
    )
    durable_0014_path = terminal_0014_path.parent / "supervisor-state/terminal_record.json"

    def verify_0014_seal() -> dict[str, Any]:
        for path in (terminal_0014_path, durable_0014_path):
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0014 terminal is not regular: {runner.public_path(path)}",
            )
            runner.require(
                runner.sha256_file(path) == SEALED_0014_TERMINAL_SHA256,
                f"sealed 0014 terminal changed: {runner.public_path(path)}",
            )
        terminal = runner.read_json(terminal_0014_path)
        runner.require(
            terminal == runner.read_json(durable_0014_path)
            and terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("outcome") == "child_nonzero_exit",
            "sealed 0014 exactly-once terminal state changed",
        )
        return runner.file_record(terminal_0014_path)

    def verify_predecessor_seals_0015() -> None:
        prior_verify_predecessor_seals()
        verify_0014_seal()

    runner.verify_predecessor_seals = verify_predecessor_seals_0015

    def frozen_input_scale() -> float:
        freeze, _review = rank1_reference.verify_candidate_binding(check_review=True)
        scale = float(
            freeze["quantization"]["input_payload"]["scale32"]["decoded_value"]
        )
        runner.require(
            scale == FROZEN_INPUT_SCALE,
            "candidate frozen layer-23 V input scale changed",
        )
        return scale

    def production_input_conversion(
        input_q: Any,
        source_scale: float,
    ) -> tuple[Any, dict[str, Any]]:
        target_scale = frozen_input_scale()
        source_values = [
            int(value) for value in input_q.to(runner.torch.int64).reshape(-1).tolist()
        ]
        runner.require(
            source_values
            and all(-128 <= value <= 127 for value in source_values),
            "layer-23 V source input escaped signed int8",
        )
        real_multiplier = float(source_scale) / target_scale
        multiplier, right_shift = runner.backend.canonical.derive_multiplier(
            runner.torch.tensor(real_multiplier, dtype=runner.torch.float64)
        )
        multiplier_s32 = int(multiplier.item())
        right_shift_u6 = int(right_shift.item())
        runner.require(
            0 <= multiplier_s32 < (1 << 31) and 0 <= right_shift_u6 <= 63,
            "layer-23 V input conversion metadata is not signed-int32/u6",
        )
        values = runner.torch.tensor(source_values, dtype=runner.torch.int64)
        products = values * multiplier_s32
        shifts = runner.torch.full_like(values, right_shift_u6)
        rounded = runner.localizer.round_shift_even_tensor(products, shifts)
        converted = rounded.clamp(-128, 127).to(runner.torch.int8)
        converted_values = [int(value) for value in converted.tolist()]
        saturation = [
            bool(value) for value in ((rounded < -128) | (rounded > 127)).tolist()
        ]
        oracle = input_scale_oracle.requantize_s8(
            source_values,
            float(source_scale),
            target_scale,
        )
        runner.require(
            oracle["multiplier_s32"] == multiplier_s32
            and oracle["right_shift_u6"] == right_shift_u6
            and oracle["converted_s8"] == converted_values
            and oracle["saturation"] == saturation,
            "production layer-23 V input conversion differs from independent oracle",
        )
        source_hash = runner.sha256_bytes(_signed_bytes(source_values))
        converted_hash = runner.sha256_bytes(_signed_bytes(converted_values))
        evidence = {
            "method": "explicit_integer_source_to_frozen_input_requantization",
            "source_input_scale": float(source_scale),
            "target_frozen_input_scale": target_scale,
            "source_to_frozen_scale32": {
                "real_multiplier_f64": real_multiplier,
                "multiplier_s32": multiplier_s32,
                "right_shift_u6": right_shift_u6,
            },
            "source_dynamic_bytes_s8": source_values,
            "converted_fixed_bytes_s8": converted_values,
            "source_dynamic_bytes_sha256": source_hash,
            "converted_fixed_bytes_sha256": converted_hash,
            "converted_saturation": saturation,
            "converted_saturation_count": sum(saturation),
            "product_width": "signed_int64",
            "rounding": "ties_to_even",
            "output_saturation": "signed_int8",
            "independent_integer_oracle_match": True,
            "software_projection_input_sha256": converted_hash,
            "frozen_integer_reference_input_sha256": converted_hash,
            "ace2_shell_input_sha256": converted_hash,
        }
        return converted.reshape(input_q.shape), evidence

    class FixedInputScaleProjectionCache(inherited_cache_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._pending_input_alignment: dict[str, dict[str, Any]] = {}

        @staticmethod
        def _is_layer23_v(name: str) -> bool:
            return runner.re.fullmatch(r"layer23_position\d+_v", name) is not None

        def _apply(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
            alignment = self._pending_input_alignment.get(name)
            records_before = len(self.records)
            converted = super()._apply(name, result)
            records_after = len(self.records)
            runner.require(
                records_after in {records_before, records_before + 1},
                "fixed-input-scale sidecar record lifecycle changed",
            )
            if alignment is None or records_after == records_before:
                return converted
            actual_values = [
                int(value)
                for value in result["input_q"].to(runner.torch.int8).reshape(-1).tolist()
            ]
            actual_hash = runner.sha256_bytes(_signed_bytes(actual_values))
            runner.require(
                float(result["input_scale"]) == FROZEN_INPUT_SCALE
                and actual_hash == alignment["converted_fixed_bytes_sha256"]
                and actual_values == alignment["converted_fixed_bytes_s8"],
                "baseline V projection did not consume converted fixed-scale bytes",
            )
            record = self.records[-1]
            record_alignment = dict(alignment)
            record_alignment["baseline_projection_actual_input_sha256"] = actual_hash
            record_alignment["ace2_shell_shared_payload_verified"] = self.mode == "rtl"
            if self.mode == "rtl":
                runner.require(
                    record["downstream_v_source"]
                    == "icarus_ace2_shell_mem_wdata",
                    "RTL V result did not come from ace2_shell memory write data",
                )
                record_alignment["rtl_activation_artifact"] = record["artifacts"][
                    "activation_s8"
                ]
            record["fixed_input_scale_alignment"] = record_alignment
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
            fixed_input, alignment = production_input_conversion(input_q, input_scale)
            self._pending_input_alignment[name] = alignment
            try:
                return super().derive_projection(
                    name,
                    merged,
                    fixed_input,
                    FROZEN_INPUT_SCALE,
                    float_input,
                    source_hashes,
                )
            finally:
                self._pending_input_alignment.pop(name, None)

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
            fixed_input, alignment = production_input_conversion(input_q, input_scale)
            self._pending_input_alignment[name] = alignment
            try:
                return super().from_fixed_metadata(
                    name,
                    fixed_input,
                    FROZEN_INPUT_SCALE,
                    qweight,
                    multiplier,
                    right_shift,
                    output_scale,
                )
            finally:
                self._pending_input_alignment.pop(name, None)

    runner.Rank1HybridProjectionCache = FixedInputScaleProjectionCache

    def build_input_scale_regression() -> dict[str, Any]:
        target_scale = frozen_input_scale()
        vectors = rank1_reference.load_frozen_position_vectors(check_review=True)
        frozen_records = []
        diagnostic_specs: list[tuple[str, float, list[int], list[int]]] = [
            (
                "positive_negative_half_ties",
                target_scale * 0.5,
                [0, 1, -1, 3, -3, 5, -5],
                [0, 0, 0, 2, -2, 2, -2],
            ),
            (
                "positive_negative_saturation",
                target_scale * 2.0,
                [0, 63, 64, 127, -64, -65, -128],
                [0, 126, 127, 127, -128, -128, -128],
            ),
            (
                "bounded_u6_shift",
                target_scale * (2.0**-32),
                [0, 1, -1, 127, -128],
                [0, 0, 0, 0, 0],
            ),
        ]
        for vector in vectors:
            values = [int(value) for value in vector.input_s8]
            tensor = runner.torch.tensor(values, dtype=runner.torch.int8)
            converted, evidence = production_input_conversion(tensor, target_scale)
            converted_values = [int(value) for value in converted.tolist()]
            runner.require(
                converted_values == values
                and evidence["converted_saturation_count"] == 0,
                f"old frozen position {vector.position} identity conversion changed",
            )
            frozen_records.append(
                {
                    "position": vector.position,
                    "source_input_scale": target_scale,
                    "target_frozen_input_scale": target_scale,
                    "multiplier_s32": evidence["source_to_frozen_scale32"][
                        "multiplier_s32"
                    ],
                    "right_shift_u6": evidence["source_to_frozen_scale32"][
                        "right_shift_u6"
                    ],
                    "source_sha256": evidence["source_dynamic_bytes_sha256"],
                    "converted_sha256": evidence["converted_fixed_bytes_sha256"],
                    "exact_original_frozen_bytes_preserved": True,
                }
            )
            bounded_target = [max(-63, min(63, value)) for value in values]
            dynamic_source = [value * 2 for value in bounded_target]
            diagnostic_specs.append(
                (
                    f"old_frozen_position_{vector.position}_dynamic_half_scale",
                    target_scale * 0.5,
                    dynamic_source,
                    bounded_target,
                )
            )

        diagnostics = []
        for name, source_scale, source_values, expected_values in diagnostic_specs:
            tensor = runner.torch.tensor(source_values, dtype=runner.torch.int8)
            converted, evidence = production_input_conversion(tensor, source_scale)
            converted_values = [int(value) for value in converted.tolist()]
            runner.require(
                source_values != expected_values,
                f"legacy dynamic bytes unexpectedly equal frozen reference: {name}",
            )
            runner.require(
                converted_values == expected_values,
                f"repaired bytes differ from frozen reference: {name}",
            )
            converted_hash = evidence["converted_fixed_bytes_sha256"]
            runner.require(
                evidence["software_projection_input_sha256"] == converted_hash
                and evidence["frozen_integer_reference_input_sha256"]
                == converted_hash
                and evidence["ace2_shell_input_sha256"] == converted_hash,
                f"fixed input byte consumers differ: {name}",
            )
            diagnostics.append(
                {
                    "case": name,
                    **evidence,
                    "legacy_dynamic_bytes_fail_frozen_reference_equality": True,
                    "repaired_bytes_match_frozen_reference": True,
                    "expected_frozen_bytes_s8": expected_values,
                }
            )
        return {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_LEGACY_DYNAMIC_FAILS_FIXED_INPUT_MATCHES",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "conversion_contract": {
                "source_domain": "dynamic_signed_int8_activation",
                "target_frozen_input_scale": target_scale,
                "metadata": "signed_int32_multiplier_and_u6_right_shift",
                "product_width": "signed_int64",
                "rounding": "ties_to_even",
                "output": "signed_int8_saturation",
                "baseline_projection_input": "converted_fixed_bytes",
                "frozen_integer_reference_input": "converted_fixed_bytes",
                "ace2_shell_input": "converted_fixed_bytes",
            },
            "old_frozen_positions": frozen_records,
            "diagnostic_vectors": diagnostics,
            "sources": {
                "launcher": runner.file_record(LAUNCHER),
                "independent_oracle": runner.file_record(
                    Path(input_scale_oracle.__file__).resolve(strict=True)
                ),
                "rank1_reference": runner.file_record(
                    Path(rank1_reference.__file__).resolve(strict=True)
                ),
                "sealed_0014_terminal": verify_0014_seal(),
            },
        }

    def input_scale_regression() -> int:
        runner.validate_no_execution()
        runner.require(
            DEPENDENCY_PROBE.is_file() and GATE_ORDER_REGRESSION.is_file(),
            "fixed-input-scale regression requires dependency and gate-order gates",
        )
        runner.require(
            not INPUT_SCALE_REGRESSION.exists(),
            "0015 fixed-input-scale regression already exists",
        )
        runner.write_json(INPUT_SCALE_REGRESSION, build_input_scale_regression())
        print(
            "ACE2_HYBRID_0015_FIXED_INPUT_SCALE_REGRESSION_PASS "
            f"sha256={runner.sha256_file(INPUT_SCALE_REGRESSION)}",
            flush=True,
        )
        return 0

    def verify_input_scale_regression() -> dict[str, Any]:
        runner.require(
            INPUT_SCALE_REGRESSION.is_file(),
            "0015 fixed-input-scale regression is absent",
        )
        record = runner.read_json(INPUT_SCALE_REGRESSION)
        expected = build_input_scale_regression()
        runner.require(
            record == expected
            and record["status"]
            == "PASS_LEGACY_DYNAMIC_FAILS_FIXED_INPUT_MATCHES"
            and record["model_executed"] is False
            and record["simulator_executed"] is False
            and record["process_start_count"] == 0
            and all(
                item["exact_original_frozen_bytes_preserved"]
                for item in record["old_frozen_positions"]
            )
            and all(
                item["legacy_dynamic_bytes_fail_frozen_reference_equality"]
                and item["repaired_bytes_match_frozen_reference"]
                for item in record["diagnostic_vectors"]
            ),
            "0015 fixed-input-scale regression artifact changed",
        )
        return record

    runner.input_scale_regression = input_scale_regression
    runner.verify_input_scale_regression = verify_input_scale_regression

    def dependency_probe_0015() -> int:
        runner.require(
            not INPUT_SCALE_REGRESSION.exists(),
            "fixed-input-scale regression exists before dependency probe",
        )
        result = prior_dependency_probe()
        record = runner.read_json(DEPENDENCY_PROBE)
        record["top_level_import_chain"].append(
            {
                "module": "tools.ace2_layer23_v_input_scale_oracle",
                "file": runner.file_record(
                    Path(input_scale_oracle.__file__).resolve(strict=True)
                ),
            }
        )
        record["checks"]["fixed_input_scale_oracle_source_bound"] = True
        record["checks"]["fixed_input_scale_regression_absent"] = True
        runner.write_json(DEPENDENCY_PROBE, record)
        return result

    runner.dependency_probe = dependency_probe_0015

    def tokenization_preflight_0015(prompts_path: Path) -> int:
        verify_input_scale_regression()
        return prior_tokenization_preflight(prompts_path)

    runner.tokenization_preflight = tokenization_preflight_0015

    def source_records_0015(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        for path in (
            INPUT_SCALE_REGRESSION,
            Path(input_scale_oracle.__file__).resolve(strict=True),
        ):
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0015

    def execution_bound_paths_0015(
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
                INPUT_SCALE_REGRESSION.resolve(strict=True),
                Path(input_scale_oracle.__file__).resolve(strict=True),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0015

    def package_template_0015(prompts_path: Path) -> int:
        scale = verify_input_scale_regression()
        result = prior_package_template(prompts_path)
        gate = runner.read_json(TEMPLATE_GATE)
        gate["fixed_input_scale_regression"] = runner.file_record(
            INPUT_SCALE_REGRESSION
        )
        gate["fixed_input_scale_conversion"] = scale["conversion_contract"]
        gate["checks"]["fixed_input_scale_byte_consumers_equal"] = True
        runner.write_json(TEMPLATE_GATE, gate)
        return result

    runner.package_template_invariant = package_template_0015

    def preflight_0015(prompts_path: Path) -> int:
        scale = verify_input_scale_regression()
        result = prior_preflight(prompts_path)
        record = runner.read_json(runner.PREFLIGHT)
        record["fixed_input_scale_regression"] = runner.file_record(
            INPUT_SCALE_REGRESSION
        )
        record["fixed_input_scale_conversion"] = scale["conversion_contract"]
        record["checks"]["fixed_input_scale_byte_consumers_equal"] = True
        runner.write_json(runner.PREFLIGHT, record)
        return result

    runner.preflight = preflight_0015

    def freeze_0015(prompts_path: Path) -> int:
        scale = verify_input_scale_regression()
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["fixed_input_scale_regression"] = runner.file_record(
            INPUT_SCALE_REGRESSION
        )
        frozen["fixed_input_scale_conversion"] = scale["conversion_contract"]
        frozen["sealed_0014_terminal"] = verify_0014_seal()
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0015

    def authority_0015(prompts_path: Path) -> int:
        verify_input_scale_regression()
        return prior_authority(prompts_path)

    runner.grant_authority = authority_0015

    def child_entry_attestation_0015(
        prompts_path: Path,
        expected_freeze_sha256: str,
    ) -> dict[str, Any]:
        scale = verify_input_scale_regression()
        attestation = prior_child_entry_attestation(
            prompts_path, expected_freeze_sha256
        )
        frozen = runner.read_json(runner.FREEZE)
        runner.require(
            frozen.get("fixed_input_scale_regression")
            == runner.file_record(INPUT_SCALE_REGRESSION)
            and frozen.get("fixed_input_scale_conversion")
            == scale["conversion_contract"],
            "frozen fixed-input-scale binding changed at child entry",
        )
        attestation["fixed_input_scale_regression_sha256"] = runner.sha256_file(
            INPUT_SCALE_REGRESSION
        )
        return attestation

    runner.child_entry_attestation = child_entry_attestation_0015

    def execute_0015(prompts_path: Path, expected_freeze_sha256: str) -> int:
        child_entry_attestation_0015(prompts_path, expected_freeze_sha256)
        return prior_execute(prompts_path, expected_freeze_sha256)

    runner.execute = execute_0015

    def compare_runs_0015(
        rtl: dict[str, Any],
        software: dict[str, Any],
    ) -> dict[str, Any]:
        comparison = prior_compare_runs(rtl, software)
        comparable_keys = (
            "source_input_scale",
            "target_frozen_input_scale",
            "source_to_frozen_scale32",
            "source_dynamic_bytes_s8",
            "converted_fixed_bytes_s8",
            "source_dynamic_bytes_sha256",
            "converted_fixed_bytes_sha256",
            "converted_saturation",
            "converted_saturation_count",
            "software_projection_input_sha256",
            "frozen_integer_reference_input_sha256",
            "ace2_shell_input_sha256",
            "baseline_projection_actual_input_sha256",
            "independent_integer_oracle_match",
        )
        for index, (rtl_v, software_v) in enumerate(
            zip(
                rtl["layer23_v_records"],
                software["layer23_v_records"],
                strict=True,
            )
        ):
            rtl_scale = rtl_v["fixed_input_scale_alignment"]
            software_scale = software_v["fixed_input_scale_alignment"]
            runner.require(
                all(rtl_scale[key] == software_scale[key] for key in comparable_keys),
                f"fixed-input-scale RTL/software record differs at step {index}",
            )
            runner.require(
                rtl_scale["ace2_shell_shared_payload_verified"] is True
                and software_scale["ace2_shell_shared_payload_verified"] is False,
                f"fixed-input-scale shell source classification differs at step {index}",
            )
            comparison["steps"][index].update(
                {
                    "fixed_input_scale_metadata_equal": True,
                    "fixed_input_activation_896_bytes_equal": True,
                    "baseline_reference_shell_input_hash_equal": True,
                    "independent_input_scale_oracle_equal": True,
                }
            )
        comparison["fixed_input_scale_alignment_all_steps"] = True
        return comparison

    runner.compare_runs = compare_runs_0015

    def verify_0015() -> int:
        result = prior_verify()
        verify_input_scale_regression()
        execution_result = runner.read_json(runner.RESULT)
        records = [
            record
            for prompt in execution_result["prompts"]
            for mode in ("rtl_hybrid", "software_integer_reference")
            for record in prompt[mode]["layer23_v_records"]
        ]
        runner.require(
            len(records) == 2 * 2 * runner.STEPS,
            "fixed-input-scale fresh prompt record count changed",
        )
        for record in records:
            alignment = record["fixed_input_scale_alignment"]
            converted_hash = alignment["converted_fixed_bytes_sha256"]
            runner.require(
                alignment["target_frozen_input_scale"] == FROZEN_INPUT_SCALE
                and len(alignment["source_dynamic_bytes_s8"]) == 896
                and len(alignment["converted_fixed_bytes_s8"]) == 896
                and alignment["software_projection_input_sha256"] == converted_hash
                and alignment["frozen_integer_reference_input_sha256"]
                == converted_hash
                and alignment["ace2_shell_input_sha256"] == converted_hash
                and alignment["baseline_projection_actual_input_sha256"]
                == converted_hash,
                "fixed-input-scale per-step evidence changed",
            )
        print(
            "ACE2_HYBRID_0015_DECISIVE_VERIFY_PASS "
            "prompt_count=2 generated_tokens_per_prompt=4 fixed_input_scale_steps=16",
            flush=True,
        )
        return result

    runner.verify = verify_0015


def main() -> int:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid_0010 as prior_base
        from tools import run_stage1_layer23_v_rank1_hybrid_0011 as inherited_prior
        from tools import run_stage1_layer23_v_rank1_hybrid_0014 as prior

        _configure_0014(prior)
        prior._configure_prior(inherited_prior)
        inherited_prior._configure_prior(prior_base)
        prior_base._verify_pre_import_launch()

        from tools import ace2_layer23_v_input_scale_oracle
        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as base

        _configure_runner(
            runner,
            prior,
            inherited_prior,
            prior_base,
            base,
            ace2_layer23_v_rank1_integer_correction_reference,
            ace2_layer23_v_input_scale_oracle,
            dict(os.environ),
        )
        if "--select-prompts" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--select-prompts", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.select_prompts()
        if "--check-record-lifecycle" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-record-lifecycle", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.record_lifecycle_regression()
        if "--check-gate-order" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-gate-order", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.gate_order_regression()
        if "--check-input-scale" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-input-scale", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.input_scale_regression()
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
