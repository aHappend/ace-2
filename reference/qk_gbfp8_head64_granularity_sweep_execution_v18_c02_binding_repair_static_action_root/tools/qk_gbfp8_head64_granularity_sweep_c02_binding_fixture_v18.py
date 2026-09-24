#!/usr/bin/env python3
"""Synthetic-only V18 fixture for complete C02 binding closure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import qk_gbfp8_head64_granularity_sweep_c02_binding_repair_v18 as repair


PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root"
LANE_METADATA = PROJECT_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
ACCEPTED_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
SENSITIVE_MARKER = "protected parser text must not escape"


class FixtureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"JSON object: {path}")
    return value


def load_parser() -> Any:
    spec = importlib.util.spec_from_file_location("v18_fixture_c02_parser", PARSER)
    require(spec is not None and spec.loader is not None, "parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_bundle(parser: Any, names: list[str]) -> tuple[bytes, dict[str, dict[str, Any]]]:
    dtypes = ("torch.bfloat16", "torch.int8", "torch.int16", "torch.int64")
    records: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        dtype = dtypes[index % len(dtypes)]
        shape = (1,)
        payload = b"\x00\x00" if dtype == "torch.bfloat16" else bytes(range(1, {"torch.int8": 2, "torch.int16": 3, "torch.int64": 9}[dtype]))
        digest = parser.record_sha256(dtype, shape, payload)
        records.append({"dtype": dtype, "name": name, "payload": payload, "shape": shape})
        bindings[name] = {"dtype": dtype, "sha256": digest, "shape": list(shape)}
    return parser.produce_c02_bundle(records), bindings


def expect_c02_failure(name: str, stage: str, operation: Callable[[], Any], parser: Any) -> dict[str, Any]:
    try:
        operation()
    except parser.C02Error as error:
        diagnostic = repair.sanitized_failure(error, stage)
        return {"diagnostic": diagnostic, "name": name, "passed": True}
    raise FixtureError(f"case did not fail closed: {name}")


def run_fixture() -> dict[str, Any]:
    parser = load_parser()
    lane_metadata = load_json(LANE_METADATA)
    accepted_package = load_json(ACCEPTED_PACKAGE)
    context = repair.prepare_static_context(accepted_package, lane_metadata)
    names = sorted(context["tensor_bindings"])
    require(len(names) == repair.EXPECTED_RECORD_COUNT, "fixture binding cardinality")
    bundle, bindings = synthetic_bundle(parser, names)

    producer = parser.parse_c02_producer_bytes(bundle, bindings)
    accepted = parser.parse_c02_accepted_reader(bundle, bindings)
    require(producer == accepted and len(producer) == 25, "full 25-record cross-parse")

    first_name = names[0]
    missing = deepcopy(bindings)
    missing.pop(first_name)
    extra = deepcopy(bindings)
    extra["synthetic.extra"] = {"dtype": "torch.int8", "sha256": "0" * 64, "shape": [1]}
    malformed = deepcopy(bindings)
    malformed[first_name]["unexpected"] = False
    cases = [
        expect_c02_failure("producer_missing_binding", "C02_BINDING_VALIDATION", lambda: parser.parse_c02_producer_bytes(bundle, missing), parser),
        expect_c02_failure("accepted_missing_binding", "C02_BINDING_VALIDATION", lambda: parser.parse_c02_accepted_reader(bundle, missing), parser),
        expect_c02_failure("producer_extra_binding", "C02_BINDING_VALIDATION", lambda: parser.parse_c02_producer_bytes(bundle, extra), parser),
        expect_c02_failure("accepted_extra_binding", "C02_BINDING_VALIDATION", lambda: parser.parse_c02_accepted_reader(bundle, extra), parser),
        expect_c02_failure("producer_malformed_binding", "C02_BINDING_VALIDATION", lambda: parser.parse_c02_producer_bytes(bundle, malformed), parser),
        expect_c02_failure("accepted_malformed_binding", "C02_BINDING_VALIDATION", lambda: parser.parse_c02_accepted_reader(bundle, malformed), parser),
    ]

    selected = repair.select_numerical_records(producer, context)
    require(tuple(selected) == repair.EXPECTED_SELECTED_NAMES and len(selected) == 3, "three-record numerical selection")
    diagnostic = repair.sanitized_failure(parser.C02Error(SENSITIVE_MARKER), "C02_BINDING_VALIDATION")
    require(set(diagnostic) == {"exception_type", "failure_stage"}, "diagnostic exact keys")
    require(SENSITIVE_MARKER not in compact_bytes(diagnostic).decode("ascii"), "diagnostic text leakage")

    input_bindings = accepted_package["official_benchmark"]["input_bindings"]
    identity = {
        "action_id": repair.ACTION_ID,
        "claim_boundary": repair.CLAIM_BOUNDARY,
        "lane_metadata_sha256": input_bindings["lane_metadata"]["sha256"],
        "model_identity_sha256": accepted_package["official_benchmark"]["model_identity_sha256"],
        "static_package_id": accepted_package["package_id"],
        "tensor_bundle_sha256": input_bindings["tensor_bundle"]["sha256"],
    }
    report = {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_synthetic_fixture",
        "cases": cases,
        "complete_binding_count": len(context["tensor_bindings"]),
        "cross_parser_record_count": len(producer),
        "diagnostic_contract": diagnostic,
        "identity": identity,
        "numerical_tensor_names": list(selected),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "production_mode_exercised": False,
        "status": "PASS_V18_C02_BINDING_REPAIR_SYNTHETIC_FIXTURE",
        "synthetic_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "synthetic_bytes_only": True,
        "v17_retried": False,
        "v18_executed": False,
    }
    report["report_sha256"] = hashlib.sha256(compact_bytes(report)).hexdigest()
    require(SENSITIVE_MARKER not in compact_bytes(report).decode("ascii"), "fixture report text leakage")
    return report


def main() -> int:
    print(compact_bytes(run_fixture()).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
