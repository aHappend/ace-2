#!/usr/bin/env python3
"""Non-consuming regression for the V21 production binding resolver."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import qk_gbfp8_head64_granularity_sweep_v21_binding_resolution as resolution


ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGE = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root/accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
V18_BINDING_TABLE = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root/bindings/C02_EXACT_BINDINGS_25.json"
OFFICIAL_PAYLOAD = ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
EXPECTED_CORE_PACKAGE_SHA256 = "3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832"
EXPECTED_V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"

AUDIT = {
    "official_payload_open_count": 0,
    "official_target_process_starts": 0,
}


class RegressionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        candidate = args[0]
        if isinstance(candidate, (str, bytes, os.PathLike)):
            try:
                if Path(candidate).resolve() == OFFICIAL_PAYLOAD.resolve():
                    AUDIT["official_payload_open_count"] += 1
            except (OSError, RuntimeError, ValueError):
                pass
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.posix_spawnp"}:
        AUDIT["official_target_process_starts"] += 1


def load_canonical(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"frozen hash: {path}")
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value, raw


def record_identities(records_by_name: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {name: sha256_bytes(compact_bytes(record)) for name, record in records_by_name.items()}


def expect_failure(
    name: str,
    code: str,
    operation: Callable[[], Any],
) -> dict[str, Any]:
    try:
        operation()
    except resolution.BindingResolutionError as error:
        require(error.failure_stage == "V18_BINDING", f"{name}: failure stage")
        require(error.code == code, f"{name}: failure code {error.code}")
        return {
            "failure_code": error.code,
            "failure_stage": error.failure_stage,
            "name": name,
            "passed": True,
        }
    raise RegressionError(f"{name}: not rejected")


def write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run() -> dict[str, Any]:
    core_package, _ = load_canonical(CORE_PACKAGE, EXPECTED_CORE_PACKAGE_SHA256)
    binding_table, _ = load_canonical(V18_BINDING_TABLE, EXPECTED_V18_BINDING_SHA256)
    tensor_records = core_package["official_benchmark"]["input_bindings"]["tensor_records"]
    require(type(tensor_records) is dict and len(tensor_records) == 3, "frozen tensor record cardinality")

    baseline = resolution.resolve_production_selection(tensor_records, binding_table)
    baseline_identities = record_identities(baseline.records_by_name)
    permutation_cases = []
    for permutation in itertools.permutations(tensor_records.items()):
        permuted = dict(permutation)
        observed = resolution.resolve_production_selection(permuted, binding_table)
        require(observed.canonical_names == resolution.CANONICAL_TENSOR_NAMES, "canonical output order")
        require(record_identities(observed.records_by_name) == baseline_identities, "record identity across permutation")
        for tensor_name in resolution.CANONICAL_TENSOR_NAMES:
            require(observed.records_by_name[tensor_name] is permuted[observed.source_keys_by_name[tensor_name]], "record object preservation")
        permutation_cases.append({
            "canonical_tensor_names": list(observed.canonical_names),
            "record_identities": record_identities(observed.records_by_name),
            "source_key_order": list(permuted),
        })

    duplicate = copy.deepcopy(tensor_records)
    duplicate["bf16_oracle_scores"]["tensor_name"] = "bf16.k_rope"
    missing = copy.deepcopy(tensor_records)
    del missing["realized_query_source"]
    unexpected = copy.deepcopy(tensor_records)
    unexpected["realized_query_source"]["tensor_name"] = "bf16.unexpected"
    malformed = copy.deepcopy(tensor_records)
    malformed["realized_query_source"] = {"dtype": "torch.bfloat16"}
    altered_table = copy.deepcopy(binding_table)
    altered_table["selected_tensor_names"] = list(reversed(altered_table["selected_tensor_names"]))

    negative_cases = [
        expect_failure("duplicate_tensor_name", "DUPLICATE_TENSOR_NAME", lambda: resolution.resolve_production_selection(duplicate, binding_table)),
        expect_failure("missing_tensor_name", "MISSING_TENSOR_NAME", lambda: resolution.resolve_production_selection(missing, binding_table)),
        expect_failure("unexpected_tensor_name", "UNEXPECTED_TENSOR_NAME", lambda: resolution.resolve_production_selection(unexpected, binding_table)),
        expect_failure("malformed_tensor_record", "TENSOR_RECORD_MALFORMED", lambda: resolution.resolve_production_selection(malformed, binding_table)),
        expect_failure("altered_v18_selected_names", "BINDING_TABLE_SELECTED_NAMES", lambda: resolution.resolve_production_selection(tensor_records, altered_table)),
    ]

    require(AUDIT == {"official_payload_open_count": 0, "official_target_process_starts": 0}, "non-consuming audit boundary")
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_v21_binding_resolution_regression",
        "canonical_tensor_names": list(resolution.CANONICAL_TENSOR_NAMES),
        "case_count": len(permutation_cases) + len(negative_cases),
        "frozen_core_package_sha256": EXPECTED_CORE_PACKAGE_SHA256,
        "frozen_v18_binding_table_sha256": EXPECTED_V18_BINDING_SHA256,
        "negative_cases": negative_cases,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": AUDIT["official_payload_open_count"],
        "official_target_process_starts": AUDIT["official_target_process_starts"],
        "permutation_case_count": len(permutation_cases),
        "permutation_cases": permutation_cases,
        "production_resolver_sha256": sha256_file(Path(resolution.__file__).resolve()),
        "regression_source_sha256": sha256_file(Path(__file__).resolve()),
        "schema_version": 1,
        "status": "PASS_V21_ORDER_INDEPENDENT_BINDING_RESOLUTION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    sys.addaudithook(audit_hook)
    try:
        report = run()
        raw = compact_bytes(report)
        if arguments.output is not None:
            write_once(arguments.output, raw)
        sys.stdout.buffer.write(raw)
        return 0
    except Exception as error:
        sys.stderr.write(f"V21_BINDING_REGRESSION_FAIL:{type(error).__name__}:{error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
