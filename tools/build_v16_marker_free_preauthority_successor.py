#!/usr/bin/env python3
"""Build the additive marker-free V16 static preauthority successor.

The builder reads only static predecessor/package material.  It never opens
the sealed tensor and never invokes the transport, launcher, controller, or
official evaluator.  The only executed fixture is synthetic and marker-free.
"""

from __future__ import annotations

import hashlib
import ast
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
V15_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v15_shellfree_action_root"
PRESERVED_V16_CANDIDATE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_action_root"
V16_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_shellfree_action_root"
RUNTIME_CONTAINER = PROJECT_ROOT / "runtime"
RUNTIME_ROOT = RUNTIME_CONTAINER / "qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_e4c05735"
PRIMARY_ROOT = RUNTIME_ROOT / "primary"
FALLBACK_ROOT = RUNTIME_ROOT / "fallback"
V8_STATIC_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root"
CONTRACT = PROJECT_ROOT / "design/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_V16_PREAUTHORITY_ENGINEER_TASK.json"
VERIFIER_SOURCE = PROJECT_ROOT / "tools/v16_marker_free_preauthority_verify_independent.py"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v16:execute-once:e4c05735:20260814T093500Z"
V15_ACTION_ID = "ace2:qk-gbfp8-base-v15:execute-once:c2dfe170:20260814T084500Z"
DISPOSITION = "V16_EVALUATOR_RETURN_PATH_SUCCESSOR_STATIC_PACKAGE_READY_NO_EXECUTION_AUTHORITY"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXPECTED_UID = os.getuid()
EXPECTED_GID = os.getgid()
FAILURE_STAGES = (
    "EVALUATOR_CALL",
    "CANONICAL_DECODE",
    "RESULT_SCHEMA",
    "RESULT_RECORD",
    "RESULT_PUBLICATION",
    "TERMINAL_BUILD",
    "TERMINAL_SCHEMA",
    "TERMINAL_PUBLICATION",
)


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_object(raw: bytes) -> dict[str, Any]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs)
    require(type(value) is dict and not duplicate, "JSON object shape")
    return value


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def transform_text(text: str) -> str:
    replacements = (
        (str(V15_ROOT), str(V16_ROOT)),
        ("qk_gbfp8_head64_granularity_sweep_execution_v15_c2dfe170", "qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_e4c05735"),
        (V15_ACTION_ID, ACTION_ID),
        ("V15_SHELLFREE", "V16_SHELLFREE"),
        ("v15_shellfree", "v16_shellfree"),
        ("_v15.py", "_v16.py"),
        ("Base V15", "Base V16"),
        ("V15 execute-once", "V16 execute-once"),
        (".v15-", ".v16-"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def transform_value(value: Any) -> Any:
    if isinstance(value, str):
        return transform_text(value)
    if isinstance(value, list):
        return [transform_value(item) for item in value]
    if isinstance(value, dict):
        return {key: transform_value(item) for key, item in value.items()}
    return value


def invocation(argv: list[str]) -> dict[str, Any]:
    payload = {
        "argv": argv,
        "command_representation": "ARGV_VECTOR_ONLY",
        "cwd": str(V16_ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": False,
    }
    return payload | {"invocation_sha256": sha256_bytes(compact_bytes(payload))}


def runtime_directories() -> list[Path]:
    return [
        RUNTIME_ROOT,
        PRIMARY_ROOT,
        PRIMARY_ROOT / "authority",
        PRIMARY_ROOT / "authority/base",
        PRIMARY_ROOT / "result",
        PRIMARY_ROOT / "result/base",
        FALLBACK_ROOT,
    ]


def runtime_files() -> dict[str, Path]:
    return {
        "authority": PRIMARY_ROOT / "authority/base/authority.json",
        "credential": PRIMARY_ROOT / "authority/base/credential.json",
        "ledger": PRIMARY_ROOT / "authority/base/authority-ledger.json",
        "result": PRIMARY_ROOT / "result/base/result.json",
        "first_terminal": PRIMARY_ROOT / "authority/base/first-terminal.json",
        "fallback_terminal": FALLBACK_ROOT / "first-terminal.json",
    }


def prepare_runtime_tree(staging: Path) -> None:
    require(RUNTIME_CONTAINER.is_dir() and not RUNTIME_CONTAINER.is_symlink(), "runtime container absent")
    info = os.lstat(RUNTIME_CONTAINER)
    require(stat.S_IMODE(info.st_mode) == 0o700, "runtime container mode")
    require(info.st_uid == EXPECTED_UID and info.st_gid == EXPECTED_GID, "runtime container owner")
    os.mkdir(staging, 0o700)
    for relative in ("primary", "primary/authority", "primary/authority/base", "primary/result", "primary/result/base", "fallback"):
        os.mkdir(staging / relative, 0o700)


def render_shared_module() -> bytes:
    source = '''#!/usr/bin/env python3
"""Pure evaluator-return processing shared by V16 production and fixture."""

from __future__ import annotations

import json
from typing import Any, Callable, NamedTuple


EVALUATOR_CALL = "EVALUATOR_CALL"
CANONICAL_DECODE = "CANONICAL_DECODE"
RESULT_SCHEMA = "RESULT_SCHEMA"
RESULT_RECORD = "RESULT_RECORD"
RESULT_PUBLICATION = "RESULT_PUBLICATION"
TERMINAL_BUILD = "TERMINAL_BUILD"
TERMINAL_SCHEMA = "TERMINAL_SCHEMA"
TERMINAL_PUBLICATION = "TERMINAL_PUBLICATION"


class EvaluatorReturnPathError(RuntimeError):
    def __init__(self, failure_stage: str, error: BaseException):
        self.failure_stage = failure_stage
        self.error_type = type(error).__name__
        super().__init__(f"{failure_stage}:{self.error_type}:{error}")


class ProcessedEvaluatorReturn(NamedTuple):
    record: dict[str, Any]
    durable_result_bytes: bytes


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\\n").encode("ascii")


def _decode_canonical_json(raw: bytes) -> dict[str, Any]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(
        raw.decode("ascii", "strict"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if duplicate or type(value) is not dict or compact_bytes(value) != raw:
        raise ValueError("noncanonical evaluator-return JSON object")
    return value


def call_evaluator_once(evaluate: Callable[[], bytes]) -> bytes:
    try:
        raw = evaluate()
        if type(raw) is not bytes:
            raise TypeError("evaluator return is not bytes")
        return raw
    except EvaluatorReturnPathError:
        raise
    except BaseException as error:
        raise EvaluatorReturnPathError(EVALUATOR_CALL, error) from error


def process_evaluator_return(
    raw: bytes,
    schema_validate: Callable[[dict[str, Any]], None],
    record_validate: Callable[[dict[str, Any], bytes], None],
) -> ProcessedEvaluatorReturn:
    try:
        record = _decode_canonical_json(raw)
    except BaseException as error:
        raise EvaluatorReturnPathError(CANONICAL_DECODE, error) from error
    try:
        schema_validate(record)
    except BaseException as error:
        raise EvaluatorReturnPathError(RESULT_SCHEMA, error) from error
    try:
        record_validate(record, raw)
        durable_result_bytes = compact_bytes(record)
        if durable_result_bytes != raw:
            raise ValueError("durable result bytes differ")
    except BaseException as error:
        raise EvaluatorReturnPathError(RESULT_RECORD, error) from error
    return ProcessedEvaluatorReturn(record=record, durable_result_bytes=durable_result_bytes)


def _stage_call(stage: str, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except EvaluatorReturnPathError:
        raise
    except BaseException as error:
        raise EvaluatorReturnPathError(stage, error) from error


def complete_result_path(
    processed: ProcessedEvaluatorReturn,
    publish_result: Callable[[bytes], None],
    build_terminal: Callable[[dict[str, Any], bytes], dict[str, Any]],
    validate_terminal: Callable[[dict[str, Any]], None],
    publish_terminal: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    _stage_call(RESULT_PUBLICATION, lambda: publish_result(processed.durable_result_bytes))
    terminal = _stage_call(TERMINAL_BUILD, lambda: build_terminal(processed.record, processed.durable_result_bytes))
    _stage_call(TERMINAL_SCHEMA, lambda: validate_terminal(terminal))
    _stage_call(TERMINAL_PUBLICATION, lambda: publish_terminal(terminal))
    return terminal
'''
    return source.encode("utf-8")


def render_fixture() -> bytes:
    source = '''#!/usr/bin/env python3
"""Marker-free synthetic fixture for the shared V16 evaluator-return path."""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

from qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16 import (
    EvaluatorReturnPathError,
    call_evaluator_once,
    compact_bytes,
    complete_result_path,
    process_evaluator_return,
)


LABELS = ("G8", "G4", "G2", "G1")
EXPECTED_KEYS = {"binding_sha256", "candidate_results", "selected_candidate", "selection", "terminal"}
SYNTHETIC_BINDING = "a" * 64


def make_result(first_passing: str | None) -> dict[str, Any]:
    candidates = [
        {"label": label, "all_hard_gates_pass": first_passing is not None and LABELS.index(label) >= LABELS.index(first_passing)}
        for label in LABELS
    ]
    passing = [item["label"] for item in candidates if item["all_hard_gates_pass"]]
    selected = passing[0] if passing else None
    return {
        "binding_sha256": SYNTHETIC_BINDING,
        "candidate_results": candidates,
        "selected_candidate": selected,
        "selection": {"passing_candidates": passing, "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selected_candidate": selected},
        "terminal": {
            "reason_code": "HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED",
            "status": "SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL",
        },
    }


def schema_validate(record: dict[str, Any]) -> None:
    if not isinstance(record.get("binding_sha256"), str):
        raise ValueError("binding type")
    if not isinstance(record.get("candidate_results"), list) or len(record["candidate_results"]) != 4:
        raise ValueError("candidate schema")
    if record.get("selected_candidate") is not None and record.get("selected_candidate") not in LABELS:
        raise ValueError("selected candidate schema")
    if not isinstance(record.get("selection"), dict) or not isinstance(record.get("terminal"), dict):
        raise ValueError("nested schema")


def record_validate(record: dict[str, Any], raw: bytes) -> None:
    if set(record) != EXPECTED_KEYS:
        raise ValueError("exact keys")
    if record["binding_sha256"] != SYNTHETIC_BINDING:
        raise ValueError("binding mismatch")
    if compact_bytes(record) != raw:
        raise ValueError("canonical record bytes")
    labels = [item.get("label") for item in record["candidate_results"]]
    if labels != list(LABELS):
        raise ValueError("candidate order")
    passing = [item["label"] for item in record["candidate_results"] if item.get("all_hard_gates_pass") is True]
    selected = passing[0] if passing else None
    if record["selected_candidate"] != selected:
        raise ValueError("selection mismatch")
    if record["selection"] != {"passing_candidates": passing, "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selected_candidate": selected}:
        raise ValueError("selection record mismatch")


def pipeline(raw: bytes, *, result_exists: bool = False, terminal_build_error: bool = False, terminal_schema_error: bool = False, terminal_publication_error: bool = False) -> dict[str, Any]:
    processed = process_evaluator_return(raw, schema_validate, record_validate)
    store: dict[str, Any] = {"result": b"occupied"} if result_exists else {}

    def publish_result(value: bytes) -> None:
        if "result" in store:
            raise FileExistsError("create-once result collision")
        store["result"] = value

    def build_terminal(record: dict[str, Any], durable: bytes) -> dict[str, Any]:
        if terminal_build_error:
            raise ValueError("synthetic terminal build failure")
        return {"failure_stage": None, "result_sha256": __import__("hashlib").sha256(durable).hexdigest(), **record["terminal"]}

    def validate_terminal(record: dict[str, Any]) -> None:
        if terminal_schema_error:
            raise ValueError("synthetic terminal schema failure")
        if set(record) != {"failure_stage", "reason_code", "result_sha256", "status"} or record["failure_stage"] is not None:
            raise ValueError("terminal record")

    def publish_terminal(record: dict[str, Any]) -> None:
        if terminal_publication_error:
            raise OSError("synthetic terminal publication failure")
        if "terminal" in store:
            raise FileExistsError("create-once terminal collision")
        store["terminal"] = copy.deepcopy(record)

    terminal = complete_result_path(processed, publish_result, build_terminal, validate_terminal, publish_terminal)
    return {"result": store["result"], "terminal": terminal}


def observe(name: str, expected_stage: str | None, operation: Callable[[], Any]) -> dict[str, Any]:
    observed_stage: str | None = None
    detail = ""
    try:
        operation()
    except EvaluatorReturnPathError as error:
        observed_stage = error.failure_stage
        detail = error.error_type
    except BaseException as error:
        detail = f"UNSTRUCTURED:{type(error).__name__}"
    return {
        "detail_class": detail,
        "expected_stage": expected_stage,
        "name": name,
        "observed_stage": observed_stage,
        "passed": observed_stage == expected_stage,
    }


def run_fixture() -> dict[str, Any]:
    all_fail = make_result(None)
    first_pass = make_result("G4")
    schema_bad = copy.deepcopy(all_fail)
    schema_bad["selected_candidate"] = 7
    exact_key_bad = copy.deepcopy(all_fail)
    exact_key_bad["unexpected"] = True
    binding_bad = copy.deepcopy(all_fail)
    binding_bad["binding_sha256"] = "b" * 64
    cases = [
        observe("all_hard_gates_fail", None, lambda: pipeline(compact_bytes(all_fail))),
        observe("first_passing_candidate", None, lambda: pipeline(compact_bytes(first_pass))),
        observe("evaluator_call_failure", "EVALUATOR_CALL", lambda: call_evaluator_once(lambda: (_ for _ in ()).throw(RuntimeError("synthetic evaluator call failure")))),
        observe("noncanonical_json", "CANONICAL_DECODE", lambda: process_evaluator_return((json.dumps(all_fail, indent=2, sort_keys=True) + "\\n").encode("ascii"), schema_validate, record_validate)),
        observe("schema_mismatch", "RESULT_SCHEMA", lambda: process_evaluator_return(compact_bytes(schema_bad), schema_validate, record_validate)),
        observe("exact_key_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(exact_key_bad), schema_validate, record_validate)),
        observe("binding_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(binding_bad), schema_validate, record_validate)),
        observe("create_once_result_collision", "RESULT_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), result_exists=True)),
        observe("terminal_build_failure", "TERMINAL_BUILD", lambda: pipeline(compact_bytes(all_fail), terminal_build_error=True)),
        observe("terminal_validation_failure", "TERMINAL_SCHEMA", lambda: pipeline(compact_bytes(all_fail), terminal_schema_error=True)),
        observe("terminal_publication_failure", "TERMINAL_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), terminal_publication_error=True)),
    ]
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_marker_free_fixture_report",
        "case_count": len(cases),
        "cases": cases,
        "fixture_uses_official_evaluator": False,
        "fixture_uses_official_payload": False,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE" if all(case["passed"] for case in cases) else "FAIL_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE",
    }


def main() -> int:
    report = run_fixture()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return source.encode("utf-8")


def render_result_validation_module(result_schema_bytes: bytes) -> bytes:
    source = '''#!/usr/bin/env python3
"""Exact frozen result schema and record validators shared by production and fixture."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from typing import Any, Callable, NamedTuple

import jsonschema


FROZEN_RESULT_SCHEMA_BYTES = ''' + repr(result_schema_bytes) + '''
RESULT_SCHEMA_SHA256 = "''' + sha256_bytes(result_schema_bytes) + '''"
RESULT_KEYS = frozenset({
    "authority_sha256", "candidate_results", "consumed_ledger_sha256", "evaluator_sha256",
    "fresh_l2_acceptance_sha256", "generation_id", "input_bindings", "invocation_sha256",
    "irreversible_action_id", "lane_label", "model_identity_sha256", "namespace_label", "package_id",
    "package_sha256", "result_sha256", "schema_id", "selected_candidate", "selection", "terminal",
})
CANDIDATE_RESULT_KEYS = frozenset({
    "bytes_per_head", "exponent_bytes_per_head", "group_count", "group_size", "label",
    "mantissa_bytes_per_head", "metrics", "threshold_evaluation",
})
SELECTION_KEYS = frozenset({"passing_candidates", "policy", "selected_candidate"})
TERMINAL_KEYS = frozenset({
    "first_record_immutable", "invocation_count_performed", "metrics_published", "reason_code",
    "retry_replay_resume_repair_permitted", "status", "tensor_open_count", "thresholds_evaluated",
})
CANDIDATE_SPECS = (
    ("G8", 8, 8, 80, 16),
    ("G4", 4, 16, 96, 32),
    ("G2", 2, 32, 128, 64),
    ("G1", 1, 64, 192, 128),
)


class ResultValidationError(RuntimeError):
    pass


class ResultBindings(NamedTuple):
    acceptance_sha256: str
    authority_sha256: str
    evaluator_sha256: str
    invocation_sha256: str
    ledger_sha256: str
    model_identity_sha256: str
    package_sha256: str
    tensor_bundle_sha256: str


class ResultValidationBindings(NamedTuple):
    validate_result_record: Callable[[dict[str, Any], bytes], None]
    validate_result_schema: Callable[[dict[str, Any]], None]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultValidationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_result_schema() -> dict[str, Any]:
    require(sha256_bytes(FROZEN_RESULT_SCHEMA_BYTES) == RESULT_SCHEMA_SHA256, "embedded result schema checksum")
    value = json.loads(FROZEN_RESULT_SCHEMA_BYTES.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == FROZEN_RESULT_SCHEMA_BYTES, "embedded canonical result schema")
    return value


def construct_result_schema_validator(result_schema: dict[str, Any]) -> Any:
    require(compact_bytes(result_schema) == FROZEN_RESULT_SCHEMA_BYTES, "production result schema bytes")
    validator_class = jsonschema.Draft202012Validator
    validator_class.check_schema(result_schema)
    return validator_class(result_schema)


def validate_result_schema_exact(result_validator: Any, result: dict[str, Any]) -> None:
    result_validator.validate(result)


def _valid_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_result_record_exact(
    result: dict[str, Any],
    result_bytes: bytes,
    *,
    bindings: ResultBindings,
    action_id: str,
) -> None:
    require(compact_bytes(result) == result_bytes, "result canonical bytes")
    require(set(result) == RESULT_KEYS, "result exact keys")
    require(result["package_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE", "result package id")
    require(result["schema_id"] == "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1", "result schema id")
    require(result["generation_id"] == "qk-gbfp8-head64-granularity-sweep-v8-base", "result generation")
    require(result["lane_label"] == "Base" and result["namespace_label"] == "base", "result Base lane")
    require(result["package_sha256"] == bindings.package_sha256, "result package checksum")
    require(result["evaluator_sha256"] == bindings.evaluator_sha256, "result evaluator checksum")
    require(result["authority_sha256"] == bindings.authority_sha256, "result authority checksum")
    require(result["consumed_ledger_sha256"] == bindings.ledger_sha256, "result ledger checksum")
    require(result["fresh_l2_acceptance_sha256"] == bindings.acceptance_sha256, "result acceptance checksum")
    require(result["invocation_sha256"] == bindings.invocation_sha256, "result invocation checksum")
    require(result["irreversible_action_id"] == action_id, "result action identity")
    require(result["model_identity_sha256"] == bindings.model_identity_sha256, "result model checksum")
    for key in (
        "authority_sha256", "consumed_ledger_sha256", "evaluator_sha256", "fresh_l2_acceptance_sha256",
        "invocation_sha256", "model_identity_sha256", "package_sha256", "result_sha256",
    ):
        require(_valid_sha256(result[key]), f"result checksum syntax: {key}")
    require(result["input_bindings"] == {
        "sealed_set_id": "w4a8-c02-attention-substage-trace-v2",
        "tensor_bundle_sha256": bindings.tensor_bundle_sha256,
        "tensor_record_count": 25,
    }, "result input bindings")
    candidates = result["candidate_results"]
    require(type(candidates) is list and len(candidates) == len(CANDIDATE_SPECS), "result candidate cardinality")
    passing: list[str] = []
    for candidate, (label, group_size, group_count, bytes_per_head, exponent_bytes_per_head) in zip(candidates, CANDIDATE_SPECS):
        require(type(candidate) is dict and set(candidate) == CANDIDATE_RESULT_KEYS, f"result candidate keys: {label}")
        require(candidate["label"] == label, f"result candidate label: {label}")
        require(candidate["group_size"] == group_size and candidate["group_count"] == group_count, f"result candidate groups: {label}")
        require(candidate["bytes_per_head"] == bytes_per_head, f"result candidate bytes: {label}")
        require(candidate["exponent_bytes_per_head"] == exponent_bytes_per_head, f"result candidate exponent bytes: {label}")
        require(candidate["mantissa_bytes_per_head"] == 64, f"result candidate mantissa bytes: {label}")
        if candidate["threshold_evaluation"]["all_hard_gates_pass"]:
            passing.append(label)
    selected = passing[0] if passing else None
    require(result["selected_candidate"] == selected, "result selected candidate")
    selection = result["selection"]
    require(type(selection) is dict and set(selection) == SELECTION_KEYS, "result selection keys")
    require(selection == {
        "passing_candidates": passing,
        "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER",
        "selected_candidate": selected,
    }, "result selection")
    terminal = result["terminal"]
    require(type(terminal) is dict and set(terminal) == TERMINAL_KEYS, "result terminal keys")
    require(terminal["first_record_immutable"] is True, "result terminal immutability")
    require(terminal["invocation_count_performed"] == 1 and terminal["tensor_open_count"] == 1, "result terminal counters")
    require(terminal["metrics_published"] is True and terminal["thresholds_evaluated"] is True, "result terminal publication")
    require(terminal["retry_replay_resume_repair_permitted"] is False, "result terminal replay")
    require(terminal["status"] == ("SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL"), "result terminal status")
    require(terminal["reason_code"] == ("HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED"), "result terminal reason")
    payload = dict(result)
    observed = payload.pop("result_sha256")
    require(sha256_bytes(compact_bytes(payload)) == observed, "result self-checksum")


def bind_result_validators(result_validator: Any, bindings: ResultBindings, action_id: str) -> ResultValidationBindings:
    return ResultValidationBindings(
        validate_result_record=partial(validate_result_record_exact, bindings=bindings, action_id=action_id),
        validate_result_schema=partial(validate_result_schema_exact, result_validator),
    )
'''
    return source.encode("utf-8")


def render_repaired_fixture() -> bytes:
    source = '''#!/usr/bin/env python3
"""Marker-free synthetic fixture using the exact V16 production result validators."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable

from qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16 import (
    EvaluatorReturnPathError,
    call_evaluator_once,
    compact_bytes,
    complete_result_path,
    process_evaluator_return,
)
from qk_gbfp8_head64_granularity_sweep_result_validation_v16 import (
    RESULT_SCHEMA_SHA256,
    ResultBindings,
    bind_result_validators,
    construct_result_schema_validator,
    frozen_result_schema,
)


ACTION_ID = "ace2:qk-gbfp8-base-v16:execute-once:e4c05735:20260814T093500Z"
LABELS = ("G8", "G4", "G2", "G1")
CANDIDATE_SPECS = {
    "G8": (8, 8, 80, 16),
    "G4": (4, 16, 96, 32),
    "G2": (2, 32, 128, 64),
    "G1": (1, 64, 192, 128),
}
SYNTHETIC_BINDINGS = ResultBindings(
    acceptance_sha256="a" * 64,
    authority_sha256="b" * 64,
    evaluator_sha256="c" * 64,
    invocation_sha256="d" * 64,
    ledger_sha256="e" * 64,
    model_identity_sha256="f" * 64,
    package_sha256="0" * 64,
    tensor_bundle_sha256="1" * 64,
)
RESULT_VALIDATOR = construct_result_schema_validator(frozen_result_schema())
RESULT_VALIDATION = bind_result_validators(RESULT_VALIDATOR, SYNTHETIC_BINDINGS, ACTION_ID)


def count_gate(actual: int) -> dict[str, Any]:
    return {"actual": actual, "limit": 0, "pass": actual == 0}


def make_metrics(passing: bool) -> dict[str, Any]:
    saturation_events = 0 if passing else 1
    return {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": saturation_events,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": 1,
            "preserved_positive_margin_fraction": {"denominator": 346, "numerator": 346},
            "preserved_positive_margin_row_count": 346,
            "unique_oracle_top_row_count": 346,
            "violation_count": 0,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": 0,
            "sum_absolute_error_q12_20_lsb": 0,
            "sum_signed_error_q12_20_lsb": 0,
            "sum_squared_error_q40_40_lsb2": 0,
            "valid_value_count": 12054,
        },
        "top_key": {
            "matching_fraction": {"denominator": 574, "numerator": 574},
            "matching_row_count": 574,
            "mismatch_count": 0,
            "row_count": 574,
        },
    }


def make_thresholds(passing: bool) -> dict[str, Any]:
    saturation_events = 0 if passing else 1
    return {
        "all_hard_gates_pass": passing,
        "cross_lane_record_count_maximum": count_gate(0),
        "invalid_or_non_finite_value_count_maximum": count_gate(0),
        "normalization_rejection_count_maximum": count_gate(0),
        "positive_centered_realized_score_count_maximum": count_gate(0),
        "rank_margin_violation_count_maximum": count_gate(0),
        "saturation_event_count_maximum": count_gate(saturation_events),
        "top_key_matching_fraction_minimum": {"actual": {"denominator": 574, "numerator": 574}, "limit": {"denominator": 1, "numerator": 1}, "pass": True},
        "top_key_mismatch_count_maximum": count_gate(0),
        "unique_oracle_positive_margin_preserved_fraction_minimum": {"actual": {"denominator": 346, "numerator": 346}, "limit": {"denominator": 1, "numerator": 1}, "pass": True},
    }


def make_candidate(label: str, passing: bool) -> dict[str, Any]:
    group_size, group_count, bytes_per_head, exponent_bytes_per_head = CANDIDATE_SPECS[label]
    return {
        "bytes_per_head": bytes_per_head,
        "exponent_bytes_per_head": exponent_bytes_per_head,
        "group_count": group_count,
        "group_size": group_size,
        "label": label,
        "mantissa_bytes_per_head": 64,
        "metrics": make_metrics(passing),
        "threshold_evaluation": make_thresholds(passing),
    }


def make_result(first_passing: str | None) -> dict[str, Any]:
    passing_index = LABELS.index(first_passing) if first_passing is not None else len(LABELS)
    candidates = [make_candidate(label, index >= passing_index) for index, label in enumerate(LABELS)]
    passing = [item["label"] for item in candidates if item["threshold_evaluation"]["all_hard_gates_pass"]]
    selected = passing[0] if passing else None
    result = {
        "authority_sha256": SYNTHETIC_BINDINGS.authority_sha256,
        "candidate_results": candidates,
        "consumed_ledger_sha256": SYNTHETIC_BINDINGS.ledger_sha256,
        "evaluator_sha256": SYNTHETIC_BINDINGS.evaluator_sha256,
        "fresh_l2_acceptance_sha256": SYNTHETIC_BINDINGS.acceptance_sha256,
        "generation_id": "qk-gbfp8-head64-granularity-sweep-v8-base",
        "input_bindings": {"sealed_set_id": "w4a8-c02-attention-substage-trace-v2", "tensor_bundle_sha256": SYNTHETIC_BINDINGS.tensor_bundle_sha256, "tensor_record_count": 25},
        "invocation_sha256": SYNTHETIC_BINDINGS.invocation_sha256,
        "irreversible_action_id": ACTION_ID,
        "lane_label": "Base",
        "model_identity_sha256": SYNTHETIC_BINDINGS.model_identity_sha256,
        "namespace_label": "base",
        "package_id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE",
        "package_sha256": SYNTHETIC_BINDINGS.package_sha256,
        "schema_id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_RESULT_V1",
        "selected_candidate": selected,
        "selection": {"passing_candidates": passing, "policy": "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER", "selected_candidate": selected},
        "terminal": {
            "first_record_immutable": True,
            "invocation_count_performed": 1,
            "metrics_published": True,
            "reason_code": "HARD_THRESHOLDS_PASSED" if selected is not None else "HARD_THRESHOLD_FAILED",
            "retry_replay_resume_repair_permitted": False,
            "status": "SUCCEEDED_TERMINAL" if selected is not None else "FAILED_TERMINAL",
            "tensor_open_count": 1,
            "thresholds_evaluated": True,
        },
    }
    result["result_sha256"] = hashlib.sha256(compact_bytes(result)).hexdigest()
    return result


def pipeline(raw: bytes, *, result_exists: bool = False, terminal_build_error: bool = False, terminal_schema_error: bool = False, terminal_publication_error: bool = False) -> dict[str, Any]:
    processed = process_evaluator_return(raw, RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)
    store: dict[str, Any] = {"result": b"occupied"} if result_exists else {}

    def publish_result(value: bytes) -> None:
        if "result" in store:
            raise FileExistsError("create-once result collision")
        store["result"] = value

    def build_terminal(record: dict[str, Any], durable: bytes) -> dict[str, Any]:
        if terminal_build_error:
            raise ValueError("synthetic terminal build failure")
        return {"failure_stage": None, "result_sha256": hashlib.sha256(durable).hexdigest(), "reason_code": record["terminal"]["reason_code"], "status": record["terminal"]["status"]}

    def validate_terminal(record: dict[str, Any]) -> None:
        if terminal_schema_error:
            raise ValueError("synthetic terminal schema failure")
        if set(record) != {"failure_stage", "reason_code", "result_sha256", "status"} or record["failure_stage"] is not None:
            raise ValueError("terminal record")

    def publish_terminal(record: dict[str, Any]) -> None:
        if terminal_publication_error:
            raise OSError("synthetic terminal publication failure")
        if "terminal" in store:
            raise FileExistsError("create-once terminal collision")
        store["terminal"] = copy.deepcopy(record)

    terminal = complete_result_path(processed, publish_result, build_terminal, validate_terminal, publish_terminal)
    return {"result": store["result"], "terminal": terminal}


def observe(name: str, expected_stage: str | None, operation: Callable[[], Any]) -> dict[str, Any]:
    observed_stage: str | None = None
    detail = ""
    try:
        operation()
    except EvaluatorReturnPathError as error:
        observed_stage = error.failure_stage
        detail = error.error_type
    except BaseException as error:
        detail = f"UNSTRUCTURED:{type(error).__name__}"
    return {"detail_class": detail, "expected_stage": expected_stage, "name": name, "observed_stage": observed_stage, "passed": observed_stage == expected_stage}


def callable_identity(value: Any) -> str:
    target = getattr(value, "func", value)
    return f"{target.__module__}.{target.__qualname__}"


def run_fixture() -> dict[str, Any]:
    all_fail = make_result(None)
    first_pass = make_result("G4")
    for nominal in (all_fail, first_pass):
        raw = compact_bytes(nominal)
        RESULT_VALIDATION.validate_result_schema(nominal)
        RESULT_VALIDATION.validate_result_record(nominal, raw)
    schema_bad = copy.deepcopy(all_fail)
    schema_bad["candidate_results"][0]["metrics"]["top_key"]["row_count"] = 573
    exact_key_bad = copy.deepcopy(all_fail)
    exact_key_bad["unexpected"] = True
    binding_bad = copy.deepcopy(all_fail)
    binding_bad["package_sha256"] = "2" * 64
    checksum_bad = copy.deepcopy(all_fail)
    checksum_bad["result_sha256"] = "3" * 64
    cases = [
        observe("all_hard_gates_fail", None, lambda: pipeline(compact_bytes(all_fail))),
        observe("first_passing_candidate", None, lambda: pipeline(compact_bytes(first_pass))),
        observe("evaluator_call_failure", "EVALUATOR_CALL", lambda: call_evaluator_once(lambda: (_ for _ in ()).throw(RuntimeError("synthetic evaluator call failure")))),
        observe("noncanonical_json", "CANONICAL_DECODE", lambda: process_evaluator_return((json.dumps(all_fail, indent=2, sort_keys=True) + "\\n").encode("ascii"), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("schema_mismatch", "RESULT_SCHEMA", lambda: process_evaluator_return(compact_bytes(schema_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("exact_key_mismatch", "RESULT_SCHEMA", lambda: process_evaluator_return(compact_bytes(exact_key_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("binding_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(binding_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("self_checksum_mismatch", "RESULT_RECORD", lambda: process_evaluator_return(compact_bytes(checksum_bad), RESULT_VALIDATION.validate_result_schema, RESULT_VALIDATION.validate_result_record)),
        observe("create_once_result_collision", "RESULT_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), result_exists=True)),
        observe("terminal_build_failure", "TERMINAL_BUILD", lambda: pipeline(compact_bytes(all_fail), terminal_build_error=True)),
        observe("terminal_validation_failure", "TERMINAL_SCHEMA", lambda: pipeline(compact_bytes(all_fail), terminal_schema_error=True)),
        observe("terminal_publication_failure", "TERMINAL_PUBLICATION", lambda: pipeline(compact_bytes(all_fail), terminal_publication_error=True)),
    ]
    return {
        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_marker_free_fixture_report",
        "case_count": len(cases),
        "cases": cases,
        "fixture_reads_result_schema_file": False,
        "fixture_uses_official_evaluator": False,
        "fixture_uses_official_payload": False,
        "nominal_record_valid_case_count": 2,
        "nominal_schema_conforming_case_count": 2,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "result_schema_sha256": RESULT_SCHEMA_SHA256,
        "schema_validator_class": f"{type(RESULT_VALIDATOR).__module__}.{type(RESULT_VALIDATOR).__qualname__}",
        "schema_validator_callable": callable_identity(RESULT_VALIDATION.validate_result_schema),
        "record_validator_callable": callable_identity(RESULT_VALIDATION.validate_result_record),
        "validator_factory": "qk_gbfp8_head64_granularity_sweep_result_validation_v16.bind_result_validators",
        "status": "PASS_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE" if all(case["passed"] for case in cases) else "FAIL_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE",
    }


def main() -> int:
    report = run_fixture()
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return source.encode("utf-8")


def patch_terminal_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = transform_value(schema)
    schema["properties"]["failure_stage"] = {
        "oneOf": [
            {"enum": [*FAILURE_STAGES, "TRANSPORT", "PREFLIGHT", "LIVE_MATERIALIZATION"], "type": "string"},
            {"type": "null"},
        ]
    }
    schema["required"] = sorted(set(schema["required"]) | {"failure_stage"})
    for conditional in schema.get("allOf", []):
        status = conditional.get("if", {}).get("properties", {}).get("status", {}).get("const")
        then = conditional.get("then", {})
        if status == "SUCCEEDED_TERMINAL":
            then.setdefault("properties", {})["failure_stage"] = {"type": "null"}
        elif status == "PREFLIGHT_FAILED_TERMINAL":
            then.setdefault("properties", {})["failure_stage"] = {"enum": ["TRANSPORT", "PREFLIGHT"]}
        elif status == "CONSUMED_ORPHAN":
            then.setdefault("properties", {})["failure_stage"] = {"enum": list(FAILURE_STAGES)}
        elif status == "FAILED_TERMINAL":
            for branch in then.get("oneOf", []):
                reason = branch.get("properties", {}).get("reason_code", {}).get("const")
                if reason == "HARD_THRESHOLD_FAILED":
                    branch.setdefault("properties", {})["failure_stage"] = {"type": "null"}
                elif reason == "LIVE_MATERIALIZATION_FAILED":
                    branch.setdefault("properties", {})["failure_stage"] = {"const": "LIVE_MATERIALIZATION"}
    return schema


def patch_acceptance_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = transform_value(schema)
    additions = {
        "fixture_report_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        "independent_verifier_report_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        "official_payload_open_count": {"const": 0},
        "official_target_process_starts": {"const": 0},
        "required_disposition": {"const": DISPOSITION},
        "runtime_namespace_file_count": {"const": 0},
    }
    schema["properties"].update(additions)
    schema["required"] = sorted(set(schema["required"]) | set(additions))
    return schema


def render_launcher(accepted_paths: dict[str, Path], accepted_hash_sources: dict[str, Path], local_paths: dict[str, Path]) -> bytes:
    source_path = V15_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v15.py"
    launcher = transform_text(source_path.read_text(encoding="utf-8"))
    launcher = launcher.replace(
        "from typing import Any, Callable, NamedTuple\n",
        "from typing import Any, Callable, NamedTuple\n\nfrom qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16 import (\n    EvaluatorReturnPathError,\n    call_evaluator_once,\n    complete_result_path,\n    process_evaluator_return,\n)\n",
        1,
    )
    launcher = launcher.replace(
        ")\n\n\nROOT = Path(",
        ")\nfrom qk_gbfp8_head64_granularity_sweep_result_validation_v16 import (\n    ResultBindings,\n    bind_result_validators,\n    construct_result_schema_validator,\n)\n\n\nROOT = Path(",
        1,
    )
    launcher = replace_block(launcher, "RESULT_KEYS = frozenset({", "class ExecutionError", "")
    launcher = replace_block(launcher, "class ResultBindings(NamedTuple):", "class PreflightContext", "")
    launcher = launcher.replace("    RUNTIME_CONTAINER,\n    RUNTIME_ROOT,", "    RUNTIME_ROOT,", 1)
    accepted_block = "ACCEPTED_PATHS = {\n" + "".join(f'    "{key}": Path("{path}"),\n' for key, path in accepted_paths.items()) + "}\n"
    launcher = replace_block(launcher, "ACCEPTED_PATHS = {", "EXPECTED_BINDING_HASHES = {", accepted_block)
    hashes_block = "EXPECTED_BINDING_HASHES = {\n" + "".join(f'    "{key}": "{sha256_file(accepted_hash_sources[key])}",\n' for key in accepted_paths) + "}\n"
    launcher = replace_block(launcher, "EXPECTED_BINDING_HASHES = {", "EXPECTED_BINDING_ORDER =", hashes_block)
    local_block = "LOCAL_PATHS = {\n" + "".join(f'    "{key}": ROOT / "{path.relative_to(V16_ROOT).as_posix()}",\n' for key, path in local_paths.items()) + "}\n"
    launcher = replace_block(launcher, "LOCAL_PATHS = {", "LOCAL_BINDING_ORDER =", local_block)
    validate_block = '''def _validate_package(package: dict[str, Any]) -> None:
    require(package["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_package", "package kind")
    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_action_id": V15_ACTION_ID,
        "replacement_is_distinct": True,
        "retry_or_resume": False,
        "reuse_permitted": False,
    }, "action identity")
    transport = invocation_record(EXACT_TRANSPORT_ARGV)
    launcher = invocation_record(EXACT_LAUNCHER_ARGV)
    require(package["future_invocation"] == transport | {"invocation_sha256": sha256_bytes(compact_bytes(transport))}, "future transport invocation")
    require(package["launcher_invocation"] == launcher | {"invocation_sha256": sha256_bytes(compact_bytes(launcher))}, "launcher invocation")
    require([item["id"] for item in package["canonical_bindings"]] == EXPECTED_BINDING_ORDER, "canonical binding order")
    require([item["id"] for item in package["local_artifact_bindings"]] == LOCAL_BINDING_ORDER, "local binding order")
    require(package["official_identities"] == OFFICIAL_IDENTITIES, "official identities")
    require(package["fixed_populations"] == FIXED_POPULATIONS, "fixed populations")
    require(package["selection"]["candidate_order"] == list(CANDIDATE_ORDER), "candidate order")
    require(package["selection"]["policy"] == "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER_WITH_EVERY_HARD_GATE_REQUIRED", "selection policy")
    require(package["required_disposition"] == "V16_EVALUATOR_RETURN_PATH_SUCCESSOR_STATIC_PACKAGE_READY_NO_EXECUTION_AUTHORITY", "disposition")
    require(package["failure_stage_contract"]["stages"] == ["EVALUATOR_CALL", "CANONICAL_DECODE", "RESULT_SCHEMA", "RESULT_RECORD", "RESULT_PUBLICATION", "TERMINAL_BUILD", "TERMINAL_SCHEMA", "TERMINAL_PUBLICATION"], "failure stages")
    require(package["claim_boundary"]["execution_authorized"] is False, "execution authority")
    require(package["claim_boundary"]["official_payload_open_count"] == 0, "payload opens")
    require(package["claim_boundary"]["official_target_process_starts"] == 0, "target starts")
    runtime = package["runtime_namespace"]
    require(runtime["runtime_root"] == str(RUNTIME_ROOT), "runtime root")
    require(runtime["precreated_directories"] == [str(path) for path in RUNTIME_DIRECTORIES], "runtime directories")
    require(runtime["paths"] == {key: str(path) for key, path in LIVE_PATHS.items()}, "runtime paths")
    require(runtime["fallback_terminal"] == str(FALLBACK_TERMINAL), "fallback terminal")
    require(runtime["file_count"] == 0, "runtime file count")


'''
    launcher = replace_block(launcher, "def _validate_package(package: dict[str, Any]) -> None:", "def _verify_retained_terminals", validate_block)
    retirement_block = '''def _verify_v15_retirement() -> None:
    prior_package, _ = read_canonical_json(ACCEPTED_PATHS["v15_package"], "retired V15 package")
    require(prior_package["action_identity"]["future_action_id"] == V15_ACTION_ID, "retired V15 action")
    require(prior_package["claim_boundary"]["execution_authorized"] is False, "retired V15 package authority declaration")
    prior_acceptance, _ = read_canonical_json(ACCEPTED_PATHS["v15_static_acceptance"], "retired V15 static acceptance")
    require(prior_acceptance["static_acceptance_grants_execution_authority"] is False, "retired V15 acceptance authority")


'''
    launcher = replace_block(launcher, "def _verify_retained_terminals() -> None:", "def _load_frozen_module", retirement_block)
    launcher = launcher.replace("    _verify_retained_terminals()\n", "    _verify_v15_retirement()\n", 1)
    launcher = launcher.replace('require(acceptance["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_fresh_l2_static_acceptance", "acceptance kind")', 'require(acceptance["artifact_kind"] == "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_fresh_l2_static_acceptance", "acceptance kind")\n    require(acceptance["official_payload_open_count"] == 0 and acceptance["official_target_process_starts"] == 0, "acceptance no-execution counters")\n    require(acceptance["runtime_namespace_file_count"] == 0, "acceptance runtime file count")\n    require(acceptance["required_disposition"] == "V16_EVALUATOR_RETURN_PATH_SUCCESSOR_STATIC_PACKAGE_READY_NO_EXECUTION_AUTHORITY", "acceptance disposition")')
    launcher = launcher.replace('controller = _load_frozen_module("v15_frozen_accepted_v8_controller"', 'controller = _load_frozen_module("v16_frozen_accepted_v8_controller"')
    launcher = launcher.replace('evaluator = _load_frozen_module("v15_frozen_accepted_v8_evaluator"', 'evaluator = _load_frozen_module("v16_frozen_accepted_v8_evaluator"')
    launcher = launcher.replace('parser_module = _load_frozen_module("v15_frozen_accepted_v8_c02_parser"', 'parser_module = _load_frozen_module("v16_frozen_accepted_v8_c02_parser"')
    validator_constructor = '''def _construct_result_validator(result_schema: dict[str, Any]) -> Any:
    return construct_result_schema_validator(result_schema)


'''
    launcher = replace_block(launcher, "def _construct_result_validator(result_schema: dict[str, Any]) -> Any:", "def _construct_tensor_bindings", validator_constructor)
    launcher = replace_block(launcher, "def _valid_sha256(value: Any) -> bool:", "def _freeze_execution_plan", "")
    old_runtime_validators = '''    def validate_result_once(result: dict[str, Any], result_bytes: bytes) -> None:
        _validate_result_record(result, result_bytes, result_bindings)

    def validate_result_schema_once(result: dict[str, Any]) -> None:
        result_validator.validate(result)

    def validate_terminal_once(record: dict[str, Any]) -> None:
        terminal_validator.validate(record)

    runtime = RuntimeBindings(
        evaluate_bundle_once=evaluate_once,
        validate_result_record=validate_result_once,
        validate_result_schema=validate_result_schema_once,
        validate_terminal_record=validate_terminal_once,
    )
'''
    new_runtime_validators = '''    result_validation = bind_result_validators(result_validator, result_bindings, ACTION_ID)

    def validate_terminal_once(record: dict[str, Any]) -> None:
        terminal_validator.validate(record)

    runtime = RuntimeBindings(
        evaluate_bundle_once=evaluate_once,
        validate_result_record=result_validation.validate_result_record,
        validate_result_schema=result_validation.validate_result_schema,
        validate_terminal_record=validate_terminal_once,
    )
'''
    require(old_runtime_validators in launcher, "production result-validator closure marker")
    launcher = launcher.replace(old_runtime_validators, new_runtime_validators, 1)
    terminal_block = '''def _terminal_record(
    *,
    status: str,
    reason_code: str,
    authority_sha256: str | None,
    consumed_ledger_sha256: str | None,
    result_file_sha256: str | None,
    credential_consumed: bool,
    invocation_count: int,
    payload_open_count: int,
    orphaned: bool,
    failure_detail: str,
    failure_stage: str | None,
) -> dict[str, Any]:
    return sealed(
        {
            "action_id": ACTION_ID,
            "action_retired": True,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_first_terminal",
            "authority_sha256": authority_sha256,
            "consumed_ledger_sha256": consumed_ledger_sha256,
            "credential_consumed": credential_consumed,
            "failure_detail_sha256": sha256_bytes(failure_detail.encode("utf-8")),
            "failure_stage": failure_stage,
            "first_record_immutable": True,
            "invocation_count_performed": invocation_count,
            "orphaned_after_consumption": orphaned,
            "payload_open_count": payload_open_count,
            "reason_code": reason_code,
            "result_file_sha256": result_file_sha256,
            "retry_replay_resume_repair_replacement_permitted": False,
            "status": status,
        },
        "first_terminal_sha256",
    )


'''
    launcher = replace_block(launcher, "def _terminal_record(", "def _publish_terminal", terminal_block)
    retire_block = '''def _retire_preflight_failure(error: BaseException, reason_code: str) -> int:
    record = _terminal_record(
        status="PREFLIGHT_FAILED_TERMINAL",
        reason_code=reason_code,
        authority_sha256=None,
        consumed_ledger_sha256=None,
        result_file_sha256=None,
        credential_consumed=False,
        invocation_count=0,
        payload_open_count=0,
        orphaned=False,
        failure_detail=f"{type(error).__name__}:{error}",
        failure_stage="TRANSPORT" if reason_code == "TRANSPORT_ATTESTATION_FAILED" else "PREFLIGHT",
    )
    try:
        _publish_terminal(record)
    except Exception:
        return 4
    return 3


'''
    launcher = replace_block(launcher, "def _retire_preflight_failure(error: BaseException, reason_code: str) -> int:", "def _validate_generated", retire_block)
    consumed_block = '''def _run_consumed(plan: ExecutionPlan) -> int:
    counter = {"invocation_count": 0, "payload_open_count": 0}
    result_file_sha256: str | None = None
    try:
        counter["invocation_count"] = 1
        result_bytes = call_evaluator_once(lambda: plan.runtime.evaluate_bundle_once(lambda: _read_tensor_once(counter)))
        processed = process_evaluator_return(result_bytes, plan.runtime.validate_result_schema, plan.runtime.validate_result_record)

        def publish_result(raw: bytes) -> None:
            nonlocal result_file_sha256
            _durable_create(LIVE_PATHS["result"], raw)
            result_file_sha256 = sha256_bytes(raw)

        def build_terminal(result: dict[str, Any], raw: bytes) -> dict[str, Any]:
            terminal = result["terminal"]
            return _terminal_record(
                status=terminal["status"],
                reason_code=terminal["reason_code"],
                authority_sha256=plan.authority_sha256,
                consumed_ledger_sha256=plan.ledger_sha256,
                result_file_sha256=sha256_bytes(raw),
                credential_consumed=True,
                invocation_count=1,
                payload_open_count=1,
                orphaned=False,
                failure_detail="",
                failure_stage=None,
            )

        terminal_record = complete_result_path(
            processed,
            publish_result,
            build_terminal,
            plan.runtime.validate_terminal_record,
            _publish_terminal,
        )
        return 0 if terminal_record["status"] == "SUCCEEDED_TERMINAL" else 1
    except Exception as error:
        failure_stage = error.failure_stage if isinstance(error, EvaluatorReturnPathError) else "TERMINAL_BUILD"
        try:
            terminal_record = _terminal_record(
                status="CONSUMED_ORPHAN",
                reason_code="POST_CONSUMPTION_AMBIGUITY",
                authority_sha256=plan.authority_sha256,
                consumed_ledger_sha256=plan.ledger_sha256,
                result_file_sha256=result_file_sha256,
                credential_consumed=True,
                invocation_count=counter["invocation_count"],
                payload_open_count=counter["payload_open_count"],
                orphaned=True,
                failure_detail=f"{type(error).__name__}:{error}",
                failure_stage=failure_stage,
            )
            plan.runtime.validate_terminal_record(terminal_record)
            _publish_terminal(terminal_record)
        except Exception:
            pass
        return 2


'''
    launcher = replace_block(launcher, "def _run_consumed(plan: ExecutionPlan) -> int:", "def main() -> int:", consumed_block)
    launcher = launcher.replace(
        '                failure_detail=f"{type(error).__name__}:{error}",\n            )',
        '                failure_detail=f"{type(error).__name__}:{error}",\n                failure_stage="LIVE_MATERIALIZATION",\n            )',
        1,
    )
    launcher = launcher.replace('V13_ACTION_ID = "ace2:qk-gbfp8-base-v13:execute-once:81a8edc1:20260814T075838Z"\n', f'V15_ACTION_ID = "{V15_ACTION_ID}"\n', 1)
    require("_verify_retained_terminals" not in launcher, "retained-terminal verifier remains")
    require("failure_stage=None" in launcher, "successful failure-stage null absent")
    return launcher.encode("utf-8")


def render_transport() -> bytes:
    source_path = V15_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v15.py"
    transport = transform_text(source_path.read_text(encoding="utf-8"))
    transport = transport.replace("RUNTIME_DIRECTORIES = (RUNTIME_CONTAINER, RUNTIME_ROOT,", "RUNTIME_DIRECTORIES = (RUNTIME_ROOT,", 1)
    marker = '        "failure_detail_sha256": sha256_bytes(f"{type(error).__name__}:{error}".encode("utf-8")),\n'
    require(marker in transport, "transport terminal detail marker")
    transport = transport.replace(marker, marker + '        "failure_stage": "TRANSPORT",\n', 1)
    ast.parse(transport, filename="qk_gbfp8_head64_granularity_sweep_transport_shellfree_v16.py")
    require('"failure_stage": "TRANSPORT"' in transport, "transport failure-stage regression")
    return transport.encode("utf-8")


def bind(binding_id: str, final_path: Path, source_path: Path | None = None) -> dict[str, Any]:
    return {"id": binding_id, "path": str(final_path), "sha256": sha256_file(source_path or final_path)}


def v15_static_hashes() -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(V15_ROOT.rglob("*")):
        if path.is_file() and not path.is_symlink():
            records[path.relative_to(V15_ROOT).as_posix()] = sha256_file(path)
    return records


def preserved_v16_candidate_static_hashes() -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(PRESERVED_V16_CANDIDATE_ROOT.rglob("*")):
        if path.is_file() and not path.is_symlink():
            records[path.relative_to(PRESERVED_V16_CANDIDATE_ROOT).as_posix()] = sha256_file(path)
    return records


def main() -> int:
    require(not V16_ROOT.exists(), f"refusing to overwrite V16 package: {V16_ROOT}")
    require(not RUNTIME_ROOT.exists(), f"refusing to overwrite V16 runtime: {RUNTIME_ROOT}")
    require(V15_ROOT.is_dir(), "retired V15 package absent")
    require(PRESERVED_V16_CANDIDATE_ROOT.is_dir() and not PRESERVED_V16_CANDIDATE_ROOT.is_symlink(), "preserved rejected V16 candidate absent")
    require(stat.S_IMODE(os.lstat(PRESERVED_V16_CANDIDATE_ROOT).st_mode) == 0o555, "preserved rejected V16 candidate mode")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == "7f7e672a264924c103eda7568edac54be0a8b9a72643e75c8ac91f136e63152b", "sealed contract hash")
    require(VERIFIER_SOURCE.is_file(), "independent verifier source absent")
    require(INTERPRETER.is_file(), "frozen interpreter absent")
    package_stage = V16_ROOT.with_name(f".{V16_ROOT.name}.staging-{os.getpid()}")
    runtime_stage = RUNTIME_CONTAINER / f".{RUNTIME_ROOT.name}.staging-{os.getpid()}"
    require(not package_stage.exists() and not runtime_stage.exists(), "staging path exists")
    package_stage.mkdir(parents=True, mode=0o700)
    prepare_runtime_tree(runtime_stage)
    try:
        accepted_package_source = V15_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
        accepted_schema_source = V15_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
        accepted_package_stage = package_stage / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
        accepted_schema_stage = package_stage / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
        write_bytes(accepted_package_stage, accepted_package_source.read_bytes())
        write_bytes(accepted_schema_stage, accepted_schema_source.read_bytes())

        schema_names = ("AUTHORITY_SCHEMA", "CREDENTIAL_SCHEMA", "FIRST_TERMINAL_SCHEMA", "FRESH_L2_ACCEPTANCE_SCHEMA", "LEDGER_SCHEMA")
        schema_stages: dict[str, Path] = {}
        for name in schema_names:
            source = V15_ROOT / f"reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_{name}.json"
            target = package_stage / f"reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_{name}.json"
            schema = strict_object(source.read_bytes())
            if name == "FIRST_TERMINAL_SCHEMA":
                schema = patch_terminal_schema(schema)
            elif name == "FRESH_L2_ACCEPTANCE_SCHEMA":
                schema = patch_acceptance_schema(schema)
            else:
                schema = transform_value(schema)
            write_bytes(target, compact_bytes(schema))
            schema_stages[name] = target

        shared_stage = package_stage / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v16.py"
        result_validation_stage = package_stage / "tools/qk_gbfp8_head64_granularity_sweep_result_validation_v16.py"
        fixture_stage = package_stage / "tools/qk_gbfp8_head64_granularity_sweep_marker_free_fixture_v16.py"
        verifier_stage = package_stage / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree.py"
        fixture_report_stage = package_stage / "evidence/MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE_REPORT.json"
        write_bytes(shared_stage, render_shared_module())
        write_bytes(result_validation_stage, render_result_validation_module(accepted_schema_stage.read_bytes()))
        write_bytes(fixture_stage, render_repaired_fixture())
        write_bytes(verifier_stage, VERIFIER_SOURCE.read_bytes())

        fixture_run = subprocess.run(
            [str(INTERPRETER), str(fixture_stage)],
            cwd=package_stage,
            env=EXACT_ENVIRONMENT | {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
        require(fixture_run.returncode == 0, f"synthetic fixture failed: {fixture_run.stdout.decode('utf-8', 'replace')}")
        fixture_report = strict_object(fixture_run.stdout)
        require(fixture_report.get("status") == "PASS_MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE", "fixture report status")
        require(fixture_report.get("official_payload_open_count") == 0 and fixture_report.get("official_target_process_starts") == 0, "fixture execution boundary")
        write_bytes(fixture_report_stage, compact_bytes(fixture_report))

        accepted_paths = {
            "preauthority_contract": CONTRACT,
            "v15_package": V15_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json",
            "v15_static_acceptance": V15_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json",
            "package": V16_ROOT / accepted_package_stage.relative_to(package_stage),
            "result_schema": V16_ROOT / accepted_schema_stage.relative_to(package_stage),
            "controller": V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
            "evaluator": V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
            "c02_parser": V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py",
        }
        accepted_hash_sources = dict(accepted_paths)
        accepted_hash_sources["package"] = accepted_package_stage
        accepted_hash_sources["result_schema"] = accepted_schema_stage
        local_paths = {
            "authority_schema": V16_ROOT / schema_stages["AUTHORITY_SCHEMA"].relative_to(package_stage),
            "credential_schema": V16_ROOT / schema_stages["CREDENTIAL_SCHEMA"].relative_to(package_stage),
            "ledger_schema": V16_ROOT / schema_stages["LEDGER_SCHEMA"].relative_to(package_stage),
            "first_terminal_schema": V16_ROOT / schema_stages["FIRST_TERMINAL_SCHEMA"].relative_to(package_stage),
            "fresh_l2_acceptance_schema": V16_ROOT / schema_stages["FRESH_L2_ACCEPTANCE_SCHEMA"].relative_to(package_stage),
            "shared_result_path": V16_ROOT / shared_stage.relative_to(package_stage),
            "result_validation": V16_ROOT / result_validation_stage.relative_to(package_stage),
            "marker_free_fixture": V16_ROOT / fixture_stage.relative_to(package_stage),
            "fixture_report": V16_ROOT / fixture_report_stage.relative_to(package_stage),
            "transport": V16_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v16.py",
            "launcher": V16_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v16.py",
            "static_verifier": V16_ROOT / verifier_stage.relative_to(package_stage),
        }
        launcher_stage = package_stage / local_paths["launcher"].relative_to(V16_ROOT)
        transport_stage = package_stage / local_paths["transport"].relative_to(V16_ROOT)
        write_bytes(launcher_stage, render_launcher(accepted_paths, accepted_hash_sources, local_paths))
        write_bytes(transport_stage, render_transport())

        local_stage_lookup = {
            local_paths["authority_schema"]: schema_stages["AUTHORITY_SCHEMA"],
            local_paths["credential_schema"]: schema_stages["CREDENTIAL_SCHEMA"],
            local_paths["ledger_schema"]: schema_stages["LEDGER_SCHEMA"],
            local_paths["first_terminal_schema"]: schema_stages["FIRST_TERMINAL_SCHEMA"],
            local_paths["fresh_l2_acceptance_schema"]: schema_stages["FRESH_L2_ACCEPTANCE_SCHEMA"],
            local_paths["shared_result_path"]: shared_stage,
            local_paths["result_validation"]: result_validation_stage,
            local_paths["marker_free_fixture"]: fixture_stage,
            local_paths["fixture_report"]: fixture_report_stage,
            local_paths["transport"]: transport_stage,
            local_paths["launcher"]: launcher_stage,
            local_paths["static_verifier"]: verifier_stage,
        }

        old_package = strict_object((V15_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V15_SHELLFREE_PACKAGE.json").read_bytes())
        package = transform_value(old_package)
        package.pop("package_content_sha256", None)
        package["artifact_kind"] = "qk_gbfp8_head64_granularity_sweep_execution_v16_shellfree_package"
        package["root_id"] = "qk_gbfp8_head64_granularity_sweep_execution_v16_schema_fixture_repair_shellfree_action_root"
        package["required_disposition"] = DISPOSITION
        package["action_identity"] = {
            "future_action_id": ACTION_ID,
            "prior_action_id": V15_ACTION_ID,
            "replacement_is_distinct": True,
            "retry_or_resume": False,
            "reuse_permitted": False,
        }
        package["canonical_bindings"] = [bind(key, path, accepted_hash_sources[key]) for key, path in accepted_paths.items()]
        package["local_artifact_bindings"] = [bind(key, path, local_stage_lookup[path]) for key, path in local_paths.items()]
        package["preauthority_contract"] = {
            "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
            "path": str(CONTRACT),
            "sha256": sha256_file(CONTRACT),
            "stop_before_authority_or_credential": True,
        }
        package["v15_retirement"] = {
            "action_id": V15_ACTION_ID,
            "mutation_permitted": False,
            "replay_or_repair_permitted": False,
            "root": str(V15_ROOT),
            "static_file_sha256": v15_static_hashes(),
            "status": "CONSUMED_ORPHAN_RETIRED",
        }
        package["preserved_rejected_v16_candidate"] = {
            "action_id": "ace2:qk-gbfp8-base-v16:execute-once:7f7e672a:20260814T100000Z",
            "mutation_permitted": False,
            "root": str(PRESERVED_V16_CANDIDATE_ROOT),
            "static_file_sha256": preserved_v16_candidate_static_hashes(),
            "status": "REJECTED_CANDIDATE_PRESERVED_UNMODIFIED",
        }
        package["failure_stage_contract"] = {
            "failure_detail_sha256_retained": True,
            "stages": list(FAILURE_STAGES),
            "successful_terminal_value": None,
        }
        package["evaluator_return_path"] = {
            "fixture_calls_shared_routine": True,
            "operations_in_order": ["CANONICAL_DECODE", "RESULT_SCHEMA", "RESULT_RECORD", "PREPARE_DURABLE_RESULT_BYTES"],
            "production_and_fixture_bind_exact_result_validators": True,
            "production_calls_shared_routine_after_evaluator_return": True,
            "result_validation_module": str(local_paths["result_validation"]),
            "result_validator_factory": "bind_result_validators",
            "shared_module": str(local_paths["shared_result_path"]),
            "shared_routine": "process_evaluator_return",
        }
        package["marker_free_fixture"] = {
            "official_evaluator_invoked": False,
            "official_payload_open_count": 0,
            "official_target_process_starts": 0,
            "report_path": str(local_paths["fixture_report"]),
            "report_sha256": sha256_file(fixture_report_stage),
            "synthetic_only": True,
        }
        package["selection"]["candidate_order"] = ["G8", "G4", "G2", "G1"]
        package["selection"]["policy"] = "FIRST_PASSING_CANDIDATE_IN_FIXED_ORDER_WITH_EVERY_HARD_GATE_REQUIRED"
        package["runtime_namespace"] = {
            "directory_mode_octal": "0700",
            "fallback_terminal": str(runtime_files()["fallback_terminal"]),
            "file_count": 0,
            "file_mode_octal": "0400",
            "owner_gid": EXPECTED_GID,
            "owner_uid": EXPECTED_UID,
            "paths": {key: str(path) for key, path in runtime_files().items() if key != "fallback_terminal"},
            "precreated_directories": [str(path) for path in runtime_directories()],
            "primary_root": str(PRIMARY_ROOT),
            "publication": "O_CREAT|O_EXCL then file fsync then parent-directory fsync",
            "runtime_root": str(RUNTIME_ROOT),
            "static_preparation_only": True,
            "terminal_publication_order": ["primary", "fallback"],
        }
        package["claim_boundary"] = {
            "acceptance_materialized": False,
            "authority_materialized": False,
            "controller_or_evaluator_invoked": False,
            "credential_materialized": False,
            "execution_authorized": False,
            "first_terminal_materialized": False,
            "ledger_materialized": False,
            "official_payload_open_count": 0,
            "official_target_process_starts": 0,
            "result_materialized": False,
            "runtime_namespace_file_count": 0,
            "transport_or_launcher_invoked": False,
        }
        package["accepted_byte_derivation"] = {
            "package": {
                "canonical_path": str(accepted_paths["package"]),
                "canonical_sha256": sha256_file(accepted_package_stage),
                "source_path": str(accepted_package_source),
                "source_raw_sha256": sha256_file(accepted_package_source),
                "semantic_json_equal": True,
            },
            "result_schema": {
                "canonical_path": str(accepted_paths["result_schema"]),
                "canonical_sha256": sha256_file(accepted_schema_stage),
                "source_path": str(accepted_schema_source),
                "source_raw_sha256": sha256_file(accepted_schema_source),
                "semantic_json_equal": True,
            },
        }
        transport_final = local_paths["transport"]
        launcher_final = local_paths["launcher"]
        package_final = V16_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_PACKAGE.json"
        acceptance_final = V16_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
        transport_argv = [str(INTERPRETER), str(transport_final), "--package", str(package_final), "--acceptance", str(acceptance_final), "--irreversible-action-id", ACTION_ID]
        launcher_argv = [str(INTERPRETER), str(launcher_final), "--package", str(package_final), "--acceptance", str(acceptance_final), "--irreversible-action-id", ACTION_ID]
        package["future_invocation"] = invocation(transport_argv)
        package["launcher_invocation"] = invocation(launcher_argv)
        attestation = {
            "api": "os.posix_spawn",
            "immediate_parent_argv": transport_argv,
            "immediate_parent_environment": EXACT_ENVIRONMENT,
            "immediate_parent_executable_path": str(INTERPRETER),
            "immediate_parent_executable_sha256": sha256_file(INTERPRETER),
            "immediate_parent_transport_path": str(transport_final),
            "immediate_parent_transport_sha256": sha256_file(transport_stage),
            "launcher_argv": launcher_argv,
            "launcher_environment": EXACT_ENVIRONMENT,
            "launcher_sha256": sha256_file(launcher_stage),
            "shell": False,
        }
        package["transport_contract"]["attestation_sha256"] = sha256_bytes(compact_bytes(attestation))
        package["transport_contract"]["direct_exec_api"] = "os.posix_spawn"
        package["transport_contract"]["environment_inheritance_permitted"] = False
        package["transport_contract"]["shell"] = False
        allowed_files = [
            "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_PACKAGE.json",
            accepted_package_stage.relative_to(package_stage).as_posix(),
            accepted_schema_stage.relative_to(package_stage).as_posix(),
            fixture_report_stage.relative_to(package_stage).as_posix(),
            *[path.relative_to(package_stage).as_posix() for path in schema_stages.values()],
            shared_stage.relative_to(package_stage).as_posix(),
            result_validation_stage.relative_to(package_stage).as_posix(),
            fixture_stage.relative_to(package_stage).as_posix(),
            transport_stage.relative_to(package_stage).as_posix(),
            launcher_stage.relative_to(package_stage).as_posix(),
            verifier_stage.relative_to(package_stage).as_posix(),
        ]
        package["static_file_policy"] = {
            "allowed_relative_files": sorted(allowed_files),
            "forbidden_directories": ["build", "live", "__pycache__"],
            "generated_python_artifacts_permitted": False,
            "reviewer_optional_relative_files": ["review/FRESH_L2_STATIC_ACCEPTANCE.json"],
            "runtime_namespace_outside_package_root": str(RUNTIME_ROOT),
            "static_files_only": True,
        }
        for record in package.get("record_formats", {}).values():
            if "schema_path" in record:
                record["schema_path"] = transform_text(record["schema_path"])
                stage_schema = package_stage / Path(record["schema_path"]).relative_to(V16_ROOT)
                if stage_schema.is_file():
                    record["schema_sha256"] = sha256_file(stage_schema)
        package["package_content_sha256"] = sha256_bytes(compact_bytes(package))
        package_stage_path = package_stage / package_final.relative_to(V16_ROOT)
        write_bytes(package_stage_path, compact_bytes(package))

        for path in sorted(package_stage.rglob("*"), reverse=True):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted((item for item in package_stage.rglob("*") if item.is_dir()), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(package_stage, 0o555)
        os.rename(runtime_stage, RUNTIME_ROOT)
        os.rename(package_stage, V16_ROOT)
    except Exception:
        if package_stage.exists():
            os.chmod(package_stage, 0o700)
            shutil.rmtree(package_stage)
        if runtime_stage.exists():
            shutil.rmtree(runtime_stage)
        raise

    output = {
        "action_id": ACTION_ID,
        "fixture_report_sha256": sha256_file(V16_ROOT / "evidence/MARKER_FREE_EVALUATOR_RETURN_PATH_FIXTURE_REPORT.json"),
        "manifest_raw_sha256": sha256_file(V16_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V16_SHELLFREE_PACKAGE.json"),
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_root": str(V16_ROOT),
        "runtime_file_count": 0,
        "runtime_root": str(RUNTIME_ROOT),
        "status": "BUILT_IMMUTABLE_V16_MARKER_FREE_PREAUTHORITY_NO_EXECUTION_AUTHORITY",
    }
    print(compact_bytes(output).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
