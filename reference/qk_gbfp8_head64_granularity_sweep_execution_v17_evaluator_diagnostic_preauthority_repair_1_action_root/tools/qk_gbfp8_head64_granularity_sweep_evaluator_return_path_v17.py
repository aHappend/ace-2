#!/usr/bin/env python3
"""Pure evaluator-return processing shared by V17 production and fixture."""

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
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


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
