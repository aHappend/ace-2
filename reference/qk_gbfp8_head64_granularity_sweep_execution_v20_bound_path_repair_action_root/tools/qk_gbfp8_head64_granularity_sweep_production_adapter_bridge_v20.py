#!/usr/bin/env python3
"""Frozen V20 production binding for the shared evaluator adapter."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Callable

from qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v20 import (
    AdapterResult,
    EvaluatorInvocationError,
    InvocationSpec,
    compact_bytes,
    invoke_evaluator,
)


ACTION_ID = 'ace2:qk-gbfp8-base-v20:execute-once:a81f2916:additive-0001'
ROOT = '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root'
WORKER = ROOT + "/tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v20.py"
PACKAGE = ROOT + "/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json"
ACCEPTANCE = '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json'
EXACT_ENVIRONMENT = (('LANG', 'C'), ('LC_ALL', 'C'), ('PYTHONDONTWRITEBYTECODE', '1'), ('PYTHONHASHSEED', '0'), ('TZ', 'UTC'))
PRODUCTION_SPEC = InvocationSpec(
    argv=(
        '/home/argustest/miniconda3/bin/python3.13',
        WORKER,
        "--mode",
        "production",
        "--package",
        PACKAGE,
        "--acceptance",
        ACCEPTANCE,
        "--irreversible-action-id",
        ACTION_ID,
    ),
    cwd=ROOT,
    environment=EXACT_ENVIRONMENT,
    max_capture_bytes=16 * 1024 * 1024,
)


def pack_production_input(payload: bytes, bindings: dict[str, Any], context: dict[str, Any]) -> bytes:
    if type(payload) is not bytes or type(bindings) is not dict or type(context) is not dict:
        raise TypeError("production adapter envelope ABI")
    return compact_bytes({
        "bindings": bindings,
        "context": context,
        "payload_base64": base64.b64encode(payload).decode("ascii"),
    })


def invoke_production_evaluator(
    payload: bytes,
    bindings: dict[str, Any],
    context: dict[str, Any],
    schema_validate: Callable[[dict[str, Any]], None],
    record_validate: Callable[[dict[str, Any], bytes], None],
    publish_result: Callable[[bytes], None],
) -> AdapterResult:
    envelope = pack_production_input(payload, bindings, context)
    return invoke_evaluator(PRODUCTION_SPEC, envelope, schema_validate, record_validate, publish_result)


def terminal_diagnostic_fields(error: EvaluatorInvocationError) -> dict[str, Any]:
    diagnostic = error.diagnostic
    failure_stage = "RESULT_PUBLICATION" if diagnostic["failure_class"] == "PUBLICATION" else "EVALUATOR_CALL"
    return {
        "evaluator_diagnostic": diagnostic,
        "failure_detail_sha256": hashlib.sha256(compact_bytes(diagnostic)).hexdigest(),
        "failure_stage": failure_stage,
    }


def main() -> int:
    raise RuntimeError("PREAUTHORITY_ONLY: V20 has no execution authority")


if __name__ == "__main__":
    raise SystemExit(main())
