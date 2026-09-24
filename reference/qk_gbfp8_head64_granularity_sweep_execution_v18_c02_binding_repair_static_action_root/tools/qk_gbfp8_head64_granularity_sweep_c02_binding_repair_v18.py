#!/usr/bin/env python3
"""Static-only V18 C02 binding closure.

This module consumes caller-provided public metadata only.  It has no payload
reader, evaluator launcher, authority materializer, or runtime entry point.
"""

from __future__ import annotations

import re
from typing import Any


ACTION_ID = "ace2:qk-gbfp8-base-v18:static-c02-binding-repair:additive-0001"
CLAIM_BOUNDARY = "STATIC_ONLY_NO_EXECUTION_AUTHORITY"
EXPECTED_RECORD_COUNT = 25
SELECTED_ALIASES = (
    "bf16_oracle_scores",
    "realized_key_source",
    "realized_query_source",
)
EXPECTED_SELECTED_NAMES = (
    "bf16.k_rope",
    "bf16.q_rope",
    "bf16.qk_scaled_scores",
)
STABLE_FAILURE_STAGES = frozenset({
    "C02_BINDING_VALIDATION",
    "C02_CROSS_PARSE",
    "NUMERICAL_SELECTION",
    "RESULT_PUBLICATION",
    "EVALUATOR_INTERNAL",
})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class V18BindingError(ValueError):
    """Public metadata or static V18 context is not exactly bound."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise V18BindingError(message)


def valid_sha256(value: Any) -> bool:
    return type(value) is str and SHA256_PATTERN.fullmatch(value) is not None


def exact_binding_table(lane_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract the exact 25-record table from sealed non-payload metadata."""

    require(type(lane_metadata) is dict, "lane metadata type")
    bundle = lane_metadata.get("tensor_bundle")
    require(
        type(bundle) is dict
        and set(bundle) == {"bytes", "path", "sha256", "tensor_count", "tensors"},
        "lane tensor bundle exact keys",
    )
    require(bundle["tensor_count"] == EXPECTED_RECORD_COUNT, "lane tensor record count")
    require(type(bundle["bytes"]) is int and bundle["bytes"] > 0, "lane tensor bundle bytes")
    require(valid_sha256(bundle["sha256"]), "lane tensor bundle sha256")
    tensors = bundle["tensors"]
    require(type(tensors) is dict and len(tensors) == EXPECTED_RECORD_COUNT, "lane tensor table cardinality")

    bindings: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        source = tensors[name]
        require(type(name) is str and name, "lane tensor name")
        require(type(source) is dict and set(source) == {"dtype", "sha256", "shape"}, f"lane tensor exact keys: {name}")
        dtype = source["dtype"]
        shape = source["shape"]
        digest = source["sha256"]
        require(type(dtype) is str and dtype.startswith("torch."), f"lane tensor dtype: {name}")
        require(
            type(shape) is list
            and 1 <= len(shape) <= 8
            and all(type(dimension) is int and dimension > 0 for dimension in shape),
            f"lane tensor shape: {name}",
        )
        require(valid_sha256(digest), f"lane tensor sha256: {name}")
        bindings[name] = {"dtype": dtype, "sha256": digest, "shape": list(shape)}
    require(len(bindings) == EXPECTED_RECORD_COUNT, "complete binding table cardinality")
    return bindings


def selected_tensor_names(accepted_package: dict[str, Any]) -> tuple[str, ...]:
    """Return the frozen three-record numerical selection."""

    require(type(accepted_package) is dict, "accepted package type")
    try:
        records = accepted_package["official_benchmark"]["input_bindings"]["tensor_records"]
    except (KeyError, TypeError) as error:
        raise V18BindingError("accepted package tensor records") from error
    require(type(records) is dict and tuple(sorted(records)) == SELECTED_ALIASES, "selected aliases")
    names: list[str] = []
    for alias in SELECTED_ALIASES:
        record = records[alias]
        require(
            type(record) is dict and set(record) == {"dtype", "sha256", "shape", "tensor_name"},
            f"selected record exact keys: {alias}",
        )
        names.append(record["tensor_name"])
    selected = tuple(sorted(names))
    require(selected == EXPECTED_SELECTED_NAMES, "selected tensor names")
    return selected


def prepare_static_context(
    accepted_package: dict[str, Any], lane_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Construct V18's complete parser bindings and frozen numerical context."""

    bindings = exact_binding_table(lane_metadata)
    selected = selected_tensor_names(accepted_package)
    input_bindings = accepted_package["official_benchmark"]["input_bindings"]
    for alias in SELECTED_ALIASES:
        record = input_bindings["tensor_records"][alias]
        expected = {key: record[key] for key in ("dtype", "sha256", "shape")}
        require(bindings.get(record["tensor_name"]) == expected, f"selected record metadata agreement: {alias}")
    bundle = lane_metadata["tensor_bundle"]
    require(bundle["sha256"] == input_bindings["tensor_bundle"]["sha256"], "bundle hash agreement")
    require(bundle["bytes"] == input_bindings["tensor_bundle"]["byte_count"], "bundle size agreement")
    return {
        "action_id": ACTION_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "model_identity_sha256": accepted_package["official_benchmark"]["model_identity_sha256"],
        "numerical_tensor_names": list(selected),
        "official_input_bindings": input_bindings,
        "static_package_id": accepted_package["package_id"],
        "tensor_bindings": bindings,
    }


def select_numerical_records(
    parsed_records: dict[str, dict[str, Any]], context: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Expose exactly the frozen three tensors to a numerical consumer."""

    require(type(parsed_records) is dict, "parsed records type")
    bindings = context.get("tensor_bindings")
    names = context.get("numerical_tensor_names")
    require(type(bindings) is dict and set(parsed_records) == set(bindings), "parsed record closure")
    require(type(names) is list and tuple(names) == EXPECTED_SELECTED_NAMES, "numerical selection")
    return {name: parsed_records[name] for name in names}


def sanitized_failure(error: BaseException, failure_stage: str) -> dict[str, str]:
    """Preserve exception type and stable stage without retaining message text."""

    require(isinstance(error, BaseException), "diagnostic exception")
    require(failure_stage in STABLE_FAILURE_STAGES, "diagnostic failure stage")
    exception_type = type(error).__name__
    require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", exception_type) is not None, "diagnostic exception type")
    return {"exception_type": exception_type, "failure_stage": failure_stage}


def main() -> int:
    raise RuntimeError("STATIC_ONLY_NO_EXECUTION_AUTHORITY")


if __name__ == "__main__":
    raise SystemExit(main())
