#!/usr/bin/env python3
"""Profile authenticated 4/6/8-token ACE-2 chat simulation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_TOKEN_COUNTS = (4, 6, 8)
PRODUCT_STATUS = "PASS_STAGE1_RTL_CHAT_PRODUCT_COHERENT"
PROFILE_SCOPE = (
    "computer-local RTL simulation and host/orchestration profiling only; "
    "not FPGA, deployed-hardware, synthesis, STA, PPA, or silicon performance"
)
ATTEMPT_REQUIRED_MEMBERS = {
    "attempt-manifest.json",
    "attempt-result.json",
    "runtime-output/product_result.json",
    "runtime-output/run_summary.json",
    "timing.json",
}


class ProfileError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"invalid JSON evidence: {path}") from error
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def parse_checksum_lines(raw: bytes, label: str) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProfileError(f"{label} SHA256SUMS is not ASCII") from error
    require(bool(lines), f"{label} SHA256SUMS is empty")
    records: dict[str, str] = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        relative = Path(name)
        require(
            separator == "  "
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"{label} SHA256SUMS has an invalid line",
        )
        require(
            bool(name)
            and not relative.is_absolute()
            and ".." not in relative.parts
            and name not in records,
            f"{label} SHA256SUMS has an invalid member path",
        )
        records[name] = digest
    return records


def verify_seal(
    directory: Path,
    *,
    label: str,
    required_members: set[str],
    require_tree_root: bool,
) -> dict[str, Any]:
    directory = directory.resolve()
    require(directory.is_dir(), f"missing {label} directory: {directory}")
    sums_path = directory / "SHA256SUMS"
    require(sums_path.is_file(), f"missing {label} SHA256SUMS")
    raw = sums_path.read_bytes()
    records = parse_checksum_lines(raw, label)
    require(
        required_members <= records.keys(),
        f"{label} seal is incomplete: missing required members",
    )
    bytes_checked = 0
    for name, expected_digest in records.items():
        member = directory / name
        require(
            member.is_file() and member.resolve().is_relative_to(directory),
            f"{label} sealed member is missing or escapes its directory: {name}",
        )
        require(
            sha256_file(member) == expected_digest,
            f"{label} sealed member hash mismatch: {name}",
        )
        bytes_checked += member.stat().st_size
    tree_root = hashlib.sha256(raw).hexdigest()
    if require_tree_root:
        root_path = directory / "TREE_ROOT.sha256"
        require(root_path.is_file(), f"missing {label} TREE_ROOT.sha256")
        require(
            root_path.read_text(encoding="ascii")
            == f"{tree_root}  SHA256SUMS\n",
            f"{label} SHA256SUMS tree root mismatch",
        )
    return {
        "tree_root_sha256": tree_root,
        "entries_checked": len(records),
        "authenticated_payload_bytes": bytes_checked,
    }


def numeric(value: object, name: str, *, positive: bool = False) -> float:
    require(
        type(value) in (int, float) and math.isfinite(value),
        f"{name} must be a finite number",
    )
    converted = float(value)
    require(
        converted > 0 if positive else converted >= 0,
        f"{name} must be {'positive' if positive else 'non-negative'}",
    )
    return converted


def metric(value: int | float, unit: str, scope: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "measurement_scope": scope}


def require_equal(left: object, right: object, message: str) -> None:
    require(left == right, message)


def validate_case(
    token_count: int,
    attempt: Path,
    oracle: Path,
) -> dict[str, Any]:
    require(
        token_count in EXPECTED_TOKEN_COUNTS,
        "token count must be exactly one of 4, 6, or 8",
    )
    attempt_seal = verify_seal(
        attempt,
        label=f"{token_count}-token attempt",
        required_members=ATTEMPT_REQUIRED_MEMBERS,
        require_tree_root=True,
    )
    oracle_seal = verify_seal(
        oracle,
        label=f"{token_count}-token oracle",
        required_members={"result.json"},
        require_tree_root=False,
    )

    attempt = attempt.resolve()
    manifest = load_object(attempt / "attempt-manifest.json")
    attempt_result = load_object(attempt / "attempt-result.json")
    product = load_object(attempt / "runtime-output/product_result.json")
    summary = load_object(attempt / "runtime-output/run_summary.json")
    timing = load_object(attempt / "timing.json")
    oracle_result = load_object(oracle.resolve() / "result.json")

    bounds = manifest.get("tokenization", {}).get("generation_bounds", {})
    require_equal(
        bounds.get("max_new_tokens"),
        token_count,
        f"{token_count}-token manifest token contract mismatched",
    )
    source_binding = manifest.get("source_and_rtl", {}).get("sha256")
    require(
        isinstance(source_binding, str) and len(source_binding) == 64,
        f"{token_count}-token manifest source binding is missing",
    )
    require_equal(
        attempt_result.get("status"),
        "PASS",
        f"{token_count}-token attempt is not PASS",
    )
    require(
        attempt_result.get("timed_out") is False,
        f"{token_count}-token attempt timed out or lacks explicit false",
    )
    require_equal(
        product.get("status"),
        PRODUCT_STATUS,
        f"{token_count}-token product is incomplete",
    )
    require_equal(
        summary.get("software_transformer_or_logits_fallback"),
        False,
        f"{token_count}-token summary fallback is enabled or unspecified",
    )
    require_equal(
        product.get("software_transformer_or_logits_fallback"),
        False,
        f"{token_count}-token product fallback is enabled or unspecified",
    )
    generated = summary.get("generated_token_ids")
    require(
        isinstance(generated, list) and len(generated) == token_count,
        f"{token_count}-token summary generated-token count mismatched",
    )
    for label, candidate in (
        ("attempt", attempt_result.get("generated_token_ids")),
        ("product", product.get("generated_token_ids")),
    ):
        require_equal(
            candidate,
            generated,
            f"{token_count}-token {label} token sequence mismatched",
        )

    require_equal(
        oracle_result.get("status"),
        "PASS",
        f"{token_count}-token oracle is not PASS",
    )
    require_equal(
        oracle_result.get("numerical_status"),
        "PASS",
        f"{token_count}-token oracle numerical status is not PASS",
    )
    require_equal(
        oracle_result.get("constraints", {}).get("max_new_tokens"),
        token_count,
        f"{token_count}-token oracle token contract mismatched",
    )
    oracle_attempt = oracle_result.get("attempt", {})
    require_equal(
        oracle_attempt.get("tree_root_sha256"),
        attempt_seal["tree_root_sha256"],
        f"{token_count}-token evidence is mixed-source: attempt tree root differs",
    )
    require_equal(
        oracle_result.get("source_binding", {}).get("sha256"),
        source_binding,
        f"{token_count}-token evidence is mixed-source: source binding differs",
    )
    agreement = oracle_result.get("agreement", {})
    require_equal(
        agreement.get("generated_token_ids"),
        generated,
        f"{token_count}-token oracle token sequence mismatched",
    )
    require_equal(
        agreement.get("full_vocabulary_head_steps_checked"),
        token_count,
        f"{token_count}-token oracle head-step coverage is incomplete",
    )
    for name in (
        "integer_byte_mismatches",
        "selected_token_mismatches",
    ):
        require_equal(
            agreement.get(name),
            0,
            f"{token_count}-token oracle reports {name}",
        )
    for name in (
        "kv_append_bytes_compared",
        "full_cache_bytes_compared",
        "quantized_layer_output_bytes_compared",
        "full_vocabulary_logit_bytes_compared",
        "retained_payload_bytes_compared",
    ):
        numeric(
            agreement.get(name),
            f"{token_count}-token oracle {name}",
            positive=True,
        )

    latency = timing.get("backend_latency")
    require(
        isinstance(latency, dict),
        f"{token_count}-token backend latency is missing",
    )
    oracle_timing = oracle_result.get("timing", {})
    reused_timing = oracle_timing.get("reused_rtl_attempt")
    require(
        isinstance(reused_timing, dict),
        f"{token_count}-token oracle reused timing is missing",
    )
    phase_fields = {
        "compile": "compile_wall_seconds",
        "host_orchestration": "model_wall_seconds",
        "rtl_simulation": "simulation_wall_seconds",
    }
    phase_values: dict[str, float] = {}
    for phase, field in phase_fields.items():
        phase_values[phase] = numeric(
            latency.get(field),
            f"{token_count}-token {field}",
            positive=True,
        )
        require_equal(
            reused_timing.get(field),
            latency.get(field),
            f"{token_count}-token oracle reused {field} mismatched",
        )
    wall_time = numeric(
        timing.get("supervisor_total_wall_seconds"),
        f"{token_count}-token supervisor wall time",
        positive=True,
    )
    oracle_host = oracle_timing.get("independent_oracle_host")
    require(
        isinstance(oracle_host, dict),
        f"{token_count}-token independent oracle timing is missing",
    )
    oracle_verification = numeric(
        oracle_host.get("total_host_wall_seconds"),
        f"{token_count}-token oracle verification wall time",
        positive=True,
    )
    process_tree = product.get("process_tree_rss")
    require(
        isinstance(process_tree, dict) and process_tree.get("complete") is True,
        f"{token_count}-token process-tree memory evidence is incomplete",
    )
    attempt_peak_rss = numeric(
        process_tree.get("peak_rss_kib"),
        f"{token_count}-token attempt peak RSS",
        positive=True,
    )
    oracle_peak_rss = numeric(
        oracle_host.get("peak_rss_kib"),
        f"{token_count}-token oracle peak RSS",
        positive=True,
    )

    decode_transitions = token_count - 1
    total_storage = (
        attempt_seal["authenticated_payload_bytes"]
        + oracle_seal["authenticated_payload_bytes"]
    )
    additive = {
        **phase_values,
        "oracle_verification": oracle_verification,
        "wall_time": wall_time,
        "authenticated_evidence_storage": total_storage,
    }
    raw = {
        name: metric(
            value,
            "byte" if name == "authenticated_evidence_storage" else "second",
            (
                "authenticated sealed evidence payload"
                if name == "authenticated_evidence_storage"
                else PROFILE_SCOPE
            ),
        )
        for name, value in additive.items()
    }
    raw["available_memory_metrics"] = {
        "attempt_process_tree_peak_rss": metric(
            int(attempt_peak_rss),
            "KiB",
            "complete sampled computer-local attempt process tree",
        ),
        "independent_oracle_peak_rss": metric(
            int(oracle_peak_rss),
            "KiB",
            "computer-local independent oracle host process",
        ),
        "host_mem_available": None,
        "limitation": "host-wide MemAvailable was not retained by these sealed attempts",
    }

    def normalized(divisor: int, suffix: str) -> dict[str, Any]:
        return {
            name: metric(
                value / divisor,
                (
                    f"byte/{suffix}"
                    if name == "authenticated_evidence_storage"
                    else f"second/{suffix}"
                ),
                raw[name]["measurement_scope"],
            )
            for name, value in additive.items()
        }

    return {
        "generated_tokens": token_count,
        "decode_transitions": decode_transitions,
        "authentication": {
            "attempt_tree_root_sha256": attempt_seal["tree_root_sha256"],
            "source_and_rtl_sha256": source_binding,
            "attempt_entries_checked": attempt_seal["entries_checked"],
            "oracle_entries_checked": oracle_seal["entries_checked"],
            "fallback": False,
            "oracle_numerical_status": "PASS",
        },
        "raw_metrics": raw,
        "normalized_metrics": {
            "per_generated_token": normalized(token_count, "generated_token"),
            "per_decode_transition": normalized(
                decode_transitions,
                "decode_transition",
            ),
        },
    }


def build_profile(cases: list[tuple[int, Path, Path]]) -> dict[str, Any]:
    require(
        len(cases) == len(EXPECTED_TOKEN_COUNTS),
        "exactly three --case values are required",
    )
    counts = [case[0] for case in cases]
    require(
        sorted(counts) == list(EXPECTED_TOKEN_COUNTS) and len(set(counts)) == 3,
        "--case values must provide exactly one 4-, 6-, and 8-token evidence set",
    )
    results = [validate_case(*case) for case in sorted(cases)]
    source_bindings = {
        result["authentication"]["source_and_rtl_sha256"] for result in results
    }
    require(
        len(source_bindings) == 1,
        "4/6/8-token cohort is mixed-source: source/RTL bindings differ",
    )
    return {
        "schema": "ace2-chat-token-scaling-profile-v1",
        "status": "PASS",
        "measurement_scope": PROFILE_SCOPE,
        "normalization_definitions": {
            "per_generated_token": "raw additive metric divided by generated_tokens",
            "per_decode_transition": (
                "raw additive metric divided by generated_tokens minus one"
            ),
            "peak_memory": "not normalized because peak RSS is not additive",
        },
        "cases": results,
    }


def parse_case(values: list[str]) -> tuple[int, Path, Path]:
    try:
        token_count = int(values[0])
    except ValueError as error:
        raise ProfileError("--case token count must be an integer") from error
    return token_count, Path(values[1]), Path(values[2])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile authenticated ACE-2 chat evidence as computer-local "
            "RTL-simulation and host measurements."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("TOKENS", "ATTEMPT", "ORACLE"),
        required=True,
        help="repeat for exactly the 4-, 6-, and 8-token evidence sets",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    profile = build_profile([parse_case(values) for values in args.case])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_bytes(profile))
    print(
        "ACE2_CHAT_TOKEN_SCALING_PROFILE_PASS "
        f"scope=computer-local-simulation-host output={output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
