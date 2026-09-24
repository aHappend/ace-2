#!/usr/bin/env python3
"""Run deterministic multi-prompt RTL chat and full-chain oracle regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RegressionError(RuntimeError):
    pass


EXPECTED_PROMPTS = 3
EXPECTED_LAYERS = 24
TOKENS_PER_PROMPT = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionError(message)


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), f"artifact is outside project: {path}")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def project_path(value: str) -> Path:
    relative = Path(value)
    require(not relative.is_absolute(), f"path is not project-relative: {value}")
    require(".." not in relative.parts, f"path escapes project: {value}")
    return ROOT / relative


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verify_output_checksums(output: Path) -> dict[str, Any]:
    sums_path = output / "SHA256SUMS"
    require(sums_path.is_file(), f"missing output checksums: {sums_path}")
    entries = 0
    for line in sums_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative_name = line.partition("  ")
        require(separator == "  ", f"invalid checksum line in {sums_path}")
        member_name = Path(relative_name)
        require(
            not member_name.is_absolute() and ".." not in member_name.parts,
            f"unsafe checksum member: {relative_name}",
        )
        member = output / member_name
        require(member.is_file(), f"missing checksum member: {member}")
        require(sha256_file(member) == digest, f"checksum mismatch: {member}")
        entries += 1
    require(entries > 0, f"empty checksum manifest: {sums_path}")
    return {
        "path": sums_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(sums_path),
        "entries_checked": entries,
    }


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("schema") == "ace2-prompt-suite-rtl-oracle-config-v1",
        "unsupported prompt-suite schema",
    )
    require(
        config.get("max_new_tokens") == TOKENS_PER_PROMPT,
        "suite must generate exactly four tokens",
    )
    cases = config.get("cases")
    require(
        isinstance(cases, list) and len(cases) == EXPECTED_PROMPTS,
        "suite requires exactly three prompts",
    )
    identifiers: set[str] = set()
    prompts: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), "suite case is not an object")
        identifier = case.get("id")
        prompt = case.get("prompt")
        require(
            isinstance(identifier, str)
            and identifier
            and all(character.isalnum() or character == "-" for character in identifier),
            "case id is not a non-empty slug",
        )
        require(
            isinstance(prompt, str) and bool(prompt.strip()),
            f"case {identifier} has an empty prompt",
        )
        require(identifier not in identifiers, f"duplicate case id: {identifier}")
        require(prompt not in prompts, f"duplicate prompt: {prompt}")
        require("reuse" not in case, f"case {identifier} is not a fresh execution")
        identifiers.add(identifier)
        prompts.add(prompt)


def validate_kv_growth(
    case_id: str,
    generated: list[int],
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    require(
        run_summary.get("software_transformer_or_logits_fallback") is False,
        f"case {case_id} used a software transformer or logits fallback",
    )
    require(
        run_summary.get("generated_token_ids") == generated,
        f"case {case_id} run-summary token records differ",
    )
    context = run_summary.get("context_contract")
    require(isinstance(context, dict), f"case {case_id} has no context contract")
    prompt_tokens = context.get("prompt_token_count")
    require(
        isinstance(prompt_tokens, int) and prompt_tokens > 0,
        f"case {case_id} has an invalid prompt-token count",
    )
    executions = run_summary.get("token_executions")
    expected_positions = prompt_tokens + len(generated) - 1
    require(
        isinstance(executions, list) and len(executions) == expected_positions,
        f"case {case_id} has incomplete token executions",
    )
    kv_append_bytes = 0
    for position, execution in enumerate(executions):
        require(
            isinstance(execution, dict)
            and execution.get("absolute_position") == position,
            f"case {case_id} has discontinuous position {position}",
        )
        layers = execution.get("layers")
        require(
            isinstance(layers, list) and len(layers) == EXPECTED_LAYERS,
            f"case {case_id} position {position} does not have 24 layers",
        )
        require(
            all(isinstance(layer, dict) for layer in layers)
            and [layer.get("layer_id") for layer in layers]
            == list(range(EXPECTED_LAYERS)),
            f"case {case_id} position {position} has discontinuous layer ids",
        )
        for layer in layers:
            require(
                layer.get("cache_length_before") == position
                and layer.get("cache_length_after") == position + 1,
                f"case {case_id} position {position} layer "
                f"{layer.get('layer_id')} has discontinuous K/V growth",
            )
            append_bytes = layer.get("kv_append_bytes")
            require(
                isinstance(append_bytes, int) and append_bytes > 0,
                f"case {case_id} position {position} layer "
                f"{layer.get('layer_id')} has no K/V append",
            )
            kv_append_bytes += append_bytes
    decode_transitions = len(executions[prompt_tokens:])
    require(
        decode_transitions == TOKENS_PER_PROMPT - 1,
        f"case {case_id} does not have three decode transitions",
    )
    return {
        "continuous_24_layer_kv_growth": True,
        "positions_checked": len(executions),
        "decode_transitions_checked": decode_transitions,
        "decode_layer_growth_checks": decode_transitions * EXPECTED_LAYERS,
        "kv_append_bytes_checked": kv_append_bytes,
        "software_transformer_or_logits_fallback": False,
    }


def validate_execution_result(
    case: dict[str, Any],
    result: dict[str, Any],
    generated: list[int],
) -> dict[str, Any]:
    attempt = project_path(result["attempt"]["path"]).resolve()
    run_summary_record = result["manifests"]["run_summary"]
    run_summary_path = project_path(run_summary_record["path"]).resolve()
    require(
        run_summary_path.is_relative_to(attempt / "runtime-output"),
        f"case {case['id']} run summary is outside its fresh attempt",
    )
    require(
        file_record(run_summary_path) == run_summary_record,
        f"case {case['id']} run-summary binding differs",
    )
    return validate_kv_growth(case["id"], generated, load_json(run_summary_path))


def validate_oracle_result(
    case: dict[str, Any],
    oracle_output: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    require(result.get("status") == "PASS", f"case {case['id']} oracle did not pass")
    tokenization = result["tokenization"]
    agreement = result["agreement"]
    prompt_sha256 = sha256_bytes(case["prompt"].encode("utf-8"))
    require(
        result["reused_inputs"]["prompt"]["sha256"] == prompt_sha256,
        f"case {case['id']} prompt hash differs from oracle input",
    )
    generated = agreement["generated_token_ids"]
    decoded = agreement["decoded_text"]
    require(
        isinstance(generated, list) and len(generated) == 4,
        f"case {case['id']} did not produce four tokens",
    )
    require(
        tokenization["generated_token_ids"] == generated,
        f"case {case['id']} token records differ",
    )
    require(
        isinstance(decoded, str)
        and bool(decoded.strip())
        and "\ufffd" not in decoded
        and any(character.isalpha() for character in decoded),
        f"case {case['id']} continuation is not readable",
    )
    require(
        agreement["integer_byte_mismatches"] == 0,
        f"case {case['id']} has integer mismatches",
    )
    require(
        agreement["selected_token_mismatches"] == 0,
        f"case {case['id']} has selected-token mismatches",
    )
    require(
        agreement["kv_append_bytes_compared"] > 0
        and agreement["full_cache_bytes_compared"] > 0
        and agreement["quantized_layer_output_bytes_compared"] > 0
        and agreement["full_vocabulary_logit_bytes_compared"] > 0,
        f"case {case['id']} did not compare every required oracle surface",
    )
    require(
        agreement["rank1_positions_checked"] == agreement["positions_checked"],
        f"case {case['id']} quantized decode coverage is incomplete",
    )
    require(
        agreement["full_vocabulary_head_steps_checked"] == len(generated),
        f"case {case['id']} logit coverage is incomplete",
    )
    require(
        result["constraints"]["max_new_tokens"] == 4,
        f"case {case['id']} generation bound differs",
    )
    execution = validate_execution_result(case, result, generated)
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "prompt_sha256": prompt_sha256,
        "origin": "fresh_rtl_and_oracle",
        "attempt": result["attempt"],
        "oracle_output": {
            "path": oracle_output.relative_to(ROOT).as_posix(),
            "result": file_record(oracle_output / "result.json"),
            "checksums": verify_output_checksums(oracle_output),
            "implementation_sha256": result["oracle_implementation"]["aggregate_sha256"],
        },
        "manifests": result["manifests"],
        "source_binding": result["source_binding"],
        "reused_inputs": result["reused_inputs"],
        "tokenization": tokenization,
        "agreement": agreement,
        "execution": execution,
        "timing": result["timing"],
        "runtime": result["runtime"],
    }


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return completed.returncode


def failure_cone(
    case: dict[str, Any],
    stage: str,
    command: list[str],
    returncode: int,
    stdout_path: Path,
    stderr_path: Path,
    attempt: Path,
) -> dict[str, Any]:
    cone: dict[str, Any] = {
        "case_id": case["id"],
        "stage": stage,
        "returncode": returncode,
        "command": command,
        "stdout": file_record(stdout_path),
        "stderr": file_record(stderr_path),
        "dependency_chain": [
            "fixed prompt and tokenizer",
            "24-layer quantized RTL simulation with persistent K/V",
            "full-vocabulary RTL logits and selected tokens",
            "independent full-chain host oracle",
            "suite aggregation",
        ],
    }
    attempt_result = attempt / "attempt-result.json"
    if attempt_result.is_file():
        cone["attempt_result"] = load_json(attempt_result)
        cone["attempt_result_record"] = file_record(attempt_result)
    return cone


def run_fresh_case(
    case: dict[str, Any],
    output: Path,
    max_new_tokens: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    case_dir = output / "cases" / case["id"]
    case_dir.mkdir(parents=True)
    prompt_path = case_dir / "prompt.utf8"
    prompt_path.write_text(case["prompt"], encoding="utf-8")
    attempt = case_dir / "rtl-attempt"
    oracle_output = case_dir / "full-chain-oracle"
    attempt_stdout = case_dir / "rtl-attempt.stdout.log"
    attempt_stderr = case_dir / "rtl-attempt.stderr.log"
    attempt_command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(ROOT / "tools/ace2_v73_stage1_chat_attempt.py"),
        "--prompt-file",
        str(prompt_path),
        "--output",
        str(attempt),
        "--max-new-tokens",
        str(max_new_tokens),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    returncode = run_logged(attempt_command, attempt_stdout, attempt_stderr)
    if returncode != 0:
        return None, failure_cone(
            case,
            "rtl_attempt",
            attempt_command,
            returncode,
            attempt_stdout,
            attempt_stderr,
            attempt,
        )

    oracle_stdout = case_dir / "full-chain-oracle.stdout.log"
    oracle_stderr = case_dir / "full-chain-oracle.stderr.log"
    oracle_command = [
        str(Path(sys.executable).resolve()),
        "-B",
        str(ROOT / "scripts/verify_full_chain_independent_oracle.py"),
        "--attempt",
        str(attempt),
        "--output",
        str(oracle_output),
    ]
    returncode = run_logged(oracle_command, oracle_stdout, oracle_stderr)
    if returncode != 0:
        return None, failure_cone(
            case,
            "full_chain_independent_oracle",
            oracle_command,
            returncode,
            oracle_stdout,
            oracle_stderr,
            attempt,
        )
    return validate_oracle_result(case, oracle_output, load_json(oracle_output / "result.json")), None


def common_binding(cases: list[dict[str, Any]], field: str) -> str:
    values = {case[field]["sha256"] for case in cases}
    require(len(values) == 1, f"suite cases use different {field} bindings")
    return values.pop()


def build_result(
    config_path: Path,
    output: Path,
    cases: list[dict[str, Any]],
    status: str,
    failure: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "ace2-prompt-suite-rtl-oracle-regression-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "reproduction_command": (
            "make prompt-suite-rtl-oracle-regression "
            f"ACE2_CHAT_PROMPT_SUITE={config_path.relative_to(ROOT).as_posix()} "
            f"ACE2_CHAT_PROMPT_SUITE_OUTPUT={output.relative_to(ROOT).as_posix()}"
        ),
        "suite_config": file_record(config_path),
        "implementation": file_record(Path(__file__)),
        "cases": cases,
        "coverage": {
            "configured_prompts": len(load_json(config_path)["cases"]),
            "completed_prompts": len(cases),
            "fresh_rtl_prompts": sum(
                case["origin"] == "fresh_rtl_and_oracle" for case in cases
            ),
            "generated_tokens": sum(
                len(case["agreement"]["generated_token_ids"]) for case in cases
            ),
            "positions_checked": sum(
                case["agreement"]["positions_checked"] for case in cases
            ),
            "layers_checked": sum(case["agreement"]["layers_checked"] for case in cases),
            "integer_byte_mismatches": sum(
                case["agreement"]["integer_byte_mismatches"] for case in cases
            ),
            "selected_token_mismatches": sum(
                case["agreement"]["selected_token_mismatches"] for case in cases
            ),
            "kv_append_bytes_compared": sum(
                case["agreement"]["kv_append_bytes_compared"] for case in cases
            ),
            "full_cache_bytes_compared": sum(
                case["agreement"]["full_cache_bytes_compared"] for case in cases
            ),
            "quantized_layer_output_bytes_compared": sum(
                case["agreement"]["quantized_layer_output_bytes_compared"]
                for case in cases
            ),
            "full_vocabulary_logit_bytes_compared": sum(
                case["agreement"]["full_vocabulary_logit_bytes_compared"]
                for case in cases
            ),
            "full_vocabulary_head_steps_checked": sum(
                case["agreement"]["full_vocabulary_head_steps_checked"]
                for case in cases
            ),
            "continuous_24_layer_kv_growth_prompts": sum(
                case["execution"]["continuous_24_layer_kv_growth"] for case in cases
            ),
            "decode_transitions_checked": sum(
                case["execution"]["decode_transitions_checked"] for case in cases
            ),
            "decode_layer_growth_checks": sum(
                case["execution"]["decode_layer_growth_checks"] for case in cases
            ),
            "software_transformer_or_logits_fallback": any(
                case["execution"]["software_transformer_or_logits_fallback"]
                for case in cases
            ),
        },
        "measurement_scope": (
            "computer-local host/orchestration and RTL simulation only; "
            "no hardware timing"
        ),
        "limitations": [
            "Three fixed natural-language prompts with four-token greedy continuations; this is not an exhaustive language-quality evaluation.",
            "The independent oracle shares accepted fixed-point primitives with vector generation; implementation-diverse arithmetic is not claimed.",
            "The RTL runner generates input-specific testbenches, so no compatible compiled-simulator artifact was reused across different prompts.",
            "No FPGA, synthesis, STA, PPA, bitstream, deployed-hardware, or silicon work was run or claimed.",
        ],
    }
    if cases:
        result["bindings"] = {
            "source_and_rtl_sha256": common_binding(cases, "source_binding"),
            "model_sha256": common_binding(
                [{"model": case["reused_inputs"]["model"]} for case in cases],
                "model",
            ),
            "adapter_sha256": common_binding(
                [{"adapter": case["reused_inputs"]["adapter"]} for case in cases],
                "adapter",
            ),
            "tokenizer_source_sha256": next(
                iter(
                    {
                        case["reused_inputs"]["tokenizer"]["source_files_sha256"]
                        for case in cases
                    }
                )
            ),
        }
        require(
            len(
                {
                    case["reused_inputs"]["tokenizer"]["source_files_sha256"]
                    for case in cases
                }
            )
            == 1,
            "suite cases use different tokenizer bindings",
        )
    if failure is not None:
        result["failing_dependency_cone"] = failure
    return result


def validate_pass_coverage(coverage: dict[str, Any]) -> None:
    require(
        coverage["configured_prompts"] == EXPECTED_PROMPTS,
        "suite prompt count differs",
    )
    require(coverage["completed_prompts"] == EXPECTED_PROMPTS, "suite is incomplete")
    require(coverage["fresh_rtl_prompts"] == EXPECTED_PROMPTS, "suite reused a prompt")
    require(
        coverage["generated_tokens"] == EXPECTED_PROMPTS * TOKENS_PER_PROMPT,
        "suite did not generate exactly 12 tokens",
    )
    require(
        coverage["continuous_24_layer_kv_growth_prompts"] == EXPECTED_PROMPTS
        and coverage["decode_transitions_checked"]
        == EXPECTED_PROMPTS * (TOKENS_PER_PROMPT - 1)
        and coverage["decode_layer_growth_checks"]
        == EXPECTED_PROMPTS * (TOKENS_PER_PROMPT - 1) * EXPECTED_LAYERS,
        "suite K/V growth coverage is incomplete",
    )
    for field in (
        "kv_append_bytes_compared",
        "full_cache_bytes_compared",
        "quantized_layer_output_bytes_compared",
        "full_vocabulary_logit_bytes_compared",
        "full_vocabulary_head_steps_checked",
    ):
        require(coverage[field] > 0, f"suite {field} is zero")
    require(coverage["integer_byte_mismatches"] == 0, "suite has integer mismatches")
    require(coverage["selected_token_mismatches"] == 0, "suite has token mismatches")
    require(
        coverage["software_transformer_or_logits_fallback"] is False,
        "suite used a software transformer or logits fallback",
    )


def write_report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Deterministic multi-prompt RTL/oracle regression",
        "",
        f"**Status:** {result['status']}",
        "",
        "## Reproduction",
        "",
        "```bash",
        result["reproduction_command"],
        "```",
        "",
        "## Prompt coverage",
        "",
        "| Case | Origin | Prompt | Continuation | Tokens | Positions | Integer / token mismatches |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in result["cases"]:
        prompt = json.dumps(case["prompt"], ensure_ascii=False)
        decoded = json.dumps(case["agreement"]["decoded_text"], ensure_ascii=False)
        mismatch = (
            f"{case['agreement']['integer_byte_mismatches']} / "
            f"{case['agreement']['selected_token_mismatches']}"
        )
        lines.append(
            f"| `{case['id']}` | {case['origin']} | `{prompt}` | `{decoded}` | "
            f"{len(case['agreement']['generated_token_ids'])} | "
            f"{case['agreement']['positions_checked']} | {mismatch} |"
        )
    lines.extend(
        [
            "",
            "## Exact bindings",
            "",
            f"- Suite manifest: `{result['suite_config']['sha256']}`",
            f"- Suite implementation: `{result['implementation']['sha256']}`",
        ]
    )
    if "bindings" in result:
        lines.extend(
            [
                f"- Source/RTL aggregate: `{result['bindings']['source_and_rtl_sha256']}`",
                f"- Model: `{result['bindings']['model_sha256']}`",
                f"- Adapter: `{result['bindings']['adapter_sha256']}`",
                f"- Tokenizer sources: `{result['bindings']['tokenizer_source_sha256']}`",
            ]
        )
    for case in result["cases"]:
        lines.extend(
            [
                f"- `{case['id']}` attempt root: `{case['attempt']['tree_root_sha256']}`",
                f"- `{case['id']}` attempt manifest: `{case['manifests']['attempt_manifest']['sha256']}`",
                f"- `{case['id']}` run summary: `{case['manifests']['run_summary']['sha256']}`",
                f"- `{case['id']}` oracle result: `{case['oracle_output']['result']['sha256']}`",
            ]
        )
    lines.extend(["", "## Simulation/host-only timing", ""])
    for case in result["cases"]:
        rtl = case["timing"]["reused_rtl_attempt"]
        host = case["timing"]["independent_oracle_host"]
        lines.append(
            f"- `{case['id']}`: RTL-attempt wall {rtl['total_wall_seconds']:.6f} s; "
            f"Icarus compile {rtl['compile_wall_seconds']:.6f} s; "
            f"Icarus simulation {rtl['simulation_wall_seconds']:.6f} s; "
            f"generation host/orchestration {rtl['model_wall_seconds']:.6f} s; "
            f"independent-oracle host {host['total_host_wall_seconds']:.6f} s."
        )
    lines.extend(
        [
            "",
            "These measurements are computer-local simulation and host timings, not hardware latency.",
            "",
        ]
    )
    if "failing_dependency_cone" in result:
        cone = result["failing_dependency_cone"]
        lines.extend(
            [
                "## Failing dependency cone",
                "",
                f"- Case: `{cone['case_id']}`",
                f"- First failing stage: `{cone['stage']}`",
                f"- Return code: {cone['returncode']}",
                f"- Standard output: `{cone['stdout']['path']}`",
                f"- Standard error: `{cone['stderr']['path']}`",
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(output: Path, result: dict[str, Any]) -> None:
    (output / "result.json").write_bytes(canonical_bytes(result))
    write_report(output, result)
    members = [output / "result.json", output / "REPORT.md", output / "suite-config.json"]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="ascii",
    )


def run(config_path: Path, output: Path, timeout_seconds: int) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    require(config_path.is_relative_to(ROOT / "tests"), "--config must be under tests")
    require(
        output.is_relative_to(ROOT / "reports" / "verification"),
        "--output must be under reports/verification",
    )
    require(not output.exists(), f"suite output already exists: {output}")
    config = load_json(config_path)
    validate_config(config)
    require(timeout_seconds > 0, "--timeout-seconds must be positive")
    output.mkdir(parents=True)
    (output / "suite-config.json").write_bytes(canonical_bytes(config))

    completed_cases: list[dict[str, Any]] = []
    for case in config["cases"]:
        summary, failure = run_fresh_case(
            case,
            output,
            config["max_new_tokens"],
            timeout_seconds,
        )
        if failure is not None:
            result = build_result(
                config_path,
                output,
                completed_cases,
                "FAIL",
                failure,
            )
            finalize(output, result)
            return result
        require(summary is not None, f"case {case['id']} produced no result")
        completed_cases.append(summary)
        (output / "progress.json").write_bytes(
            canonical_bytes(
                {
                    "schema": "ace2-prompt-suite-progress-v1",
                    "status": "RUNNING",
                    "completed_case_ids": [item["id"] for item in completed_cases],
                }
            )
        )
    result = build_result(config_path, output, completed_cases, "PASS", None)
    validate_pass_coverage(result["coverage"])
    finalize(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=72_000)
    args = parser.parse_args()
    result = run(args.config, args.output, args.timeout_seconds)
    print(
        "PROMPT_SUITE_RTL_ORACLE_REGRESSION_"
        f"{result['status']} prompts={result['coverage']['completed_prompts']} "
        f"tokens={result['coverage']['generated_tokens']} "
        "continuous_24_layer_kv_growth="
        f"{result['coverage']['continuous_24_layer_kv_growth_prompts']} "
        f"integer_mismatches={result['coverage']['integer_byte_mismatches']} "
        f"token_mismatches={result['coverage']['selected_token_mismatches']} "
        "software_transformer_or_logits_fallback="
        f"{str(result['coverage']['software_transformer_or_logits_fallback']).lower()} "
        f"output={args.output}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PROMPT_SUITE_RTL_ORACLE_REGRESSION_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(1)
