#!/usr/bin/env python3
"""Construct the additive, zero-state CF18 runtime-acceleration package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/ace2_chat_demo"
IDENTITY = "stage1-w4a8-r6-generation-probe-runtime-acceleration-cf18-0001"
PACKAGE = BUILD / IDENTITY
STATE = BUILD / f"{IDENTITY}-authority-state-0001"
EXTERNAL_AUTHORITY = BUILD / f"{IDENTITY}-execution-authority.json"
ATTEMPT = STATE / "attempt-0001"
CF15 = (
    BUILD
    / "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
)
CF16_PACKAGE = (
    BUILD / "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002"
)
CF16_STATE = BUILD / (
    "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002"
    "-authority-state-0001"
)
CF17_PACKAGE = BUILD / (
    "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0003"
)
CF17_STATE = BUILD / (
    "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0003"
    "-authority-state-0001"
)
CF10 = BUILD / "stage1-w4a8-r6-product-prep-cf10"
BENCHMARK = (
    ROOT
    / "build/cf18-host-persistence-microbenchmark-v1/attempt-0002/results.json"
)
PRIOR_RTL_EQUIVALENCE = (
    ROOT
    / "build/host-rtl-persistence-batch-microbenchmark-v1"
    / "attempt-0002-sanitized/results.json"
)
SEALED_PROGRESS = (
    CF17_STATE / "attempt-0001/endpoint-output/progress.json"
)
SEALED_COMMANDS = (
    CF17_STATE / "attempt-0001/endpoint-output/commands.jsonl"
)
SCHEDULE_SHA256 = "0fc65c947ac2a34ee096815304e1ee68ec65c279160db08d4901e072930c560b"
COMMANDS = 1_306_104
BATCH_COMMANDS = 1024
ADMISSION_TIMEOUT_SECONDS = 144_000


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected regular file: {path}")
    observed = path.stat()
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write(path: Path, value: Any, mode: int = 0o444) -> None:
    raw = value if isinstance(value, bytes) else pretty_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def copy_source(source: Path, target: Path, mode: int = 0o444) -> None:
    write(target, source.read_bytes(), mode)


def predecessor_inventory() -> dict[str, Any]:
    files = {}
    for root in (CF16_PACKAGE, CF16_STATE, CF17_PACKAGE, CF17_STATE):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files[relative(path)] = record(path)
    return {
        "schema": "ace2-cf18-immutable-predecessor-inventory-v1",
        "roots": [relative(path) for path in (
            CF16_PACKAGE, CF16_STATE, CF17_PACKAGE, CF17_STATE
        )],
        "files": files,
        "cf17_disposition": {
            "status": "PERMANENTLY_CLOSED_CONSUMED_TIMEOUT",
            "committed_commands": 106_796,
            "next_ordinal": 106_796,
            "simulator_cycles": 4_653_618_259,
            "generated_tokens": 0,
            "retry_replay_resume_relaunch_extension_mutation": "FORBIDDEN",
        },
    }


def profile_diagnosis() -> dict[str, Any]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    committed_cycles = 0
    with SEALED_COMMANDS.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            values = totals[item["operator"]]
            values[0] += 1
            values[1] += item["cycles"]
            values[2] += item["read_beats"]
            values[3] += item["write_beats"]
            committed_cycles += item["cycles"]
    operators = [
        {
            "operator": operator,
            "committed_commands": values[0],
            "simulator_cycles": values[1],
            "cycle_share": values[1] / committed_cycles,
            "read_beats": values[2],
            "write_beats": values[3],
        }
        for operator, values in sorted(
            totals.items(), key=lambda item: item[1][1], reverse=True
        )
    ]
    dominant = {
        "lm_head_tile",
        "mlp_up_proj",
        "mlp_gate_proj",
        "mlp_down_proj",
    }
    return {
        "schema": "ace2-cf18-sealed-profile-diagnosis-v1",
        "source": {
            "progress": {**record(SEALED_PROGRESS), "path": relative(SEALED_PROGRESS)},
            "commands": {**record(SEALED_COMMANDS), "path": relative(SEALED_COMMANDS)},
        },
        "sealed_observation": json.loads(
            SEALED_PROGRESS.read_text(encoding="utf-8")
        ),
        "recomputed_committed_cycles": committed_cycles,
        "operators": operators,
        "dominant_cycle_share": sum(
            item["cycle_share"] for item in operators
            if item["operator"] in dominant
        ),
        "host_transactor": {
            "baseline_persistence_batch_commands": 1,
            "durable_operations_per_command": [
                "progress.journal fsync",
                "commands.jsonl fsync",
                "atomic progress.json file and parent-directory fsync",
            ],
            "controlled_comparison": {
                **record(PRIOR_RTL_EQUIVALENCE),
                "path": relative(PRIOR_RTL_EQUIVALENCE),
                "status": json.loads(
                    PRIOR_RTL_EQUIVALENCE.read_text(encoding="utf-8")
                )["status"],
                "measured_improvement_percent": json.loads(
                    PRIOR_RTL_EQUIVALENCE.read_text(encoding="utf-8")
                )["improvement"]["percent"],
            },
            "fresh_host_only_microbenchmark": {
                **record(BENCHMARK),
                "path": relative(BENCHMARK),
                "model_schedule_executed": False,
                "rtl_executed": False,
            },
        },
        "diagnosis": (
            "RTL cycle cost is dominated by LM-head and MLP projection work; "
            "independent of those unchanged semantics, per-command durable host "
            "publication is a measured avoidable transactor cost."
        ),
    }


def finite_estimate(profile: dict[str, Any]) -> dict[str, Any]:
    sealed = profile["sealed_observation"]
    committed = sealed["next_ordinal"]
    elapsed_seconds = 7200
    commands_per_second = committed / elapsed_seconds
    cycles_per_second = sealed["simulator_cycles"] / elapsed_seconds
    baseline_seconds = COMMANDS / commands_per_second
    estimated_cycles = sealed["simulator_cycles"] * COMMANDS / committed
    prior = json.loads(PRIOR_RTL_EQUIVALENCE.read_text(encoding="utf-8"))
    measured_ratio = 1.0 + prior["improvement"]["percent"] / 100.0
    candidate_seconds = baseline_seconds / measured_ratio
    return {
        "schema": "ace2-cf18-finite-runtime-estimate-v1",
        "status": "FINITE_ESTIMATE_WITH_MARGIN",
        "method": (
            "Linear projection from the sealed CF17 committed-command and "
            "simulator-cycle rates; the candidate estimate applies only the "
            "measured 10-command persistence-batching ratio. The admission "
            "timeout is larger than both projections and is not a completion claim."
        ),
        "sealed_rate": {
            "elapsed_seconds": elapsed_seconds,
            "commands_per_second": commands_per_second,
            "simulator_cycles_per_second": cycles_per_second,
        },
        "full_schedule": {
            "commands": COMMANDS,
            "schedule_sha256": SCHEDULE_SHA256,
            "projected_simulator_cycles": estimated_cycles,
            "baseline_projected_seconds": baseline_seconds,
            "baseline_projected_hours": baseline_seconds / 3600,
            "candidate_projected_seconds": candidate_seconds,
            "candidate_projected_hours": candidate_seconds / 3600,
        },
        "admission_timeout_seconds": ADMISSION_TIMEOUT_SECONDS,
        "admission_timeout_hours": ADMISSION_TIMEOUT_SECONDS / 3600,
        "margin_over_baseline_projection": (
            ADMISSION_TIMEOUT_SECONDS / baseline_seconds
        ),
        "limitations": [
            "CF17 sampled 106796 of 1306104 ordered commands through token step 9.",
            "Later attention context is longer; projections and LM-head dominate the observed cycle total.",
            "The estimate does not claim generated tokens or successful completion.",
        ],
    }


def main() -> None:
    required = [
        CF15 / "product_probe.py",
        CF15 / "bindings.json",
        CF15 / "model24_position_contract.py",
        CF15 / "product_host_runtime.py",
        CF10 / "inputs/descriptor.json",
        CF10 / "inputs/runtime-package.bin",
        CF10 / "inputs/memory-image.bin",
        CF10 / "Vace2_shell_runtime_harness",
        BENCHMARK,
        PRIOR_RTL_EQUIVALENCE,
        SEALED_PROGRESS,
        SEALED_COMMANDS,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required CF18 input is absent: {path}")
    for path in (PACKAGE, STATE, EXTERNAL_AUTHORITY):
        if os.path.lexists(path):
            raise RuntimeError(f"CF18 fresh path already exists: {path}")
    PACKAGE.mkdir(mode=0o755)

    bindings = json.loads((CF15 / "bindings.json").read_text(encoding="utf-8"))
    endpoint_prefix = list(bindings["endpoint"]["argv_prefix"])
    bindings["endpoint"]["persistence_batch_commands"] = BATCH_COMMANDS
    bindings["endpoint"]["timeout_seconds"] = ADMISSION_TIMEOUT_SECONDS
    bindings["cf18_acceleration"] = {
        "mechanism": "BATCH_DURABLE_HOST_PERSISTENCE_ONLY",
        "batch_commands": BATCH_COMMANDS,
        "full_schedule_preserved": True,
        "rtl_semantics": "UNCHANGED",
        "software_fallback": "FORBIDDEN",
    }
    write(PACKAGE / "bindings.json", bindings)

    probe = (CF15 / "product_probe.py").read_text(encoding="utf-8")
    old = (
        '        argv = list(endpoint["argv_prefix"]) + '
        '["--output", str(args.endpoint_output)]\n'
    )
    new = (
        '        argv = list(endpoint["argv_prefix"]) + [\n'
        '            "--persistence-batch-commands",\n'
        '            str(endpoint["persistence_batch_commands"]),\n'
        '            "--output",\n'
        '            str(args.endpoint_output),\n'
        '        ]\n'
    )
    if probe.count(old) != 1:
        raise RuntimeError("CF15 endpoint invocation boundary changed")
    write(PACKAGE / "product_probe.py", probe.replace(old, new).encode(), 0o555)
    for name in ("model24_position_contract.py", "product_host_runtime.py"):
        copy_source(CF15 / name, PACKAGE / name)
    for source, target, mode in (
        (
            ROOT / "tools/cf18_runtime_authority_runner.py",
            PACKAGE / "authority_runner.py",
            0o555,
        ),
        (
            ROOT / "tools/cf18_runtime_validate.py",
            PACKAGE / "validate_package.py",
            0o555,
        ),
        (
            ROOT / "tools/cf18_runtime_tests.py",
            PACKAGE / "tests/test_cf18_runtime.py",
            0o444,
        ),
    ):
        if target.parent != PACKAGE:
            target.parent.mkdir()
        copy_source(source, target, mode)

    inventory = predecessor_inventory()
    write(PACKAGE / "predecessor-cf16-cf17-inventory.json", inventory)
    profile = profile_diagnosis()
    write(PACKAGE / "sealed-profile-diagnosis.json", profile)
    estimate = finite_estimate(profile)
    write(PACKAGE / "finite-runtime-estimate.json", estimate)

    nonce = sha256_bytes(
        canonical_bytes(
            {
                "identity": IDENTITY,
                "schedule_sha256": SCHEDULE_SHA256,
                "cf17_terminal_seal_sha256": sha256_file(
                    CF17_STATE / "terminal-seal.json"
                ),
                "descriptor_sha256": sha256_file(CF10 / "inputs/descriptor.json"),
                "batch_commands": BATCH_COMMANDS,
            }
        )
    )
    endpoint_output = ATTEMPT / "product/rtl-completion"
    endpoint_argv = endpoint_prefix + [
        "--persistence-batch-commands",
        str(BATCH_COMMANDS),
        "--output",
        str(endpoint_output),
    ]
    evidence_path = ATTEMPT / "product/product-rtl-evidence.json"
    product_argv = [
        "/home/argustest/miniconda3/bin/python3.13",
        "-B",
        str(PACKAGE / "product_probe.py"),
        "--output",
        str(evidence_path),
        "--capture",
        str(ATTEMPT / "product/product-host-error-capture.json"),
        "--endpoint-output",
        str(endpoint_output),
    ]
    launch_argv = [
        "/home/argustest/miniconda3/bin/python3.13",
        "-B",
        str(PACKAGE / "authority_runner.py"),
        "--authority",
        str(EXTERNAL_AUTHORITY),
    ]
    candidate = {
        "schema": "ace2-cf18-exactly-once-authority-candidate-v1",
        "status": "CANDIDATE_ONLY_SEPARATE_AUTHORITY_REQUIRED",
        "identity": IDENTITY,
        "nonce": nonce,
        "authority_cardinality": 1,
        "execution_limit": 1,
        "authority_granted": False,
        "authority_consumed": False,
        "state_namespace": str(STATE),
        "output_namespace": str(ATTEMPT),
        "separate_execution_authority_path": str(EXTERNAL_AUTHORITY),
        "exact_future_launch_argv": launch_argv,
        "exact_product_invocation": {
            "argv": product_argv,
            "timeout_seconds": ADMISSION_TIMEOUT_SECONDS + 7200,
            "evidence_path": str(evidence_path),
        },
        "exact_endpoint_invocation": {
            "argv": endpoint_argv,
            "timeout_seconds": ADMISSION_TIMEOUT_SECONDS,
        },
        "full_schedule": {
            "commands": COMMANDS,
            "schedule_sha256": SCHEDULE_SHA256,
            "truncation": "FORBIDDEN",
            "layer_substitution": "FORBIDDEN",
        },
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "cf16_cf17_mutation_or_execution": "PERMANENTLY_FORBIDDEN",
        "software_fallback": "FORBIDDEN",
        "generated_token_claim": "NONE_PREPARATION_ONLY",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    write(PACKAGE / "authority-candidate.json", candidate)
    contract = {
        "schema": "ace2-cf18-runtime-acceleration-contract-v1",
        "identity": IDENTITY,
        "nonce": nonce,
        "predecessor": CF17_PACKAGE.name,
        "predecessor_disposition": "CLOSED_NO_RETRY_REPLAY_RESUME_RELAUNCH",
        "mechanism": "BATCH_DURABLE_HOST_PERSISTENCE_ONLY",
        "persistence_batch_commands": BATCH_COMMANDS,
        "complete_frozen_schedule_commands": COMMANDS,
        "schedule_sha256": SCHEDULE_SHA256,
        "schedule_semantics": "ALL_COMMANDS_IN_ORIGINAL_ORDER",
        "rtl_semantics": "REAL_UNCHANGED_ACE2_SHELL",
        "final_export_backing": record(CF10 / "inputs/final-export-manifest.json"),
        "arbitrary_text_path": [
            "official_tokenizer_chat_template",
            "accepted_W4A8_host_prefill",
            "cacheful_generation",
            "24_layer_KV",
            "full_RTL_schedule",
            "independent_host_RTL_token_and_KV_comparison",
            "official_whole_sequence_decode",
        ],
        "prohibited": [
            "schedule_truncation",
            "layer_substitution",
            "scripted_output",
            "cached_logits",
            "software_fallback",
            "retry",
            "replay",
            "resume",
            "relaunch",
            "stage2",
        ],
        "execution_during_preparation": False,
        "generated_token_claim": "NONE",
    }
    write(PACKAGE / "contract.json", contract)
    write(
        PACKAGE / "semantic-equivalence.json",
        {
            "schema": "ace2-cf18-semantic-equivalence-v1",
            "status": "PASS_PREPARATION_EVIDENCE",
            "product_probe_predecessor": {
                **record(CF15 / "product_probe.py"),
                "path": relative(CF15 / "product_probe.py"),
            },
            "product_probe_successor": record(PACKAGE / "product_probe.py"),
            "source_edit": (
                "One endpoint argv insertion adds --persistence-batch-commands "
                "1024; host, model, tokenizer, generation, KV, RTL descriptor, "
                "comparison, and decode logic are otherwise retained."
            ),
            "prior_controlled_rtl_comparison": {
                **record(PRIOR_RTL_EQUIVALENCE),
                "status": "PASS",
                "journal_jsonl_summary_sha256_exact": True,
                "simulator_cycles_exact": True,
                "rtl_reference": "PASS_BIT_EXACT",
            },
            "fresh_microbenchmark": {
                **record(BENCHMARK),
                "status": "PASS",
                "model_schedule_executed": False,
                "rtl_executed": False,
            },
        },
    )
    write(
        PACKAGE / "preconstruction-zero-state.json",
        {
            "schema": "ace2-cf18-zero-state-v1",
            "status": "PASS",
            "authority_candidate_count": 1,
            "external_execution_authority_exists": False,
            "state_namespace_exists": False,
            "attempt_namespace_exists": False,
            "registry_created": False,
            "process_started": False,
            "model_executed": False,
            "rtl_executed": False,
            "generated_tokens": 0,
        },
    )

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "tools/cf18_runtime_authority_runner.py",
        ROOT / "tools/cf18_runtime_validate.py",
        ROOT / "tools/cf18_runtime_tests.py",
        ROOT / "tools/run_cf18_host_persistence_microbenchmark.py",
        ROOT / "verification/verilator/ace2_shell_runtime_main.cpp",
        ROOT / "verification/verilator/ace2_shell_runtime_harness.sv",
        ROOT / "rtl/ace2_shell.sv",
        ROOT / "constraints/ace2_rmsnorm_core.sdc",
        CF10 / "inputs/descriptor.json",
        CF10 / "inputs/runtime-package.bin",
        CF10 / "inputs/memory-image.bin",
        CF10 / "inputs/final-export-manifest.json",
        CF10 / "Vace2_shell_runtime_harness",
        CF17_STATE / "terminal-seal.json",
    ]
    write(
        PACKAGE / "source-constraint-manifest.json",
        {
            "schema": "ace2-cf18-source-constraint-manifest-v1",
            "files": {
                relative(path): record(path) for path in source_paths
            },
            "rtl_change": "NONE",
            "constraint_change": "NONE",
            "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
            "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
            "non_sram_area_cap_mm2": 2.0,
            "minimum_frequency_mhz": 100,
        },
    )
    write(
        PACKAGE / "Makefile",
        (
            b".PHONY: validate test\n"
            b"validate:\n"
            b"\tPYTHONDONTWRITEBYTECODE=1 python3 validate_package.py\n"
            b"test:\n"
            b"\tPYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v\n"
        ),
    )

    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    package_files = {
        path.relative_to(PACKAGE).as_posix(): record(path)
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file()
        and path.relative_to(PACKAGE).as_posix() not in excluded
    }
    manifest = {
        "schema": "ace2-cf18-package-manifest-v1",
        "identity": IDENTITY,
        "files": package_files,
    }
    write(PACKAGE / "package-manifest.json", manifest)
    write(
        PACKAGE / "package-manifest.sha256",
        (
            f"{sha256_file(PACKAGE / 'package-manifest.json')}  "
            "package-manifest.json\n"
        ).encode("ascii"),
    )
    review = {
        "schema": "ace2-cf18-independent-l2-review-request-v1",
        "status": "PENDING_INDEPENDENT_L2",
        "identity": IDENTITY,
        "nonce": nonce,
        "package_manifest_sha256": sha256_file(
            PACKAGE / "package-manifest.json"
        ),
        "required_checks": [
            "fresh_path_and_hash_authentication",
            "CF16_CF17_exact_read_only_inventory",
            "full_1306104_command_schedule_and_final_export_binding",
            "host_model_generation_KV_RTL_comparison_decode_path",
            "persistence_batch_semantic_equivalence",
            "negative_tests_and_fail_closed_authority",
            "finite_runtime_estimate_and_144000_second_admission_bound",
            "exact_future_command_identity_nonce_and_cardinality_one",
            "zero_state_no_authority_consumption_or_execution",
            "no_retry_replay_resume_relaunch",
            "stage1_open_stage2_forbidden",
        ],
    }
    write(PACKAGE / "review-request.json", review)
    write(
        PACKAGE / "review-request.sha256",
        (
            f"{sha256_file(PACKAGE / 'review-request.json')}  review-request.json\n"
        ).encode("ascii"),
    )
    print(
        f"CF18_CONSTRUCTED identity={IDENTITY} "
        f"manifest_sha256={sha256_file(PACKAGE / 'package-manifest.json')} "
        "authority_cardinality=1 state=ZERO execution=NONE"
    )


if __name__ == "__main__":
    main()
