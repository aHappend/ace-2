#!/usr/bin/env python3
"""Freeze and execute the bounded layer-23 V RTL-sidecar W4A8 hybrid once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import peft
import numpy as np
import safetensors
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

IMPLEMENTATION_FILE = Path(__file__).resolve()
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import ace2_checkpoint176_hidden_state_localizer as localizer
from tools import ace2_exact_once_runtime_supervisor as supervisor
from tools import rtl_arbitrary_text_generation_backend as backend
from tools import run_rtl_arbitrary_text_generation as generation_runner
from tools import run_stage1_layer23_v_rank1_integer_candidate as candidate

MISSION_ID = "stage1rank1hybrid03"
CLASSIFICATION = "bounded_nonofficial_autoregressive_rtl_sidecar_hybrid"
ATTEMPT_ID = "nonofficial-hybrid-0003"
STEPS = 4
TARGET_LAYER = 23
EXPECTED_PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
EXPECTED_PYTHON_SHA256 = (
    "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
)
CHILD_HOME = "/home/argustest"
CHILD_PYTHONPATH = "/home/argustest/ace-2/.venv/lib/python3.13/site-packages"
CHILD_PATH = (
    "/home/argustest/argustest2/.venv/bin:/home/argustest/bin:"
    "/home/argustest/.bun/bin:/home/argustest/.krew/bin:"
    "/home/argustest/.local/bin:/home/argustest/bin:/home/argustest/.krew/bin:"
    "/home/argustest/.vscode-server/data/User/globalStorage/"
    "github.copilot-chat/debugCommand:"
    "/home/argustest/.vscode-server/data/User/globalStorage/"
    "github.copilot-chat/copilotCli:"
    "/home/argustest/.vscode-server/cli/servers/Stable-"
    "a5b500951314efd502d07465bd138dfbd714a960/server/bin/remote-cli:"
    "/home/argustest/.elan/bin:/home/argustest/.local/bin:"
    "/home/argustest/bin:/home/argustest/Argus_BoXiuLi8T_MOF/code/"
    "Argus_mofgen_uv:/home/argustest/bin:/home/argustest/.bun/bin:"
    "/home/argustest/.krew/bin:/home/argustest/.local/bin:"
    "/home/argustest/bin:/home/argustest/.krew/bin:"
    "/home/argustest/.nvm/versions/node/v22.23.1/bin:"
    "/home/argustest/miniconda3/bin:/home/argustest/miniconda3/condabin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
    "/usr/games:/usr/local/games:/snap/bin"
)
INTEGRATION_RESULT_SHA256 = (
    "36496c88ea1192a5d22f5b9e3593ddc556ceb4f229bdfc6c3444191c555e23a1"
)
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / ATTEMPT_ID
)
PREFLIGHT = OUTPUT / "preflight.json"
FREEZE = OUTPUT / "freeze.json"
PACKAGE = OUTPUT / "execution-package.json"
AUTHORITY = OUTPUT / "execution-authority.json"
LIVE = OUTPUT / "live"
RESULT = LIVE / "result.json"
FAILURE = LIVE / "failure.json"
SUMS = LIVE / "SHA256SUMS"
STATE = OUTPUT / "supervisor-state"
STDOUT = OUTPUT / "execution.stdout.log"
STDERR = OUTPUT / "execution.stderr.log"
TERMINAL = OUTPUT / "terminal-record.json"
DEFAULT_PROMPTS = (
    ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0003.json"
)
TEMPLATE_GATE = (
    ROOT
    / "build/stage1-layer23-v-rank1-hybrid-v1/private"
    / "package-template-invariant-0003.json"
)
DEPENDENCY_PROBE = (
    ROOT
    / "build/stage1-layer23-v-rank1-hybrid-v1/private"
    / "dependency-probe-0003.json"
)
PREDECESSOR_TERMINAL_SEALS = {
    "nonofficial-hybrid-0001": {
        "sha256": "c97541d51a481812fbf41cb8fccdb035fe560ec97db179ca2a2aa4d6afb2780e",
        "process_start_count": 0,
        "process_started": False,
    },
    "nonofficial-hybrid-0002": {
        "sha256": "c8facc07691df72aab7fed0c6a6c51c15d6bc2c6fa222e622c3ee4c088bf4573",
        "process_start_count": 0,
        "process_started": False,
    },
}
ADAPTER = (
    ROOT
    / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4"
    / "attempt-0001/checkpoints/checkpoint-176"
    / "adapter_model.safetensors"
)
INTEGRATION_RESULT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-integration-v1"
    / "latest/RESULT.json"
)
INTEGRATION_SUMS = INTEGRATION_RESULT.parent / "SHA256SUMS"
TB = ROOT / "verification/tb/ace2_layer23_v_rank1_hybrid_shell_tb.sv"
TB_TOP = "ace2_layer23_v_rank1_hybrid_shell_tb"
RTL_SOURCES = (
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_dynamic_scale32_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_dynamic_rope_head_core.sv",
    "rtl/ace2_fixed_q7_rope_score_core.sv",
    "rtl/ace2_relative_rope_score_fusion_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv",
    "rtl/ace2_shell.sv",
)
PASS_PATTERN = re.compile(
    r"ACE2_LAYER23_V_RANK1_HYBRID_SHELL_PASS "
    r"outputs=128 descriptor_accept=1 disabled_control=1 rank_exact=1 "
    r"saturation=(?P<saturation>[01]) overflow=(?P<overflow>[01]) "
    r"cycles=(?P<cycles>\d+) reads=(?P<reads>\d+) writes=(?P<writes>\d+)"
)
V_BEAT_PATTERN = re.compile(
    r"^ACE2_HYBRID_V_BEAT beat=(?P<beat>\d+) "
    r"rank_acc=(?P<rank_acc>-?\d+) "
    r"rank_rounded=(?P<rank_rounded>-?\d+) "
    r"rank_s8=(?P<rank_s8>-?\d+) "
    r"baseline=(?P<baseline>[0-9a-fA-F]{32}) "
    r"corrected=(?P<corrected>[0-9a-fA-F]{32})$",
    re.MULTILINE,
)


class HybridError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise HybridError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external:{resolved.name}>"


def file_record(path: Path, *, expose_path: bool = True) -> dict[str, Any]:
    require(path.is_file(), f"required file is absent: {public_path(path)}")
    record: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if expose_path:
        record["path"] = public_path(path)
    return record


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {public_path(path)}")
    return value


def child_environment() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HOME": CHILD_HOME,
        "LC_ALL": "C",
        "PATH": CHILD_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": CHILD_PYTHONPATH,
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def verify_exact_interpreter() -> None:
    interpreter = Path(sys.executable).resolve()
    require(interpreter == EXPECTED_PYTHON, "Python interpreter binding changed")
    require(
        sha256_file(interpreter) == EXPECTED_PYTHON_SHA256,
        "Python interpreter SHA-256 changed",
    )


def verify_predecessor_seals() -> None:
    for attempt_id, expected in PREDECESSOR_TERMINAL_SEALS.items():
        terminal = OUTPUT.parent / attempt_id / "terminal-record.json"
        require(terminal.is_file(), f"predecessor terminal absent: {attempt_id}")
        require(
            sha256_file(terminal) == expected["sha256"],
            f"predecessor terminal seal changed: {attempt_id}",
        )
        record = read_json(terminal)
        require(
            record.get("process_start_count") == expected["process_start_count"]
            and record.get("process_started") is expected["process_started"],
            f"predecessor process-start state changed: {attempt_id}",
        )


def dependency_probe() -> int:
    require(not DEPENDENCY_PROBE.exists(), "dependency probe already exists")
    require(not TEMPLATE_GATE.exists(), "package-template invariant already exists")
    require(not OUTPUT.exists(), f"{ATTEMPT_ID} output namespace already exists")
    verify_predecessor_seals()
    verify_exact_interpreter()
    observed_environment = dict(os.environ)
    require(
        observed_environment == child_environment(),
        "dependency probe environment differs from the complete frozen child environment",
    )
    project_modules = {
        "tools.ace2_checkpoint176_hidden_state_localizer": localizer,
        "tools.ace2_exact_once_runtime_supervisor": supervisor,
        "tools.rtl_arbitrary_text_generation_backend": backend,
        "tools.run_rtl_arbitrary_text_generation": generation_runner,
        "tools.run_stage1_layer23_v_rank1_integer_candidate": candidate,
    }
    imports = {
        "peft": str(peft.__version__),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "numpy": str(np.__version__),
        "safetensors": str(safetensors.__version__),
    }
    record = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "attempt_identity": ATTEMPT_ID,
        "status": "PASS_NONCONSUMING_EXACT_CHILD_DEPENDENCY_PROBE",
        "created_at_utc": utc_now(),
        "interpreter": file_record(EXPECTED_PYTHON),
        "environment": child_environment(),
        "environment_sha256": sha256_bytes(canonical_bytes(child_environment())),
        "imports": imports,
        "project_module_chain": {
            name: file_record(Path(module.__file__).resolve())
            for name, module in project_modules.items()
        },
        "sys_path": list(sys.path),
        "sys_prefix": sys.prefix,
        "checks": {
            "process_not_started": True,
            "output_namespace_absent": True,
            "exact_interpreter_path": True,
            "exact_interpreter_sha256": True,
            "complete_child_environment_exact": True,
            "required_third_party_imports_succeeded": True,
            "runner_project_module_chain_imported": True,
        },
    }
    write_json(DEPENDENCY_PROBE, record)
    print(
        "ACE2_HYBRID_DEPENDENCY_PROBE_PASS "
        f"sha256={sha256_file(DEPENDENCY_PROBE)} process_start_count=0",
        flush=True,
    )
    return 0


def verify_dependency_probe() -> dict[str, Any]:
    require(DEPENDENCY_PROBE.is_file(), "exact child dependency probe is absent")
    record = read_json(DEPENDENCY_PROBE)
    require(
        record.get("mission_id") == MISSION_ID
        and record.get("attempt_identity") == ATTEMPT_ID
        and record.get("status") == "PASS_NONCONSUMING_EXACT_CHILD_DEPENDENCY_PROBE",
        "dependency probe identity/status changed",
    )
    require(
        record["interpreter"]["sha256"] == EXPECTED_PYTHON_SHA256,
        "probed interpreter SHA-256 changed",
    )
    require(record["environment"] == child_environment(), "probed child environment changed")
    require(
        all(value is True for value in record["checks"].values()),
        "dependency probe predicate changed",
    )
    return record


def require_resolved_regular_file(path: Path, label: str) -> None:
    resolved = path.resolve(strict=True)
    require(path == resolved, f"{label} binding is not resolved")
    require(stat.S_ISREG(os.lstat(resolved).st_mode), f"{label} binding is not a regular file")


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def tool_identity(argv: list[str]) -> dict[str, Any]:
    executable = Path(
        subprocess.run(
            ["which", argv[0]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    require(completed.returncode == 0, f"tool identity command failed: {argv[0]}")
    return {
        "executable": file_record(executable),
        "version_sha256": sha256_bytes(
            (completed.stdout + completed.stderr).encode("utf-8")
        ),
        "version_first_line": (
            (completed.stdout + completed.stderr).splitlines() or [""]
        )[0],
    }


def load_prompts(path: Path) -> list[dict[str, str]]:
    value = read_json(path)
    prompts = value.get("prompts")
    require(isinstance(prompts, list) and len(prompts) == 2, "exactly two prompts required")
    normalized: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for item in prompts:
        require(isinstance(item, dict) and set(item) == {"id", "text"}, "prompt schema changed")
        prompt_id = item["id"]
        text = item["text"]
        require(
            isinstance(prompt_id, str)
            and prompt_id
            and isinstance(text, str)
            and bool(text.strip()),
            "prompt id/text must be nonempty strings",
        )
        require(prompt_id not in identifiers, "prompt ids must be unique")
        identifiers.add(prompt_id)
        normalized.append({"id": prompt_id, "text": text})
    return normalized


def prompt_records(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": item["id"],
            "utf8_bytes": len(item["text"].encode("utf-8")),
            "sha256": sha256_bytes(item["text"].encode("utf-8")),
        }
        for item in load_prompts(path)
    ]


def verify_unseen_prompt_hashes(records: list[dict[str, Any]]) -> dict[str, Any]:
    roots = [
        ROOT / "evidence",
        ROOT / "diagnosis",
        ROOT / "research",
    ]
    scans = []
    for record in records:
        completed = subprocess.run(
            [
                "rg",
                "--fixed-strings",
                "--files-with-matches",
                record["sha256"],
                *[str(path) for path in roots if path.exists()],
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        require(completed.returncode in (0, 1), "prompt-history hash scan failed")
        matches = [line for line in completed.stdout.splitlines() if line]
        require(not matches, f"prompt hash was already present: {record['prompt_id']}")
        scans.append(
            {
                "prompt_id": record["prompt_id"],
                "sha256": record["sha256"],
                "prior_hash_matches": 0,
            }
        )
    return {"method": "prior artifact fixed-string SHA-256 scan", "records": scans}


def resolve_runtime_inputs() -> tuple[Path, Path, Path, Path]:
    snapshot = generation_runner.resolve_snapshot().resolve()
    model = (snapshot / "model.safetensors").resolve(strict=True)
    tokenizer_json = (snapshot / "tokenizer.json").resolve(strict=True)
    tokenizer_config = (snapshot / "tokenizer_config.json").resolve(strict=True)
    for path in (model, ADAPTER, tokenizer_json, tokenizer_config):
        require(path.is_file(), f"runtime input absent: {public_path(path)}")
    return snapshot, model, tokenizer_json, tokenizer_config


def compile_argv(binary: Path) -> list[str]:
    return [
        "iverilog",
        "-g2012",
        "-Wall",
        "-Irtl",
        "-s",
        TB_TOP,
        "-o",
        str(binary),
        *[str(ROOT / path) for path in RTL_SOURCES],
        str(TB),
    ]


def vvp_template(binary: Path) -> list[str]:
    return [
        "vvp",
        str(binary),
        "+ACTIVATION={activation}",
        "+V_WEIGHT={v_weight}",
        "+V_META={v_meta}",
        "+BASELINE={baseline}",
        "+CORRECTED={corrected}",
        "+RANK={rank}",
        "+RANK_S8={rank_s8}",
        "+SATURATION={saturation}",
    ]


def source_records(
    model: Path, tokenizer_json: Path, tokenizer_config: Path, prompts: Path
) -> dict[str, Any]:
    paths = [ROOT / item for item in RTL_SOURCES]
    paths.extend(
        [
            Path(__file__).resolve(),
            IMPLEMENTATION_FILE,
            TB,
            model,
            ADAPTER,
            tokenizer_json,
            tokenizer_config,
            candidate.FREEZE,
            INTEGRATION_RESULT,
            INTEGRATION_SUMS,
            prompts,
            DEPENDENCY_PROBE,
        ]
    )
    freeze = read_json(candidate.FREEZE)
    for record in freeze["artifacts"].values():
        paths.append(ROOT / record["path"])
    unique = {str(path.resolve()): path.resolve() for path in paths}
    return {
        public_path(path): file_record(path)
        for path in sorted(unique.values(), key=lambda item: str(item))
    }


def execution_argv(prompts_path: Path, freeze_sha256: str) -> list[str]:
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", freeze_sha256)),
        "freeze binding must be a lowercase SHA-256",
    )
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--execute",
        "--prompts",
        str(prompts_path.resolve(strict=True)),
        "--freeze-sha256",
        freeze_sha256,
    ]


def execution_bound_paths(
    prompts_path: Path,
    model: Path,
    tokenizer_json: Path,
    tokenizer_config: Path,
    *,
    include_frozen_stage_files: bool,
) -> list[Path]:
    paths = [
        Path(sys.executable).resolve(),
        Path(__file__).resolve(),
        IMPLEMENTATION_FILE,
        prompts_path.resolve(strict=True),
        *[ROOT / item for item in RTL_SOURCES],
        TB,
        model,
        ADAPTER,
        tokenizer_json,
        tokenizer_config,
        candidate.FREEZE,
        INTEGRATION_RESULT,
        INTEGRATION_SUMS,
        DEPENDENCY_PROBE,
    ]
    if include_frozen_stage_files:
        paths.extend([TEMPLATE_GATE, PREFLIGHT, FREEZE])
    return sorted(
        {path.resolve(strict=True) for path in paths},
        key=lambda item: str(item),
    )


def package_bindings(
    prompts_path: Path,
    freeze_sha256: str,
    required_absent_outputs: list[str],
    model: Path,
    tokenizer_json: Path,
    tokenizer_config: Path,
    *,
    include_frozen_stage_files: bool,
) -> dict[str, Any]:
    argv = execution_argv(prompts_path, freeze_sha256)
    return {
        "bound_files": [
            supervisor.file_binding(path)
            for path in execution_bound_paths(
                prompts_path,
                model,
                tokenizer_json,
                tokenizer_config,
                include_frozen_stage_files=include_frozen_stage_files,
            )
        ],
        "source_tree": supervisor.source_tree_binding(ROOT / "rtl"),
        "cwd": str(ROOT),
        "argv": argv,
        "argv_sha256": supervisor.canonical_value_sha256(argv),
        "environment": child_environment(),
        "required_absent_outputs": required_absent_outputs,
    }


def intended_runtime_paths() -> tuple[list[Path], dict[str, str]]:
    outputs = [LIVE, STDOUT, STDERR, TERMINAL]
    runtime = {
        "state_dir": str(STATE),
        "stdout_path": str(STDOUT),
        "stderr_path": str(STDERR),
        "terminal_record_path": str(TERMINAL),
    }
    return outputs, runtime


def validate_no_execution() -> None:
    forbidden = [LIVE, STATE, STDOUT, STDERR, TERMINAL]
    require(not any(path.exists() for path in forbidden), "qualifying execution already exists")


def preflight(prompts_path: Path) -> int:
    verify_exact_interpreter()
    probe = verify_dependency_probe()
    require(TEMPLATE_GATE.is_file(), "package-template invariant gate is absent")
    gate = read_json(TEMPLATE_GATE)
    require(
        gate.get("status") == "PASS_NONCONSUMING_PACKAGE_TEMPLATE_INVARIANT"
        and gate.get("mission_id") == MISSION_ID
        and gate.get("attempt_identity") == ATTEMPT_ID,
        "package-template invariant gate identity/status changed",
    )
    require(
        all(value is True for value in gate["checks"].values()),
        "package-template invariant predicate changed",
    )
    require(not PREFLIGHT.exists(), "preflight artifact already exists")
    require(not FREEZE.exists(), "freeze already exists")
    require(not OUTPUT.exists(), f"{ATTEMPT_ID} output namespace already exists before preflight")
    verify_predecessor_seals()
    validate_no_execution()
    prompts = prompt_records(prompts_path)
    require(prompts == gate["prompts"], "prompts changed after package-template gate")
    unseen = verify_unseen_prompt_hashes(prompts)
    snapshot, model, tokenizer_json, tokenizer_config = resolve_runtime_inputs()
    require(
        sha256_file(INTEGRATION_RESULT) == INTEGRATION_RESULT_SHA256,
        "accepted integration RESULT hash changed",
    )
    candidate.validate_freeze(read_json(candidate.FREEZE))
    record = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "PASS_NONCONSUMING_HYBRID_PREFLIGHT",
        "classification": CLASSIFICATION,
        "created_at_utc": utc_now(),
        "prior_qualifying_execution_exists": False,
        "prompts": prompts,
        "unseen_check": unseen,
        "runtime": {
            "snapshot_revision": snapshot.name,
            "model": file_record(model),
            "adapter": file_record(ADAPTER),
            "tokenizer_json": file_record(tokenizer_json),
            "tokenizer_config": file_record(tokenizer_config),
            "dependency_probe": file_record(DEPENDENCY_PROBE),
            "environment_sha256": probe["environment_sha256"],
        },
        "accepted_integration": file_record(INTEGRATION_RESULT),
        "tools": {
            "python": file_record(Path(sys.executable).resolve()),
            "iverilog": tool_identity(["iverilog", "-V"]),
            "vvp": tool_identity(["vvp", "-V"]),
        },
        "checks": {
            "model_executed": False,
            "simulator_executed": False,
            "icarus_compile_executed": False,
            "output_namespace_absent": True,
            "descriptor_binding_static": True,
            "exactly_once_supervisor_available": True,
            "package_template_invariant_passed": True,
            "predecessor_terminal_seals_immutable": True,
        },
    }
    write_json(PREFLIGHT, record)
    print(
        f"ACE2_HYBRID_PREFLIGHT_PASS sha256={sha256_file(PREFLIGHT)}",
        flush=True,
    )
    return 0


def freeze(prompts_path: Path) -> int:
    verify_exact_interpreter()
    probe = verify_dependency_probe()
    require(PREFLIGHT.is_file(), "preflight artifact is absent")
    require(not FREEZE.exists(), "freeze already exists")
    require(not PACKAGE.exists() and not AUTHORITY.exists(), "execution package already exists")
    require(TEMPLATE_GATE.is_file(), "package-template invariant gate is absent")
    verify_predecessor_seals()
    validate_no_execution()
    preflight_record = read_json(PREFLIGHT)
    require(
        preflight_record["status"] == "PASS_NONCONSUMING_HYBRID_PREFLIGHT",
        "preflight status changed",
    )
    prompts = prompt_records(prompts_path)
    require(prompts == preflight_record["prompts"], "prompt hashes changed after preflight")
    snapshot, model, tokenizer_json, tokenizer_config = resolve_runtime_inputs()
    tokenizer_config_value = read_json(tokenizer_config)
    chat_template = tokenizer_config_value.get("chat_template")
    require(isinstance(chat_template, str) and chat_template, "chat template is absent")
    binary = LIVE / "sim/ace2_layer23_v_rank1_hybrid_shell.vvp"
    sources = source_records(model, tokenizer_json, tokenizer_config, prompts_path)
    frozen = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "classification": CLASSIFICATION,
        "frozen_at_utc": utc_now(),
        "attempt_identity": ATTEMPT_ID,
        "claim_boundary": "bounded nonofficial hybrid evidence; not Stage-1 completion",
        "prompts": prompts,
        "private_prompt_artifact": file_record(prompts_path, expose_path=False),
        "exact_child_dependency_probe": file_record(DEPENDENCY_PROBE),
        "exact_child_environment": {
            "values": child_environment(),
            "sha256": probe["environment_sha256"],
            "sys_path": probe["sys_path"],
            "sys_prefix": probe["sys_prefix"],
        },
        "tokenizer": {
            "snapshot_revision": snapshot.name,
            "tokenizer_json": file_record(tokenizer_json),
            "tokenizer_config": file_record(tokenizer_config),
            "chat_template_utf8_sha256": sha256_bytes(chat_template.encode("utf-8")),
            "apply_policy": "single user message plus tokenizer generation prompt",
        },
        "decoding": {
            "policy": "greedy argmax over all 151936 deterministic W4A8 logits; lowest token id wins exact ties",
            "sampling": False,
            "generated_tokens_per_prompt": STEPS,
            "early_stop": False,
        },
        "model": {
            "numeric_path": "unchanged accepted W4A8 software inference",
            "model": file_record(model),
            "adapter": file_record(ADAPTER),
            "layer23_v_only": "RTL sidecar replaces only scored autoregressive layer-23 V A8 bytes",
            "kv_policy": "causal per-layer K/V caches carried across prefill and four decode scores",
        },
        "accepted_rank1_payload": {
            "candidate_freeze": file_record(candidate.FREEZE),
            "generated_payload": file_record(
                ROOT / "rtl/generated/ace2_layer23_v_rank1_integer_correction_payload.svh"
            ),
            "refit": False,
            "rescale": False,
            "tuning": False,
        },
        "accepted_descriptor_integration": {
            "result": file_record(INTEGRATION_RESULT),
            "required_result_sha256": INTEGRATION_RESULT_SHA256,
            "descriptor": {
                "opcode": "0x0b",
                "flags": "0x00",
                "layer_id": 23,
                "m": 1,
                "n": 896,
                "k": 896,
                "src0": "0x0000001000000700",
                "src1": "0x000000010a384000",
                "dst": "0x0000001000000a80",
                "scale": "0x0000000200472800",
                "scratch": "0x0000000000000000",
            },
        },
        "icarus": {
            "compile_argv": compile_argv(binary),
            "per_step_vvp_argv_template": vvp_template(binary),
            "compile_count": 1,
            "simulation_count": 2 * STEPS,
            "top_module": TB_TOP,
        },
        "sources": sources,
        "source_set_sha256": sha256_bytes(canonical_bytes(sources)),
        "preflight": file_record(PREFLIGHT),
        "prohibitions": {
            "predecessor_attempt_replay_or_resume": True,
            "fresh_attempt_retry_replay_or_resume": True,
            "official_attempt": True,
            "prompt_substitution": True,
            "retry_or_replay": True,
            "factor_refit_rescale_or_tuning": True,
            "synthesis_ppa_opensta": True,
            "u280_xrt_stage2": True,
        },
    }
    write_json(FREEZE, frozen)
    freeze_measurement = file_record(FREEZE)
    print(
        "ACE2_HYBRID_FREEZE_PASS "
        f"freeze_sha256={freeze_measurement['sha256']}",
        flush=True,
    )
    return 0


def package_execution(prompts_path: Path) -> int:
    verify_exact_interpreter()
    verify_dependency_probe()
    require(FREEZE.is_file(), "freeze is absent")
    require(not PACKAGE.exists(), "execution package already exists")
    require(not AUTHORITY.exists(), "execution authority already exists")
    verify_predecessor_seals()
    validate_no_execution()
    frozen = read_json(FREEZE)
    verify_frozen_sources(frozen, prompts_path)
    require(
        prompt_records(prompts_path) == frozen["prompts"],
        "prompt substitution detected before package",
    )
    _snapshot, model, tokenizer_json, tokenizer_config = resolve_runtime_inputs()
    required_outputs, runtime = supervisor.prepare_runtime_bindings(
        source_root=ROOT / "rtl",
        state_dir=STATE,
        stdout_path=STDOUT,
        stderr_path=STDERR,
        terminal_record_path=TERMINAL,
        additional_outputs=(LIVE,),
    )
    require(str(STATE) not in required_outputs, "state_dir entered required-absent outputs")
    require(
        all(not is_within(Path(path), STATE) for path in required_outputs),
        "required-absent output is within state_dir",
    )
    freeze_sha256 = sha256_file(FREEZE)
    bindings = package_bindings(
        prompts_path,
        freeze_sha256,
        required_outputs,
        model,
        tokenizer_json,
        tokenizer_config,
        include_frozen_stage_files=True,
    )
    package = {
        "schema": supervisor.PACKAGE_SCHEMA,
        "package_id": f"{MISSION_ID}-{ATTEMPT_ID}",
        "bindings": bindings,
    }
    write_json(PACKAGE, package)
    require(runtime["state_dir"] == str(STATE), "runtime state_dir changed during packaging")
    print(
        "ACE2_HYBRID_PACKAGE_PASS "
        f"package_sha256={sha256_file(PACKAGE)}",
        flush=True,
    )
    return 0


def grant_authority(prompts_path: Path) -> int:
    verify_exact_interpreter()
    verify_dependency_probe()
    require(PACKAGE.is_file(), "execution package is absent")
    require(not AUTHORITY.exists(), "execution authority already exists")
    verify_predecessor_seals()
    validate_no_execution()
    frozen = read_json(FREEZE)
    verify_frozen_sources(frozen, prompts_path)
    _snapshot, model, tokenizer_json, tokenizer_config = resolve_runtime_inputs()
    required_outputs, runtime = supervisor.prepare_runtime_bindings(
        source_root=ROOT / "rtl",
        state_dir=STATE,
        stdout_path=STDOUT,
        stderr_path=STDERR,
        terminal_record_path=TERMINAL,
        additional_outputs=(LIVE,),
    )
    expected_bindings = package_bindings(
        prompts_path,
        sha256_file(FREEZE),
        required_outputs,
        model,
        tokenizer_json,
        tokenizer_config,
        include_frozen_stage_files=True,
    )
    package = read_json(PACKAGE)
    require(package["bindings"] == expected_bindings, "packaged bindings changed before authority")
    package_measurement = supervisor.measure_file(PACKAGE)
    authority = {
        "schema": supervisor.AUTHORITY_SCHEMA,
        "authority_id": f"operator-{MISSION_ID}-{ATTEMPT_ID}-execute-once",
        "decision": "execute_once",
        "package": {
            "path": str(PACKAGE),
            "byte_count": package_measurement.byte_count,
            "sha256": package_measurement.sha256,
            "package_id": package["package_id"],
        },
        "bindings": expected_bindings,
        "runtime": runtime,
    }
    write_json(AUTHORITY, authority)
    supervisor.validate_run(PACKAGE, AUTHORITY)
    print(
        "ACE2_HYBRID_AUTHORITY_PASS "
        f"authority_sha256={sha256_file(AUTHORITY)}",
        flush=True,
    )
    return 0


def verify_frozen_sources(frozen: dict[str, Any], prompts_path: Path) -> None:
    require(
        sha256_bytes(canonical_bytes(frozen["sources"]))
        == frozen["source_set_sha256"],
        "frozen source-set digest changed",
    )
    _snapshot, model, tokenizer_json, tokenizer_config = resolve_runtime_inputs()
    require(
        source_records(model, tokenizer_json, tokenizer_config, prompts_path)
        == frozen["sources"],
        "one or more frozen sources changed",
    )


def write_hex_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value:02x}\n" for value in raw), encoding="ascii")
    return file_record(path)


def int8_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().to(torch.int8).numpy().astype("i1").tobytes()


def parse_rtl_v_trace(stdout: str) -> dict[str, Any]:
    matches = list(V_BEAT_PATTERN.finditer(stdout))
    require(len(matches) == 8, "Icarus did not emit exactly eight corrected V beats")
    beats: dict[int, dict[str, Any]] = {}
    for match in matches:
        values = match.groupdict()
        beat = int(values["beat"])
        require(0 <= beat < 8, f"Icarus emitted out-of-range V beat {beat}")
        require(beat not in beats, f"Icarus emitted duplicate V beat {beat}")
        beats[beat] = {
            "rank_accumulator_s32": int(values["rank_acc"]),
            "rank_rounded_s32": int(values["rank_rounded"]),
            "rank_intermediate_s8": int(values["rank_s8"]),
            "baseline": bytes.fromhex(values["baseline"]),
            "corrected": bytes.fromhex(values["corrected"]),
        }
    require(set(beats) == set(range(8)), "Icarus corrected V beat coverage changed")
    rank_values = {
        (
            item["rank_accumulator_s32"],
            item["rank_rounded_s32"],
            item["rank_intermediate_s8"],
        )
        for item in beats.values()
    }
    require(len(rank_values) == 1, "Icarus rank intermediates changed across V beats")
    rank_accumulator, rank_rounded, rank_s8 = rank_values.pop()
    baseline = bytearray()
    corrected = bytearray()
    for beat in range(8):
        # %032x prints lane 15 first; tensor byte order starts with lane 0.
        baseline.extend(reversed(beats[beat]["baseline"]))
        corrected.extend(reversed(beats[beat]["corrected"]))
    return {
        "rank_accumulator_s32": rank_accumulator,
        "rank_rounded_s32": rank_rounded,
        "rank_intermediate_s8": rank_s8,
        "baseline": bytes(baseline),
        "corrected": bytes(corrected),
    }


def synthetic_v_trace(baseline: bytes, corrected: bytes) -> str:
    require(len(baseline) == 128 and len(corrected) == 128, "synthetic V vectors changed")
    lines = []
    for beat in range(8):
        start = beat * 16
        stop = start + 16
        lines.append(
            "ACE2_HYBRID_V_BEAT "
            f"beat={beat} rank_acc=-17 rank_rounded=-1 rank_s8=-1 "
            f"baseline={bytes(reversed(baseline[start:stop])).hex()} "
            f"corrected={bytes(reversed(corrected[start:stop])).hex()}"
        )
    return "\n".join(lines) + "\n"


def trace_is_rejected(stdout: str) -> bool:
    try:
        parse_rtl_v_trace(stdout)
    except HybridError:
        return True
    return False


def package_template_invariant(prompts_path: Path) -> int:
    verify_exact_interpreter()
    probe = verify_dependency_probe()
    require(not TEMPLATE_GATE.exists(), "package-template invariant already exists")
    require(not OUTPUT.exists(), f"{ATTEMPT_ID} output namespace already exists")
    verify_predecessor_seals()
    require_resolved_regular_file(prompts_path.resolve(strict=True), "private prompt")
    prompts = prompt_records(prompts_path)
    unseen = verify_unseen_prompt_hashes(prompts)
    _snapshot, model, tokenizer_json, tokenizer_config = resolve_runtime_inputs()
    for label, path in (
        ("model", model),
        ("tokenizer JSON", tokenizer_json),
        ("tokenizer config", tokenizer_config),
        ("adapter", ADAPTER.resolve(strict=True)),
    ):
        require_resolved_regular_file(path, label)
    require(
        sha256_file(INTEGRATION_RESULT) == INTEGRATION_RESULT_SHA256,
        "accepted integration RESULT hash changed",
    )
    candidate.validate_freeze(read_json(candidate.FREEZE))

    intended_outputs, intended_runtime = intended_runtime_paths()
    require(STATE not in intended_outputs, "state_dir entered intended outputs")
    require(
        all(not is_within(path, STATE) for path in intended_outputs),
        "intended output is within state_dir",
    )
    placeholder_freeze_sha256 = "0" * 64
    exact_argv = execution_argv(prompts_path, placeholder_freeze_sha256)
    with tempfile.TemporaryDirectory(prefix=f"ace2-{ATTEMPT_ID}-template-") as temporary:
        sandbox = Path(temporary)
        sandbox_output = sandbox / "output"
        sandbox_output.mkdir()
        required_outputs, runtime = supervisor.prepare_runtime_bindings(
            source_root=ROOT / "rtl",
            state_dir=sandbox / "supervisor-state",
            stdout_path=sandbox_output / "execution.stdout.log",
            stderr_path=sandbox_output / "execution.stderr.log",
            terminal_record_path=sandbox_output / "terminal-record.json",
            additional_outputs=(sandbox_output / "live",),
        )
        bindings = package_bindings(
            prompts_path,
            placeholder_freeze_sha256,
            required_outputs,
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=False,
        )
        template_package = {
            "schema": supervisor.PACKAGE_SCHEMA,
            "package_id": f"{MISSION_ID}-{ATTEMPT_ID}-template",
            "bindings": bindings,
        }
        package_path = sandbox / "execution-package.json"
        authority_path = sandbox / "execution-authority.json"
        write_json(package_path, template_package)
        package_measurement = supervisor.measure_file(package_path)
        template_authority = {
            "schema": supervisor.AUTHORITY_SCHEMA,
            "authority_id": f"operator-{MISSION_ID}-{ATTEMPT_ID}-template-validation",
            "decision": "execute_once",
            "package": {
                "path": str(package_path),
                "byte_count": package_measurement.byte_count,
                "sha256": package_measurement.sha256,
                "package_id": template_package["package_id"],
            },
            "bindings": bindings,
            "runtime": runtime,
        }
        write_json(authority_path, template_authority)
        validated = supervisor.validate_run(package_path, authority_path)
        require(validated.package["bindings"]["argv"] == exact_argv, "template argv changed")
        template_package_sha256 = package_measurement.sha256

    baseline = bytes(range(128))
    corrected = bytes((value * 29 + 7) & 0xFF for value in range(128))
    valid_trace = synthetic_v_trace(baseline, corrected)
    reconstructed = parse_rtl_v_trace(valid_trace)
    missing_trace = "\n".join(valid_trace.splitlines()[:-1]) + "\n"
    duplicate_trace = valid_trace + valid_trace.splitlines()[0] + "\n"
    out_of_range_trace = valid_trace.replace("beat=7 ", "beat=8 ", 1)
    require(reconstructed["baseline"] == baseline, "baseline emitted-byte reconstruction changed")
    require(reconstructed["corrected"] == corrected, "corrected emitted-byte reconstruction changed")
    require(trace_is_rejected(missing_trace), "missing V beat was not rejected")
    require(trace_is_rejected(duplicate_trace), "duplicate V beat was not rejected")
    require(trace_is_rejected(out_of_range_trace), "out-of-range V beat was not rejected")
    require(
        not OUTPUT.exists(),
        f"{ATTEMPT_ID} output namespace was consumed by template gate",
    )
    record = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "attempt_identity": ATTEMPT_ID,
        "status": "PASS_NONCONSUMING_PACKAGE_TEMPLATE_INVARIANT",
        "created_at_utc": utc_now(),
        "prompts": prompts,
        "unseen_check": unseen,
        "integration_result_sha256": INTEGRATION_RESULT_SHA256,
        "dependency_probe": file_record(DEPENDENCY_PROBE),
        "environment_sha256": probe["environment_sha256"],
        "template": {
            "package_sha256": template_package_sha256,
            "argv_sha256": supervisor.canonical_value_sha256(exact_argv),
            "intended_runtime": {
                "required_absent_output_count": len(intended_outputs),
                "state_dir_distinct": intended_runtime["state_dir"]
                not in {str(path) for path in intended_outputs},
            },
        },
        "checks": {
            "predecessor_terminal_seals_immutable": True,
            "new_prompt_hashes_unseen": True,
            "resolved_model_tokenizer_bindings_regular": True,
            "integration_result_binding_exact": True,
            "output_namespace_absent_before_and_after": True,
            "supervisor_template_validation_passed": True,
            "state_dir_excluded_from_required_absent_outputs": True,
            "all_required_absent_outputs_outside_state_dir": True,
            "execution_argv_exact": True,
            "icarus_emitted_byte_reconstruction_exact": True,
            "malformed_missing_beat_rejected": True,
            "malformed_duplicate_beat_rejected": True,
            "malformed_out_of_range_beat_rejected": True,
            "process_not_started": True,
            "exact_child_dependency_probe_passed": True,
        },
    }
    write_json(TEMPLATE_GATE, record)
    print(
        "ACE2_HYBRID_PACKAGE_TEMPLATE_INVARIANT_PASS "
        f"sha256={sha256_file(TEMPLATE_GATE)} process_start_count=0",
        flush=True,
    )
    return 0


def pack_w4(qweight: torch.Tensor) -> bytes:
    rows = qweight.detach().cpu().contiguous().to(torch.int16).numpy()
    require(list(rows.shape) == [128, 896], "V qweight shape changed")
    require(bool(np.all(rows >= -8)) and bool(np.all(rows <= 7)), "V qweight escaped int4")
    low = rows[:, 0::2] & 0xF
    high = (rows[:, 1::2] & 0xF) << 4
    return (low | high).astype(np.uint8).tobytes()


def pack_metadata(result: dict[str, Any]) -> bytes:
    multiplier = result["multiplier"].detach().cpu().to(torch.int64).flatten().tolist()
    shifts = result["right_shift"].detach().cpu().to(torch.int64).flatten().tolist()
    require(len(multiplier) == 128 and len(shifts) == 128, "V metadata shape changed")
    raw = bytearray()
    for mult, shift in zip(multiplier, shifts, strict=True):
        require(-(1 << 31) <= mult < (1 << 31), "V multiplier escaped int32")
        require(0 <= shift <= 63, "V shift escaped u6")
        record = bytearray(16)
        record[0:4] = struct.pack("<i", mult)
        record[4] = shift
        raw.extend(record)
    return bytes(raw)


def score_w4a8(
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    head: dict[str, torch.Tensor],
) -> tuple[np.ndarray, torch.Tensor, float]:
    item = backend.derive_final_rmsnorm_case(state, norm_gain)
    float_chunks = []
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        float_chunks.append(
            torch.mv(embedding[start:stop].to(torch.float32), item["float_norm"])
        )
    output_scale = float(backend.canonical.scale_for(torch.cat(float_chunks)))
    activation = item["final_q"].to(torch.int32)
    accumulator = torch.empty((backend.MODEL_OUTPUT_DOMAIN,), dtype=torch.int64)
    for start in range(0, backend.MODEL_OUTPUT_DOMAIN, 4096):
        stop = min(backend.MODEL_OUTPUT_DOMAIN, start + 4096)
        accumulator[start:stop] = (
            head["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    multiplier, right_shift = backend.canonical.derive_multiplier(
        float(item["final_scale"]) * head["weight_scale"] / output_scale
    )
    rounded = localizer.round_shift_even_tensor(accumulator * multiplier, right_shift)
    output_q = rounded.clamp(-128, 127).to(torch.int8).cpu().contiguous()
    return output_q.to(torch.float64).numpy() * output_scale, output_q, output_scale


class Rank1HybridProjectionCache(localizer.FastProjectionCache):
    def __init__(
        self,
        weights: Any,
        adapter: Any,
        *,
        mode: str,
        prompt_id: str,
        first_scored_position: int,
        binary: Path,
    ) -> None:
        super().__init__(weights, adapter)
        require(mode in {"rtl", "software"}, "unsupported hybrid cache mode")
        self.mode = mode
        self.prompt_id = prompt_id
        self.first_scored_position = first_scored_position
        self.binary = binary
        self.records: list[dict[str, Any]] = []
        self.simulation_seconds = 0.0

    def _apply(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        match = re.fullmatch(r"layer23_position(?P<position>\d+)_v", name)
        if match is None:
            return result
        position = int(match.group("position"))
        if position < self.first_scored_position:
            return result
        step = position - self.first_scored_position
        require(0 <= step < STEPS, "sidecar activation escaped four scored steps")
        input_dequantized = (
            result["input_q"].to(torch.float64) * float(result["input_scale"])
        ).reshape(1, 896)
        baseline = (
            result["output_q"].to(torch.float64) * float(result["output_scale"])
        ).reshape(1, 128)
        integer_evidence, _correction, corrected = candidate.integer_arithmetic(
            read_json(candidate.FREEZE),
            input_dequantized,
            baseline,
            write_artifacts=False,
        )
        input_scale = float(
            read_json(candidate.FREEZE)["quantization"]["input_payload"]["scale32"][
                "decoded_value"
            ]
        )
        reference_input_q = torch.round(input_dequantized / input_scale).clamp(
            -128, 127
        ).to(torch.int8)
        require(
            torch.equal(reference_input_q.reshape(-1), result["input_q"].to(torch.int8)),
            "shell activation bytes differ from frozen integer-reference input bytes",
        )
        corrected_q = torch.round(
            corrected
            / float(candidate.RETAINED_OUTPUT_SCALE)
        ).clamp(-128, 127).to(torch.int8).reshape(128)
        baseline_q = result["output_q"].to(torch.int8).reshape(128)
        rank_record = integer_evidence["rank_intermediate"]
        saturation_byte = (
            int(rank_record["saturation"]["count"] > 0)
            | (int(integer_evidence["correction_output"]["saturation"]["count"] > 0) << 1)
            | (int(integer_evidence["final_saturating_add"]["saturation"]["count"] > 0) << 2)
        )
        record: dict[str, Any] = {
            "generation_index": step,
            "absolute_position": position,
            "descriptor_accepted": True,
            "rank_accumulator_s32": int(rank_record["accumulator_s32"]["min"]),
            "rank_rounded_s32": int(rank_record["rounded_s32"]["min"]),
            "rank_intermediate_s8": int(
                torch.round(
                    torch.tensor(rank_record["rounded_s32"]["min"])
                ).clamp(-128, 127).item()
            ),
            "baseline_v_sha256": sha256_bytes(int8_bytes(baseline_q)),
            "corrected_v_sha256": sha256_bytes(int8_bytes(corrected_q)),
            "corrected_v_bytes": 128,
            "downstream_v_source": "software_integer_reference",
            "saturation": {
                "rank": bool(saturation_byte & 1),
                "correction": bool(saturation_byte & 2),
                "add": bool(saturation_byte & 4),
                "numeric_overflow": False,
            },
            "reference_hardware_representable": integer_evidence[
                "hardware_representable"
            ],
        }
        require(
            rank_record["accumulator_s32"]["min"]
            == rank_record["accumulator_s32"]["max"],
            "rank accumulator reference is not scalar",
        )
        require(
            rank_record["rounded_s32"]["min"]
            == rank_record["rounded_s32"]["max"],
            "rank rounded reference is not scalar",
        )
        if self.mode == "rtl":
            step_dir = LIVE / "traces" / self.prompt_id / f"step-{step:02d}"
            vectors = step_dir / "vectors"
            artifacts = {
                "activation_s8": write_hex_bytes(
                    vectors / "activation.hex", int8_bytes(result["input_q"])
                ),
                "v_weight_w4": write_hex_bytes(
                    vectors / "v-weight.hex", pack_w4(result["qweight"])
                ),
                "v_metadata": write_hex_bytes(
                    vectors / "v-metadata.hex", pack_metadata(result)
                ),
                "baseline_v_s8": write_hex_bytes(
                    vectors / "baseline-v.hex", int8_bytes(baseline_q)
                ),
                "software_corrected_v_reference_s8": write_hex_bytes(
                    vectors / "corrected-v.hex", int8_bytes(corrected_q)
                ),
                "rank_s32": write_hex_bytes(
                    vectors / "rank.hex",
                    struct.pack(
                        ">II",
                        record["rank_accumulator_s32"] & 0xFFFFFFFF,
                        record["rank_rounded_s32"] & 0xFFFFFFFF,
                    ),
                ),
                "rank_s8": write_hex_bytes(
                    vectors / "rank-s8.hex",
                    bytes([record["rank_intermediate_s8"] & 0xFF]),
                ),
                "saturation": write_hex_bytes(
                    vectors / "saturation.hex", bytes([saturation_byte])
                ),
            }
            # rank.hex is two 32-bit records, not eight byte records.
            (vectors / "rank.hex").write_text(
                f"{record['rank_accumulator_s32'] & 0xFFFFFFFF:08x}\n"
                f"{record['rank_rounded_s32'] & 0xFFFFFFFF:08x}\n",
                encoding="ascii",
            )
            artifacts["rank_s32"] = file_record(vectors / "rank.hex")
            replacements = {
                "activation": vectors / "activation.hex",
                "v_weight": vectors / "v-weight.hex",
                "v_meta": vectors / "v-metadata.hex",
                "baseline": vectors / "baseline-v.hex",
                "corrected": vectors / "corrected-v.hex",
                "rank": vectors / "rank.hex",
                "rank_s8": vectors / "rank-s8.hex",
                "saturation": vectors / "saturation.hex",
            }
            argv = [
                item.format(**{key: str(value) for key, value in replacements.items()})
                for item in vvp_template(self.binary)
            ]
            started = time.monotonic()
            completed = subprocess.run(
                argv,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            elapsed = time.monotonic() - started
            self.simulation_seconds += elapsed
            (step_dir / "simulation.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (step_dir / "simulation.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            require(
                completed.returncode == 0,
                f"ace2_shell Icarus failed prompt={self.prompt_id} step={step}",
            )
            marker = PASS_PATTERN.search(completed.stdout)
            require(marker is not None, "ace2_shell hybrid PASS marker absent")
            metrics = {key: int(value) for key, value in marker.groupdict().items()}
            require(metrics["overflow"] == 0, "RTL numeric overflow observed")
            require(
                metrics["saturation"] == int(bool(saturation_byte & 0x7)),
                "RTL saturation summary differs",
            )
            rtl_trace = parse_rtl_v_trace(completed.stdout)
            require(
                rtl_trace["rank_accumulator_s32"] == record["rank_accumulator_s32"],
                "Icarus rank accumulator differs from software integer reference",
            )
            require(
                rtl_trace["rank_rounded_s32"] == record["rank_rounded_s32"],
                "Icarus rounded rank differs from software integer reference",
            )
            require(
                rtl_trace["rank_intermediate_s8"] == record["rank_intermediate_s8"],
                "Icarus rank s8 differs from software integer reference",
            )
            require(
                rtl_trace["baseline"] == int8_bytes(baseline_q),
                "Icarus disabled-sidecar V bytes differ from software baseline",
            )
            require(
                rtl_trace["corrected"] == int8_bytes(corrected_q),
                "Icarus corrected V bytes differ from software integer reference",
            )
            rtl_corrected_q = torch.from_numpy(
                np.frombuffer(rtl_trace["corrected"], dtype=np.int8).copy()
            )
            artifacts["rtl_emitted_corrected_v_s8"] = write_hex_bytes(
                vectors / "rtl-emitted-corrected-v.hex", rtl_trace["corrected"]
            )
            record.update(
                {
                    "rank_accumulator_s32": rtl_trace["rank_accumulator_s32"],
                    "rank_rounded_s32": rtl_trace["rank_rounded_s32"],
                    "rank_intermediate_s8": rtl_trace["rank_intermediate_s8"],
                    "baseline_v_sha256": sha256_bytes(rtl_trace["baseline"]),
                    "corrected_v_sha256": sha256_bytes(rtl_trace["corrected"]),
                    "downstream_v_source": "icarus_ace2_shell_mem_wdata",
                }
            )
            record.update(
                {
                    "rtl": {
                        "status": "PASS_BYTE_EXACT",
                        "argv": argv,
                        "wall_seconds": elapsed,
                        "metrics": metrics,
                        "stdout": file_record(step_dir / "simulation.stdout.log"),
                        "stderr": file_record(step_dir / "simulation.stderr.log"),
                    },
                    "artifacts": artifacts,
                }
            )
        result = dict(result)
        result["output_q"] = rtl_corrected_q if self.mode == "rtl" else corrected_q
        self.records.append(record)
        return result

    def derive_projection(
        self,
        name: str,
        merged: torch.Tensor,
        input_q: torch.Tensor,
        input_scale: float,
        float_input: torch.Tensor,
        source_hashes: dict[str, str],
    ) -> dict[str, Any]:
        return self._apply(
            name,
            super().derive_projection(
                name, merged, input_q, input_scale, float_input, source_hashes
            ),
        )

    def from_fixed_metadata(
        self,
        name: str,
        input_q: torch.Tensor,
        input_scale: float,
        qweight: torch.Tensor,
        multiplier: torch.Tensor,
        right_shift: torch.Tensor,
        output_scale: float,
    ) -> dict[str, Any]:
        return self._apply(
            name,
            super().from_fixed_metadata(
                name,
                input_q,
                input_scale,
                qweight,
                multiplier,
                right_shift,
                output_scale,
            ),
        )


def run_sequence(
    prompt_id: str,
    token_ids: list[int],
    tokenizer: Any,
    weights: Any,
    adapter: Any,
    binary: Path,
    mode: str,
) -> dict[str, Any]:
    embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
    norm_gain = weights.get_tensor("model.norm.weight").contiguous()
    head = localizer.derive_lm_head_weights(embedding)
    layer_caches = [backend.empty_layer_cache() for _ in range(backend.LAYERS)]
    templates: list[dict[str, Any] | None] = [None] * backend.LAYERS
    selected: list[int] = []
    steps: list[dict[str, Any]] = []
    first_scored = len(token_ids) - 1
    cache = Rank1HybridProjectionCache(
        weights,
        adapter,
        mode=mode,
        prompt_id=prompt_id,
        first_scored_position=first_scored,
        binary=binary,
    )
    guard = localizer.SubstitutionRequireGuard()
    started = time.monotonic()
    cache.install()
    guard.install()
    try:
        with torch.no_grad():
            for position in range(len(token_ids) + STEPS - 1):
                token_id = (
                    token_ids[position]
                    if position < len(token_ids)
                    else selected[position - len(token_ids)]
                )
                state = backend.embedding_state(weights, token_id, position)
                for layer_id in range(backend.LAYERS):
                    _derived, state, templates[layer_id] = backend.derive_layer_token(
                        layer_id,
                        state,
                        layer_caches[layer_id],
                        templates[layer_id],
                        weights,
                        adapter,
                    )
                    del _derived
                if position < first_scored:
                    continue
                step = position - first_scored
                scores, output_q, output_scale = score_w4a8(
                    state, norm_gain, embedding, head
                )
                selected_token = int(np.argmax(scores))
                selected.append(selected_token)
                logits_bytes = int8_bytes(output_q) + struct.pack("<d", output_scale)
                steps.append(
                    {
                        "generation_index": step,
                        "absolute_position": position,
                        "context_token_count": position + 1,
                        "selected_token_id": selected_token,
                        "logits_representation": {
                            "dtype": "signed_int8_plus_float64_le_scale",
                            "elements": backend.MODEL_OUTPUT_DOMAIN,
                            "sha256": sha256_bytes(logits_bytes),
                            "output_s8_sha256": sha256_bytes(int8_bytes(output_q)),
                            "scale_f64le_sha256": sha256_bytes(
                                struct.pack("<d", output_scale)
                            ),
                        },
                        "kv_cache_lengths_after_layer23": {
                            "k": len(layer_caches[TARGET_LAYER]["k"]),
                            "v": len(layer_caches[TARGET_LAYER]["v"]),
                        },
                    }
                )
    finally:
        guard.restore()
        cache.restore()
    require(len(selected) == STEPS, "generation did not produce exactly four tokens")
    require(len(cache.records) == STEPS, "sidecar/reference V record count changed")
    require(
        [item["generation_index"] for item in cache.records] == list(range(STEPS)),
        "sidecar/reference V step order changed",
    )
    elapsed = time.monotonic() - started
    return {
        "mode": mode,
        "generated_token_ids": selected,
        "decoded_text": tokenizer.decode(
            selected,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "steps": steps,
        "layer23_v_records": cache.records,
        "latency_seconds": {
            "total": elapsed,
            "simulation": cache.simulation_seconds,
            "software": elapsed - cache.simulation_seconds,
        },
        "suppressed_nonfunctional_cache_assertions": {
            "count": len(guard.suppressed),
            "messages": sorted(set(guard.suppressed)),
        },
    }


def compare_runs(rtl: dict[str, Any], software: dict[str, Any]) -> dict[str, Any]:
    require(
        rtl["generated_token_ids"] == software["generated_token_ids"],
        "RTL/software generated token sequence mismatch",
    )
    require(rtl["decoded_text"] == software["decoded_text"], "decoded text mismatch")
    comparisons = []
    for rtl_step, software_step, rtl_v, software_v in zip(
        rtl["steps"],
        software["steps"],
        rtl["layer23_v_records"],
        software["layer23_v_records"],
        strict=True,
    ):
        require(
            rtl_step["logits_representation"]["sha256"]
            == software_step["logits_representation"]["sha256"],
            "downstream logits representation mismatch",
        )
        require(
            rtl_step["selected_token_id"] == software_step["selected_token_id"],
            "selected token mismatch",
        )
        require(
            rtl_v["downstream_v_source"] == "icarus_ace2_shell_mem_wdata",
            "RTL hybrid did not consume Icarus-emitted corrected V bytes",
        )
        require(
            software_v["downstream_v_source"] == "software_integer_reference",
            "software reference V source changed",
        )
        for key in (
            "rank_accumulator_s32",
            "rank_rounded_s32",
            "rank_intermediate_s8",
            "baseline_v_sha256",
            "corrected_v_sha256",
            "saturation",
        ):
            require(rtl_v[key] == software_v[key], f"V comparison mismatch: {key}")
        comparisons.append(
            {
                "generation_index": rtl_step["generation_index"],
                "descriptor_acceptance": True,
                "rank_accumulator_and_intermediate_equal": True,
                "corrected_v_128_bytes_equal": True,
                "rtl_emitted_corrected_v_consumed_downstream": True,
                "downstream_logits_equal": True,
                "selected_token_equal": True,
            }
        )
    return {
        "all_steps_byte_exact": True,
        "generated_sequence_equal": True,
        "decoded_text_equal": True,
        "steps": comparisons,
    }


def write_sums() -> None:
    files = sorted(
        path
        for path in LIVE.rglob("*")
        if path.is_file() and path != SUMS
    )
    SUMS.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(ROOT)}\n" for path in files),
        encoding="utf-8",
    )


def execute(prompts_path: Path, expected_freeze_sha256: str) -> int:
    verify_exact_interpreter()
    verify_dependency_probe()
    require(FREEZE.is_file(), "freeze is absent")
    require(
        sha256_file(FREEZE) == expected_freeze_sha256,
        "freeze hash differs from exact command binding",
    )
    require(not LIVE.exists(), "live output already exists")
    frozen = read_json(FREEZE)
    verify_frozen_sources(frozen, prompts_path)
    require(
        prompt_records(prompts_path) == frozen["prompts"],
        "prompt substitution detected after freeze",
    )
    require(
        frozen["decoding"]["generated_tokens_per_prompt"] == STEPS,
        "frozen token count changed",
    )
    LIVE.mkdir(parents=True)
    write_json(
        LIVE / "execution-start.json",
        {
            "schema_version": 1,
            "mission_id": MISSION_ID,
            "started_at_utc": utc_now(),
            "freeze": file_record(FREEZE),
            "official_attempt": False,
            "retry_or_replay": False,
        },
    )
    binary = LIVE / "sim/ace2_layer23_v_rank1_hybrid_shell.vvp"
    binary.parent.mkdir(parents=True)
    compile_started = time.monotonic()
    compiled = subprocess.run(
        frozen["icarus"]["compile_argv"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    compile_seconds = time.monotonic() - compile_started
    (LIVE / "sim/iverilog.stdout.log").write_text(compiled.stdout, encoding="utf-8")
    (LIVE / "sim/iverilog.stderr.log").write_text(compiled.stderr, encoding="utf-8")
    require(compiled.returncode == 0, "frozen Icarus compile failed")
    require(binary.is_file(), "frozen Icarus binary was not produced")

    snapshot, model, _tokenizer_json, _tokenizer_config = resolve_runtime_inputs()
    backend.configure_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    prompts = load_prompts(prompts_path)
    prompt_results = []
    with safe_open(model, framework="pt", device="cpu") as weights, safe_open(
        ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        for prompt, frozen_prompt in zip(prompts, frozen["prompts"], strict=True):
            require(prompt["id"] == frozen_prompt["prompt_id"], "prompt order changed")
            token_ids = generation_runner.canonical_chat_token_ids(
                tokenizer, prompt["text"]
            )
            require(bool(token_ids), "tokenizer produced an empty prompt")
            rtl = run_sequence(
                prompt["id"], token_ids, tokenizer, weights, adapter, binary, "rtl"
            )
            software = run_sequence(
                prompt["id"], token_ids, tokenizer, weights, adapter, binary, "software"
            )
            comparison = compare_runs(rtl, software)
            prompt_results.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_sha256": frozen_prompt["sha256"],
                    "prompt_token_count": len(token_ids),
                    "rtl_hybrid": rtl,
                    "software_integer_reference": software,
                    "comparison": comparison,
                }
            )
            print(
                "ACE2_HYBRID_PROMPT_COMPLETE "
                f"prompt_id={prompt['id']} tokens={rtl['generated_token_ids']} "
                f"decoded={json.dumps(rtl['decoded_text'], ensure_ascii=True)}",
                flush=True,
            )
    require(len(prompt_results) == 2, "two-prompt cardinality changed")
    require(
        all(len(item["rtl_hybrid"]["generated_token_ids"]) == STEPS for item in prompt_results),
        "four-token cardinality changed",
    )
    result = {
        "schema_version": 1,
        "mission_id": MISSION_ID,
        "status": "PASS_BOUNDED_NONOFFICIAL_HYBRID_BYTE_EXACT",
        "classification": CLASSIFICATION,
        "completed_at_utc": utc_now(),
        "claim_boundary": "bounded nonofficial hybrid evidence only; not Stage-1 completion",
        "exactly_once": {
            "attempt_identity": frozen["attempt_identity"],
            "process_start_count": 1,
            "retry": False,
            "replay": False,
            "prompt_substitution": False,
            "refit_rescale_or_tuning": False,
        },
        "freeze": file_record(FREEZE),
        "compile": {
            "argv": frozen["icarus"]["compile_argv"],
            "wall_seconds": compile_seconds,
            "stdout": file_record(LIVE / "sim/iverilog.stdout.log"),
            "stderr": file_record(LIVE / "sim/iverilog.stderr.log"),
            "binary": file_record(binary),
        },
        "prompts": prompt_results,
        "aggregate": {
            "prompt_count": 2,
            "generated_tokens_per_prompt": STEPS,
            "autoregressive_steps": 2 * STEPS,
            "corrected_v_bytes_compared": 2 * STEPS * 128,
            "descriptor_acceptance_all_steps": True,
            "rank_accumulator_and_intermediates_all_steps": True,
            "all_corrected_v_bytes_equal": True,
            "rtl_emitted_corrected_v_consumed_downstream_all_steps": True,
            "downstream_logits_all_steps_equal": True,
            "selected_tokens_all_steps_equal": True,
            "four_token_sequences_equal": True,
            "disabled_sidecar_control_all_steps": True,
            "simulation_latency_seconds": sum(
                item["rtl_hybrid"]["latency_seconds"]["simulation"]
                for item in prompt_results
            ),
            "hybrid_software_latency_seconds": sum(
                item["rtl_hybrid"]["latency_seconds"]["software"]
                for item in prompt_results
            ),
            "reference_software_latency_seconds": sum(
                item["software_integer_reference"]["latency_seconds"]["software"]
                for item in prompt_results
            ),
            "icarus_compile_latency_seconds": compile_seconds,
        },
        "scope_guards": frozen["prohibitions"],
    }
    write_json(RESULT, result)
    write_sums()
    print(
        f"ACE2_{MISSION_ID.upper()}_TERMINAL "
        f"status={result['status']} result_sha256={sha256_file(RESULT)}",
        flush=True,
    )
    return 0


def verify() -> int:
    verify_exact_interpreter()
    verify_dependency_probe()
    require(RESULT.is_file(), "hybrid result is absent")
    require(SUMS.is_file(), "hybrid SHA256SUMS is absent")
    require(TERMINAL.is_file(), "supervisor terminal record is absent")
    result = read_json(RESULT)
    terminal = read_json(TERMINAL)
    verify_predecessor_seals()
    require(
        result["status"] == "PASS_BOUNDED_NONOFFICIAL_HYBRID_BYTE_EXACT",
        "hybrid result is not PASS",
    )
    require(terminal["outcome"] == "success", "supervisor did not seal success")
    require(terminal["process_start_count"] == 1, "supervisor start count differs")
    require(terminal["process_started"] is True, "supervisor process-start flag differs")
    require(
        result["aggregate"]["prompt_count"] == 2
        and result["aggregate"]["generated_tokens_per_prompt"] == 4
        and result["aggregate"]["autoregressive_steps"] == 8,
        "bounded prompt/token cardinality changed",
    )
    require(result["mission_id"] == MISSION_ID, "result mission identity changed")
    require(
        result["exactly_once"]["attempt_identity"] == ATTEMPT_ID,
        "result attempt identity changed",
    )
    required_aggregate_checks = (
        "descriptor_acceptance_all_steps",
        "rank_accumulator_and_intermediates_all_steps",
        "all_corrected_v_bytes_equal",
        "rtl_emitted_corrected_v_consumed_downstream_all_steps",
        "downstream_logits_all_steps_equal",
        "selected_tokens_all_steps_equal",
        "four_token_sequences_equal",
        "disabled_sidecar_control_all_steps",
    )
    require(
        all(result["aggregate"][key] is True for key in required_aggregate_checks),
        "aggregate byte-exact acceptance predicate changed",
    )
    for prompt in result["prompts"]:
        rtl = prompt["rtl_hybrid"]
        software = prompt["software_integer_reference"]
        require(len(rtl["steps"]) == STEPS, "RTL step count changed")
        require(len(rtl["layer23_v_records"]) == STEPS, "RTL V record count changed")
        require(
            rtl["generated_token_ids"] == software["generated_token_ids"],
            "generated sequence equality changed",
        )
        for index, (rtl_step, software_step, rtl_v, comparison) in enumerate(
            zip(
                rtl["steps"],
                software["steps"],
                rtl["layer23_v_records"],
                prompt["comparison"]["steps"],
                strict=True,
            )
        ):
            require(
                rtl_step["generation_index"] == index
                and software_step["generation_index"] == index,
                "decode step order changed",
            )
            require(
                rtl_step["kv_cache_lengths_after_layer23"]["k"]
                == rtl_step["context_token_count"]
                == rtl_step["kv_cache_lengths_after_layer23"]["v"],
                "RTL KV/decode carry length changed",
            )
            require(
                software_step["kv_cache_lengths_after_layer23"]
                == rtl_step["kv_cache_lengths_after_layer23"],
                "RTL/software KV carry differs",
            )
            require(
                rtl_v["downstream_v_source"] == "icarus_ace2_shell_mem_wdata"
                and rtl_v["corrected_v_bytes"] == 128,
                "Icarus-emitted V downstream binding changed",
            )
            require(
                all(value is True for value in comparison.values() if isinstance(value, bool)),
                "per-step byte-exact predicate changed",
            )
        for run in (rtl, software):
            latency = run["latency_seconds"]
            require(
                latency["total"] >= 0
                and latency["simulation"] >= 0
                and latency["software"] >= 0,
                "negative split latency observed",
            )
            require(
                abs(
                    latency["total"]
                    - latency["simulation"]
                    - latency["software"]
                )
                < 1e-6,
                "split latency does not reconstruct total latency",
            )
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == digest, f"trace hash mismatch: {relative}")
    print(
        "ACE2_HYBRID_DECISIVE_VERIFY_PASS "
        f"result_sha256={sha256_file(RESULT)} terminal_sha256={sha256_file(TERMINAL)}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--probe", action="store_true")
    actions.add_argument("--check-package-template", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--package", action="store_true")
    actions.add_argument("--authority", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--verify", action="store_true")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--freeze-sha256", default="")
    args = parser.parse_args()
    try:
        if args.probe:
            return dependency_probe()
        if args.check_package_template:
            return package_template_invariant(args.prompts.resolve())
        if args.preflight:
            return preflight(args.prompts.resolve())
        if args.freeze:
            return freeze(args.prompts.resolve())
        if args.package:
            return package_execution(args.prompts.resolve())
        if args.authority:
            return grant_authority(args.prompts.resolve())
        if args.execute:
            require(bool(args.freeze_sha256), "--freeze-sha256 is required")
            return execute(args.prompts.resolve(), args.freeze_sha256)
        return verify()
    except Exception as error:
        if args.execute and not RESULT.exists():
            LIVE.mkdir(parents=True, exist_ok=True)
            write_json(
                FAILURE,
                {
                    "schema_version": 1,
                    "mission_id": MISSION_ID,
                    "status": "FAILED_SEALED_NO_RERUN",
                    "failed_at_utc": utc_now(),
                    "failure_taxonomy": "hybrid_execution_failure",
                    "root_cause_hypothesis": f"{type(error).__name__}: {error}",
                    "regression": "no replay; preserve supervisor captures and live traces",
                },
            )
            print(
                f"ACE2_{MISSION_ID.upper()}_TERMINAL "
                f"status=FAILED_SEALED_NO_RERUN failure_sha256={sha256_file(FAILURE)}",
                flush=True,
            )
        elif not args.verify and not TERMINAL.exists():
            write_json(
                TERMINAL,
                {
                    "schema_version": 1,
                    "mission_id": MISSION_ID,
                    "attempt_identity": ATTEMPT_ID,
                    "status": "FAILED_SEALED_NO_EXECUTION",
                    "outcome": "pre_execution_failure",
                    "sealed_at_utc": utc_now(),
                    "process_start_count": 0,
                    "process_started": False,
                    "failure": {
                        "failure_taxonomy": "pre_execution_gate_failure",
                        "phase": next(
                            name
                            for name in (
                                "probe",
                                "check_package_template",
                                "preflight",
                                "freeze",
                                "package",
                                "authority",
                            )
                            if getattr(args, name)
                        ),
                        "root_cause_hypothesis": f"{type(error).__name__}: {error}",
                        "regression": "No retry, replay, or resume; preserve the failed gate artifacts.",
                    },
                },
            )
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
