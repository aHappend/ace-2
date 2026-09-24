#!/usr/bin/env python3
"""Pristine-environment preloader for nonofficial hybrid attempt 0005."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
IMPLEMENTATION = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid.py"
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0005"
)
TERMINAL = OUTPUT / "terminal-record.json"
EXPECTED_PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
EXPECTED_PYTHON_SHA256 = (
    "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
)
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/home/argustest",
    "LC_ALL": "C",
    "PATH": (
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
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "/home/argustest/ace-2/.venv/lib/python3.13/site-packages",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
REQUIRED_PREIMPORT_ABSENT = (
    "numpy",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "tools",
)

# Capture the Python process launch state before any dependency or project import.
PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)


class PreloaderError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreloaderError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid05",
        "attempt_identity": "nonofficial-hybrid-0005",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": _utc_now(),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "regression": "No retry, replay, or resume; preserve the failed gate artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _verify_pre_import_launch() -> None:
    interpreter = Path(sys.executable).resolve()
    _require(interpreter == EXPECTED_PYTHON, "Python interpreter binding changed")
    _require(
        _sha256_file(interpreter) == EXPECTED_PYTHON_SHA256,
        "Python interpreter SHA-256 changed",
    )
    _require(PRE_IMPORT_CWD == str(ROOT), "final-child cwd binding changed")
    _require(
        Path(PRE_IMPORT_ARGV[0]).resolve() == LAUNCHER,
        "launcher argv[0] binding changed",
    )
    _require(
        PRE_IMPORT_ENVIRONMENT == EXPECTED_ENVIRONMENT,
        "pristine pre-import OS environment differs from the frozen launch environment",
    )
    contaminated = sorted(
        name
        for name in PRE_IMPORT_MODULES
        if any(name == root or name.startswith(f"{root}.") for root in REQUIRED_PREIMPORT_ABSENT)
    )
    _require(not contaminated, f"dependency/project modules loaded before snapshot: {contaminated}")
    _require(
        stat.S_ISREG(os.lstat(LAUNCHER).st_mode)
        and stat.S_ISREG(os.lstat(IMPLEMENTATION).st_mode),
        "launcher or runner is not a regular file",
    )


def _environment_delta(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, Any]:
    return {
        "added": {key: after[key] for key in sorted(after.keys() - before.keys())},
        "removed": {key: before[key] for key in sorted(before.keys() - after.keys())},
        "changed": {
            key: {"before": before[key], "after": after[key]}
            for key in sorted(before.keys() & after.keys())
            if before[key] != after[key]
        },
    }


def _configure_runner(runner: Any, post_import_environment: dict[str, str]) -> None:
    runner.MISSION_ID = "stage1rank1hybrid05"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0005"
    runner.OUTPUT = OUTPUT
    runner.PREFLIGHT = OUTPUT / "preflight.json"
    runner.FREEZE = OUTPUT / "freeze.json"
    runner.PACKAGE = OUTPUT / "execution-package.json"
    runner.AUTHORITY = OUTPUT / "execution-authority.json"
    runner.LIVE = OUTPUT / "live"
    runner.RESULT = runner.LIVE / "result.json"
    runner.FAILURE = runner.LIVE / "failure.json"
    runner.SUMS = runner.LIVE / "SHA256SUMS"
    runner.STATE = OUTPUT / "supervisor-state"
    runner.STDOUT = OUTPUT / "execution.stdout.log"
    runner.STDERR = OUTPUT / "execution.stderr.log"
    runner.TERMINAL = TERMINAL
    runner.DEFAULT_PROMPTS = (
        ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0005.json"
    )
    runner.TEMPLATE_GATE = (
        ROOT
        / "build/stage1-layer23-v-rank1-hybrid-v1/private"
        / "package-template-invariant-0005.json"
    )
    runner.DEPENDENCY_PROBE = (
        ROOT
        / "build/stage1-layer23-v-rank1-hybrid-v1/private"
        / "dependency-probe-0005.json"
    )
    runner.PREDECESSOR_TERMINAL_SEALS = {
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
        "nonofficial-hybrid-0003": {
            "sha256": "defc1c23ce2c85548a8f95971c18f0f2a9c155a8816ce0078060203ec746e50f",
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0004": {
            "sha256": "c4ec89eaddfdbb05e5efcd91c6e5ce54cb1bc193be02732c0da4cc3ccb6b6917",
            "process_start_count": 0,
            "process_started": False,
        },
    }
    runner.__file__ = str(LAUNCHER)

    def dependency_probe_0005() -> int:
        runner.require(
            not runner.DEPENDENCY_PROBE.exists(), "dependency probe already exists"
        )
        runner.require(
            not runner.TEMPLATE_GATE.exists(), "package-template invariant already exists"
        )
        runner.require(
            not runner.OUTPUT.exists(),
            f"{runner.ATTEMPT_ID} output namespace already exists",
        )
        runner.verify_predecessor_seals()
        runner.verify_exact_interpreter()
        runner.require(
            PRE_IMPORT_ENVIRONMENT == runner.child_environment(),
            "pristine pre-import environment differs from the complete frozen child environment",
        )

        snapshot, model, tokenizer_json, tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        resolved_files = {
            "model": model,
            "adapter": runner.ADAPTER.resolve(strict=True),
            "tokenizer_json": tokenizer_json,
            "tokenizer_config": tokenizer_config,
        }
        for label, path in resolved_files.items():
            runner.require_resolved_regular_file(path, label)

        import_chain = [
            ("tools.run_stage1_layer23_v_rank1_hybrid", runner),
            ("peft", runner.peft),
            ("numpy", runner.np),
            ("safetensors", runner.safetensors),
            ("torch", runner.torch),
            ("transformers", runner.transformers),
            ("tools.ace2_checkpoint176_hidden_state_localizer", runner.localizer),
            ("tools.ace2_exact_once_runtime_supervisor", runner.supervisor),
            ("tools.rtl_arbitrary_text_generation_backend", runner.backend),
            ("tools.run_rtl_arbitrary_text_generation", runner.generation_runner),
            ("tools.run_stage1_layer23_v_rank1_integer_candidate", runner.candidate),
        ]
        chain_records = []
        for name, module in import_chain:
            module_path = (
                IMPLEMENTATION
                if module is runner
                else Path(module.__file__).resolve(strict=True)
            )
            runner.require(
                stat.S_ISREG(os.lstat(module_path).st_mode),
                f"top-level import is not backed by a regular file: {name}",
            )
            chain_records.append(
                {"module": name, "file": runner.file_record(module_path)}
            )

        paths = {
            "output_namespace": runner.OUTPUT,
            "package": runner.PACKAGE,
            "state": runner.STATE,
            "live_output": runner.LIVE,
            "stdout": runner.STDOUT,
            "stderr": runner.STDERR,
            "terminal": runner.TERMINAL,
        }
        runner.require(
            len({str(path) for path in paths.values()}) == len(paths),
            "package/state/output paths are not distinct",
        )
        delta = _environment_delta(
            PRE_IMPORT_ENVIRONMENT, post_import_environment
        )
        record = {
            "schema_version": 2,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_NONCONSUMING_EXACT_CHILD_DEPENDENCY_PROBE",
            "created_at_utc": runner.utc_now(),
            "interpreter": runner.file_record(EXPECTED_PYTHON),
            "launcher": runner.file_record(LAUNCHER),
            "runner": runner.file_record(IMPLEMENTATION),
            "cwd": PRE_IMPORT_CWD,
            "argv": PRE_IMPORT_ARGV,
            "argv_sha256": runner.sha256_bytes(
                runner.canonical_bytes(PRE_IMPORT_ARGV)
            ),
            "environment": PRE_IMPORT_ENVIRONMENT,
            "environment_keys": sorted(PRE_IMPORT_ENVIRONMENT),
            "environment_sha256": runner.sha256_bytes(
                runner.canonical_bytes(PRE_IMPORT_ENVIRONMENT)
            ),
            "pre_import_runtime": {
                "sys_path": PRE_IMPORT_SYS_PATH,
                "sys_prefix": PRE_IMPORT_SYS_PREFIX,
                "loaded_module_names": sorted(PRE_IMPORT_MODULES),
            },
            "post_import_runtime": {
                "sys_path": list(sys.path),
                "sys_prefix": sys.prefix,
            },
            "post_import_environment": post_import_environment,
            "post_import_environment_keys": sorted(post_import_environment),
            "import_side_effects": {
                "classification": "post_import_delta_not_launch_binding",
                "environment_delta": delta,
            },
            "imports": {
                "peft": str(runner.peft.__version__),
                "torch": str(runner.torch.__version__),
                "transformers": str(runner.transformers.__version__),
                "numpy": str(runner.np.__version__),
                "safetensors": str(runner.safetensors.__version__),
            },
            "top_level_import_chain": chain_records,
            "resolved_runtime_inputs": {
                label: runner.file_record(path)
                for label, path in resolved_files.items()
            },
            "snapshot_revision": snapshot.name,
            "path_disjointness": {
                name: runner.public_path(path) for name, path in paths.items()
            },
            "sys_path": list(sys.path),
            "sys_prefix": sys.prefix,
            "checks": {
                "process_not_started": True,
                "output_namespace_absent": True,
                "exact_interpreter_path": True,
                "exact_interpreter_sha256": True,
                "pristine_pre_import_environment_exact": True,
                "all_pre_import_environment_keys_recorded": True,
                "post_import_environment_delta_recorded": True,
                "post_import_delta_not_misclassified_as_launch_mismatch": True,
                "required_third_party_imports_succeeded": True,
                "complete_top_level_import_chain_recorded": True,
                "runner_and_launcher_hashes_recorded": True,
                "resolved_model_tokenizer_bindings_regular": True,
                "package_state_output_paths_distinct": True,
            },
        }
        runner.write_json(runner.DEPENDENCY_PROBE, record)
        print(
            "ACE2_HYBRID_DEPENDENCY_PROBE_PASS "
            f"sha256={runner.sha256_file(runner.DEPENDENCY_PROBE)} "
            "process_start_count=0",
            flush=True,
        )
        return 0

    runner.dependency_probe = dependency_probe_0005


def main() -> int:
    try:
        _verify_pre_import_launch()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid as runner

        post_import_environment = dict(os.environ)
        _configure_runner(runner, post_import_environment)
        return runner.main()
    except BaseException as error:
        _seal_preloader_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
