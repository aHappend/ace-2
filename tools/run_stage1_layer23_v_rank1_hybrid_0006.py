#!/usr/bin/env python3
"""Pristine-environment preloader for nonofficial hybrid attempt 0006."""

from __future__ import annotations

import argparse
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
    / "nonofficial-hybrid-0006"
)
TERMINAL = OUTPUT / "terminal-record.json"
TOKENIZATION_PREFLIGHT = (
    ROOT
    / "build/stage1-layer23-v-rank1-hybrid-v1/private"
    / "exact-tokenization-preflight-0006.json"
)
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
MAX_CONTEXT_TOKENS = 40

# Capture the Python launch state before any dependency or project import.
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
        "mission_id": "stage1rank1hybrid06",
        "attempt_identity": "nonofficial-hybrid-0006",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": _utc_now(),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_exact_tokenization",
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
    runner.MISSION_ID = "stage1rank1hybrid06"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0006"
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
        ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0006.json"
    )
    runner.TEMPLATE_GATE = (
        ROOT
        / "build/stage1-layer23-v-rank1-hybrid-v1/private"
        / "package-template-invariant-0006.json"
    )
    runner.DEPENDENCY_PROBE = (
        ROOT
        / "build/stage1-layer23-v-rank1-hybrid-v1/private"
        / "dependency-probe-0006.json"
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
        "nonofficial-hybrid-0005": {
            "sha256": "8be6597ec736cf469fa898576801719f288cf752c25c1dd8d526d785ab113cf6",
            "process_start_count": 1,
            "process_started": True,
        },
    }
    runner.__file__ = str(LAUNCHER)

    original_verify_predecessor_seals = runner.verify_predecessor_seals
    original_source_records = runner.source_records
    original_execution_bound_paths = runner.execution_bound_paths
    original_package_template_invariant = runner.package_template_invariant
    original_preflight = runner.preflight
    original_freeze = runner.freeze
    original_execute = runner.execute
    original_run_sequence = runner.run_sequence
    original_verify = runner.verify

    def verify_predecessor_seals_0006() -> None:
        original_verify_predecessor_seals()
        failure_path = (
            OUTPUT.parent / "nonofficial-hybrid-0005/live/failure.json"
        )
        runner.require(
            runner.sha256_file(failure_path)
            == "390569ddaa36949c90e3e665746b7a9efd57d217cbf035158d719e6fb31e3738",
            "nonofficial-hybrid-0005 failure seal changed",
        )
        failure = runner.read_json(failure_path)
        runner.require(
            failure.get("root_cause_hypothesis")
            == "BackendError: attention context exceeds accepted RTL core bound",
            "nonofficial-hybrid-0005 failure reason changed",
        )

    runner.verify_predecessor_seals = verify_predecessor_seals_0006

    def dependency_probe_0006() -> int:
        runner.require(
            not runner.DEPENDENCY_PROBE.exists(), "dependency probe already exists"
        )
        runner.require(
            not TOKENIZATION_PREFLIGHT.exists(),
            "exact-tokenization preflight already exists",
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
            "exact_tokenization_preflight": TOKENIZATION_PREFLIGHT,
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
                "exact_tokenization_preflight_absent": True,
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
                "predecessor_0005_terminal_and_failure_seals_immutable": True,
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

    runner.dependency_probe = dependency_probe_0006

    def exact_tokenization_payload(prompts_path: Path) -> dict[str, Any]:
        runner.verify_exact_interpreter()
        runner.verify_dependency_probe()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        runner.require(
            runner.backend.MAX_CONTEXT_TOKENS == MAX_CONTEXT_TOKENS,
            "accepted RTL context bound changed",
        )
        resolved_prompts = prompts_path.resolve(strict=True)
        runner.require_resolved_regular_file(resolved_prompts, "private prompt")
        prompt_hashes = runner.prompt_records(resolved_prompts)
        snapshot, _model, tokenizer_json, tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        for label, path in (
            ("tokenizer JSON", tokenizer_json),
            ("tokenizer config", tokenizer_config),
        ):
            runner.require_resolved_regular_file(path, label)
        tokenizer_config_value = runner.read_json(tokenizer_config)
        configured_template = tokenizer_config_value.get("chat_template")
        runner.require(
            isinstance(configured_template, str) and bool(configured_template),
            "resolved tokenizer chat template is absent",
        )
        tokenizer = runner.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        runner.require(tokenizer.is_fast is True, "final tokenizer is not fast")
        runner.require(
            tokenizer.chat_template == configured_template,
            "AutoTokenizer chat template differs from resolved tokenizer config",
        )
        prompt_records = []
        for prompt, prompt_hash in zip(
            runner.load_prompts(resolved_prompts), prompt_hashes, strict=True
        ):
            token_ids = runner.generation_runner.canonical_chat_token_ids(
                tokenizer, prompt["text"]
            )
            runner.require(bool(token_ids), "tokenizer produced an empty prompt")
            prompt_token_count = len(token_ids)
            total_context_token_count = prompt_token_count + runner.STEPS
            maximum_processed_position = total_context_token_count - 1
            runner.require(
                total_context_token_count <= MAX_CONTEXT_TOKENS,
                "exact final prompt tokenization exceeds accepted RTL context bound",
            )
            runner.require(
                maximum_processed_position < MAX_CONTEXT_TOKENS,
                "exact final prompt processed position exceeds accepted RTL context bound",
            )
            prompt_records.append(
                {
                    **prompt_hash,
                    "prompt_token_ids": token_ids,
                    "prompt_token_count": prompt_token_count,
                    "generation_count": runner.STEPS,
                    "total_context_token_count": total_context_token_count,
                    "maximum_processed_position": maximum_processed_position,
                    "checks": {
                        "prompt_plus_generation_within_context_bound": True,
                        "maximum_processed_position_within_context_bound": True,
                    },
                }
            )
        return {
            "prompts": prompt_records,
            "tokenizer": {
                "snapshot_revision": snapshot.name,
                "auto_tokenizer": {
                    "class": type(tokenizer).__name__,
                    "use_fast": True,
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                "resolved_files": {
                    "tokenizer_json": runner.file_record(tokenizer_json),
                    "tokenizer_config": runner.file_record(tokenizer_config),
                },
                "chat_template_utf8_sha256": runner.sha256_bytes(
                    configured_template.encode("utf-8")
                ),
                "canonical_chat_token_ids_source": runner.file_record(
                    Path(runner.generation_runner.__file__).resolve(strict=True)
                ),
                "apply_policy": "single user message plus tokenizer generation prompt",
            },
            "context_contract": {
                "max_context_tokens": MAX_CONTEXT_TOKENS,
                "generation_count": runner.STEPS,
                "total_context_formula": "len(prompt_token_ids) + generation_count",
                "maximum_processed_position_formula": "len(prompt_token_ids) + generation_count - 1",
            },
        }

    def tokenization_preflight_0006(prompts_path: Path) -> int:
        runner.require(
            not TOKENIZATION_PREFLIGHT.exists(),
            "exact-tokenization preflight already exists",
        )
        runner.require(
            not runner.TEMPLATE_GATE.exists(),
            "package-template invariant already exists before exact tokenization",
        )
        runner.require(
            not runner.OUTPUT.exists(),
            f"{runner.ATTEMPT_ID} output namespace already exists before exact tokenization",
        )
        payload = exact_tokenization_payload(prompts_path)
        unseen = runner.verify_unseen_prompt_hashes(
            runner.prompt_records(prompts_path)
        )
        record = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_NONCONSUMING_EXACT_TOKENIZATION_PREFLIGHT",
            "created_at_utc": runner.utc_now(),
            **payload,
            "unseen_check": unseen,
            "checks": {
                "model_executed": False,
                "simulator_executed": False,
                "icarus_compile_executed": False,
                "process_not_started": True,
                "output_namespace_absent": True,
                "exact_final_auto_tokenizer_path_executed": True,
                "canonical_chat_token_ids_path_executed": True,
                "resolved_tokenizer_files_bound": True,
                "chat_template_hash_bound": True,
                "exactly_two_new_unseen_prompts": True,
                "generation_count_is_four": True,
                "all_total_context_counts_at_most_40": True,
                "all_maximum_processed_positions_below_40": True,
                "accepted_rtl_context_bound_unchanged": True,
                "predecessor_0005_terminal_and_failure_seals_immutable": True,
            },
        }
        runner.write_json(TOKENIZATION_PREFLIGHT, record)
        print(
            "ACE2_HYBRID_EXACT_TOKENIZATION_PREFLIGHT_PASS "
            f"sha256={runner.sha256_file(TOKENIZATION_PREFLIGHT)} "
            "process_start_count=0",
            flush=True,
        )
        return 0

    runner.tokenization_preflight = tokenization_preflight_0006

    def verify_tokenization_preflight(prompts_path: Path) -> dict[str, Any]:
        runner.require(
            TOKENIZATION_PREFLIGHT.is_file(),
            "exact-tokenization preflight is absent",
        )
        record = runner.read_json(TOKENIZATION_PREFLIGHT)
        runner.require(
            record.get("mission_id") == runner.MISSION_ID
            and record.get("attempt_identity") == runner.ATTEMPT_ID
            and record.get("status")
            == "PASS_NONCONSUMING_EXACT_TOKENIZATION_PREFLIGHT",
            "exact-tokenization preflight identity/status changed",
        )
        runner.require(
            all(value is True for value in record["checks"].values()),
            "exact-tokenization preflight predicate changed",
        )
        payload = exact_tokenization_payload(prompts_path)
        for key in ("prompts", "tokenizer", "context_contract"):
            runner.require(
                record[key] == payload[key],
                f"exact-tokenization preflight binding changed: {key}",
            )
        runner.require(
            record["unseen_check"]["records"]
            == [
                {
                    "prompt_id": prompt["prompt_id"],
                    "sha256": prompt["sha256"],
                    "prior_hash_matches": 0,
                }
                for prompt in payload["prompts"]
            ],
            "exact-tokenization unseen-prompt certificate changed",
        )
        return record

    runner.verify_tokenization_preflight = verify_tokenization_preflight

    def source_records_0006(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = original_source_records(
            model, tokenizer_json, tokenizer_config, prompts
        )
        records[runner.public_path(TOKENIZATION_PREFLIGHT)] = runner.file_record(
            TOKENIZATION_PREFLIGHT
        )
        return dict(sorted(records.items()))

    runner.source_records = source_records_0006

    def execution_bound_paths_0006(
        prompts_path: Path,
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        *,
        include_frozen_stage_files: bool,
    ) -> list[Path]:
        paths = original_execution_bound_paths(
            prompts_path,
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=include_frozen_stage_files,
        )
        return sorted(
            {*paths, TOKENIZATION_PREFLIGHT.resolve(strict=True)},
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0006

    def package_template_invariant_0006(prompts_path: Path) -> int:
        certificate = verify_tokenization_preflight(prompts_path)
        result = original_package_template_invariant(prompts_path)
        gate = runner.read_json(runner.TEMPLATE_GATE)
        gate["exact_tokenization_preflight"] = runner.file_record(
            TOKENIZATION_PREFLIGHT
        )
        gate["tokenization_context_contract"] = certificate["context_contract"]
        gate["checks"]["exact_tokenization_preflight_passed"] = True
        gate["checks"]["all_prompt_contexts_within_40"] = True
        runner.write_json(runner.TEMPLATE_GATE, gate)
        print(
            "ACE2_HYBRID_PACKAGE_TEMPLATE_TOKENIZATION_BINDING_PASS "
            f"sha256={runner.sha256_file(runner.TEMPLATE_GATE)}",
            flush=True,
        )
        return result

    runner.package_template_invariant = package_template_invariant_0006

    def preflight_0006(prompts_path: Path) -> int:
        certificate = verify_tokenization_preflight(prompts_path)
        result = original_preflight(prompts_path)
        preflight_record = runner.read_json(runner.PREFLIGHT)
        preflight_record["exact_tokenization_preflight"] = runner.file_record(
            TOKENIZATION_PREFLIGHT
        )
        preflight_record["exact_prompt_tokenization"] = certificate["prompts"]
        preflight_record["context_contract"] = certificate["context_contract"]
        preflight_record["checks"]["exact_final_tokenization_preflight_passed"] = True
        preflight_record["checks"]["all_prompt_contexts_within_40"] = True
        runner.write_json(runner.PREFLIGHT, preflight_record)
        print(
            "ACE2_HYBRID_PREFLIGHT_TOKENIZATION_BINDING_PASS "
            f"sha256={runner.sha256_file(runner.PREFLIGHT)}",
            flush=True,
        )
        return result

    runner.preflight = preflight_0006

    def freeze_0006(prompts_path: Path) -> int:
        certificate = verify_tokenization_preflight(prompts_path)
        result = original_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["exact_tokenization_preflight"] = runner.file_record(
            TOKENIZATION_PREFLIGHT
        )
        frozen["tokenizer"]["exact_prompt_tokenization"] = certificate["prompts"]
        frozen["tokenizer"]["context_contract"] = certificate["context_contract"]
        runner.write_json(runner.FREEZE, frozen)
        print(
            "ACE2_HYBRID_FREEZE_TOKENIZATION_BINDING_PASS "
            f"freeze_sha256={runner.sha256_file(runner.FREEZE)}",
            flush=True,
        )
        return result

    runner.freeze = freeze_0006

    def run_sequence_0006(
        prompt_id: str,
        token_ids: list[int],
        tokenizer: Any,
        weights: Any,
        adapter: Any,
        binary: Path,
        mode: str,
    ) -> dict[str, Any]:
        runner.require(
            len(token_ids) + runner.STEPS <= MAX_CONTEXT_TOKENS,
            "attention context exceeds accepted RTL core bound",
        )
        runner.require(
            len(token_ids) + runner.STEPS - 1 < MAX_CONTEXT_TOKENS,
            "attention processed position exceeds accepted RTL core bound",
        )
        return original_run_sequence(
            prompt_id, token_ids, tokenizer, weights, adapter, binary, mode
        )

    runner.run_sequence = run_sequence_0006

    def execute_0006(prompts_path: Path, expected_freeze_sha256: str) -> int:
        certificate = verify_tokenization_preflight(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        runner.require(
            frozen.get("exact_tokenization_preflight")
            == runner.file_record(TOKENIZATION_PREFLIGHT),
            "frozen exact-tokenization preflight binding changed",
        )
        runner.require(
            frozen["tokenizer"].get("exact_prompt_tokenization")
            == certificate["prompts"],
            "frozen exact prompt tokenization changed",
        )
        return original_execute(prompts_path, expected_freeze_sha256)

    runner.execute = execute_0006

    def verify_0006() -> int:
        result = original_verify()
        certificate = verify_tokenization_preflight(runner.DEFAULT_PROMPTS)
        frozen = runner.read_json(runner.FREEZE)
        execution_result = runner.read_json(runner.RESULT)
        runner.require(
            frozen["tokenizer"]["exact_prompt_tokenization"]
            == certificate["prompts"],
            "frozen tokenization certificate changed",
        )
        for certified, observed in zip(
            certificate["prompts"], execution_result["prompts"], strict=True
        ):
            runner.require(
                certified["prompt_id"] == observed["prompt_id"]
                and certified["prompt_token_count"]
                == observed["prompt_token_count"],
                "executed prompt tokenization differs from exact preflight",
            )
            runner.require(
                certified["total_context_token_count"] <= MAX_CONTEXT_TOKENS
                and certified["maximum_processed_position"]
                < MAX_CONTEXT_TOKENS,
                "certified prompt context bound changed",
            )
            runner.require(
                max(
                    step["absolute_position"]
                    for step in observed["rtl_hybrid"]["steps"]
                )
                < certified["maximum_processed_position"],
                "observed RTL processed position exceeds certified maximum",
            )
        print(
            "ACE2_HYBRID_EXACT_TOKENIZATION_DECISIVE_VERIFY_PASS "
            f"certificate_sha256={runner.sha256_file(TOKENIZATION_PREFLIGHT)}",
            flush=True,
        )
        return result

    runner.verify = verify_0006


def main() -> int:
    try:
        _verify_pre_import_launch()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid as runner

        post_import_environment = dict(os.environ)
        _configure_runner(runner, post_import_environment)
        if "--tokenization-preflight" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--tokenization-preflight", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_preflight(arguments.prompts.resolve())
        return runner.main()
    except BaseException as error:
        _seal_preloader_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
