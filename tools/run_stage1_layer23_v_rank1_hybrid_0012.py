#!/usr/bin/env python3
"""Fresh-prompt successor launcher for nonofficial hybrid attempt 0012."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
PRIOR_LAUNCHER = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0011.py"
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0012"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0012.json"
SELECTION = PRIVATE / "candidate-selection-0012.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0012.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0012.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0012.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0012.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0012.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0012.json"
SUPERVISOR = ROOT / "tools/ace2_exact_once_runtime_supervisor.py"

PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)

PREDECESSOR_0011_SHA256 = {
    "build/stage1-layer23-v-rank1-hybrid-v1/private/dependency-probe-0011.json": (
        "259f9f859e0af40345405f18a47235710f415d791a24cccbf241a2abec81aa6b"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0011.json": (
        "56fb96e74f557e076fcd4bbf229be7d33c6a6c035cda8dd2ace93dbebb98d281"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0011/terminal-record.json": (
        "950e5f660ddf92d0cb3c8b304995fef3dbdfc07039a289ead41fbf7c459dde5b"
    ),
    "tools/run_stage1_layer23_v_rank1_hybrid_0011.py": (
        "7f8bc65b629cd82c9ce8c805319d77d839a7aca86e27fd3c9128d3fd881e51f7"
    ),
}

HISTORICAL_PROMPT_SHA256 = {
    "prompts.json": "1eaeaf4d8c2bf6f730bc498d879646e74e9f0a3a1f4e50ba4d8e7d8290e565e9",
    "prompts-0002.json": "53b2f79a0683f9436f20ad6a85bec2e79839adaf10386805508eb8bf71a50bd1",
    "prompts-0003.json": "32c8bc206dc6fbfc7bd47fbebf70ad042448f356050d1ff6b1133247e1580068",
    "prompts-0004.json": "298f3b2a4cd2b94273247a8b349b99304d43d3d400dc65c23ad864f7883e9121",
    "prompts-0005.json": "ba2e9a195031615954c9a5173912d3bb00ca44d0265a5d09a0b1bba3d45beca7",
    "prompts-0006.json": "5714eaabc25889a8af751a5844a97797598c148a6ef4fa2c03630bfbeaa4c39d",
    "prompts-0007.json": "61e77834ec049f8db34d0931da65318c74b1ec0a687f81d178f5e77470583d37",
    "prompts-0008.json": "1b65465fde96252d8dbfc0774d12735070c37a9477e312f168d7851adaabe00e",
    "prompts-0009.json": "82238b38e8a723538ee280f4447e956a1d21433f750a378dc12f3d502b0f1752",
    "prompts-0010.json": "aa7a88d40160711209244faeddbf86f7908687b8ebf2cea922f01f6665c7a205",
    "prompts-0011.json": "56fb96e74f557e076fcd4bbf229be7d33c6a6c035cda8dd2ace93dbebb98d281",
}

EXPECTED_DENY_SET_DIGEST = (
    "b655f229cda8395221ab241cdaa5c16bccc5364bf600c7a2a68f0b79f9af806b"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid12",
        "attempt_identity": "nonofficial-hybrid-0012",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_nonconsuming_gate",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "regression": "No retry, replay, or resume; preserve the failed gate artifacts.",
        },
    }
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _configure_prior(prior: Any) -> None:
    prior.LAUNCHER = LAUNCHER
    prior.PRIOR_LAUNCHER = PRIOR_LAUNCHER
    prior.OUTPUT = OUTPUT
    prior.TERMINAL = TERMINAL
    prior.PRIVATE = PRIVATE
    prior.PROMPTS = PROMPTS
    prior.TOKENIZATION_PREFLIGHT = TOKENIZATION_PREFLIGHT
    prior.TOKENIZATION_REGRESSION = TOKENIZATION_REGRESSION
    prior.SCALE_REGRESSION = SCALE_REGRESSION
    prior.TEMPLATE_GATE = TEMPLATE_GATE
    prior.DEPENDENCY_PROBE = DEPENDENCY_PROBE
    prior.LAUNCH_REGRESSION = LAUNCH_REGRESSION
    prior.PRE_IMPORT_ENVIRONMENT = PRE_IMPORT_ENVIRONMENT
    prior.PRE_IMPORT_SYS_PATH = PRE_IMPORT_SYS_PATH
    prior.PRE_IMPORT_SYS_PREFIX = PRE_IMPORT_SYS_PREFIX
    prior.PRE_IMPORT_ARGV = PRE_IMPORT_ARGV
    prior.PRE_IMPORT_CWD = PRE_IMPORT_CWD
    prior.PRE_IMPORT_MODULES = PRE_IMPORT_MODULES


def _configure_runner(
    runner: Any,
    prior: Any,
    prior_base: Any,
    base: Any,
    rank1_reference: Any,
    post_import_environment: dict[str, str],
) -> None:
    prior._configure_prior(prior_base)
    prior._configure_runner(
        runner,
        prior_base,
        base,
        rank1_reference,
        post_import_environment,
    )
    runner.MISSION_ID = "stage1rank1hybrid12"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0012"
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
    runner.DEFAULT_PROMPTS = PROMPTS
    runner.TEMPLATE_GATE = TEMPLATE_GATE
    runner.DEPENDENCY_PROBE = DEPENDENCY_PROBE
    runner.PREDECESSOR_TERMINAL_SEALS = {
        **runner.PREDECESSOR_TERMINAL_SEALS,
        "nonofficial-hybrid-0011": {
            "sha256": PREDECESSOR_0011_SHA256[
                "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
                "nonofficial-hybrid-0011/terminal-record.json"
            ],
            "process_start_count": 0,
            "process_started": False,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_unseen_check = runner.verify_unseen_prompt_hashes
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_freeze = runner.freeze
    prior_verify = runner.verify

    def predecessor_0011_records() -> dict[str, Any]:
        records: dict[str, Any] = {}
        for relative, expected_sha256 in sorted(PREDECESSOR_0011_SHA256.items()):
            path = ROOT / relative
            path_stat = os.lstat(path)
            runner.require(
                stat.S_ISREG(path_stat.st_mode),
                f"0011 evidence is not regular: {relative}",
            )
            runner.require(
                runner.sha256_file(path) == expected_sha256,
                f"sealed 0011 evidence hash changed: {relative}",
            )
            records[relative] = runner.file_record(path)
        terminal = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0011/terminal-record.json"
        )
        runner.require(
            terminal.get("outcome") == "pre_execution_failure"
            and terminal.get("process_start_count") == 0
            and terminal.get("process_started") is False,
            "sealed 0011 zero-start terminal state changed",
        )
        return records

    def historical_prompt_deny_set() -> tuple[set[str], set[str], str]:
        prompt_ids: set[str] = set()
        prompt_hashes: set[str] = set()
        for name, expected_sha256 in sorted(HISTORICAL_PROMPT_SHA256.items()):
            path = PRIVATE / name
            runner.require(
                runner.sha256_file(path) == expected_sha256,
                f"sealed historical prompt artifact changed: {name}",
            )
            for prompt in runner.load_prompts(path):
                prompt_ids.add(prompt["id"])
                prompt_hashes.add(
                    runner.sha256_bytes(prompt["text"].encode("utf-8"))
                )
        digest = runner.supervisor.canonical_value_sha256(sorted(prompt_hashes))
        runner.require(
            len(prompt_ids) == 22
            and len(prompt_hashes) == 20
            and digest == EXPECTED_DENY_SET_DIGEST,
            "complete historical prompt deny-set changed",
        )
        return prompt_ids, prompt_hashes, digest

    def verify_selection_certificate() -> dict[str, Any]:
        runner.require(SELECTION.is_file(), "candidate-selection certificate is absent")
        certificate = runner.read_json(SELECTION)
        prompt_ids, prompt_hashes, digest = historical_prompt_deny_set()
        selected = []
        snapshot, _model, _tokenizer_json, _tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        tokenizer = runner.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        for prompt in runner.load_prompts(PROMPTS):
            prompt_hash = runner.sha256_bytes(prompt["text"].encode("utf-8"))
            prompt_token_count = len(
                runner.generation_runner.canonical_chat_token_ids(
                    tokenizer, prompt["text"]
                )
            )
            runner.require(
                prompt["id"] not in prompt_ids
                and prompt_hash not in prompt_hashes
                and prompt_token_count <= 35
                and prompt_token_count + runner.STEPS <= 40,
                f"selected prompt is not fresh and bounded: {prompt['id']}",
            )
            selected.append(
                {
                    **prompt,
                    "sha256": prompt_hash,
                    "prompt_token_count": prompt_token_count,
                    "total_context_token_count": prompt_token_count + runner.STEPS,
                }
            )
        runner.require(
            certificate
            == {
                "schema_version": 1,
                "mission_id": runner.MISSION_ID,
                "attempt_identity": runner.ATTEMPT_ID,
                "status": "PASS_NONCONSUMING_CANDIDATE_SELECTION",
                "method": (
                    "deterministic short natural-language question assembly; exact "
                    "UTF-8 SHA-256; canonical Qwen chat-template tokenization; "
                    "complete sealed-attempt ID/hash rejection"
                ),
                "sealed_attempt_count": 11,
                "deny_prompt_hash_count": len(prompt_hashes),
                "deny_prompt_id_count": len(prompt_ids),
                "deny_set_digest": digest,
                "generated_candidate_count": 6,
                "selected": selected,
            },
            "candidate-selection certificate changed",
        )
        runner.require(
            len({item["id"] for item in selected}) == 2
            and len({item["sha256"] for item in selected}) == 2,
            "selected prompt IDs or hashes are not distinct",
        )
        return certificate

    def verify_predecessor_seals_0012() -> None:
        prior_verify_predecessor_seals()
        predecessor_0011_records()

    runner.verify_predecessor_seals = verify_predecessor_seals_0012

    def verify_unseen_prompt_hashes_0012(
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scan = prior_unseen_check(records)
        prompt_ids, prompt_hashes, digest = historical_prompt_deny_set()
        runner.require(
            all(
                record["prompt_id"] not in prompt_ids
                and record["sha256"] not in prompt_hashes
                for record in records
            ),
            "prompt ID or hash was already present in the complete sealed deny-set",
        )
        return {
            **scan,
            "method": (
                "complete sealed hybrid prompt ID/hash deny-set plus prior artifact "
                "fixed-string SHA-256 scan"
            ),
            "deny_set_digest": digest,
            "deny_prompt_hash_count": len(prompt_hashes),
            "deny_prompt_id_count": len(prompt_ids),
        }

    runner.verify_unseen_prompt_hashes = verify_unseen_prompt_hashes_0012

    def source_records_0012(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        paths = [
            SELECTION,
            *(PRIVATE / name for name in HISTORICAL_PROMPT_SHA256),
            *(ROOT / relative for relative in PREDECESSOR_0011_SHA256),
        ]
        for path in paths:
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0012

    def execution_bound_paths_0012(
        prompts_path: Path,
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        *,
        include_frozen_stage_files: bool,
    ) -> list[Path]:
        paths = prior_execution_bound_paths(
            prompts_path,
            model,
            tokenizer_json,
            tokenizer_config,
            include_frozen_stage_files=include_frozen_stage_files,
        )
        return sorted(
            {
                *paths,
                SELECTION.resolve(strict=True),
                *(PRIVATE / name for name in HISTORICAL_PROMPT_SHA256),
                *(ROOT / relative for relative in PREDECESSOR_0011_SHA256),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0012

    def freeze_0012(prompts_path: Path) -> int:
        verify_selection_certificate()
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["candidate_selection"] = runner.file_record(SELECTION)
        frozen["predecessor_0011_evidence"] = predecessor_0011_records()
        frozen["prompt_history_deny_set"] = {
            "digest": EXPECTED_DENY_SET_DIGEST,
            "hash_count": 20,
            "id_count": 22,
            "sealed_attempt_count": 11,
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0012

    def verify_0012() -> int:
        result = prior_verify()
        verify_selection_certificate()
        execution_result = runner.read_json(runner.RESULT)
        runner.require(
            execution_result["mission_id"] == runner.MISSION_ID
            and execution_result["exactly_once"]["attempt_identity"]
            == runner.ATTEMPT_ID,
            "0012 result identity changed",
        )
        print(
            "ACE2_HYBRID_0012_FRESH_PROMPT_VERIFY_PASS "
            f"selection_sha256={runner.sha256_file(SELECTION)} "
            "prompt_count=2 generated_tokens_per_prompt=4",
            flush=True,
        )
        return result

    runner.verify = verify_0012
    runner.verify_selection_certificate = verify_selection_certificate


def main() -> int:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid_0010 as prior_base
        from tools import run_stage1_layer23_v_rank1_hybrid_0011 as prior

        _configure_prior(prior)
        prior._configure_prior(prior_base)
        prior_base._verify_pre_import_launch()

        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as base

        _configure_runner(
            runner,
            prior,
            prior_base,
            base,
            ace2_layer23_v_rank1_integer_correction_reference,
            dict(os.environ),
        )
        if "--tokenization-preflight" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--tokenization-preflight", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_preflight(arguments.prompts.resolve())
        if "--check-tokenization-verifier" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-tokenization-verifier", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.tokenization_verifier_regression(
                arguments.prompts.resolve()
            )
        if "--check-scale-alignment" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-scale-alignment", action="store_true")
            parser.add_argument("--prompts", type=Path, default=runner.DEFAULT_PROMPTS)
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.scale_regression(arguments.prompts.resolve())
        return runner.main()
    except BaseException as error:
        _seal_preloader_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
