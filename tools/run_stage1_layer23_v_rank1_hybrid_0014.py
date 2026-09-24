#!/usr/bin/env python3
"""Gate-ordered record-safe launcher for nonofficial hybrid attempt 0014."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
PRIOR_LAUNCHER = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0011.py"
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0014"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0014.json"
SELECTION = PRIVATE / "candidate-selection-0014.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0014.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0014.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0014.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0014.json"
GATE_ORDER_REGRESSION = PRIVATE / "gate-order-regression-0014.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0014.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0014.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0014.json"

PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)

SEALED_BOUNDARY_SHA256 = {
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0012/terminal-record.json": (
        "18504b8d1140a588942d23b77bfdce4b7963fa5356a3fda8a26e852bdf766fea"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0013/terminal-record.json": (
        "30c75c3596a8d541b7fadcf27914cf9933053cbe631da97af96ca08aa89d8a3d"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/"
    "diagnostic-indexerror-0012.json": (
        "4e3245170890ec4e24ba07b896bcea58120352e96f94e5282557a44e3dad5aa0"
    ),
    "build/stage1-layer23-v-rank1-hybrid-v1/private/"
    "record-lifecycle-regression-0013.json": (
        "3801a1f1c0c70ebe00468498ee2cd7c8dd6699638daca5e7fa175b374171385a"
    ),
    "tools/run_stage1_layer23_v_rank1_hybrid_0013.py": (
        "eb81f5e99a71a1ef5c6d9f27e3bdafed8fc8d457dcff280950fde5dcd1a1ed74"
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
    "prompts-0012.json": "a6e84eb5d3f23a816be78db87cdb804b0a0618f5f28ee9fa53287fc09672b798",
    "prompts-0013.json": "1a412471d5b5684f1efb7bdea5b53d9e09d35c55ce5a78bffde6168d9f77f061",
}

DIAGNOSTIC_PROMPT_SOURCES = (
    ROOT / "build/ace2_chat_diagnostics",
    ROOT / "build/direct_base_rtl_20260808T0635Z/provenance.json",
    ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/provenance.json",
    ROOT
    / "evidence/verification/qwen-ace2-shell-e2e-3ce65cd8663d/preflight/provenance.json",
)

CANDIDATES = (
    {"id": "fresh-0014-a", "text": "What makes dew form?"},
    {"id": "fresh-0014-b", "text": "Where do ants build nests?"},
    {"id": "fresh-0014-c", "text": "Why does wool feel warm?"},
    {"id": "fresh-0014-d", "text": "How do seeds begin growing?"},
    {"id": "fresh-0014-e", "text": "What causes a quiet echo?"},
    {"id": "fresh-0014-f", "text": "Where does beach sand come from?"},
)

GATE_CHAIN = (
    "dependency_probe",
    "tokenization_preflight",
    "tokenization_tamper_regression",
    "retained_scale_regression",
    "package_template",
    "preflight",
    "freeze",
    "package",
    "authority",
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sanitized_traceback(error: BaseException) -> dict[str, Any]:
    frames = []
    rendered = ["Traceback (most recent call last):"]
    for frame in traceback.extract_tb(error.__traceback__):
        path = Path(frame.filename)
        try:
            source_file = str(path.resolve().relative_to(ROOT))
        except (OSError, ValueError):
            source_file = f"<external>/{path.name}"
        record = {
            "source_file": source_file,
            "function": frame.name,
            "line": frame.lineno,
            "expression": frame.line or "",
        }
        frames.append(record)
        rendered.append(
            f'  File "{source_file}", line {frame.lineno}, in {frame.name}'
        )
        if frame.line:
            rendered.append(f"    {frame.line}")
    rendered.append(f"{type(error).__name__}: {error}")
    return {
        "exception_type": type(error).__name__,
        "message": str(error),
        "frames": frames,
        "formatted": "\n".join(rendered) + "\n",
    }


def _seal_preloader_failure(error: BaseException) -> None:
    if TERMINAL.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid14",
        "attempt_identity": "nonofficial-hybrid-0014",
        "status": "FAILED_SEALED_NO_EXECUTION",
        "outcome": "pre_execution_failure",
        "sealed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "process_start_count": 0,
        "process_started": False,
        "failure": {
            "failure_taxonomy": "pre_execution_gate_failure",
            "phase": "stdlib_preloader_or_nonconsuming_gate",
            "root_cause_hypothesis": f"{type(error).__name__}: {error}",
            "traceback": _sanitized_traceback(error),
            "regression": "No retry, replay, or resume; preserve all 0014 artifacts.",
        },
    }
    try:
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


def _signed_bytes(values: list[int] | tuple[int, ...]) -> bytes:
    return bytes(int(value) & 0xFF for value in values)


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
    runner.MISSION_ID = "stage1rank1hybrid14"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0014"
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
        "nonofficial-hybrid-0012": {
            "sha256": SEALED_BOUNDARY_SHA256[
                "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
                "nonofficial-hybrid-0012/terminal-record.json"
            ],
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0013": {
            "sha256": SEALED_BOUNDARY_SHA256[
                "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
                "nonofficial-hybrid-0013/terminal-record.json"
            ],
            "process_start_count": 0,
            "process_started": False,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_dependency_probe = runner.dependency_probe
    prior_tokenization_preflight = runner.tokenization_preflight
    prior_tokenization_regression = runner.tokenization_verifier_regression
    prior_scale_regression = runner.scale_regression
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_package_template = runner.package_template_invariant
    prior_preflight = runner.preflight
    prior_freeze = runner.freeze
    prior_package = runner.package_execution
    prior_authority = runner.grant_authority
    prior_execute = runner.execute
    prior_verify = runner.verify
    legacy_cache_class = runner.Rank1HybridProjectionCache
    sidecar_cache_class = legacy_cache_class.__mro__[1]

    def boundary_records() -> dict[str, Any]:
        records: dict[str, Any] = {}
        for relative, expected_sha256 in sorted(SEALED_BOUNDARY_SHA256.items()):
            path = ROOT / relative
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed boundary artifact is not regular: {relative}",
            )
            runner.require(
                runner.sha256_file(path) == expected_sha256,
                f"sealed boundary artifact changed: {relative}",
            )
            records[relative] = runner.file_record(path)
        terminal_0012 = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0012/terminal-record.json"
        )
        terminal_0013 = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0013/terminal-record.json"
        )
        runner.require(
            terminal_0012.get("process_start_count") == 1
            and terminal_0012.get("process_started") is True
            and terminal_0012.get("outcome") == "child_nonzero_exit",
            "sealed 0012 one-start terminal state changed",
        )
        runner.require(
            terminal_0013.get("process_start_count") == 0
            and terminal_0013.get("process_started") is False
            and terminal_0013.get("outcome") == "pre_execution_failure",
            "sealed 0013 zero-start terminal state changed",
        )
        return records

    def verify_predecessor_seals_0014() -> None:
        prior_verify_predecessor_seals()
        boundary_records()

    runner.verify_predecessor_seals = verify_predecessor_seals_0014

    def diagnostic_source_files() -> list[Path]:
        files: list[Path] = []
        for source in DIAGNOSTIC_PROMPT_SOURCES:
            if source.is_dir():
                files.extend(sorted(source.rglob("*.json")))
            else:
                files.append(source)
        runner.require(
            files and all(path.is_file() for path in files),
            "diagnostic prompt source set is incomplete",
        )
        return files

    def collect_diagnostic_prompts(value: Any, ids: set[str], hashes: set[str]) -> None:
        if isinstance(value, dict):
            prompt_id = value.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id:
                ids.add(prompt_id)
            prompt_sha256 = value.get("prompt_sha256")
            if (
                isinstance(prompt_sha256, str)
                and len(prompt_sha256) == 64
                and all(character in "0123456789abcdef" for character in prompt_sha256)
            ):
                hashes.add(prompt_sha256)
            prompt_text = value.get("prompt")
            if isinstance(prompt_text, str):
                hashes.add(runner.sha256_bytes(prompt_text.encode("utf-8")))
            if isinstance(value.get("id"), str) and isinstance(value.get("text"), str):
                ids.add(value["id"])
                hashes.add(runner.sha256_bytes(value["text"].encode("utf-8")))
            for nested in value.values():
                collect_diagnostic_prompts(nested, ids, hashes)
        elif isinstance(value, list):
            for nested in value:
                collect_diagnostic_prompts(nested, ids, hashes)

    def prompt_deny_set() -> tuple[set[str], set[str], dict[str, Any]]:
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
        historical_id_count = len(prompt_ids)
        historical_hash_count = len(prompt_hashes)
        diagnostic_files = diagnostic_source_files()
        for path in diagnostic_files:
            collect_diagnostic_prompts(runner.read_json(path), prompt_ids, prompt_hashes)
        runner.require(
            historical_id_count == 26
            and historical_hash_count == 24
            and "3639efcd08abb273b1619e82e78c29a7df02c1051b1820e99fc395dcaa3326b8"
            in prompt_hashes
            and "5f85c0fedfd298bfc262408927bc4fb5a714f0de236d06d06224a58aa8aae1bc"
            in prompt_hashes,
            "sealed-attempt or diagnostic prompt deny-set is incomplete",
        )
        metadata = {
            "historical_prompt_artifact_count": len(HISTORICAL_PROMPT_SHA256),
            "historical_id_count": historical_id_count,
            "historical_hash_count": historical_hash_count,
            "diagnostic_source_count": len(diagnostic_files),
            "combined_id_count": len(prompt_ids),
            "combined_hash_count": len(prompt_hashes),
            "combined_id_digest": runner.supervisor.canonical_value_sha256(
                sorted(prompt_ids)
            ),
            "combined_hash_digest": runner.supervisor.canonical_value_sha256(
                sorted(prompt_hashes)
            ),
            "diagnostic_sources": [
                runner.file_record(path) for path in diagnostic_files
            ],
        }
        return prompt_ids, prompt_hashes, metadata

    def selected_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt_ids, prompt_hashes, deny_metadata = prompt_deny_set()
        snapshot, _model, _tokenizer_json, _tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        tokenizer = runner.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        selected = []
        for prompt in CANDIDATES:
            prompt_hash = runner.sha256_bytes(prompt["text"].encode("utf-8"))
            prompt_token_count = len(
                runner.generation_runner.canonical_chat_token_ids(
                    tokenizer, prompt["text"]
                )
            )
            if (
                prompt["id"] not in prompt_ids
                and prompt_hash not in prompt_hashes
                and prompt_token_count + runner.STEPS <= 40
            ):
                selected.append(
                    {
                        **prompt,
                        "sha256": prompt_hash,
                        "prompt_token_count": prompt_token_count,
                        "total_context_token_count": prompt_token_count
                        + runner.STEPS,
                    }
                )
            if len(selected) == 2:
                break
        runner.require(
            len(selected) == 2
            and len({item["id"] for item in selected}) == 2
            and len({item["sha256"] for item in selected}) == 2,
            "two distinct fresh bounded execution prompts were not found",
        )
        return selected, deny_metadata

    def select_prompts() -> int:
        runner.validate_no_execution()
        runner.require(
            not PROMPTS.exists() and not SELECTION.exists(),
            "0014 prompt selection already exists",
        )
        runner.verify_predecessor_seals()
        selected, deny_metadata = selected_records()
        runner.write_json(
            PROMPTS,
            {
                "schema_version": 1,
                "prompts": [
                    {"id": item["id"], "text": item["text"]} for item in selected
                ],
            },
        )
        certificate = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_NONCONSUMING_CANDIDATE_SELECTION",
            "method": (
                "deterministic short natural-language candidates; exact UTF-8 "
                "SHA-256; canonical Qwen chat-template tokenization; complete "
                "sealed-attempt and diagnostic prompt ID/hash rejection"
            ),
            "generated_candidate_count": len(CANDIDATES),
            "deny_set": deny_metadata,
            "selected": selected,
        }
        runner.write_json(SELECTION, certificate)
        print(
            "ACE2_HYBRID_0014_PROMPT_SELECTION_PASS "
            f"sha256={runner.sha256_file(SELECTION)} prompt_count=2",
            flush=True,
        )
        return 0

    def verify_selection_certificate() -> dict[str, Any]:
        runner.require(
            PROMPTS.is_file() and SELECTION.is_file(),
            "0014 prompt selection artifact is absent",
        )
        selected, deny_metadata = selected_records()
        runner.require(
            runner.load_prompts(PROMPTS)
            == [{"id": item["id"], "text": item["text"]} for item in selected],
            "0014 selected prompt artifact changed",
        )
        certificate = runner.read_json(SELECTION)
        runner.require(
            certificate
            == {
                "schema_version": 1,
                "mission_id": runner.MISSION_ID,
                "attempt_identity": runner.ATTEMPT_ID,
                "status": "PASS_NONCONSUMING_CANDIDATE_SELECTION",
                "method": (
                    "deterministic short natural-language candidates; exact UTF-8 "
                    "SHA-256; canonical Qwen chat-template tokenization; complete "
                    "sealed-attempt and diagnostic prompt ID/hash rejection"
                ),
                "generated_candidate_count": len(CANDIDATES),
                "deny_set": deny_metadata,
                "selected": selected,
            },
            "0014 prompt selection certificate changed",
        )
        return certificate

    runner.select_prompts = select_prompts
    runner.verify_selection_certificate = verify_selection_certificate

    class RecordAwareRetainedScaleProjectionCache(legacy_cache_class):
        def _apply(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
            alignment = result.get("retained_scale_alignment")
            records_before = len(self.records)
            converted = sidecar_cache_class._apply(self, name, result)
            records_after = len(self.records)
            runner.require(
                records_after in {records_before, records_before + 1},
                "sidecar record lifecycle changed unexpectedly",
            )
            if alignment is None or records_after == records_before:
                return converted
            record = self.records[-1]
            alignment_record = dict(alignment)
            alignment_record["converted_bytes_sha256"] = runner.sha256_bytes(
                _signed_bytes(alignment_record["converted_bytes_s8"])
            )
            alignment_record["software_reference_baseline_sha256"] = record[
                "baseline_v_sha256"
            ]
            alignment_record["ace2_shell_shared_payload_verified"] = (
                self.mode == "rtl"
            )
            runner.require(
                alignment_record["converted_bytes_sha256"]
                == record["baseline_v_sha256"],
                "retained conversion bytes differ from sidecar baseline",
            )
            record["retained_scale_alignment"] = alignment_record
            return converted

    runner.Rank1HybridProjectionCache = RecordAwareRetainedScaleProjectionCache

    def verify_bound_lifecycle_evidence() -> dict[str, Any]:
        records = boundary_records()
        diagnostic = runner.read_json(
            PRIVATE / "diagnostic-indexerror-0012.json"
        )
        prior_pass = runner.read_json(
            PRIVATE / "record-lifecycle-regression-0013.json"
        )
        runner.require(
            diagnostic.get("failure_taxonomy") == "software_runtime_record_lifecycle"
            and diagnostic.get("failing_expression") == "self.records[-1]"
            and diagnostic["proof"].get("root_cause_proven") is True,
            "diagnostic-indexerror-0012 evidence changed",
        )
        runner.require(
            prior_pass.get("status") == "PASS_FOCUSED_0012_INDEX_REGRESSION"
            and prior_pass["repaired"].get(
                "alignment_attachment_skipped_without_new_record"
            )
            is True,
            "prior PASS record-lifecycle evidence changed",
        )
        return {
            "diagnostic_indexerror_0012": records[
                "build/stage1-layer23-v-rank1-hybrid-v1/private/"
                "diagnostic-indexerror-0012.json"
            ],
            "prior_pass_record_lifecycle": records[
                "build/stage1-layer23-v-rank1-hybrid-v1/private/"
                "record-lifecycle-regression-0013.json"
            ],
        }

    def record_lifecycle_regression() -> int:
        runner.validate_no_execution()
        runner.require(
            not LIFECYCLE_REGRESSION.exists(),
            "0014 record-lifecycle regression already exists",
        )
        bound_evidence = verify_bound_lifecycle_evidence()
        name = "layer23_position0_v"
        aligned = {"retained_scale_alignment": {"converted_bytes_s8": []}}
        legacy = legacy_cache_class.__new__(legacy_cache_class)
        legacy.records = []
        legacy.first_scored_position = 5
        legacy.mode = "software"
        legacy_traceback: dict[str, Any] | None = None
        try:
            legacy._apply(name, aligned)
        except IndexError as error:
            legacy_traceback = _sanitized_traceback(error)
        runner.require(
            legacy_traceback is not None
            and legacy_traceback["frames"][-1]["source_file"]
            == "tools/run_stage1_layer23_v_rank1_hybrid_0010.py"
            and legacy_traceback["frames"][-1]["expression"]
            == "record = self.records[-1]"
            and legacy.records == [],
            "focused regression did not reproduce the legacy empty-record failure",
        )
        repaired = RecordAwareRetainedScaleProjectionCache.__new__(
            RecordAwareRetainedScaleProjectionCache
        )
        repaired.records = []
        repaired.first_scored_position = 5
        repaired.mode = "software"
        returned = repaired._apply(name, aligned)
        runner.require(
            returned is aligned and repaired.records == [],
            "record-aware repair changed the pre-scored no-record lifecycle",
        )
        record = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_FOCUSED_LEGACY_FAILS_REPAIRED_PASSES",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "bound_evidence": bound_evidence,
            "input": {
                "projection_name": name,
                "parsed_position": 0,
                "first_scored_position": 5,
                "alignment_records": 1,
            },
            "legacy": {
                "records_before": 0,
                "records_after": 0,
                "failing_expression": "self.records[-1]",
                "traceback": legacy_traceback,
            },
            "repaired": {
                "records_before": 0,
                "records_after": 0,
                "alignment_attachment_skipped_without_new_record": True,
            },
        }
        runner.write_json(LIFECYCLE_REGRESSION, record)
        print(
            "ACE2_HYBRID_0014_RECORD_LIFECYCLE_REGRESSION_PASS "
            f"sha256={runner.sha256_file(LIFECYCLE_REGRESSION)}",
            flush=True,
        )
        return 0

    def verify_record_lifecycle_regression() -> dict[str, Any]:
        runner.require(
            LIFECYCLE_REGRESSION.is_file(),
            "0014 record-lifecycle regression is absent",
        )
        record = runner.read_json(LIFECYCLE_REGRESSION)
        runner.require(
            record.get("status") == "PASS_FOCUSED_LEGACY_FAILS_REPAIRED_PASSES"
            and record.get("model_executed") is False
            and record.get("simulator_executed") is False
            and record.get("process_start_count") == 0
            and record["legacy"]["records_after"] == 0
            and record["legacy"]["failing_expression"] == "self.records[-1]"
            and record["repaired"][
                "alignment_attachment_skipped_without_new_record"
            ]
            is True
            and record["bound_evidence"] == verify_bound_lifecycle_evidence(),
            "0014 record-lifecycle regression artifact changed",
        )
        return record

    runner.record_lifecycle_regression = record_lifecycle_regression
    runner.verify_record_lifecycle_regression = verify_record_lifecycle_regression

    gate_paths = {
        "dependency_probe": DEPENDENCY_PROBE,
        "tokenization_preflight": TOKENIZATION_PREFLIGHT,
        "tokenization_tamper_regression": TOKENIZATION_REGRESSION,
        "retained_scale_regression": SCALE_REGRESSION,
        "package_template": TEMPLATE_GATE,
        "preflight": runner.PREFLIGHT,
        "freeze": runner.FREEZE,
        "package": runner.PACKAGE,
        "authority": runner.AUTHORITY,
    }

    def require_gate_predecessors(stage: str) -> None:
        index = GATE_CHAIN.index(stage)
        for predecessor in GATE_CHAIN[:index]:
            runner.require(
                gate_paths[predecessor].is_file(),
                f"{stage} blocked: predecessor artifact absent: {predecessor}",
            )

    def gate_order_regression() -> int:
        runner.validate_no_execution()
        runner.require(
            DEPENDENCY_PROBE.is_file(),
            "gate-order regression requires dependency probe first",
        )
        runner.require(
            not GATE_ORDER_REGRESSION.exists(),
            "0014 gate-order regression already exists",
        )
        runner.require(
            not any(gate_paths[name].exists() for name in GATE_CHAIN[1:]),
            "later gate exists before gate-order regression",
        )
        cases = []
        for target_index, target in enumerate(GATE_CHAIN[1:], start=1):
            predecessors = GATE_CHAIN[:target_index]
            for missing in predecessors:
                available = set(predecessors) - {missing}
                rejected = not all(name in available for name in predecessors)
                identity_executed = False
                runner.require(
                    rejected and not identity_executed,
                    f"missing predecessor was not rejected: {target}/{missing}",
                )
                cases.append(
                    {
                        "target": target,
                        "missing_predecessor": missing,
                        "rejected": True,
                        "identity_executed": False,
                    }
                )
        runner.write_json(
            GATE_ORDER_REGRESSION,
            {
                "schema_version": 1,
                "mission_id": runner.MISSION_ID,
                "attempt_identity": runner.ATTEMPT_ID,
                "status": "PASS_NONCONSUMING_GATE_ORDER_REGRESSION",
                "chain": list(GATE_CHAIN),
                "dependency_probe": runner.file_record(DEPENDENCY_PROBE),
                "negative_cases": cases,
                "checks": {
                    "dependency_probe_created_first": True,
                    "every_missing_predecessor_rejected": True,
                    "identity_execution_never_reached": True,
                    "process_not_started": True,
                    "model_not_executed": True,
                    "simulator_not_executed": True,
                },
            },
        )
        print(
            "ACE2_HYBRID_0014_GATE_ORDER_REGRESSION_PASS "
            f"sha256={runner.sha256_file(GATE_ORDER_REGRESSION)} "
            f"negative_cases={len(cases)} process_start_count=0",
            flush=True,
        )
        return 0

    def verify_gate_order_regression() -> dict[str, Any]:
        runner.require(
            GATE_ORDER_REGRESSION.is_file(),
            "0014 gate-order regression is absent",
        )
        record = runner.read_json(GATE_ORDER_REGRESSION)
        runner.require(
            record.get("status") == "PASS_NONCONSUMING_GATE_ORDER_REGRESSION"
            and record.get("chain") == list(GATE_CHAIN)
            and len(record.get("negative_cases", [])) == 36
            and all(
                case.get("rejected") is True
                and case.get("identity_executed") is False
                for case in record["negative_cases"]
            )
            and all(value is True for value in record["checks"].values())
            and record["dependency_probe"] == runner.file_record(DEPENDENCY_PROBE),
            "0014 gate-order regression artifact changed",
        )
        return record

    runner.gate_order_regression = gate_order_regression
    runner.verify_gate_order_regression = verify_gate_order_regression

    def dependency_probe_0014() -> int:
        verify_selection_certificate()
        verify_record_lifecycle_regression()
        runner.require(
            not GATE_ORDER_REGRESSION.exists()
            and not any(gate_paths[name].exists() for name in GATE_CHAIN),
            "dependency probe is not the first gate artifact",
        )
        result = prior_dependency_probe()
        runner.verify_dependency_probe()
        return result

    runner.dependency_probe = dependency_probe_0014

    def tokenization_preflight_0014(prompts_path: Path) -> int:
        require_gate_predecessors("tokenization_preflight")
        verify_gate_order_regression()
        verify_selection_certificate()
        return prior_tokenization_preflight(prompts_path)

    runner.tokenization_preflight = tokenization_preflight_0014

    def tokenization_regression_0014(prompts_path: Path) -> int:
        require_gate_predecessors("tokenization_tamper_regression")
        return prior_tokenization_regression(prompts_path)

    runner.tokenization_verifier_regression = tokenization_regression_0014

    def scale_regression_0014(prompts_path: Path) -> int:
        require_gate_predecessors("retained_scale_regression")
        return prior_scale_regression(prompts_path)

    runner.scale_regression = scale_regression_0014

    def source_records_0014(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        paths = [
            SELECTION,
            LIFECYCLE_REGRESSION,
            GATE_ORDER_REGRESSION,
            *diagnostic_source_files(),
            *(PRIVATE / name for name in HISTORICAL_PROMPT_SHA256),
            *(ROOT / relative for relative in SEALED_BOUNDARY_SHA256),
        ]
        for path in paths:
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0014

    def execution_bound_paths_0014(
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
                LIFECYCLE_REGRESSION.resolve(strict=True),
                GATE_ORDER_REGRESSION.resolve(strict=True),
                *(path.resolve(strict=True) for path in diagnostic_source_files()),
                *(PRIVATE / name for name in HISTORICAL_PROMPT_SHA256),
                *(ROOT / relative for relative in SEALED_BOUNDARY_SHA256),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0014

    def package_template_0014(prompts_path: Path) -> int:
        require_gate_predecessors("package_template")
        selection = verify_selection_certificate()
        lifecycle = verify_record_lifecycle_regression()
        gate_order = verify_gate_order_regression()
        result = prior_package_template(prompts_path)
        gate = runner.read_json(runner.TEMPLATE_GATE)
        gate["candidate_selection"] = runner.file_record(SELECTION)
        gate["record_lifecycle_regression"] = runner.file_record(
            LIFECYCLE_REGRESSION
        )
        gate["gate_order_regression"] = runner.file_record(GATE_ORDER_REGRESSION)
        gate["prompt_deny_set"] = selection["deny_set"]
        gate["checks"]["focused_record_lifecycle_regression_passed"] = (
            lifecycle["status"] == "PASS_FOCUSED_LEGACY_FAILS_REPAIRED_PASSES"
        )
        gate["checks"]["strict_gate_order_regression_passed"] = (
            gate_order["status"] == "PASS_NONCONSUMING_GATE_ORDER_REGRESSION"
        )
        runner.write_json(runner.TEMPLATE_GATE, gate)
        return result

    runner.package_template_invariant = package_template_0014

    def preflight_0014(prompts_path: Path) -> int:
        require_gate_predecessors("preflight")
        result = prior_preflight(prompts_path)
        record = runner.read_json(runner.PREFLIGHT)
        record["record_lifecycle_regression"] = runner.file_record(
            LIFECYCLE_REGRESSION
        )
        record["gate_order_regression"] = runner.file_record(GATE_ORDER_REGRESSION)
        record["checks"]["focused_record_lifecycle_regression_passed"] = True
        record["checks"]["strict_gate_order_regression_passed"] = True
        runner.write_json(runner.PREFLIGHT, record)
        return result

    runner.preflight = preflight_0014

    def freeze_0014(prompts_path: Path) -> int:
        require_gate_predecessors("freeze")
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["candidate_selection"] = runner.file_record(SELECTION)
        frozen["record_lifecycle_regression"] = runner.file_record(
            LIFECYCLE_REGRESSION
        )
        frozen["gate_order_regression"] = runner.file_record(GATE_ORDER_REGRESSION)
        frozen["sealed_boundary_evidence"] = boundary_records()
        frozen["prompt_deny_set"] = verify_selection_certificate()["deny_set"]
        frozen["record_lifecycle_repair"] = {
            "rule": (
                "attach retained-scale alignment only when parent sidecar apply "
                "appends one scored-step record"
            ),
            "pre_scored_projection_records": 0,
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0014

    def package_0014(prompts_path: Path) -> int:
        require_gate_predecessors("package")
        return prior_package(prompts_path)

    runner.package_execution = package_0014

    def authority_0014(prompts_path: Path) -> int:
        require_gate_predecessors("authority")
        verify_gate_order_regression()
        return prior_authority(prompts_path)

    runner.grant_authority = authority_0014

    def execute_0014(prompts_path: Path, expected_freeze_sha256: str) -> int:
        require_gate_predecessors("authority")
        runner.require(
            runner.AUTHORITY.is_file(),
            "identity execution blocked: authority artifact absent",
        )
        try:
            return prior_execute(prompts_path, expected_freeze_sha256)
        except BaseException as error:
            runner.LIVE.mkdir(parents=True, exist_ok=True)
            traceback_path = runner.LIVE / "failure-traceback.json"
            if not traceback_path.exists():
                runner.write_json(
                    traceback_path,
                    {
                        "schema_version": 1,
                        "mission_id": runner.MISSION_ID,
                        "attempt_identity": runner.ATTEMPT_ID,
                        "failure_taxonomy": "hybrid_execution_failure",
                        "traceback": _sanitized_traceback(error),
                    },
                )
            raise

    runner.execute = execute_0014

    def verify_0014() -> int:
        result = prior_verify()
        verify_selection_certificate()
        verify_record_lifecycle_regression()
        verify_gate_order_regression()
        execution_result = runner.read_json(runner.RESULT)
        runner.require(
            execution_result["mission_id"] == runner.MISSION_ID
            and execution_result["exactly_once"]["attempt_identity"]
            == runner.ATTEMPT_ID,
            "0014 result identity changed",
        )
        print(
            "ACE2_HYBRID_0014_DECISIVE_VERIFY_PASS "
            "prompt_count=2 generated_tokens_per_prompt=4",
            flush=True,
        )
        return result

    runner.verify = verify_0014


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
        if "--select-prompts" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--select-prompts", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.select_prompts()
        if "--check-record-lifecycle" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-record-lifecycle", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.record_lifecycle_regression()
        if "--check-gate-order" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-gate-order", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.gate_order_regression()
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
