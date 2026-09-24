#!/usr/bin/env python3
"""Record-safe fresh-prompt launcher for nonofficial hybrid attempt 0013."""

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
    / "nonofficial-hybrid-0013"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0013.json"
SELECTION = PRIVATE / "candidate-selection-0013.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0013.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0013.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0013.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0013.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0013.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0013.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0013.json"
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

PREDECESSOR_0012_SHA256 = {
    "build/stage1-layer23-v-rank1-hybrid-v1/private/prompts-0012.json": (
        "a6e84eb5d3f23a816be78db87cdb804b0a0618f5f28ee9fa53287fc09672b798"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0012/execution.stderr.log": (
        "90f2dc740174968311c37987c42fa6c0ed3de1feced37266a48d3ff68e7f46b5"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0012/execution.stdout.log": (
        "6751778f619756e9420f3c5846bb03eaaa961c295fad8394b95a676dcaee5294"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0012/supervisor-state/process_started.json": (
        "37d2720715ce807ee426781a15a1509bd48c2df2edef05a3646855e43b411ab3"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0012/supervisor-state/terminal_record.json": (
        "18504b8d1140a588942d23b77bfdce4b7963fa5356a3fda8a26e852bdf766fea"
    ),
    "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
    "nonofficial-hybrid-0012/terminal-record.json": (
        "18504b8d1140a588942d23b77bfdce4b7963fa5356a3fda8a26e852bdf766fea"
    ),
    "tools/run_stage1_layer23_v_rank1_hybrid_0012.py": (
        "9c1ade1abb48dce063e4eaafcc994c4d12532d2b49d1e711ff741ea23206f043"
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
}

EXPECTED_DENY_SET_DIGEST = (
    "a81194bc365c578bd1044ff8dc334466b882110bff3cd5b6b5787919c15278a0"
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
            parts = path.parts
            if "site-packages" in parts:
                source_file = "<site-packages>/" + "/".join(
                    parts[parts.index("site-packages") + 1 :]
                )
            else:
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
        "mission_id": "stage1rank1hybrid13",
        "attempt_identity": "nonofficial-hybrid-0013",
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
    runner.MISSION_ID = "stage1rank1hybrid13"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0013"
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
        "nonofficial-hybrid-0012": {
            "sha256": PREDECESSOR_0012_SHA256[
                "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
                "nonofficial-hybrid-0012/terminal-record.json"
            ],
            "process_start_count": 1,
            "process_started": True,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_unseen_check = runner.verify_unseen_prompt_hashes
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_package_template = runner.package_template_invariant
    prior_preflight = runner.preflight
    prior_freeze = runner.freeze
    prior_execute = runner.execute
    prior_verify = runner.verify
    retained_cache_class = runner.Rank1HybridProjectionCache
    sidecar_cache_class = retained_cache_class.__mro__[1]

    def predecessor_records(
        expected: dict[str, str],
        identity: str,
    ) -> dict[str, Any]:
        records: dict[str, Any] = {}
        for relative, expected_sha256 in sorted(expected.items()):
            path = ROOT / relative
            path_stat = os.lstat(path)
            runner.require(
                stat.S_ISREG(path_stat.st_mode),
                f"{identity} evidence is not regular: {relative}",
            )
            runner.require(
                runner.sha256_file(path) == expected_sha256,
                f"sealed {identity} evidence hash changed: {relative}",
            )
            records[relative] = runner.file_record(path)
        return records

    def predecessor_0011_records() -> dict[str, Any]:
        records = predecessor_records(PREDECESSOR_0011_SHA256, "0011")
        terminal = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0011/terminal-record.json"
        )
        runner.require(
            terminal.get("process_start_count") == 0
            and terminal.get("process_started") is False
            and terminal.get("outcome") == "pre_execution_failure",
            "sealed 0011 zero-start terminal state changed",
        )
        return records

    def predecessor_0012_records() -> dict[str, Any]:
        records = predecessor_records(PREDECESSOR_0012_SHA256, "0012")
        terminal = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0012/terminal-record.json"
        )
        durable = runner.read_json(
            ROOT
            / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0012/supervisor-state/terminal_record.json"
        )
        runner.require(
            terminal == durable
            and terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("outcome") == "child_nonzero_exit",
            "sealed 0012 one-start terminal state changed",
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
            len(prompt_ids) == 24
            and len(prompt_hashes) == 22
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
                "sealed_attempt_count": 12,
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

    def verify_predecessor_seals_0013() -> None:
        prior_verify_predecessor_seals()
        predecessor_0011_records()
        predecessor_0012_records()

    runner.verify_predecessor_seals = verify_predecessor_seals_0013

    def verify_unseen_prompt_hashes_0013(
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

    runner.verify_unseen_prompt_hashes = verify_unseen_prompt_hashes_0013

    class RecordAwareRetainedScaleProjectionCache(retained_cache_class):
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

    def record_lifecycle_regression() -> int:
        runner.validate_no_execution()
        name = "layer23_position0_v"
        aligned = {"retained_scale_alignment": {"converted_bytes_s8": []}}

        legacy = retained_cache_class.__new__(retained_cache_class)
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
            and legacy_traceback["frames"][-1]
            == {
                "source_file": "tools/run_stage1_layer23_v_rank1_hybrid_0010.py",
                "function": "_apply",
                "line": 862,
                "expression": "record = self.records[-1]",
            }
            and len(legacy.records) == 0,
            "focused regression did not reproduce the 0012 empty-record IndexError",
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
            "status": "PASS_FOCUSED_0012_INDEX_REGRESSION",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
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
                "failing_index": -1,
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
            "ACE2_HYBRID_0013_RECORD_LIFECYCLE_REGRESSION_PASS "
            f"sha256={runner.sha256_file(LIFECYCLE_REGRESSION)}",
            flush=True,
        )
        return 0

    runner.record_lifecycle_regression = record_lifecycle_regression

    def verify_record_lifecycle_regression() -> dict[str, Any]:
        runner.require(
            LIFECYCLE_REGRESSION.is_file(),
            "record-lifecycle regression artifact is absent",
        )
        record = runner.read_json(LIFECYCLE_REGRESSION)
        runner.require(
            record.get("status") == "PASS_FOCUSED_0012_INDEX_REGRESSION"
            and record.get("model_executed") is False
            and record.get("simulator_executed") is False
            and record.get("process_start_count") == 0
            and record["legacy"]["records_before"] == 0
            and record["legacy"]["records_after"] == 0
            and record["legacy"]["failing_expression"] == "self.records[-1]"
            and record["legacy"]["failing_index"] == -1
            and record["repaired"][
                "alignment_attachment_skipped_without_new_record"
            ]
            is True,
            "record-lifecycle regression artifact changed",
        )
        return record

    def source_records_0013(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        paths = [
            SELECTION,
            LIFECYCLE_REGRESSION,
            *(PRIVATE / name for name in HISTORICAL_PROMPT_SHA256),
            *(ROOT / relative for relative in PREDECESSOR_0011_SHA256),
            *(ROOT / relative for relative in PREDECESSOR_0012_SHA256),
        ]
        for path in paths:
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0013

    def execution_bound_paths_0013(
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
                *(PRIVATE / name for name in HISTORICAL_PROMPT_SHA256),
                *(ROOT / relative for relative in PREDECESSOR_0011_SHA256),
                *(ROOT / relative for relative in PREDECESSOR_0012_SHA256),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0013

    def package_template_0013(prompts_path: Path) -> int:
        verify_selection_certificate()
        verify_record_lifecycle_regression()
        result = prior_package_template(prompts_path)
        gate = runner.read_json(runner.TEMPLATE_GATE)
        gate["record_lifecycle_regression"] = runner.file_record(
            LIFECYCLE_REGRESSION
        )
        gate["checks"]["focused_0012_index_regression_passed"] = True
        runner.write_json(runner.TEMPLATE_GATE, gate)
        return result

    runner.package_template_invariant = package_template_0013

    def preflight_0013(prompts_path: Path) -> int:
        verify_selection_certificate()
        verify_record_lifecycle_regression()
        result = prior_preflight(prompts_path)
        record = runner.read_json(runner.PREFLIGHT)
        record["record_lifecycle_regression"] = runner.file_record(
            LIFECYCLE_REGRESSION
        )
        record["checks"]["focused_0012_index_regression_passed"] = True
        runner.write_json(runner.PREFLIGHT, record)
        return result

    runner.preflight = preflight_0013

    def freeze_0013(prompts_path: Path) -> int:
        verify_selection_certificate()
        verify_record_lifecycle_regression()
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["candidate_selection"] = runner.file_record(SELECTION)
        frozen["record_lifecycle_regression"] = runner.file_record(
            LIFECYCLE_REGRESSION
        )
        frozen["predecessor_0011_evidence"] = predecessor_0011_records()
        frozen["predecessor_0012_evidence"] = predecessor_0012_records()
        frozen["prompt_history_deny_set"] = {
            "digest": EXPECTED_DENY_SET_DIGEST,
            "hash_count": 22,
            "id_count": 24,
            "sealed_attempt_count": 12,
        }
        frozen["indexerror_repair"] = {
            "failing_expression": "self.records[-1]",
            "legacy_source": "tools/run_stage1_layer23_v_rank1_hybrid_0010.py:862",
            "repair": "attach retained-scale alignment only when sidecar appends a record",
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0013

    def execute_0013(prompts_path: Path, expected_freeze_sha256: str) -> int:
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

    runner.execute = execute_0013

    def verify_0013() -> int:
        result = prior_verify()
        verify_selection_certificate()
        verify_record_lifecycle_regression()
        execution_result = runner.read_json(runner.RESULT)
        runner.require(
            execution_result["mission_id"] == runner.MISSION_ID
            and execution_result["exactly_once"]["attempt_identity"]
            == runner.ATTEMPT_ID,
            "0013 result identity changed",
        )
        print(
            "ACE2_HYBRID_0013_FRESH_PROMPT_VERIFY_PASS "
            f"selection_sha256={runner.sha256_file(SELECTION)} "
            "prompt_count=2 generated_tokens_per_prompt=4",
            flush=True,
        )
        return result

    runner.verify = verify_0013
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
        if "--check-record-lifecycle" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-record-lifecycle", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.record_lifecycle_regression()
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
