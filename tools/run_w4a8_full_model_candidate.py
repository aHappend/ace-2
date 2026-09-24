#!/usr/bin/env python3
"""Prepare, review, execute, and verify immutable non-official W4A8 commands."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

import ace2_full_model_fixed_point as fixed
import run_w4a8_full_model_software_contract as frozen
import run_w4a8_full_model_software_contract_repair as repair
import w4a8_full_model_evaluator as evaluator


ROOT = frozen.ROOT
RUNNER_PATH = Path(__file__).resolve()
EVALUATOR_PATH = ROOT / "tools/w4a8_full_model_evaluator.py"
HANDOFF_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-full-model-software-contract-stabilization-v1/round-0002.json"
)
C00_DISPOSITION_HANDOFF_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-full-model-software-contract-stabilization-v1/round-0001.json"
)
MISSION_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "w4a8-full-model-software-contract-stabilization-v1/mission.json"
)
NEXT_ACTION = (
    "Repair all Scale32-consuming arithmetic and candidate selection, add a "
    "regression for this divergence, preserve the existing bundles, then prepare "
    "additive replacement c01 bundles for fresh L2 review."
)
EXECUTION_REVIEW_NEXT_ACTION = (
    "Return the independently accepted base and checkpoint-176 c01 command SHA-256 "
    "values for a separate manager execution directive; do not execute either command."
)
COMMAND_NAMESPACE = "attempt-0001-command-scale32-corrected-v1"
SUPERSEDED_COMMAND_NAMESPACE = "attempt-0001-command"
COMMAND_NAME = "command.json"
COMMAND_SHA_NAME = "command.json.sha256"
APPROVAL_NAME = "fresh-l2-execution-approval.json"
APPROVAL_SHA_NAME = "fresh-l2-execution-approval.json.sha256"
MANAGER_DIRECTIVE_NAME = "manager-execution-directive.json"
MANAGER_DIRECTIVE_SHA_NAME = "manager-execution-directive.json.sha256"
MANAGER_DIRECTIVE_ROOT = (
    ROOT / "evidence/approvals/w4a8-c01-scale32-corrected-v1"
)
MANAGER_DIRECTIVE_PATH = MANAGER_DIRECTIVE_ROOT / MANAGER_DIRECTIVE_NAME
MANAGER_DIRECTIVE_SHA_PATH = MANAGER_DIRECTIVE_ROOT / MANAGER_DIRECTIVE_SHA_NAME
C00_DISPOSITION_PATH = (
    ROOT
    / "evidence/diagnostics/w4a8-c00-shared-terminal-disposition-v1/record.json"
)
C00_DISPOSITION_SHA_PATH = C00_DISPOSITION_PATH.with_name("record.json.sha256")
PREPARABLE_CANDIDATE_ID = "c01-mse-clip-grid"
SUPPORTED_CANDIDATE_IDS = ("c00-rtn-absmax", PREPARABLE_CANDIDATE_ID)
MODEL_ALIASES = frozen.MODEL_ALIASES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def record_for(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"bound source is missing: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": relative_or_absolute(path),
        "sha256": evaluator.sha256_file(path),
    }


def verify_sha256_companion(
    artifact_path: Path,
    companion_path: Path,
    artifact_name: str,
) -> str:
    fields = companion_path.read_text(encoding="utf-8").strip().split()
    require(
        len(fields) == 2
        and len(fields[0]) == 64
        and fields[1] == artifact_name
        and all(character in "0123456789abcdef" for character in fields[0]),
        f"{artifact_name} SHA companion differs",
    )
    digest = evaluator.sha256_file(artifact_path)
    require(fields[0] == digest, f"{artifact_name} SHA256 differs")
    return digest


def path_from_record(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else ROOT / path


def verify_source_records(records: list[dict[str, Any]]) -> None:
    require(
        [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "bound source records are not ordered",
    )
    require(
        len({record["path"] for record in records}) == len(records),
        "bound source records contain duplicates",
    )
    for record in records:
        require(set(record) == {"bytes", "path", "sha256"}, "source record fields differ")
        path = path_from_record(record)
        require(path.is_file(), f"bound source disappeared: {record['path']}")
        require(path.stat().st_size == record["bytes"], f"bound source size differs: {record['path']}")
        require(
            evaluator.sha256_file(path) == record["sha256"],
            f"bound source hash differs: {record['path']}",
        )


def command_directory(spec: dict[str, Any], candidate_id: str) -> Path:
    return ROOT / spec["artifact_root"] / candidate_id / COMMAND_NAMESPACE


def superseded_command_directory(spec: dict[str, Any], candidate_id: str) -> Path:
    return ROOT / spec["artifact_root"] / candidate_id / SUPERSEDED_COMMAND_NAMESPACE


def attempt_directory(
    contract: dict[str, Any],
    spec: dict[str, Any],
    candidate_id: str,
) -> Path:
    return (
        ROOT
        / spec["artifact_root"]
        / candidate_id
        / contract["artifact_policy"]["candidate_attempt_name"]
    )


def load_preparation_authority() -> dict[str, Any]:
    handoff = evaluator.load_json(HANDOFF_PATH)
    require(handoff["kind"] == "round_reviewed_handoff", "Fresh-L2 handoff kind differs")
    require(handoff["producer_role"] == "reviewer", "Fresh-L2 producer differs")
    require(handoff["review"]["status"] == "continue", "Fresh-L2 did not continue")
    require(handoff["review"]["next_action"] == NEXT_ACTION, "Fresh-L2 next action differs")
    mission = evaluator.load_json(MISSION_PATH)
    require(
        mission["mission_id"] == handoff["mission_id"],
        "Fresh-L2 mission identity differs",
    )
    return {
        "handoff": record_for(HANDOFF_PATH),
        "mission": record_for(MISSION_PATH),
        "review_status": handoff["review"]["status"],
        "round": handoff["round"],
    }


def verify_c00_disposition() -> dict[str, Any]:
    verify_sha256_companion(
        C00_DISPOSITION_PATH,
        C00_DISPOSITION_SHA_PATH,
        C00_DISPOSITION_PATH.name,
    )
    disposition = evaluator.require_canonical_json(C00_DISPOSITION_PATH)
    require(
        disposition["kind"] == "w4a8_c00_shared_terminal_disposition",
        "c00 disposition kind differs",
    )
    require(
        disposition["status"] == "SEALED_TERMINAL_UNSELECTABLE",
        "c00 disposition status differs",
    )
    require(
        disposition["shared_c00_disposition"]
        == "TERMINAL_UNSELECTABLE_PROCEDURAL_NO_QUALITY_CONCLUSION",
        "shared c00 disposition differs",
    )
    require(
        disposition["current_mission"] == record_for(MISSION_PATH),
        "c00 disposition mission binding differs",
    )
    require(
        disposition["contract"] == record_for(frozen.CONTRACT_PATH),
        "c00 disposition contract binding differs",
    )
    require(
        disposition["reviewer_handoff"] == record_for(C00_DISPOSITION_HANDOFF_PATH),
        "c00 disposition reviewer binding differs",
    )
    preserved_records = sorted(
        (
            record
            for model in disposition["artifacts"].values()
            for record in model["files"]
        ),
        key=lambda record: record["path"],
    )
    verify_source_records(preserved_records)
    verify_source_records([disposition["prior_containment"]])
    contract, _ = frozen.load_contract()
    for alias in MODEL_ALIASES:
        require(
            not attempt_directory(
                contract,
                frozen.model_spec(contract, alias),
                "c00-rtn-absmax",
            ).exists(),
            f"published c00 attempt unexpectedly exists for {alias}",
        )
    return disposition


def repaired_control_paths(contract: dict[str, Any], alias: str) -> list[Path]:
    spec = frozen.model_spec(contract, alias)
    root = ROOT / spec["artifact_root"] / repair.REPAIR_NAMESPACE
    require(root.is_dir(), f"repaired control directory is missing for {alias}")
    paths = sorted(path for path in root.iterdir() if path.is_file())
    require(
        [path.name for path in paths]
        == [
            "identity-oracle-self-test.json",
            "replay-0001.json",
            "replay-0002.json",
            "summary.json",
        ],
        f"repaired control file set differs for {alias}",
    )
    return paths


def superseded_command_paths(
    spec: dict[str, Any],
    candidate_id: str,
) -> list[Path]:
    root = superseded_command_directory(spec, candidate_id)
    require(root.is_dir(), f"superseded c01 command directory is missing for {spec['alias']}")
    paths = sorted(path for path in root.iterdir() if path.is_file())
    require(
        [path.name for path in paths] == [COMMAND_NAME, COMMAND_SHA_NAME],
        f"superseded c01 command file set differs for {spec['alias']}",
    )
    verify_sha256_companion(root / COMMAND_NAME, root / COMMAND_SHA_NAME, COMMAND_NAME)
    command = evaluator.require_canonical_json(root / COMMAND_NAME)
    require(
        command["candidate"]["candidate_id"] == candidate_id,
        f"superseded c01 candidate identity differs for {spec['alias']}",
    )
    require(
        command["model"]["alias"] == spec["alias"],
        f"superseded c01 model identity differs for {spec['alias']}",
    )
    require(
        command["pre_execution_review"]["status"]
        == "PENDING_FRESH_L2_PRE_EXECUTION_REVIEW",
        f"superseded c01 command status differs for {spec['alias']}",
    )
    return paths


def quality_dataset_source_paths() -> list[Path]:
    manifest = evaluator.load_json(ROOT / "benchmark/quality/PROMPT_MANIFEST.json")
    paths: set[Path] = set()
    for spec in manifest["datasets"].values():
        paths.update(evaluator.local_dataset_files(spec))
    task_root = ROOT / "benchmark/quality/lm_eval_tasks"
    paths.update(path for path in task_root.iterdir() if path.is_file())
    datasets_cache = Path.home() / ".cache/huggingface/datasets"
    for spec in manifest["lm_eval"]["tasks"].values():
        snapshot = evaluator.dataset_snapshot_path(
            spec["repository"], spec["revision"]
        )
        require(snapshot.is_dir(), f"lm-eval snapshot is missing: {spec['repository']}")
        paths.update(path for path in snapshot.rglob("*") if path.is_file())
        processed = (
            datasets_cache
            / spec["repository"].replace("/", "___")
            / spec["config"]
            / "0.0.0"
            / spec["revision"]
        )
        if processed.is_dir():
            paths.update(
                path
                for path in processed.rglob("*")
                if path.is_file() and path.suffix in {".arrow", ".json"}
            )
    piqa_source = manifest["lm_eval"]["tasks"]["piqa"]["raw_sources"][
        "train_dev"
    ]
    downloads = datasets_cache / "downloads"
    archive_matches: list[Path] = []
    for metadata_path in downloads.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        archive = metadata_path.with_suffix("")
        if metadata.get("url") == piqa_source["url"] and archive.is_file():
            archive_matches.append(archive)
    require(len(archive_matches) == 1, "verified PIQA archive source count differs")
    require(
        evaluator.sha256_file(archive_matches[0]) == piqa_source["sha256"],
        "PIQA archive source hash differs",
    )
    paths.add(archive_matches[0])
    return sorted(paths, key=relative_or_absolute)


def bound_source_paths(contract: dict[str, Any], spec: dict[str, Any]) -> list[Path]:
    paths = {
        C00_DISPOSITION_PATH,
        C00_DISPOSITION_SHA_PATH,
        frozen.CONTRACT_PATH,
        frozen.CONTRACT_SHA_PATH,
        ROOT / "benchmark/quality/QUALITY_CONFIG.json",
        ROOT / "benchmark/quality/PROMPT_MANIFEST.json",
        ROOT / "tools/ace2_absolute_rope_online_attention_reference.py",
        ROOT / "tools/ace2_full_model_fixed_point.py",
        ROOT / "tools/ace2_quality_contracts.py",
        ROOT / "tools/run_w4a8_full_model_software_contract.py",
        ROOT / "tools/run_w4a8_full_model_software_contract_repair.py",
        ROOT / "tools/verify_w4a8_full_model_software_contract.py",
        ROOT / "verification/test_w4a8_full_model_candidate_package.py",
        ROOT / "verification/test_w4a8_full_model_evaluator.py",
        EVALUATOR_PATH,
        RUNNER_PATH,
        C00_DISPOSITION_HANDOFF_PATH,
        HANDOFF_PATH,
        MISSION_PATH,
        *repaired_control_paths(contract, spec["alias"]),
        *superseded_command_paths(spec, PREPARABLE_CANDIDATE_ID),
        *quality_dataset_source_paths(),
    }
    base = contract["model_identities"]["qwen2.5-0.5b-instruct"]
    snapshot = frozen.snapshot_path(base["repository"], base["revision"])
    paths.update(snapshot / filename for filename in base["files"])
    if spec["alias"] == "checkpoint-176":
        paths.update(
            ROOT / spec["adapter"][field]["path"]
            for field in ("adapter_config", "adapter_model")
        )
    return sorted(paths, key=relative_or_absolute)


def load_context(alias: str, candidate_id: str) -> dict[str, Any]:
    require(candidate_id in SUPPORTED_CANDIDATE_IDS, "candidate is not supported")
    contract, contract_sha256 = frozen.load_contract()
    environment = repair.exact_runtime_environment(contract)
    quality_config = evaluator.load_json(
        ROOT / "benchmark/quality/QUALITY_CONFIG.json"
    )
    evaluator.validate_contract_support(contract, quality_config)
    fixed.validate_runtime(quality_config)
    spec = frozen.model_spec(contract, alias)
    frozen.verify_local_model_inputs(contract, spec)
    control = repair.verify_repair(contract, contract_sha256, alias, environment)
    preparation_authority = load_preparation_authority()
    c00_disposition = verify_c00_disposition()
    return {
        "candidate": evaluator.candidate_spec(contract, candidate_id),
        "c00_disposition": c00_disposition,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "control": control,
        "environment": environment,
        "preparation_authority": preparation_authority,
        "spec": spec,
    }


def execution_argv(
    alias: str,
    candidate_id: str,
    command_path: Path,
    command_sha_path: Path,
    approval_path: Path,
    approval_sha_path: Path,
    manager_directive_path: Path,
    manager_directive_sha_path: Path,
    threads: int,
) -> list[str]:
    return [
        "./.venv/bin/python",
        RUNNER_PATH.relative_to(ROOT).as_posix(),
        "--action",
        "execute-command",
        "--model",
        alias,
        "--candidate",
        candidate_id,
        "--command",
        command_path.relative_to(ROOT).as_posix(),
        "--command-sha256-file",
        command_sha_path.relative_to(ROOT).as_posix(),
        "--fresh-l2-approval",
        approval_path.relative_to(ROOT).as_posix(),
        "--fresh-l2-approval-sha256-file",
        approval_sha_path.relative_to(ROOT).as_posix(),
        "--manager-execution-directive",
        manager_directive_path.relative_to(ROOT).as_posix(),
        "--manager-execution-directive-sha256-file",
        manager_directive_sha_path.relative_to(ROOT).as_posix(),
        "--threads",
        str(threads),
    ]


def build_command(context: dict[str, Any], threads: int, created_at_utc: str) -> dict[str, Any]:
    contract = context["contract"]
    spec = context["spec"]
    candidate = context["candidate"]
    candidate_id = candidate["candidate_id"]
    command_root = command_directory(spec, candidate_id)
    command_path = command_root / COMMAND_NAME
    command_sha_path = command_root / COMMAND_SHA_NAME
    approval_path = command_root / APPROVAL_NAME
    approval_sha_path = command_root / APPROVAL_SHA_NAME
    superseded_root = superseded_command_directory(spec, candidate_id)
    source_records = [record_for(path) for path in bound_source_paths(contract, spec)]
    source_records.sort(key=lambda record: record["path"])
    return {
        "attempt": contract["artifact_policy"]["candidate_attempt_name"],
        "candidate": candidate,
        "c00_terminal_disposition": record_for(C00_DISPOSITION_PATH),
        "command_id": f"{spec['alias']}-{candidate_id}-attempt-0001-scale32-corrected-v2",
        "contract_sha256": context["contract_sha256"],
        "created_at_utc": created_at_utc,
        "environment": context["environment"],
        "execution": {
            "argv": execution_argv(
                spec["alias"],
                candidate_id,
                command_path,
                command_sha_path,
                approval_path,
                approval_sha_path,
                MANAGER_DIRECTIVE_PATH,
                MANAGER_DIRECTIVE_SHA_PATH,
                threads,
            ),
            "network_access": False,
            "official_generation_attempt": False,
            "threads": threads,
        },
        "kind": "immutable_non_official_candidate_attempt_command",
        "model": {
            "alias": spec["alias"],
            "model_id": spec["model_id"],
            "model_identity_sha256": evaluator.canonical_sha256(
                spec["contract_identity"]
            ),
        },
        "pre_execution_review": {
            "preparation_authority": context["preparation_authority"],
            "required_approval": {
                "path": approval_path.relative_to(ROOT).as_posix(),
                "review_next_action": EXECUTION_REVIEW_NEXT_ACTION,
                "sha256_path": approval_sha_path.relative_to(ROOT).as_posix(),
            },
            "required_manager_execution_directive": {
                "accepted_command_aliases": list(MODEL_ALIASES),
                "must_name_both_accepted_command_sha256_values": True,
                "path": MANAGER_DIRECTIVE_PATH.relative_to(ROOT).as_posix(),
                "sha256_path": MANAGER_DIRECTIVE_SHA_PATH.relative_to(ROOT).as_posix(),
            },
            "status": "PENDING_FRESH_L2_PRE_EXECUTION_REVIEW",
        },
        "repaired_control": context["control"],
        "schema_version": 3,
        "source_records": source_records,
        "supersedes": {
            "command": record_for(superseded_root / COMMAND_NAME),
            "command_sha256_companion": record_for(
                superseded_root / COMMAND_SHA_NAME
            ),
            "reason": "UNEXECUTED_REJECTED_SCALE32_ORACLE_MISMATCH",
        },
        "target": attempt_directory(contract, spec, candidate_id).relative_to(ROOT).as_posix(),
    }


def prepare_command(alias: str, candidate_id: str, threads: int) -> dict[str, Any]:
    require(candidate_id == PREPARABLE_CANDIDATE_ID, "only c01 command preparation is authorized")
    context = load_context(alias, candidate_id)
    spec = context["spec"]
    target = command_directory(spec, candidate_id)
    require(not target.exists(), f"immutable command already exists: {target.relative_to(ROOT)}")
    require(
        not attempt_directory(context["contract"], spec, candidate_id).exists(),
        f"candidate attempt already exists for {alias}",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-command-", dir=target.parent))
    try:
        command = build_command(context, threads, evaluator.utc_now())
        command_path = temporary / COMMAND_NAME
        evaluator.write_json(command_path, command)
        command_sha256 = evaluator.sha256_file(command_path)
        (temporary / COMMAND_SHA_NAME).write_text(
            f"{command_sha256}  {COMMAND_NAME}\n",
            encoding="utf-8",
        )
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "alias": alias,
        "candidate_id": candidate_id,
        "command_path": (target / COMMAND_NAME).relative_to(ROOT).as_posix(),
        "command_sha256": command_sha256,
        "execution_argv": command["execution"]["argv"],
        "status": "PREPARED_PENDING_FRESH_L2_PRE_EXECUTION_REVIEW",
    }


def load_bound_command(
    alias: str,
    candidate_id: str,
    command_argument: str,
    command_sha_argument: str,
    threads: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    context = load_context(alias, candidate_id)
    expected_root = command_directory(context["spec"], candidate_id)
    command_path = (ROOT / command_argument).resolve()
    command_sha_path = (ROOT / command_sha_argument).resolve()
    require(command_path == expected_root / COMMAND_NAME, "command path differs")
    require(command_sha_path == expected_root / COMMAND_SHA_NAME, "command SHA path differs")
    command_sha256 = verify_sha256_companion(
        command_path,
        command_sha_path,
        COMMAND_NAME,
    )
    command = evaluator.require_canonical_json(command_path)
    expected = build_command(context, threads, command["created_at_utc"])
    require(command == expected, "hash-bound command semantics or sources differ")
    verify_source_records(command["source_records"])
    return command, context, command_sha256


def load_execution_approval(
    command: dict[str, Any],
    command_sha256: str,
    approval_argument: str,
    approval_sha_argument: str,
) -> tuple[dict[str, Any], str]:
    required = command["pre_execution_review"]["required_approval"]
    approval_path = (ROOT / approval_argument).resolve()
    approval_sha_path = (ROOT / approval_sha_argument).resolve()
    require(approval_path == (ROOT / required["path"]).resolve(), "approval path differs")
    require(
        approval_sha_path == (ROOT / required["sha256_path"]).resolve(),
        "approval SHA path differs",
    )
    approval_sha256 = verify_sha256_companion(
        approval_path,
        approval_sha_path,
        APPROVAL_NAME,
    )
    approval = evaluator.require_canonical_json(approval_path)
    require(
        set(approval)
        == {
            "candidate_id",
            "command",
            "kind",
            "mission",
            "model_alias",
            "reviewer_handoff",
            "schema_version",
            "status",
        },
        "approval fields differ",
    )
    require(
        approval["kind"] == "fresh_l2_w4a8_candidate_execution_approval",
        "approval kind differs",
    )
    require(
        approval["status"] == "APPROVED_FOR_SINGLE_NON_OFFICIAL_EXECUTION",
        "approval status differs",
    )
    require(approval["schema_version"] == 1, "approval schema differs")
    require(approval["candidate_id"] == command["candidate"]["candidate_id"], "approval candidate differs")
    require(approval["model_alias"] == command["model"]["alias"], "approval model differs")
    require(
        approval["command"]
        == {
            "path": command["execution"]["argv"][
                command["execution"]["argv"].index("--command") + 1
            ],
            "sha256": command_sha256,
        },
        "approval command binding differs",
    )
    require(
        approval["mission"]
        == command["pre_execution_review"]["preparation_authority"]["mission"],
        "approval mission binding differs",
    )
    verify_source_records([approval["reviewer_handoff"]])
    handoff = evaluator.load_json(path_from_record(approval["reviewer_handoff"]))
    require(handoff["kind"] == "round_reviewed_handoff", "approval handoff kind differs")
    require(handoff["producer_role"] == "reviewer", "approval producer differs")
    require(handoff["review"]["status"] == "continue", "approval handoff did not continue")
    require(
        handoff["review"]["next_action"] == required["review_next_action"],
        "approval next action differs",
    )
    require(handoff["mission_context"] == MISSION_PATH.as_posix(), "approval mission path differs")
    require(
        handoff["mission_id"]
        == evaluator.load_json(MISSION_PATH)["mission_id"],
        "approval mission identity differs",
    )
    return approval, approval_sha256


def accepted_command_bindings(contract: dict[str, Any]) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for alias in MODEL_ALIASES:
        spec = frozen.model_spec(contract, alias)
        root = command_directory(spec, PREPARABLE_CANDIDATE_ID)
        digest = verify_sha256_companion(
            root / COMMAND_NAME,
            root / COMMAND_SHA_NAME,
            COMMAND_NAME,
        )
        bindings[alias] = {
            "path": (root / COMMAND_NAME).relative_to(ROOT).as_posix(),
            "sha256": digest,
        }
    return bindings


def load_manager_execution_directive(
    command: dict[str, Any],
    approval: dict[str, Any],
    contract: dict[str, Any],
    directive_argument: str,
    directive_sha_argument: str,
) -> tuple[dict[str, Any], str]:
    required = command["pre_execution_review"][
        "required_manager_execution_directive"
    ]
    directive_path = (ROOT / directive_argument).resolve()
    directive_sha_path = (ROOT / directive_sha_argument).resolve()
    require(
        directive_path == (ROOT / required["path"]).resolve(),
        "manager execution directive path differs",
    )
    require(
        directive_sha_path == (ROOT / required["sha256_path"]).resolve(),
        "manager execution directive SHA path differs",
    )
    directive_sha256 = verify_sha256_companion(
        directive_path,
        directive_sha_path,
        MANAGER_DIRECTIVE_NAME,
    )
    directive = evaluator.require_canonical_json(directive_path)
    require(
        set(directive)
        == {
            "accepted_commands",
            "kind",
            "mission",
            "reviewer_handoff",
            "schema_version",
            "status",
        },
        "manager execution directive fields differ",
    )
    require(
        directive["kind"] == "manager_w4a8_c01_execution_directive",
        "manager execution directive kind differs",
    )
    require(
        directive["status"] == "AUTHORIZED_FOR_SINGLE_NON_OFFICIAL_EXECUTION",
        "manager execution directive status differs",
    )
    require(directive["schema_version"] == 1, "manager execution directive schema differs")
    require(
        directive["accepted_commands"] == accepted_command_bindings(contract),
        "manager directive does not name both accepted command SHA256 values",
    )
    require(
        directive["mission"]
        == command["pre_execution_review"]["preparation_authority"]["mission"],
        "manager execution directive mission differs",
    )
    require(
        directive["reviewer_handoff"] == approval["reviewer_handoff"],
        "manager execution directive reviewer handoff differs",
    )
    return directive, directive_sha256


def attempt_source_hashes(
    command: dict[str, Any],
    command_sha256: str,
    approval: dict[str, Any],
    approval_sha256: str,
    manager_directive_sha256: str,
) -> dict[str, str]:
    argv = command["execution"]["argv"]
    return {
        "approval_reviewer_handoff_sha256": approval["reviewer_handoff"]["sha256"],
        "approval_sha256": approval_sha256,
        "bound_source_records_sha256": evaluator.canonical_sha256(
            command["source_records"]
        ),
        "command_path": argv[argv.index("--command") + 1],
        "command_sha256": command_sha256,
        "manager_execution_directive_sha256": manager_directive_sha256,
        "repaired_control_identity_sha256": command["repaired_control"][
            "identity_sha256"
        ],
    }


def execute_command(
    alias: str,
    candidate_id: str,
    command_argument: str,
    command_sha_argument: str,
    approval_argument: str,
    approval_sha_argument: str,
    manager_directive_argument: str,
    manager_directive_sha_argument: str,
    threads: int,
) -> dict[str, Any]:
    command, context, command_sha256 = load_bound_command(
        alias,
        candidate_id,
        command_argument,
        command_sha_argument,
        threads,
    )
    approval, approval_sha256 = load_execution_approval(
        command,
        command_sha256,
        approval_argument,
        approval_sha_argument,
    )
    _manager_directive, manager_directive_sha256 = (
        load_manager_execution_directive(
            command,
            approval,
            context["contract"],
            manager_directive_argument,
            manager_directive_sha_argument,
        )
    )
    actual_argv = ["./.venv/bin/python", RUNNER_PATH.relative_to(ROOT).as_posix(), *sys.argv[1:]]
    require(actual_argv == command["execution"]["argv"], "execution argv differs from command")
    source_hashes = attempt_source_hashes(
        command,
        command_sha256,
        approval,
        approval_sha256,
        manager_directive_sha256,
    )
    target = evaluator.execute_candidate_attempt(
        context["contract"],
        context["contract_sha256"],
        context["spec"],
        source_hashes,
        candidate_id,
    )
    result = evaluator.verify_candidate_attempt(
        context["contract"],
        context["contract_sha256"],
        context["spec"],
        source_hashes,
        candidate_id,
    )
    require(target == ROOT / result["path"], "executed and verified attempt paths differ")
    return result


def review_command(alias: str, candidate_id: str, threads: int) -> dict[str, Any]:
    context = load_context(alias, candidate_id)
    root = command_directory(context["spec"], candidate_id)
    command, context, command_sha256 = load_bound_command(
        alias,
        candidate_id,
        (root / COMMAND_NAME).relative_to(ROOT).as_posix(),
        (root / COMMAND_SHA_NAME).relative_to(ROOT).as_posix(),
        threads,
    )
    required = command["pre_execution_review"]["required_approval"]
    require(not (ROOT / required["path"]).exists(), "execution approval already exists")
    require(not (ROOT / required["sha256_path"]).exists(), "execution approval SHA already exists")
    manager_required = command["pre_execution_review"][
        "required_manager_execution_directive"
    ]
    require(
        not (ROOT / manager_required["path"]).exists(),
        "manager execution directive already exists",
    )
    require(
        not (ROOT / manager_required["sha256_path"]).exists(),
        "manager execution directive SHA already exists",
    )
    require(
        not attempt_directory(context["contract"], context["spec"], candidate_id).exists(),
        "candidate attempt already exists",
    )
    return {
        "alias": alias,
        "candidate_id": candidate_id,
        "command_path": (root / COMMAND_NAME).relative_to(ROOT).as_posix(),
        "command_sha256": command_sha256,
        "mission_sha256": command["pre_execution_review"]["preparation_authority"][
            "mission"
        ]["sha256"],
        "status": "PASS_READY_FOR_FRESH_L2_PRE_EXECUTION_REVIEW",
    }


def verify_attempt(
    alias: str,
    candidate_id: str,
    command_argument: str,
    command_sha_argument: str,
    approval_argument: str,
    approval_sha_argument: str,
    manager_directive_argument: str,
    manager_directive_sha_argument: str,
    threads: int,
) -> dict[str, Any]:
    command, context, command_sha256 = load_bound_command(
        alias,
        candidate_id,
        command_argument,
        command_sha_argument,
        threads,
    )
    approval, approval_sha256 = load_execution_approval(
        command,
        command_sha256,
        approval_argument,
        approval_sha_argument,
    )
    _manager_directive, manager_directive_sha256 = (
        load_manager_execution_directive(
            command,
            approval,
            context["contract"],
            manager_directive_argument,
            manager_directive_sha_argument,
        )
    )
    return evaluator.verify_candidate_attempt(
        context["contract"],
        context["contract_sha256"],
        context["spec"],
        attempt_source_hashes(
            command,
            command_sha256,
            approval,
            approval_sha256,
            manager_directive_sha256,
        ),
        candidate_id,
    )


def self_test(alias: str, candidate_id: str) -> dict[str, Any]:
    evaluator.self_test()
    context = load_context(alias, candidate_id)
    verify_source_records(
        sorted(
            (record_for(path) for path in bound_source_paths(context["contract"], context["spec"])),
            key=lambda record: record["path"],
        )
    )
    return {
        "alias": alias,
        "candidate_id": candidate_id,
        "contract_sha256": context["contract_sha256"],
        "control_identity_sha256": context["control"]["identity_sha256"],
        "status": "PASS",
    }


def synthetic_smoke(candidate_id: str) -> dict[str, Any]:
    require(
        candidate_id == PREPARABLE_CANDIDATE_ID,
        "synthetic smoke supports only additive c01",
    )
    contract, contract_sha256 = frozen.load_contract()
    candidate = evaluator.candidate_spec(contract, candidate_id)
    source = torch.tensor(
        [
            [0.29296875, -0.1318359375, 0.0],
            [0.625, -0.625, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    first = evaluator.mse_clip_grid_quantization_scale32(source, candidate)
    second = evaluator.mse_clip_grid_quantization_scale32(source, candidate)
    require(
        all(torch.equal(left, right) for left, right in zip(first, second, strict=True)),
        "c01 synthetic selector is nondeterministic",
    )
    qweight, scales, numerators, records = first
    require(
        scales.tolist() == [evaluator.scale32_value(int(record)) for record in records],
        "c01 synthetic Scale32 values differ",
    )
    return {
        "candidate_id": candidate_id,
        "contract_sha256": contract_sha256,
        "quality_data_accessed": False,
        "qweight_sha256": evaluator.tensor_sha256(qweight),
        "scale32_records_sha256": evaluator.tensor_sha256(records),
        "scales_sha256": evaluator.tensor_sha256(scales),
        "scoring_performed": False,
        "selected_clip_ratio_numerators": numerators.tolist(),
        "status": "PASS_DETERMINISTIC_SYNTHETIC_NON_SCORING",
    }


def harness_smoke(alias: str, candidate_id: str) -> dict[str, Any]:
    require(
        candidate_id == PREPARABLE_CANDIDATE_ID,
        "c00 harness execution is terminally prohibited",
    )
    context = load_context(alias, candidate_id)
    contract = context["contract"]
    spec = context["spec"]
    config = evaluator.load_json(ROOT / "benchmark/quality/QUALITY_CONFIG.json")
    manifest = evaluator.load_json(ROOT / "benchmark/quality/PROMPT_MANIFEST.json")
    fixed.seed_everything(config)
    tokenizer = evaluator.load_tokenizer(spec)
    inputs, observations = evaluator.load_bounded_inputs(tokenizer, contract, manifest)
    smoke_input = inputs["calibration"][0][:, :64]
    model = evaluator.load_model(spec)
    ranges, operator_ranges = fixed.calibrate(model, [smoke_input])
    require(
        all(value.input_absmax > 0.0 and value.output_absmax > 0.0 for value in ranges.values()),
        "smoke calibration did not observe every linear",
    )
    evaluator.replace_linears_for_candidate(
        model,
        ranges,
        context["candidate"],
        rope_diagnostic_mechanism=fixed.ACTIVE_ROPE_MECHANISM,
    )
    runtime_operator_ranges = evaluator.replace_fixed_operators_for_candidate(
        model,
        operator_ranges,
        context["candidate"],
        rope_diagnostic_mechanism=fixed.ACTIVE_ROPE_MECHANISM,
    )
    evaluator.enable_w4a8_kv_cache(model)
    temporary = Path(tempfile.mkdtemp(prefix=f"w4a8-{candidate_id}-smoke-"))
    try:
        packed = temporary / "packed-w4.bin"
        layout = evaluator._write_packed_w4(model, packed)
        scale_hashes = evaluator._write_scale_artifacts(
            model, ranges, runtime_operator_ranges, temporary
        )
        generation = dict(contract["shared_w4a8_contract"]["generation"])
        generation["max_new_tokens"] = 1
        output = evaluator.generate_w4a8(
            model,
            tokenizer,
            [contract["evaluation_contract"]["canonical_generation_prompts"][0]],
            generation,
        )[0]
        require(len(output["generated_token_ids"]) == 1, "smoke decode length differs")
        return {
            "alias": alias,
            "candidate_id": candidate_id,
            "calibration_record_sha256": observations["c4_calibration"][
                "record_sha256"
            ],
            "generated_token_id": output["generated_token_ids"][0],
            "packed_bytes": packed.stat().st_size,
            "projection_count": len(layout),
            "scale_hashes": scale_hashes,
            "status": "PASS_NON_SCORING_HARNESS_SMOKE",
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=(
            "self-test",
            "synthetic-smoke",
            "harness-smoke",
            "prepare-command",
            "review-command",
            "execute-command",
            "verify-attempt",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=MODEL_ALIASES, required=True)
    parser.add_argument(
        "--candidate",
        choices=SUPPORTED_CANDIDATE_IDS,
        default=PREPARABLE_CANDIDATE_ID,
    )
    parser.add_argument("--command")
    parser.add_argument("--command-sha256-file")
    parser.add_argument("--fresh-l2-approval")
    parser.add_argument("--fresh-l2-approval-sha256-file")
    parser.add_argument("--manager-execution-directive")
    parser.add_argument("--manager-execution-directive-sha256-file")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    require(args.threads >= 1, "thread count must be positive")
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.action == "self-test":
        require(
            args.command is None
            and args.command_sha256_file is None
            and args.fresh_l2_approval is None
            and args.fresh_l2_approval_sha256_file is None
            and args.manager_execution_directive is None
            and args.manager_execution_directive_sha256_file is None,
            "self-test command arguments differ",
        )
        result = self_test(args.model, args.candidate)
    elif args.action == "synthetic-smoke":
        require(
            args.command is None
            and args.command_sha256_file is None
            and args.fresh_l2_approval is None
            and args.fresh_l2_approval_sha256_file is None
            and args.manager_execution_directive is None
            and args.manager_execution_directive_sha256_file is None,
            "synthetic-smoke command arguments differ",
        )
        result = synthetic_smoke(args.candidate)
    elif args.action == "harness-smoke":
        require(
            args.command is None
            and args.command_sha256_file is None
            and args.fresh_l2_approval is None
            and args.fresh_l2_approval_sha256_file is None
            and args.manager_execution_directive is None
            and args.manager_execution_directive_sha256_file is None,
            "harness-smoke command arguments differ",
        )
        result = harness_smoke(args.model, args.candidate)
    elif args.action == "prepare-command":
        require(
            args.command is None
            and args.command_sha256_file is None
            and args.fresh_l2_approval is None
            and args.fresh_l2_approval_sha256_file is None
            and args.manager_execution_directive is None
            and args.manager_execution_directive_sha256_file is None,
            "prepare command arguments differ",
        )
        result = prepare_command(args.model, args.candidate, args.threads)
    elif args.action == "review-command":
        require(
            args.command is None
            and args.command_sha256_file is None
            and args.fresh_l2_approval is None
            and args.fresh_l2_approval_sha256_file is None
            and args.manager_execution_directive is None
            and args.manager_execution_directive_sha256_file is None,
            "review command arguments differ",
        )
        result = review_command(args.model, args.candidate, args.threads)
    else:
        require(args.command is not None, "command path is required")
        require(args.command_sha256_file is not None, "command SHA path is required")
        require(args.fresh_l2_approval is not None, "Fresh-L2 approval path is required")
        require(
            args.fresh_l2_approval_sha256_file is not None,
            "Fresh-L2 approval SHA path is required",
        )
        require(
            args.manager_execution_directive is not None,
            "manager execution directive path is required",
        )
        require(
            args.manager_execution_directive_sha256_file is not None,
            "manager execution directive SHA path is required",
        )
        if args.action == "execute-command":
            result = execute_command(
                args.model,
                args.candidate,
                args.command,
                args.command_sha256_file,
                args.fresh_l2_approval,
                args.fresh_l2_approval_sha256_file,
                args.manager_execution_directive,
                args.manager_execution_directive_sha256_file,
                args.threads,
            )
        else:
            result = verify_attempt(
                args.model,
                args.candidate,
                args.command,
                args.command_sha256_file,
                args.fresh_l2_approval,
                args.fresh_l2_approval_sha256_file,
                args.manager_execution_directive,
                args.manager_execution_directive_sha256_file,
                args.threads,
            )
    print(
        "ACE2_W4A8_CANDIDATE "
        + json.dumps(
            {
                "action": args.action,
                "candidate": args.candidate,
                "model": args.model,
                "result": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
