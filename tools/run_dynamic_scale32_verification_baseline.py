#!/usr/bin/env python3
"""Verification-stage, baseline-only, exactly-once Dynamic Scale32 runner.

This runner has no command-line subprocess surface.  It will execute only one
Manager-frozen command with an exact schema and provenance binding.  Missing or
ambiguous authority fails before reservation.  Recovery never reruns a command
whose durable reservation or execution mark already exists.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from ace2_software_identity import evaluate_distribution_versions


SCHEMA_VERSION = 1
SUCCESSOR_LEDGER_SCHEMA_VERSION = 2
CONTRACT_ID = "shared_token_group_dynamic_scale32_v1"
REQUIRED_STAGE = "verification"
MANAGER_TRANSITION_AT = "2026-08-02T09:13:08.974409Z"
AUTHORITY_KIND = "manager_verification_baseline_execution_authority_v1"
SUCCESSOR_AUTHORITY_KIND = "manager_verification_baseline_successor_execution_authority_v1"
PREDECESSOR_RUN_ID = "dynamic-scale32-baseline-official-v1"
SUCCESSOR_GENERATION = 1
RUN_ID_RE = re.compile(r"dynamic-scale32-baseline-[a-z0-9][a-z0-9._-]{0,47}\Z")

PIPELINE_REL = Path("research/PIPELINE_STATE.json")
RTL_REVIEW_REL = Path(
    "evidence/review/rtl_stage_closing_shared_token_group_dynamic_scale32_v1/decision-v2.json"
)
RTL_REVIEW_COMPANION_REL = RTL_REVIEW_REL.with_suffix(".sha256")
PROPOSAL_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/architecture/PROPOSAL.json"
)
PROPOSAL_COMPANION_REL = PROPOSAL_REL.with_suffix(".sha256")
MANAGER_FREEZE_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/architecture/MANAGER_FREEZE.json"
)
PROMPT_MANIFEST_REL = Path("benchmark/quality/PROMPT_MANIFEST.json")
QUALITY_CONFIG_REL = Path("benchmark/quality/QUALITY_CONFIG.json")
QUALITY_REQUIREMENTS_REL = Path("benchmark/quality/requirements.txt")
EXECUTOR_REL = Path("tools/ace2_dynamic_scale32_baseline.py")
SOFTWARE_IDENTITY_REL = Path("tools/ace2_software_identity.py")
AUTHORITY_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/EXECUTION_AUTHORITY.json"
)
V1_MANAGER_ISSUANCE_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/MANAGER_ISSUANCE.json"
)
LEGACY_ROOT_SUCCESSOR_AUTHORITY_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/"
    "SUCCESSOR_EXECUTION_AUTHORITY.json"
)
RECOVERY_ROOT_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery"
)
SUCCESSOR_AUTHORITY_BUNDLE_NAME = "successor_authority_generation_1_v1"
SUCCESSOR_AUTHORITY_BUNDLE_REL = RECOVERY_ROOT_REL / SUCCESSOR_AUTHORITY_BUNDLE_NAME
SUCCESSOR_AUTHORITY_REL = SUCCESSOR_AUTHORITY_BUNDLE_REL / "SUCCESSOR_EXECUTION_AUTHORITY.json"
SUCCESSOR_AUTHORITY_COMPANION_REL = (
    SUCCESSOR_AUTHORITY_BUNDLE_REL / "SUCCESSOR_EXECUTION_AUTHORITY.sha256"
)
SUCCESSOR_MANAGER_ISSUANCE_REL = SUCCESSOR_AUTHORITY_BUNDLE_REL / "MANAGER_ISSUANCE.json"
SUCCESSOR_AUTHORITY_BUNDLE_FILES = {
    SUCCESSOR_AUTHORITY_REL.name,
    SUCCESSOR_AUTHORITY_COMPANION_REL.name,
    SUCCESSOR_MANAGER_ISSUANCE_REL.name,
}
STATE_ROOT_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/state"
)
PREDECESSOR_LEDGER_REL = STATE_ROOT_REL / "RUN_LEDGER.json"
SUCCESSOR_LEDGER_REL = STATE_ROOT_REL / "RUN_LEDGER.successor-v2.json"
SUCCESSOR_LEDGER_LOCK_REL = STATE_ROOT_REL / "RUN_LEDGER.successor-v2.lock"
PREDECESSOR_TERMINAL_REL = (
    STATE_ROOT_REL
    / PREDECESSOR_RUN_ID
    / "terminal.bundle/TERMINAL.json"
)
SUCCESSOR_DECISION_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery/"
    "successor_manager_decision_v1/SUCCESSOR_DECISION.json"
)
ACCEPTED_ENVIRONMENT_AUDIT_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/"
    "environment_recertification_v1/ENVIRONMENT_AUDIT.json"
)
ENVIRONMENT_REPRODUCTION_REL = ACCEPTED_ENVIRONMENT_AUDIT_REL.with_name("REPRODUCE.json")
ACCEPTED_RECOVERY_HANDOFF_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery/"
    "version_identity_remediation_v1/RECOVERY_HANDOFF.json"
)
V1_AUTHORITY_ARCHIVE_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery/"
    "successor_policy_v1/v1_authority_archive/EXECUTION_AUTHORITY.json"
)
POST_POLICY_AUDIT_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/recovery/"
    "successor_policy_v1/POST_POLICY_ENVIRONMENT_AUDIT.json"
)
POLICY_MIGRATION_HANDOFF_REL = POST_POLICY_AUDIT_REL.with_name("POLICY_MIGRATION_HANDOFF.json")
POST_POLICY_REVIEW_REL = Path(
    "evidence/review/verification_baseline_successor_policy_scale32_v1/decision.json"
)
PUBLISHED_ROOT_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/runs"
)
BLOCKER_ROOT_REL = Path(
    "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline/blockers"
)

AUTHORITY_KEYS = {
    "schema_version",
    "kind",
    "contract_id",
    "stage",
    "run_id",
    "execution_authorized",
    "command",
    "scope",
    "inputs",
    "seeds",
    "device",
    "software_versions",
    "environment",
    "environment_sha256",
    "authority_binding_sha256",
    "timeout_seconds",
    "model_counts",
    "integrity",
}
SUCCESSOR_AUTHORITY_KEYS = AUTHORITY_KEYS | {"successor_policy"}
SUCCESSOR_POLICY_KEYS = {
    "recovery_generation",
    "decision_sha256",
    "post_policy_environment_audit_sha256",
    "independent_review_sha256",
    "runner_sha256",
    "executor_sha256",
    "software_identity_helper_sha256",
    "predecessor_ledger_sha256",
    "predecessor_terminal_sha256",
}
MANAGER_ISSUANCE_KEYS = {
    "schema_version",
    "kind",
    "contract_id",
    "stage",
    "stage_closing",
    "recovery_generation",
    "run_id",
    "authority",
    "companion",
    "final_path",
    "slot_identity",
    "immutable_bindings",
    "publication",
    "authority_effect",
    "exact_command",
    "conditional_option",
    "predecessor_relationship",
    "claim_boundary",
}
MANAGER_ISSUANCE_BINDING_KEYS = {
    "accepted_policy_migration_handoff_sha256",
    "accepted_post_policy_environment_audit_sha256",
    "candidate_free_executor_sha256",
    "independent_policy_review_projection_sha256",
    "manager_successor_decision_sha256",
    "runner_sha256",
    "sealed_v1_authority_sha256",
    "sealed_v1_run_ledger_sha256",
    "sealed_v1_terminal_sha256",
    "software_identity_helper_sha256",
    "successor_policy_tests_sha256",
    "versioned_successor_ledger_sha256",
}
SCOPE_KEYS = {
    "mode",
    "mechanism",
    "mechanism_disabled",
    "model_repository",
    "model_revision",
    "datasets",
    "layer_count",
    "required_outputs",
    "smoke_token_limit",
}
INPUT_RELS = (
    PROMPT_MANIFEST_REL,
    QUALITY_CONFIG_REL,
    QUALITY_REQUIREMENTS_REL,
    EXECUTOR_REL,
    SOFTWARE_IDENTITY_REL,
)
REQUIRED_OUTPUTS = (
    "ordered_layer_traces.json",
    "final_outputs.json",
    "results.json",
    "run_contract.json",
    "input_observations.json",
)
ENVIRONMENT_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "TMPDIR",
    "TZ",
    "XDG_CACHE_HOME",
)
FORBIDDEN_COMMAND_FRAGMENTS = (
    "candidate",
    "diagnostic",
    "localize",
    "discriminator",
    "synth",
    "openroad",
    "benchmark",
    "prototype",
    "signoff",
)
FORBIDDEN_EXECUTOR_MARKERS = (
    "ace2_down_projection_residual_fusion_hook",
    "ace2_cross_layer_error_carry_hook",
    "--candidate-evidence",
    "--diagnostic-rope-mechanism",
    "ace2_full_model_fixed_point",
)
ALLOWED_EXECUTOR_OPTIONS = {
    "--mode",
    "--output-dir",
    "--smoke-token-limit",
}
FORBIDDEN_EXECUTOR_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "os.system",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.run",
}
FAILPOINTS = {
    "after_reservation_record",
    "after_reservation",
    "after_execution_mark",
    "after_bundle_publish",
}

EXPECTED_SUCCESSOR_DECISION_SHA256 = "e6abadc30f1ad97012e012b7eb8a843d46ae390c1ea397d538d8677f2686846b"
EXPECTED_PREDECESSOR_LEDGER_SHA256 = "4945a5cb78661bb2289428d5baa6e0c3b0d72bd20842c1dca6810710a928464f"
EXPECTED_PREDECESSOR_TERMINAL_SHA256 = "742cd3c2af41773534dd8b3010657ea5595bda0d421a098f7d0ec0e63c890ffe"
EXPECTED_V1_AUTHORITY_SHA256 = "194ec47d40615891d433bdf9dfacf8982db346e36ff5ceebde55f5079e041c75"
EXPECTED_V1_AUTHORITY_COMPANION_SHA256 = "e8a6c8bbd11a5bafc0da273650fdf0b22198f035df165f386c3bfa30ec1f8646"
EXPECTED_V1_MANAGER_ISSUANCE_SHA256 = "29b05ed86cb5e19d805283548b581c17e89cf37f8ad8aed8908e58af489344c1"
EXPECTED_SUCCESSOR_LEDGER_SHA256 = "b343e3f1acb45e912d620f45155b42752319dc90da3e5c503e53c2ec784f629c"
EXPECTED_ENVIRONMENT_AUDIT_SHA256 = "4c3050acec5142018e7815257ce589fefe3caf4cd327d9df5aab7d299e179eb0"
EXPECTED_ENVIRONMENT_REPRODUCTION_SHA256 = "4179106c36cd638e58b58a7938dac51e8cd2dc413b152e625a2895fbd77ab4a8"
EXPECTED_RECOVERY_HANDOFF_SHA256 = "ca69abebe6880b6034d8413018b87b84e43db4e13faa43a78ec630b8de5eb7a2"
EXPECTED_EXECUTOR_SHA256 = "f950d6e9b8973a9fda5839e2471f4fc3aac38eba71afce6ab38970b9ce4e4620"
EXPECTED_SOFTWARE_IDENTITY_SHA256 = "516825a3bf8229495408328ab339373f8ebefdfdca079930886b82d57be46b94"
PRE_POLICY_RUNNER_SHA256 = "10fc7749aa0540d37d2e01e1cde738db46a47bc41310d68ec74b6350c3c78a24"
MIGRATION_RUNNER_RECORD = {
    "path": "tools/run_dynamic_scale32_verification_baseline.py",
    "bytes": 121060,
    "sha256": "535d82d4f5de421306c337ab44952a69847126f5f02142e5746d59369a06d6ea",
}

AUTHORITY_SNAPSHOT_NAME = "EXECUTION_AUTHORITY.json"
RESERVATION_NAME = "RESERVATION.json"
FINAL_LEDGER_NAME = "FINAL_RUN_LEDGER.json"
LAUNCH_ENVIRONMENT_NAME = "LAUNCH_ENVIRONMENT.json"
RESERVED_INPUTS_NAME = "reserved_inputs"


class BaselineError(RuntimeError):
    """A fail-closed baseline-runner error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot load JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_json_bytes(path: Path, content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot load JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_durable(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(root)


def write_companion_pair(directory: Path, name: str, content: bytes) -> dict[str, Any]:
    path = directory / name
    digest = sha256_bytes(content)
    write_durable(path, content)
    companion = path.with_suffix(".sha256")
    write_durable(companion, f"{digest}  {path.name}\n".encode("ascii"))
    return {"path": path.name, "bytes": len(content), "sha256": digest}


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} keys must be exactly {sorted(expected)}")


def artifact(root: Path, path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def canonical_integrity(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return sha256_bytes(encoded)


def detached_canonical_integrity(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.pop("integrity", None)
    return sha256_bytes(canonical_json_bytes(clone))


def verify_companion(path: Path, companion: Path) -> None:
    require(companion.is_file(), f"missing companion hash: {companion}")
    fields = companion.read_text(encoding="ascii").strip().split()
    require(len(fields) == 2, f"invalid companion hash: {companion}")
    require(fields[1] == path.name, f"companion names the wrong artifact: {companion}")
    require(fields[0] == sha256_file(path), f"companion hash mismatch: {path}")


def contains_forbidden_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "candidate_model" and item == 0:
                continue
            if "candidate" in lowered or contains_forbidden_value(item):
                return True
        return False
    if isinstance(value, list):
        return any(contains_forbidden_value(item) for item in value)
    if isinstance(value, str):
        return "candidate" in value.lower()
    return False


class DynamicScale32BaselineRunner:
    """One-command, verification-only baseline runner."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.pipeline_path = self.root / PIPELINE_REL
        self.review_path = self.root / RTL_REVIEW_REL
        self.review_companion_path = self.root / RTL_REVIEW_COMPANION_REL
        self.proposal_path = self.root / PROPOSAL_REL
        self.proposal_companion_path = self.root / PROPOSAL_COMPANION_REL
        self.manager_freeze_path = self.root / MANAGER_FREEZE_REL
        self.prompt_manifest_path = self.root / PROMPT_MANIFEST_REL
        self.quality_config_path = self.root / QUALITY_CONFIG_REL
        self.requirements_path = self.root / QUALITY_REQUIREMENTS_REL
        self.executor_path = self.root / EXECUTOR_REL
        self.software_identity_path = self.root / SOFTWARE_IDENTITY_REL
        self.authority_path = self.root / AUTHORITY_REL
        self.v1_manager_issuance_path = self.root / V1_MANAGER_ISSUANCE_REL
        self.legacy_root_successor_authority_path = self.root / LEGACY_ROOT_SUCCESSOR_AUTHORITY_REL
        self.legacy_root_successor_companion_path = self.legacy_root_successor_authority_path.with_suffix(".sha256")
        self.recovery_root = self.root / RECOVERY_ROOT_REL
        self.successor_authority_bundle_path = self.root / SUCCESSOR_AUTHORITY_BUNDLE_REL
        self.successor_authority_path = self.root / SUCCESSOR_AUTHORITY_REL
        self.successor_authority_companion_path = self.root / SUCCESSOR_AUTHORITY_COMPANION_REL
        self.successor_manager_issuance_path = self.root / SUCCESSOR_MANAGER_ISSUANCE_REL
        self.state_root = self.root / STATE_ROOT_REL
        self.ledger_path = self.state_root / "RUN_LEDGER.json"
        self.lock_path = self.state_root / "RUN_LEDGER.lock"
        self.predecessor_ledger_path = self.root / PREDECESSOR_LEDGER_REL
        self.successor_ledger_path = self.root / SUCCESSOR_LEDGER_REL
        self.successor_lock_path = self.root / SUCCESSOR_LEDGER_LOCK_REL
        self.predecessor_terminal_path = self.root / PREDECESSOR_TERMINAL_REL
        self.successor_decision_path = self.root / SUCCESSOR_DECISION_REL
        self.accepted_environment_audit_path = self.root / ACCEPTED_ENVIRONMENT_AUDIT_REL
        self.environment_reproduction_path = self.root / ENVIRONMENT_REPRODUCTION_REL
        self.accepted_recovery_handoff_path = self.root / ACCEPTED_RECOVERY_HANDOFF_REL
        self.v1_authority_archive_path = self.root / V1_AUTHORITY_ARCHIVE_REL
        self.post_policy_audit_path = self.root / POST_POLICY_AUDIT_REL
        self.policy_migration_handoff_path = self.root / POLICY_MIGRATION_HANDOFF_REL
        self.post_policy_review_path = self.root / POST_POLICY_REVIEW_REL
        self.published_root = self.root / PUBLISHED_ROOT_REL
        self.blocker_root = self.root / BLOCKER_ROOT_REL
        self.runner_path = self.root / "tools/run_dynamic_scale32_verification_baseline.py"
        self.successor_policy_test_path = self.root / "tools/test_dynamic_scale32_successor_policy.py"

    def _normalized_environment(self, seeds: dict[str, Any]) -> dict[str, str]:
        python_seed = seeds.get("python_seed")
        require(isinstance(python_seed, int), "frozen python seed is missing")
        interpreter_bin = str(Path(sys.executable).parent)
        return {
            "HOME": "{RUN_HOME}",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.pathsep.join((interpreter_bin, "/usr/bin", "/bin")),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": str(python_seed),
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": "{RUN_TMP}",
            "TZ": "UTC",
            "XDG_CACHE_HOME": "{RUN_HOME}/.cache",
        }

    def _render_environment(
        self,
        template: dict[str, str],
        state_dir: Path,
    ) -> dict[str, str]:
        replacements = {
            "{RUN_HOME}": str(state_dir / "runtime" / "home"),
            "{RUN_TMP}": str(state_dir / "runtime" / "tmp"),
        }
        rendered: dict[str, str] = {}
        for key, value in template.items():
            rendered_value = value
            for placeholder, replacement in replacements.items():
                rendered_value = rendered_value.replace(placeholder, replacement)
            require("{" not in rendered_value and "}" not in rendered_value, f"unresolved environment placeholder: {key}")
            rendered[key] = rendered_value
        return rendered

    def _render_reserved_command(
        self,
        authority: dict[str, Any],
        state_dir: Path,
    ) -> list[str]:
        command = list(authority["command"])
        require(
            len(command) >= 2 and command[1] == EXECUTOR_REL.as_posix(),
            "authority command does not name the frozen executor",
        )
        command[1] = str(state_dir / RESERVED_INPUTS_NAME / EXECUTOR_REL)
        command = [
            str(state_dir / "raw") if value == "{OUTPUT_DIR}" else value
            for value in command
        ]
        require(
            all("{" not in value and "}" not in value for value in command),
            "unresolved command placeholder",
        )
        return command

    def _stage_authority(self) -> dict[str, Any]:
        verify_companion(self.review_path, self.review_companion_path)
        verify_companion(self.proposal_path, self.proposal_companion_path)
        pipeline = load_json(self.pipeline_path)
        review = load_json(self.review_path)
        proposal = load_json(self.proposal_path)
        manager_freeze = load_json(self.manager_freeze_path)

        require(pipeline.get("current_stage") == REQUIRED_STAGE, "Manager stage is not verification")
        transitions = [
            item
            for item in pipeline.get("stage_history", [])
            if item.get("at") == MANAGER_TRANSITION_AT
        ]
        require(len(transitions) == 1, "Manager RTL-to-verification transition is missing or duplicated")
        transition = transitions[0]
        require(
            transition.get("by") == "manager"
            and transition.get("from_stage") == "rtl"
            and transition.get("to_stage") == REQUIRED_STAGE,
            "Manager transition fields differ",
        )
        require(review.get("contract_id") == CONTRACT_ID, "RTL review contract differs")
        require(review.get("stage") == "rtl", "RTL review stage differs")
        require(review.get("stage_closing") is True, "RTL review is not stage-closing")
        require(review.get("verdict") == "go", "RTL review verdict is not go")
        require(review.get("status") == "review_complete_stage_closing_true", "RTL review status differs")
        integrity = review.get("integrity", {})
        require(
            integrity.get("canonical_sha256") == canonical_integrity(review),
            "RTL review canonical integrity mismatch",
        )
        require(proposal.get("contract_id") == CONTRACT_ID, "architecture proposal contract differs")
        require(manager_freeze.get("contract_id") == CONTRACT_ID, "Manager freeze contract differs")

        binding = {
            "contract_id": CONTRACT_ID,
            "stage": REQUIRED_STAGE,
            "manager_transition": transition,
            "pipeline": artifact(self.root, self.pipeline_path),
            "rtl_stage_closing_review": artifact(self.root, self.review_path),
            "architecture_proposal": artifact(self.root, self.proposal_path),
            "manager_architecture_freeze": artifact(self.root, self.manager_freeze_path),
        }
        binding["authority_binding_sha256"] = sha256_bytes(canonical_json_bytes(binding))
        return binding

    def _frozen_inputs(self) -> dict[str, Any]:
        prompt = load_json(self.prompt_manifest_path)
        quality = load_json(self.quality_config_path)
        model = prompt.get("model")
        require(model == {
            "repository": quality.get("baseline", {}).get("model_repository"),
            "revision": quality.get("baseline", {}).get("model_revision"),
        }, "model identity differs between frozen inputs")
        seeds = quality.get("determinism")
        software = quality.get("software")
        require(isinstance(seeds, dict) and seeds, "frozen seeds are missing")
        require(isinstance(software, dict) and software, "frozen software pins are missing")
        inputs = [artifact(self.root, self.root / relative) for relative in INPUT_RELS]
        launch_environment = self._normalized_environment(seeds)
        return {
            "model": model,
            "datasets": prompt.get("datasets"),
            "seeds": seeds,
            "software_versions": software,
            "device": "cpu",
            "inputs": inputs,
            "runner": artifact(self.root, self.runner_path),
            "launch_environment": {
                "variables": launch_environment,
                "sha256": sha256_bytes(canonical_json_bytes(launch_environment)),
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
        }

    def _runtime_version_blockers(self, expected: dict[str, str]) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        for name, record in evaluate_distribution_versions(expected).items():
            if not record["matches"]:
                blockers.append(
                    {
                        "code": "software_version_mismatch",
                        "package": name,
                        **record,
                    }
                )
        return blockers

    def _executor_blockers(self) -> list[dict[str, Any]]:
        try:
            source = self.executor_path.read_text(encoding="utf-8")
        except OSError as exc:
            return [
                {
                    "code": "existing_executor_unavailable",
                    "path": EXECUTOR_REL.as_posix(),
                    "detail": str(exc),
                }
            ]
        blockers = []
        markers = [marker for marker in FORBIDDEN_EXECUTOR_MARKERS if marker in source]
        if markers:
            blockers.append(
                {
                    "code": "existing_executor_has_candidate_surface",
                    "path": EXECUTOR_REL.as_posix(),
                    "markers": markers,
                    "detail": "The baseline process would import or expose legacy candidate entrypoints.",
                }
            )
        try:
            tree = ast.parse(source, filename=EXECUTOR_REL.as_posix())
        except SyntaxError as exc:
            blockers.append(
                {
                    "code": "baseline_executor_static_parse_failed",
                    "path": EXECUTOR_REL.as_posix(),
                    "detail": str(exc),
                }
            )
            tree = None
        if tree is not None:
            imported_modules = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            } | {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            forbidden_imports = sorted(
                name
                for name in imported_modules
                if name == "importlib"
                or name.startswith("importlib.")
                or name == "subprocess"
                or name.startswith("subprocess.")
                or "candidate" in name.lower()
                or "diagnostic" in name.lower()
                or "localize" in name.lower()
                or "discriminator" in name.lower()
            )

            def dotted_name(node: ast.AST) -> str | None:
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, ast.Attribute):
                    base = dotted_name(node.value)
                    return f"{base}.{node.attr}" if base else None
                return None

            forbidden_calls = sorted(
                {
                    name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    if (name := dotted_name(node.func)) in FORBIDDEN_EXECUTOR_CALLS
                }
            )
            observed_options = {
                argument.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and dotted_name(node.func) is not None
                and dotted_name(node.func).endswith("add_argument")
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            }
            if forbidden_imports or forbidden_calls or observed_options != ALLOWED_EXECUTOR_OPTIONS:
                blockers.append(
                    {
                        "code": "baseline_executor_not_statically_allowlisted",
                        "path": EXECUTOR_REL.as_posix(),
                        "forbidden_imports": forbidden_imports,
                        "forbidden_calls": forbidden_calls,
                        "expected_options": sorted(ALLOWED_EXECUTOR_OPTIONS),
                        "observed_options": sorted(observed_options),
                        "detail": "The baseline executor must retain the closed, fixed invocation surface.",
                    }
                )
        missing_outputs = [name for name in REQUIRED_OUTPUTS if name not in source]
        if missing_outputs:
            blockers.append(
                {
                    "code": "existing_executor_lacks_required_comparison_outputs",
                    "path": EXECUTOR_REL.as_posix(),
                    "missing_output_markers": missing_outputs,
                    "detail": "The frozen comparison requires ordered layer traces and final outputs before reservation.",
                }
            )
        return blockers

    def _expected_command(self, scope: dict[str, Any]) -> list[str]:
        mode = scope["mode"]
        command = [
            sys.executable,
            EXECUTOR_REL.as_posix(),
            "--mode",
            mode,
            "--output-dir",
            "{OUTPUT_DIR}",
        ]
        if mode == "smoke":
            command.extend(["--smoke-token-limit", str(scope["smoke_token_limit"])])
        return command

    def _validate_authority(
        self,
        authority: dict[str, Any],
        stage_authority: dict[str, Any],
        frozen: dict[str, Any],
    ) -> None:
        successor = authority.get("kind") == SUCCESSOR_AUTHORITY_KIND
        exact_keys(
            authority,
            SUCCESSOR_AUTHORITY_KEYS if successor else AUTHORITY_KEYS,
            "execution authority",
        )
        require(authority["schema_version"] == SCHEMA_VERSION, "execution authority schema differs")
        require(
            authority["kind"] == (SUCCESSOR_AUTHORITY_KIND if successor else AUTHORITY_KIND),
            "execution authority kind differs",
        )
        require(authority["contract_id"] == CONTRACT_ID, "execution authority contract differs")
        require(authority["stage"] == REQUIRED_STAGE, "execution authority stage differs")
        require(authority["execution_authorized"] is True, "baseline execution is not authorized")
        require(RUN_ID_RE.fullmatch(authority["run_id"]) is not None, "invalid run_id")
        require(authority["authority_binding_sha256"] == stage_authority["authority_binding_sha256"], "execution authority is stale")
        require(authority["model_counts"] == {"baseline_model": 1, "candidate_model": 0}, "model counts differ")
        require(authority["device"] == "cpu", "only the frozen CPU device is permitted")
        require(authority["seeds"] == frozen["seeds"], "seed binding differs")
        require(authority["software_versions"] == frozen["software_versions"], "software binding differs")
        require(authority["inputs"] == frozen["inputs"], "model/data/config input binding differs")
        require(isinstance(authority["environment"], dict), "environment must be an object")
        exact_keys(authority["environment"], set(ENVIRONMENT_KEYS), "environment")
        require(
            all(isinstance(key, str) and isinstance(value, str) for key, value in authority["environment"].items()),
            "environment keys and values must be strings",
        )
        require(
            authority["environment"] == frozen["launch_environment"]["variables"],
            "normalized launch environment differs",
        )
        require(
            authority["environment_sha256"] == frozen["launch_environment"]["sha256"]
            == sha256_bytes(canonical_json_bytes(authority["environment"])),
            "launch environment hash differs",
        )
        require(
            isinstance(authority["timeout_seconds"], int)
            and 1 <= authority["timeout_seconds"] <= 86400,
            "timeout is outside 1..86400 seconds",
        )

        scope = authority["scope"]
        require(isinstance(scope, dict), "scope must be an object")
        exact_keys(scope, SCOPE_KEYS, "scope")
        require(scope["mode"] in {"smoke", "official"}, "unsupported baseline mode")
        require(scope["mechanism"] == CONTRACT_ID, "scope mechanism differs")
        require(scope["mechanism_disabled"] is True, "mechanism is not disabled")
        require(scope["model_repository"] == frozen["model"]["repository"], "model repository differs")
        require(scope["model_revision"] == frozen["model"]["revision"], "model revision differs")
        require(scope["datasets"] == ["wikitext2", "c4_en_512"], "dataset scope differs")
        require(scope["layer_count"] == 24, "layer count differs")
        require(scope["required_outputs"] == list(REQUIRED_OUTPUTS), "required output set differs")
        if scope["mode"] == "smoke":
            require(
                isinstance(scope["smoke_token_limit"], int)
                and 2 <= scope["smoke_token_limit"] <= 512,
                "smoke token limit is outside 2..512",
            )
        else:
            require(scope["smoke_token_limit"] is None, "official mode cannot set a smoke token limit")

        require(authority["command"] == self._expected_command(scope), "command is not the sole allowlisted invocation")
        lowered_command = " ".join(authority["command"]).lower()
        require(
            not any(fragment in lowered_command for fragment in FORBIDDEN_COMMAND_FRAGMENTS),
            "command contains a forbidden candidate/downstream entrypoint",
        )
        integrity = authority["integrity"]
        require(
            isinstance(integrity, dict)
            and integrity.get("algorithm") == "sha256-canonical-json-v1"
            and integrity.get("canonical_sha256") == canonical_integrity(authority),
            "execution authority canonical integrity mismatch",
        )
        if successor:
            self._validate_successor_authority_policy(
                authority,
                self._load_successor_ledger_readonly(),
            )

    def _validate_successor_manager_issuance(
        self,
        issuance: dict[str, Any],
        authority: dict[str, Any],
        content_by_name: dict[str, bytes],
    ) -> None:
        exact_keys(issuance, MANAGER_ISSUANCE_KEYS, "successor Manager issuance")
        require(issuance["schema_version"] == SCHEMA_VERSION, "successor issuance schema differs")
        require(
            issuance["kind"] == "manager_dynamic_scale32_successor_authority_issuance_v1",
            "successor issuance kind differs",
        )
        require(issuance["contract_id"] == CONTRACT_ID, "successor issuance contract differs")
        require(issuance["stage"] == REQUIRED_STAGE, "successor issuance stage differs")
        require(issuance["stage_closing"] is False, "successor issuance cannot close the stage")
        require(
            issuance["recovery_generation"] == SUCCESSOR_GENERATION,
            "successor issuance generation differs",
        )
        require(issuance["run_id"] == authority.get("run_id"), "successor issuance run_id differs")
        require(
            issuance["final_path"] == SUCCESSOR_MANAGER_ISSUANCE_REL.as_posix(),
            "successor issuance final path is not canonical",
        )

        authority_record = issuance.get("authority")
        require(isinstance(authority_record, dict), "successor issuance authority binding is missing")
        exact_keys(authority_record, {"final_path", "sha256"}, "successor issuance authority binding")
        require(
            authority_record["final_path"] == SUCCESSOR_AUTHORITY_REL.as_posix(),
            "successor authority final path is not canonical",
        )
        require(
            authority_record["sha256"]
            == sha256_bytes(content_by_name[SUCCESSOR_AUTHORITY_REL.name]),
            "successor issuance authority hash differs",
        )
        companion_record = issuance.get("companion")
        require(isinstance(companion_record, dict), "successor issuance companion binding is missing")
        exact_keys(companion_record, {"final_path", "sha256"}, "successor issuance companion binding")
        require(
            companion_record["final_path"] == SUCCESSOR_AUTHORITY_COMPANION_REL.as_posix(),
            "successor companion final path is not canonical",
        )
        require(
            companion_record["sha256"]
            == sha256_bytes(content_by_name[SUCCESSOR_AUTHORITY_COMPANION_REL.name]),
            "successor issuance companion file hash differs",
        )

        slot = issuance.get("slot_identity")
        require(isinstance(slot, dict), "successor issuance slot binding is missing")
        exact_keys(
            slot,
            {"ledger_path", "ledger_sha256", "recovery_generation", "slot_index", "status_required_before_and_after_publication"},
            "successor issuance slot binding",
        )
        require(slot["ledger_path"] == SUCCESSOR_LEDGER_REL.as_posix(), "successor issuance ledger path differs")
        require(slot["ledger_sha256"] == EXPECTED_SUCCESSOR_LEDGER_SHA256, "successor issuance ledger hash differs")
        require(slot["recovery_generation"] == SUCCESSOR_GENERATION, "successor issuance slot generation differs")
        require(slot["slot_index"] == 0, "successor issuance slot index differs")
        require(
            slot["status_required_before_and_after_publication"] == "available",
            "successor issuance publication must not consume the slot",
        )

        publication = issuance.get("publication")
        require(isinstance(publication, dict), "successor issuance publication contract is missing")
        require(
            publication
            == {
                "allowed_artifact_count": 3,
                "atomic_required": True,
                "becomes_effective_only_at_final_paths": True,
                "manager_self_review_can_close_gate": False,
                "requires_fresh_independent_reviewer_acceptance_before_publication": True,
            },
            "successor issuance publication contract differs",
        )
        require(
            issuance.get("authority_effect")
            == {
                "candidate_model": 0,
                "cumulative_baseline_process_attempt_ceiling": 2,
                "execution_authorized": True,
                "publication_is_execution": False,
                "publication_is_l2_acceptance": False,
                "publication_reserves_or_consumes_slot": False,
            },
            "successor issuance authority effect differs",
        )
        require(issuance.get("exact_command") == authority.get("command"), "successor issuance command differs")

        bindings = issuance.get("immutable_bindings")
        require(isinstance(bindings, dict), "successor issuance immutable bindings are missing")
        exact_keys(bindings, MANAGER_ISSUANCE_BINDING_KEYS, "successor issuance immutable bindings")
        policy = authority["successor_policy"]
        require(
            bindings["accepted_policy_migration_handoff_sha256"]
            == sha256_file(self.policy_migration_handoff_path),
            "successor issuance policy handoff binding differs",
        )
        require(
            bindings["accepted_post_policy_environment_audit_sha256"]
            == policy["post_policy_environment_audit_sha256"]
            == sha256_file(self.post_policy_audit_path),
            "successor issuance post-policy audit binding differs",
        )
        require(
            bindings["independent_policy_review_projection_sha256"]
            == policy["independent_review_sha256"]
            == sha256_file(self.post_policy_review_path),
            "successor issuance independent review binding differs",
        )
        require(
            bindings["manager_successor_decision_sha256"] == EXPECTED_SUCCESSOR_DECISION_SHA256,
            "successor issuance Manager decision binding differs",
        )
        require(bindings["runner_sha256"] == sha256_file(self.runner_path), "successor issuance runner binding differs")
        require(bindings["candidate_free_executor_sha256"] == EXPECTED_EXECUTOR_SHA256, "successor issuance executor binding differs")
        require(
            bindings["software_identity_helper_sha256"] == EXPECTED_SOFTWARE_IDENTITY_SHA256,
            "successor issuance helper binding differs",
        )
        require(bindings["sealed_v1_authority_sha256"] == EXPECTED_V1_AUTHORITY_SHA256, "successor issuance v1 authority binding differs")
        require(bindings["sealed_v1_run_ledger_sha256"] == EXPECTED_PREDECESSOR_LEDGER_SHA256, "successor issuance v1 ledger binding differs")
        require(bindings["sealed_v1_terminal_sha256"] == EXPECTED_PREDECESSOR_TERMINAL_SHA256, "successor issuance v1 terminal binding differs")
        require(bindings["versioned_successor_ledger_sha256"] == EXPECTED_SUCCESSOR_LEDGER_SHA256, "successor issuance v2 ledger binding differs")
        require(
            bindings["successor_policy_tests_sha256"] == sha256_file(self.successor_policy_test_path),
            "successor issuance policy-test binding differs",
        )

    def _successor_authority_bundle_snapshot_at(
        self,
        directory: Path,
        ledger: dict[str, Any],
        *,
        final_required: bool,
    ) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        if final_required:
            require(directory == self.successor_authority_bundle_path, "successor authority directory is not canonical")
        else:
            require(directory.parent == self.recovery_root, "successor staging directory is not a sibling of final")
            require(
                directory.name.startswith(f".staging-{SUCCESSOR_AUTHORITY_BUNDLE_NAME}-"),
                "successor staging directory name differs",
            )
        require(directory.is_dir() and not directory.is_symlink(), f"successor authority bundle missing: {directory}")
        require(
            not self.legacy_root_successor_authority_path.exists()
            and not self.legacy_root_successor_companion_path.exists(),
            "unversioned root successor authority is forbidden",
        )
        self._require_frozen_artifact(
            self.authority_path,
            EXPECTED_V1_AUTHORITY_SHA256,
            "sealed v1 authority",
        )
        self._require_frozen_artifact(
            self.v1_manager_issuance_path,
            EXPECTED_V1_MANAGER_ISSUANCE_SHA256,
            "sealed v1 Manager issuance",
            companion=False,
        )
        names = {item.name for item in directory.iterdir()}
        require(names == SUCCESSOR_AUTHORITY_BUNDLE_FILES, "successor authority bundle contents differ")
        for item in directory.iterdir():
            require(item.is_file() and not item.is_symlink(), f"successor authority bundle entry is not a regular file: {item}")
        content_by_name = {
            name: (directory / name).read_bytes()
            for name in sorted(SUCCESSOR_AUTHORITY_BUNDLE_FILES)
        }
        authority_bytes = content_by_name[SUCCESSOR_AUTHORITY_REL.name]
        companion_bytes = content_by_name[SUCCESSOR_AUTHORITY_COMPANION_REL.name]
        companion_fields = companion_bytes.decode("ascii").strip().split()
        require(
            companion_fields == [sha256_bytes(authority_bytes), SUCCESSOR_AUTHORITY_REL.name],
            "successor authority companion differs",
        )
        authority = load_json_bytes(directory / SUCCESSOR_AUTHORITY_REL.name, authority_bytes)
        require(authority.get("kind") == SUCCESSOR_AUTHORITY_KIND, "successor authority kind is required")
        self._validate_successor_ledger(ledger)
        self._validate_successor_authority_policy(authority, ledger)
        issuance = load_json_bytes(
            directory / SUCCESSOR_MANAGER_ISSUANCE_REL.name,
            content_by_name[SUCCESSOR_MANAGER_ISSUANCE_REL.name],
        )
        self._validate_successor_manager_issuance(issuance, authority, content_by_name)
        files = [
            {
                "path": (SUCCESSOR_AUTHORITY_BUNDLE_REL / name).as_posix(),
                "bytes": len(content_by_name[name]),
                "sha256": sha256_bytes(content_by_name[name]),
            }
            for name in sorted(content_by_name)
        ]
        bundle = {
            "path": SUCCESSOR_AUTHORITY_BUNDLE_REL.as_posix(),
            "files": files,
        }
        bundle["manifest_sha256"] = sha256_bytes(canonical_json_bytes(bundle))
        source = next(item for item in files if item["path"] == SUCCESSOR_AUTHORITY_REL.as_posix())
        return authority, authority_bytes, {**source, "bundle": bundle}

    def publish_successor_authority_bundle(
        self,
        content_by_name: dict[str, bytes],
    ) -> dict[str, Any]:
        require(set(content_by_name) == SUCCESSOR_AUTHORITY_BUNDLE_FILES, "successor publication file set differs")
        ledger = self._load_successor_ledger_readonly()
        final = self.successor_authority_bundle_path
        if final.exists():
            authority, _, source = self._successor_authority_bundle_snapshot_at(
                final,
                ledger,
                final_required=True,
            )
            existing = {name: (final / name).read_bytes() for name in SUCCESSOR_AUTHORITY_BUNDLE_FILES}
            require(existing == content_by_name, "conflicting successor authority publication is forbidden")
            return {"created": False, "authority": authority, "source": source}
        self.recovery_root.mkdir(parents=True, exist_ok=True)
        fsync_directory(self.recovery_root.parent)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".staging-{SUCCESSOR_AUTHORITY_BUNDLE_NAME}-",
                dir=self.recovery_root,
            )
        )
        fsync_directory(self.recovery_root)
        for name in sorted(content_by_name):
            write_durable(staging / name, content_by_name[name])
        fsync_directory(staging)
        self._successor_authority_bundle_snapshot_at(staging, ledger, final_required=False)
        self._failpoint("before_successor_authority_bundle_publish")
        os.replace(staging, final)
        fsync_directory(self.recovery_root)
        self._failpoint("after_successor_authority_bundle_publish")
        authority, _, source = self._successor_authority_bundle_snapshot_at(
            final,
            ledger,
            final_required=True,
        )
        return {"created": True, "authority": authority, "source": source}

    def _authority_snapshot(self) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        if self.successor_ledger_path.is_file():
            return self._successor_authority_bundle_snapshot_at(
                self.successor_authority_bundle_path,
                self._load_successor_ledger_readonly(),
                final_required=True,
            )
        path = self.authority_path
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise BaselineError(
                f"cannot read execution authority {path}: {exc}"
            ) from exc
        authority = load_json_bytes(path, content)
        source = {
            "path": path.relative_to(self.root).as_posix(),
            "bytes": len(content),
            "sha256": sha256_bytes(content),
        }
        return authority, content, source

    def _validated_authority_snapshot(
        self,
        preflight: dict[str, Any],
    ) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        authority, content, source = self._authority_snapshot()
        require(
            source == preflight.get("execution_authority"),
            "execution authority changed after preflight",
        )
        stage_authority = self._stage_authority()
        frozen = self._frozen_inputs()
        self._validate_authority(authority, stage_authority, frozen)
        require(
            stage_authority == preflight.get("stage_authority"),
            "stage authority changed after preflight",
        )
        require(
            frozen == preflight.get("frozen_provenance"),
            "frozen provenance changed after preflight",
        )
        return authority, content, source

    def _revalidate_successor_authority_bundle_at_reservation_boundary(
        self,
        ledger: dict[str, Any],
        expected_authority: dict[str, Any],
        expected_content: bytes,
        expected_source: dict[str, Any],
    ) -> None:
        authority, content, source = self._successor_authority_bundle_snapshot_at(
            self.successor_authority_bundle_path,
            ledger,
            final_required=True,
        )
        require(
            authority == expected_authority
            and content == expected_content
            and source == expected_source,
            "complete successor authority bundle changed at reservation boundary",
        )

    def _blocked_after_ready_preflight(
        self,
        preflight: dict[str, Any],
        detail: str,
    ) -> dict[str, Any]:
        report = copy.deepcopy(preflight)
        report["generated_at_utc"] = now_utc()
        report["status"] = "blocked_fail_closed"
        report["reservation_created"] = False
        report["execution_counts"] = {"baseline_model": 0, "candidate_model": 0}
        report["recovery_state"] = "no_reservation_no_execution"
        report["blockers"] = [
            {
                "code": "validated_authority_snapshot_changed",
                "detail": detail,
            }
        ]
        report.pop("preflight_sha256", None)
        report["preflight_sha256"] = sha256_bytes(canonical_json_bytes(report))
        return report

    def preflight(self) -> dict[str, Any]:
        blockers: list[dict[str, Any]] = []
        stage_authority: dict[str, Any] | None = None
        frozen: dict[str, Any] | None = None
        try:
            stage_authority = self._stage_authority()
        except BaselineError as exc:
            blockers.append({"code": "stage_authority_invalid", "detail": str(exc)})
        try:
            frozen = self._frozen_inputs()
        except BaselineError as exc:
            blockers.append({"code": "frozen_input_invalid", "detail": str(exc)})

        blockers.extend(self._executor_blockers())
        authority = None
        authority_source = None
        active_authority_path = (
            self.successor_authority_bundle_path
            if self.successor_ledger_path.is_file()
            else self.authority_path
        )
        active_authority_exists = (
            active_authority_path.is_dir()
            if self.successor_ledger_path.is_file()
            else active_authority_path.is_file()
        )
        if not active_authority_exists:
            blockers.append(
                {
                    "code": "missing_manager_frozen_baseline_command_authority",
                    "path": active_authority_path.relative_to(self.root).as_posix(),
                    "detail": "No Manager-issued object freezes one command, mode/token scope, inputs, seeds, device, software, timeout, and baseline_model=1/candidate_model=0.",
                }
            )
        elif stage_authority is not None and frozen is not None:
            try:
                authority, _, authority_source = self._authority_snapshot()
                self._validate_authority(authority, stage_authority, frozen)
            except BaselineError as exc:
                blockers.append({"code": "execution_authority_invalid", "detail": str(exc)})
                authority = None
                authority_source = None

        if frozen is not None and authority is not None:
            blockers.extend(self._runtime_version_blockers(frozen["software_versions"]))
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "dynamic_scale32_verification_baseline_preflight",
            "generated_at_utc": now_utc(),
            "contract_id": CONTRACT_ID,
            "stage": REQUIRED_STAGE,
            "status": "blocked_fail_closed" if blockers else "ready",
            "reservation_created": False,
            "execution_counts": {"baseline_model": 0, "candidate_model": 0},
            "stage_authority": stage_authority,
            "frozen_provenance": frozen,
            "execution_authority": authority_source if authority is not None else None,
            "candidate_entrypoint_interlock": {
                "enabled": True,
                "forbidden_command_fragments": list(FORBIDDEN_COMMAND_FRAGMENTS),
                "forbidden_executor_markers": list(FORBIDDEN_EXECUTOR_MARKERS),
                "candidate_model_count": 0,
            },
            "blockers": blockers,
            "recovery_state": "no_reservation_no_execution",
        }
        report["preflight_sha256"] = sha256_bytes(canonical_json_bytes(report))
        return report

    def _publish_blocker(self, report: dict[str, Any]) -> dict[str, Any]:
        self.blocker_root.mkdir(parents=True, exist_ok=True)
        fsync_directory(self.blocker_root.parent)
        suffix = report["preflight_sha256"][:16]
        final = self.blocker_root / f"preflight-{suffix}.bundle"
        if final.exists():
            return self._verify_blocker(final)
        staging = self.blocker_root / f".staging-preflight-{suffix}-{os.getpid()}"
        require(not staging.exists(), f"blocker staging collision: {staging}")
        staging.mkdir()
        blocker_path = staging / "BLOCKER.json"
        blocker_bytes = json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        blocker_sha256 = sha256_bytes(blocker_bytes)
        write_durable(blocker_path, blocker_bytes)
        write_durable(
            staging / "BLOCKER.sha256",
            f"{blocker_sha256}  BLOCKER.json\n".encode("ascii"),
        )
        fsync_directory(staging)
        os.replace(staging, final)
        fsync_directory(self.blocker_root)
        return self._verify_blocker(final)

    def _verify_blocker(self, bundle: Path) -> dict[str, Any]:
        require(bundle.is_dir(), f"blocker bundle missing: {bundle}")
        require(
            {item.name for item in bundle.iterdir()} == {"BLOCKER.json", "BLOCKER.sha256"},
            "blocker bundle contents differ",
        )
        verify_companion(bundle / "BLOCKER.json", bundle / "BLOCKER.sha256")
        report = load_json(bundle / "BLOCKER.json")
        require(report.get("status") == "blocked_fail_closed", "blocker status differs")
        require(report.get("reservation_created") is False, "blocked preflight claims a reservation")
        require(report.get("execution_counts") == {"baseline_model": 0, "candidate_model": 0}, "blocked execution counts differ")
        return {
            "bundle": bundle,
            "artifact": bundle / "BLOCKER.json",
            "artifact_sha256": sha256_file(bundle / "BLOCKER.json"),
            "report": report,
        }

    @contextlib.contextmanager
    def _locked_ledger(self) -> Iterator[dict[str, Any]]:
        self.state_root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.ledger_path.exists():
                ledger = load_json(self.ledger_path)
                require(ledger.get("schema_version") == SCHEMA_VERSION, "ledger schema differs")
                require(isinstance(ledger.get("runs"), dict), "ledger runs must be an object")
            else:
                ledger = {"schema_version": SCHEMA_VERSION, "runs": {}}
            yield ledger

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        atomic_write_json(self.ledger_path, ledger)

    def _require_frozen_artifact(
        self,
        path: Path,
        expected_sha256: str,
        label: str,
        *,
        companion: bool = True,
    ) -> dict[str, Any]:
        require(path.is_file(), f"missing {label}: {path}")
        require(sha256_file(path) == expected_sha256, f"{label} hash mismatch")
        if companion:
            verify_companion(path, path.with_suffix(".sha256"))
        return artifact(self.root, path)

    def _migration_bindings(self) -> tuple[dict[str, Any], dict[str, Any]]:
        predecessor = self._require_frozen_artifact(
            self.predecessor_ledger_path,
            EXPECTED_PREDECESSOR_LEDGER_SHA256,
            "sealed predecessor ledger",
            companion=False,
        )
        terminal = self._require_frozen_artifact(
            self.predecessor_terminal_path,
            EXPECTED_PREDECESSOR_TERMINAL_SHA256,
            "sealed predecessor terminal",
            companion=False,
        )
        decision = self._require_frozen_artifact(
            self.successor_decision_path,
            EXPECTED_SUCCESSOR_DECISION_SHA256,
            "Manager successor decision",
        )
        environment_audit = self._require_frozen_artifact(
            self.accepted_environment_audit_path,
            EXPECTED_ENVIRONMENT_AUDIT_SHA256,
            "accepted environment audit",
        )
        environment_reproduction = self._require_frozen_artifact(
            self.environment_reproduction_path,
            EXPECTED_ENVIRONMENT_REPRODUCTION_SHA256,
            "environment reproduction",
        )
        recovery = self._require_frozen_artifact(
            self.accepted_recovery_handoff_path,
            EXPECTED_RECOVERY_HANDOFF_SHA256,
            "accepted recovery handoff",
        )
        executor = self._require_frozen_artifact(
            self.executor_path,
            EXPECTED_EXECUTOR_SHA256,
            "baseline executor",
            companion=False,
        )
        helper = self._require_frozen_artifact(
            self.software_identity_path,
            EXPECTED_SOFTWARE_IDENTITY_SHA256,
            "software identity helper",
            companion=False,
        )
        original_authority = self._require_frozen_artifact(
            self.authority_path,
            EXPECTED_V1_AUTHORITY_SHA256,
            "sealed v1 authority",
        )
        original_companion = self._require_frozen_artifact(
            self.authority_path.with_suffix(".sha256"),
            EXPECTED_V1_AUTHORITY_COMPANION_SHA256,
            "sealed v1 authority companion",
            companion=False,
        )
        self._require_frozen_artifact(
            self.v1_manager_issuance_path,
            EXPECTED_V1_MANAGER_ISSUANCE_SHA256,
            "sealed v1 Manager issuance",
            companion=False,
        )
        archived_authority = self._require_frozen_artifact(
            self.v1_authority_archive_path,
            EXPECTED_V1_AUTHORITY_SHA256,
            "archived v1 authority",
        )
        archived_companion = self._require_frozen_artifact(
            self.v1_authority_archive_path.with_suffix(".sha256"),
            EXPECTED_V1_AUTHORITY_COMPANION_SHA256,
            "archived v1 authority companion",
            companion=False,
        )
        require(
            self.authority_path.read_bytes() == self.v1_authority_archive_path.read_bytes(),
            "archived v1 authority is not byte-identical",
        )
        require(
            self.authority_path.with_suffix(".sha256").read_bytes()
            == self.v1_authority_archive_path.with_suffix(".sha256").read_bytes(),
            "archived v1 authority companion is not byte-identical",
        )

        decision_value = load_json(self.successor_decision_path)
        require(decision_value.get("authority") is False, "successor decision unexpectedly grants authority")
        require(decision_value.get("execution") is False, "successor decision unexpectedly grants execution")
        require(decision_value.get("stage_closing") is False, "successor decision unexpectedly closes the stage")
        require(
            decision_value.get("integrity", {}).get("canonical_sha256")
            == detached_canonical_integrity(decision_value),
            "successor decision canonical integrity mismatch",
        )

        predecessor_value = load_json_bytes(
            self.predecessor_ledger_path,
            self.predecessor_ledger_path.read_bytes(),
        )
        require(
            predecessor_value.get("schema_version") == SCHEMA_VERSION
            and set(predecessor_value.get("runs", {})) == {PREDECESSOR_RUN_ID},
            "sealed predecessor ledger structure differs",
        )
        predecessor_record = predecessor_value["runs"][PREDECESSOR_RUN_ID]
        require(predecessor_record.get("run_id") == PREDECESSOR_RUN_ID, "sealed predecessor run_id differs")
        require(predecessor_record.get("state") == "failed_fail_closed", "sealed predecessor state differs")
        require(predecessor_record.get("execution_count") == 1, "sealed predecessor execution count differs")
        require(
            predecessor_record.get("model_counts") == {"baseline_model": 1, "candidate_model": 0},
            "sealed predecessor model counts differ",
        )
        require(predecessor_record.get("recovery_policy") == "never_retry", "sealed predecessor recovery policy differs")

        current_runner_sha256 = sha256_file(self.runner_path)
        require(current_runner_sha256 != PRE_POLICY_RUNNER_SHA256, "runner still has the pre-policy hash")
        bindings = {
            "predecessor_ledger": predecessor,
            "predecessor_terminal": terminal,
            "manager_decision": decision,
            "accepted_environment_audit": environment_audit,
            "environment_reproduction": environment_reproduction,
            "accepted_recovery_handoff": recovery,
            "v1_authority": original_authority,
            "v1_authority_companion": original_companion,
            "v1_authority_archive": archived_authority,
            "v1_authority_archive_companion": archived_companion,
            "source_identity": {
                "runner": copy.deepcopy(MIGRATION_RUNNER_RECORD),
                "executor": executor,
                "software_identity_helper": helper,
            },
        }
        return bindings, predecessor_value

    def _expected_successor_ledger(self) -> dict[str, Any]:
        bindings, predecessor = self._migration_bindings()
        return {
            "schema_version": SUCCESSOR_LEDGER_SCHEMA_VERSION,
            "kind": "dynamic_scale32_baseline_successor_ledger_v2",
            "contract_id": CONTRACT_ID,
            "stage": REQUIRED_STAGE,
            "stage_closing": False,
            "migration_bindings": bindings,
            "migration_binding_sha256": sha256_bytes(canonical_json_bytes(bindings)),
            "runs": copy.deepcopy(predecessor["runs"]),
            "recovery_generation_slots": [
                {
                    "recovery_generation": SUCCESSOR_GENERATION,
                    "status": "available",
                    "consumed_by_run_id": None,
                    "reserved_at_utc": None,
                }
            ],
            "successor_reservations": 0,
            "cumulative_baseline_process_attempts": 1,
            "cumulative_model_counts": {"baseline_model": 1, "candidate_model": 0},
            "publication": {
                "successor_slot_reopened": False,
                "baseline_l2_accepted": False,
            },
        }

    def migrate_successor_ledger(self) -> dict[str, Any]:
        expected = self._expected_successor_ledger()
        self.state_root.mkdir(parents=True, exist_ok=True)
        with self.successor_lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.successor_ledger_path.exists():
                observed_bytes = self.successor_ledger_path.read_bytes()
                observed = load_json_bytes(self.successor_ledger_path, observed_bytes)
                self._validate_successor_ledger(observed)
                require(observed == expected, "existing successor ledger is not the deterministic migration")
                require(observed_bytes == json_bytes(expected), "existing successor ledger bytes differ")
                created = False
            else:
                atomic_write_bytes(self.successor_ledger_path, json_bytes(expected))
                created = True
            return {
                "created": created,
                "ledger": artifact(self.root, self.successor_ledger_path),
                "cumulative_baseline_process_attempts": 1,
                "cumulative_model_counts": {"baseline_model": 1, "candidate_model": 0},
                "successor_reservations": 0,
                "stage_closing": False,
            }

    def _load_successor_ledger_readonly(self) -> dict[str, Any]:
        require(self.successor_ledger_path.is_file(), "successor ledger migration is missing")
        ledger = load_json(self.successor_ledger_path)
        self._validate_successor_ledger(ledger)
        return ledger

    @contextlib.contextmanager
    def _locked_successor_ledger(self) -> Iterator[dict[str, Any]]:
        require(self.successor_ledger_path.is_file(), "successor ledger migration is missing")
        with self.successor_lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            ledger = load_json(self.successor_ledger_path)
            self._validate_successor_ledger(ledger)
            yield ledger

    def _save_successor_ledger(self, ledger: dict[str, Any]) -> None:
        self._validate_successor_ledger(ledger)
        atomic_write_json(self.successor_ledger_path, ledger)

    def _validate_successor_ledger(self, ledger: dict[str, Any]) -> None:
        expected = self._expected_successor_ledger()
        exact_keys(ledger, set(expected), "successor ledger")
        for key in (
            "schema_version",
            "kind",
            "contract_id",
            "stage",
            "stage_closing",
            "migration_bindings",
            "migration_binding_sha256",
            "publication",
        ):
            require(ledger.get(key) == expected[key], f"successor ledger {key} differs")
        require(isinstance(ledger.get("runs"), dict), "successor ledger runs must be an object")
        require(PREDECESSOR_RUN_ID in ledger["runs"], "successor ledger omits the sealed predecessor")
        require(
            ledger["runs"][PREDECESSOR_RUN_ID] == expected["runs"][PREDECESSOR_RUN_ID],
            "successor ledger changed the sealed predecessor record",
        )
        successor_ids = sorted(set(ledger["runs"]) - {PREDECESSOR_RUN_ID})
        require(len(successor_ids) <= 1, "multiple successor records are forbidden")
        slots = ledger.get("recovery_generation_slots")
        require(isinstance(slots, list) and len(slots) == 1, "exactly one recovery-generation slot is required")
        slot = slots[0]
        exact_keys(
            slot,
            {"recovery_generation", "status", "consumed_by_run_id", "reserved_at_utc"},
            "successor slot",
        )
        require(slot["recovery_generation"] == SUCCESSOR_GENERATION, "unknown recovery generation")
        require(
            ledger.get("publication")
            == {"successor_slot_reopened": False, "baseline_l2_accepted": False},
            "publication cannot reopen the slot or encode L2 acceptance",
        )
        require(
            ledger.get("cumulative_model_counts", {}).get("candidate_model") == 0,
            "candidate_model count must remain zero",
        )
        if not successor_ids:
            require(slot["status"] == "available", "unconsumed successor slot must be available")
            require(slot["consumed_by_run_id"] is None and slot["reserved_at_utc"] is None, "available slot has consumption state")
            require(ledger.get("successor_reservations") == 0, "successor reservation count differs")
            require(ledger.get("cumulative_baseline_process_attempts") == 1, "migration changed cumulative attempts")
            require(
                ledger.get("cumulative_model_counts") == {"baseline_model": 1, "candidate_model": 0},
                "migration changed cumulative model counts",
            )
            return

        run_id = successor_ids[0]
        require(run_id != PREDECESSOR_RUN_ID and RUN_ID_RE.fullmatch(run_id) is not None, "invalid successor run_id")
        record = ledger["runs"][run_id]
        require(isinstance(record, dict), "successor record must be an object")
        require(record.get("run_id") == run_id, "successor record run_id differs")
        require(record.get("recovery_generation") == SUCCESSOR_GENERATION, "successor generation differs")
        require(record.get("recovery_policy") == "never_retry", "successor recovery policy differs")
        require(record.get("model_counts", {}).get("candidate_model") == 0, "successor candidate count differs")
        require(not contains_forbidden_value(record), "candidate surface found in successor state")
        for field in ("authority_sha256", "reservation_sha256"):
            value = record.get(field)
            require(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                f"successor record {field} is missing or malformed",
            )
        require(isinstance(record.get("reserved_at_utc"), str), "successor reservation time is missing")
        require(slot["status"] == "consumed", "successor reservation did not consume the slot")
        require(slot["consumed_by_run_id"] == run_id, "slot consumption run_id differs")
        require(slot["reserved_at_utc"] == record.get("reserved_at_utc"), "slot reservation time differs")
        require(ledger.get("successor_reservations") == 1, "successor reservation count differs")
        execution_count = record.get("execution_count")
        require(execution_count in {0, 1}, "successor execution count differs")
        state = record.get("state")
        states_before_invocation = {
            "reserved",
            "reservation_validation_fail_closed",
            "interrupted_before_execution_fail_closed",
        }
        states_after_invocation = {
            "running",
            "timeout_fail_closed",
            "command_launch_fail_closed",
            "failed_fail_closed",
            "validation_fail_closed",
            "publication_fail_closed",
            "interrupted_after_execution_fail_closed",
            "committed",
        }
        require(
            state in (states_before_invocation if execution_count == 0 else states_after_invocation),
            "successor state is malformed for its invocation count",
        )
        require(
            record.get("model_counts")
            == {"baseline_model": execution_count, "candidate_model": 0},
            "successor per-generation model counts differ",
        )
        expected_attempts = 1 + execution_count
        require(
            ledger.get("cumulative_baseline_process_attempts") == expected_attempts,
            "cumulative attempts may change only at successor invocation",
        )
        require(
            ledger.get("cumulative_model_counts")
            == {"baseline_model": expected_attempts, "candidate_model": 0},
            "cumulative model counts differ",
        )

    def _validate_successor_authority_policy(
        self,
        authority: dict[str, Any],
        ledger: dict[str, Any],
    ) -> None:
        policy = authority.get("successor_policy")
        require(isinstance(policy, dict), "successor authority policy is missing")
        exact_keys(policy, SUCCESSOR_POLICY_KEYS, "successor authority policy")
        require(authority.get("run_id") != PREDECESSOR_RUN_ID, "sealed v1 run_id is permanently rejected")
        require(authority.get("run_id") not in ledger.get("runs", {}), "successor run_id is not unique")
        require(policy["recovery_generation"] == SUCCESSOR_GENERATION, "successor authority generation differs")
        require(policy["decision_sha256"] == EXPECTED_SUCCESSOR_DECISION_SHA256, "successor decision hash differs")
        require(policy["predecessor_ledger_sha256"] == EXPECTED_PREDECESSOR_LEDGER_SHA256, "predecessor ledger binding differs")
        require(policy["predecessor_terminal_sha256"] == EXPECTED_PREDECESSOR_TERMINAL_SHA256, "predecessor terminal binding differs")
        runner_sha256 = sha256_file(self.runner_path)
        require(policy["runner_sha256"] == runner_sha256, "successor runner hash differs")
        require(policy["executor_sha256"] == EXPECTED_EXECUTOR_SHA256, "successor executor hash differs")
        require(
            policy["software_identity_helper_sha256"] == EXPECTED_SOFTWARE_IDENTITY_SHA256,
            "successor software identity helper hash differs",
        )
        source_identity = ledger["migration_bindings"]["source_identity"]
        require(
            source_identity["runner"] == MIGRATION_RUNNER_RECORD,
            "immutable migration-time runner binding differs",
        )
        require(source_identity["executor"]["sha256"] == EXPECTED_EXECUTOR_SHA256, "ledger executor binding differs")
        require(
            source_identity["software_identity_helper"]["sha256"] == EXPECTED_SOFTWARE_IDENTITY_SHA256,
            "ledger helper binding differs",
        )

        require(self.post_policy_audit_path.is_file(), "post-policy environment audit is missing")
        require(
            sha256_file(self.post_policy_audit_path) == policy["post_policy_environment_audit_sha256"],
            "post-policy environment audit hash differs",
        )
        audit = load_json(self.post_policy_audit_path)
        require(audit.get("status") == "pass", "post-policy environment audit is not passing")
        require(audit.get("authority") is False and audit.get("execution") is False, "post-policy audit exceeds no-execution scope")
        require(audit.get("stage_closing") is False, "post-policy audit cannot close the stage")
        require(
            audit.get("source_hashes")
            == {
                "runner": runner_sha256,
                "executor": EXPECTED_EXECUTOR_SHA256,
                "software_identity_helper": EXPECTED_SOFTWARE_IDENTITY_SHA256,
            },
            "post-policy audit source binding differs",
        )
        require(
            audit.get("observed_counts") == {"baseline_model": 1, "candidate_model": 0},
            "post-policy audit counts differ",
        )
        no_execution = audit.get("no_execution_state", {})
        require(
            no_execution.get("authority_created") is False
            and no_execution.get("reservation_created") is False
            and no_execution.get("run_created") is False
            and no_execution.get("candidate_task_created") is False
            and no_execution.get("process_invoked") is False,
            "post-policy audit does not prove no execution",
        )

        require(self.post_policy_review_path.is_file(), "independent post-policy review is missing")
        require(
            sha256_file(self.post_policy_review_path) == policy["independent_review_sha256"],
            "independent post-policy review hash differs",
        )
        review = load_json(self.post_policy_review_path)
        require(review.get("verdict") == "go", "independent post-policy review is not accepted")
        require(review.get("independent_reviewer") is True, "post-policy review is not independent")
        require(review.get("stage_closing") is False, "post-policy review cannot close the stage")
        require(
            review.get("bindings", {}).get("post_policy_environment_audit_sha256")
            == policy["post_policy_environment_audit_sha256"],
            "review does not bind the post-policy audit",
        )
        require(
            review.get("bindings", {}).get("runner_sha256") == runner_sha256,
            "review does not bind the changed runner",
        )
        require(not contains_forbidden_value(authority), "candidate surface found in successor authority")

    def reserve_successor_slot(
        self,
        ledger: dict[str, Any],
        *,
        run_id: str,
        authority_sha256: str,
        reservation_sha256: str,
        reserved_at_utc: str,
    ) -> dict[str, Any]:
        self._validate_successor_ledger(ledger)
        require(run_id != PREDECESSOR_RUN_ID, "sealed v1 run_id is permanently rejected")
        require(RUN_ID_RE.fullmatch(run_id) is not None, "invalid successor run_id")
        require(run_id not in ledger["runs"], "duplicate successor run_id is forbidden")
        require(len(ledger["runs"]) == 1, "successor slot is already consumed")
        slot = ledger["recovery_generation_slots"][0]
        require(slot["status"] == "available", "successor slot is already consumed")
        record = {
            "run_id": run_id,
            "state": "reserved",
            "recovery_generation": SUCCESSOR_GENERATION,
            "reserved_at_utc": reserved_at_utc,
            "execution_count": 0,
            "model_counts": {"baseline_model": 0, "candidate_model": 0},
            "authority_sha256": authority_sha256,
            "reservation_sha256": reservation_sha256,
            "recovery_policy": "never_retry",
        }
        ledger["runs"][run_id] = record
        slot.update(
            {
                "status": "consumed",
                "consumed_by_run_id": run_id,
                "reserved_at_utc": reserved_at_utc,
            }
        )
        ledger["successor_reservations"] = 1
        self._validate_successor_ledger(ledger)
        return record

    def mark_successor_invoked(self, ledger: dict[str, Any], run_id: str) -> None:
        self._validate_successor_ledger(ledger)
        require(run_id != PREDECESSOR_RUN_ID, "sealed v1 can never be reinvoked")
        record = ledger.get("runs", {}).get(run_id)
        require(isinstance(record, dict), "successor reservation is missing")
        require(record.get("state") == "reserved", "duplicate or recovered successor cannot invoke")
        require(record.get("execution_count") == 0, "successor process was already invoked")
        record["state"] = "running"
        record["execution_count"] = 1
        record["model_counts"] = {"baseline_model": 1, "candidate_model": 0}
        record["execution_started_at_utc"] = now_utc()
        ledger["cumulative_baseline_process_attempts"] = 2
        ledger["cumulative_model_counts"] = {"baseline_model": 2, "candidate_model": 0}
        self._validate_successor_ledger(ledger)

    def terminalize_successor_policy(
        self,
        ledger: dict[str, Any],
        run_id: str,
        state: str,
        detail: str,
    ) -> None:
        self._validate_successor_ledger(ledger)
        require(state != "reserved" and state != "running", "terminal state required")
        record = ledger.get("runs", {}).get(run_id)
        require(isinstance(record, dict), "successor reservation is missing")
        record["state"] = state
        record["terminal_at_utc"] = now_utc()
        record["error"] = detail
        ledger["publication"] = {
            "successor_slot_reopened": False,
            "baseline_l2_accepted": False,
        }
        self._validate_successor_ledger(ledger)

    def recover_successor_policy(self, ledger: dict[str, Any], run_id: str) -> str:
        self._validate_successor_ledger(ledger)
        require(run_id != PREDECESSOR_RUN_ID, "sealed v1 recovery is permanently rejected")
        record = ledger.get("runs", {}).get(run_id)
        require(isinstance(record, dict), "successor reservation is missing")
        if record["state"] == "reserved":
            self.terminalize_successor_policy(
                ledger,
                run_id,
                "interrupted_before_execution_fail_closed",
                "reserved successor was interrupted; process rerun forbidden",
            )
            return "terminalized_no_process_invocation"
        if record["state"] == "running":
            self.terminalize_successor_policy(
                ledger,
                run_id,
                "interrupted_after_execution_fail_closed",
                "successor invocation was already consumed; process rerun forbidden",
            )
            return "terminalized_no_process_invocation"
        return "already_terminal_no_process_invocation"

    def _failpoint(self, point: str) -> None:
        if os.environ.get("ACE2_BASELINE_TEST_FAILPOINT") != point:
            return
        require(os.environ.get("ACE2_BASELINE_TEST_MODE") == "1", "failpoints require test mode")
        os._exit(97)

    def _state_dir(self, run_id: str) -> Path:
        return self.state_root / run_id

    def _bundle(self, run_id: str) -> Path:
        return self.published_root / f"{run_id}.bundle"

    def _load_reservation(self, run_id: str) -> dict[str, Any]:
        state_dir = self._state_dir(run_id)
        reservation_path = state_dir / RESERVATION_NAME
        authority_path = state_dir / AUTHORITY_SNAPSHOT_NAME
        verify_companion(reservation_path, reservation_path.with_suffix(".sha256"))
        verify_companion(authority_path, authority_path.with_suffix(".sha256"))
        reservation = load_json(reservation_path)
        require(reservation.get("run_id") == run_id, "reservation run_id differs")
        require(reservation.get("stage_closing") is False, "reservation must keep stage_closing=false")
        require(
            reservation.get("authority_snapshot", {}).get("sha256") == sha256_file(authority_path),
            "reservation authority snapshot differs",
        )
        return reservation

    def _reserved_record(self, reservation: dict[str, Any]) -> dict[str, Any]:
        reservation_path = self._state_dir(reservation["run_id"]) / RESERVATION_NAME
        return {
            "run_id": reservation["run_id"],
            "state": "reserved",
            "reserved_at_utc": reservation["reserved_at_utc"],
            "execution_count": 0,
            "model_counts": {"baseline_model": 0, "candidate_model": 0},
            "authority_sha256": reservation["authority_snapshot"]["sha256"],
            "reservation_sha256": sha256_file(reservation_path),
            "preflight_sha256": reservation["preflight_sha256"],
            "command_sha256": reservation["command_sha256"],
            "environment_sha256": reservation["environment_sha256"],
            "recovery_policy": "never_retry",
        }

    def _snapshot_reserved_inputs(
        self,
        staging: Path,
        authority: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expected = {record["path"]: record for record in authority["inputs"]}
        require(
            set(expected) == {path.as_posix() for path in INPUT_RELS},
            "authority input path set differs",
        )
        snapshot_root = staging / RESERVED_INPUTS_NAME
        snapshot_root.mkdir()
        snapshots: list[dict[str, Any]] = []
        for source_path in sorted(expected):
            source_record = expected[source_path]
            relative = Path(source_path)
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe authority input path")
            source = self.root / relative
            try:
                content = source.read_bytes()
            except OSError as exc:
                raise BaselineError(f"cannot snapshot authority input {source}: {exc}") from exc
            require(len(content) == source_record["bytes"], f"authority input byte count changed: {source_path}")
            require(sha256_bytes(content) == source_record["sha256"], f"authority input hash changed: {source_path}")
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_durable(destination, content)
            os.chmod(destination, 0o444)
            snapshot = artifact(staging, destination)
            snapshot["source_path"] = source_path
            snapshots.append(snapshot)
        fsync_tree_directories(snapshot_root)
        directories = [snapshot_root, *(path for path in snapshot_root.rglob("*") if path.is_dir())]
        for directory in directories:
            os.chmod(directory, 0o555)
        return snapshots

    def _verify_reserved_inputs(
        self,
        base: Path,
        reservation: dict[str, Any],
        authority: dict[str, Any],
    ) -> None:
        snapshot_root = base / RESERVED_INPUTS_NAME
        require(snapshot_root.is_dir() and not snapshot_root.is_symlink(), "reserved input root is missing or unsafe")
        snapshots = reservation.get("reserved_inputs")
        require(isinstance(snapshots, list), "reservation input snapshot list is missing")
        expected_sources = {record["path"]: record for record in authority["inputs"]}
        require(len(snapshots) == len(expected_sources), "reserved input count differs")
        expected_snapshot_paths: set[str] = set()
        seen_sources: set[str] = set()
        for snapshot in snapshots:
            require(isinstance(snapshot, dict), "reserved input record must be an object")
            exact_keys(snapshot, {"source_path", "path", "bytes", "sha256"}, "reserved input record")
            source_path = snapshot["source_path"]
            require(source_path in expected_sources and source_path not in seen_sources, "reserved input source differs")
            seen_sources.add(source_path)
            source_record = expected_sources[source_path]
            expected_path = (Path(RESERVED_INPUTS_NAME) / source_path).as_posix()
            require(snapshot["path"] == expected_path, f"reserved input path differs: {source_path}")
            require(snapshot["bytes"] == source_record["bytes"], f"reserved input byte count binding differs: {source_path}")
            require(snapshot["sha256"] == source_record["sha256"], f"reserved input hash binding differs: {source_path}")
            relative = Path(snapshot["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe reserved input path")
            path = base / relative
            require(path.is_file() and not path.is_symlink(), f"reserved input is missing or unsafe: {source_path}")
            require(path.stat().st_size == snapshot["bytes"], f"reserved input byte count differs: {source_path}")
            require(sha256_file(path) == snapshot["sha256"], f"reserved input hash differs: {source_path}")
            require(path.stat().st_mode & 0o222 == 0, f"reserved input is writable: {source_path}")
            expected_snapshot_paths.add(snapshot["path"])
        observed_paths = {
            path.relative_to(base).as_posix()
            for path in snapshot_root.rglob("*")
            if path.is_file()
        }
        require(observed_paths == expected_snapshot_paths, "reserved input file set differs")
        require(seen_sources == set(expected_sources), "reserved input source set differs")

    def _load_execution_plan(self, run_id: str) -> dict[str, Any]:
        state_dir = self._state_dir(run_id)
        reservation = self._load_reservation(run_id)
        authority_path = state_dir / AUTHORITY_SNAPSHOT_NAME
        authority = load_json(authority_path)
        require(authority.get("run_id") == run_id, "reserved authority run_id differs")
        require(
            reservation["authority_snapshot"]["sha256"] == sha256_file(authority_path),
            "reserved authority hash differs",
        )
        self._verify_reserved_inputs(state_dir, reservation, authority)

        environment_path = state_dir / LAUNCH_ENVIRONMENT_NAME
        verify_companion(environment_path, environment_path.with_suffix(".sha256"))
        require(
            artifact(state_dir, environment_path) == reservation.get("launch_environment"),
            "reserved launch environment artifact differs",
        )
        environment_record = load_json(environment_path)
        exact_keys(
            environment_record,
            {"schema_version", "kind", "variables", "environment_sha256"},
            "launch environment record",
        )
        require(environment_record["schema_version"] == SCHEMA_VERSION, "launch environment schema differs")
        require(
            environment_record["kind"] == "dynamic_scale32_verification_baseline_launch_environment",
            "launch environment kind differs",
        )
        environment = environment_record["variables"]
        require(isinstance(environment, dict), "reserved launch environment variables must be an object")
        exact_keys(environment, set(ENVIRONMENT_KEYS), "reserved launch environment")
        require(
            all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()),
            "reserved launch environment keys and values must be strings",
        )
        environment_sha256 = sha256_bytes(canonical_json_bytes(environment))
        require(environment_record["environment_sha256"] == environment_sha256, "reserved launch environment hash differs")
        require(reservation["environment_sha256"] == environment_sha256, "reservation environment hash differs")
        require(
            environment == self._render_environment(authority["environment"], state_dir),
            "reserved launch environment differs from authority",
        )

        command = self._render_reserved_command(authority, state_dir)
        require(reservation["command"] == command, "reserved launch command differs")
        require(
            reservation["command_sha256"] == sha256_bytes(canonical_json_bytes(command)),
            "reserved launch command hash differs",
        )
        require(
            reservation["command_template_sha256"]
            == sha256_bytes(canonical_json_bytes(authority["command"])),
            "reserved command template hash differs",
        )
        working_directory = state_dir / reservation["working_directory"]
        require(
            working_directory == state_dir / RESERVED_INPUTS_NAME
            and working_directory.is_dir()
            and not working_directory.is_symlink(),
            "reserved working directory differs",
        )
        for runtime_directory in (state_dir / "runtime" / "home", state_dir / "runtime" / "tmp"):
            require(runtime_directory.is_dir() and not runtime_directory.is_symlink(), "reserved runtime directory differs")
        return {
            "authority": authority,
            "command": command,
            "environment": environment,
            "working_directory": working_directory,
            "reservation": reservation,
        }

    def _create_reservation(
        self,
        authority: dict[str, Any],
        authority_bytes: bytes,
        authority_source: dict[str, Any],
        preflight: dict[str, Any],
        before_publish: Callable[[dict[str, Any], str], None] | None = None,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        run_id = authority["run_id"]
        self.state_root.mkdir(parents=True, exist_ok=True)
        final = self._state_dir(run_id)
        require(not final.exists(), f"run already has a durable reservation: {final}")
        staging = self.state_root / f".staging-reservation-{run_id}-{os.getpid()}"
        require(not staging.exists(), f"reservation staging collision: {staging}")
        staging.mkdir()
        (staging / "runtime" / "home").mkdir(parents=True)
        (staging / "runtime" / "tmp").mkdir()
        authority_snapshot = write_companion_pair(
            staging,
            AUTHORITY_SNAPSHOT_NAME,
            authority_bytes,
        )
        reserved_inputs = self._snapshot_reserved_inputs(staging, authority)
        launch_environment = self._render_environment(authority["environment"], final)
        environment_sha256 = sha256_bytes(canonical_json_bytes(launch_environment))
        launch_environment_record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "dynamic_scale32_verification_baseline_launch_environment",
            "variables": launch_environment,
            "environment_sha256": environment_sha256,
        }
        launch_environment_artifact = write_companion_pair(
            staging,
            LAUNCH_ENVIRONMENT_NAME,
            json_bytes(launch_environment_record),
        )
        command = self._render_reserved_command(authority, final)
        reservation = {
            "schema_version": SCHEMA_VERSION,
            "kind": "dynamic_scale32_verification_baseline_reservation",
            "contract_id": CONTRACT_ID,
            "stage": REQUIRED_STAGE,
            "stage_closing": False,
            "run_id": run_id,
            "reserved_at_utc": now_utc(),
            "execution_count": 0,
            "model_counts": {"baseline_model": 0, "candidate_model": 0},
            "preflight_sha256": preflight["preflight_sha256"],
            "authority_source": authority_source,
            "authority_snapshot": authority_snapshot,
            "authority_binding_sha256": authority["authority_binding_sha256"],
            "reserved_inputs": reserved_inputs,
            "working_directory": RESERVED_INPUTS_NAME,
            "command": command,
            "command_sha256": sha256_bytes(canonical_json_bytes(command)),
            "command_template_sha256": sha256_bytes(canonical_json_bytes(authority["command"])),
            "launch_environment": launch_environment_artifact,
            "environment_sha256": environment_sha256,
            "recovery_policy": "never_retry_after_durable_reservation",
        }
        write_companion_pair(staging, RESERVATION_NAME, json_bytes(reservation))
        fsync_tree_directories(staging)
        if before_publish is not None:
            try:
                before_publish(reservation, sha256_file(staging / RESERVATION_NAME))
            except Exception:
                for directory in [
                    staging,
                    *(path for path in staging.rglob("*") if path.is_dir()),
                ]:
                    os.chmod(directory, 0o755)
                shutil.rmtree(staging)
                fsync_directory(self.state_root)
                raise
        os.replace(staging, final)
        fsync_directory(self.state_root)
        self._failpoint("after_reservation_record")
        return final, reservation, self._reserved_record(reservation)

    def _event_bundle(self, run_id: str, event: str) -> Path:
        return self._state_dir(run_id) / f"{event}.bundle"

    def _verify_event_packet(self, bundle: Path, label: str) -> dict[str, Any]:
        require(bundle.is_dir(), f"event packet missing: {bundle}")
        packet_path = bundle / f"{label}.json"
        verify_companion(packet_path, packet_path.with_suffix(".sha256"))
        packet = load_json(packet_path)
        require(packet.get("contract_id") == CONTRACT_ID, "event packet contract differs")
        require(packet.get("stage_closing") is False, "event packet must keep stage_closing=false")
        for record in packet.get("artifact_inventory", []):
            relative = Path(record["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe event artifact path")
            path = bundle / relative
            require(path.is_file(), f"event artifact missing: {record['path']}")
            require(path.stat().st_size == record["bytes"], f"event artifact size differs: {record['path']}")
            require(sha256_file(path) == record["sha256"], f"event artifact hash differs: {record['path']}")
        return {"bundle": bundle, "packet": packet_path, "report": packet}

    def _publish_event_packet(
        self,
        run_id: str,
        event: str,
        label: str,
        record: dict[str, Any],
        detail: str,
        *,
        attempted_run_id: str | None = None,
    ) -> dict[str, Any]:
        state_dir = self._state_dir(run_id)
        self._load_reservation(run_id)
        final = self._event_bundle(run_id, event)
        if final.exists():
            return self._verify_event_packet(final, label)
        staging = state_dir / f".staging-{event}-{os.getpid()}"
        require(not staging.exists(), f"event packet staging collision: {staging}")
        staging.mkdir()
        for name in (
            AUTHORITY_SNAPSHOT_NAME,
            Path(AUTHORITY_SNAPSHOT_NAME).with_suffix(".sha256").name,
            RESERVATION_NAME,
            Path(RESERVATION_NAME).with_suffix(".sha256").name,
            LAUNCH_ENVIRONMENT_NAME,
            Path(LAUNCH_ENVIRONMENT_NAME).with_suffix(".sha256").name,
        ):
            shutil.copy2(state_dir / name, staging / name)
        shutil.copytree(state_dir / RESERVED_INPUTS_NAME, staging / RESERVED_INPUTS_NAME)
        if (state_dir / "stdout.log").is_file():
            shutil.copy2(state_dir / "stdout.log", staging / "stdout.log")
        ledger_snapshot = {"schema_version": SCHEMA_VERSION, "runs": {run_id: copy.deepcopy(record)}}
        write_companion_pair(staging, "RUN_LEDGER_SNAPSHOT.json", json_bytes(ledger_snapshot))
        packet = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"dynamic_scale32_verification_baseline_{event}",
            "contract_id": CONTRACT_ID,
            "stage": REQUIRED_STAGE,
            "stage_closing": False,
            "run_id": run_id,
            "attempted_run_id": attempted_run_id,
            "created_at_utc": now_utc(),
            "terminal_state": record["state"],
            "execution_count": record["execution_count"],
            "model_counts": record["model_counts"],
            "detail": detail,
            "artifact_inventory": artifact_inventory(staging),
        }
        write_companion_pair(staging, f"{label}.json", json_bytes(packet))
        fsync_directory(staging)
        os.replace(staging, final)
        fsync_directory(state_dir)
        return self._verify_event_packet(final, label)

    def _validate_outputs(
        self,
        raw: Path,
        authority: dict[str, Any],
        environment: dict[str, str],
    ) -> None:
        require(raw.is_dir(), f"baseline output directory is missing: {raw}")
        expected = set(authority["scope"]["required_outputs"])
        entries = list(raw.iterdir())
        require(all(path.is_file() for path in entries), "baseline output directory contains non-files")
        observed = {path.name for path in entries}
        require(observed == expected, f"baseline output set differs: expected={sorted(expected)} observed={sorted(observed)}")
        for name in expected:
            value = load_json(raw / name)
            require(not contains_forbidden_value(value), f"candidate content found in baseline output: {name}")
            require(value.get("schema_version") == SCHEMA_VERSION, f"output schema differs: {name}")
            require(value.get("contract_id") == CONTRACT_ID, f"output contract differs: {name}")
        traces = load_json(raw / "ordered_layer_traces.json")
        exact_keys(
            traces,
            {"schema_version", "contract_id", "kind", "ordering", "sequences"},
            "ordered layer traces",
        )
        require(
            traces["kind"] == "dynamic_scale32_disabled_baseline_ordered_layer_traces",
            "ordered layer trace kind differs",
        )
        require(isinstance(traces["sequences"], list), "ordered layer trace sequences must be a list")
        for sequence in traces["sequences"]:
            exact_keys(sequence, {"dataset", "sequence_index", "input_ids", "traces"}, "trace sequence")
            require(sequence["dataset"] in authority["scope"]["datasets"], "trace dataset differs")
            require(isinstance(sequence["traces"], list), "trace records must be a list")
            require(
                [record.get("ordinal") for record in sequence["traces"]]
                == list(range(len(sequence["traces"]))),
                "trace ordinals are not contiguous and ordered",
            )
            for record in sequence["traces"]:
                exact_keys(record, {"ordinal", "operator", "tensor"}, "trace record")
                require(isinstance(record["operator"], str) and record["operator"], "trace operator is missing")
                exact_keys(
                    record["tensor"],
                    {"dtype", "shape", "sha256", "minimum", "maximum", "squared_l2"},
                    "trace tensor descriptor",
                )
        final_outputs = load_json(raw / "final_outputs.json")
        exact_keys(
            final_outputs,
            {"schema_version", "contract_id", "kind", "datasets", "sequences"},
            "final outputs",
        )
        require(final_outputs["datasets"] == authority["scope"]["datasets"], "final output datasets differ")
        require(isinstance(final_outputs["sequences"], list), "final output sequences must be a list")
        results = load_json(raw / "results.json")
        exact_keys(
            results,
            {
                "schema_version",
                "contract_id",
                "kind",
                "classification",
                "gate_passed",
                "mechanism_disabled",
                "mode",
                "model",
                "metrics",
                "runtime",
                "seeds",
            },
            "results",
        )
        require(results["mechanism_disabled"] is True, "results do not prove disabled mechanism")
        require(results["mode"] == authority["scope"]["mode"], "result mode differs")
        require(results.get("model", {}).get("repository") == authority["scope"]["model_repository"], "result model repository differs")
        require(results.get("model", {}).get("revision") == authority["scope"]["model_revision"], "result model revision differs")
        require(results.get("seeds") == authority["seeds"], "result seeds differ")
        require(results.get("runtime", {}).get("device") == authority["device"], "result device differs")
        require(results.get("runtime", {}).get("packages") == authority["software_versions"], "result software versions differ")
        run_contract = load_json(raw / "run_contract.json")
        exact_keys(
            run_contract,
            {
                "schema_version",
                "contract_id",
                "kind",
                "artifact_hashes",
                "architecture_boundary",
                "baseline_model",
                "candidate_model",
                "datasets",
                "determinism",
                "mechanism",
                "mechanism_disabled",
                "mode_scope",
                "model",
                "numerical_contract",
                "output_schema",
                "software_versions",
                "source_contract_hashes",
                "status",
            },
            "run contract",
        )
        require(run_contract.get("mechanism_disabled") is True, "run contract does not prove disabled mechanism")
        require(run_contract.get("baseline_model") == 1, "run contract baseline_model count differs")
        require(run_contract.get("candidate_model") == 0, "run contract candidate_model count differs")
        require(run_contract.get("mechanism") == CONTRACT_ID, "run contract mechanism differs")
        require(
            run_contract.get("architecture_boundary")
            == {
                "first_unsupported_layer_operator": "layer_0.rope_q",
                "mode": "ADVANCE",
                "ordered_supported_layer_operator_prefix": [
                    "layer_0.input_rmsnorm",
                    "layer_0.q_proj",
                    "layer_0.k_proj",
                    "layer_0.v_proj",
                ],
            },
            "run contract architecture boundary differs",
        )
        require(
            run_contract.get("output_schema")
            == {"files": list(REQUIRED_OUTPUTS), "schema_version": SCHEMA_VERSION},
            "run contract output schema differs",
        )
        expected_hashes = {
            name: sha256_file(raw / name)
            for name in REQUIRED_OUTPUTS
            if name != "run_contract.json"
        }
        require(run_contract.get("artifact_hashes") == expected_hashes, "run contract artifact hashes differ")
        observations = load_json(raw / "input_observations.json")
        exact_keys(
            observations,
            {"schema_version", "contract_id", "kind", "datasets", "environment", "environment_sha256", "mode_scope"},
            "input observations",
        )
        require(observations.get("environment") == environment, "subprocess environment differs")
        require(
            observations.get("environment_sha256") == sha256_bytes(canonical_json_bytes(environment)),
            "subprocess environment hash differs",
        )

    def _publish_success(
        self,
        run_id: str,
        state_dir: Path,
        authority: dict[str, Any],
        preflight: dict[str, Any],
        final_ledger: dict[str, Any],
    ) -> dict[str, Any]:
        self.published_root.mkdir(parents=True, exist_ok=True)
        fsync_directory(self.published_root.parent)
        final = self._bundle(run_id)
        require(not final.exists(), f"published bundle already exists: {final}")
        staging = self.published_root / f".staging-{run_id}-{os.getpid()}"
        require(not staging.exists(), f"publication staging collision: {staging}")
        staging.mkdir()
        shutil.copytree(state_dir / "raw", staging / "raw")
        shutil.copy2(state_dir / "stdout.log", staging / "stdout.log")
        for name in (
            AUTHORITY_SNAPSHOT_NAME,
            Path(AUTHORITY_SNAPSHOT_NAME).with_suffix(".sha256").name,
            RESERVATION_NAME,
            Path(RESERVATION_NAME).with_suffix(".sha256").name,
            LAUNCH_ENVIRONMENT_NAME,
            Path(LAUNCH_ENVIRONMENT_NAME).with_suffix(".sha256").name,
        ):
            shutil.copy2(state_dir / name, staging / name)
        shutil.copytree(state_dir / RESERVED_INPUTS_NAME, staging / RESERVED_INPUTS_NAME)
        write_companion_pair(staging, FINAL_LEDGER_NAME, json_bytes(final_ledger))

        reservation = load_json(state_dir / RESERVATION_NAME)

        provenance = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "command": authority["command"],
            "command_sha256": sha256_bytes(canonical_json_bytes(authority["command"])),
            "scope": authority["scope"],
            "authority_source": preflight["execution_authority"],
            "authority_snapshot": artifact(staging, staging / AUTHORITY_SNAPSHOT_NAME),
            "authority_binding_sha256": authority["authority_binding_sha256"],
            "stage_authority": preflight["stage_authority"],
            "inputs": authority["inputs"],
            "seeds": authority["seeds"],
            "device": authority["device"],
            "software_versions": authority["software_versions"],
            "reserved_inputs": reservation["reserved_inputs"],
            "launch_environment": reservation["launch_environment"],
            "environment_sha256": reservation["environment_sha256"],
            "runner": preflight["frozen_provenance"]["runner"],
        }
        write_durable(staging / "PROVENANCE.json", json_bytes(provenance))

        handoff = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "stage": REQUIRED_STAGE,
            "stage_closing": False,
            "run_id": run_id,
            "terminal_status": "success",
            "execution_count": 1,
            "model_counts": {"baseline_model": 1, "candidate_model": 0},
            "reservation": artifact(staging, staging / RESERVATION_NAME),
            "final_run_ledger": artifact(staging, staging / FINAL_LEDGER_NAME),
            "candidate_entrypoint_interlock": preflight["candidate_entrypoint_interlock"],
            "environment_sha256": reservation["environment_sha256"],
            "crash_duplicate_policy": "never_retry_finalize_valid_published_bundle_or_fail_closed",
            "artifact_inventory": artifact_inventory(staging),
            "fresh_evidence_only_l2_required": True,
            "cumulative_baseline_process_attempts": final_ledger.get(
                "cumulative_baseline_process_attempts",
                1,
            ),
            "cumulative_model_counts": final_ledger.get(
                "cumulative_model_counts",
                {"baseline_model": 1, "candidate_model": 0},
            ),
            "successor_slot_reopened": False,
            "baseline_l2_accepted": False,
        }
        handoff["handoff_sha256"] = sha256_bytes(canonical_json_bytes(handoff))
        write_durable(staging / "HANDOFF.json", json_bytes(handoff))

        inventory = artifact_inventory(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "artifact_count": len(inventory),
            "artifacts": inventory,
        }
        write_companion_pair(staging, "MANIFEST.json", json_bytes(manifest))
        fsync_directory(staging)
        os.replace(staging, final)
        fsync_directory(self.published_root)
        return self._verify_bundle(final)

    def _verify_bundle(self, bundle: Path) -> dict[str, Any]:
        require(bundle.is_dir(), f"published bundle missing: {bundle}")
        for name in (
            "MANIFEST.json",
            AUTHORITY_SNAPSHOT_NAME,
            RESERVATION_NAME,
            FINAL_LEDGER_NAME,
            LAUNCH_ENVIRONMENT_NAME,
        ):
            path = bundle / name
            verify_companion(path, path.with_suffix(".sha256"))
        manifest = load_json(bundle / "MANIFEST.json")
        require(manifest.get("contract_id") == CONTRACT_ID, "manifest contract differs")
        require(isinstance(manifest.get("artifacts"), list), "manifest artifacts must be a list")
        require(manifest.get("artifact_count") == len(manifest["artifacts"]), "manifest count differs")
        for record in manifest["artifacts"]:
            relative = Path(record["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe manifest artifact path")
            path = bundle / relative
            require(path.is_file(), f"manifest artifact missing: {record['path']}")
            require(path.stat().st_size == record["bytes"], f"manifest byte count differs: {record['path']}")
            require(sha256_file(path) == record["sha256"], f"manifest hash differs: {record['path']}")
        handoff = load_json(bundle / "HANDOFF.json")
        require(handoff.get("stage_closing") is False, "handoff must keep stage_closing=false")
        require(handoff.get("execution_count") == 1, "handoff execution count differs")
        require(handoff.get("model_counts") == {"baseline_model": 1, "candidate_model": 0}, "handoff model counts differ")
        require(handoff.get("successor_slot_reopened") is False, "publication reopened the successor slot")
        require(handoff.get("baseline_l2_accepted") is False, "publication encoded baseline L2 acceptance")
        reservation = load_json(bundle / RESERVATION_NAME)
        require(reservation.get("execution_count") == 0, "reservation execution count differs")
        authority = load_json(bundle / AUTHORITY_SNAPSHOT_NAME)
        self._verify_reserved_inputs(bundle, reservation, authority)
        launch_environment = load_json(bundle / LAUNCH_ENVIRONMENT_NAME)
        require(
            launch_environment.get("environment_sha256") == reservation.get("environment_sha256"),
            "published launch environment hash differs",
        )
        final_ledger = load_json(bundle / FINAL_LEDGER_NAME)
        run_id = handoff.get("run_id")
        record = final_ledger.get("runs", {}).get(run_id, {})
        require(record.get("state") == "committed", "final ledger state differs")
        require(record.get("execution_count") == 1, "final ledger execution count differs")
        require(record.get("model_counts") == {"baseline_model": 1, "candidate_model": 0}, "final ledger model counts differ")
        if final_ledger.get("schema_version") == SUCCESSOR_LEDGER_SCHEMA_VERSION:
            self._validate_successor_ledger(final_ledger)
            require(
                handoff.get("cumulative_baseline_process_attempts") == 2,
                "successor handoff cumulative attempts differ",
            )
            require(
                handoff.get("cumulative_model_counts") == {"baseline_model": 2, "candidate_model": 0},
                "successor handoff cumulative model counts differ",
            )
        return {
            "bundle": bundle,
            "manifest": bundle / "MANIFEST.json",
            "manifest_sha256": sha256_file(bundle / "MANIFEST.json"),
            "handoff": bundle / "HANDOFF.json",
            "final_ledger": final_ledger,
        }

    def _terminalize(
        self,
        ledger: dict[str, Any],
        record: dict[str, Any],
        state: str,
        detail: str,
        **fields: Any,
    ) -> BaselineError:
        record["state"] = state
        record["terminal_at_utc"] = now_utc()
        record["error"] = detail
        record.update(fields)
        self._save_ledger(ledger)
        self._publish_event_packet(record["run_id"], "terminal", "TERMINAL", record, detail)
        return BaselineError(detail)

    def _terminalize_successor_runtime(
        self,
        ledger: dict[str, Any],
        run_id: str,
        state: str,
        detail: str,
        **fields: Any,
    ) -> BaselineError:
        self.terminalize_successor_policy(ledger, run_id, state, detail)
        record = ledger["runs"][run_id]
        record.update(fields)
        self._save_successor_ledger(ledger)
        self._publish_event_packet(run_id, "terminal", "TERMINAL", record, detail)
        return BaselineError(detail)

    def _run_successor(self) -> dict[str, Any]:
        preflight = self.preflight()
        if preflight["blockers"]:
            blocked = self._publish_blocker(preflight)
            raise BaselineError(
                f"preflight blocked before reservation; evidence={blocked['bundle']} sha256={blocked['artifact_sha256']}"
            )
        try:
            authority, authority_bytes, authority_source = self._validated_authority_snapshot(preflight)
        except BaselineError as exc:
            blocked_report = self._blocked_after_ready_preflight(preflight, str(exc))
            blocked = self._publish_blocker(blocked_report)
            raise BaselineError(
                f"validated authority changed before reservation; evidence={blocked['bundle']} sha256={blocked['artifact_sha256']}"
            ) from exc
        require(authority.get("kind") == SUCCESSOR_AUTHORITY_KIND, "successor authority kind is required")
        run_id = authority["run_id"]
        with self._locked_successor_ledger() as ledger:
            self._validate_successor_authority_policy(authority, ledger)
            require(run_id not in ledger["runs"], "successor duplicate is permanently rejected")
            require(len(ledger["runs"]) == 1, "successor slot is already consumed; third attempt forbidden")
            orphaned = sorted(
                path.name
                for path in self.state_root.iterdir()
                if path.is_dir()
                and RUN_ID_RE.fullmatch(path.name)
                and path.name != PREDECESSOR_RUN_ID
            )
            require(not orphaned, "unbound or partially migrated successor state exists")

            def consume_slot_before_reservation_publish(
                reservation: dict[str, Any],
                reservation_sha256: str,
            ) -> None:
                self._revalidate_successor_authority_bundle_at_reservation_boundary(
                    ledger,
                    authority,
                    authority_bytes,
                    authority_source,
                )
                record = self.reserve_successor_slot(
                    ledger,
                    run_id=run_id,
                    authority_sha256=authority_source["sha256"],
                    reservation_sha256=reservation_sha256,
                    reserved_at_utc=reservation["reserved_at_utc"],
                )
                record.update(
                    {
                        "preflight_sha256": reservation["preflight_sha256"],
                        "command_sha256": reservation["command_sha256"],
                        "environment_sha256": reservation["environment_sha256"],
                    }
                )
                self._save_successor_ledger(ledger)

            state_dir, reservation, _ = self._create_reservation(
                authority,
                authority_bytes,
                authority_source,
                preflight,
                before_publish=consume_slot_before_reservation_publish,
            )
            self._failpoint("after_reservation")

            try:
                execution_plan = self._load_execution_plan(run_id)
            except (BaselineError, OSError, ValueError, TypeError) as exc:
                raise self._terminalize_successor_runtime(
                    ledger,
                    run_id,
                    "reservation_validation_fail_closed",
                    f"reservation-bound execution plan validation failed: {exc}; retry forbidden",
                ) from exc
            authority = execution_plan["authority"]
            reservation = execution_plan["reservation"]
            command = execution_plan["command"]
            environment = execution_plan["environment"]
            working_directory = execution_plan["working_directory"]
            raw = state_dir / "raw"
            self.mark_successor_invoked(ledger, run_id)
            record = ledger["runs"][run_id]
            self._save_successor_ledger(ledger)
            self._failpoint("after_execution_mark")

            try:
                completed = subprocess.run(
                    command,
                    cwd=working_directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=authority["timeout_seconds"],
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                atomic_write_bytes(state_dir / "stdout.log", output.encode("utf-8"))
                raise self._terminalize_successor_runtime(
                    ledger,
                    run_id,
                    "timeout_fail_closed",
                    "successor baseline command timed out; retry forbidden",
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                atomic_write_bytes(state_dir / "stdout.log", b"")
                raise self._terminalize_successor_runtime(
                    ledger,
                    run_id,
                    "command_launch_fail_closed",
                    f"successor baseline command could not launch: {exc}; retry forbidden",
                ) from exc
            atomic_write_bytes(state_dir / "stdout.log", completed.stdout.encode("utf-8"))
            if completed.returncode != 0:
                raise self._terminalize_successor_runtime(
                    ledger,
                    run_id,
                    "failed_fail_closed",
                    f"successor baseline command failed with exit {completed.returncode}; retry forbidden",
                    exit_status=completed.returncode,
                )
            try:
                self._verify_reserved_inputs(state_dir, reservation, authority)
                self._validate_outputs(raw, authority, environment)
            except (BaselineError, OSError, ValueError, TypeError) as exc:
                raise self._terminalize_successor_runtime(
                    ledger,
                    run_id,
                    "validation_fail_closed",
                    f"successor baseline output validation failed: {exc}; retry forbidden",
                ) from exc

            final_record = copy.deepcopy(record)
            final_record["state"] = "committed"
            final_record["terminal_at_utc"] = now_utc()
            final_record["bundle"] = self._bundle(run_id).relative_to(self.root).as_posix()
            final_ledger = copy.deepcopy(ledger)
            final_ledger["runs"][run_id] = final_record
            try:
                published = self._publish_success(
                    run_id,
                    state_dir,
                    authority,
                    preflight,
                    final_ledger,
                )
            except (BaselineError, OSError, ValueError, TypeError, shutil.Error) as exc:
                raise self._terminalize_successor_runtime(
                    ledger,
                    run_id,
                    "publication_fail_closed",
                    f"successor baseline publication failed: {exc}; retry forbidden",
                ) from exc
            self._failpoint("after_bundle_publish")
            ledger["runs"][run_id] = final_record
            self._save_successor_ledger(ledger)
            return published

    def run(self) -> dict[str, Any]:
        if self.successor_ledger_path.is_file():
            return self._run_successor()
        preflight = self.preflight()
        if preflight["blockers"]:
            blocked = self._publish_blocker(preflight)
            raise BaselineError(
                f"preflight blocked before reservation; evidence={blocked['bundle']} sha256={blocked['artifact_sha256']}"
            )
        try:
            authority, authority_bytes, authority_source = self._validated_authority_snapshot(preflight)
        except BaselineError as exc:
            blocked_report = self._blocked_after_ready_preflight(preflight, str(exc))
            blocked = self._publish_blocker(blocked_report)
            raise BaselineError(
                f"validated authority changed before reservation; evidence={blocked['bundle']} sha256={blocked['artifact_sha256']}"
            ) from exc
        run_id = authority["run_id"]
        with self._locked_ledger() as ledger:
            if ledger["runs"]:
                existing_run_id, existing_record = next(iter(ledger["runs"].items()))
                if self._state_dir(existing_run_id).is_dir():
                    self._publish_event_packet(
                        existing_run_id,
                        "duplicate",
                        "DUPLICATE",
                        existing_record,
                        "a baseline reservation already exists; retry forbidden",
                        attempted_run_id=run_id,
                    )
                raise BaselineError("a baseline reservation already exists; retry forbidden")
            orphaned = sorted(
                path.name
                for path in self.state_root.iterdir()
                if path.is_dir() and RUN_ID_RE.fullmatch(path.name)
            )
            if orphaned:
                orphaned_run_id = orphaned[0]
                reservation = self._load_reservation(orphaned_run_id)
                orphaned_record = self._reserved_record(reservation)
                orphaned_record["state"] = "interrupted_before_ledger_fail_closed"
                orphaned_record["terminal_at_utc"] = now_utc()
                orphaned_record["error"] = "durable reservation exists without a ledger record"
                ledger["runs"][orphaned_run_id] = orphaned_record
                self._save_ledger(ledger)
                self._publish_event_packet(
                    orphaned_run_id,
                    "recovery",
                    "RECOVERY",
                    orphaned_record,
                    orphaned_record["error"],
                    attempted_run_id=run_id,
                )
                raise BaselineError("an orphaned durable reservation exists; retry forbidden")

            state_dir, _, record = self._create_reservation(
                authority,
                authority_bytes,
                authority_source,
                preflight,
            )
            ledger["runs"][run_id] = record
            self._save_ledger(ledger)
            self._failpoint("after_reservation")

            try:
                execution_plan = self._load_execution_plan(run_id)
            except (BaselineError, OSError, ValueError, TypeError) as exc:
                raise self._terminalize(
                    ledger,
                    record,
                    "reservation_validation_fail_closed",
                    f"reservation-bound execution plan validation failed: {exc}; retry forbidden",
                ) from exc
            authority = execution_plan["authority"]
            reservation = execution_plan["reservation"]
            command = execution_plan["command"]
            environment = execution_plan["environment"]
            working_directory = execution_plan["working_directory"]
            raw = state_dir / "raw"
            record["state"] = "running"
            record["execution_count"] = 1
            record["model_counts"] = {"baseline_model": 1, "candidate_model": 0}
            record["execution_started_at_utc"] = now_utc()
            self._save_ledger(ledger)
            self._failpoint("after_execution_mark")

            try:
                completed = subprocess.run(
                    command,
                    cwd=working_directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=authority["timeout_seconds"],
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                atomic_write_bytes(state_dir / "stdout.log", output.encode("utf-8"))
                raise self._terminalize(
                    ledger,
                    record,
                    "timeout_fail_closed",
                    "baseline command timed out; retry forbidden",
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                atomic_write_bytes(state_dir / "stdout.log", b"")
                raise self._terminalize(
                    ledger,
                    record,
                    "command_launch_fail_closed",
                    f"baseline command could not launch: {exc}; retry forbidden",
                ) from exc
            atomic_write_bytes(state_dir / "stdout.log", completed.stdout.encode("utf-8"))
            if completed.returncode != 0:
                raise self._terminalize(
                    ledger,
                    record,
                    "failed_fail_closed",
                    f"baseline command failed with exit {completed.returncode}; retry forbidden",
                    exit_status=completed.returncode,
                )
            try:
                self._verify_reserved_inputs(state_dir, reservation, authority)
                self._validate_outputs(raw, authority, environment)
            except (BaselineError, OSError, ValueError, TypeError) as exc:
                raise self._terminalize(
                    ledger,
                    record,
                    "validation_fail_closed",
                    f"baseline output validation failed: {exc}; retry forbidden",
                ) from exc

            final_record = copy.deepcopy(record)
            final_record["state"] = "committed"
            final_record["terminal_at_utc"] = now_utc()
            final_record["bundle"] = self._bundle(run_id).relative_to(self.root).as_posix()
            final_ledger = copy.deepcopy(ledger)
            final_ledger["runs"][run_id] = final_record
            try:
                published = self._publish_success(
                    run_id,
                    state_dir,
                    authority,
                    preflight,
                    final_ledger,
                )
            except (BaselineError, OSError, ValueError, TypeError, shutil.Error) as exc:
                raise self._terminalize(
                    ledger,
                    record,
                    "publication_fail_closed",
                    f"baseline publication failed: {exc}; retry forbidden",
                ) from exc
            self._failpoint("after_bundle_publish")
            ledger["runs"][run_id] = final_record
            self._save_ledger(ledger)
            return published

    def recover(self, run_id: str) -> dict[str, Any]:
        require(RUN_ID_RE.fullmatch(run_id) is not None, "invalid run_id")
        if run_id == PREDECESSOR_RUN_ID and self.predecessor_ledger_path.is_file():
            raise BaselineError("sealed v1 recovery is permanently rejected; process rerun forbidden")
        if self.successor_ledger_path.is_file():
            with self._locked_successor_ledger() as ledger:
                require(run_id in ledger["runs"], "no durable successor reservation to recover")
                record = ledger["runs"][run_id]
                if record["state"] == "committed":
                    return self._verify_bundle(self._bundle(run_id))
                if record["state"] == "running" and self._bundle(run_id).is_dir():
                    published = self._verify_bundle(self._bundle(run_id))
                    ledger["runs"][run_id] = published["final_ledger"]["runs"][run_id]
                    self._save_successor_ledger(ledger)
                    return published
                self.recover_successor_policy(ledger, run_id)
                self._save_successor_ledger(ledger)
                raise BaselineError(f"successor recovery never reruns: {ledger['runs'][run_id]['state']}")
        with self._locked_ledger() as ledger:
            if run_id not in ledger["runs"] and self._state_dir(run_id).is_dir():
                ledger["runs"][run_id] = self._reserved_record(self._load_reservation(run_id))
                self._save_ledger(ledger)
            require(run_id in ledger["runs"], "no durable reservation to recover")
            record = ledger["runs"][run_id]
            if record["state"] == "committed":
                published = self._verify_bundle(self._bundle(run_id))
                if self._event_bundle(run_id, "recovery").exists():
                    self._verify_event_packet(self._event_bundle(run_id, "recovery"), "RECOVERY")
                return published
            if record["state"] == "reserved":
                record["state"] = "interrupted_before_execution_fail_closed"
                record["terminal_at_utc"] = now_utc()
                record["error"] = "reserved run was interrupted; retry forbidden"
                self._save_ledger(ledger)
                self._publish_event_packet(
                    run_id,
                    "recovery",
                    "RECOVERY",
                    record,
                    record["error"],
                )
                raise BaselineError(record["error"])
            if record["state"] == "running":
                try:
                    published = self._verify_bundle(self._bundle(run_id))
                except BaselineError as exc:
                    record["state"] = "interrupted_after_execution_fail_closed"
                    record["terminal_at_utc"] = now_utc()
                    record["error"] = "execution started without a valid atomic bundle; retry forbidden"
                    self._save_ledger(ledger)
                    self._publish_event_packet(
                        run_id,
                        "recovery",
                        "RECOVERY",
                        record,
                        f"{record['error']}: {exc}",
                    )
                    raise BaselineError(record["error"]) from exc
                recovered_record = published["final_ledger"]["runs"][run_id]
                ledger["runs"][run_id] = recovered_record
                self._save_ledger(ledger)
                self._publish_event_packet(
                    run_id,
                    "recovery",
                    "RECOVERY",
                    recovered_record,
                    "valid atomic bundle finalized without re-execution",
                )
                return published
            event = "terminal" if self._event_bundle(run_id, "terminal").exists() else "recovery"
            label = "TERMINAL" if event == "terminal" else "RECOVERY"
            self._publish_event_packet(
                run_id,
                event,
                label,
                record,
                f"run is terminally fail-closed: {record['state']}",
            )
            raise BaselineError(f"run is terminally fail-closed: {record['state']}")

    def verify(self, run_id: str) -> dict[str, Any]:
        require(RUN_ID_RE.fullmatch(run_id) is not None, "invalid run_id")
        if run_id == PREDECESSOR_RUN_ID and self.predecessor_ledger_path.is_file():
            raise BaselineError("sealed v1 verify-as-run is permanently rejected")
        if self.successor_ledger_path.is_file():
            with self._locked_successor_ledger() as ledger:
                require(run_id in ledger["runs"], "run_id is absent from successor ledger")
                record = ledger["runs"][run_id]
                require(record["state"] == "committed", "successor run is not committed")
                require(record["execution_count"] == 1, "successor execution count differs")
                require(record["model_counts"] == {"baseline_model": 1, "candidate_model": 0}, "successor model counts differ")
                published = self._verify_bundle(self._bundle(run_id))
                require(
                    published["final_ledger"]["runs"][run_id] == record,
                    "durable successor ledger differs from hash-bound final ledger snapshot",
                )
                return published
        with self._locked_ledger() as ledger:
            require(run_id in ledger["runs"], "run_id is absent from ledger")
            record = ledger["runs"][run_id]
            require(record["state"] == "committed", "run is not committed")
            require(record["execution_count"] == 1, "execution count differs")
            require(record["model_counts"] == {"baseline_model": 1, "candidate_model": 0}, "ledger model counts differ")
            published = self._verify_bundle(self._bundle(run_id))
            require(
                published["final_ledger"]["runs"][run_id] == record,
                "durable ledger differs from hash-bound final ledger snapshot",
            )
            return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("migrate-successor-ledger")
    subparsers.add_parser("run")
    for action in ("recover", "verify"):
        child = subparsers.add_parser(action)
        child.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner = DynamicScale32BaselineRunner(args.root)
    try:
        if args.action == "migrate-successor-ledger":
            result = runner.migrate_successor_ledger()
            print(
                "DYNAMIC_SCALE32_SUCCESSOR_MIGRATION_PASS "
                f"created={str(result['created']).lower()} sha256={result['ledger']['sha256']}"
            )
            return 0
        if args.action == "preflight":
            report = runner.preflight()
            if report["blockers"]:
                blocked = runner._publish_blocker(report)
                raise BaselineError(
                    f"preflight blocked before reservation; evidence={blocked['bundle']} sha256={blocked['artifact_sha256']}"
                )
            print(f"DYNAMIC_SCALE32_BASELINE_PREFLIGHT_PASS sha256={report['preflight_sha256']}")
            return 0
        if args.action == "run":
            result = runner.run()
        elif args.action == "recover":
            result = runner.recover(args.run_id)
        else:
            result = runner.verify(args.run_id)
    except BaselineError as exc:
        print(f"DYNAMIC_SCALE32_BASELINE_FAIL {exc}", file=sys.stderr)
        return 2
    print(
        "DYNAMIC_SCALE32_BASELINE_PASS "
        f"bundle={result['bundle']} manifest_sha256={result['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
