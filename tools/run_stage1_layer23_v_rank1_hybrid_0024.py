#!/usr/bin/env python3
"""Phase-safe fixed-input-scale launcher for nonofficial hybrid attempt 0024."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0024"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0024.json"
SELECTION = PRIVATE / "candidate-selection-0024.json"
CANDIDATE_BUDGET_REGRESSION = PRIVATE / "candidate-budget-regression-0024.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0024.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0024.json"
INPUT_SCALE_REGRESSION = PRIVATE / "fixed-input-scale-regression-0024.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0024.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0024.json"
GATE_ORDER_REGRESSION = PRIVATE / "gate-order-regression-0024.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0024.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0024.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0024.json"
ARGV_SHAPE_REGRESSION = PRIVATE / "argv-shape-regression-0024.json"
CLOSURE_CERTIFICATE = PRIVATE / "closure-certificate-0024.json"
PHASE_TRANSITION_REGRESSION = PRIVATE / "phase-transition-regression-0024.json"
INVOCATION_REGRESSION = PRIVATE / "invocation-fidelity-regression-0024.json"
ORCHESTRATION_REGRESSION = PRIVATE / "orchestration-regression-0024.json"
PACKAGE = OUTPUT / "execution-package.json"
AUTHORITY = OUTPUT / "execution-authority.json"
STATE = OUTPUT / "supervisor-state"
SUPERVISOR = ROOT / "tools/ace2_exact_once_runtime_supervisor.py"

SEALED_0023_OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0023"
)
SEALED_0023_TERMINAL = SEALED_0023_OUTPUT / "terminal-record.json"
SEALED_0023_LAUNCHER = (
    ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0023.py"
)
SEALED_0023_PROMPTS = PRIVATE / "prompts-0023.json"
SEALED_0023_SELECTION = PRIVATE / "candidate-selection-0023.json"

PRE_IMPORT_ENVIRONMENT = dict(os.environ)
PRE_IMPORT_SYS_PATH = list(sys.path)
PRE_IMPORT_SYS_PREFIX = sys.prefix
PRE_IMPORT_ARGV = list(sys.argv)
PRE_IMPORT_CWD = str(Path.cwd().resolve())
PRE_IMPORT_MODULES = frozenset(sys.modules)

FROZEN_EXPECTED_ENVIRONMENT = {
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
EXPECTED_PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
REQUIRED_PREIMPORT_ABSENT = (
    "numpy",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "tools",
)

SEALED_0023_TERMINAL_SHA256 = (
    "44d8aaebc6af9e492f5731013d9457796e503d0832e90d6e5ecc83527948626f"
)
SEALED_0023_LAUNCHER_SHA256 = (
    "4109d2da32702afaa7e3bcdfd05db639d218b59110d7e7aeaa4e92de49cd35b6"
)
SEALED_0023_PROMPTS_SHA256 = (
    "fe029b7ff2ebb47338cce49a46cf83b653d1ac77b32f3c164f4cdfb72ffba786"
)
SEALED_0023_SELECTION_SHA256 = (
    "8e748b5b3a45b3dfb8fea1050dfe234d0275bed27eb738e18cef8fb8f4949256"
)

EXPECTED_DENY_ID_COUNT = 42
EXPECTED_DENY_HASH_COUNT = 52
EXPECTED_DENY_ID_DIGEST = (
    "ccf30e4e14552c3966ed5cb28222bc9aa29b6c6237544b6ac0c6264fbc4965d6"
)
EXPECTED_DENY_HASH_DIGEST = (
    "10c1f494c5684f8e285780ae33ba849f54d63304a0d96b4059f8d841dd1d1154"
)

CANDIDATES = (
    {"id": "fresh-0024-a", "text": "How do willows sway?"},
    {"id": "fresh-0024-b", "text": "Where do otters sleep?"},
)
EXPECTED_SELECTED_CANDIDATES = (
    {
        **CANDIDATES[0],
        "sha256": "e0ebe0809c483d974752ea968317b66e4601b276c401c67e7a7dc02b6bc6a638",
        "prompt_token_count": 35,
        "total_context_token_count": 39,
    },
    {
        **CANDIDATES[1],
        "sha256": "af7566dec521f3ec5c069cb169976cecdfdd1e7aa381fc7b26ae832c7cf3d515",
        "prompt_token_count": 35,
        "total_context_token_count": 39,
    },
)
OVER_BUDGET_CANDIDATE = {
    "id": "fresh-0024-over-budget",
    "text": "Why do pebbles feel smooth?",
}
REUSED_ID_CANDIDATE = {
    "id": "fresh-0023-a",
    "text": "How do willows sway?",
}
REUSED_HASH_CANDIDATE = {
    "id": "fresh-0024-hash-reuse",
    "text": "Why do reeds bend?",
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_environment_snapshot(path: Path) -> dict[str, str]:
    _require(
        stat.S_ISREG(os.lstat(path).st_mode),
        "ordinary inherited environment snapshot is not a regular file",
    )
    environment: dict[str, str] = {}
    for entry in path.read_bytes().split(b"\0"):
        if not entry:
            continue
        name, separator, value = entry.partition(b"=")
        _require(
            bool(separator) and bool(name),
            "ordinary inherited environment snapshot contains a malformed entry",
        )
        decoded_name = os.fsdecode(name)
        _require(
            decoded_name not in environment,
            "ordinary inherited environment snapshot contains a duplicate key",
        )
        environment[decoded_name] = os.fsdecode(value)
    return environment


def _environment_accepted(environment: dict[str, str]) -> bool:
    return environment == FROZEN_EXPECTED_ENVIRONMENT


def _verify_0023_stdlib_seal() -> dict[str, Any]:
    paths = (
        SEALED_0023_TERMINAL,
        SEALED_0023_LAUNCHER,
        SEALED_0023_PROMPTS,
        SEALED_0023_SELECTION,
    )
    for path in paths:
        _require(
            stat.S_ISREG(os.lstat(path).st_mode),
            f"sealed 0023 artifact is not regular: {path}",
        )
    terminal = json.loads(SEALED_0023_TERMINAL.read_text(encoding="utf-8"))
    prompts = json.loads(SEALED_0023_PROMPTS.read_text(encoding="utf-8"))
    selection = json.loads(SEALED_0023_SELECTION.read_text(encoding="utf-8"))
    selected_prompts = [
        {"id": item["id"], "text": item["text"]} for item in selection["selected"]
    ]
    _require(
        _sha256_file(SEALED_0023_TERMINAL) == SEALED_0023_TERMINAL_SHA256
        and _sha256_file(SEALED_0023_LAUNCHER) == SEALED_0023_LAUNCHER_SHA256
        and _sha256_file(SEALED_0023_PROMPTS) == SEALED_0023_PROMPTS_SHA256
        and _sha256_file(SEALED_0023_SELECTION) == SEALED_0023_SELECTION_SHA256
        and terminal.get("attempt_identity") == "nonofficial-hybrid-0023"
        and terminal.get("status") == "FAILED_SEALED_NO_EXECUTION"
        and terminal.get("outcome") == "pre_execution_failure"
        and terminal.get("process_start_count") == 0
        and terminal.get("process_started") is False
        and selection.get("attempt_identity") == "nonofficial-hybrid-0023"
        and selection.get("status") == "PASS_NONCONSUMING_CANDIDATE_SELECTION"
        and selection.get("deny_set", {}).get("combined_id_count") == 40
        and selection.get("deny_set", {}).get("combined_hash_count") == 50
        and prompts == {"schema_version": 1, "prompts": selected_prompts},
        "sealed 0023 terminal, launcher, prompts, or selection changed",
    )
    return {
        "terminal_sha256": SEALED_0023_TERMINAL_SHA256,
        "launcher_sha256": SEALED_0023_LAUNCHER_SHA256,
        "prompts_sha256": SEALED_0023_PROMPTS_SHA256,
        "selection_sha256": SEALED_0023_SELECTION_SHA256,
        "process_start_count": 0,
        "process_started": False,
        "status": "FAILED_SEALED_NO_EXECUTION",
    }


def _invocation_regression_record(
    inherited_environment: dict[str, str],
) -> dict[str, Any]:
    expected = FROZEN_EXPECTED_ENVIRONMENT
    missing = sorted(set(expected) - set(inherited_environment))
    changed = sorted(
        name
        for name in expected.keys() & inherited_environment.keys()
        if inherited_environment[name] != expected[name]
    )
    inherited_expected_key_count = len(set(expected) & set(inherited_environment))
    inherited_extra_key_count = len(set(inherited_environment) - set(expected))
    forbidden_present = sorted(
        name
        for name in REQUIRED_PREIMPORT_ABSENT
        if any(
            module == name or module.startswith(f"{name}.")
            for module in PRE_IMPORT_MODULES
        )
    )
    extra_environment = {**expected, "ACE2_INVOCATION_FIDELITY_EXTRA": "1"}
    missing_environment = dict(expected)
    del missing_environment["HF_HUB_OFFLINE"]
    changed_environment = {**expected, "HF_HUB_OFFLINE": "0"}
    ordinary_differs = bool(missing or changed or inherited_extra_key_count)
    checks = {
        "ordinary_inherited_environment_differs": ordinary_differs,
        "ordinary_inherited_environment_rejected": not _environment_accepted(
            inherited_environment
        ),
        "exact_env_i_snapshot_equals_frozen_map": PRE_IMPORT_ENVIRONMENT == expected,
        "exact_env_i_snapshot_accepted": _environment_accepted(
            PRE_IMPORT_ENVIRONMENT
        ),
        "extra_variable_rejected": not _environment_accepted(extra_environment),
        "missing_variable_rejected": not _environment_accepted(missing_environment),
        "changed_variable_rejected": not _environment_accepted(changed_environment),
        "no_project_or_dependency_module_imported_before_snapshot": (
            not forbidden_present
        ),
        "sealed_0023_failed_without_execution": (
            _verify_0023_stdlib_seal()["process_start_count"] == 0
        ),
        "process_not_started": True,
    }
    return {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid24execute",
        "attempt_identity": "nonofficial-hybrid-0024",
        "status": "PASS_EXACT_ENV_I_INVOCATION_FIDELITY",
        "model_executed": False,
        "simulator_executed": False,
        "process_start_count": 0,
        "frozen_expected_environment": expected,
        "exact_env_i_snapshot": PRE_IMPORT_ENVIRONMENT,
        "ordinary_inherited_environment": {
            "entry_count": len(inherited_environment),
            "expected_key_count": inherited_expected_key_count,
            "extra_key_count": inherited_extra_key_count,
            "missing_expected_keys": missing,
            "changed_expected_keys": changed,
            "accepted": False,
        },
        "rejection_cases": {
            "extra": {"accepted": False},
            "missing": {"accepted": False},
            "changed": {"accepted": False},
        },
        "pre_import_snapshot": {
            "module_count": len(PRE_IMPORT_MODULES),
            "forbidden_modules_present": forbidden_present,
        },
        "sealed_0023": _verify_0023_stdlib_seal(),
        "checks": checks,
    }


def _verify_invocation_regression() -> dict[str, Any]:
    _require(
        stat.S_ISREG(os.lstat(INVOCATION_REGRESSION).st_mode),
        "0024 invocation-fidelity regression is absent or not regular",
    )
    record = json.loads(INVOCATION_REGRESSION.read_text(encoding="utf-8"))
    ordinary = record.get("ordinary_inherited_environment", {})
    ordinary_differs = bool(
        ordinary.get("extra_key_count")
        or ordinary.get("missing_expected_keys")
        or ordinary.get("changed_expected_keys")
    )
    _require(
        record.get("schema_version") == 1
        and record.get("mission_id") == "stage1rank1hybrid24execute"
        and record.get("attempt_identity") == "nonofficial-hybrid-0024"
        and record.get("status") == "PASS_EXACT_ENV_I_INVOCATION_FIDELITY"
        and record.get("model_executed") is False
        and record.get("simulator_executed") is False
        and record.get("process_start_count") == 0
        and record.get("frozen_expected_environment")
        == FROZEN_EXPECTED_ENVIRONMENT
        and record.get("exact_env_i_snapshot") == FROZEN_EXPECTED_ENVIRONMENT
        and ordinary.get("accepted") is False
        and ordinary_differs
        and ordinary.get("entry_count")
        == ordinary.get("expected_key_count") + ordinary.get("extra_key_count")
        and record.get("sealed_0023") == _verify_0023_stdlib_seal()
        and record.get("checks")
        and all(record["checks"].values()),
        "0024 invocation-fidelity regression changed or is incomplete",
    )
    return record


def _run_invocation_regression(inherited_snapshot: Path) -> int:
    _require(
        Path(sys.executable).resolve() == EXPECTED_PYTHON,
        "Python interpreter binding changed",
    )
    _require(
        PRE_IMPORT_ENVIRONMENT == FROZEN_EXPECTED_ENVIRONMENT,
        "invocation-fidelity gate was not launched with the exact env-i map",
    )
    _require(PRE_IMPORT_CWD == str(ROOT), "invocation-fidelity gate cwd changed")
    _require(
        Path(PRE_IMPORT_ARGV[0]).resolve() == LAUNCHER,
        "invocation-fidelity gate launcher changed",
    )
    _require(
        not OUTPUT.exists() and not INVOCATION_REGRESSION.exists(),
        "0024 invocation or execution namespace already exists",
    )
    record = _invocation_regression_record(
        _read_environment_snapshot(inherited_snapshot)
    )
    _require(
        all(record["checks"].values()),
        "0024 invocation-fidelity regression failed",
    )
    INVOCATION_REGRESSION.parent.mkdir(parents=True, exist_ok=True)
    with INVOCATION_REGRESSION.open("xb") as stream:
        stream.write(_canonical_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    _verify_invocation_regression()
    print(
        "ACE2_HYBRID_0024_INVOCATION_FIDELITY_REGRESSION_PASS "
        "exact_env_i=true inherited_rejected=true mutations_rejected=3 "
        "sealed_0023_process_start_count=0 process_start_count=0",
        flush=True,
    )
    return 0


def _sanitized_traceback(error: BaseException) -> dict[str, Any]:
    frames = []
    rendered = ["Traceback (most recent call last):"]
    for frame in traceback.extract_tb(error.__traceback__):
        path = Path(frame.filename)
        try:
            source_file = str(path.resolve().relative_to(ROOT))
        except (OSError, ValueError):
            source_file = f"<external>/{path.name}"
        frames.append(
            {
                "source_file": source_file,
                "function": frame.name,
                "line": frame.lineno,
                "expression": frame.line or "",
            }
        )
        rendered.append(f'  File "{source_file}", line {frame.lineno}, in {frame.name}')
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
        "mission_id": "stage1rank1hybrid24execute",
        "attempt_identity": "nonofficial-hybrid-0024",
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
            "regression": "No retry, replay, or resume; preserve all 0024 artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _collect_prompts(value: Any, ids: set[str], hashes: set[str]) -> None:
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
            hashes.add(hashlib.sha256(prompt_text.encode("utf-8")).hexdigest())
        if isinstance(value.get("id"), str) and isinstance(value.get("text"), str):
            ids.add(value["id"])
            hashes.add(hashlib.sha256(value["text"].encode("utf-8")).hexdigest())
        for nested in value.values():
            _collect_prompts(nested, ids, hashes)
    elif isinstance(value, list):
        for nested in value:
            _collect_prompts(nested, ids, hashes)


def _diagnostic_files(prior_0014: Any) -> list[Path]:
    files: list[Path] = []
    for source in prior_0014.DIAGNOSTIC_PROMPT_SOURCES:
        if source.is_dir():
            files.extend(sorted(source.rglob("*.json")))
        else:
            files.append(source)
    _require(
        files
        and len(files) == len(set(files))
        and all(path.is_file() for path in files),
        "0024 diagnostic prompt source set is incomplete",
    )
    return files


def _recompute_deny_set(
    runner: Any,
    prior_0014: Any,
) -> tuple[set[str], set[str], dict[str, Any]]:
    prompt_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for name, expected_sha256 in sorted(
        prior_0014.HISTORICAL_PROMPT_SHA256.items()
    ):
        path = PRIVATE / name
        _require(
            _sha256_file(path) == expected_sha256,
            f"sealed historical prompt artifact changed: {name}",
        )
        for prompt in runner.load_prompts(path):
            prompt_ids.add(prompt["id"])
            prompt_hashes.add(
                hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
            )
    historical_id_count = len(prompt_ids)
    historical_hash_count = len(prompt_hashes)
    diagnostic_files = _diagnostic_files(prior_0014)
    for path in diagnostic_files:
        _collect_prompts(runner.read_json(path), prompt_ids, prompt_hashes)
    metadata = {
        "historical_prompt_artifact_count": len(
            prior_0014.HISTORICAL_PROMPT_SHA256
        ),
        "historical_id_count": historical_id_count,
        "historical_hash_count": historical_hash_count,
        "diagnostic_source_count": len(diagnostic_files),
        "combined_id_count": len(prompt_ids),
        "combined_hash_count": len(prompt_hashes),
        "combined_id_digest": _canonical_digest(sorted(prompt_ids)),
        "combined_hash_digest": _canonical_digest(sorted(prompt_hashes)),
        "diagnostic_sources": [
            runner.file_record(path) for path in diagnostic_files
        ],
    }
    return prompt_ids, prompt_hashes, metadata


def _reservation_allows_child() -> bool:
    intent_path = STATE / "pre_start_intent.json"
    if not intent_path.is_file():
        return False
    try:
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        intent.get("schema") == "ace2-exact-once-pre-start-intent-v1"
        and intent.get("process_start_count") == 0
    )


def _run_orchestration_regression() -> int:
    _require(
        Path(sys.executable).resolve() == EXPECTED_PYTHON,
        "Python interpreter binding changed",
    )
    _require(
        PRE_IMPORT_ENVIRONMENT == FROZEN_EXPECTED_ENVIRONMENT,
        "orchestration gate requires exact env-i",
    )
    _require(PRE_IMPORT_CWD == str(ROOT), "orchestration gate cwd changed")
    _require(
        PACKAGE.is_file() and AUTHORITY.is_file(),
        "package and authority must precede orchestration gate",
    )
    _require(
        not ORCHESTRATION_REGRESSION.exists(),
        "0024 orchestration regression already exists",
    )
    _require(
        not STATE.exists() and not TERMINAL.exists(),
        "orchestration gate must be nonconsuming",
    )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools import ace2_exact_once_runtime_supervisor as supervisor

    validated = supervisor.validate_run(PACKAGE, AUTHORITY)
    package = validated.package
    authority = validated.authority
    child_argv = package["bindings"]["argv"]
    child_argv_sha256 = supervisor.canonical_value_sha256(child_argv)
    supervisor_command = [
        str(EXPECTED_PYTHON),
        str(SUPERVISOR),
        "--package",
        str(PACKAGE),
        "--authority",
        str(AUTHORITY),
    ]
    tampered_argv = [*child_argv[:-1], "f" * 64]
    tampered_rejected = (
        tampered_argv != child_argv
        and supervisor.canonical_value_sha256(tampered_argv)
        != child_argv_sha256
    )
    with tempfile.TemporaryDirectory(prefix="ace2-0024-duplicate-fixture-") as temporary:
        duplicate_state = Path(temporary)
        (duplicate_state / "pre_start_intent.json").write_text(
            "{}\n", encoding="utf-8"
        )
        try:
            supervisor._check_state_available(duplicate_state)
        except supervisor.DuplicateExecution:
            duplicate_rejected = True
        else:
            duplicate_rejected = False

    supervisor_source = SUPERVISOR.read_text(encoding="utf-8")
    run_once_source = supervisor_source[supervisor_source.index("def run_once(") :]
    supervisor_order = [
        run_once_source.index("_reserve_pre_start_intent(validated)"),
        run_once_source.index("subprocess.Popen("),
        run_once_source.index("_write_started_metadata("),
    ]
    attestation_source = (
        ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0022.py"
    ).read_text(encoding="utf-8")
    attestation_source = attestation_source[
        attestation_source.index("def child_entry_attestation_0022(") :
        attestation_source.index(
            "runner.child_entry_attestation = child_entry_attestation_0022"
        )
    ]
    child_order = [
        attestation_source.index(
            'intent = runner.read_json(runner.STATE / "pre_start_intent.json")'
        ),
        attestation_source.index("started = wait_for_process_started()"),
        attestation_source.index(
            'runner.write_json(runner.STATE / "child_entry_attestation.json"'
        ),
    ]
    checks = {
        "direct_execute_without_reservation_forbidden": (
            not _reservation_allows_child()
        ),
        "supervisor_command_binds_exact_package_and_authority": (
            supervisor_command[2:]
            == ["--package", str(PACKAGE), "--authority", str(AUTHORITY)]
        ),
        "package_authority_bind_exact_seven_element_child_argv": (
            len(child_argv) == 7
            and authority["bindings"]["argv"] == child_argv
            and package["bindings"]["argv_sha256"] == child_argv_sha256
            and authority["bindings"]["argv_sha256"] == child_argv_sha256
        ),
        "intent_before_popen_before_process_started": (
            supervisor_order == sorted(supervisor_order)
        ),
        "intent_and_started_before_child_attestation": (
            child_order == sorted(child_order)
        ),
        "duplicate_child_reservation_rejected": duplicate_rejected,
        "tampered_child_argv_rejected": tampered_rejected,
        "no_attempt_state_consumed": not STATE.exists() and not TERMINAL.exists(),
        "model_and_simulator_not_executed": True,
    }
    record = {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid24execute",
        "attempt_identity": "nonofficial-hybrid-0024",
        "status": "PASS_EXTERNAL_EXACT_ONCE_ORCHESTRATION_REGRESSION",
        "model_executed": False,
        "simulator_executed": False,
        "process_start_count": 0,
        "package": {
            "path": str(PACKAGE),
            "byte_count": validated.package_measurement.byte_count,
            "sha256": validated.package_measurement.sha256,
        },
        "authority": {
            "path": str(AUTHORITY),
            "byte_count": validated.authority_measurement.byte_count,
            "sha256": validated.authority_measurement.sha256,
        },
        "supervisor_command": supervisor_command,
        "child_argv": child_argv,
        "child_argv_sha256": child_argv_sha256,
        "ordering": [
            "pre_start_intent_fsync",
            "popen",
            "process_started_fsync",
            "child_attestation",
        ],
        "checks": checks,
    }
    _require(all(checks.values()), "0024 orchestration regression failed")
    ORCHESTRATION_REGRESSION.parent.mkdir(parents=True, exist_ok=True)
    with ORCHESTRATION_REGRESSION.open("xb") as stream:
        stream.write(_canonical_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "ACE2_HYBRID_0024_ORCHESTRATION_REGRESSION_PASS "
        f"package_sha256={validated.package_measurement.sha256} "
        f"authority_sha256={validated.authority_measurement.sha256} "
        f"child_argv_sha256={child_argv_sha256} process_start_count=0",
        flush=True,
    )
    return 0


def _configure_0024(
    template: Any,
    runner: Any,
    prior_0014: Any,
    model_rtl_execute: Any,
) -> None:
    runner.MISSION_ID = "stage1rank1hybrid24execute"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0024"
    runner.PREDECESSOR_TERMINAL_SEALS = {
        **runner.PREDECESSOR_TERMINAL_SEALS,
        "nonofficial-hybrid-0023": {
            "sha256": SEALED_0023_TERMINAL_SHA256,
            "process_start_count": 0,
            "process_started": False,
        },
    }

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_verify_exact_interpreter = runner.verify_exact_interpreter
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_freeze = runner.freeze
    prior_verify = runner.verify
    current_child_entry_attestation = runner.child_entry_attestation

    def verify_predecessor_seals_0024() -> None:
        prior_verify_predecessor_seals()
        _verify_0023_stdlib_seal()

    runner.verify_predecessor_seals = verify_predecessor_seals_0024

    def verify_exact_interpreter_0024() -> None:
        prior_verify_exact_interpreter()
        _verify_invocation_regression()

    runner.verify_exact_interpreter = verify_exact_interpreter_0024

    def probe_candidate(
        tokenizer: Any,
        candidate: dict[str, str],
        rejection: str,
    ) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(candidate["text"].encode("utf-8")).hexdigest()
        prompt_token_count = len(
            runner.generation_runner.canonical_chat_token_ids(
                tokenizer, candidate["text"]
            )
        )
        return {
            **candidate,
            "sha256": prompt_hash,
            "prompt_token_count": prompt_token_count,
            "total_context_token_count": prompt_token_count + runner.STEPS,
            "rejection": rejection,
        }

    def expected_candidate_budget_regression() -> dict[str, Any]:
        selection = runner.verify_selection_certificate()
        prompt_ids, prompt_hashes, deny_metadata = _recompute_deny_set(
            runner, prior_0014
        )
        snapshot, _model, _tokenizer_json, _tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        tokenizer = runner.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        over_budget = probe_candidate(
            tokenizer,
            OVER_BUDGET_CANDIDATE,
            "prompt_plus_four_exceeds_40",
        )
        reused_id = probe_candidate(
            tokenizer,
            REUSED_ID_CANDIDATE,
            "sealed_prompt_id_reused",
        )
        reused_hash = probe_candidate(
            tokenizer,
            REUSED_HASH_CANDIDATE,
            "sealed_prompt_hash_reused",
        )
        selected = selection["selected"]
        deleted_id_set = set(prompt_ids)
        deleted_id = min(deleted_id_set)
        deleted_id_set.remove(deleted_id)
        deleted_hash_set = set(prompt_hashes)
        deleted_hash = min(deleted_hash_set)
        deleted_hash_set.remove(deleted_hash)
        added_id_set = {*prompt_ids, "fresh-0024-addition-probe"}
        added_hash_set = {*prompt_hashes, "0" * 64}
        mutations = {
            "deleted_id": {
                "entry_sha256": hashlib.sha256(deleted_id.encode("utf-8")).hexdigest(),
                "count": len(deleted_id_set),
                "digest": _canonical_digest(sorted(deleted_id_set)),
            },
            "deleted_hash": {
                "entry_sha256": hashlib.sha256(
                    deleted_hash.encode("utf-8")
                ).hexdigest(),
                "count": len(deleted_hash_set),
                "digest": _canonical_digest(sorted(deleted_hash_set)),
            },
            "added_id": {
                "entry_sha256": hashlib.sha256(
                    b"fresh-0024-addition-probe"
                ).hexdigest(),
                "count": len(added_id_set),
                "digest": _canonical_digest(sorted(added_id_set)),
            },
            "added_hash": {
                "entry_sha256": hashlib.sha256(("0" * 64).encode("utf-8")).hexdigest(),
                "count": len(added_hash_set),
                "digest": _canonical_digest(sorted(added_hash_set)),
            },
        }
        checks = {
            "exactly_two_selected": (
                len(selected) == 2
                and len({item["id"] for item in selected}) == 2
                and len({item["sha256"] for item in selected}) == 2
            ),
            "selected_counts_and_hashes_match_external_screen": (
                selected == list(EXPECTED_SELECTED_CANDIDATES)
            ),
            "selected_prompt_plus_four_within_40": all(
                item["prompt_token_count"] + runner.STEPS <= 40
                and item["total_context_token_count"]
                == item["prompt_token_count"] + runner.STEPS
                for item in selected
            ),
            "complete_sealed_deny_set_recomputed": (
                deny_metadata == selection["deny_set"]
                and len(prompt_ids) == EXPECTED_DENY_ID_COUNT
                and len(prompt_hashes) == EXPECTED_DENY_HASH_COUNT
                and _canonical_digest(sorted(prompt_ids)) == EXPECTED_DENY_ID_DIGEST
                and _canonical_digest(sorted(prompt_hashes))
                == EXPECTED_DENY_HASH_DIGEST
            ),
            "selected_ids_are_not_reused": not (
                {item["id"] for item in selected} & prompt_ids
            ),
            "selected_hashes_are_not_reused": not (
                {item["sha256"] for item in selected} & prompt_hashes
            ),
            "over_budget_candidate_rejected": (
                over_budget["total_context_token_count"] > 40
            ),
            "sealed_prompt_id_reuse_rejected": (
                reused_id["id"] in prompt_ids
                and reused_id["sha256"] not in prompt_hashes
            ),
            "sealed_prompt_hash_reuse_rejected": (
                reused_hash["id"] not in prompt_ids
                and reused_hash["sha256"] in prompt_hashes
            ),
            "deleting_id_entry_fails_binding": (
                mutations["deleted_id"]["count"] != EXPECTED_DENY_ID_COUNT
                and mutations["deleted_id"]["digest"] != EXPECTED_DENY_ID_DIGEST
            ),
            "deleting_hash_entry_fails_binding": (
                mutations["deleted_hash"]["count"] != EXPECTED_DENY_HASH_COUNT
                and mutations["deleted_hash"]["digest"] != EXPECTED_DENY_HASH_DIGEST
            ),
            "adding_id_entry_fails_binding": (
                mutations["added_id"]["count"] != EXPECTED_DENY_ID_COUNT
                and mutations["added_id"]["digest"] != EXPECTED_DENY_ID_DIGEST
            ),
            "adding_hash_entry_fails_binding": (
                mutations["added_hash"]["count"] != EXPECTED_DENY_HASH_COUNT
                and mutations["added_hash"]["digest"] != EXPECTED_DENY_HASH_DIGEST
            ),
            "sealed_0023_bound_without_execution": (
                _verify_0023_stdlib_seal()["process_start_count"] == 0
            ),
            "process_not_started": True,
        }
        return {
            "schema_version": 2,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_EXACT_SEALED_DENY_SET_AND_CANDIDATE_BUDGET_REGRESSION",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "selection": runner.file_record(SELECTION),
            "sealed_0023": _verify_0023_stdlib_seal(),
            "deny_set_binding": {
                "combined_id_count": EXPECTED_DENY_ID_COUNT,
                "combined_hash_count": EXPECTED_DENY_HASH_COUNT,
                "combined_id_digest": EXPECTED_DENY_ID_DIGEST,
                "combined_hash_digest": EXPECTED_DENY_HASH_DIGEST,
            },
            "selected": selected,
            "negative_probes": [over_budget, reused_id, reused_hash],
            "mutation_probes": mutations,
            "checks": checks,
        }

    def verify_candidate_budget_regression() -> dict[str, Any]:
        runner.require(
            stat.S_ISREG(os.lstat(CANDIDATE_BUDGET_REGRESSION).st_mode),
            "0024 candidate-budget regression is absent or not regular",
        )
        record = runner.read_json(CANDIDATE_BUDGET_REGRESSION)
        runner.require(
            record == expected_candidate_budget_regression()
            and all(record["checks"].values()),
            "0024 candidate-budget regression changed",
        )
        return record

    def candidate_budget_regression() -> int:
        runner.verify_exact_interpreter()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        runner.require(
            PROMPTS.is_file() and SELECTION.is_file(),
            "candidate-budget gate requires prompt selection first",
        )
        runner.require(
            not CANDIDATE_BUDGET_REGRESSION.exists(),
            "0024 candidate-budget regression already exists",
        )
        record = expected_candidate_budget_regression()
        runner.require(
            all(record["checks"].values()),
            "0024 candidate-budget regression failed",
        )
        runner.write_json(CANDIDATE_BUDGET_REGRESSION, record)
        verify_candidate_budget_regression()
        print(
            "ACE2_HYBRID_0024_CANDIDATE_BUDGET_REGRESSION_PASS "
            "deny_ids=42 deny_hashes=52 deletion_addition_reuse_rejected=true "
            "selected=2 process_start_count=0",
            flush=True,
        )
        return 0

    runner.candidate_budget_regression = candidate_budget_regression
    runner.verify_candidate_budget_regression = verify_candidate_budget_regression

    sealed_0023_paths = (
        SEALED_0023_TERMINAL,
        SEALED_0023_LAUNCHER,
        SEALED_0023_PROMPTS,
        SEALED_0023_SELECTION,
    )

    def source_records_0024(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        for path in sealed_0023_paths:
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0024

    def execution_bound_paths_0024(
        prompts_path: Path,
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        *,
        include_frozen_stage_files: bool,
    ) -> list[Path]:
        return sorted(
            {
                *prior_execution_bound_paths(
                    prompts_path,
                    model,
                    tokenizer_json,
                    tokenizer_config,
                    include_frozen_stage_files=include_frozen_stage_files,
                ),
                *(path.resolve(strict=True) for path in sealed_0023_paths),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0024

    def freeze_0024(prompts_path: Path) -> int:
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["sealed_0023"] = {
            "terminal": runner.file_record(SEALED_0023_TERMINAL),
            "launcher": runner.file_record(SEALED_0023_LAUNCHER),
            "private_prompts": runner.file_record(
                SEALED_0023_PROMPTS, expose_path=False
            ),
            "selection": runner.file_record(SEALED_0023_SELECTION),
            "process_start_count": 0,
            "process_started": False,
            "status": "FAILED_SEALED_NO_EXECUTION",
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0024

    def execute_0024(prompts_path: Path, expected_freeze_sha256: str) -> int:
        return template._execute_attested_body(
            current_child_entry_attestation,
            model_rtl_execute,
            prompts_path,
            expected_freeze_sha256,
        )

    runner.execute = execute_0024

    def verify_0024() -> int:
        result = prior_verify()
        terminal = runner.read_json(runner.TERMINAL)
        execution_result = runner.read_json(runner.RESULT)
        candidate_regression = verify_candidate_budget_regression()
        orchestration_regression = runner.read_json(ORCHESTRATION_REGRESSION)
        attestation = runner.read_json(
            runner.STATE / "child_entry_attestation.json"
        )
        runner.require(
            terminal.get("outcome") == "success"
            and terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("provenance", {})
            .get("package", {})
            .get("package_id")
            == "stage1rank1hybrid24execute-nonofficial-hybrid-0024"
            and attestation.get("attempt_identity") == "nonofficial-hybrid-0024"
            and attestation.get("mission_id") == "stage1rank1hybrid24execute"
            and orchestration_regression.get("status")
            == "PASS_EXTERNAL_EXACT_ONCE_ORCHESTRATION_REGRESSION"
            and all(orchestration_regression.get("checks", {}).values())
            and candidate_regression["deny_set_binding"]
            == {
                "combined_id_count": EXPECTED_DENY_ID_COUNT,
                "combined_hash_count": EXPECTED_DENY_HASH_COUNT,
                "combined_id_digest": EXPECTED_DENY_ID_DIGEST,
                "combined_hash_digest": EXPECTED_DENY_HASH_DIGEST,
            }
            and execution_result["aggregate"]["prompt_count"] == 2
            and execution_result["aggregate"]["generated_tokens_per_prompt"] == 4,
            "0024 terminal or exact deny-set binding changed",
        )
        print(
            "ACE2_HYBRID_0024_DECISIVE_VERIFY_PASS "
            "prompt_count=2 generated_tokens_per_prompt=4 rtl_v_bytes=1024 "
            "deny_ids=42 deny_hashes=52 process_start_count=1 "
            "external_supervisor=true reconstructed_process_argv=true "
            "phase_safe_binding=true",
            flush=True,
        )
        return result

    runner.verify = verify_0024


def _patch_template(template: Any) -> None:
    for name, value in {
        "LAUNCHER": LAUNCHER,
        "OUTPUT": OUTPUT,
        "TERMINAL": TERMINAL,
        "PRIVATE": PRIVATE,
        "PROMPTS": PROMPTS,
        "SELECTION": SELECTION,
        "CANDIDATE_BUDGET_REGRESSION": CANDIDATE_BUDGET_REGRESSION,
        "TOKENIZATION_PREFLIGHT": TOKENIZATION_PREFLIGHT,
        "TOKENIZATION_REGRESSION": TOKENIZATION_REGRESSION,
        "INPUT_SCALE_REGRESSION": INPUT_SCALE_REGRESSION,
        "SCALE_REGRESSION": SCALE_REGRESSION,
        "LIFECYCLE_REGRESSION": LIFECYCLE_REGRESSION,
        "GATE_ORDER_REGRESSION": GATE_ORDER_REGRESSION,
        "TEMPLATE_GATE": TEMPLATE_GATE,
        "DEPENDENCY_PROBE": DEPENDENCY_PROBE,
        "LAUNCH_REGRESSION": LAUNCH_REGRESSION,
        "ARGV_SHAPE_REGRESSION": ARGV_SHAPE_REGRESSION,
        "CLOSURE_CERTIFICATE": CLOSURE_CERTIFICATE,
        "PHASE_TRANSITION_REGRESSION": PHASE_TRANSITION_REGRESSION,
        "INVOCATION_REGRESSION": INVOCATION_REGRESSION,
        "PRE_IMPORT_ENVIRONMENT": PRE_IMPORT_ENVIRONMENT,
        "PRE_IMPORT_SYS_PATH": PRE_IMPORT_SYS_PATH,
        "PRE_IMPORT_SYS_PREFIX": PRE_IMPORT_SYS_PREFIX,
        "PRE_IMPORT_ARGV": PRE_IMPORT_ARGV,
        "PRE_IMPORT_CWD": PRE_IMPORT_CWD,
        "PRE_IMPORT_MODULES": PRE_IMPORT_MODULES,
        "FROZEN_EXPECTED_ENVIRONMENT": FROZEN_EXPECTED_ENVIRONMENT,
        "EXPECTED_PYTHON": EXPECTED_PYTHON,
        "CANDIDATES": CANDIDATES,
        "EXPECTED_SELECTED_CANDIDATES": EXPECTED_SELECTED_CANDIDATES,
        "OVER_BUDGET_CANDIDATE": OVER_BUDGET_CANDIDATE,
        "REUSED_CANDIDATE": REUSED_ID_CANDIDATE,
        "_verify_invocation_regression": _verify_invocation_regression,
        "_seal_preloader_failure": _seal_preloader_failure,
    }.items():
        setattr(template, name, value)

    original_patch_inherited_chain = template._patch_inherited_chain
    original_configure_runner = template._configure_runner
    configured: dict[str, Any] = {}

    def patch_inherited_chain(prior: Any, prior_0014: Any) -> None:
        original_patch_inherited_chain(prior, prior_0014)
        prior_0014.DIAGNOSTIC_PROMPT_SOURCES = tuple(
            dict.fromkeys(
                (
                    *prior_0014.DIAGNOSTIC_PROMPT_SOURCES,
                    PRIVATE / "prompts-0022.json",
                    SEALED_0023_PROMPTS,
                )
            )
        )
        prior_0014.SEALED_BOUNDARY_SHA256 = {
            **prior_0014.SEALED_BOUNDARY_SHA256,
            str(SEALED_0023_TERMINAL.relative_to(ROOT)): (
                SEALED_0023_TERMINAL_SHA256
            ),
            str(SEALED_0023_LAUNCHER.relative_to(ROOT)): (
                SEALED_0023_LAUNCHER_SHA256
            ),
            str(SEALED_0023_PROMPTS.relative_to(ROOT)): SEALED_0023_PROMPTS_SHA256,
            str(SEALED_0023_SELECTION.relative_to(ROOT)): (
                SEALED_0023_SELECTION_SHA256
            ),
        }
        configured["prior_0014"] = prior_0014

    def configure_runner(
        runner: Any,
        prior: Any,
        predecessor_modules: tuple[Any, ...],
        model_rtl_execute: Any,
    ) -> None:
        _require("prior_0014" in configured, "0024 inherited chain was not configured")
        original_configure_runner(
            runner,
            prior,
            (*predecessor_modules, template),
            model_rtl_execute,
        )
        _configure_0024(
            template,
            runner,
            configured["prior_0014"],
            model_rtl_execute,
        )

    template._patch_inherited_chain = patch_inherited_chain
    template._configure_runner = configure_runner


def main() -> int:
    try:
        if "--execute" in PRE_IMPORT_ARGV[1:]:
            _require(
                _reservation_allows_child(),
                "direct --execute is forbidden without external supervisor reservation",
            )
        if "--check-invocation-fidelity" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-invocation-fidelity", action="store_true")
            parser.add_argument(
                "--inherited-environment-snapshot",
                type=Path,
                required=True,
            )
            arguments = parser.parse_args(PRE_IMPORT_ARGV[1:])
            return _run_invocation_regression(
                arguments.inherited_environment_snapshot.resolve(strict=True)
            )
        if "--check-orchestration" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-orchestration", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return _run_orchestration_regression()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid_0022 as template

        _patch_template(template)
        return template.main()
    except BaseException as error:
        _seal_preloader_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
