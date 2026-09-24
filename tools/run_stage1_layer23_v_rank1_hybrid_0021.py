#!/usr/bin/env python3
"""Phase-safe fixed-input-scale launcher for nonofficial hybrid attempt 0021."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
OUTPUT = (
    ROOT
    / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1"
    / "nonofficial-hybrid-0021"
)
TERMINAL = OUTPUT / "terminal-record.json"
PRIVATE = ROOT / "build/stage1-layer23-v-rank1-hybrid-v1/private"
PROMPTS = PRIVATE / "prompts-0021.json"
SELECTION = PRIVATE / "candidate-selection-0021.json"
CANDIDATE_BUDGET_REGRESSION = PRIVATE / "candidate-budget-regression-0021.json"
TOKENIZATION_PREFLIGHT = PRIVATE / "exact-tokenization-preflight-0021.json"
TOKENIZATION_REGRESSION = PRIVATE / "tokenization-verifier-regression-0021.json"
INPUT_SCALE_REGRESSION = PRIVATE / "fixed-input-scale-regression-0021.json"
SCALE_REGRESSION = PRIVATE / "retained-scale-regression-0021.json"
LIFECYCLE_REGRESSION = PRIVATE / "record-lifecycle-regression-0021.json"
GATE_ORDER_REGRESSION = PRIVATE / "gate-order-regression-0021.json"
TEMPLATE_GATE = PRIVATE / "package-template-invariant-0021.json"
DEPENDENCY_PROBE = PRIVATE / "dependency-probe-0021.json"
LAUNCH_REGRESSION = PRIVATE / "launch-fidelity-regression-0021.json"
ARGV_SHAPE_REGRESSION = PRIVATE / "argv-shape-regression-0021.json"
CLOSURE_CERTIFICATE = PRIVATE / "closure-certificate-0021.json"
PHASE_TRANSITION_REGRESSION = PRIVATE / "phase-transition-regression-0021.json"
INVOCATION_REGRESSION = PRIVATE / "invocation-fidelity-regression-0021.json"

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

SEALED_0015_TERMINAL_SHA256 = (
    "6b2c3e97460f37dafe41e7f0bf0c02c937be7326642074f3de9b70433fbd2186"
)
SEALED_0015_LAUNCHER_SHA256 = (
    "163e168bfc9c8f8aff111dfe412a4aa726fcbe74bdc4446bed61211772c307be"
)
SEALED_0016_TERMINAL_SHA256 = (
    "40940183e82ddc86aa74421db4d509d2f4ea2295c05953a41e5ce160ebbaecfe"
)
SEALED_0016_LAUNCHER_SHA256 = (
    "f5155317dd15e61f55bba495757ffdeee3b54ffa9cad15c0554f11a40e7d8aa4"
)
SEALED_0017_TERMINAL_SHA256 = (
    "2761f66b627358a97bd344cc94bc79806613bcdf290098149683835eeaf05ba6"
)
SEALED_0017_LAUNCHER_SHA256 = (
    "594d10c4280251a0d01150b978951fc18b3c52c1f08a6849128130b3b35f5f94"
)
SEALED_0017_CANONICAL_ARGV_SHA256 = (
    "1cd750c1003a2f1d2709b5ebd976aed47f002936c0e088ab39206877d4f1bcc7"
)
SEALED_0018_TERMINAL_SHA256 = (
    "8353f11d8352ca053d7659d8fd34bc43e77b5306793f198489b31055f929ef65"
)
SEALED_0018_LAUNCHER_SHA256 = (
    "ba254307b900272fe977cfcf781f9a2faae4e99da2e9dee021e5e66cb3fe310d"
)
SEALED_0019_TERMINAL_SHA256 = (
    "b541616e771744c8f5e702b538fe33fbbb912ce43b2cd32636d9921aa2734f78"
)
SEALED_0019_LAUNCHER_SHA256 = (
    "e2f04199f533d4317d63b19f2b944204c84701c025346ae17acf100276465de1"
)
SEALED_0020_TERMINAL_SHA256 = (
    "56c20d29060fddd58ed1f0c3bf78b1849c4f0084836f581fd884df502180b424"
)
SEALED_0020_LAUNCHER_SHA256 = (
    "345e8526c69c02aef835d1dd94800fe85954dccc1689ddc622ea1ae21162a2ba"
)

CANDIDATES = (
    {"id": "fresh-0021-a", "text": "Explain rain."},
    {"id": "fresh-0021-b", "text": "Describe moss."},
)
EXPECTED_SELECTED_CANDIDATES = (
    {
        **CANDIDATES[0],
        "sha256": "c79c6db097b4fad5030410e7eda962409d2723d74fdd6e08173f85f88018c801",
        "prompt_token_count": 33,
        "total_context_token_count": 37,
    },
    {
        **CANDIDATES[1],
        "sha256": "49337aca62849440b2b413f2a63d15c88385dfe5057497b00325e50d89c560dc",
        "prompt_token_count": 32,
        "total_context_token_count": 36,
    },
)
OVER_BUDGET_CANDIDATE = {
    "id": "fresh-0021-over-budget",
    "text": (
        "Explain how rainfall, rivers, oceans, clouds, winds, mountains, forests, "
        "farms, cities, and seasons shape a region over many years."
    ),
}
REUSED_CANDIDATE = {
    "id": "fresh-0014-a",
    "text": "What makes dew form?",
}
EXPECTED_NEGATIVE_PROBES = (
    {
        **OVER_BUDGET_CANDIDATE,
        "sha256": "e0db100821c738ed02178e0c5885f426ff938432b0caa4556eaed59b65dc2632",
        "prompt_token_count": 59,
        "total_context_token_count": 63,
        "rejection": "prompt_plus_four_exceeds_40",
    },
    {
        **REUSED_CANDIDATE,
        "sha256": "ba5d30e99769fcf8e515063cac6eeaa49cc0a38880a000aa5550109b066f8acd",
        "prompt_token_count": 34,
        "total_context_token_count": 38,
        "rejection": "sealed_prompt_id_and_hash_reused",
    },
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _actual_process_argv(pre_import_argv: list[str]) -> list[str]:
    return [str(Path(sys.executable).resolve()), *pre_import_argv]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preloader_require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_environment_snapshot(path: Path) -> dict[str, str]:
    _preloader_require(
        stat.S_ISREG(os.lstat(path).st_mode),
        "ordinary inherited environment snapshot is not a regular file",
    )
    environment: dict[str, str] = {}
    for entry in path.read_bytes().split(b"\0"):
        if not entry:
            continue
        name, separator, value = entry.partition(b"=")
        _preloader_require(
            bool(separator) and bool(name),
            "ordinary inherited environment snapshot contains a malformed entry",
        )
        decoded_name = os.fsdecode(name)
        _preloader_require(
            decoded_name not in environment,
            "ordinary inherited environment snapshot contains a duplicate key",
        )
        environment[decoded_name] = os.fsdecode(value)
    return environment


def _environment_accepted(environment: dict[str, str]) -> bool:
    return environment == FROZEN_EXPECTED_ENVIRONMENT


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
        "exact_env_i_snapshot_equals_frozen_map": (
            PRE_IMPORT_ENVIRONMENT == expected
        ),
        "exact_env_i_snapshot_accepted": _environment_accepted(
            PRE_IMPORT_ENVIRONMENT
        ),
        "extra_variable_rejected": not _environment_accepted(extra_environment),
        "missing_variable_rejected": not _environment_accepted(missing_environment),
        "changed_variable_rejected": not _environment_accepted(changed_environment),
        "no_project_or_dependency_module_imported_before_snapshot": (
            not forbidden_present
        ),
        "process_not_started": True,
    }
    return {
        "schema_version": 1,
        "mission_id": "stage1rank1hybrid21execute",
        "attempt_identity": "nonofficial-hybrid-0021",
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
        "checks": checks,
    }


def _verify_invocation_regression() -> dict[str, Any]:
    _preloader_require(
        stat.S_ISREG(os.lstat(INVOCATION_REGRESSION).st_mode),
        "0021 invocation-fidelity regression is absent or not regular",
    )
    record = json.loads(INVOCATION_REGRESSION.read_text(encoding="utf-8"))
    ordinary = record.get("ordinary_inherited_environment", {})
    ordinary_differs = bool(
        ordinary.get("extra_key_count")
        or ordinary.get("missing_expected_keys")
        or ordinary.get("changed_expected_keys")
    )
    _preloader_require(
        record.get("schema_version") == 1
        and record.get("mission_id") == "stage1rank1hybrid21execute"
        and record.get("attempt_identity") == "nonofficial-hybrid-0021"
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
        and record.get("rejection_cases")
        == {
            "extra": {"accepted": False},
            "missing": {"accepted": False},
            "changed": {"accepted": False},
        }
        and record.get("pre_import_snapshot", {}).get(
            "forbidden_modules_present"
        )
        == []
        and record.get("checks")
        and all(record["checks"].values()),
        "0021 invocation-fidelity regression changed or is incomplete",
    )
    return record


def _verify_0020_stdlib_seal() -> None:
    terminal = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0020/terminal-record.json"
    )
    launcher = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0020.py"
    for path in (terminal, launcher):
        _preloader_require(
            stat.S_ISREG(os.lstat(path).st_mode),
            f"sealed 0020 artifact is not regular: {path}",
        )
    terminal_record = json.loads(terminal.read_text(encoding="utf-8"))
    _preloader_require(
        _sha256_file(terminal) == SEALED_0020_TERMINAL_SHA256
        and _sha256_file(launcher) == SEALED_0020_LAUNCHER_SHA256
        and terminal_record.get("status") == "FAILED_SEALED_NO_EXECUTION"
        and terminal_record.get("outcome") == "pre_execution_failure"
        and terminal_record.get("process_start_count") == 0
        and terminal_record.get("process_started") is False,
        "sealed 0020 terminal or launcher changed",
    )


def _run_invocation_regression(inherited_snapshot: Path) -> int:
    _preloader_require(
        Path(sys.executable).resolve() == EXPECTED_PYTHON,
        "Python interpreter binding changed",
    )
    _preloader_require(
        PRE_IMPORT_ENVIRONMENT == FROZEN_EXPECTED_ENVIRONMENT,
        "invocation-fidelity gate was not launched with the exact env-i map",
    )
    _preloader_require(
        PRE_IMPORT_CWD == str(ROOT),
        "invocation-fidelity gate cwd changed",
    )
    _preloader_require(
        Path(PRE_IMPORT_ARGV[0]).resolve() == LAUNCHER,
        "invocation-fidelity gate launcher changed",
    )
    _preloader_require(
        not OUTPUT.exists() and not INVOCATION_REGRESSION.exists(),
        "0021 invocation or execution namespace already exists",
    )
    _verify_0020_stdlib_seal()
    record = _invocation_regression_record(
        _read_environment_snapshot(inherited_snapshot)
    )
    _preloader_require(
        all(record["checks"].values()),
        "0021 invocation-fidelity regression failed",
    )
    INVOCATION_REGRESSION.parent.mkdir(parents=True, exist_ok=True)
    with INVOCATION_REGRESSION.open("xb") as stream:
        stream.write(_canonical_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    _verify_invocation_regression()
    print(
        "ACE2_HYBRID_0021_INVOCATION_FIDELITY_REGRESSION_PASS "
        "exact_env_i=true inherited_rejected=true mutations_rejected=3 "
        "pre_import_project_modules=0 process_start_count=0",
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
        "mission_id": "stage1rank1hybrid21execute",
        "attempt_identity": "nonofficial-hybrid-0021",
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
            "regression": "No retry, replay, or resume; preserve all 0021 artifacts.",
        },
    }
    try:
        with TERMINAL.open("xb") as stream:
            stream.write(_canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return


def _patch_inherited_chain(prior: Any, prior_0014: Any) -> None:
    for name, value in {
        "LAUNCHER": LAUNCHER,
        "OUTPUT": OUTPUT,
        "TERMINAL": TERMINAL,
        "PRIVATE": PRIVATE,
        "PROMPTS": PROMPTS,
        "SELECTION": SELECTION,
        "TOKENIZATION_PREFLIGHT": TOKENIZATION_PREFLIGHT,
        "TOKENIZATION_REGRESSION": TOKENIZATION_REGRESSION,
        "INPUT_SCALE_REGRESSION": INPUT_SCALE_REGRESSION,
        "SCALE_REGRESSION": SCALE_REGRESSION,
        "LIFECYCLE_REGRESSION": LIFECYCLE_REGRESSION,
        "GATE_ORDER_REGRESSION": GATE_ORDER_REGRESSION,
        "TEMPLATE_GATE": TEMPLATE_GATE,
        "DEPENDENCY_PROBE": DEPENDENCY_PROBE,
        "LAUNCH_REGRESSION": LAUNCH_REGRESSION,
        "CANDIDATES": CANDIDATES,
        "PRE_IMPORT_ENVIRONMENT": PRE_IMPORT_ENVIRONMENT,
        "PRE_IMPORT_SYS_PATH": PRE_IMPORT_SYS_PATH,
        "PRE_IMPORT_SYS_PREFIX": PRE_IMPORT_SYS_PREFIX,
        "PRE_IMPORT_ARGV": PRE_IMPORT_ARGV,
        "PRE_IMPORT_CWD": PRE_IMPORT_CWD,
        "PRE_IMPORT_MODULES": PRE_IMPORT_MODULES,
    }.items():
        setattr(prior, name, value)
    prior._configure_0014(prior_0014)
    prior_0014.DIAGNOSTIC_PROMPT_SOURCES = (
        *prior_0014.DIAGNOSTIC_PROMPT_SOURCES,
        PRIVATE / "prompts-0015.json",
        PRIVATE / "prompts-0016.json",
        PRIVATE / "prompts-0017.json",
        PRIVATE / "prompts-0018.json",
    )
    prior_0014.SEALED_BOUNDARY_SHA256 = {
        **prior_0014.SEALED_BOUNDARY_SHA256,
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0015/terminal-record.json"
        ): SEALED_0015_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0015.py": (
            SEALED_0015_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0016/terminal-record.json"
        ): SEALED_0016_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0016.py": (
            SEALED_0016_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0017/terminal-record.json"
        ): SEALED_0017_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0017.py": (
            SEALED_0017_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0018/terminal-record.json"
        ): SEALED_0018_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0018.py": (
            SEALED_0018_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0019/terminal-record.json"
        ): SEALED_0019_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0019.py": (
            SEALED_0019_LAUNCHER_SHA256
        ),
        (
            "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
            "nonofficial-hybrid-0020/terminal-record.json"
        ): SEALED_0020_TERMINAL_SHA256,
        "tools/run_stage1_layer23_v_rank1_hybrid_0020.py": (
            SEALED_0020_LAUNCHER_SHA256
        ),
    }


def _configure_runner(
    runner: Any,
    prior: Any,
    predecessor_modules: tuple[Any, ...],
) -> None:
    runner.MISSION_ID = "stage1rank1hybrid21execute"
    runner.ATTEMPT_ID = "nonofficial-hybrid-0021"
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
        "nonofficial-hybrid-0015": {
            "sha256": SEALED_0015_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0016": {
            "sha256": SEALED_0016_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0017": {
            "sha256": SEALED_0017_TERMINAL_SHA256,
            "process_start_count": 1,
            "process_started": True,
        },
        "nonofficial-hybrid-0018": {
            "sha256": SEALED_0018_TERMINAL_SHA256,
            "process_start_count": 0,
            "process_started": False,
        },
        "nonofficial-hybrid-0019": {
            "sha256": SEALED_0019_TERMINAL_SHA256,
            "process_start_count": 0,
            "process_started": False,
        },
        "nonofficial-hybrid-0020": {
            "sha256": SEALED_0020_TERMINAL_SHA256,
            "process_start_count": 0,
            "process_started": False,
        },
    }
    runner.__file__ = str(LAUNCHER)

    prior_verify_predecessor_seals = runner.verify_predecessor_seals
    prior_source_records = runner.source_records
    prior_execution_bound_paths = runner.execution_bound_paths
    prior_package_bindings = runner.package_bindings
    prior_package_template = runner.package_template_invariant
    prior_preflight = runner.preflight
    prior_freeze = runner.freeze
    prior_package = runner.package_execution
    prior_authority = runner.grant_authority
    prior_execute = runner.execute
    prior_verify = runner.verify
    prior_verify_exact_interpreter = runner.verify_exact_interpreter

    terminal_0017 = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0017/terminal-record.json"
    )
    durable_0017 = terminal_0017.parent / "supervisor-state/terminal_record.json"
    launcher_0017 = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0017.py"
    package_0017 = terminal_0017.parent / "execution-package.json"
    authority_0017 = terminal_0017.parent / "execution-authority.json"
    intent_0017 = terminal_0017.parent / "supervisor-state/pre_start_intent.json"
    started_0017 = terminal_0017.parent / "supervisor-state/process_started.json"
    terminal_0018 = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0018/terminal-record.json"
    )
    launcher_0018 = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0018.py"
    terminal_0019 = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0019/terminal-record.json"
    )
    launcher_0019 = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0019.py"
    terminal_0020 = (
        ROOT
        / "evidence/verification/stage1-layer23-v-rank1-hybrid-v1/"
        "nonofficial-hybrid-0020/terminal-record.json"
    )
    launcher_0020 = ROOT / "tools/run_stage1_layer23_v_rank1_hybrid_0020.py"

    def verify_0017_seal() -> dict[str, Any]:
        sealed_paths = (
            terminal_0017,
            durable_0017,
            launcher_0017,
            package_0017,
            authority_0017,
            intent_0017,
            started_0017,
        )
        for path in sealed_paths:
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0017 artifact is not regular: {runner.public_path(path)}",
            )
        terminal = runner.read_json(terminal_0017)
        package = runner.read_json(package_0017)
        authority = runner.read_json(authority_0017)
        intent = runner.read_json(intent_0017)
        started = runner.read_json(started_0017)
        argv = terminal["process"]["argv"]
        argv_sha256 = runner.supervisor.canonical_value_sha256(argv)
        package_measurement = runner.supervisor.measure_file(package_0017)
        authority_measurement = runner.supervisor.measure_file(authority_0017)
        expected_package_provenance = {
            "path": str(package_0017),
            "byte_count": package_measurement.byte_count,
            "sha256": package_measurement.sha256,
            "package_id": package["package_id"],
        }
        expected_authority_provenance = {
            "path": str(authority_0017),
            "byte_count": authority_measurement.byte_count,
            "sha256": authority_measurement.sha256,
            "authority_id": authority["authority_id"],
        }
        runner.require(
            runner.sha256_file(terminal_0017) == SEALED_0017_TERMINAL_SHA256
            and runner.sha256_file(durable_0017) == SEALED_0017_TERMINAL_SHA256
            and runner.sha256_file(launcher_0017) == SEALED_0017_LAUNCHER_SHA256
            and terminal == runner.read_json(durable_0017)
            and terminal.get("process_start_count") == 1
            and terminal.get("process_started") is True
            and terminal.get("outcome") == "child_nonzero_exit",
            "sealed 0017 terminal or launcher changed",
        )
        runner.require(
            len(argv) == 7
            and argv_sha256 == SEALED_0017_CANONICAL_ARGV_SHA256
            and terminal["process"]["argv_sha256"] == argv_sha256
            and package["bindings"]["argv"] == argv
            and authority["bindings"]["argv"] == argv
            and intent["bindings"]["argv"] == argv
            and package["bindings"]["argv_sha256"] == argv_sha256
            and authority["bindings"]["argv_sha256"] == argv_sha256
            and intent["bindings"]["argv_sha256"] == argv_sha256,
            "sealed 0017 package/authority/supervisor argv changed",
        )
        runner.require(
            authority["package"] == expected_package_provenance
            and intent["provenance"]["package"] == expected_package_provenance
            and terminal["provenance"]["package"] == expected_package_provenance
            and intent["provenance"]["authority"] == expected_authority_provenance
            and started["provenance"] == intent["provenance"]
            and terminal["provenance"] == intent["provenance"]
            and started["process_start_count"] == 1,
            "sealed 0017 provenance chain changed",
        )
        return {
            "terminal": runner.file_record(terminal_0017),
            "launcher": runner.file_record(launcher_0017),
            "package": runner.file_record(package_0017),
            "authority": runner.file_record(authority_0017),
            "pre_start_intent": runner.file_record(intent_0017),
            "process_started": runner.file_record(started_0017),
            "canonical_argv_sha256": argv_sha256,
        }

    def verify_0018_seal() -> dict[str, Any]:
        for path in (terminal_0018, launcher_0018):
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0018 artifact is not regular: {runner.public_path(path)}",
            )
        terminal = runner.read_json(terminal_0018)
        runner.require(
            runner.sha256_file(terminal_0018) == SEALED_0018_TERMINAL_SHA256
            and runner.sha256_file(launcher_0018) == SEALED_0018_LAUNCHER_SHA256
            and terminal.get("status") == "FAILED_SEALED_NO_EXECUTION"
            and terminal.get("outcome") == "pre_execution_failure"
            and terminal.get("process_start_count") == 0
            and terminal.get("process_started") is False
            and terminal.get("failure", {}).get("phase") == "check_package_template",
            "sealed 0018 terminal or launcher changed",
        )
        return {
            "terminal": runner.file_record(terminal_0018),
            "launcher": runner.file_record(launcher_0018),
        }

    def verify_0019_seal() -> dict[str, Any]:
        for path in (terminal_0019, launcher_0019):
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0019 artifact is not regular: {runner.public_path(path)}",
            )
        terminal = runner.read_json(terminal_0019)
        runner.require(
            runner.sha256_file(terminal_0019) == SEALED_0019_TERMINAL_SHA256
            and runner.sha256_file(launcher_0019) == SEALED_0019_LAUNCHER_SHA256
            and terminal.get("status") == "FAILED_SEALED_NO_EXECUTION"
            and terminal.get("outcome") == "pre_execution_failure"
            and terminal.get("process_start_count") == 0
            and terminal.get("process_started") is False
            and terminal.get("failure", {}).get("phase")
            == "stdlib_preloader_or_nonconsuming_gate",
            "sealed 0019 terminal or launcher changed",
        )
        return {
            "terminal": runner.file_record(terminal_0019),
            "launcher": runner.file_record(launcher_0019),
        }

    def verify_0020_seal() -> dict[str, Any]:
        for path in (terminal_0020, launcher_0020):
            runner.require(
                stat.S_ISREG(os.lstat(path).st_mode),
                f"sealed 0020 artifact is not regular: {runner.public_path(path)}",
            )
        terminal = runner.read_json(terminal_0020)
        runner.require(
            runner.sha256_file(terminal_0020) == SEALED_0020_TERMINAL_SHA256
            and runner.sha256_file(launcher_0020) == SEALED_0020_LAUNCHER_SHA256
            and terminal.get("status") == "FAILED_SEALED_NO_EXECUTION"
            and terminal.get("outcome") == "pre_execution_failure"
            and terminal.get("process_start_count") == 0
            and terminal.get("process_started") is False
            and terminal.get("failure", {}).get("phase")
            == "stdlib_preloader_or_nonconsuming_gate",
            "sealed 0020 terminal or launcher changed",
        )
        return {
            "terminal": runner.file_record(terminal_0020),
            "launcher": runner.file_record(launcher_0020),
        }

    def verify_predecessor_seals_0021() -> None:
        prior_verify_predecessor_seals()
        verify_0017_seal()
        verify_0018_seal()
        verify_0019_seal()
        verify_0020_seal()

    runner.verify_predecessor_seals = verify_predecessor_seals_0021

    def verify_exact_interpreter_0021() -> None:
        prior_verify_exact_interpreter()
        _verify_invocation_regression()

    runner.verify_exact_interpreter = verify_exact_interpreter_0021

    def module_sources() -> list[dict[str, Any]]:
        return [
            {
                "module": module.__name__,
                "source": runner.file_record(Path(module.__file__).resolve(strict=True)),
            }
            for module in predecessor_modules
        ]

    def expected_candidate_budget_regression() -> dict[str, Any]:
        selection = runner.verify_selection_certificate()
        snapshot, _model, _tokenizer_json, _tokenizer_config = (
            runner.resolve_runtime_inputs()
        )
        tokenizer = runner.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )

        def probe(candidate: dict[str, str], rejection: str) -> dict[str, Any]:
            prompt_hash = runner.sha256_bytes(candidate["text"].encode("utf-8"))
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

        probes = (
            probe(OVER_BUDGET_CANDIDATE, "prompt_plus_four_exceeds_40"),
            probe(REUSED_CANDIDATE, "sealed_prompt_id_and_hash_reused"),
        )
        reused_prompts = runner.load_prompts(PRIVATE / "prompts-0014.json")
        deny_set = selection["deny_set"]
        selected = selection["selected"]
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
                deny_set["combined_id_count"] == 36
                and deny_set["combined_hash_count"] == 46
            ),
            "over_budget_count_and_hash_match_external_screen": (
                probes[0] == EXPECTED_NEGATIVE_PROBES[0]
                and probes[0]["total_context_token_count"] > 40
            ),
            "reused_count_and_hash_match_external_screen": (
                probes[1] == EXPECTED_NEGATIVE_PROBES[1]
                and {
                    "id": probes[1]["id"],
                    "text": probes[1]["text"],
                }
                in reused_prompts
            ),
            "over_budget_candidate_rejected": (
                probes[0]["total_context_token_count"] > 40
            ),
            "reused_candidate_rejected": (
                probes[1]["id"] in {item["id"] for item in reused_prompts}
                and probes[1]["sha256"]
                in {
                    runner.sha256_bytes(item["text"].encode("utf-8"))
                    for item in reused_prompts
                }
            ),
            "process_not_started": True,
        }
        return {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_FOCUSED_CANDIDATE_BUDGET_REGRESSION",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "selection": runner.file_record(SELECTION),
            "selected": selected,
            "negative_probes": list(probes),
            "checks": checks,
        }

    def verify_candidate_budget_regression() -> dict[str, Any]:
        runner.require(
            stat.S_ISREG(os.lstat(CANDIDATE_BUDGET_REGRESSION).st_mode),
            "0021 candidate-budget regression is absent or not regular",
        )
        record = runner.read_json(CANDIDATE_BUDGET_REGRESSION)
        runner.require(
            record == expected_candidate_budget_regression()
            and all(record["checks"].values()),
            "0021 candidate-budget regression changed",
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
            "0021 candidate-budget regression already exists",
        )
        record = expected_candidate_budget_regression()
        runner.require(
            all(record["checks"].values()),
            "0021 candidate-budget regression failed",
        )
        runner.write_json(CANDIDATE_BUDGET_REGRESSION, record)
        verify_candidate_budget_regression()
        print(
            "ACE2_HYBRID_0021_CANDIDATE_BUDGET_REGRESSION_PASS "
            "selected=2 rejected_over_budget=1 rejected_reused=1 "
            "process_start_count=0",
            flush=True,
        )
        return 0

    runner.candidate_budget_regression = candidate_budget_regression
    runner.verify_candidate_budget_regression = verify_candidate_budget_regression

    def expected_argv_shape_regression() -> dict[str, Any]:
        pre_import_argv = [
            str(LAUNCHER),
            "--execute",
            "--prompts",
            str(PROMPTS.resolve()),
            "--freeze-sha256",
            "0" * 64,
        ]
        process_argv = [
            str(Path(sys.executable).resolve()),
            *pre_import_argv,
        ]
        reconstructed = _actual_process_argv(pre_import_argv)
        tampered_interpreter = [str(ROOT / "tampered-python"), *pre_import_argv]
        tampered_script = [
            process_argv[0],
            str(ROOT / "tools/tampered-launcher.py"),
            *pre_import_argv[1:],
        ]
        tampered_argument = [*process_argv[:-1], "f" * 64]
        process_argv_sha256 = runner.supervisor.canonical_value_sha256(process_argv)

        def tamper_fails(candidate: list[str]) -> bool:
            return (
                candidate != process_argv
                and runner.supervisor.canonical_value_sha256(candidate)
                != process_argv_sha256
            )

        return {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_RECONSTRUCTED_PROCESS_ARGV_SHAPE_AND_TAMPER_REJECTION",
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "legacy_pre_import_argv_length": len(pre_import_argv),
            "supervisor_process_argv_length": len(process_argv),
            "process_argv_sha256": process_argv_sha256,
            "checks": {
                "legacy_six_vs_seven_direct_comparison_fails": (
                    len(pre_import_argv) == 6
                    and len(process_argv) == 7
                    and pre_import_argv != process_argv
                ),
                "resolved_interpreter_plus_pre_import_argv_passes": (
                    reconstructed == process_argv
                ),
                "interpreter_tampering_fails": tamper_fails(tampered_interpreter),
                "script_tampering_fails": tamper_fails(tampered_script),
                "argument_tampering_fails": tamper_fails(tampered_argument),
                "pre_import_script_resolves_to_exact_launcher": (
                    Path(pre_import_argv[0]).resolve() == LAUNCHER
                ),
                "process_not_started": True,
            },
        }

    def verify_argv_shape_regression() -> dict[str, Any]:
        runner.require(
            stat.S_ISREG(os.lstat(ARGV_SHAPE_REGRESSION).st_mode),
            "0021 argv-shape regression is absent or not regular",
        )
        record = runner.read_json(ARGV_SHAPE_REGRESSION)
        runner.require(
            record == expected_argv_shape_regression()
            and all(record["checks"].values()),
            "0021 argv-shape regression changed",
        )
        return record

    def argv_shape_regression() -> int:
        runner.verify_exact_interpreter()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        dependencies = (
            INVOCATION_REGRESSION,
            SELECTION,
            CANDIDATE_BUDGET_REGRESSION,
            LIFECYCLE_REGRESSION,
            DEPENDENCY_PROBE,
            GATE_ORDER_REGRESSION,
            INPUT_SCALE_REGRESSION,
            TOKENIZATION_PREFLIGHT,
            TOKENIZATION_REGRESSION,
            SCALE_REGRESSION,
        )
        runner.require(
            all(path.is_file() for path in dependencies),
            "argv-shape gate requires all dependency-first regressions",
        )
        runner.require(
            not ARGV_SHAPE_REGRESSION.exists(),
            "0021 argv-shape regression already exists",
        )
        runner.write_json(ARGV_SHAPE_REGRESSION, expected_argv_shape_regression())
        record = verify_argv_shape_regression()
        print(
            "ACE2_HYBRID_0021_ARGV_SHAPE_REGRESSION_PASS "
            f"argv_sha256={record['process_argv_sha256']} "
            "legacy_argv_length=6 process_argv_length=7 process_start_count=0",
            flush=True,
        )
        return 0

    runner.argv_shape_regression = argv_shape_regression
    runner.verify_argv_shape_regression = verify_argv_shape_regression

    def execution_envelope() -> dict[str, Any]:
        argv_template = [
            str(Path(sys.executable).resolve()),
            str(LAUNCHER),
            "--execute",
            "--prompts",
            str(PROMPTS.resolve(strict=True)),
            "--freeze-sha256",
            "<sha256-of-freeze.json>",
        ]
        modules = sorted(PRE_IMPORT_MODULES)
        return {
            "argv_template": argv_template,
            "argv_freeze_binding_policy": (
                "replace only the final placeholder with the current freeze.json SHA-256"
            ),
            "environment": PRE_IMPORT_ENVIRONMENT,
            "environment_sha256": runner.supervisor.canonical_value_sha256(
                PRE_IMPORT_ENVIRONMENT
            ),
            "cwd": str(ROOT),
            "pre_import_modules": modules,
            "pre_import_modules_sha256": runner.supervisor.canonical_value_sha256(
                modules
            ),
            "reservation": {
                "state_dir": str(runner.STATE),
                "pre_start_intent": str(runner.STATE / "pre_start_intent.json"),
                "process_started": str(runner.STATE / "process_started.json"),
                "terminal_record": str(runner.STATE / "terminal_record.json"),
                "process_start_count": 1,
            },
        }

    def binding_core() -> dict[str, Any]:
        return {
            "schema": "ace2-phase-safe-binding-core-v1",
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "claim_boundary": (
                "bounded nonofficial hybrid evidence only; not Stage-1 certification"
            ),
            "execution_envelope": execution_envelope(),
            "launcher": runner.file_record(LAUNCHER),
            "private_prompts": runner.file_record(PROMPTS, expose_path=False),
            "supervisor": runner.file_record(
                Path(runner.supervisor.__file__).resolve(strict=True)
            ),
            "predecessor_modules": module_sources(),
            "sealed_predecessor": verify_0020_seal(),
            "invocation_fidelity_regression": runner.file_record(
                INVOCATION_REGRESSION
            ),
            "candidate_budget_regression": runner.file_record(
                CANDIDATE_BUDGET_REGRESSION
            ),
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
        }

    def historical_observation_contract() -> dict[str, Any]:
        return {
            "phase": "pre_freeze_preauthority",
            "authority_present": False,
            "package_present": False,
            "freeze_present": False,
            "pre_start_intent_present": False,
            "process_started_present": False,
            "terminal_present": False,
            "result_present": False,
            "process_start_count": 0,
        }

    def observe_preauthority() -> dict[str, Any]:
        return {
            "phase": "pre_freeze_preauthority",
            "authority_present": runner.AUTHORITY.exists(),
            "package_present": runner.PACKAGE.exists(),
            "freeze_present": runner.FREEZE.exists(),
            "pre_start_intent_present": (
                runner.STATE / "pre_start_intent.json"
            ).exists(),
            "process_started_present": (
                runner.STATE / "process_started.json"
            ).exists(),
            "terminal_present": runner.TERMINAL.exists(),
            "result_present": runner.RESULT.exists(),
            "process_start_count": 0,
        }

    def verify_closure_certificate() -> dict[str, Any]:
        runner.require(
            stat.S_ISREG(os.lstat(CLOSURE_CERTIFICATE).st_mode),
            "0021 closure certificate is absent or not regular",
        )
        certificate = runner.read_json(CLOSURE_CERTIFICATE)
        core = binding_core()
        core_sha256 = runner.supervisor.canonical_value_sha256(core)
        observation = certificate.get("preauthority_observation")
        runner.require(
            set(certificate)
            == {
                "schema_version",
                "status",
                "binding_core",
                "binding_core_sha256",
                "preauthority_observation",
                "preauthority_observation_sha256",
            }
            and certificate.get("schema_version") == 2
            and certificate.get("status") == "PASS_IMMUTABLE_PHASE_SAFE_BINDING"
            and certificate.get("binding_core") == core
            and certificate.get("binding_core_sha256") == core_sha256
            and observation == historical_observation_contract()
            and certificate.get("preauthority_observation_sha256")
            == runner.supervisor.canonical_value_sha256(observation),
            "0021 immutable closure certificate changed",
        )
        return certificate

    def expected_phase_regression(certificate: dict[str, Any]) -> dict[str, Any]:
        original_whole = {
            "binding_core": certificate["binding_core"],
            "preauthority_observation": certificate["preauthority_observation"],
        }
        transitioned_observation = {
            **certificate["preauthority_observation"],
            "phase": "child_entry_after_authority_and_start",
            "authority_present": True,
            "package_present": True,
            "freeze_present": True,
            "pre_start_intent_present": True,
            "process_started_present": True,
            "process_start_count": 1,
        }
        legacy_recomputed = {
            "binding_core": binding_core(),
            "preauthority_observation": transitioned_observation,
        }
        tampered_core = copy.deepcopy(certificate["binding_core"])
        tampered_core["execution_envelope"]["cwd"] = str(ROOT / "tampered")
        original_whole_sha256 = runner.supervisor.canonical_value_sha256(
            original_whole
        )
        legacy_sha256 = runner.supervisor.canonical_value_sha256(legacy_recomputed)
        tampered_sha256 = runner.supervisor.canonical_value_sha256(tampered_core)
        return {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": (
                "PASS_LEGACY_WHOLE_RECORD_FAILS_PHASE_SEPARATED_PASSES_TAMPER_FAILS"
            ),
            "model_executed": False,
            "simulator_executed": False,
            "process_start_count": 0,
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "original_whole_record_sha256": original_whole_sha256,
            "legacy_transitioned_whole_record_sha256": legacy_sha256,
            "binding_core_sha256": certificate["binding_core_sha256"],
            "tampered_binding_core_sha256": tampered_sha256,
            "checks": {
                "legacy_whole_record_recomputation_fails_after_authority_start": (
                    legacy_sha256 != original_whole_sha256
                ),
                "historical_preauthority_observation_retained_without_recomputation": (
                    original_whole["preauthority_observation"]
                    == historical_observation_contract()
                ),
                "phase_separated_binding_core_recomputation_passes": (
                    runner.supervisor.canonical_value_sha256(binding_core())
                    == certificate["binding_core_sha256"]
                ),
                "binding_core_tamper_fails_digest_verification": (
                    tampered_sha256 != certificate["binding_core_sha256"]
                ),
                "process_not_started": True,
            },
        }

    def verify_phase_transition_regression() -> dict[str, Any]:
        runner.require(
            PHASE_TRANSITION_REGRESSION.is_file(),
            "0021 phase-transition regression is absent",
        )
        certificate = verify_closure_certificate()
        record = runner.read_json(PHASE_TRANSITION_REGRESSION)
        runner.require(
            record == expected_phase_regression(certificate)
            and all(record["checks"].values()),
            "0021 phase-transition regression changed",
        )
        return record

    def phase_binding_regression() -> int:
        runner.verify_exact_interpreter()
        runner.verify_predecessor_seals()
        runner.validate_no_execution()
        required = (
            INVOCATION_REGRESSION,
            SELECTION,
            CANDIDATE_BUDGET_REGRESSION,
            LIFECYCLE_REGRESSION,
            DEPENDENCY_PROBE,
            GATE_ORDER_REGRESSION,
            INPUT_SCALE_REGRESSION,
            TOKENIZATION_PREFLIGHT,
            TOKENIZATION_REGRESSION,
            SCALE_REGRESSION,
            ARGV_SHAPE_REGRESSION,
        )
        runner.require(
            all(path.is_file() for path in required),
            "phase-binding gate requires all dependency-first regressions",
        )
        runner.require(
            not CLOSURE_CERTIFICATE.exists()
            and not PHASE_TRANSITION_REGRESSION.exists(),
            "0021 phase-binding artifacts already exist",
        )
        verify_argv_shape_regression()
        observation = observe_preauthority()
        runner.require(
            observation == historical_observation_contract(),
            "0021 preauthority observation is not pristine",
        )
        core = binding_core()
        certificate = {
            "schema_version": 2,
            "status": "PASS_IMMUTABLE_PHASE_SAFE_BINDING",
            "binding_core": core,
            "binding_core_sha256": runner.supervisor.canonical_value_sha256(core),
            "preauthority_observation": observation,
            "preauthority_observation_sha256": (
                runner.supervisor.canonical_value_sha256(observation)
            ),
        }
        runner.write_json(CLOSURE_CERTIFICATE, certificate)
        runner.write_json(
            PHASE_TRANSITION_REGRESSION,
            expected_phase_regression(certificate),
        )
        verify_phase_transition_regression()
        print(
            "ACE2_HYBRID_0021_PHASE_BINDING_REGRESSION_PASS "
            f"certificate_sha256={runner.sha256_file(CLOSURE_CERTIFICATE)} "
            f"binding_core_sha256={certificate['binding_core_sha256']} "
            "process_start_count=0",
            flush=True,
        )
        return 0

    runner.phase_binding_regression = phase_binding_regression
    runner.verify_closure_certificate = verify_closure_certificate
    runner.verify_phase_transition_regression = verify_phase_transition_regression

    def phase_binding_manifest() -> dict[str, Any]:
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        envelope = certificate["binding_core"]["execution_envelope"]
        return {
            "certificate": runner.supervisor.file_binding(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "execution_envelope": envelope,
            "execution_envelope_sha256": (
                runner.supervisor.canonical_value_sha256(envelope)
            ),
            "phase_transition_regression": runner.supervisor.file_binding(
                PHASE_TRANSITION_REGRESSION
            ),
        }

    def source_records_0021(
        model: Path,
        tokenizer_json: Path,
        tokenizer_config: Path,
        prompts: Path,
    ) -> dict[str, Any]:
        records = prior_source_records(model, tokenizer_json, tokenizer_config, prompts)
        for path in (
            INVOCATION_REGRESSION,
            CANDIDATE_BUDGET_REGRESSION,
            ARGV_SHAPE_REGRESSION,
            CLOSURE_CERTIFICATE,
            PHASE_TRANSITION_REGRESSION,
        ):
            records[runner.public_path(path)] = runner.file_record(path)
        return dict(sorted(records.items()))

    runner.source_records = source_records_0021

    def execution_bound_paths_0021(
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
                INVOCATION_REGRESSION.resolve(strict=True),
                CANDIDATE_BUDGET_REGRESSION.resolve(strict=True),
                ARGV_SHAPE_REGRESSION.resolve(strict=True),
                CLOSURE_CERTIFICATE.resolve(strict=True),
                PHASE_TRANSITION_REGRESSION.resolve(strict=True),
            },
            key=lambda item: str(item),
        )

    runner.execution_bound_paths = execution_bound_paths_0021

    def package_bindings_0021(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bindings = prior_package_bindings(*args, **kwargs)
        bindings["phase_binding"] = phase_binding_manifest()
        return bindings

    runner.package_bindings = package_bindings_0021

    def package_template_0021(prompts_path: Path) -> int:
        verify_phase_transition_regression()
        result = prior_package_template(prompts_path)
        gate = runner.read_json(runner.TEMPLATE_GATE)
        gate["phase_safe_binding"] = {
            "invocation_fidelity_regression": runner.file_record(
                INVOCATION_REGRESSION
            ),
            "candidate_budget_regression": runner.file_record(
                CANDIDATE_BUDGET_REGRESSION
            ),
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": verify_closure_certificate()[
                "binding_core_sha256"
            ],
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        gate["checks"]["phase_safe_binding_template_passed"] = True
        runner.write_json(runner.TEMPLATE_GATE, gate)
        return result

    runner.package_template_invariant = package_template_0021

    def preflight_0021(prompts_path: Path) -> int:
        verify_phase_transition_regression()
        result = prior_preflight(prompts_path)
        record = runner.read_json(runner.PREFLIGHT)
        record["phase_safe_binding"] = {
            "invocation_fidelity_regression": runner.file_record(
                INVOCATION_REGRESSION
            ),
            "candidate_budget_regression": runner.file_record(
                CANDIDATE_BUDGET_REGRESSION
            ),
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": verify_closure_certificate()[
                "binding_core_sha256"
            ],
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        record["checks"]["phase_safe_binding_regression_passed"] = True
        runner.write_json(runner.PREFLIGHT, record)
        return result

    runner.preflight = preflight_0021

    def freeze_0021(prompts_path: Path) -> int:
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        result = prior_freeze(prompts_path)
        frozen = runner.read_json(runner.FREEZE)
        frozen["sealed_0019"] = verify_0019_seal()
        frozen["sealed_0020"] = verify_0020_seal()
        frozen["phase_safe_binding"] = {
            "candidate_budget_regression": runner.file_record(
                CANDIDATE_BUDGET_REGRESSION
            ),
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "preauthority_observation_sha256": certificate[
                "preauthority_observation_sha256"
            ],
            "execution_envelope": certificate["binding_core"][
                "execution_envelope"
            ],
            "execution_envelope_sha256": (
                runner.supervisor.canonical_value_sha256(
                    certificate["binding_core"]["execution_envelope"]
                )
            ),
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        runner.write_json(runner.FREEZE, frozen)
        return result

    runner.freeze = freeze_0021

    def verify_frozen_phase_binding() -> dict[str, Any]:
        certificate = verify_closure_certificate()
        regression = verify_phase_transition_regression()
        frozen = runner.read_json(runner.FREEZE)
        expected = {
            "candidate_budget_regression": runner.file_record(
                CANDIDATE_BUDGET_REGRESSION
            ),
            "argv_shape_regression": runner.file_record(ARGV_SHAPE_REGRESSION),
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "preauthority_observation_sha256": certificate[
                "preauthority_observation_sha256"
            ],
            "execution_envelope": certificate["binding_core"][
                "execution_envelope"
            ],
            "execution_envelope_sha256": (
                runner.supervisor.canonical_value_sha256(
                    certificate["binding_core"]["execution_envelope"]
                )
            ),
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
        }
        runner.require(
            frozen.get("phase_safe_binding") == expected
            and regression["binding_core_sha256"]
            == certificate["binding_core_sha256"],
            "frozen phase-safe binding changed",
        )
        return expected

    def package_0021(prompts_path: Path) -> int:
        verify_frozen_phase_binding()
        result = prior_package(prompts_path)
        package = runner.read_json(runner.PACKAGE)
        runner.require(
            package["bindings"].get("phase_binding") == phase_binding_manifest(),
            "execution package lost phase-safe binding",
        )
        return result

    runner.package_execution = package_0021

    def authority_0021(prompts_path: Path) -> int:
        verify_frozen_phase_binding()
        result = prior_authority(prompts_path)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        runner.require(
            package["bindings"]["phase_binding"] == phase_binding_manifest()
            and authority["bindings"]["phase_binding"]
            == package["bindings"]["phase_binding"],
            "authority lost package phase-safe binding",
        )
        return result

    runner.grant_authority = authority_0021

    def wait_for_process_started() -> dict[str, Any]:
        started_path = runner.STATE / "process_started.json"
        deadline = time.monotonic() + 5.0
        while not started_path.is_file():
            runner.require(
                time.monotonic() < deadline,
                "supervisor process-start phase handoff timed out",
            )
            time.sleep(0.01)
        return runner.read_json(started_path)

    def child_entry_attestation_0021(
        prompts_path: Path,
        expected_freeze_sha256: str,
    ) -> dict[str, Any]:
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        frozen_binding = verify_frozen_phase_binding()
        package_measurement = runner.supervisor.measure_file(runner.PACKAGE)
        authority_measurement = runner.supervisor.measure_file(runner.AUTHORITY)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        phase_binding = phase_binding_manifest()
        intent = runner.read_json(runner.STATE / "pre_start_intent.json")
        started = wait_for_process_started()
        expected_package_provenance = {
            "path": str(runner.PACKAGE),
            "byte_count": package_measurement.byte_count,
            "sha256": package_measurement.sha256,
            "package_id": package["package_id"],
        }
        expected_authority_provenance = {
            "path": str(runner.AUTHORITY),
            "byte_count": authority_measurement.byte_count,
            "sha256": authority_measurement.sha256,
            "authority_id": authority["authority_id"],
        }
        expected_argv = runner.execution_argv(
            prompts_path,
            expected_freeze_sha256,
        )
        actual_process_argv = _actual_process_argv(PRE_IMPORT_ARGV)
        actual_process_argv_sha256 = runner.supervisor.canonical_value_sha256(
            actual_process_argv
        )
        expected_modules = sorted(PRE_IMPORT_MODULES)
        runner.require(
            runner.sha256_file(runner.FREEZE) == expected_freeze_sha256,
            "actual freeze differs from child argv binding",
        )
        runner.require(
            package["bindings"]["phase_binding"] == phase_binding
            and authority["bindings"]["phase_binding"] == phase_binding
            and frozen_binding["binding_core_sha256"]
            == certificate["binding_core_sha256"],
            "certificate bytes/hash or binding core lost package/authority binding",
        )
        runner.require(
            intent.get("schema") == runner.supervisor.INTENT_SCHEMA
            and intent.get("process_start_count") == 0
            and intent.get("bindings") == package["bindings"]
            and intent.get("runtime") == authority["runtime"]
            and intent.get("provenance", {}).get("package")
            == expected_package_provenance
            and intent.get("provenance", {}).get("authority")
            == expected_authority_provenance,
            "durable pre-start intent or reservation changed",
        )
        runner.require(
            started.get("schema") == runner.supervisor.STARTED_SCHEMA
            and started.get("process_start_count") == 1
            and started.get("pid") == os.getpid()
            and started.get("provenance") == intent["provenance"]
            and sum(
                path.name == "process_started.json"
                for path in runner.STATE.iterdir()
            )
            == 1,
            "sole process_started=1 handoff changed",
        )
        runner.require(
            Path(PRE_IMPORT_ARGV[0]).resolve() == LAUNCHER,
            "pre-import argv[0] does not resolve to the exact 0021 launcher",
        )
        runner.require(
            actual_process_argv == expected_argv
            and package["bindings"]["argv"] == actual_process_argv
            and authority["bindings"]["argv"] == actual_process_argv
            and intent["bindings"]["argv"] == actual_process_argv
            and package["bindings"]["argv_sha256"]
            == actual_process_argv_sha256
            and authority["bindings"]["argv_sha256"]
            == actual_process_argv_sha256
            and intent["bindings"]["argv_sha256"]
            == actual_process_argv_sha256
            and runner.supervisor.canonical_value_sha256(expected_argv)
            == actual_process_argv_sha256
            and len(PRE_IMPORT_ARGV) == 6
            and len(actual_process_argv) == 7,
            "reconstructed process argv differs from package/authority/intent snapshot",
        )
        runner.require(
            started["provenance"] == intent["provenance"]
            and started["provenance"]["package"] == expected_package_provenance
            and started["provenance"]["authority"] == expected_authority_provenance,
            "process-started provenance does not bind the attested child argv",
        )
        runner.require(
            package["bindings"]["argv"] == expected_argv
            and authority["bindings"]["argv"] == expected_argv
            and intent["bindings"]["argv"] == expected_argv,
            "expected execution argv differs across provenance-bound records",
        )
        runner.require(
            PRE_IMPORT_ENVIRONMENT
            == certificate["binding_core"]["execution_envelope"]["environment"]
            == package["bindings"]["environment"]
            == authority["bindings"]["environment"]
            and PRE_IMPORT_CWD
            == certificate["binding_core"]["execution_envelope"]["cwd"]
            == package["bindings"]["cwd"]
            and expected_modules
            == certificate["binding_core"]["execution_envelope"][
                "pre_import_modules"
            ],
            "actual child environment/cwd/module snapshot changed",
        )
        for module in predecessor_modules:
            runner.require(
                module.PRE_IMPORT_ARGV == PRE_IMPORT_ARGV
                and module.PRE_IMPORT_ENVIRONMENT == PRE_IMPORT_ENVIRONMENT
                and module.PRE_IMPORT_CWD == PRE_IMPORT_CWD
                and sorted(module.PRE_IMPORT_MODULES) == expected_modules,
                f"effective module snapshot changed: {module.__name__}",
            )
        runner.require(
            authority["package"] == expected_package_provenance
            and authority["runtime"]["state_dir"] == str(runner.STATE)
            and authority["runtime"]["terminal_record_path"] == str(runner.TERMINAL),
            "current package/authority hash or reservation binding changed",
        )
        attestation = {
            "schema_version": 1,
            "mission_id": runner.MISSION_ID,
            "attempt_identity": runner.ATTEMPT_ID,
            "status": "PASS_PHASE_SAFE_RECONSTRUCTED_ARGV_CHILD_ENTRY_ATTESTATION",
            "actual_process_argv": actual_process_argv,
            "actual_process_argv_sha256": actual_process_argv_sha256,
            "certificate": runner.file_record(CLOSURE_CERTIFICATE),
            "binding_core_sha256": certificate["binding_core_sha256"],
            "historical_preauthority_observation_sha256": certificate[
                "preauthority_observation_sha256"
            ],
            "phase_transition_regression": runner.file_record(
                PHASE_TRANSITION_REGRESSION
            ),
            "package": runner.file_record(runner.PACKAGE),
            "authority": runner.file_record(runner.AUTHORITY),
            "pre_start_intent": runner.file_record(
                runner.STATE / "pre_start_intent.json"
            ),
            "process_started": runner.file_record(
                runner.STATE / "process_started.json"
            ),
            "checks": {
                "original_certificate_bytes_and_hash_verified": True,
                "historical_observation_verified_without_phase_recomputation": True,
                "binding_core_recomputed": True,
                "current_package_and_authority_hashes_verified": True,
                "pre_start_intent_verified": True,
                "sole_process_started_equals_one": True,
                "pre_import_launcher_resolution_verified_separately": True,
                "reconstructed_process_argv_and_canonical_sha_verified": True,
                "process_started_provenance_binds_child_argv": True,
                "actual_environment_cwd_modules_verified": True,
                "reservation_verified": True,
            },
        }
        runner.write_json(runner.STATE / "child_entry_attestation.json", attestation)
        return attestation

    runner.child_entry_attestation = child_entry_attestation_0021

    def execute_0021(prompts_path: Path, expected_freeze_sha256: str) -> int:
        child_entry_attestation_0021(prompts_path, expected_freeze_sha256)
        return prior_execute(prompts_path, expected_freeze_sha256)

    runner.execute = execute_0021

    def verify_0021() -> int:
        result = prior_verify()
        certificate = verify_closure_certificate()
        verify_phase_transition_regression()
        verify_argv_shape_regression()
        verify_frozen_phase_binding()
        attestation = runner.read_json(
            runner.STATE / "child_entry_attestation.json"
        )
        terminal = runner.read_json(runner.TERMINAL)
        package = runner.read_json(runner.PACKAGE)
        authority = runner.read_json(runner.AUTHORITY)
        intent = runner.read_json(runner.STATE / "pre_start_intent.json")
        started = runner.read_json(runner.STATE / "process_started.json")
        execution_result = runner.read_json(runner.RESULT)
        attested_argv = attestation["actual_process_argv"]
        attested_argv_sha256 = runner.supervisor.canonical_value_sha256(
            attested_argv
        )
        runner.require(
            attestation.get("status")
            == "PASS_PHASE_SAFE_RECONSTRUCTED_ARGV_CHILD_ENTRY_ATTESTATION"
            and attestation.get("binding_core_sha256")
            == certificate["binding_core_sha256"]
            and all(attestation.get("checks", {}).values())
            and attestation["actual_process_argv_sha256"] == attested_argv_sha256
            and terminal["process"]["argv"] == attested_argv
            and terminal["process"]["argv_sha256"] == attested_argv_sha256
            and package["bindings"]["argv"] == attested_argv
            and authority["bindings"]["argv"] == attested_argv
            and intent["bindings"]["argv"] == attested_argv
            and package["bindings"]["argv_sha256"] == attested_argv_sha256
            and authority["bindings"]["argv_sha256"] == attested_argv_sha256
            and intent["bindings"]["argv_sha256"] == attested_argv_sha256
            and started["provenance"] == intent["provenance"]
            and terminal["provenance"] == intent["provenance"]
            and terminal.get("outcome") == "success"
            and terminal.get("process_start_count") == 1
            and execution_result["aggregate"]["prompt_count"] == 2
            and execution_result["aggregate"]["generated_tokens_per_prompt"] == 4
            and execution_result["aggregate"]["corrected_v_bytes_compared"]
            == 2 * 4 * 128
            and execution_result["aggregate"][
                "rtl_emitted_corrected_v_consumed_downstream_all_steps"
            ]
            is True
            and execution_result["aggregate"]["downstream_logits_all_steps_equal"]
            is True
            and execution_result["aggregate"]["selected_tokens_all_steps_equal"]
            is True
            and execution_result["aggregate"]["four_token_sequences_equal"] is True
            and execution_result["aggregate"][
                "disabled_sidecar_control_all_steps"
            ]
            is True,
            "0021 terminal argv, phase-safe, or RTL-in-loop evidence changed",
        )
        print(
            "ACE2_HYBRID_0021_DECISIVE_VERIFY_PASS "
            "prompt_count=2 generated_tokens_per_prompt=4 rtl_v_bytes=1024 "
            "process_start_count=1 reconstructed_process_argv=true "
            "phase_safe_binding=true",
            flush=True,
        )
        return result

    runner.verify = verify_0021


def main() -> int:
    try:
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
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools import run_stage1_layer23_v_rank1_hybrid_0010 as prior_0010
        from tools import run_stage1_layer23_v_rank1_hybrid_0011 as prior_0011
        from tools import run_stage1_layer23_v_rank1_hybrid_0012 as prior_0012
        from tools import run_stage1_layer23_v_rank1_hybrid_0013 as prior_0013
        from tools import run_stage1_layer23_v_rank1_hybrid_0014 as prior_0014
        from tools import run_stage1_layer23_v_rank1_hybrid_0015 as prior
        from tools import run_stage1_layer23_v_rank1_hybrid_0016 as prior_0016
        from tools import run_stage1_layer23_v_rank1_hybrid_0017 as prior_0017
        from tools import run_stage1_layer23_v_rank1_hybrid_0018 as prior_0018
        from tools import run_stage1_layer23_v_rank1_hybrid_0019 as prior_0019
        from tools import run_stage1_layer23_v_rank1_hybrid_0020 as prior_0020

        predecessor_modules = (
            prior_0010,
            prior_0011,
            prior_0012,
            prior_0013,
            prior_0014,
            prior,
            prior_0016,
            prior_0017,
            prior_0018,
            prior_0019,
            prior_0020,
        )
        _preloader_require(
            prior_0010.EXPECTED_ENVIRONMENT == FROZEN_EXPECTED_ENVIRONMENT,
            "0021 invocation map differs from the frozen 0010 environment",
        )
        prior_0016._bind_predecessor_snapshots(
            predecessor_modules,
            argv=PRE_IMPORT_ARGV,
            environment=PRE_IMPORT_ENVIRONMENT,
            cwd=PRE_IMPORT_CWD,
            imported_modules=PRE_IMPORT_MODULES,
        )
        _patch_inherited_chain(prior, prior_0014)
        prior_0014._configure_prior(prior_0011)
        prior_0011._configure_prior(prior_0010)
        prior_0010._verify_pre_import_launch()

        from tools import ace2_layer23_v_input_scale_oracle
        from tools import ace2_layer23_v_rank1_integer_correction_reference
        from tools import run_stage1_layer23_v_rank1_hybrid as runner
        from tools import run_stage1_layer23_v_rank1_hybrid_0008 as base

        prior._configure_runner(
            runner,
            prior_0014,
            prior_0011,
            prior_0010,
            base,
            ace2_layer23_v_rank1_integer_correction_reference,
            ace2_layer23_v_input_scale_oracle,
            dict(os.environ),
        )
        _configure_runner(runner, prior, predecessor_modules)
        if "--select-prompts" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--select-prompts", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.select_prompts()
        if "--check-candidate-budget" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-candidate-budget", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.candidate_budget_regression()
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
        if "--check-input-scale" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-input-scale", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.input_scale_regression()
        if "--check-argv-shape" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-argv-shape", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.argv_shape_regression()
        if "--check-phase-binding" in PRE_IMPORT_ARGV[1:]:
            parser = argparse.ArgumentParser()
            parser.add_argument("--check-phase-binding", action="store_true")
            parser.parse_args(PRE_IMPORT_ARGV[1:])
            return runner.phase_binding_regression()
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
