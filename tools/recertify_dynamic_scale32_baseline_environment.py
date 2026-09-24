#!/usr/bin/env python3
"""Candidate-free Dynamic Scale32 CPU-baseline environment recertification.

This tool performs only read-only metadata/static probes plus synthetic process,
CPU, and temporary-filesystem checks.  It never imports or invokes the baseline
runner or executor, and it never loads model or dataset payloads.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from ace2_software_identity import evaluate_distribution_versions


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "shared_token_group_dynamic_scale32_v1"
RUN_ID = "dynamic-scale32-baseline-official-v1"
BASELINE = ROOT / "evidence/shared_token_group_dynamic_scale32_v1/verification/baseline"
SEALED_STATE = BASELINE / "state" / RUN_ID
RECOVERY = BASELINE / "recovery/version_identity_remediation_v1/RECOVERY_HANDOFF.json"
RECOVERY_COMPANION = RECOVERY.with_suffix(".sha256")
LEDGER = BASELINE / "state/RUN_LEDGER.json"
TERMINAL = SEALED_STATE / "terminal.bundle/TERMINAL.json"
AUTHORITY = SEALED_STATE / "EXECUTION_AUTHORITY.json"
PROMPT_MANIFEST = SEALED_STATE / "reserved_inputs/benchmark/quality/PROMPT_MANIFEST.json"
QUALITY_CONFIG = SEALED_STATE / "reserved_inputs/benchmark/quality/QUALITY_CONFIG.json"
DEFAULT_OUTPUT = BASELINE / "environment_recertification_v1"

EXPECTED_RECOVERY_SHA256 = "ca69abebe6880b6034d8413018b87b84e43db4e13faa43a78ec630b8de5eb7a2"
EXPECTED_LEDGER_SHA256 = "4945a5cb78661bb2289428d5baa6e0c3b0d72bd20842c1dca6810710a928464f"
EXPECTED_TERMINAL_SHA256 = "742cd3c2af41773534dd8b3010657ea5595bda0d421a098f7d0ec0e63c890ffe"
EXPECTED_TORCH_RUNTIME_OBSERVATION = "2.11.0+cu130"
TERMINAL_STATES = {"failed_fail_closed", "succeeded", "blocked", "cancelled"}

SOURCE_PATHS = {
    "executor": ROOT / "tools/ace2_dynamic_scale32_baseline.py",
    "runner": ROOT / "tools/run_dynamic_scale32_verification_baseline.py",
    "software_identity_helper": ROOT / "tools/ace2_software_identity.py",
    "executor_test": ROOT / "tools/test_dynamic_scale32_baseline_executor.py",
    "software_identity_test": ROOT / "tools/test_dynamic_scale32_software_identity.py",
    "runner_test": ROOT / "tools/test_dynamic_scale32_verification_baseline.py",
    "recertifier": Path(__file__).resolve(),
    "recertifier_test": ROOT / "tools/test_dynamic_scale32_environment_recertification.py",
}

PROTECTED_PATHS = {
    "pipeline_state": ROOT / "research/PIPELINE_STATE.json",
    "repository_execution_authority": BASELINE / "EXECUTION_AUTHORITY.json",
    "repository_execution_authority_companion": BASELINE / "EXECUTION_AUTHORITY.sha256",
    "manager_issuance": BASELINE / "MANAGER_ISSUANCE.json",
    "state_tree": BASELINE / "state",
    "published_runs_tree": BASELINE / "runs",
}


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def canonical_integrity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("integrity", None)
    return sha256_bytes(canonical_json_bytes(payload))


def add_integrity(value: dict[str, Any]) -> dict[str, Any]:
    value["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonical_sha256": canonical_integrity(value),
    }
    return value


def snapshot_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": relative_path(path)}
    if path.is_file():
        return {
            "bytes": path.stat().st_size,
            "exists": True,
            "kind": "file",
            "path": relative_path(path),
            "sha256": sha256_file(path),
        }
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    for current, dirnames, filenames in os.walk(path):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        directories.append(current_path.relative_to(path).as_posix())
        for name in filenames:
            item = current_path / name
            files.append(
                {
                    "bytes": item.stat().st_size,
                    "path": item.relative_to(path).as_posix(),
                    "sha256": sha256_file(item),
                }
            )
    manifest = {"directories": directories, "files": files}
    return {
        "exists": True,
        "file_count": len(files),
        "kind": "directory",
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "path": relative_path(path),
    }


def protected_snapshot() -> dict[str, Any]:
    return {name: snapshot_path(path) for name, path in sorted(PROTECTED_PATHS.items())}


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                if value is None:
                    break
                return ast.literal_eval(value)
    raise ValueError(f"literal assignment {name} not found in {path}")


def read_companion(path: Path, expected_name: str) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != expected_name or len(parts[0]) != 64:
        raise ValueError(f"invalid companion: {path}")
    return parts[0]


def source_identity(recovery: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    accepted = {
        item["path"]: item["new_sha256"]
        for item in recovery["remediation"]["changed_artifacts"]
    }
    records: dict[str, Any] = {}
    accepted_match = True
    for role, path in sorted(SOURCE_PATHS.items()):
        observed = sha256_file(path)
        expected = accepted.get(relative_path(path))
        record = {
            "path": relative_path(path),
            "sha256": observed,
            "accepted_recovery_sha256": expected,
            "matches_accepted_recovery": expected is None or observed == expected,
        }
        records[role] = record
        if expected is not None and observed != expected:
            accepted_match = False
    return records, accepted_match


def static_source_probe() -> tuple[dict[str, Any], bool]:
    records: dict[str, Any] = {}
    passed = True
    for role, path in sorted(SOURCE_PATHS.items()):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", ast.PyCF_ONLY_AST)
            record = {"parsed": True, "path": relative_path(path)}
        except (OSError, SyntaxError) as exc:
            record = {"error": f"{type(exc).__name__}: {exc}", "parsed": False, "path": relative_path(path)}
            passed = False
        records[role] = record
    return records, passed


def python_identity_probe() -> tuple[dict[str, Any], bool]:
    expected_executable = ROOT / ".venv/bin/python"
    expected_prefix = ROOT / ".venv"
    observed = {
        "base_prefix": sys.base_prefix,
        "byteorder": sys.byteorder,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "prefix": sys.prefix,
        "python_version": platform.python_version(),
        "uname": list(platform.uname()),
    }
    gate = (
        Path(sys.executable) == expected_executable
        and Path(sys.prefix) == expected_prefix
        and sys.prefix != sys.base_prefix
    )
    return observed, gate


def distribution_probe(expected: dict[str, str]) -> tuple[dict[str, Any], bool, dict[str, Any], bool]:
    records = evaluate_distribution_versions(expected)
    public_gate = all(record["matches"] for record in records.values())
    module_names = {
        "datasets": "datasets",
        "lm_eval": "lm_eval",
        "torch": "torch",
        "transformers": "transformers",
    }
    modules: dict[str, Any] = {}
    module_gate = True
    venv = (ROOT / ".venv").resolve()
    for package, module_name in module_names.items():
        spec = importlib.util.find_spec(module_name)
        origin = None if spec is None else spec.origin
        under_venv = False
        if origin and origin not in {"built-in", "frozen"}:
            try:
                under_venv = Path(origin).resolve().is_relative_to(venv)
            except OSError:
                under_venv = False
        modules[package] = {
            "module": module_name,
            "origin": origin,
            "origin_under_venv": under_venv,
            "spec_present": spec is not None,
        }
        module_gate = module_gate and spec is not None and under_venv
    return records, public_gate, modules, module_gate


def torch_runtime_probe(expected_public: str, seed: int) -> tuple[dict[str, Any], bool, bool]:
    import torch

    runtime_raw = str(torch.__version__)
    try:
        runtime_public = Version(runtime_raw).public
    except InvalidVersion:
        runtime_public = None
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_supported = False
    repeatable = False
    synthetic_cpu_ok = False
    synthetic_digest = None
    try:
        torch.use_deterministic_algorithms(True)
        deterministic_supported = torch.are_deterministic_algorithms_enabled()
        torch.manual_seed(seed)
        first = torch.rand(8, device="cpu")
        torch.manual_seed(seed)
        second = torch.rand(8, device="cpu")
        repeatable = bool(torch.equal(first, second))
        value = (torch.arange(8, dtype=torch.float32, device="cpu").square() + 1).sum()
        synthetic_cpu_ok = float(value.item()) == 148.0
        synthetic_digest = sha256_bytes(first.detach().cpu().numpy().tobytes())
    finally:
        torch.use_deterministic_algorithms(original_deterministic)
    cpu_capability = None
    capability_getter = getattr(getattr(torch, "backends", None), "cpu", None)
    if capability_getter is not None:
        getter = getattr(capability_getter, "get_cpu_capability", None)
        if callable(getter):
            cpu_capability = getter()
    observed = {
        "cpu_capability": cpu_capability,
        "cuda_build": getattr(torch.version, "cuda", None),
        "deterministic_algorithms_original": original_deterministic,
        "deterministic_algorithms_supported": deterministic_supported,
        "module_path": str(Path(torch.__file__).resolve()),
        "repeatable_seeded_cpu_tensor": repeatable,
        "runtime_matches_recovery_observation": runtime_raw == EXPECTED_TORCH_RUNTIME_OBSERVATION,
        "runtime_public": runtime_public,
        "runtime_version": runtime_raw,
        "synthetic_cpu_digest": synthetic_digest,
        "synthetic_cpu_operation_ok": synthetic_cpu_ok,
    }
    runtime_gate = runtime_public == Version(expected_public).public
    deterministic_gate = deterministic_supported and repeatable and synthetic_cpu_ok
    return observed, runtime_gate, deterministic_gate


def exact_subprocess_environment(temp_root: Path, seed: int) -> dict[str, str]:
    home = temp_root / "home"
    tmp = temp_root / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), "/usr/bin", "/bin")),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": str(seed),
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(tmp),
        "TZ": "UTC",
        "XDG_CACHE_HOME": str(home / ".cache"),
    }


def process_probe(seed: int) -> tuple[dict[str, Any], dict[str, bool]]:
    with tempfile.TemporaryDirectory(prefix="ace2-scale32-process-") as temporary:
        temp_root = Path(temporary)
        environment = exact_subprocess_environment(temp_root, seed)
        hash_code = "import os; print(os.environ['PYTHONHASHSEED']); print(hash('ace2-scale32'))"
        first = subprocess.run(
            [sys.executable, "-c", hash_code],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=10,
        )
        second = subprocess.run(
            [sys.executable, "-c", hash_code],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=10,
        )
        alternate_environment = dict(environment)
        alternate_environment["PYTHONHASHSEED"] = str(seed + 1)
        alternate = subprocess.run(
            [sys.executable, "-c", hash_code],
            capture_output=True,
            check=False,
            env=alternate_environment,
            text=True,
            timeout=10,
        )
        success = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ.get('ACE2_SENTINEL', 'absent'))"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=10,
        )
        nonzero = subprocess.run(
            [sys.executable, "-c", "import sys; print('synthetic-nonzero'); sys.exit(7)"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=10,
        )
        timed_out = False
        timeout_stdout = ""
        try:
            subprocess.run(
                [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(2)"],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
                timeout=0.15,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            captured = exc.stdout or ""
            timeout_stdout = captured.decode("utf-8", errors="replace") if isinstance(captured, bytes) else captured
    first_lines = first.stdout.strip().splitlines()
    second_lines = second.stdout.strip().splitlines()
    alternate_lines = alternate.stdout.strip().splitlines()
    hash_gate = (
        first.returncode == second.returncode == alternate.returncode == 0
        and len(first_lines) == len(second_lines) == len(alternate_lines) == 2
        and first_lines == second_lines
        and first_lines[0] == str(seed)
        and alternate_lines[0] == str(seed + 1)
        and first_lines[1] != alternate_lines[1]
    )
    success_gate = success.returncode == 0 and success.stdout.strip() == "absent"
    nonzero_gate = nonzero.returncode == 7 and nonzero.stdout.strip() == "synthetic-nonzero"
    timeout_gate = timed_out and "started" in timeout_stdout
    observed = {
        "hash_seed": {
            "alternate": alternate_lines,
            "first": first_lines,
            "second": second_lines,
        },
        "nonzero": {"returncode": nonzero.returncode, "stdout": nonzero.stdout.strip()},
        "success": {"returncode": success.returncode, "stdout": success.stdout.strip()},
        "timeout": {"raised_timeout_expired": timed_out, "stdout": timeout_stdout.strip()},
    }
    return observed, {
        "pythonhashseed": hash_gate,
        "subprocess_success": success_gate,
        "subprocess_nonzero": nonzero_gate,
        "subprocess_timeout": timeout_gate,
    }


def memory_probe() -> tuple[dict[str, Any], bool]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition(":")
            if separator and raw.strip().endswith(" kB"):
                values[key] = int(raw.strip().split()[0]) * 1024
    observed = {
        "available_bytes": values.get("MemAvailable"),
        "contract_minimum_bytes": None,
        "minimum_policy": "No operator-owned RAM minimum is frozen; readiness is limited to positive observable capacity.",
        "swap_free_bytes": values.get("SwapFree"),
        "total_bytes": values.get("MemTotal"),
    }
    gate = isinstance(observed["available_bytes"], int) and observed["available_bytes"] > 0
    return observed, gate


def cpu_probe(torch_observed: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    flags: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            lowered = line.lower()
            if lowered.startswith("flags") or lowered.startswith("features"):
                _, _, values = line.partition(":")
                flags.update(values.strip().split())
    observed = {
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "instruction_flags": sorted(flags),
        "torch_cpu_capability": torch_observed.get("cpu_capability"),
        "torch_synthetic_cpu_operation_ok": torch_observed.get("synthetic_cpu_operation_ok"),
    }
    gate = bool(observed["architecture"]) and bool(observed["cpu_count"]) and bool(flags) and bool(
        observed["torch_synthetic_cpu_operation_ok"]
    )
    return observed, gate


def filesystem_probe(output_dir: Path, required_outputs: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, bool]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_dir)
    permission_records: dict[str, Any] = {}
    atomic_record: dict[str, Any] = {}
    permissions_gate = True
    atomic_gate = False
    with tempfile.TemporaryDirectory(prefix=".capability-", dir=output_dir) as temporary:
        root = Path(temporary)
        for name in ("run_home", "run_tmp", "cache", "output"):
            directory = root / name
            directory.mkdir()
            source = directory / ".probe.tmp"
            final = directory / "probe.json"
            content = pretty_json_bytes({"directory": name, "synthetic": True})
            with source.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(source, final)
            fsync_directory(directory)
            gate = (
                os.access(directory, os.R_OK | os.W_OK | os.X_OK)
                and final.read_bytes() == content
            )
            permission_records[name] = {
                "atomic_replace": True,
                "permissions_rwx": os.access(directory, os.R_OK | os.W_OK | os.X_OK),
                "round_trip_sha256": sha256_file(final),
            }
            permissions_gate = permissions_gate and gate
        staging = root / ".five-output.staging"
        final_bundle = root / "five-output.bundle"
        staging.mkdir()
        expected_hashes: dict[str, str] = {}
        for index, name in enumerate(required_outputs):
            content = pretty_json_bytes({"index": index, "name": name, "synthetic": True})
            path = staging / name
            with path.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            expected_hashes[name] = sha256_bytes(content)
        fsync_directory(staging)
        os.replace(staging, final_bundle)
        fsync_directory(root)
        observed_names = sorted(path.name for path in final_bundle.iterdir())
        observed_hashes = {name: sha256_file(final_bundle / name) for name in observed_names}
        atomic_gate = observed_names == sorted(required_outputs) and observed_hashes == expected_hashes
        atomic_record = {
            "directory_replace": True,
            "expected_names": list(required_outputs),
            "observed_names": observed_names,
            "output_hashes": observed_hashes,
        }
    observed = {
        "atomic_publication": atomic_record,
        "disk": {
            "available_bytes": disk.free,
            "contract_minimum_bytes": None,
            "minimum_policy": "No operator-owned disk minimum is frozen; readiness is limited to positive observable capacity and successful same-filesystem publication probes.",
            "total_bytes": disk.total,
        },
        "isolated_permissions": permission_records,
        "probe_parent": relative_path(output_dir),
    }
    return observed, {
        "disk": disk.free > 0,
        "isolated_permissions": permissions_gate,
        "atomic_publication": atomic_gate,
    }


def cache_root() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface/hub"
    return Path.home() / ".cache/huggingface/hub"


def repository_cache_path(kind: str, repository: str, revision: str) -> Path:
    prefix = "models" if kind == "model" else "datasets"
    return cache_root() / f"{prefix}--{repository.replace('/', '--')}" / "snapshots" / revision


def head_probe(kind: str, repository: str, revision: str) -> dict[str, Any]:
    quoted_repository = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    namespace = "models" if kind == "model" else "datasets"
    url = f"https://huggingface.co/api/{namespace}/{quoted_repository}/revision/{quoted_revision}"
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ACE2-environment-recertification/1"})
    result: dict[str, Any] = {
        "body_bytes_read": 0,
        "error": None,
        "final_url": None,
        "method": "HEAD",
        "status": None,
        "url": url,
    }
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result["status"] = response.status
            result["final_url"] = response.geturl()
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["final_url"] = exc.geturl()
        result["error"] = f"HTTPError: {exc.reason}"
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["reachable"] = isinstance(result["status"], int) and 200 <= result["status"] < 400
    return result


def repository_probe(prompt: dict[str, Any], authority: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    model = prompt["model"]
    datasets = prompt["datasets"]
    contract_gate = (
        authority["scope"]["model_repository"] == model["repository"]
        and authority["scope"]["model_revision"] == model["revision"]
        and authority["scope"]["datasets"] == ["wikitext2", "c4_en_512"]
        and set(datasets) == {"c4_calibration", "wikitext2", "c4_en_512"}
    )
    unique: dict[tuple[str, str, str], list[str]] = {
        ("model", model["repository"], model["revision"]): ["model"]
    }
    for name, spec in datasets.items():
        unique.setdefault(("dataset", spec["repository"], spec["revision"]), []).append(name)
    records: list[dict[str, Any]] = []
    network_gate = True
    cache_probe_gate = True
    access_gate = True
    for (kind, repository, revision), uses in sorted(unique.items()):
        snapshot = repository_cache_path(kind, repository, revision)
        cache_observed = {
            "entry_count": len(list(snapshot.iterdir())) if snapshot.is_dir() else 0,
            "path": str(snapshot),
            "snapshot_present": snapshot.is_dir(),
        }
        network = head_probe(kind, repository, revision)
        record = {
            "cache": cache_observed,
            "kind": kind,
            "network": network,
            "repository": repository,
            "revision": revision,
            "uses": sorted(uses),
        }
        records.append(record)
        network_gate = network_gate and bool(network["reachable"])
        cache_probe_gate = cache_probe_gate and isinstance(cache_observed["snapshot_present"], bool)
        access_gate = access_gate and (cache_observed["snapshot_present"] or bool(network["reachable"]))
    observed = {
        "ambient_cache_root": str(cache_root()),
        "payload_imported_or_loaded": False,
        "records": records,
    }
    return observed, {
        "repository_contract": contract_gate,
        "network_metadata_reachability": network_gate,
        "cache_presence_probed": cache_probe_gate,
        "repository_revision_access": access_gate,
    }


def executor_candidate_probe(path: Path) -> tuple[dict[str, Any], bool]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported = {
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
        for name in imported
        if name == "subprocess"
        or name.startswith("subprocess.")
        or name == "importlib"
        or name.startswith("importlib.")
        or any(marker in name.lower() for marker in ("candidate", "diagnostic", "localize", "discriminator"))
    )
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    candidate_strings = sorted(value for value in strings if "candidate" in value.lower())
    markers = sorted(
        marker
        for marker in (
            "ace2_down_projection_residual_fusion_hook",
            "ace2_cross_layer_error_carry_hook",
            "--candidate-evidence",
            "--diagnostic-rope-mechanism",
            "ace2_full_model_fixed_point",
        )
        if marker in source
    )
    observed = {
        "candidate_string_literals": candidate_strings,
        "forbidden_imports": forbidden_imports,
        "forbidden_markers": markers,
        "path": relative_path(path),
    }
    gate = not forbidden_imports and not markers and candidate_strings == ["candidate_model"]
    return observed, gate


def candidate_task_paths(output_dir: Path) -> list[str]:
    candidates = [
        BASELINE / "candidate",
        BASELINE / "CANDIDATE_TASK.json",
        BASELINE / "candidate_task.json",
        BASELINE / "tasks/candidate.json",
    ]
    for pattern in ("*candidate*task*", "*task*candidate*"):
        candidates.extend(BASELINE.glob(pattern))
        candidates.extend((BASELINE / "state").glob(pattern))
        candidates.extend((BASELINE / "runs").glob(pattern))
    return sorted(
        {relative_path(path) for path in candidates if path.exists() and not path.resolve().is_relative_to(output_dir.resolve())}
    )


def active_processes() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    markers = ("tools/run_dynamic_scale32_verification_baseline.py", "tools/ace2_dynamic_scale32_baseline.py")
    proc = Path("/proc")
    if not proc.is_dir():
        return [{"error": "/proc unavailable"}]
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if any(marker in command for marker in markers):
            hits.append({"command": command, "pid": int(entry.name)})
    return hits


def execution_state_probe(output_dir: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    ledger = load_json(LEDGER)
    runs = ledger.get("runs", {})
    active = {
        run_id: record
        for run_id, record in runs.items()
        if record.get("state") not in TERMINAL_STATES
    }
    reservation_documents = sorted(relative_path(path) for path in (BASELINE / "state").rglob("RESERVATION.json"))
    published_runs = sorted(path.name for path in (BASELINE / "runs").iterdir()) if (BASELINE / "runs").is_dir() else []
    candidate_tasks = candidate_task_paths(output_dir)
    processes = active_processes()
    observed = {
        "active_processes": processes,
        "active_reservations": active,
        "candidate_task_paths": candidate_tasks,
        "historical_reservation_document_count": len(reservation_documents),
        "historical_reservation_documents": reservation_documents,
        "ledger_run_count": len(runs),
        "published_run_count": len(published_runs),
        "published_runs": published_runs,
        "repository_authority_source_count": int((BASELINE / "EXECUTION_AUTHORITY.json").is_file()),
    }
    return observed, {
        "candidate_tasks_absent": not candidate_tasks,
        "no_active_execution_process": not processes,
        "no_active_reservation": not active,
        "no_published_successor_run": not published_runs,
    }


def state_counts(output_dir: Path) -> dict[str, Any]:
    ledger = load_json(LEDGER)
    runs = ledger.get("runs", {})
    execution_total = sum(int(record.get("execution_count", 0)) for record in runs.values())
    model_totals = {"baseline_model": 0, "candidate_model": 0}
    for record in runs.values():
        counts = record.get("model_counts", {})
        for name in model_totals:
            model_totals[name] += int(counts.get(name, 0))
    return {
        "candidate_task_count": len(candidate_task_paths(output_dir)),
        "execution_count": execution_total,
        "model_counts": model_totals,
        "published_run_count": len(list((BASELINE / "runs").iterdir())) if (BASELINE / "runs").is_dir() else 0,
        "repository_authority_source_count": int((BASELINE / "EXECUTION_AUTHORITY.json").is_file()),
        "reservation_document_count": len(list((BASELINE / "state").rglob("RESERVATION.json"))),
        "run_ledger_entry_count": len(runs),
    }


def add_probe(
    probes: list[dict[str, Any]],
    required_gates: dict[str, bool],
    *,
    probe_id: str,
    probe_class: str,
    command: str,
    expected: Any,
    observed: Any,
    gate: bool,
    remediation: str,
    required: bool = True,
) -> None:
    gate = bool(gate)
    probes.append(
        {
            "class": probe_class,
            "command": command,
            "expected": expected,
            "gate": gate,
            "id": probe_id,
            "observed": observed,
            "remediation": remediation,
            "required": required,
        }
    )
    if required:
        required_gates[probe_id] = gate


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    forbidden_roots = [
        (BASELINE / "state").resolve(),
        (BASELINE / "runs").resolve(),
        (BASELINE / "blockers").resolve(),
        (BASELINE / "recovery").resolve(),
    ]
    if any(output_dir == root or output_dir.is_relative_to(root) for root in forbidden_roots):
        raise ValueError("output directory overlaps an immutable or execution-bearing baseline path")
    output_dir.mkdir(parents=True, exist_ok=True)

    before_snapshot = protected_snapshot()
    before_counts = state_counts(output_dir)
    probes: list[dict[str, Any]] = []
    required_gates: dict[str, bool] = {}

    recovery = load_json(RECOVERY)
    recovery_sha = sha256_file(RECOVERY)
    recovery_companion_sha = read_companion(RECOVERY_COMPANION, RECOVERY.name)
    add_probe(
        probes,
        required_gates,
        probe_id="accepted_recovery_handoff_hash",
        probe_class="sealed_file_hash",
        command=f"sha256({relative_path(RECOVERY)}) and verify companion",
        expected=EXPECTED_RECOVERY_SHA256,
        observed={"file_sha256": recovery_sha, "companion_sha256": recovery_companion_sha},
        gate=recovery_sha == recovery_companion_sha == EXPECTED_RECOVERY_SHA256,
        remediation="Restore the independently accepted recovery handoff bytes and companion before any successor decision.",
    )

    ledger_sha = sha256_file(LEDGER)
    terminal_sha = sha256_file(TERMINAL)
    add_probe(
        probes,
        required_gates,
        probe_id="sealed_run_ledger_byte_identity",
        probe_class="sealed_file_hash",
        command=f"sha256({relative_path(LEDGER)})",
        expected=EXPECTED_LEDGER_SHA256,
        observed=ledger_sha,
        gate=ledger_sha == EXPECTED_LEDGER_SHA256,
        remediation="Restore the sealed RUN_LEDGER bytes; do not repair or retry the sealed run.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="sealed_terminal_byte_identity",
        probe_class="sealed_file_hash",
        command=f"sha256({relative_path(TERMINAL)})",
        expected=EXPECTED_TERMINAL_SHA256,
        observed=terminal_sha,
        gate=terminal_sha == EXPECTED_TERMINAL_SHA256,
        remediation="Restore the sealed TERMINAL bytes; do not edit terminal evidence.",
    )

    ledger = load_json(LEDGER)
    terminal = load_json(TERMINAL)
    sealed_record = ledger.get("runs", {}).get(RUN_ID, {})
    sealed_contract_gate = (
        set(ledger.get("runs", {})) == {RUN_ID}
        and sealed_record.get("state") == "failed_fail_closed"
        and sealed_record.get("execution_count") == 1
        and sealed_record.get("recovery_policy") == "never_retry"
        and sealed_record.get("model_counts") == {"baseline_model": 1, "candidate_model": 0}
        and terminal.get("execution_count") == 1
        and terminal.get("model_counts") == {"baseline_model": 1, "candidate_model": 0}
        and terminal.get("terminal_state") == "failed_fail_closed"
        and terminal.get("stage_closing") is False
    )
    add_probe(
        probes,
        required_gates,
        probe_id="sealed_run_terminal_contract",
        probe_class="sealed_json_contract",
        command=f"parse {relative_path(LEDGER)} and {relative_path(TERMINAL)}",
        expected={
            "candidate_model": 0,
            "execution_count": 1,
            "baseline_model": 1,
            "recovery_policy": "never_retry",
            "run_id": RUN_ID,
            "state": "failed_fail_closed",
        },
        observed={"ledger_record": sealed_record, "terminal": terminal},
        gate=sealed_contract_gate,
        remediation="Restore the sealed exactly-once failed-run contract; never create a retry for this run_id.",
    )

    sources, sources_gate = source_identity(recovery)
    add_probe(
        probes,
        required_gates,
        probe_id="current_executor_runner_helper_test_hashes",
        probe_class="source_hash_binding",
        command="sha256 current executor, runner, software-identity helper, and focused tests",
        expected="All recovery-remediated sources match the accepted handoff; recertifier sources are freshly recorded.",
        observed=sources,
        gate=sources_gate,
        remediation="Restore the accepted recovery source versions or obtain a new independently reviewed remediation handoff.",
    )
    static_records, static_gate = static_source_probe()
    add_probe(
        probes,
        required_gates,
        probe_id="static_python_parse",
        probe_class="compile_static_check",
        command="compile(source, path, 'exec', ast.PyCF_ONLY_AST) for bound Python sources",
        expected=True,
        observed=static_records,
        gate=static_gate,
        remediation="Repair syntax in the identified source without weakening tests or frozen contracts.",
    )

    python_observed, python_gate = python_identity_probe()
    add_probe(
        probes,
        required_gates,
        probe_id="exact_venv_interpreter",
        probe_class="python_runtime_identity",
        command="sys.executable, sys.prefix, sys.base_prefix, platform metadata",
        expected={"executable": str(ROOT / ".venv/bin/python"), "prefix": str(ROOT / ".venv")},
        observed=python_observed,
        gate=python_gate,
        remediation=f"Invoke this recertification with {ROOT / '.venv/bin/python'} and repair/recreate that venv if its prefix is not isolated.",
    )

    quality = load_json(QUALITY_CONFIG)
    prompt = load_json(PROMPT_MANIFEST)
    authority = load_json(AUTHORITY)
    expected_software = quality["software"]
    distributions, distributions_gate, modules, modules_gate = distribution_probe(expected_software)
    add_probe(
        probes,
        required_gates,
        probe_id="pep440_public_distribution_versions",
        probe_class="importlib_metadata_pep440",
        command="importlib.metadata.version for each frozen distribution; compare normalized PEP 440 public versions",
        expected=expected_software,
        observed=distributions,
        gate=distributions_gate,
        remediation="Install the exact frozen public distribution versions in .venv; do not relax or rewrite any pin.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="runtime_module_origins_in_venv",
        probe_class="module_spec_identity",
        command="importlib.util.find_spec for datasets, lm_eval, torch, and transformers without importing payload libraries",
        expected="Every module spec exists under the exact .venv.",
        observed=modules,
        gate=modules_gate,
        remediation="Repair the exact .venv so every frozen distribution resolves from its site-packages tree.",
    )

    torch_observed, torch_public_gate, torch_determinism_gate = torch_runtime_probe(
        expected_software["torch"], quality["determinism"]["torch_seed"]
    )
    add_probe(
        probes,
        required_gates,
        probe_id="torch_public_and_runtime_identity",
        probe_class="runtime_local_build_identity",
        command="import torch; record torch.__version__ separately from importlib.metadata public identity",
        expected={
            "frozen_public": "2.11.0",
            "local_build_binding_adopted": False,
            "recovered_runtime_observation": EXPECTED_TORCH_RUNTIME_OBSERVATION,
        },
        observed=torch_observed,
        gate=torch_public_gate,
        remediation="Install a Torch build whose normalized public version is exactly 2.11.0; any local label remains observational unless separately frozen by higher authority.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="torch_determinism_support",
        probe_class="synthetic_torch_cpu",
        command="temporarily enable torch deterministic algorithms; repeat a seeded 8-element CPU tensor and restore prior setting",
        expected={"deterministic_algorithms": True, "seed": quality["determinism"]["torch_seed"]},
        observed=torch_observed,
        gate=torch_determinism_gate,
        remediation="Use a CPU Torch build supporting deterministic algorithms and repeatable seeded CPU tensors under the frozen seed.",
    )

    process_observed, process_gates = process_probe(quality["determinism"]["python_seed"])
    add_probe(
        probes,
        required_gates,
        probe_id="pythonhashseed_support",
        probe_class="synthetic_subprocess_determinism",
        command=f"{sys.executable} -c <fixed-hash-probe> under exact environment, twice with PYTHONHASHSEED={quality['determinism']['python_seed']} and once with seed+1",
        expected="Same seed produces identical hash output; different seed changes it.",
        observed=process_observed["hash_seed"],
        gate=process_gates["pythonhashseed"],
        remediation="Use a CPython runtime honoring PYTHONHASHSEED before interpreter startup and preserve the frozen seed in the launch environment.",
    )
    for probe_id, label, expectation, remediation in (
        ("subprocess_success", "synthetic_subprocess_success", "Captured stdout and returncode 0 under an exact non-inherited environment.", "Repair Python subprocess execution and exact environment passing."),
        ("subprocess_nonzero", "synthetic_subprocess_nonzero", "Captured stdout and returncode 7 without check=True masking.", "Repair nonzero return-code capture required for fail-closed terminalization."),
        ("subprocess_timeout", "synthetic_subprocess_timeout", "TimeoutExpired raised and flushed partial stdout captured.", "Repair subprocess timeout and partial-output capture required by the fixed runner."),
    ):
        key = probe_id.removeprefix("subprocess_")
        add_probe(
            probes,
            required_gates,
            probe_id=probe_id,
            probe_class=label,
            command=f"{sys.executable} -c <{key}-fixture> with subprocess.run(..., timeout=...) and exact env",
            expected=expectation,
            observed=process_observed[key],
            gate=process_gates[probe_id],
            remediation=remediation,
        )

    memory_observed, memory_gate = memory_probe()
    add_probe(
        probes,
        required_gates,
        probe_id="ram_capacity_observable",
        probe_class="host_memory_capability",
        command="read /proc/meminfo MemAvailable, MemTotal, and SwapFree",
        expected="Positive MemAvailable; no operator-owned minimum RAM threshold is frozen.",
        observed=memory_observed,
        gate=memory_gate,
        remediation="Provide a Linux host exposing positive available RAM; a new numeric minimum requires explicit operator ownership.",
    )
    cpu_observed, cpu_gate = cpu_probe(torch_observed)
    add_probe(
        probes,
        required_gates,
        probe_id="cpu_architecture_and_instructions",
        probe_class="host_cpu_capability",
        command="platform.machine, os.cpu_count, /proc/cpuinfo flags/features, and one synthetic Torch CPU operation",
        expected="Nonempty architecture/instruction inventory, at least one CPU, and successful synthetic Torch CPU operation.",
        observed=cpu_observed,
        gate=cpu_gate,
        remediation="Use a host CPU supported by the installed Torch build and expose /proc CPU feature metadata.",
    )

    runner_outputs = tuple(literal_assignment(SOURCE_PATHS["runner"], "REQUIRED_OUTPUTS"))
    executor_outputs = tuple(literal_assignment(SOURCE_PATHS["executor"], "OUTPUT_NAMES"))
    authority_outputs = tuple(authority["scope"]["required_outputs"])
    output_contract_gate = runner_outputs == executor_outputs == authority_outputs and len(runner_outputs) == 5
    add_probe(
        probes,
        required_gates,
        probe_id="five_output_contract_identity",
        probe_class="static_output_contract",
        command="AST literal extraction of runner REQUIRED_OUTPUTS and executor OUTPUT_NAMES; compare sealed authority required_outputs",
        expected=list(runner_outputs),
        observed={"authority": list(authority_outputs), "executor": list(executor_outputs), "runner": list(runner_outputs)},
        gate=output_contract_gate,
        remediation="Restore the exact five-output contract across runner, executor, and sealed contract inputs.",
    )
    filesystem_observed, filesystem_gates = filesystem_probe(output_dir, runner_outputs)
    add_probe(
        probes,
        required_gates,
        probe_id="disk_capacity_observable",
        probe_class="host_disk_capability",
        command=f"shutil.disk_usage({output_dir})",
        expected="Positive free bytes; no operator-owned minimum disk threshold is frozen.",
        observed=filesystem_observed["disk"],
        gate=filesystem_gates["disk"],
        remediation="Free space on the workspace filesystem; a new numeric minimum requires explicit operator ownership.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="isolated_run_path_permissions",
        probe_class="synthetic_temporary_filesystem",
        command="create isolated RUN_HOME/RUN_TMP/cache/output fixtures; write, fsync, os.replace, read, and remove under recertification-only temporary storage",
        expected="Read/write/execute plus atomic replace succeeds for all four isolated path classes.",
        observed=filesystem_observed["isolated_permissions"],
        gate=filesystem_gates["isolated_permissions"],
        remediation="Repair workspace ownership/mount permissions for isolated home, temp, cache, and output directories.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="five_output_atomic_publication",
        probe_class="synthetic_atomic_directory_publication",
        command="write and fsync five synthetic files in staging; fsync directory; os.replace staging to final; fsync parent; verify exact names and hashes; clean temporary tree",
        expected=list(runner_outputs),
        observed=filesystem_observed["atomic_publication"],
        gate=filesystem_gates["atomic_publication"],
        remediation="Use one filesystem supporting file/directory fsync and same-filesystem atomic directory replacement for publication.",
    )

    repository_observed, repository_gates = repository_probe(prompt, authority)
    repository_expectation = {
        "datasets": prompt["datasets"],
        "model": prompt["model"],
        "network_method": "HEAD",
        "payload_downloaded": False,
    }
    add_probe(
        probes,
        required_gates,
        probe_id="repository_revision_contract",
        probe_class="frozen_repository_metadata",
        command=f"parse {relative_path(PROMPT_MANIFEST)} and sealed authority scope",
        expected=repository_expectation,
        observed=repository_observed,
        gate=repository_gates["repository_contract"],
        remediation="Restore exact model/dataset repository and revision metadata; do not substitute revisions.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="repository_network_metadata_reachability",
        probe_class="network_control_plane_head",
        command="HTTPS HEAD to Hugging Face model/dataset revision metadata endpoints with 15-second timeout; read zero body bytes",
        expected="Every exact frozen repository revision returns HTTP 2xx/3xx without payload read.",
        observed=repository_observed["records"],
        gate=repository_gates["network_metadata_reachability"],
        remediation="Restore DNS/TLS/HTTPS access to huggingface.co metadata endpoints or preseed and separately bind complete exact-revision caches before successor authorization.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="repository_cache_presence_probed",
        probe_class="cache_metadata_only",
        command="stat expected Hugging Face snapshot directories for exact repositories/revisions; do not open payload files",
        expected="Cache presence is recorded as a capability observation, not silently assumed.",
        observed=repository_observed,
        gate=repository_gates["cache_presence_probed"],
        remediation="Repair cache-path visibility so exact-revision snapshot presence can be inspected without payload loading.",
    )
    add_probe(
        probes,
        required_gates,
        probe_id="repository_revision_access_capability",
        probe_class="cache_or_network_capability",
        command="for each exact revision, require observed snapshot directory or successful metadata-only HEAD reachability",
        expected=True,
        observed=repository_observed["records"],
        gate=repository_gates["repository_revision_access"],
        remediation="Provide metadata reachability or a separately verified exact-revision cache; this recertification may not download payloads.",
    )

    candidate_observed, candidate_gate = executor_candidate_probe(SOURCE_PATHS["executor"])
    add_probe(
        probes,
        required_gates,
        probe_id="candidate_import_surface_absent",
        probe_class="executor_ast_candidate_interlock",
        command=f"AST parse {relative_path(SOURCE_PATHS['executor'])}; inspect imports, forbidden markers, and candidate string literals",
        expected={"candidate_string_literals": ["candidate_model"], "forbidden_imports": [], "forbidden_markers": []},
        observed=candidate_observed,
        gate=candidate_gate,
        remediation="Remove candidate/diagnostic imports or entrypoints from the baseline executor and obtain fresh independent review.",
    )
    execution_observed, execution_gates = execution_state_probe(output_dir)
    for probe_id, key, expected, remediation in (
        ("candidate_tasks_absent", "candidate_tasks_absent", [], "Remove any candidate execution task under baseline authority/state paths; do not execute it."),
        ("no_active_execution_process", "no_active_execution_process", [], "Stop the active baseline runner/executor process and independently reconcile authoritative state."),
        ("no_active_reservation", "no_active_reservation", {}, "Resolve active reservation state without rerunning the sealed run; a successor requires a new run_id and authority."),
        ("no_published_successor_run", "no_published_successor_run", [], "Quarantine any unauthorized published successor run and obtain Manager direction."),
    ):
        observed_key = {
            "candidate_tasks_absent": "candidate_task_paths",
            "no_active_execution_process": "active_processes",
            "no_active_reservation": "active_reservations",
            "no_published_successor_run": "published_runs",
        }[probe_id]
        add_probe(
            probes,
            required_gates,
            probe_id=probe_id,
            probe_class="authoritative_execution_inactivity",
            command="parse RUN_LEDGER and enumerate baseline task/run paths plus /proc command lines; no runner action invoked",
            expected=expected,
            observed=execution_observed[observed_key],
            gate=execution_gates[key],
            remediation=remediation,
        )

    after_snapshot = protected_snapshot()
    after_counts = state_counts(output_dir)
    protected_unchanged = before_snapshot == after_snapshot
    counts_unchanged = before_counts == after_counts
    no_new_authority = after_counts["repository_authority_source_count"] == before_counts["repository_authority_source_count"] == 1
    no_new_reservation = after_counts["reservation_document_count"] == before_counts["reservation_document_count"]
    no_new_run = (
        after_counts["run_ledger_entry_count"] == before_counts["run_ledger_entry_count"] == 1
        and after_counts["published_run_count"] == before_counts["published_run_count"] == 0
    )
    no_execution_change = (
        counts_unchanged
        and after_counts["execution_count"] == 1
        and after_counts["model_counts"] == {"baseline_model": 1, "candidate_model": 0}
    )
    for probe_id, gate, expected, observed, remediation in (
        ("protected_state_byte_identity", protected_unchanged, before_snapshot, after_snapshot, "Restore every protected authority/state/ledger/run/PIPELINE_STATE byte and investigate unauthorized mutation."),
        ("no_new_execution_authority", no_new_authority, 1, after_counts["repository_authority_source_count"], "Remove any newly created authority; this task grants none."),
        ("no_new_reservation", no_new_reservation, before_counts["reservation_document_count"], after_counts["reservation_document_count"], "Remove and investigate any reservation created during recertification without invoking recovery or execution."),
        ("no_new_run_or_task", no_new_run and after_counts["candidate_task_count"] == 0, {"ledger_entries": 1, "published_runs": 0, "candidate_tasks": 0}, after_counts, "Remove and investigate any new run/task artifact; a successor requires separate Manager authority."),
        ("no_execution_count_change", no_execution_change, {"execution_count": 1, "model_counts": {"baseline_model": 1, "candidate_model": 0}}, after_counts, "Restore sealed counts and investigate any execution; never retry the sealed run."),
    ):
        add_probe(
            probes,
            required_gates,
            probe_id=probe_id,
            probe_class="before_after_protected_state",
            command="hash protected files/trees and count authority/reservation/run/task/execution state before and after all probes",
            expected=expected,
            observed=observed,
            gate=gate,
            remediation=remediation,
        )

    all_pass = all(required_gates.values()) and bool(required_gates)
    failed = [probe for probe in probes if probe["required"] and not probe["gate"]]
    audit: dict[str, Any] = {
        "authority_granted": False,
        "claim_boundary": "Candidate-free CPU-baseline environment readiness only. No model or dataset payload was imported, loaded, or downloaded; no runner preflight/run/recover/verify action or executor was invoked; no authority, reservation, run, candidate task, PIPELINE_STATE change, stage transition, PPA, benchmark, signoff, tapeout, or silicon claim is made.",
        "contract_id": CONTRACT_ID,
        "created_at_utc": now_utc(),
        "failed_required_gates": [
            {"id": probe["id"], "remediation": probe["remediation"]} for probe in failed
        ],
        "kind": "dynamic_scale32_baseline_environment_recertification_v1",
        "no_execution_counts": {
            "after": after_counts,
            "before": before_counts,
            "forbidden_actions_invoked": [],
            "model_or_data_loaded": False,
            "model_or_data_payload_downloaded": False,
            "runner_or_executor_invoked": False,
        },
        "policy": {
            "pins_relaxed": False,
            "public_version_authority": "importlib.metadata distribution metadata with PEP 440 normalized public-version equality",
            "torch_frozen_public": expected_software["torch"],
            "torch_local_build_binding_adopted": False,
            "torch_runtime_observed": torch_observed["runtime_version"],
        },
        "probes": probes,
        "protected_state": {"after": after_snapshot, "before": before_snapshot, "unchanged": protected_unchanged},
        "required_gates": required_gates,
        "required_gates_all_pass": all_pass,
        "review": {
            "independent_review_required": True,
            "stage_closing": False,
            "status": "pending_fresh_reviewer",
        },
        "schema_version": 1,
        "sealed_run": {
            "execution_count": sealed_record.get("execution_count"),
            "model_counts": sealed_record.get("model_counts"),
            "recovery_handoff_sha256": recovery_sha,
            "recovery_policy": sealed_record.get("recovery_policy"),
            "run_id": RUN_ID,
            "run_ledger_sha256": ledger_sha,
            "state": sealed_record.get("state"),
            "terminal_sha256": terminal_sha,
        },
        "source_identity": sources,
        "stage_closing": False,
        "status": "all_required_gates_passed_pending_independent_review" if all_pass else "blocked_failed_required_environment_gates",
        "successor_authority_recommendation": {
            "authorized_by_this_audit": False,
            "new_run_id_required": True,
            "recommend_separate_manager_decision_after_independent_acceptance": all_pass,
            "reuse_sealed_run_id": False,
        },
    }
    add_integrity(audit)
    audit_bytes = pretty_json_bytes(audit)
    audit_path = output_dir / "ENVIRONMENT_AUDIT.json"
    atomic_write(audit_path, audit_bytes)
    audit_sha = sha256_bytes(audit_bytes)
    atomic_write(output_dir / "ENVIRONMENT_AUDIT.sha256", f"{audit_sha}  {audit_path.name}\n".encode("utf-8"))

    companion: dict[str, Any] = {
        "audit": {"path": relative_path(audit_path), "sha256": audit_sha},
        "authority_granted": False,
        "claim_boundary": audit["claim_boundary"],
        "created_at_utc": now_utc(),
        "forbidden_actions": [
            "runner preflight/run/recover/verify",
            "baseline executor invocation",
            "model or dataset import/load/download",
            "EXECUTION_AUTHORITY/state/ledger/runs/candidate-task/PIPELINE_STATE mutation",
        ],
        "independent_review": {
            "compare_fields": [
                "required_gates",
                "sealed_run",
                "policy",
                "source_identity",
                "no_execution_counts",
            ],
            "reproduction_command": [
                sys.executable,
                relative_path(Path(__file__).resolve()),
                "--output-dir",
                "{REVIEW_TEMP_DIR}",
            ],
            "required": True,
            "stage_closing": False,
        },
        "interpreter": python_observed,
        "kind": "dynamic_scale32_baseline_environment_recertification_reproduction_v1",
        "permitted_probe_classes": sorted({probe["class"] for probe in probes}),
        "recertifier": sources["recertifier"],
        "required_gate_ids": sorted(required_gates),
        "schema_version": 1,
        "stage_closing": False,
    }
    add_integrity(companion)
    companion_bytes = pretty_json_bytes(companion)
    companion_path = output_dir / "REPRODUCE.json"
    atomic_write(companion_path, companion_bytes)
    companion_sha = sha256_bytes(companion_bytes)
    atomic_write(output_dir / "REPRODUCE.sha256", f"{companion_sha}  {companion_path.name}\n".encode("utf-8"))
    return {
        "audit": relative_path(audit_path),
        "audit_sha256": audit_sha,
        "companion": relative_path(companion_path),
        "companion_sha256": companion_sha,
        "failed_required_gate_ids": [probe["id"] for probe in failed],
        "required_gates_all_pass": all_pass,
        "status": audit["status"],
    }


def verify_existing(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    audit_path = output_dir / "ENVIRONMENT_AUDIT.json"
    companion_path = output_dir / "REPRODUCE.json"
    audit_sha = read_companion(output_dir / "ENVIRONMENT_AUDIT.sha256", audit_path.name)
    companion_sha = read_companion(output_dir / "REPRODUCE.sha256", companion_path.name)
    if sha256_file(audit_path) != audit_sha:
        raise ValueError("audit byte hash differs from companion")
    if sha256_file(companion_path) != companion_sha:
        raise ValueError("reproduction companion byte hash differs")
    audit = load_json(audit_path)
    companion = load_json(companion_path)
    if audit.get("integrity", {}).get("canonical_sha256") != canonical_integrity(audit):
        raise ValueError("audit canonical integrity differs")
    if companion.get("integrity", {}).get("canonical_sha256") != canonical_integrity(companion):
        raise ValueError("reproduction companion canonical integrity differs")
    if companion.get("audit") != {"path": relative_path(audit_path), "sha256": audit_sha}:
        raise ValueError("reproduction companion does not bind audit bytes")
    if audit.get("stage_closing") is not False or audit.get("authority_granted") is not False:
        raise ValueError("audit authority/stage boundary differs")
    if companion.get("stage_closing") is not False or companion.get("authority_granted") is not False:
        raise ValueError("companion authority/stage boundary differs")
    required_gates = audit.get("required_gates")
    if not isinstance(required_gates, dict) or not required_gates or not all(isinstance(value, bool) for value in required_gates.values()):
        raise ValueError("required gates are not an explicit boolean map")
    if audit.get("required_gates_all_pass") is not all(required_gates.values()):
        raise ValueError("required gate summary differs")
    live_hashes = {
        "recovery_handoff_sha256": sha256_file(RECOVERY),
        "run_ledger_sha256": sha256_file(LEDGER),
        "terminal_sha256": sha256_file(TERMINAL),
    }
    if live_hashes != {
        "recovery_handoff_sha256": EXPECTED_RECOVERY_SHA256,
        "run_ledger_sha256": EXPECTED_LEDGER_SHA256,
        "terminal_sha256": EXPECTED_TERMINAL_SHA256,
    }:
        raise ValueError(f"live sealed hashes differ: {live_hashes}")
    for record in audit.get("source_identity", {}).values():
        path = ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"bound source hash differs: {record['path']}")
    return {
        "audit_sha256": audit_sha,
        "companion_sha256": companion_sha,
        "required_gate_count": len(required_gates),
        "required_gates_all_pass": audit["required_gates_all_pass"],
        "sealed_hashes": live_hashes,
        "status": audit["status"],
        "verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = verify_existing(args.output_dir) if args.verify_existing else generate(args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
