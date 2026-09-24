#!/usr/bin/env python3
"""Sequence-aware checkpoint-176 W4A8 Icarus generation backend.

This module reuses the accepted ACE-2 integer references and RTL cores without
mutating or replaying either accepted sealed attempt.  It executes one absolute
token position at a time, which gives prompt prefill and generated-token decode
the same causal K/V-cache semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import struct
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_lora_v4_two_token_24layer_rtl_authorized as transformer
import run_lora_v4_lm_head_top_token_rtl as lm_head
import ace2_layer23_v_rank1_integer_correction_reference as rank1_reference


canonical = transformer.canonical
frontier = transformer.frontier
layer_runner = transformer.layer_runner

LAYERS = transformer.LAYERS
HIDDEN = transformer.HIDDEN
INTERMEDIATE = transformer.INTERMEDIATE
Q_HEADS = transformer.Q_HEADS
KV_HEADS = transformer.KV_HEADS
HEAD_DIM = transformer.HEAD_DIM
PROJECTION_ORDER = tuple(transformer.PROJECTION_ORDER)
PROJECTION_CHANNELS = dict(transformer.PROJECTION_CHANNELS)
# Small Stage-1 headroom above the canonical 30-token Qwen chat prefix plus
# four generated tokens.  This is a total prompt-plus-generation bound; the
# backend processes at most one fewer causal positions because the last token
# is emitted rather than fed back through the transformer.
MAX_CONTEXT_TOKENS = 43
MODEL_OUTPUT_DOMAIN = lm_head.VOCAB
LM_HEAD_TILE_OUTPUTS = lm_head.TILE_OUTPUTS
RANK1_INPUT_SCALE = 0.25991058349609375
RANK1_RETAINED_V_SCALE = rank1_reference.RETAINED_OUTPUT_SCALE
RANK1_SIDECAR = ROOT / "rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv"
RANK1_TB = ROOT / "verification/tb/ace2_layer23_v_rank1_hybrid_shell_tb.sv"
RANK1_TOP = "ace2_layer23_v_rank1_hybrid_shell_tb"
RANK1_RTL_SOURCES = (
    ROOT / "rtl/ace2_pkg.sv",
    ROOT / "rtl/ace2_rmsnorm_core.sv",
    ROOT / "rtl/ace2_dynamic_scale32_core.sv",
    ROOT / "rtl/ace2_w4a8_proj_core.sv",
    ROOT / "rtl/ace2_rope_core.sv",
    ROOT / "rtl/ace2_dynamic_rope_head_core.sv",
    ROOT / "rtl/ace2_fixed_q7_rope_score_core.sv",
    ROOT / "rtl/ace2_relative_rope_score_fusion_core.sv",
    ROOT / "rtl/ace2_attention_score_core.sv",
    ROOT / "rtl/ace2_softmax_core.sv",
    ROOT / "rtl/ace2_attention_compose_core.sv",
    ROOT / "rtl/ace2_silu_gate_core.sv",
    RANK1_SIDECAR,
    ROOT / "rtl/ace2_shell.sv",
)
RANK1_PASS_PATTERN = re.compile(
    r"ACE2_LAYER23_V_RANK1_HYBRID_SHELL_PASS "
    r"outputs=128 descriptor_accept=1 disabled_control=1 rank_exact=1 "
    r"saturation=(?P<saturation>[01]) overflow=(?P<overflow>[01]) "
    r"cycles=(?P<cycles>\d+) reads=(?P<reads>\d+) writes=(?P<writes>\d+)"
)
RANK1_BEAT_PATTERN = re.compile(
    r"^ACE2_HYBRID_V_BEAT beat=(?P<beat>\d+) "
    r"rank_acc=(?P<rank_acc>-?\d+) "
    r"rank_rounded=(?P<rank_rounded>-?\d+) "
    r"rank_s8=(?P<rank_s8>-?\d+) "
    r"baseline=(?P<baseline>[0-9a-fA-F]{32}) "
    r"corrected=(?P<corrected>[0-9a-fA-F]{32})$",
    re.MULTILINE,
)
PROJECTION_RESULT_PATTERN = re.compile(
    r"^RTL_PROJ_RESULT operator=(\d+) channel=(\d+) "
    r"out_u8=(\d+) acc=(-?\d+) saturation=(\d+) overflow=(\d+)$"
)


class BackendError(RuntimeError):
    """The real Icarus generation backend rejected or failed an execution."""


@dataclass
class CarriedGenerationState:
    caches: list[dict[str, Any]]
    templates: list[dict[str, Any] | None]
    context_token_ids: list[int]
    execution_mode: str
    snapshot_model_sha256: str | None = None
    snapshot_config_sha256: str | None = None
    valid: bool = True


def new_carried_generation_state(execution_mode: str) -> CarriedGenerationState:
    require(
        execution_mode in {"official", "stage1_product"},
        "generation execution mode is unsupported",
    )
    return CarriedGenerationState(
        caches=[empty_layer_cache() for _ in range(LAYERS)],
        templates=[None] * LAYERS,
        context_token_ids=[],
        execution_mode=execution_mode,
    )


def validate_carried_generation_state(
    state: CarriedGenerationState,
    prompt_token_ids: list[int],
    snapshot_record: dict[str, Any],
    execution_mode: str,
) -> int:
    require(isinstance(state, CarriedGenerationState), "carried state has the wrong type")
    require(state.valid, "carried state is invalid after an incomplete execution")
    require(state.execution_mode == execution_mode, "carried-state execution mode differs")
    require(len(state.caches) == LAYERS, "carried state does not contain 24 layer caches")
    require(len(state.templates) == LAYERS, "carried state does not contain 24 layer templates")
    retained = len(state.context_token_ids)
    require(retained < len(prompt_token_ids), "turn has no uncached suffix to execute")
    require(
        prompt_token_ids[:retained] == state.context_token_ids,
        "turn prefix differs from retained carried state",
    )
    for layer_id, cache in enumerate(state.caches):
        require(
            isinstance(cache, dict)
            and set(cache) == {"k", "v", "float_k", "float_v"},
            f"carried layer {layer_id} cache fields differ",
        )
        require(
            all(len(cache[field]) == retained for field in cache),
            f"carried layer {layer_id} cache length differs",
        )
    if retained:
        require(
            all(isinstance(template, dict) for template in state.templates),
            "carried state has an incomplete layer template",
        )
    else:
        require(
            all(template is None for template in state.templates),
            "fresh carried state unexpectedly contains layer templates",
        )
    model_sha256 = snapshot_record["model"]["sha256"]
    config_sha256 = snapshot_record["config"]["sha256"]
    if state.snapshot_model_sha256 is not None:
        require(
            state.snapshot_model_sha256 == model_sha256
            and state.snapshot_config_sha256 == config_sha256,
            "carried-state model snapshot differs",
        )
    return retained


class SimulatorProcessError(BackendError):
    """A simulator child failed or produced no executable RTL evidence."""

    def __init__(
        self,
        message: str,
        *,
        classification: str,
        process_kind: str,
        returncode: int | None,
        rtl_executed: bool,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.process_kind = process_kind
        self.returncode = returncode
        self.rtl_executed = rtl_executed


class ProcessTreeRssTracker:
    """Sample aggregate RSS for this process and every live descendant."""

    def __init__(self, *, interval_seconds: float = 0.01) -> None:
        require(interval_seconds > 0, "RSS sampling interval must be positive")
        self.root_pid = os.getpid()
        self.interval_seconds = float(interval_seconds)
        self.started_monotonic = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._sampling_errors = 0
        self._sampled_peak_rss_kib = 0
        self._root_peak_rss_kib = 0
        self._spawned: list[dict[str, Any]] = []
        self._active: dict[
            int, tuple[subprocess.Popen[str], dict[str, Any]]
        ] = {}

    @staticmethod
    def _children(pid: int) -> list[int]:
        path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            raw = path.read_text(encoding="ascii").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return []
        return [int(value) for value in raw.split()] if raw else []

    @staticmethod
    def _rss_kib(pid: int) -> int | None:
        path = Path(f"/proc/{pid}/status")
        try:
            for line in path.read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            return None
        return None

    def _snapshot(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        rss: dict[int, int] = {}
        children: dict[int, list[int]] = {}
        pending = [self.root_pid]
        seen: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            child_pids = self._children(pid)
            children[pid] = child_pids
            pending.extend(child_pids)
            value = self._rss_kib(pid)
            if value is not None:
                rss[pid] = value
        return rss, children

    @staticmethod
    def _subtree(pid: int, children: dict[int, list[int]]) -> set[int]:
        members: set[int] = set()
        pending = [pid]
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(children.get(current, []))
        return members

    def sample(self) -> None:
        try:
            rss, children = self._snapshot()
        except OSError:
            with self._lock:
                self._sampling_errors += 1
            return
        with self._lock:
            self._sample_count += 1
            self._sampled_peak_rss_kib = max(
                self._sampled_peak_rss_kib, sum(rss.values())
            )
            self._root_peak_rss_kib = max(
                self._root_peak_rss_kib, rss.get(self.root_pid, 0)
            )
            for pid, (_, record) in self._active.items():
                if pid not in rss:
                    continue
                members = self._subtree(pid, children)
                record["observed_samples"] += 1
                record["peak_process_rss_kib"] = max(
                    int(record["peak_process_rss_kib"] or 0), rss[pid]
                )
                record["peak_subtree_rss_kib"] = max(
                    int(record["peak_subtree_rss_kib"] or 0),
                    sum(rss.get(member, 0) for member in members),
                )

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def start(self) -> None:
        require(Path("/proc/self/status").is_file(), "process-tree RSS requires Linux /proc")
        require(self._thread is None, "process-tree RSS tracker was already started")
        self.sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="ace2-process-tree-rss",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 10))
        self.sample()

    def note_spawn(self, process: subprocess.Popen[str], command: list[str]) -> None:
        program = Path(command[0]).name
        executable = shutil.which(command[0]) or command[0]
        with self._lock:
            require(
                process.pid not in self._active,
                f"duplicate tracked child pid: {process.pid}",
            )
            record = {
                "pid": process.pid,
                "program": program,
                "executable": display_path(executable),
                "command": display_command(command),
                "started_monotonic": time.monotonic(),
                "finished_monotonic": None,
                "returncode": None,
                "observed_samples": 0,
                "peak_process_rss_kib": None,
                "peak_subtree_rss_kib": None,
            }
            self._spawned.append(record)
            self._active[process.pid] = (process, record)
        self.sample()

    def note_exit(self, process: subprocess.Popen[str]) -> dict[str, Any] | None:
        self.sample()
        with self._lock:
            active = self._active.get(process.pid)
            if active is None:
                return None
            tracked_process, record = active
            require(
                tracked_process is process,
                f"tracked child pid owner mismatch: {process.pid}",
            )
            record["finished_monotonic"] = time.monotonic()
            record["returncode"] = process.returncode
            del self._active[process.pid]
            return dict(record)

    def summary(self, *, require_complete: bool = True) -> dict[str, Any]:
        self.sample()
        with self._lock:
            children = [dict(record) for record in self._spawned]
            sample_count = self._sample_count
            sampling_errors = self._sampling_errors
            sampled_peak = self._sampled_peak_rss_kib
            sampled_root_peak = self._root_peak_rss_kib
        icarus_children = [
            record for record in children if record["program"] in {"iverilog", "vvp"}
        ]
        required_icarus_programs = {"iverilog", "vvp"}
        observed_icarus_programs = {
            str(record["program"]) for record in icarus_children
        }
        missing_icarus_programs = sorted(
            required_icarus_programs - observed_icarus_programs
        )
        unterminated = [
            record
            for record in icarus_children
            if record["finished_monotonic"] is None or record["returncode"] is None
        ]
        unsampled = [
            record for record in icarus_children if int(record["observed_samples"]) < 1
        ]
        all_icarus_children_observed = (
            not missing_icarus_programs and not unterminated
        )
        all_icarus_children_sampled = (
            not missing_icarus_programs and not unsampled
        )
        complete = (
            bool(sample_count)
            and all_icarus_children_observed
            and not sampling_errors
        )
        if require_complete:
            require(
                not missing_icarus_programs,
                "process-tree RSS missed required Icarus programs: "
                + ", ".join(missing_icarus_programs),
            )
            require(
                not unterminated,
                "process-tree RSS has an unterminated iverilog/vvp child",
            )
            require(sampling_errors == 0, "process-tree RSS sampling reported errors")
        root_ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {
            "root_pid": self.root_pid,
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": sample_count,
            "sampling_errors": sampling_errors,
            "sampled_process_tree_peak_rss_kib": int(sampled_peak),
            "sampled_root_peak_rss_kib": int(sampled_root_peak),
            "root_ru_maxrss_kib": root_ru_maxrss,
            "peak_rss_kib": max(int(sampled_peak), root_ru_maxrss),
            "spawned_process_count": len(children),
            "icarus_child_count": len(icarus_children),
            "required_icarus_programs": sorted(required_icarus_programs),
            "observed_icarus_programs": sorted(observed_icarus_programs),
            "missing_icarus_programs": missing_icarus_programs,
            "all_icarus_children_observed": all_icarus_children_observed,
            "all_icarus_children_terminally_accounted": all_icarus_children_observed,
            "all_icarus_children_sampled": all_icarus_children_sampled,
            "sampled_icarus_child_count": len(icarus_children) - len(unsampled),
            "unsampled_icarus_child_count": len(unsampled),
            "complete": complete,
            "children": children,
            "definition": (
                "Linux /proc sampling of aggregate VmRSS for the Python root and every "
                "recursively discovered descendant; every spawned iverilog/vvp PID must "
                "be synchronously registered and terminally accounted, while per-child "
                "sample coverage is reported separately because short-lived children can "
                "exit between sampler ticks"
            ),
        }


_ACTIVE_PROCESS_TREE_RSS_TRACKER: ProcessTreeRssTracker | None = None


@contextmanager
def process_tree_rss_tracking(
    *, interval_seconds: float = 0.01
) -> Iterable[ProcessTreeRssTracker]:
    global _ACTIVE_PROCESS_TREE_RSS_TRACKER
    require(_ACTIVE_PROCESS_TREE_RSS_TRACKER is None, "process-tree RSS tracker is already active")
    tracker = ProcessTreeRssTracker(interval_seconds=interval_seconds)
    _ACTIVE_PROCESS_TREE_RSS_TRACKER = tracker
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        _ACTIVE_PROCESS_TREE_RSS_TRACKER = None


def tracked_run(
    command: list[str],
    *,
    cwd: Path,
    timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tracker = _ACTIVE_PROCESS_TREE_RSS_TRACKER
    if tracker is not None:
        tracker.note_spawn(process, command)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        if tracker is not None:
            tracker.note_exit(process)
        raise
    child_record = tracker.note_exit(process) if tracker is not None else None
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        child_record,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: Path) -> str:
    """Return a reversible repository-root-relative artifact identity."""
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def runtime_path(path: Path) -> str:
    """Return a repository-root-relative path usable with simulator cwd=ROOT."""
    return public_path(path)


def display_path(value: str | Path) -> str:
    """Render process evidence without absolute host paths or ``..`` escapes."""
    path = Path(value)
    if not path.is_absolute():
        return str(value)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def validate_context_request(
    prompt_token_ids: list[int], max_new_tokens: int
) -> dict[str, int]:
    require(bool(prompt_token_ids), "prompt token sequence must be non-empty")
    require(max_new_tokens >= 1, "generated token count must be positive")
    total_context_tokens = len(prompt_token_ids) + max_new_tokens
    require(
        total_context_tokens <= MAX_CONTEXT_TOKENS,
        (
            f"prompt plus generation requires {total_context_tokens} total tokens; "
            f"backend context bound is {MAX_CONTEXT_TOKENS}"
        ),
    )
    return {
        "prompt_token_count": len(prompt_token_ids),
        "max_new_tokens": max_new_tokens,
        "total_context_tokens": total_context_tokens,
        "processed_positions": total_context_tokens - 1,
        "backend_context_bound": MAX_CONTEXT_TOKENS,
    }


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required file is absent: {public_path(path)}")
    return {
        "path": public_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def timing_record(
    total_wall_seconds: float,
    icarus_wall_seconds: float,
    parent_process_cpu_seconds: float,
) -> dict[str, float | str]:
    require(total_wall_seconds >= 0, "timing total wall seconds is negative")
    require(icarus_wall_seconds >= 0, "timing Icarus wall seconds is negative")
    require(parent_process_cpu_seconds >= 0, "timing parent CPU seconds is negative")
    return {
        "total_wall_seconds": float(total_wall_seconds),
        "cpu_wall_seconds": max(0.0, float(total_wall_seconds) - float(icarus_wall_seconds)),
        "icarus_wall_seconds": float(icarus_wall_seconds),
        "parent_process_cpu_seconds": float(parent_process_cpu_seconds),
        "cpu_wall_definition": "host orchestration wall time outside recorded iverilog/vvp child intervals",
        "icarus_wall_definition": "sum of recorded iverilog compile and vvp simulation wall intervals",
    }


def simulator_phase_timing(executions: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(executions)
    compile_wall_seconds = sum(
        float(record["iverilog_wall_seconds"]) for record in records
    )
    simulation_wall_seconds = sum(
        float(record["vvp_wall_seconds"]) for record in records
    )
    icarus_wall_seconds = sum(
        float(record["icarus_wall_seconds"]) for record in records
    )
    require(
        math.isclose(
            compile_wall_seconds + simulation_wall_seconds,
            icarus_wall_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "Icarus compile/simulation timing does not sum to recorded child wall time",
    )
    return {
        "compile_wall_seconds": compile_wall_seconds,
        "simulation_wall_seconds": simulation_wall_seconds,
        "icarus_wall_seconds": icarus_wall_seconds,
    }


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return file_record(path)


TRANSIENT_EXECUTION_DIRECTORIES = (
    "execution_sources",
    "sim",
    "vectors",
)
TRANSIENT_EXECUTION_FILES = ("lm_head_weight.hex",)
RETAINED_COMPARISON_TENSOR_FILES = frozenset(
    {
        "layer_output_s8.bin",
        "layer_output_scale_f64le.bin",
        "final_rmsnorm_s8.bin",
        "final_rmsnorm_scale_f64le.bin",
        "lm_head_output_s8.bin",
        "lm_head_output_scale_f64le.bin",
    }
)


def prune_transient_execution_artifacts(working: Path) -> dict[str, Any]:
    root = working.resolve()
    require(root.is_dir(), f"execution artifact root is absent: {public_path(root)}")
    removed_bytes = 0
    removed_files = 0
    removed_directories: list[str] = []
    for name in TRANSIENT_EXECUTION_DIRECTORIES:
        child = root / name
        if not child.exists():
            continue
        require(
            not child.is_symlink() and child.resolve().parent == root,
            f"transient execution artifact path escaped its root: {public_path(child)}",
        )
        for path in child.rglob("*"):
            if path.is_file():
                removed_bytes += path.stat().st_size
                removed_files += 1
        shutil.rmtree(child)
        removed_directories.append(name)
    retained_tensor_file_names: list[str] = []
    retained_tensor_bytes = 0
    tensor_directory = root / "tensors"
    if tensor_directory.exists():
        require(
            tensor_directory.is_dir()
            and not tensor_directory.is_symlink()
            and tensor_directory.resolve().parent == root,
            f"tensor artifact path escaped its root: {public_path(tensor_directory)}",
        )
        for child in tensor_directory.iterdir():
            require(
                child.is_file()
                and not child.is_symlink()
                and child.resolve().parent == tensor_directory.resolve(),
                f"tensor artifact is not a direct regular file: {public_path(child)}",
            )
            if child.name in RETAINED_COMPARISON_TENSOR_FILES:
                retained_tensor_file_names.append(child.name)
                retained_tensor_bytes += child.stat().st_size
                continue
            removed_bytes += child.stat().st_size
            removed_files += 1
            child.unlink()
        if not retained_tensor_file_names:
            tensor_directory.rmdir()
            removed_directories.append("tensors")
    removed_file_names: list[str] = []
    for name in TRANSIENT_EXECUTION_FILES:
        child = root / name
        if not child.exists():
            continue
        require(
            child.is_file()
            and not child.is_symlink()
            and child.resolve().parent == root,
            f"transient execution artifact path escaped its root: {public_path(child)}",
        )
        removed_bytes += child.stat().st_size
        removed_files += 1
        child.unlink()
        removed_file_names.append(name)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "status": "TRANSIENT_ARTIFACTS_PRUNED_AFTER_DURABLE_RESULT",
        "removed_directories": removed_directories,
        "removed_file_names": removed_file_names,
        "removed_files": removed_files,
        "removed_logical_bytes": removed_bytes,
        "retained_tensor_file_names": sorted(retained_tensor_file_names),
        "retained_tensor_files": len(retained_tensor_file_names),
        "retained_tensor_logical_bytes": retained_tensor_bytes,
        "retained_classes": [
            "result_json",
            "simulator_logs",
            "rtl_observed_bytes",
            "strict_oracle_comparison_payload_bytes",
        ],
    }


def _tree_usage(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "files": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
    }


def run_storage_lifecycle_construction_check(output: Path) -> dict[str, Any]:
    """Exercise two synthetic construction cycles without model or RTL execution."""
    output = output.resolve()
    require(
        not output.exists(),
        f"storage lifecycle check output already exists: {public_path(output)}",
    )
    slot = output / "construction-slot"
    slot.mkdir(parents=True)
    cycles: list[dict[str, Any]] = []
    expected_post_prune: dict[str, int] | None = None
    for cycle in range(2):
        write_text(slot / "logs/simulator.stdout.log", "retained simulator log\n")
        write_binary(slot / "rtl_observed/bytes.bin", bytes(range(32)))
        write_json(slot / "result.json", {"cycle": cycle, "status": "DURABLE"})
        for index, name in enumerate(TRANSIENT_EXECUTION_DIRECTORIES):
            write_binary(
                slot / name / "construction.bin",
                bytes([index]) * 1024,
            )
        write_binary(slot / "lm_head_weight.hex", b"0\n" * 2048)
        before = _tree_usage(slot)
        pruned = prune_transient_execution_artifacts(slot)
        after = _tree_usage(slot)
        require(
            pruned["removed_logical_bytes"] == 8192
            and pruned["removed_files"] == 5,
            "storage lifecycle check removed-byte accounting differs",
        )
        require(
            all(not (slot / name).exists() for name in TRANSIENT_EXECUTION_DIRECTORIES)
            and not (slot / "lm_head_weight.hex").exists(),
            "storage lifecycle check retained a reconstruction-only artifact",
        )
        if expected_post_prune is None:
            expected_post_prune = after
        require(
            after == expected_post_prune,
            "repeated construction increased the retained workspace footprint",
        )
        cycles.append(
            {
                "cycle": cycle,
                "before_prune": before,
                "after_prune": after,
                "pruning": pruned,
            }
        )
    idempotent = prune_transient_execution_artifacts(slot)
    require(
        idempotent["removed_files"] == 0
        and idempotent["removed_logical_bytes"] == 0,
        "storage lifecycle pruning is not idempotent",
    )
    result = {
        "schema_version": 1,
        "status": "PASS_CONSTRUCTION_ONLY_STORAGE_LIFECYCLE",
        "cycles": cycles,
        "stable_post_prune_usage": expected_post_prune,
        "idempotent_pruning": idempotent,
        "model_execution_count": 0,
        "rtl_execution_count": 0,
        "benchmark_execution_count": 0,
        "official_attempt_consumed": False,
        "claim_boundary": (
            "construction-only storage lifecycle check; no Stage-1 correctness, "
            "latency, PPA, or deployment claim"
        ),
    }
    write_json(output / "storage_lifecycle_check.json", result)
    return result


def write_text(path: Path, value: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return file_record(path)


def write_binary(path: Path, raw: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return file_record(path)


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes(order="C")


def raw_int8(value: torch.Tensor | list[int]) -> bytes:
    values = value.to(torch.int64).reshape(-1).tolist() if isinstance(value, torch.Tensor) else value
    return bytes(int(item) & 0xFF for item in values)


def raw_int16(values: Iterable[int], *, unsigned: bool = False) -> bytes:
    code = "<H" if unsigned else "<h"
    return b"".join(struct.pack(code, int(value)) for value in values)


def raw_int32(value: torch.Tensor | list[int]) -> bytes:
    values = value.to(torch.int64).reshape(-1).tolist() if isinstance(value, torch.Tensor) else value
    return b"".join(struct.pack("<i", int(item)) for item in values)


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def pack(values: Iterable[int], width: int) -> int:
    packed = 0
    mask = (1 << width) - 1
    for index, value in enumerate(values):
        packed |= (int(value) & mask) << (index * width)
    return packed


def hex_literal(value: int, width: int) -> str:
    return f"{width}'h{int(value) & ((1 << width) - 1):0{width // 4}x}"


def write_hex(path: Path, values: Iterable[int], width: int) -> dict[str, Any]:
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(value) & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )
    return file_record(path)


def requantize_rank1_input(
    input_q: torch.Tensor, source_scale: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    multiplier, right_shift = canonical.derive_multiplier(
        torch.tensor(source_scale / RANK1_INPUT_SCALE, dtype=torch.float64)
    )
    multiplier_s32 = int(multiplier.item())
    right_shift_u6 = int(right_shift.item())
    converted = [
        rank1_reference.saturate_s8(
            rank1_reference.round_shift_even_signed(
                int(value) * multiplier_s32, right_shift_u6
            )
        )[0]
        for value in input_q.to(torch.int64).reshape(-1).tolist()
    ]
    result = torch.tensor(converted, dtype=torch.int8).reshape(input_q.shape)
    return result, {
        "method": "accepted_0026_fixed_input_scale32_requantization",
        "source_scale": float(source_scale),
        "target_scale": RANK1_INPUT_SCALE,
        "multiplier_s32": multiplier_s32,
        "right_shift_u6": right_shift_u6,
        "source_sha256": sha256_bytes(raw_int8(input_q)),
        "converted_sha256": sha256_bytes(raw_int8(result)),
    }


def pack_rank1_w4(qweight: torch.Tensor) -> bytes:
    rows = qweight.to(torch.int16).contiguous()
    require(list(rows.shape) == [KV_HEADS * HEAD_DIM, HIDDEN], "rank-1 V qweight shape differs")
    require(bool(torch.all(rows >= -8)) and bool(torch.all(rows <= 7)), "rank-1 V qweight escaped int4")
    packed = (rows[:, 0::2] & 0xF) | ((rows[:, 1::2] & 0xF) << 4)
    return bytes(int(value) for value in packed.reshape(-1).tolist())


def pack_rank1_metadata(projection: dict[str, Any]) -> bytes:
    multipliers = projection["multiplier"].to(torch.int64).reshape(-1).tolist()
    shifts = projection["right_shift"].to(torch.int64).reshape(-1).tolist()
    require(
        len(multipliers) == KV_HEADS * HEAD_DIM
        and len(shifts) == KV_HEADS * HEAD_DIM,
        "rank-1 V metadata shape differs",
    )
    payload = bytearray()
    for multiplier, shift in zip(multipliers, shifts, strict=True):
        require(-(1 << 31) <= multiplier < (1 << 31), "rank-1 V multiplier escaped int32")
        require(0 <= shift <= 63, "rank-1 V shift escaped u6")
        record = bytearray(16)
        record[:4] = struct.pack("<i", multiplier)
        record[4] = shift
        payload.extend(record)
    return bytes(payload)


def parse_rank1_sidecar_stdout(stdout: str) -> dict[str, Any]:
    matches = list(RANK1_BEAT_PATTERN.finditer(stdout))
    require(len(matches) == 8, "rank-1 sidecar did not emit eight corrected-V beats")
    beats: dict[int, dict[str, Any]] = {}
    for match in matches:
        values = match.groupdict()
        beat = int(values["beat"])
        require(0 <= beat < 8 and beat not in beats, "rank-1 sidecar beat coverage differs")
        beats[beat] = {
            "rank_accumulator_s32": int(values["rank_acc"]),
            "rank_rounded_s32": int(values["rank_rounded"]),
            "rank_intermediate_s8": int(values["rank_s8"]),
            "baseline": bytes.fromhex(values["baseline"]),
            "corrected": bytes.fromhex(values["corrected"]),
        }
    rank_values = {
        (
            record["rank_accumulator_s32"],
            record["rank_rounded_s32"],
            record["rank_intermediate_s8"],
        )
        for record in beats.values()
    }
    require(len(rank_values) == 1, "rank-1 sidecar intermediates differ across beats")
    rank_accumulator, rank_rounded, rank_s8 = rank_values.pop()
    baseline = b"".join(bytes(reversed(beats[beat]["baseline"])) for beat in range(8))
    corrected = b"".join(bytes(reversed(beats[beat]["corrected"])) for beat in range(8))
    return {
        "rank_accumulator_s32": rank_accumulator,
        "rank_rounded_s32": rank_rounded,
        "rank_intermediate_s8": rank_s8,
        "baseline": baseline,
        "corrected": corrected,
    }


_SIMULATOR_TOOLS = {
    "iverilog": "/usr/bin/iverilog",
    "vvp": "/usr/bin/vvp",
}


def configure_simulator_tools(iverilog_path: str, vvp_path: str) -> None:
    global _SIMULATOR_TOOLS
    tools = {"iverilog": iverilog_path, "vvp": vvp_path}
    for name, value in tools.items():
        path = Path(value)
        require(path.is_absolute() and path.is_file(), f"absolute {name} tool path required")
    _SIMULATOR_TOOLS = tools


class Rank1SidecarSession:
    def __init__(self, working: Path) -> None:
        self.working = working
        self.working.mkdir(parents=True, exist_ok=False)
        self.binary = working / "sim/ace2_layer23_v_rank1_hybrid_shell.vvp"
        self.binary.parent.mkdir(parents=True)
        self.config = rank1_reference.load_frozen_config(check_review=False)
        command = [
            _SIMULATOR_TOOLS["iverilog"],
            "-g2012",
            "-Wall",
            "-Irtl",
            "-s",
            RANK1_TOP,
            "-o",
            runtime_path(self.binary),
            *[runtime_path(path) for path in RANK1_RTL_SOURCES],
            runtime_path(RANK1_TB),
        ]
        compiled = run_process(working, "rank1_sidecar.iverilog", command)
        require(compiled.returncode == 0, "rank-1 sidecar Icarus compile failed")
        self.compile_wall_seconds = float(
            json.loads(
                (working / "logs/rank1_sidecar.iverilog.result.json").read_text(
                    encoding="utf-8"
                )
            )["elapsed_wall_seconds"]
        )
        self.simulation_wall_seconds = 0.0
        self.compile = {
            "command": display_command(command),
            "binary": file_record(self.binary),
            "source": file_record(RANK1_SIDECAR),
            "testbench": file_record(RANK1_TB),
            "wall_seconds": self.compile_wall_seconds,
        }

    def apply(
        self,
        working: Path,
        activation_q: torch.Tensor,
        baseline_projection: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        working.mkdir(parents=True, exist_ok=False)
        baseline_q = baseline_projection["output_q"].to(torch.int8).reshape(-1)
        activation_values = activation_q.to(torch.int64).reshape(-1).tolist()
        baseline_values = baseline_q.to(torch.int64).tolist()
        reference = rank1_reference.apply_rank1_correction(
            activation_values, baseline_values, self.config
        )
        saturation_byte = (
            int(reference.rank_saturation)
            | (int(any(reference.correction_saturation)) << 1)
            | (int(any(reference.add_saturation)) << 2)
            | (
                int(
                    baseline_projection["saturation"]
                    .to(torch.bool)
                    .any()
                    .item()
                )
                << 4
            )
        )
        vectors = working / "vectors"
        artifacts = {
            "activation_s8": write_hex(vectors / "activation.hex", activation_values, 8),
            "v_weight_w4": write_hex(
                vectors / "v-weight.hex",
                pack_rank1_w4(baseline_projection["qweight"]),
                8,
            ),
            "v_metadata": write_hex(
                vectors / "v-metadata.hex", pack_rank1_metadata(baseline_projection), 8
            ),
            "baseline_v_s8": write_hex(vectors / "baseline-v.hex", baseline_values, 8),
            "software_corrected_v_reference_s8": write_hex(
                vectors / "corrected-v.hex", reference.corrected_v_s8, 8
            ),
            "rank_s32": write_hex(
                vectors / "rank.hex",
                [reference.rank_accumulator_s32, reference.rank_rounded_s32],
                32,
            ),
            "rank_s8": write_hex(
                vectors / "rank-s8.hex", [reference.rank_intermediate_s8], 8
            ),
            "saturation": write_hex(
                vectors / "saturation.hex", [saturation_byte], 8
            ),
        }
        replacements = {
            "activation": vectors / "activation.hex",
            "v_weight": vectors / "v-weight.hex",
            "v_meta": vectors / "v-metadata.hex",
            "baseline": vectors / "baseline-v.hex",
            "corrected": vectors / "corrected-v.hex",
            "rank": vectors / "rank.hex",
            "rank_s8": vectors / "rank-s8.hex",
            "saturation": vectors / "saturation.hex",
        }
        command = [
            _SIMULATOR_TOOLS["vvp"],
            runtime_path(self.binary),
            *[
                f"+{name.upper()}={runtime_path(path)}"
                for name, path in replacements.items()
            ],
        ]
        simulated = run_process(working, "rank1_sidecar.vvp", command)
        require(simulated.returncode == 0, "rank-1 sidecar Icarus execution failed")
        marker = RANK1_PASS_PATTERN.search(simulated.stdout)
        require(marker is not None, "rank-1 sidecar PASS marker is absent")
        metrics = {key: int(value) for key, value in marker.groupdict().items()}
        require(metrics["overflow"] == 0, "rank-1 sidecar numeric overflow observed")
        observed = parse_rank1_sidecar_stdout(simulated.stdout)
        expected_corrected = raw_int8(list(reference.corrected_v_s8))
        require(observed["baseline"] == raw_int8(baseline_q), "rank-1 sidecar baseline differs")
        require(observed["corrected"] == expected_corrected, "rank-1 sidecar corrected V differs")
        require(
            observed["rank_accumulator_s32"] == reference.rank_accumulator_s32
            and observed["rank_rounded_s32"] == reference.rank_rounded_s32
            and observed["rank_intermediate_s8"] == reference.rank_intermediate_s8,
            "rank-1 sidecar intermediates differ",
        )
        elapsed = float(
            json.loads(
                (working / "logs/rank1_sidecar.vvp.result.json").read_text(
                    encoding="utf-8"
                )
            )["elapsed_wall_seconds"]
        )
        self.simulation_wall_seconds += elapsed
        artifacts["rtl_emitted_corrected_v_s8"] = write_binary(
            working / "rtl_observed/corrected_v_s8.bin", observed["corrected"]
        )
        corrected = torch.tensor(
            [value - 256 if value & 0x80 else value for value in observed["corrected"]],
            dtype=torch.int8,
        )
        result = {
            "status": "PASS_ACCEPTED_0026_RANK1_SIDECAR",
            "v_source": "icarus_ace2_shell_mem_wdata_layer23_rank1_corrected_v",
            "baseline_v_sha256": sha256_bytes(observed["baseline"]),
            "corrected_v_sha256": sha256_bytes(observed["corrected"]),
            "corrected_v_bytes": len(observed["corrected"]),
            "rank_accumulator_s32": observed["rank_accumulator_s32"],
            "rank_rounded_s32": observed["rank_rounded_s32"],
            "rank_intermediate_s8": observed["rank_intermediate_s8"],
            "metrics": metrics,
            "command": display_command(command),
            "simulation_wall_seconds": elapsed,
            "stdout": file_record(working / "logs/rank1_sidecar.vvp.stdout.log"),
            "stderr": file_record(working / "logs/rank1_sidecar.vvp.stderr.log"),
            "artifacts": artifacts,
            "_rtl_corrected_v_s8": corrected.to(torch.int64).tolist(),
        }
        result_path = working / "rank1_execution.json"
        write_json(
            result_path,
            {key: value for key, value in result.items() if not key.startswith("_")},
        )
        result["persistence"] = prune_transient_execution_artifacts(working)
        write_json(
            result_path,
            {key: value for key, value in result.items() if not key.startswith("_")},
        )
        return corrected, result


def configure_snapshot(snapshot: Path) -> dict[str, Any]:
    """Retarget inherited read-only helpers to a portable resolved snapshot."""
    resolved = snapshot.resolve()
    model = resolved / "model.safetensors"
    config = resolved / "config.json"
    require(model.is_file(), f"model.safetensors is absent from {public_path(resolved)}")
    require(config.is_file(), f"config.json is absent from {public_path(resolved)}")
    require(sha256_file(model) == canonical.MODEL_SHA256, "portable snapshot model hash differs")

    canonical.SNAPSHOT = resolved
    canonical.MODEL = model
    canonical.CONFIG = config
    lm_head.SNAPSHOT = resolved
    lm_head.MODEL = model
    lm_head.CONFIG = config
    return {
        "snapshot": public_path(resolved),
        "model": file_record(model),
        "config": file_record(config),
    }


def empty_layer_cache() -> dict[str, Any]:
    return {"k": [], "v": [], "float_k": [], "float_v": []}


def embedding_state(weights: Any, token_id: int, position: int) -> dict[str, Any]:
    require(0 <= token_id < MODEL_OUTPUT_DOMAIN, f"embedding token id is outside model domain: {token_id}")
    hidden = weights.get_tensor("model.embed_tokens.weight")[token_id].contiguous()
    scale = canonical.scale_for(hidden.to(torch.float32))
    return {
        "position": position,
        "token_id": token_id,
        "fixed_q": canonical.quantize_int8(hidden, scale),
        "fixed_scale": scale,
        "float_hidden": hidden.to(torch.float32).contiguous(),
    }


def projection_with_template(
    name: str,
    merged: torch.Tensor,
    fixed_input: torch.Tensor,
    input_scale: float,
    float_input: torch.Tensor,
    template: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    result = frontier.projection_from_fixed_metadata(
        name,
        fixed_input,
        input_scale,
        template["qweight"],
        template["multiplier"],
        template["right_shift"],
        template["output_scale"],
    )
    result.update(
        {
            "float_output": torch.mv(merged, float_input.to(torch.float32)).contiguous(),
            "source_hashes": source_hashes,
            "weight_scale": template["weight_scale"],
        }
    )
    return result


def derive_layer_token(
    layer_id: int,
    state: dict[str, Any],
    cache: dict[str, Any],
    template: dict[str, Any] | None,
    weights: Any,
    adapter: Any,
    *,
    rank1_sidecar: Rank1SidecarSession | None = None,
    sidecar_working: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive one causal token at one layer and append that layer's K/V."""
    position_id = int(state["position"])
    require(position_id == len(cache["k"]), f"layer {layer_id} K cache length is not causal")
    require(position_id == len(cache["v"]), f"layer {layer_id} V cache length is not causal")
    require(position_id < MAX_CONTEXT_TOKENS, "attention context exceeds accepted RTL core bound")

    input_gain = weights.get_tensor(f"model.layers.{layer_id}.input_layernorm.weight").contiguous()
    post_gain = weights.get_tensor(f"model.layers.{layer_id}.post_attention_layernorm.weight").contiguous()
    float_input_norm = canonical.float_rmsnorm(state["float_hidden"], input_gain)
    if template is None:
        input_norm_scale = canonical.derive_rmsnorm_output_scale(
            input_gain.to(torch.float64).tolist(),
            float(float_input_norm.to(torch.float64).abs().max().item()),
        )
        input_gains = canonical.derive_scaled_gains_q8(
            input_gain.to(torch.float64).tolist(), input_norm_scale
        )
        template = {
            "input_norm_scale": input_norm_scale,
            "input_gains": input_gains,
            "qkv": {},
        }
    else:
        input_norm_scale = float(template["input_norm_scale"])
        input_gains = list(template["input_gains"])

    input_norm = canonical.reference_rmsnorm(
        state["fixed_q"].to(torch.int64).tolist(), input_gains
    )
    position: dict[str, Any] = {
        "layer_id": layer_id,
        "position": position_id,
        "token_id": int(state["token_id"]),
        "embedding": state["float_hidden"],
        "embedding_q": state["fixed_q"],
        "embedding_scale": float(state["fixed_scale"]),
        "input_gain": input_gain,
        "input_gains": input_gains,
        "input_norm": input_norm,
        "input_norm_q": torch.tensor(input_norm.outputs, dtype=torch.int8),
        "input_norm_scale": input_norm_scale,
        "projections": {},
    }

    for key, suffix in (
        ("q", "self_attn.q_proj"),
        ("k", "self_attn.k_proj"),
        ("v", "self_attn.v_proj"),
    ):
        tensor_name = f"model.layers.{layer_id}.{suffix}"
        merged, hashes = canonical.merge_projection(weights, adapter, tensor_name)
        if key not in template["qkv"]:
            source_projection = canonical.derive_projection(
                f"layer{layer_id}_position{position_id}_{key}",
                merged,
                position["input_norm_q"],
                input_norm_scale,
                float_input_norm,
                hashes,
            )
            projection = source_projection
            if key == "v" and layer_id == LAYERS - 1 and rank1_sidecar is not None:
                fixed_input, input_alignment = requantize_rank1_input(
                    position["input_norm_q"], input_norm_scale
                )
                multiplier, right_shift = canonical.derive_multiplier(
                    RANK1_INPUT_SCALE
                    * source_projection["weight_scale"].to(torch.float64)
                    / RANK1_RETAINED_V_SCALE
                )
                projection = frontier.projection_from_fixed_metadata(
                    f"layer{layer_id}_position{position_id}_{key}",
                    fixed_input,
                    RANK1_INPUT_SCALE,
                    source_projection["qweight"],
                    multiplier,
                    right_shift,
                    RANK1_RETAINED_V_SCALE,
                )
                projection.update(
                    {
                        "float_output": source_projection["float_output"],
                        "source_hashes": hashes,
                        "weight_scale": source_projection["weight_scale"],
                        "rank1_input_alignment": input_alignment,
                    }
                )
            template["qkv"][key] = {
                field: projection[field]
                for field in ("qweight", "multiplier", "right_shift", "output_scale", "weight_scale")
            }
        else:
            if key == "v" and layer_id == LAYERS - 1 and rank1_sidecar is not None:
                fixed_input, input_alignment = requantize_rank1_input(
                    position["input_norm_q"], input_norm_scale
                )
                projection = projection_with_template(
                    f"layer{layer_id}_position{position_id}_{key}",
                    merged,
                    fixed_input,
                    RANK1_INPUT_SCALE,
                    float_input_norm,
                    template["qkv"][key],
                    hashes,
                )
                projection["rank1_input_alignment"] = input_alignment
            else:
                projection = projection_with_template(
                    f"layer{layer_id}_position{position_id}_{key}",
                    merged,
                    position["input_norm_q"],
                    input_norm_scale,
                    float_input_norm,
                    template["qkv"][key],
                    hashes,
                )
        if key == "v" and layer_id == LAYERS - 1 and rank1_sidecar is not None:
            require(sidecar_working is not None, "rank-1 sidecar working path is absent")
            projection["baseline_output_q"] = projection["output_q"].clone()
            corrected, sidecar_record = rank1_sidecar.apply(
                sidecar_working,
                projection["input_q"],
                projection,
            )
            projection["output_q"] = corrected
            projection["rank1_sidecar"] = sidecar_record
        position["projections"][key] = projection
        del merged

    q_values = position["projections"]["q"]["output_q"].to(torch.int64).tolist()
    k_values = position["projections"]["k"]["output_q"].to(torch.int64).tolist()
    q_cos, q_sin = frontier.rope_coefficients(position_id, Q_HEADS)
    k_cos, k_sin = frontier.rope_coefficients(position_id, KV_HEADS)
    position["rope_q"] = canonical.reference_rope(
        canonical.RopeCase(
            f"layer{layer_id}_position{position_id}_q",
            position_id,
            q_values,
            [canonical.Q9_SCALE_ONE] * HIDDEN,
            q_cos,
            q_sin,
        )
    )
    position["rope_k"] = canonical.reference_rope(
        canonical.RopeCase(
            f"layer{layer_id}_position{position_id}_k",
            position_id,
            k_values,
            [canonical.Q9_SCALE_ONE] * (KV_HEADS * HEAD_DIM),
            k_cos,
            k_sin,
        )
    )
    position["rope_q_cos"] = q_cos
    position["rope_q_sin"] = q_sin
    position["rope_k_cos"] = k_cos
    position["rope_k_sin"] = k_sin
    if position_id == 0:
        require(position["rope_q"].outputs == q_values, "position-0 Q RoPE is not identity")
        require(position["rope_k"].outputs == k_values, "position-0 K RoPE is not identity")

    float_q = layer_runner.float_rope(
        position["projections"]["q"]["float_output"], position_id, Q_HEADS
    )
    float_k = layer_runner.float_rope(
        position["projections"]["k"]["float_output"], position_id, KV_HEADS
    )
    float_v = position["projections"]["v"]["float_output"]

    cache_before = len(cache["k"])
    cache["k"].append(list(position["rope_k"].outputs))
    cache["v"].append(
        position["projections"]["v"]["output_q"].to(torch.int64).tolist()
    )
    cache["float_k"].append(float_k)
    cache["float_v"].append(float_v)
    context = len(cache["k"])

    scores = []
    probabilities = []
    composes = []
    attention_values: list[int] = []
    cached_probability_heads = 0
    cached_k_effect_heads = 0
    cached_v_effect_heads = 0
    for head in range(Q_HEADS):
        kv_head = head // (Q_HEADS // KV_HEADS)
        start = kv_head * HEAD_DIM
        stop = start + HEAD_DIM
        q_head = position["rope_q"].outputs[head * HEAD_DIM : (head + 1) * HEAD_DIM]
        keys = [row[start:stop] for row in cache["k"]]
        values = [row[start:stop] for row in cache["v"]]
        score = canonical.reference_attention_score(
            canonical.AttentionScoreCase(
                f"layer{layer_id}_position{position_id}_head{head}",
                q_head,
                keys,
                float(template["qkv"]["q"]["output_scale"]),
                float(template["qkv"]["k"]["output_scale"]),
            )
        )
        softmax = canonical.reference_softmax(
            canonical.SoftmaxCase(
                f"layer{layer_id}_position{position_id}_head{head}",
                score.core_scores_q6_9,
            ),
            context_max=MAX_CONTEXT_TOKENS,
        )
        compose = canonical.reference_attention_compose(
            canonical.AttentionComposeCase(
                f"layer{layer_id}_position{position_id}_head{head}",
                score.core_scores_q6_9,
                values,
            )
        )
        if position_id > 0:
            cached_probability_heads += int(
                any(softmax.probabilities_q0_15[:position_id])
            )
            zero_keys = [[0] * HEAD_DIM for _ in range(position_id)] + [keys[-1]]
            zero_score = canonical.reference_attention_score(
                canonical.AttentionScoreCase(
                    f"layer{layer_id}_position{position_id}_head{head}_zero_cached_k",
                    q_head,
                    zero_keys,
                    float(template["qkv"]["q"]["output_scale"]),
                    float(template["qkv"]["k"]["output_scale"]),
                )
            )
            cached_k_effect_heads += int(zero_score.accumulators != score.accumulators)
            zero_values = [[0] * HEAD_DIM for _ in range(position_id)] + [values[-1]]
            zero_compose = canonical.reference_attention_compose(
                canonical.AttentionComposeCase(
                    f"layer{layer_id}_position{position_id}_head{head}_zero_cached_v",
                    score.core_scores_q6_9,
                    zero_values,
                )
            )
            cached_v_effect_heads += int(zero_compose.outputs != compose.outputs)
        scores.append(score)
        probabilities.append(softmax)
        composes.append(compose)
        attention_values.extend(compose.outputs)

    if position_id > 0:
        require(cached_probability_heads > 0, f"layer {layer_id} decode assigns no cached probability")
        require(cached_k_effect_heads > 0, f"layer {layer_id} decode has no cached-K data effect")
        require(cached_v_effect_heads > 0, f"layer {layer_id} decode has no cached-V data effect")

    position["scores"] = scores
    position["probabilities"] = probabilities
    position["composes"] = composes
    position["attention_q"] = torch.tensor(attention_values, dtype=torch.int8)

    merged_tail: dict[str, tuple[torch.Tensor, dict[str, str]]] = {}
    for key, suffix in (
        ("o", "self_attn.o_proj"),
        ("gate", "mlp.gate_proj"),
        ("up", "mlp.up_proj"),
        ("down", "mlp.down_proj"),
    ):
        merged_tail[key] = canonical.merge_projection(
            weights, adapter, f"model.layers.{layer_id}.{suffix}"
        )

    float_attention = layer_runner.float_attention(
        position_id, float_q, cache["float_k"], cache["float_v"]
    )
    o_merged, o_hashes = merged_tail["o"]
    position["projections"]["o"] = canonical.derive_projection(
        f"layer{layer_id}_position{position_id}_o",
        o_merged,
        position["attention_q"],
        position["projections"]["v"]["output_scale"],
        float_attention,
        o_hashes,
    )

    float_attention_residual = state["float_hidden"] + position["projections"]["o"]["float_output"]
    attention_residual_scale = canonical.scale_for(float_attention_residual)
    if layer_id == 0:
        fixed_residual_source = canonical.quantize_int8(
            state["float_hidden"], attention_residual_scale
        )
    else:
        fixed_residual_source = canonical.quantize_int8(
            state["fixed_q"].to(torch.float64) * state["fixed_scale"],
            attention_residual_scale,
        )
    fixed_residual_o = canonical.quantize_int8(
        position["projections"]["o"]["output_q"].to(torch.float64)
        * position["projections"]["o"]["output_scale"],
        attention_residual_scale,
    )
    attention_values_fixed, attention_saturation = canonical.reference_residual_add(
        fixed_residual_source.to(torch.int64).tolist(),
        fixed_residual_o.to(torch.int64).tolist(),
    )
    attention_q = torch.tensor(attention_values_fixed, dtype=torch.int8)
    position["attention_residual"] = {
        "lhs": fixed_residual_source,
        "rhs": fixed_residual_o,
        "output": attention_q,
        "scale": attention_residual_scale,
        "saturation": attention_saturation,
    }

    float_post_norm = canonical.float_rmsnorm(float_attention_residual, post_gain)
    post_norm_scale = canonical.derive_rmsnorm_output_scale(
        post_gain.to(torch.float64).tolist(),
        float(float_post_norm.to(torch.float64).abs().max().item()),
    )
    post_gains = canonical.derive_scaled_gains_q8(
        post_gain.to(torch.float64).tolist(), post_norm_scale
    )
    post_norm = canonical.reference_rmsnorm(attention_values_fixed, post_gains)
    post_norm_q = torch.tensor(post_norm.outputs, dtype=torch.int8)
    position.update(
        {
            "post_gain": post_gain,
            "post_gains": post_gains,
            "post_norm": post_norm,
            "post_norm_q": post_norm_q,
            "post_norm_scale": post_norm_scale,
        }
    )

    for key in ("gate", "up"):
        merged, hashes = merged_tail[key]
        position["projections"][key] = canonical.derive_projection(
            f"layer{layer_id}_position{position_id}_{key}",
            merged,
            post_norm_q,
            post_norm_scale,
            float_post_norm,
            hashes,
        )

    float_silu = (
        F.silu(position["projections"]["gate"]["float_output"])
        * position["projections"]["up"]["float_output"]
    )
    gate_q6 = torch.round(
        position["projections"]["gate"]["output_q"].to(torch.float64)
        * position["projections"]["gate"]["output_scale"]
        * (1 << 9)
    ).clamp(-32768, 32767).to(torch.int16)
    up_q6 = torch.round(
        position["projections"]["up"]["output_q"].to(torch.float64)
        * position["projections"]["up"]["output_scale"]
        * (1 << 9)
    ).clamp(-32768, 32767).to(torch.int16)
    silu_scale = canonical.scale_for(float_silu)
    silu_multiplier, silu_shift = canonical.derive_multiplier(
        torch.tensor([1.0 / (silu_scale * (1 << 21))], dtype=torch.float64)
    )
    silu_result = canonical.reference_silu_gate(
        canonical.SiluGateCase(
            f"layer{layer_id}_position{position_id}_silu",
            gate_q6.to(torch.int64).tolist(),
            up_q6.to(torch.int64).tolist(),
            int(silu_multiplier[0]),
            int(silu_shift[0]),
            0,
        )
    )
    silu_q = torch.tensor(silu_result.outputs, dtype=torch.int8)
    position["silu"] = {
        "gate_q6": gate_q6,
        "up_q6": up_q6,
        "output": silu_q,
        "scale": silu_scale,
        "multiplier": int(silu_multiplier[0]),
        "right_shift": int(silu_shift[0]),
        "result": silu_result,
    }

    down_merged, down_hashes = merged_tail["down"]
    position["projections"]["down"] = canonical.derive_projection(
        f"layer{layer_id}_position{position_id}_down",
        down_merged,
        silu_q,
        silu_scale,
        float_silu,
        down_hashes,
    )
    float_layer_output = float_attention_residual + position["projections"]["down"]["float_output"]
    layer_output_scale = canonical.scale_for(float_layer_output)
    residual_stream = canonical.quantize_int8(
        attention_q.to(torch.float64) * attention_residual_scale,
        layer_output_scale,
    )
    residual_down = canonical.quantize_int8(
        position["projections"]["down"]["output_q"].to(torch.float64)
        * position["projections"]["down"]["output_scale"],
        layer_output_scale,
    )
    layer_values, layer_saturation = canonical.reference_residual_add(
        residual_down.to(torch.int64).tolist(), residual_stream.to(torch.int64).tolist()
    )
    layer_q = torch.tensor(layer_values, dtype=torch.int8)
    position["final_residual"] = {
        "down": residual_down,
        "stream": residual_stream,
        "output": layer_q,
        "scale": layer_output_scale,
        "saturation": layer_saturation,
    }
    position["float_layer_output"] = float_layer_output
    error = layer_q.to(torch.float64) * layer_output_scale - float_layer_output.to(torch.float64)
    denominator = torch.linalg.vector_norm(float_layer_output.to(torch.float64))
    position["metrics"] = {
        "max_abs_dequantized_error": float(error.abs().max().item()),
        "mean_abs_dequantized_error": float(error.abs().mean().item()),
        "relative_l2_error": (
            float(torch.linalg.vector_norm(error) / denominator) if float(denominator) else 0.0
        ),
        "float_output_absmax": float(float_layer_output.to(torch.float64).abs().max().item()),
        "layer_output_scale": layer_output_scale,
    }

    next_state = {
        "position": position_id,
        "token_id": int(state["token_id"]),
        "fixed_q": layer_q,
        "fixed_scale": layer_output_scale,
        "float_hidden": float_layer_output,
    }
    derived = {
        "layer_id": layer_id,
        "token_ids": [int(state["token_id"])],
        "positions": [position],
        "cached_k": cache["k"],
        "cached_v": cache["v"],
        "cache_reuse": {
            "context_length_before": cache_before,
            "context_length_after": context,
            "decode_heads": Q_HEADS if position_id > 0 else 0,
            "cached_probability_nonzero_heads": cached_probability_heads,
            "cached_k_accumulator_effect_heads": cached_k_effect_heads,
            "cached_v_compose_effect_heads": cached_v_effect_heads,
        },
    }
    return derived, next_state, template


def render_layer_rmsnorm_vectors(position: dict[str, Any]) -> str:
    cases = [
        (position["embedding_q"].to(torch.int64).tolist(), position["input_gains"], position["input_norm"]),
        (
            position["attention_residual"]["output"].to(torch.int64).tolist(),
            position["post_gains"],
            position["post_norm"],
        ),
    ]
    lines = [
        "// One streamed position: input and post-attention RMSNorm.",
        "localparam integer TEST_COUNT = 2;",
        "localparam integer TEST_BEATS = 56;",
        "reg [8*16-1:0] test_input_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [16*16-1:0] test_gain_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [8*16-1:0] test_expected_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [47:0] test_expected_sumsq [0:TEST_COUNT-1];",
        "reg [31:0] test_expected_inv [0:TEST_COUNT-1];",
        "reg test_expected_saturation [0:TEST_COUNT-1];",
        "initial begin",
    ]
    for case, (inputs, gains, result) in enumerate(cases):
        lines += [
            f"  test_expected_sumsq[{case}] = {hex_literal(result.sumsq, 48)};",
            f"  test_expected_inv[{case}] = {hex_literal(result.inv_rms_q30, 32)};",
            f"  test_expected_saturation[{case}] = 1'b{int(result.saturation_seen)};",
        ]
        for beat, (input_word, gain_word, output_word) in enumerate(
            zip(
                canonical.pack_int8_beats(inputs),
                canonical.pack_gain_beats(gains),
                canonical.pack_int8_beats(result.outputs),
                strict=True,
            )
        ):
            flat = case * 56 + beat
            lines += [
                f"  test_input_beats[{flat}] = {hex_literal(input_word, 128)};",
                f"  test_gain_beats[{flat}] = {hex_literal(gain_word, 256)};",
                f"  test_expected_beats[{flat}] = {hex_literal(output_word, 128)};",
            ]
    return "\n".join(lines + ["end", ""])


def render_rope_vectors(position: dict[str, Any]) -> str:
    k_input = position["projections"]["k"]["output_q"].to(torch.int64).tolist()
    k_input += [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
    k_cos = position["rope_k_cos"] + [canonical.Q15_ONE] * (HIDDEN - KV_HEADS * HEAD_DIM)
    k_sin = position["rope_k_sin"] + [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
    k_output = position["rope_k"].outputs + [0] * (HIDDEN - KV_HEADS * HEAD_DIM)
    cases = [
        (
            position["position"],
            position["projections"]["q"]["output_q"].to(torch.int64).tolist(),
            position["rope_q_cos"],
            position["rope_q_sin"],
            position["rope_q"].outputs,
            position["rope_q"],
        ),
        (position["position"], k_input, k_cos, k_sin, k_output, position["rope_k"]),
    ]
    lines = [
        "// One streamed absolute-position Q/K RoPE pair.",
        "localparam integer ROPE_CASE_COUNT = 2;",
        "localparam integer ROPE_HIDDEN_SIZE = 896;",
        "localparam integer ROPE_LANES = 16;",
        "localparam integer ROPE_BEATS = 56;",
        "reg [15:0] rope_sequence_position [0:ROPE_CASE_COUNT-1];",
        "reg rope_expected_saturation [0:ROPE_CASE_COUNT-1];",
        "reg rope_expected_saturation_by_beat [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "reg [127:0] rope_input_beats [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "reg [127:0] rope_scale_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_cos_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_sin_beats [0:ROPE_CASE_COUNT*ROPE_BEATS*2-1];",
        "reg [127:0] rope_expected_beats [0:ROPE_CASE_COUNT*ROPE_BEATS-1];",
        "initial begin",
    ]
    for case, (sequence_position, inputs, cosines, sines, outputs, result) in enumerate(cases):
        beat_saturation = frontier.rope_saturation_by_beat(inputs, cosines, sines)
        require(any(beat_saturation) == result.saturation_seen, "RoPE saturation reduction differs")
        lines += [
            f"  rope_sequence_position[{case}] = 16'd{sequence_position};",
            f"  rope_expected_saturation[{case}] = 1'b{int(result.saturation_seen)};",
        ]
        for beat in range(56):
            flat = case * 56 + beat
            lines += [
                f"  rope_input_beats[{flat}] = {hex_literal(pack(inputs[beat*16:(beat+1)*16], 8), 128)};",
                f"  rope_expected_beats[{flat}] = {hex_literal(pack(outputs[beat*16:(beat+1)*16], 8), 128)};",
                f"  rope_expected_saturation_by_beat[{flat}] = 1'b{int(beat_saturation[beat])};",
            ]
            for half in range(2):
                table = case * 112 + beat * 2 + half
                start = beat * 16 + half * 8
                lines += [
                    f"  rope_scale_beats[{table}] = {hex_literal(pack([canonical.Q9_SCALE_ONE] * 8, 16), 128)};",
                    f"  rope_cos_beats[{table}] = {hex_literal(pack(cosines[start:start+8], 16), 128)};",
                    f"  rope_sin_beats[{table}] = {hex_literal(pack(sines[start:start+8], 16), 128)};",
                ]
    return "\n".join(lines + ["end", ""])


def render_score_vectors(derived: dict[str, Any]) -> str:
    position = derived["positions"][0]
    context = len(derived["cached_k"])
    require(1 <= context <= MAX_CONTEXT_TOKENS, "score context exceeds backend bound")
    word_width = MAX_CONTEXT_TOKENS * 16
    lines = [
        "// Fourteen streamed decode-head score cases.",
        "localparam integer ATTN_SCORE_CASE_COUNT = 14;",
        "localparam integer ATTN_SCORE_HEAD_DIM = 64;",
        "localparam integer ATTN_SCORE_MAC_LANES = 1;",
        f"localparam integer ATTN_SCORE_CONTEXT_MAX = {MAX_CONTEXT_TOKENS};",
        "localparam integer ATTN_SCORE_BEATS_PER_VECTOR = 4;",
        "localparam [127:0] ATTN_SCORE_INVALID_NEGATIVE_MULTIPLIER = 128'd0;",
        "localparam [127:0] ATTN_SCORE_INVALID_RESERVED_BITS = 128'd0;",
        "reg [15:0] attn_score_context_count [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg signed [31:0] attn_score_multiplier [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [5:0] attn_score_right_shift [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg attn_score_expected_saturation [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg attn_score_expected_saturation_token [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg attn_score_expected_core_saturation_token [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg [31:0] attn_score_expected_acc [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX-1];",
        "reg [ATTN_SCORE_CONTEXT_MAX*16-1:0] attn_score_expected_word [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [ATTN_SCORE_CONTEXT_MAX*16-1:0] attn_score_expected_core_word [0:ATTN_SCORE_CASE_COUNT-1];",
        "reg [127:0] attn_score_q_beats [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_BEATS_PER_VECTOR-1];",
        "reg [127:0] attn_score_k_beats [0:ATTN_SCORE_CASE_COUNT*ATTN_SCORE_CONTEXT_MAX*ATTN_SCORE_BEATS_PER_VECTOR-1];",
        "initial begin",
    ]
    for head, score in enumerate(position["scores"]):
        kv_head = head // (Q_HEADS // KV_HEADS)
        q_head = position["rope_q"].outputs[head * HEAD_DIM : (head + 1) * HEAD_DIM]
        lines += [
            f"  attn_score_context_count[{head}] = 16'd{context};",
            f"  attn_score_multiplier[{head}] = 32'sd{score.multiplier};",
            f"  attn_score_right_shift[{head}] = 6'd{score.right_shift};",
            f"  attn_score_expected_saturation[{head}] = 1'b{int(score.saturation_seen)};",
            f"  attn_score_expected_word[{head}] = {hex_literal(pack(score.scores_q6_9 + [0] * (MAX_CONTEXT_TOKENS-context), 16), word_width)};",
            f"  attn_score_expected_core_word[{head}] = {hex_literal(pack(score.core_scores_q6_9 + [0] * (MAX_CONTEXT_TOKENS-context), 16), word_width)};",
        ]
        for beat in range(4):
            lines.append(
                f"  attn_score_q_beats[{head*4+beat}] = {hex_literal(pack(q_head[beat*16:(beat+1)*16], 8), 128)};"
            )
        for token in range(MAX_CONTEXT_TOKENS):
            flat = head * MAX_CONTEXT_TOKENS + token
            real = token < context
            lines += [
                f"  attn_score_expected_acc[{flat}] = {hex_literal(score.accumulators[token] if real else 0, 32)};",
                f"  attn_score_expected_saturation_token[{flat}] = 1'b{int(score.saturation_by_token[token]) if real else 0};",
                f"  attn_score_expected_core_saturation_token[{flat}] = 1'b{int(score.core_saturation_by_token[token]) if real else 0};",
            ]
            key = (
                derived["cached_k"][token][kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
                if real
                else [0] * HEAD_DIM
            )
            for beat in range(4):
                lines.append(
                    f"  attn_score_k_beats[{head*MAX_CONTEXT_TOKENS*4+token*4+beat}] = {hex_literal(pack(key[beat*16:(beat+1)*16], 8), 128)};"
                )
    return "\n".join(lines + ["end", ""])


def render_softmax_vectors(derived: dict[str, Any]) -> str:
    position = derived["positions"][0]
    context = len(derived["cached_k"])
    require(1 <= context <= MAX_CONTEXT_TOKENS, "softmax context exceeds backend bound")
    word_width = MAX_CONTEXT_TOKENS * 16
    exp_sum_width = 15 + math.ceil(math.log2(MAX_CONTEXT_TOKENS + 1))
    lines = [
        "// Fourteen streamed causal softmax cases.",
        "localparam integer SOFTMAX_CASE_COUNT = 14;",
        f"localparam integer SOFTMAX_CONTEXT_MAX = {MAX_CONTEXT_TOKENS};",
        "localparam integer SOFTMAX_EXP_SUM_WIDTH = 15 + $clog2(SOFTMAX_CONTEXT_MAX + 1);",
        "reg [15:0] softmax_context_count [0:SOFTMAX_CASE_COUNT-1];",
        "reg [SOFTMAX_CONTEXT_MAX*16-1:0] softmax_score_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [SOFTMAX_CONTEXT_MAX*16-1:0] softmax_expected_exp_word [0:SOFTMAX_CASE_COUNT-1];",
        "reg [SOFTMAX_EXP_SUM_WIDTH-1:0] softmax_expected_exp_sum [0:SOFTMAX_CASE_COUNT-1];",
        "reg [SOFTMAX_CONTEXT_MAX*16-1:0] softmax_expected_word [0:SOFTMAX_CASE_COUNT-1];",
        "initial begin",
    ]
    for head, (score, softmax) in enumerate(zip(position["scores"], position["probabilities"], strict=True)):
        lines += [
            f"  softmax_context_count[{head}] = 16'd{context};",
            f"  softmax_score_word[{head}] = {hex_literal(pack(score.core_scores_q6_9 + [0] * (MAX_CONTEXT_TOKENS-context), 16), word_width)};",
            f"  softmax_expected_exp_word[{head}] = {hex_literal(pack(softmax.exp_weights_q15, 16), word_width)};",
            f"  softmax_expected_exp_sum[{head}] = {exp_sum_width}'d{softmax.exp_sum_q15};",
            f"  softmax_expected_word[{head}] = {hex_literal(pack(softmax.probabilities_q0_15, 16), word_width)};",
        ]
    return "\n".join(lines + ["end", ""])


def render_compose_vectors(derived: dict[str, Any]) -> str:
    position = derived["positions"][0]
    context = len(derived["cached_v"])
    require(1 <= context <= MAX_CONTEXT_TOKENS, "compose context exceeds backend bound")
    protocol_context = max(9, context + 1)
    protocol_tiles = (protocol_context + 7) // 8
    lines = [
        "// Fourteen streamed attention-compose cases with exact masked padding.",
        "localparam integer ATTN_COMPOSE_CASE_COUNT = 14;",
        "localparam integer ATTN_COMPOSE_TILE = 8;",
        f"localparam integer ATTN_COMPOSE_MAX_CONTEXT = {protocol_context};",
        f"localparam integer ATTN_COMPOSE_MAX_TILES = {protocol_tiles};",
        "localparam integer ATTN_COMPOSE_BEATS = 4;",
        "reg [15:0] attn_compose_context_count [0:ATTN_COMPOSE_CASE_COUNT-1];",
        "reg [127:0] attn_compose_score_tiles [0:ATTN_COMPOSE_CASE_COUNT*ATTN_COMPOSE_MAX_TILES-1];",
        "reg [127:0] attn_compose_value_beats [0:ATTN_COMPOSE_CASE_COUNT*ATTN_COMPOSE_MAX_CONTEXT*ATTN_COMPOSE_BEATS-1];",
        "reg [127:0] attn_compose_expected_beats [0:ATTN_COMPOSE_CASE_COUNT*ATTN_COMPOSE_BEATS-1];",
        "reg attn_compose_expected_saturation [0:ATTN_COMPOSE_CASE_COUNT-1];",
        "initial begin",
    ]
    for head, (score, compose) in enumerate(zip(position["scores"], position["composes"], strict=True)):
        kv_head = head // (Q_HEADS // KV_HEADS)
        real_values = [
            row[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
            for row in derived["cached_v"]
        ]
        protocol_scores = score.core_scores_q6_9 + [-32768] * (protocol_context - context)
        protocol_values = real_values + [
            [0] * HEAD_DIM for _ in range(protocol_context - context)
        ]
        protocol = canonical.reference_attention_compose(
            canonical.AttentionComposeCase(
                f"stream_protocol_head{head}", protocol_scores, protocol_values
            )
        )
        require(protocol.outputs == compose.outputs, "compose padding alters real output")
        lines += [
            f"  attn_compose_context_count[{head}] = 16'd{protocol_context};",
            f"  attn_compose_expected_saturation[{head}] = 1'b{int(protocol.saturation_seen)};",
        ]
        for tile in range(protocol_tiles):
            tile_scores = protocol_scores[tile * 8 : (tile + 1) * 8]
            tile_scores += [0] * (8 - len(tile_scores))
            lines.append(
                f"  attn_compose_score_tiles[{head*protocol_tiles+tile}] = "
                f"{hex_literal(pack(tile_scores, 16), 128)};"
            )
        for token, row in enumerate(protocol_values):
            for beat in range(4):
                flat = head * protocol_context * 4 + token * 4 + beat
                lines.append(
                    f"  attn_compose_value_beats[{flat}] = {hex_literal(pack(row[beat*16:(beat+1)*16], 8), 128)};"
                )
        for beat in range(4):
            lines.append(
                f"  attn_compose_expected_beats[{head*4+beat}] = {hex_literal(pack(compose.outputs[beat*16:(beat+1)*16], 8), 128)};"
            )
    return "\n".join(lines + ["end", ""])


def render_silu_vectors(position: dict[str, Any]) -> str:
    silu = position["silu"]
    lines = [
        "// One streamed SiLU-gate case.",
        "localparam integer SILU_CASE_COUNT = 1;",
        "localparam integer SILU_MAX_INPUT_BEATS = 608;",
        "localparam integer SILU_MAX_OUTPUT_BEATS = 304;",
        "localparam [5:0] SILU_REQUIRED_BOUNDARY_COVERAGE = 6'd0;",
        "reg [15:0] silu_case_length [0:SILU_CASE_COUNT-1];",
        "reg signed [31:0] silu_case_multiplier [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_right_shift [0:SILU_CASE_COUNT-1];",
        "reg signed [7:0] silu_case_zero_point [0:SILU_CASE_COUNT-1];",
        "reg silu_expected_saturation [0:SILU_CASE_COUNT-1];",
        "reg [5:0] silu_case_boundary_coverage [0:SILU_CASE_COUNT-1];",
        "reg [127:0] silu_gate_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_up_beats [0:SILU_CASE_COUNT*SILU_MAX_INPUT_BEATS-1];",
        "reg [127:0] silu_expected_beats [0:SILU_CASE_COUNT*SILU_MAX_OUTPUT_BEATS-1];",
        "initial begin",
        f"  silu_case_length[0] = 16'd{INTERMEDIATE};",
        f"  silu_case_multiplier[0] = 32'sd{silu['multiplier']};",
        f"  silu_case_right_shift[0] = 6'd{silu['right_shift']};",
        "  silu_case_zero_point[0] = 8'sd0;",
        f"  silu_expected_saturation[0] = 1'b{int(silu['result'].saturation_seen)};",
        "  silu_case_boundary_coverage[0] = 6'd0;",
    ]
    gate = silu["gate_q6"].to(torch.int64).tolist()
    up = silu["up_q6"].to(torch.int64).tolist()
    output = silu["output"].to(torch.int64).tolist()
    for beat in range(608):
        lines += [
            f"  silu_gate_beats[{beat}] = {hex_literal(pack(gate[beat*8:(beat+1)*8], 16), 128)};",
            f"  silu_up_beats[{beat}] = {hex_literal(pack(up[beat*8:(beat+1)*8], 16), 128)};",
        ]
    for beat, group in enumerate(chunks(output, 16)):
        lines.append(f"  silu_expected_beats[{beat}] = {hex_literal(pack(group, 8), 128)};")
    return "\n".join(lines + ["end", ""])


def render_residual_vectors(position: dict[str, Any], *, final: bool) -> str:
    prefix = "mlp_residual" if final else "residual"
    upper = "MLP_RESIDUAL" if final else "RESIDUAL"
    lhs_name = "down" if final else "lhs"
    rhs_name = "stream" if final else "rhs"
    values = position["final_residual"] if final else position["attention_residual"]
    lhs = values[lhs_name].to(torch.int64).tolist()
    rhs = values[rhs_name].to(torch.int64).tolist()
    output = values["output"].to(torch.int64).tolist()
    case_count = 2 if final else 1
    lines = [
        "// One streamed residual payload; final residual duplicates retained case 1.",
        f"localparam integer {upper}_CASE_COUNT = {case_count};",
        f"localparam integer {upper}_BEATS = 56;",
        f"reg [127:0] {prefix}_{lhs_name}_beats [0:{upper}_CASE_COUNT*{upper}_BEATS-1];",
        f"reg [127:0] {prefix}_{rhs_name}_beats [0:{upper}_CASE_COUNT*{upper}_BEATS-1];",
        f"reg [127:0] {prefix}_expected_beats [0:{upper}_CASE_COUNT*{upper}_BEATS-1];",
        f"reg {prefix}_expected_saturation [0:{upper}_CASE_COUNT-1];",
        "initial begin",
    ]
    payloads = list(zip(chunks(lhs, 16), chunks(rhs, 16), chunks(output, 16), strict=True))
    for case_index in range(case_count):
        lines.append(
            f"  {prefix}_expected_saturation[{case_index}] = "
            f"1'b{int(values['saturation'])};"
        )
        for beat, (lhs_group, rhs_group, out_group) in enumerate(payloads):
            flat = case_index * 56 + beat
            lines += [
                f"  {prefix}_{lhs_name}_beats[{flat}] = {hex_literal(pack(lhs_group, 8), 128)};",
                f"  {prefix}_{rhs_name}_beats[{flat}] = {hex_literal(pack(rhs_group, 8), 128)};",
                f"  {prefix}_expected_beats[{flat}] = {hex_literal(pack(out_group, 8), 128)};",
            ]
    return "\n".join(lines + ["end", ""])


def persist_layer_vectors(working: Path, derived: dict[str, Any]) -> dict[str, Any]:
    position = derived["positions"][0]
    directory = working / "vectors"
    activation_words: list[int] = []
    multipliers: list[int] = []
    shifts: list[int] = []
    outputs: list[int] = []
    accumulators: list[int] = []
    saturation: list[int] = []
    for key in PROJECTION_ORDER:
        projection = position["projections"][key]
        activation_words.extend(
            pack(group, 8)
            for group in chunks(projection["input_q"].to(torch.int64).tolist(), 4)
        )
        multipliers.extend(projection["multiplier"].to(torch.int64).tolist())
        shifts.extend(projection["right_shift"].to(torch.int64).tolist())
        projection_output = projection.get("baseline_output_q", projection["output_q"])
        outputs.extend(projection_output.to(torch.int64).tolist())
        accumulators.extend(projection["accumulator"].to(torch.int64).tolist())
        saturation.extend(projection["saturation"].to(torch.int64).tolist())
    weight_words: list[int] = []
    for key in PROJECTION_ORDER:
        for row in position["projections"][key]["qweight"].to(torch.int64):
            weight_words.extend(pack(group, 4) for group in chunks(row.tolist(), 4))
    return {
        "projection_activation": write_hex(directory / "projection_activation.hex", activation_words, 32),
        "projection_weight": write_hex(directory / "projection_weight.hex", weight_words, 16),
        "projection_multiplier": write_hex(directory / "projection_multiplier.hex", multipliers, 32),
        "projection_shift": write_hex(directory / "projection_shift.hex", shifts, 8),
        "projection_expected": write_hex(directory / "projection_expected.hex", outputs, 8),
        "projection_accumulator": write_hex(directory / "projection_accumulator.hex", accumulators, 32),
        "projection_saturation": write_hex(directory / "projection_saturation.hex", saturation, 8),
        "rmsnorm": write_text(directory / "rmsnorm_vectors.svh", render_layer_rmsnorm_vectors(position)),
        "rope": write_text(directory / "rope_vectors.svh", render_rope_vectors(position)),
        "score": write_text(directory / "attention_score_vectors.svh", render_score_vectors(derived)),
        "softmax": write_text(directory / "softmax_vectors.svh", render_softmax_vectors(derived)),
        "compose": write_text(directory / "attention_compose_vectors.svh", render_compose_vectors(derived)),
        "silu": write_text(directory / "silu_gate_vectors.svh", render_silu_vectors(position)),
        "residual": write_text(directory / "residual_vectors.svh", render_residual_vectors(position, final=False)),
        "mlp_residual": write_text(
            directory / "mlp_residual_vectors.svh", render_residual_vectors(position, final=True)
        ),
    }


def instrument_cycles(source: str, family: str, instance_needle: str) -> str:
    require(instance_needle in source, f"{family} RTL instance insertion point changed")
    counter = (
        "    integer ace2_generation_cycle_count;\n"
        "    initial ace2_generation_cycle_count = 0;\n"
        "    always @(posedge clk) ace2_generation_cycle_count <= ace2_generation_cycle_count + 1;\n\n"
    )
    source = source.replace(instance_needle, counter + instance_needle, 1)
    finish = source.rfind("$finish;")
    require(finish >= 0, f"{family} testbench finish marker is absent")
    return (
        source[:finish]
        + f'$display("ACE2_GENERATION_SIM_CYCLES family={family} cycles=%0d", ace2_generation_cycle_count);\n        '
        + source[finish:]
    )


def projection_execution_source(working: Path) -> Path:
    source = layer_runner.PROJECTION_TB.read_text(encoding="utf-8")
    frozen = "evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors"
    source = source.replace(frozen, runtime_path(working / "vectors"))
    source = instrument_cycles(source, "projection", "    ace2_w4a8_proj_core #(")
    path = working / "execution_sources/projection_tb.sv"
    write_text(path, source)
    return path


def included_execution_source(
    working: Path,
    family: str,
    source_path: Path,
    generated_name: str,
    instance_needle: str,
) -> Path:
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(
        f'`include "../generated/{generated_name}"',
        f'`include "{runtime_path(working / "vectors" / generated_name)}"',
        1,
    )
    if family == "softmax":
        needle = "    localparam integer CONTEXT_MAX = 8;"
        require(source.count(needle) == 1, "softmax context parameter changed")
        source = source.replace(
            needle,
            f"    localparam integer CONTEXT_MAX = {MAX_CONTEXT_TOKENS};",
            1,
        )
    if family == "rope":
        source = source.replace(
            "saturation_seen !== rope_expected_saturation[selected_case]",
            "saturation_seen !== rope_expected_saturation_by_beat[selected_case*ROPE_BEATS + selected_beat]",
            1,
        ).replace(
            "saturation_seen, rope_expected_saturation[selected_case]",
            "saturation_seen, rope_expected_saturation_by_beat[selected_case*ROPE_BEATS + selected_beat]",
            1,
        )
        needle = "            end else begin\n                if (out_data !=="
        replacement = (
            "            end else begin\n"
            "                $display(\"RTL_GENERATION_ROPE_RESULT case=%0d beat=%0d data=%032x\", "
            "selected_case, selected_beat, out_data);\n"
            "                if (out_data !=="
        )
        require(source.count(needle) == 1, "RoPE observation insertion point changed")
        source = source.replace(needle, replacement, 1)
    source = instrument_cycles(source, family, instance_needle)
    path = working / "execution_sources" / f"{family}_tb.sv"
    write_text(path, source)
    return path


def residual_execution_source(working: Path) -> Path:
    source = layer_runner.RESIDUAL_TB.read_text(encoding="utf-8")
    source = source.replace(
        '`include "../generated/residual_vectors.svh"',
        f'`include "{runtime_path(working / "vectors/residual_vectors.svh")}"',
        1,
    ).replace(
        '`include "../generated/mlp_residual_vectors.svh"',
        f'`include "{runtime_path(working / "vectors/mlp_residual_vectors.svh")}"',
        1,
    )
    branch = r'''
        if ($test$plusargs("ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_REPLAY")) begin
            send_residual_cmd_full(0, 16'd896, RESIDUAL_LHS_BASE,
                                   RESIDUAL_RHS_BASE, RESIDUAL_OUT_BASE, 16'h7d00);
            wait_residual_done_and_compare(0, residual_expected_saturation[0], 16'h7d00);
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("RTL_GENERATION_RESIDUAL_BEAT stage=attention beat=%0d data=%032x",
                         case_index, observed_output[case_index]);
            if (residual_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);

            send_mlp_residual_cmd_full(0, 16'd896, MLP_RESIDUAL_DOWN_BASE,
                                       MLP_RESIDUAL_STREAM_BASE, MLP_RESIDUAL_OUT_BASE, 16'h7d01);
            wait_mlp_residual_done_and_compare(0, mlp_residual_expected_saturation[0], 16'h7d01);
            for (case_index = 0; case_index < observed_count; case_index = case_index + 1)
                $display("RTL_GENERATION_RESIDUAL_BEAT stage=final beat=%0d data=%032x",
                         case_index, observed_output[case_index]);
            if (mlp_residual_expected_saturation[0])
                csr_write64(ACE2_CSR_ERROR_STATUS, 64'hffff_ffff_ffff_ffff);
            if (failures != 0)
                $fatal(1, "ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_FAIL failures=%0d", failures);
            $display("ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_PASS stages=2 beats=112 cycles=%0d",
                     cycle_count);
            $finish;
        end

'''
    needle = "        if (qproj_stride_only_mode) begin"
    require(needle in source, "shell residual replay insertion point changed")
    source = source.replace(needle, branch + needle, 1)
    path = working / "execution_sources/residual_tb.sv"
    write_text(path, source)
    return path


def display_command(command: list[str]) -> list[str]:
    return [display_path(value) for value in command]


def run_process(working: Path, name: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed, child_record = tracked_run(command, cwd=ROOT)
    elapsed = time.monotonic() - started
    logs = working / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json(
        logs / f"{name}.result.json",
        {
            "command": display_command(command),
            "elapsed_wall_seconds": elapsed,
            "process_tree_rss": child_record,
            "returncode": completed.returncode,
        },
    )
    return completed


def compile_and_run(
    working: Path,
    name: str,
    sources: list[Path],
    top: str,
    plusargs: list[str] | None = None,
) -> dict[str, Any]:
    binary = working / "sim" / f"{name}.vvp"
    binary.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        _SIMULATOR_TOOLS["iverilog"],
        "-g2012",
        "-I.",
        "-Irtl",
        "-Irtl/generated",
        "-Iverification/tb",
        "-s",
        top,
        "-o",
        runtime_path(binary),
        *[runtime_path(path) for path in sources],
    ]
    compiled = run_process(working, f"{name}.iverilog", compile_command)
    if compiled.returncode != 0:
        raise SimulatorProcessError(
            f"{name} iverilog compile failed",
            classification="SIMULATOR_COMPILE_FAILURE_NO_EXECUTION",
            process_kind="iverilog",
            returncode=compiled.returncode,
            rtl_executed=False,
        )
    relevant_compile_warnings = [
        line
        for line in (compiled.stdout + "\n" + compiled.stderr).splitlines()
        if "warning" in line.lower()
        and any(
            marker in line.lower()
            for marker in ("out of bound", "out-of-bound", "selecting after")
        )
    ]
    if relevant_compile_warnings:
        raise SimulatorProcessError(
            f"{name} iverilog reported an out-of-bounds warning",
            classification="SIMULATOR_COMPILE_WARNING_NO_EXECUTION",
            process_kind="iverilog",
            returncode=compiled.returncode,
            rtl_executed=False,
        )
    simulate_command = [_SIMULATOR_TOOLS["vvp"], runtime_path(binary), *(plusargs or [])]
    simulated = run_process(working, f"{name}.vvp", simulate_command)
    if simulated.returncode != 0:
        raise SimulatorProcessError(
            f"{name} vvp simulation failed",
            classification="SIMULATOR_EXECUTION_FAILURE",
            process_kind="vvp",
            returncode=simulated.returncode,
            rtl_executed=True,
        )
    cycle_match = re.search(
        rf"ACE2_GENERATION_SIM_CYCLES family={re.escape(name)} cycles=(\d+)",
        simulated.stdout,
    )
    if name not in {"residual", "lm_head"}:
        if cycle_match is None:
            raise SimulatorProcessError(
                f"{name} cycle marker is absent",
                classification="SIMULATOR_NO_EXECUTION_EVIDENCE",
                process_kind="vvp",
                returncode=simulated.returncode,
                rtl_executed=False,
            )
    compile_result_path = working / "logs" / f"{name}.iverilog.result.json"
    simulate_result_path = working / "logs" / f"{name}.vvp.result.json"
    compile_elapsed = json.loads(compile_result_path.read_text(encoding="utf-8"))["elapsed_wall_seconds"]
    simulate_elapsed = json.loads(simulate_result_path.read_text(encoding="utf-8"))["elapsed_wall_seconds"]
    return {
        "compile_command": display_command(compile_command),
        "simulate_command": display_command(simulate_command),
        "binary": file_record(binary),
        "stdout": file_record(working / "logs" / f"{name}.vvp.stdout.log"),
        "stderr": file_record(working / "logs" / f"{name}.vvp.stderr.log"),
        "simulator_cycles": int(cycle_match.group(1)) if cycle_match else None,
        "iverilog_wall_seconds": compile_elapsed,
        "vvp_wall_seconds": simulate_elapsed,
        "icarus_wall_seconds": compile_elapsed + simulate_elapsed,
        "stdout_text": simulated.stdout,
    }


def persist_layer_tensors(working: Path, derived: dict[str, Any]) -> dict[str, Any]:
    position = derived["positions"][0]
    directory = working / "tensors"
    records: dict[str, Any] = {}
    payloads = {
        "input_hidden_s8.bin": raw_int8(position["embedding_q"]),
        "input_rmsnorm_s8.bin": raw_int8(position["input_norm_q"]),
        "rope_q_s8.bin": raw_int8(position["rope_q"].outputs),
        "rope_k_append_s8.bin": raw_int8(position["rope_k"].outputs),
        "v_cache_append_s8.bin": raw_int8(position["projections"]["v"]["output_q"]),
        "attention_s8.bin": raw_int8(position["attention_q"]),
        "attention_residual_s8.bin": raw_int8(position["attention_residual"]["output"]),
        "post_attention_rmsnorm_s8.bin": raw_int8(position["post_norm_q"]),
        "silu_output_s8.bin": raw_int8(position["silu"]["output"]),
        "layer_output_s8.bin": raw_int8(position["final_residual"]["output"]),
        "input_scale_f64le.bin": struct.pack("<d", position["embedding_scale"]),
        "layer_output_scale_f64le.bin": struct.pack("<d", position["final_residual"]["scale"]),
        "full_k_cache_s8.bin": raw_int8([item for row in derived["cached_k"] for item in row]),
        "full_v_cache_s8.bin": raw_int8([item for row in derived["cached_v"] for item in row]),
    }
    for name, raw in payloads.items():
        records[name] = write_binary(directory / name, raw)
    return records


def parse_projection_stdout(stdout: str) -> list[tuple[int, int, int, int, int, int]]:
    lines = [line for line in stdout.splitlines() if "RTL_PROJ_RESULT" in line]
    expected_rows = [
        (operator, channel)
        for operator, key in enumerate(PROJECTION_ORDER)
        for channel in range(PROJECTION_CHANNELS[key])
    ]
    require(
        len(lines) == len(expected_rows),
        "single-token projection RTL result count differs",
    )
    rows: list[tuple[int, int, int, int, int, int]] = []
    for index, (line, expected_address) in enumerate(zip(lines, expected_rows, strict=True)):
        match = PROJECTION_RESULT_PATTERN.fullmatch(line)
        require(match is not None, f"projection RTL result row {index} is malformed")
        row = tuple(int(value) for value in match.groups())
        operator, channel, out_u8, accumulator, saturation, overflow = row
        require(
            (operator, channel) == expected_address,
            f"projection RTL result row {index} address differs",
        )
        require(0 <= out_u8 <= 0xFF, f"projection RTL result row {index} output is out of range")
        require(
            -(1 << 31) <= accumulator < (1 << 31),
            f"projection RTL result row {index} accumulator is out of range",
        )
        require(saturation in (0, 1), f"projection RTL result row {index} saturation is malformed")
        require(overflow in (0, 1), f"projection RTL result row {index} overflow is malformed")
        rows.append(row)
    return rows


def verify_projection_stdout(stdout: str, position: dict[str, Any]) -> dict[str, int]:
    rows = parse_projection_stdout(stdout)
    expected_count = sum(PROJECTION_CHANNELS.values())
    mismatch = {"address": 0, "output": 0, "accumulator": 0, "saturation": 0, "overflow": 0}
    index = 0
    for operator, key in enumerate(PROJECTION_ORDER):
        projection = position["projections"][key]
        projection_output = projection.get("baseline_output_q", projection["output_q"])
        outputs = projection_output.to(torch.int64).tolist()
        accumulators = projection["accumulator"].to(torch.int64).tolist()
        saturation = projection["saturation"].to(torch.int64).tolist()
        require(
            len(outputs) == PROJECTION_CHANNELS[key]
            and len(accumulators) == PROJECTION_CHANNELS[key]
            and len(saturation) == PROJECTION_CHANNELS[key],
            f"{key} projection host result count differs",
        )
        for channel in range(len(outputs)):
            op, observed_channel, out_u8, acc, sat, overflow = rows[index]
            index += 1
            mismatch["address"] += int(op != operator or observed_channel != channel)
            mismatch["output"] += int(out_u8 != (int(outputs[channel]) & 0xFF))
            mismatch["accumulator"] += int(acc != int(accumulators[channel]))
            mismatch["saturation"] += int(sat != int(saturation[channel]))
            mismatch["overflow"] += int(overflow != 0)
    require(all(value == 0 for value in mismatch.values()), f"projection RTL mismatch: {mismatch}")
    return {"results": len(rows), **{f"{key}_mismatches": value for key, value in mismatch.items()}}


def observed_cache_append(working: Path, derived: dict[str, Any]) -> dict[str, Any]:
    position = derived["positions"][0]
    projection_rows = parse_projection_stdout(
        (working / "logs/projection.vvp.stdout.log").read_text(encoding="utf-8"),
    )
    observed_v_rows = [
        (channel, out_u8)
        for operator, channel, out_u8, _acc, _sat, _overflow in projection_rows
        if operator == 2
    ]
    require([channel for channel, _ in observed_v_rows] == list(range(KV_HEADS * HEAD_DIM)), "RTL V append order differs")
    observed_v = bytes(value for _, value in observed_v_rows)

    rope_rows = re.findall(
        r"RTL_GENERATION_ROPE_RESULT case=(\d+) beat=(\d+) data=([0-9a-fA-F]+)",
        (working / "logs/rope.vvp.stdout.log").read_text(encoding="utf-8"),
    )
    selected = [(int(beat), data) for case, beat, data in rope_rows if int(case) == 1]
    require([beat for beat, _ in selected] == list(range(56)), "RTL K append beat order differs")
    observed_k_padded = b"".join(int(data, 16).to_bytes(16, "little") for _, data in selected)
    observed_k = observed_k_padded[: KV_HEADS * HEAD_DIM]
    require(observed_k_padded[KV_HEADS * HEAD_DIM :] == bytes(HIDDEN - KV_HEADS * HEAD_DIM), "RTL K padding is nonzero")

    expected_k = raw_int8(position["rope_k"].outputs)
    v_projection = position["projections"]["v"]
    expected_v = raw_int8(v_projection.get("baseline_output_q", v_projection["output_q"]))
    k_mismatches = sum(left != right for left, right in zip(observed_k, expected_k, strict=True))
    v_mismatches = sum(left != right for left, right in zip(observed_v, expected_v, strict=True))
    require(k_mismatches + v_mismatches == 0, "RTL K/V append differs from canonical integer boundary")
    sidecar = v_projection.get("rank1_sidecar")
    if sidecar is not None:
        observed_v = raw_int8(sidecar["_rtl_corrected_v_s8"])
    observed_dir = working / "rtl_observed"
    k_record = write_binary(observed_dir / "k_cache_append_s8.bin", observed_k)
    v_record = write_binary(observed_dir / "v_cache_append_s8.bin", observed_v)
    return {
        "status": "PASS_RTL_KV_APPEND_EXACT",
        "cache_length_before": derived["cache_reuse"]["context_length_before"],
        "cache_length_after": derived["cache_reuse"]["context_length_after"],
        "k_bytes_compared": len(expected_k),
        "v_bytes_compared": len(expected_v),
        "k_byte_mismatches": k_mismatches,
        "v_byte_mismatches": v_mismatches,
        "append_bytes": len(expected_k) + len(expected_v),
        "rtl_observed_k": k_record,
        "rtl_observed_v": v_record,
        "_rtl_observed_k_s8": [
            value - 256 if value & 0x80 else value for value in observed_k
        ],
        "_rtl_observed_v_s8": [
            value - 256 if value & 0x80 else value for value in observed_v
        ],
        "full_k_cache_sha256": sha256_bytes(raw_int8([item for row in derived["cached_k"] for item in row])),
        "full_v_cache_sha256": sha256_bytes(raw_int8([item for row in derived["cached_v"] for item in row])),
        "cache_reuse": derived["cache_reuse"],
        "rank1_sidecar": sidecar,
    }


def consume_rtl_cache_append(
    cache: dict[str, Any],
    rtl_result: dict[str, Any],
    *,
    layer_id: int,
    absolute_position: int,
) -> dict[str, Any]:
    observed = rtl_result.pop("_rtl_observed_cache_append", None)
    require(isinstance(observed, dict), "RTL cache append payload is absent")
    observed_k = observed.get("k_s8")
    observed_v = observed.get("v_s8")
    require(
        isinstance(observed_k, list) and len(observed_k) == KV_HEADS * HEAD_DIM,
        "RTL K cache append payload has the wrong shape",
    )
    require(
        isinstance(observed_v, list) and len(observed_v) == KV_HEADS * HEAD_DIM,
        "RTL V cache append payload has the wrong shape",
    )
    require(
        len(cache["k"]) == absolute_position + 1
        and len(cache["v"]) == absolute_position + 1,
        "RTL cache replacement is not causal",
    )
    cache["k"][-1] = [int(value) for value in observed_k]
    cache["v"][-1] = [int(value) for value in observed_v]
    cache_record = rtl_result["kv_cache"]
    return {
        "layer_id": layer_id,
        "absolute_position": absolute_position,
        "cache_length_after": len(cache["k"]),
        "k_source": "icarus_rope_stdout",
        "v_source": (
            rtl_result["rank1_sidecar"]["v_source"]
            if rtl_result.get("rank1_sidecar") is not None
            else "icarus_projection_stdout"
        ),
        "rtl_observed_k_sha256": cache_record["rtl_observed_k"]["sha256"],
        "rtl_observed_v_sha256": cache_record["rtl_observed_v"]["sha256"],
        "host_cache_append_replaced": True,
        "downstream_consumer_position": absolute_position + 1,
        "rank1_sidecar_executed": rtl_result.get("rank1_sidecar") is not None,
    }


def verify_final_residual_stdout(stdout: str, position: dict[str, Any]) -> dict[str, Any]:
    rows = re.findall(
        r"RTL_GENERATION_RESIDUAL_BEAT stage=final beat=(\d+) data=([0-9a-fA-F]+)",
        stdout,
    )
    require([int(beat) for beat, _ in rows] == list(range(56)), "final residual RTL beat order differs")
    observed = b"".join(int(data, 16).to_bytes(16, "little") for _, data in rows)
    expected = raw_int8(position["final_residual"]["output"])
    mismatches = sum(left != right for left, right in zip(observed, expected, strict=True))
    require(mismatches == 0, "final residual RTL output differs")
    return {
        "bytes_compared": len(expected),
        "byte_mismatches": mismatches,
        "output_sha256": sha256_bytes(observed),
    }


def run_layer_rtl(working: Path, derived: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = process_cpu_seconds()
    working.mkdir(parents=True, exist_ok=False)
    vectors = persist_layer_vectors(working, derived)
    tensors = persist_layer_tensors(working, derived)
    projection_source = projection_execution_source(working)
    sources = {
        "rmsnorm": included_execution_source(
            working, "rmsnorm", layer_runner.RMS_TB, "rmsnorm_vectors.svh", "    ace2_rmsnorm_core dut ("
        ),
        "rope": included_execution_source(
            working, "rope", layer_runner.ROPE_TB, "rope_vectors.svh", "    ace2_rope_core dut ("
        ),
        "score": included_execution_source(
            working, "score", layer_runner.SCORE_TB, "attention_score_vectors.svh", "    ace2_attention_score_core #(" 
        ),
        "softmax": included_execution_source(
            working, "softmax", layer_runner.SOFTMAX_TB, "softmax_vectors.svh", "    ace2_softmax_core #(" 
        ),
        "compose": included_execution_source(
            working, "compose", layer_runner.COMPOSE_TB, "attention_compose_vectors.svh", "    ace2_attention_compose_core dut ("
        ),
        "silu": included_execution_source(
            working, "silu", layer_runner.SILU_TB, "silu_gate_vectors.svh", "    ace2_silu_gate_core dut ("
        ),
        "residual": residual_execution_source(working),
    }
    executions = {
        "projection": compile_and_run(
            working,
            "projection",
            [layer_runner.CORES["projection"], projection_source],
            "ace2_lora_v4_layer0_projection_batch_tb",
        ),
        "rmsnorm": compile_and_run(
            working, "rmsnorm", [layer_runner.CORES["rmsnorm"], sources["rmsnorm"]], "ace2_rmsnorm_tb"
        ),
        "rope": compile_and_run(
            working, "rope", [layer_runner.CORES["rope"], sources["rope"]], "ace2_rope_tb"
        ),
        "score": compile_and_run(
            working, "score", [layer_runner.CORES["score"], sources["score"]], "ace2_attention_score_tb"
        ),
        "softmax": compile_and_run(
            working, "softmax", [layer_runner.CORES["softmax"], sources["softmax"]], "ace2_softmax_tb"
        ),
        "compose": compile_and_run(
            working, "compose", [layer_runner.CORES["compose"], sources["compose"]], "ace2_attention_compose_tb"
        ),
        "silu": compile_and_run(
            working, "silu", [layer_runner.CORES["silu"], sources["silu"]], "ace2_silu_gate_tb"
        ),
    }
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    executions["residual"] = compile_and_run(
        working,
        "residual",
        [*rtl_sources, sources["residual"]],
        "ace2_shell_tb",
        ["+ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_REPLAY"],
    )

    markers = {
        "projection": "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=7 channels=12672",
        "rmsnorm": "ACE2_RMSNORM_TB_PASS cases=2 beats_per_case=56",
        "rope": "ACE2_ROPE_TB_PASS cases=2 beats_per_case=56",
        "score": f"ACE2_ATTN_SCORE_TB_PASS cases=14 context_max={MAX_CONTEXT_TOKENS}",
        "softmax": f"ACE2_SOFTMAX_TB_PASS cases=14 context_max={MAX_CONTEXT_TOKENS}",
        "compose": "ACE2_ATTN_COMPOSE_TB_PASS cases=14",
        "silu": "ACE2_SILU_GATE_TB_PASS cases=1",
        "residual": "ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_PASS stages=2 beats=112",
    }
    for name, marker in markers.items():
        require(marker in executions[name]["stdout_text"], f"{name} RTL PASS marker is absent")
    residual_observations = [
        line
        for line in executions["residual"]["stdout_text"].splitlines()
        if line.startswith("RTL_GENERATION_RESIDUAL_BEAT")
    ]
    require(bool(residual_observations), "residual RTL observations are absent")
    require(
        not any(re.search(r"data=[0-9a-fA-F]*[xXzZ]", line) for line in residual_observations),
        "residual RTL observation contains X/Z data",
    )

    position = derived["positions"][0]
    projection_check = verify_projection_stdout(executions["projection"]["stdout_text"], position)
    cache = observed_cache_append(working, derived)
    observed_cache = {
        "k_s8": cache.pop("_rtl_observed_k_s8"),
        "v_s8": cache.pop("_rtl_observed_v_s8"),
    }
    rank1_sidecar = cache.pop("rank1_sidecar")
    residual = verify_final_residual_stdout(executions["residual"]["stdout_text"], position)
    for execution in executions.values():
        execution.pop("stdout_text", None)
    simulator_timing = simulator_phase_timing(executions.values())
    sidecar_simulation_seconds = (
        float(rank1_sidecar["simulation_wall_seconds"])
        if rank1_sidecar is not None
        else 0.0
    )
    icarus_wall_seconds = (
        simulator_timing["icarus_wall_seconds"] + sidecar_simulation_seconds
    )
    timing = timing_record(
        time.monotonic() - started,
        icarus_wall_seconds,
        process_cpu_seconds() - cpu_started,
    )
    result = {
        "status": "PASS_SINGLE_TOKEN_ALL_BOUNDARIES_RTL",
        "layer_id": derived["layer_id"],
        "absolute_position": position["position"],
        "token_id": position["token_id"],
        "vectors": vectors,
        "tensors": tensors,
        "executions": executions,
        "projection_check": projection_check,
        "kv_cache": cache,
        "rank1_sidecar": (
            {key: value for key, value in rank1_sidecar.items() if not key.startswith("_")}
            if rank1_sidecar is not None
            else None
        ),
        "final_residual": residual,
        "integer_boundary_mismatches": {
            "rmsnorm": 0,
            "projection": sum(value for key, value in projection_check.items() if key.endswith("_mismatches")),
            "rope": 0,
            "attention_score": 0,
            "softmax": 0,
            "attention_compose": 0,
            "silu": 0,
            "residual": residual["byte_mismatches"],
            "kv_cache": cache["k_byte_mismatches"] + cache["v_byte_mismatches"],
        },
        "timing": timing,
        "compile_wall_seconds": simulator_timing["compile_wall_seconds"],
        "simulation_wall_seconds": (
            simulator_timing["simulation_wall_seconds"] + sidecar_simulation_seconds
        ),
        "elapsed_wall_seconds": timing["total_wall_seconds"],
        "cpu_wall_seconds": timing["cpu_wall_seconds"],
        "icarus_wall_seconds": timing["icarus_wall_seconds"],
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    require(
        all(value == 0 for value in result["integer_boundary_mismatches"].values()),
        "single-token RTL boundary mismatch aggregate is nonzero",
    )
    result_path = working / "rtl_execution.json"
    write_json(result_path, result)
    result["persistence"] = prune_transient_execution_artifacts(working)
    write_json(result_path, result)
    result["_rtl_observed_cache_append"] = observed_cache
    return result


def run_stage1_repair_regressions(working: Path) -> dict[str, Any]:
    """Run focused context-width and retained residual testbench regressions."""
    require(not working.exists(), f"repair regression output exists: {public_path(working)}")
    working.mkdir(parents=True)

    softmax_working = working / "context-40-softmax"
    softmax_vectors = softmax_working / "vectors"
    softmax_vectors.mkdir(parents=True)
    scores = [0] * MAX_CONTEXT_TOKENS
    softmax = canonical.reference_softmax(
        canonical.SoftmaxCase("context40", scores),
        context_max=MAX_CONTEXT_TOKENS,
    )
    score = SimpleNamespace(core_scores_q6_9=scores)
    softmax_derived = {
        "cached_k": [[] for _ in range(MAX_CONTEXT_TOKENS)],
        "positions": [
            {
                "scores": [score for _ in range(Q_HEADS)],
                "probabilities": [softmax for _ in range(Q_HEADS)],
            }
        ],
    }
    write_text(
        softmax_vectors / "softmax_vectors.svh",
        render_softmax_vectors(softmax_derived),
    )
    softmax_source = included_execution_source(
        softmax_working,
        "softmax",
        layer_runner.SOFTMAX_TB,
        "softmax_vectors.svh",
        "    ace2_softmax_core #(",
    )
    softmax_execution = compile_and_run(
        softmax_working,
        "softmax",
        [layer_runner.CORES["softmax"], softmax_source],
        "ace2_softmax_tb",
    )
    require(
        f"ACE2_SOFTMAX_TB_PASS cases=14 context_max={MAX_CONTEXT_TOKENS}"
        in softmax_execution["stdout_text"],
        "context-40 softmax PASS marker is absent",
    )
    require("MISMATCH" not in softmax_execution["stdout_text"], "context-40 softmax mismatch")
    require("FAIL" not in softmax_execution["stdout_text"], "context-40 softmax failed")

    residual_working = working / "retained-residual-x-clean"
    residual_vectors = residual_working / "vectors"
    residual_vectors.mkdir(parents=True)
    zeros = torch.zeros(HIDDEN, dtype=torch.int8)
    residual_position = {
        "attention_residual": {
            "lhs": zeros,
            "rhs": zeros,
            "output": zeros,
            "saturation": False,
        },
        "final_residual": {
            "down": zeros,
            "stream": zeros,
            "output": zeros,
            "saturation": False,
        },
    }
    write_text(
        residual_vectors / "residual_vectors.svh",
        render_residual_vectors(residual_position, final=False),
    )
    write_text(
        residual_vectors / "mlp_residual_vectors.svh",
        render_residual_vectors(residual_position, final=True),
    )
    residual_source = residual_execution_source(residual_working)
    rtl_sources = [ROOT / "rtl/ace2_pkg.sv"] + sorted(
        path for path in (ROOT / "rtl").glob("*.sv") if path.name != "ace2_pkg.sv"
    )
    residual_execution = compile_and_run(
        residual_working,
        "residual",
        [*rtl_sources, residual_source],
        "ace2_shell_tb",
        ["+ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_REPLAY"],
    )
    require(
        "ACE2_GENERATION_SINGLE_TOKEN_RESIDUAL_PASS stages=2 beats=112"
        in residual_execution["stdout_text"],
        "retained residual PASS marker is absent",
    )
    residual_observations = [
        line
        for line in residual_execution["stdout_text"].splitlines()
        if line.startswith("RTL_GENERATION_RESIDUAL_BEAT")
    ]
    require(len(residual_observations) == 112, "retained residual observation count differs")
    require(
        not any(re.search(r"data=[0-9a-fA-F]*[xXzZ]", line) for line in residual_observations),
        "retained residual observation contains X/Z data",
    )

    softmax_execution.pop("stdout_text", None)
    residual_execution.pop("stdout_text", None)
    result = {
        "schema_version": 1,
        "status": "PASS_STAGE1_CHAT_CONTEXT_ARTIFACT_X_CLEAN_REGRESSIONS",
        "context_bound": MAX_CONTEXT_TOKENS,
        "softmax_context_executed": MAX_CONTEXT_TOKENS,
        "softmax": softmax_execution,
        "retained_mlp_residual_case_count": 2,
        "retained_residual_observation_count": len(residual_observations),
        "retained_residual_xz_count": 0,
        "relevant_out_of_bounds_warning_count": 0,
        "residual": residual_execution,
    }
    write_json(working / "summary.json", result)
    return result


def derive_final_rmsnorm_case(state: dict[str, Any], norm_gain: torch.Tensor) -> dict[str, Any]:
    float_input = state["fixed_q"].to(torch.float64) * float(state["fixed_scale"])
    float_norm = canonical.float_rmsnorm(float_input.to(torch.float32), norm_gain).contiguous()
    output_scale = canonical.derive_rmsnorm_output_scale(
        norm_gain.to(torch.float64).tolist(),
        float(float_norm.to(torch.float64).abs().max().item()),
    )
    gains_q8 = canonical.derive_scaled_gains_q8(norm_gain.to(torch.float64).tolist(), output_scale)
    result = canonical.reference_rmsnorm(state["fixed_q"].to(torch.int64).tolist(), gains_q8)
    return {
        "input_q": state["fixed_q"],
        "input_scale": float(state["fixed_scale"]),
        "float_norm": float_norm,
        "final_q": torch.tensor(result.outputs, dtype=torch.int8),
        "final_scale": float(output_scale),
        "gains_q8": gains_q8,
        "rmsnorm_result": result,
    }


def render_final_rmsnorm_vectors(item: dict[str, Any]) -> str:
    result = item["rmsnorm_result"]
    lines = [
        "// One generation-step final RMSNorm.",
        "localparam integer TEST_COUNT = 1;",
        "localparam integer TEST_BEATS = 56;",
        "reg [8*16-1:0] test_input_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [16*16-1:0] test_gain_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [8*16-1:0] test_expected_beats [0:TEST_COUNT*TEST_BEATS-1];",
        "reg [47:0] test_expected_sumsq [0:TEST_COUNT-1];",
        "reg [31:0] test_expected_inv [0:TEST_COUNT-1];",
        "reg test_expected_saturation [0:TEST_COUNT-1];",
        "initial begin",
        f"  test_expected_sumsq[0] = {hex_literal(result.sumsq, 48)};",
        f"  test_expected_inv[0] = {hex_literal(result.inv_rms_q30, 32)};",
        f"  test_expected_saturation[0] = 1'b{int(result.saturation_seen)};",
    ]
    for beat, (input_word, gain_word, output_word) in enumerate(
        zip(
            canonical.pack_int8_beats(item["input_q"].to(torch.int64).tolist()),
            canonical.pack_gain_beats(item["gains_q8"]),
            canonical.pack_int8_beats(result.outputs),
            strict=True,
        )
    ):
        lines += [
            f"  test_input_beats[{beat}] = {hex_literal(input_word, 128)};",
            f"  test_gain_beats[{beat}] = {hex_literal(gain_word, 256)};",
            f"  test_expected_beats[{beat}] = {hex_literal(output_word, 128)};",
        ]
    return "\n".join(lines + ["end", ""])


def derive_lm_head_weights(
    shared: Path, embedding: torch.Tensor, output_count: int
) -> dict[str, Any]:
    require(output_count % LM_HEAD_TILE_OUTPUTS == 0, "LM-head output count is not tile aligned")
    qweight = torch.empty((output_count, HIDDEN), dtype=torch.int8)
    weight_scale = torch.empty((output_count,), dtype=torch.float64)
    for start in range(0, output_count, 2048):
        stop = min(output_count, start + 2048)
        rows = embedding[start:stop].to(torch.float64)
        scale = rows.abs().amax(dim=1) / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight[start:stop] = torch.round(rows / scale[:, None]).clamp(-8, 7).to(torch.int8)
        weight_scale[start:stop] = scale
    weight_path = shared / "lm_head_weight.hex"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    with weight_path.open("w", encoding="ascii") as handle:
        lookup = [f"{value:04x}\n" for value in range(1 << 16)]
        for start in range(0, output_count, 512):
            stop = min(output_count, start + 512)
            packed = lm_head.packed_w4_groups(qweight[start:stop]).tolist()
            handle.write("".join(lookup[int(value) & 0xFFFF] for value in packed))
    return {
        "qweight": qweight,
        "weight_scale": weight_scale,
        "weight_path": weight_path,
        "artifacts": {
            "lm_head_weight_hex": file_record(weight_path),
            "lm_head_qweight_raw_sha256": sha256_bytes(tensor_bytes(qweight)),
            "output_count": output_count,
        },
    }


def derive_lm_head_logits(
    working: Path,
    item: dict[str, Any],
    embedding: torch.Tensor,
    weights_record: dict[str, Any],
    output_count: int,
) -> dict[str, Any]:
    float_outputs = []
    for start in range(0, output_count, 4096):
        stop = min(output_count, start + 4096)
        float_outputs.append(torch.mv(embedding[start:stop].to(torch.float32), item["float_norm"]))
    float_output = torch.cat(float_outputs).contiguous()
    output_scale = canonical.scale_for(float_output)
    multiplier, right_shift = canonical.derive_multiplier(
        item["final_scale"] * weights_record["weight_scale"] / output_scale
    )
    accumulator = torch.empty((output_count,), dtype=torch.int64)
    activation = item["final_q"].to(torch.int32)
    for start in range(0, output_count, 4096):
        stop = min(output_count, start + 4096)
        accumulator[start:stop] = (
            weights_record["qweight"][start:stop].to(torch.int32) * activation
        ).sum(dim=1, dtype=torch.int64)
    require(bool(torch.all(accumulator >= -(1 << 31))), "LM-head accumulator underflow")
    require(bool(torch.all(accumulator < (1 << 31))), "LM-head accumulator overflow")
    rounded = lm_head.round_outputs(accumulator, multiplier, right_shift)
    output_q = rounded.clamp(-128, 127).to(torch.int8)
    saturation = ((rounded < -128) | (rounded > 127)).to(torch.uint8)
    top_token, top_logit = lm_head.top_token(output_q)

    vectors = working / "vectors"
    tensors = working / "tensors"
    activation_words = lm_head.pack_i8_groups(item["final_q"])
    artifacts = {
        "lm_head_activation_hex": lm_head.write_hex_tensor(vectors / "lm_head_activation.hex", activation_words, 32),
        "lm_head_multiplier_hex": lm_head.write_hex_tensor(vectors / "lm_head_multiplier.hex", multiplier, 32),
        "lm_head_shift_hex": lm_head.write_hex_tensor(vectors / "lm_head_shift.hex", right_shift, 8),
        "lm_head_expected_hex": lm_head.write_hex_tensor(vectors / "lm_head_expected.hex", output_q.to(torch.int64), 8),
        "lm_head_accumulator_hex": lm_head.write_hex_tensor(vectors / "lm_head_accumulator.hex", accumulator, 32),
        "lm_head_saturation_hex": lm_head.write_hex_tensor(vectors / "lm_head_saturation.hex", saturation.to(torch.int64), 8),
        "final_rmsnorm_s8": write_binary(tensors / "final_rmsnorm_s8.bin", tensor_bytes(item["final_q"])),
        "final_rmsnorm_scale_f64le": write_binary(
            tensors / "final_rmsnorm_scale_f64le.bin", struct.pack("<d", item["final_scale"])
        ),
        "lm_head_output_s8": write_binary(tensors / "lm_head_output_s8.bin", tensor_bytes(output_q)),
        "lm_head_output_scale_f64le": write_binary(
            tensors / "lm_head_output_scale_f64le.bin", struct.pack("<d", float(output_scale))
        ),
    }
    return {
        "output_count": output_count,
        "output_scale": float(output_scale),
        "output_q": output_q,
        "top_token": int(top_token),
        "top_logit_s8": int(top_logit),
        "saturation_count": int(saturation.sum().item()),
        "accumulator_min": int(accumulator.min().item()),
        "accumulator_max": int(accumulator.max().item()),
        "output_sha256": sha256_bytes(tensor_bytes(output_q)),
        "artifacts": artifacts,
    }


def render_lm_head_tb(
    working: Path, shared_weight: Path, absolute_position: int, output_count: int
) -> str:
    vectors = working / "vectors"
    module_name = f"ace2_generation_lm_head_position{absolute_position}_tb"
    return f"""`timescale 1ns/1ps
`default_nettype none

module {module_name};
    localparam integer POSITION = {absolute_position};
    localparam integer TOTAL_OUTPUTS = {output_count};
    localparam integer GROUPS = {HIDDEN // 4};
    localparam integer GROUP_INDEX_WIDTH = $clog2(GROUPS + 1);
    reg clk; reg rst_n; reg start_valid; wire start_ready;
    reg pair_valid; wire pair_ready; reg [31:0] act_data; reg [15:0] weight_data;
    reg [GROUP_INDEX_WIDTH-1:0] last_group; reg meta_valid; wire meta_ready;
    reg signed [31:0] multiplier; reg [5:0] right_shift;
    wire out_valid; reg out_ready; wire [7:0] out_data; wire signed [31:0] acc;
    wire accumulator_overflow; wire saturation_seen;
    reg [31:0] activation_mem [0:GROUPS-1];
    reg [15:0] weight_mem [0:TOTAL_OUTPUTS*GROUPS-1];
    reg [31:0] multiplier_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] shift_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_mem [0:TOTAL_OUTPUTS-1];
    reg [31:0] expected_acc_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_saturation_mem [0:TOTAL_OUTPUTS-1];
    integer channel; integer group_index; integer guard; integer failures; integer cycles;
    integer top_token; integer top_logit_s8; reg top_valid; reg signed [7:0] out_signed;

    ace2_w4a8_proj_core #(.K_SIZE({HIDDEN}), .MAC_LANES(4)) dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(1'b0), .start_valid_i(start_valid),
        .start_ready_o(start_ready), .last_group_i(last_group), .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready), .act_data_i(act_data), .weight_data_i(weight_data),
        .meta_valid_i(meta_valid), .meta_ready_o(meta_ready), .multiplier_i(multiplier),
        .right_shift_i(right_shift), .output_zero_point_i(8'sd0), .bias_accumulator_i(32'sd0),
        .out_valid_o(out_valid), .out_ready_i(out_ready), .out_data_o(out_data), .acc_o(acc),
        .accumulator_overflow_o(accumulator_overflow), .saturation_seen_o(saturation_seen));
    initial begin clk=1'b0; forever #5 clk=~clk; end
    always @(posedge clk) if (rst_n) cycles = cycles + 1;

    task apply_reset; begin
        rst_n=1'b0; start_valid=1'b0; pair_valid=1'b0; act_data=0; weight_data=0;
        last_group=0; meta_valid=1'b0; multiplier=0; right_shift=0; out_ready=1'b0; cycles=0;
        repeat (4) @(posedge clk); rst_n=1'b1; repeat (2) @(posedge clk);
    end endtask
    task run_channel; input integer selected_channel; begin
        while (!start_ready) @(posedge clk); last_group=GROUPS-1;
        @(negedge clk); start_valid=1'b1; @(posedge clk); @(negedge clk); start_valid=1'b0;
        for (group_index=0; group_index<GROUPS; group_index=group_index+1) begin
            while (!pair_ready) @(posedge clk); act_data=activation_mem[group_index];
            weight_data=weight_mem[selected_channel*GROUPS+group_index];
            @(negedge clk); pair_valid=1'b1; @(posedge clk); @(negedge clk); pair_valid=1'b0;
        end
        while (!meta_ready) @(posedge clk); multiplier=$signed(multiplier_mem[selected_channel]);
        right_shift=shift_mem[selected_channel][5:0];
        @(negedge clk); meta_valid=1'b1; @(posedge clk); @(negedge clk); meta_valid=1'b0;
        guard=0; while (!out_valid && guard<8192) begin guard=guard+1; @(posedge clk); end
        if (!out_valid) begin failures=failures+1; end else begin
            out_signed=$signed(out_data);
            if ((out_data !== expected_mem[selected_channel]) ||
                (acc !== $signed(expected_acc_mem[selected_channel])) ||
                (saturation_seen !== expected_saturation_mem[selected_channel][0]) || accumulator_overflow)
                failures=failures+1;
            if (!top_valid || out_signed>top_logit_s8 ||
                ((out_signed==top_logit_s8) && selected_channel<top_token)) begin
                top_valid=1'b1; top_logit_s8=out_signed; top_token=selected_channel;
            end
            out_ready=1'b1; @(posedge clk); @(negedge clk); out_ready=1'b0;
        end
    end endtask
    initial begin
        $readmemh("{runtime_path(vectors / 'lm_head_activation.hex')}", activation_mem);
        $readmemh("{runtime_path(shared_weight)}", weight_mem);
        $readmemh("{runtime_path(vectors / 'lm_head_multiplier.hex')}", multiplier_mem);
        $readmemh("{runtime_path(vectors / 'lm_head_shift.hex')}", shift_mem);
        $readmemh("{runtime_path(vectors / 'lm_head_expected.hex')}", expected_mem);
        $readmemh("{runtime_path(vectors / 'lm_head_accumulator.hex')}", expected_acc_mem);
        $readmemh("{runtime_path(vectors / 'lm_head_saturation.hex')}", expected_saturation_mem);
        failures=0; top_token=0; top_logit_s8=-129; top_valid=1'b0; apply_reset();
        for (channel=0; channel<TOTAL_OUTPUTS; channel=channel+1) run_channel(channel);
        if (failures != 0) $fatal(1, "ACE2_GENERATION_LM_HEAD_FAIL failures=%0d", failures);
        $display("ACE2_GENERATION_LM_HEAD_PASS position=%0d outputs=%0d top_token=%0d top_logit_s8=%0d cycles=%0d",
                 POSITION, TOTAL_OUTPUTS, top_token, top_logit_s8, cycles);
        $finish;
    end
endmodule
`default_nettype wire
"""


def run_final_head_rtl(
    working: Path,
    state: dict[str, Any],
    norm_gain: torch.Tensor,
    embedding: torch.Tensor,
    weights_record: dict[str, Any],
    output_count: int,
    *,
    prune_transient: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    cpu_started = process_cpu_seconds()
    working.mkdir(parents=True, exist_ok=False)
    item = derive_final_rmsnorm_case(state, norm_gain)
    write_text(working / "vectors/rmsnorm_vectors.svh", render_final_rmsnorm_vectors(item))
    logits = derive_lm_head_logits(working, item, embedding, weights_record, output_count)

    rms_source = included_execution_source(
        working,
        "final_rmsnorm",
        lm_head.RMS_TB,
        "rmsnorm_vectors.svh",
        "    ace2_rmsnorm_core dut (",
    )
    rms = compile_and_run(
        working,
        "final_rmsnorm",
        [lm_head.RMS_CORE, rms_source],
        "ace2_rmsnorm_tb",
    )
    require("ACE2_RMSNORM_TB_PASS cases=1 beats_per_case=56" in rms["stdout_text"], "final RMSNorm RTL marker absent")

    lm_source = working / "execution_sources/lm_head_tb.sv"
    write_text(
        lm_source,
        render_lm_head_tb(
            working,
            weights_record["weight_path"],
            int(state["position"]),
            output_count,
        ),
    )
    top = f"ace2_generation_lm_head_position{int(state['position'])}_tb"
    lm_execution = compile_and_run(
        working,
        "lm_head",
        [lm_head.PROJ_CORE, lm_source],
        top,
    )
    marker_match = re.search(
        r"ACE2_GENERATION_LM_HEAD_PASS position=(\d+) outputs=(\d+) top_token=(\d+) top_logit_s8=(-?\d+) cycles=(\d+)",
        lm_execution["stdout_text"],
    )
    require(marker_match is not None, "LM-head RTL marker absent")
    marker = {
        "position": int(marker_match.group(1)),
        "outputs": int(marker_match.group(2)),
        "top_token": int(marker_match.group(3)),
        "top_logit_s8": int(marker_match.group(4)),
        "cycles": int(marker_match.group(5)),
    }
    require(marker["outputs"] == output_count, "LM-head RTL output coverage differs")
    require(marker["top_token"] == logits["top_token"], "LM-head RTL selected token differs")
    require(marker["top_logit_s8"] == logits["top_logit_s8"], "LM-head RTL selected logit differs")
    rms.pop("stdout_text", None)
    lm_execution.pop("stdout_text", None)
    simulator_timing = simulator_phase_timing((rms, lm_execution))
    icarus_wall_seconds = simulator_timing["icarus_wall_seconds"]
    timing = timing_record(
        time.monotonic() - started,
        icarus_wall_seconds,
        process_cpu_seconds() - cpu_started,
    )
    result = {
        "status": "PASS_FINAL_RMSNORM_FULL_LM_HEAD_RTL" if output_count == MODEL_OUTPUT_DOMAIN else "PASS_FINAL_RMSNORM_LM_HEAD_TILE_PREFLIGHT_RTL",
        "absolute_position": int(state["position"]),
        "output_count": output_count,
        "full_vocabulary": output_count == MODEL_OUTPUT_DOMAIN,
        "final_rmsnorm": {
            "input_scale": item["input_scale"],
            "output_scale": item["final_scale"],
            "sumsq": int(item["rmsnorm_result"].sumsq),
            "inv_rms_q30": int(item["rmsnorm_result"].inv_rms_q30),
            "saturation_seen": bool(item["rmsnorm_result"].saturation_seen),
            "output_sha256": sha256_bytes(tensor_bytes(item["final_q"])),
            "rtl": rms,
            "integer_mismatches": 0,
        },
        "lm_head": {
            key: value for key, value in logits.items() if key != "output_q"
        }
        | {
            "rtl": lm_execution,
            "rtl_marker": marker,
            "selected_token_agreement": True,
            "selected_logit_agreement": True,
            "integer_mismatches": 0,
        },
        "timing": timing,
        "compile_wall_seconds": simulator_timing["compile_wall_seconds"],
        "simulation_wall_seconds": simulator_timing["simulation_wall_seconds"],
        "elapsed_wall_seconds": timing["total_wall_seconds"],
        "cpu_wall_seconds": timing["cpu_wall_seconds"],
        "icarus_wall_seconds": timing["icarus_wall_seconds"],
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    result_path = working / "head_execution.json"
    write_json(result_path, result)
    result["persistence"] = (
        prune_transient_execution_artifacts(working)
        if prune_transient
        else {
            "status": "TRANSIENT_ARTIFACT_PRUNING_DEFERRED",
            "required_before_pruning": "DURABLE_DOWNSTREAM_RANK_AND_TERMINAL_PUBLICATION",
        }
    )
    write_json(result_path, result)
    return result


def tokenizer_domain_decision(tokenizer: Any, token_id: int) -> dict[str, Any]:
    tokenizer_length = len(tokenizer)
    base_vocab = int(tokenizer.vocab_size)
    require(0 <= token_id < MODEL_OUTPUT_DOMAIN, "RTL argmax is outside model output domain")
    require(
        token_id < tokenizer_length,
        (
            f"RTL full-vocabulary argmax {token_id} is in padded/unmapped model rows "
            f"[{tokenizer_length}, {MODEL_OUTPUT_DOMAIN - 1}]"
        ),
    )
    piece = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    require(bool(piece), f"RTL selected token {token_id} decodes to an empty piece")
    require(not piece.isspace(), f"RTL selected token {token_id} decodes only to whitespace")
    require(all(character.isprintable() or character in "\r\n\t" for character in piece), "RTL selected token is not readable text")
    return {
        "token_id": token_id,
        "decoded_piece": piece,
        "tokenizer_base_vocab_size": base_vocab,
        "tokenizer_length": tokenizer_length,
        "model_output_domain": MODEL_OUTPUT_DOMAIN,
        "selection_policy": "unmasked full-model-domain greedy argmax; fail closed if argmax is outside tokenizer length or decodes empty/unreadable",
        "feedback_authorized": True,
    }


def run_preflight_smoke(
    working: Path,
    snapshot: Path,
    tokenizer: Any,
    prompt_token_id: int,
    *,
    lm_head_outputs: int = 32,
) -> dict[str, Any]:
    """Execute real Icarus prefill, selected-logit feedback, and decode on layer 0."""
    require(not working.exists(), f"preflight smoke output already exists: {public_path(working)}")
    working.mkdir(parents=True)
    snapshot_record = configure_snapshot(snapshot)
    cache = empty_layer_cache()
    template: dict[str, Any] | None = None
    started = time.monotonic()
    with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
        canonical.ADAPTER, framework="pt", device="cpu"
    ) as adapter:
        embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
        norm_gain = weights.get_tensor("model.norm.weight").contiguous()
        shared_weights = derive_lm_head_weights(working / "shared", embedding, lm_head_outputs)

        prefill_state = embedding_state(weights, prompt_token_id, 0)
        prefill_derived, prefill_next, template = derive_layer_token(
            0, prefill_state, cache, template, weights, adapter
        )
        prefill_rtl = run_layer_rtl(working / "prefill/layer-00", prefill_derived)
        consume_rtl_cache_append(
            cache,
            prefill_rtl,
            layer_id=0,
            absolute_position=0,
        )
        head0 = run_final_head_rtl(
            working / "head-00", prefill_next, norm_gain, embedding, shared_weights, lm_head_outputs
        )
        selected = int(head0["lm_head"]["top_token"])
        selected_piece = tokenizer.decode(
            [selected], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        decode_state = embedding_state(weights, selected, 1)
        decode_derived, decode_next, template = derive_layer_token(
            0, decode_state, cache, template, weights, adapter
        )
        decode_rtl = run_layer_rtl(working / "decode/layer-00", decode_derived)
        consume_rtl_cache_append(
            cache,
            decode_rtl,
            layer_id=0,
            absolute_position=1,
        )
        head1 = run_final_head_rtl(
            working / "head-01", decode_next, norm_gain, embedding, shared_weights, lm_head_outputs
        )

    result = {
        "status": "PASS_REAL_ICARUS_PREFILL_DECODE_BACKEND_SMOKE",
        "classification": "non-official focused implementation preflight; partial 32-row LM-head tiles only",
        "snapshot": snapshot_record,
        "prompt_token_id": prompt_token_id,
        "prefill": prefill_rtl,
        "first_head": head0,
        "selected_feedback_token": selected,
        "selected_feedback_piece": selected_piece,
        "decode": decode_rtl,
        "second_head": head1,
        "cache_growth": [
            prefill_rtl["kv_cache"]["cache_length_after"],
            decode_rtl["kv_cache"]["cache_length_after"],
        ],
        "selected_token_came_from_rtl": True,
        "software_transformer_or_logits_fallback": False,
        "full_vocabulary_execution_claim": False,
        "full_vocabulary_backend_bound": MODEL_OUTPUT_DOMAIN,
        "elapsed_wall_seconds": time.monotonic() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    require(result["cache_growth"] == [1, 2], "preflight smoke cache growth differs")
    write_json(working / "smoke_summary.json", result)
    shared_root = working / "shared"
    result["storage_lifecycle"] = (
        prune_transient_execution_artifacts(shared_root)
        if shared_root.is_dir()
        else {"status": "NO_SHARED_CONSTRUCTION_ARTIFACTS"}
    )
    write_json(working / "smoke_summary.json", result)
    return result


def run_generation_with_state(
    output: Path,
    snapshot: Path,
    tokenizer: Any,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    *,
    execution_mode: str = "official",
    carried_state: CarriedGenerationState | None = None,
) -> tuple[dict[str, Any], CarriedGenerationState]:
    """Run one causal turn and return its exact in-process K/V continuation."""
    context_contract = validate_context_request(prompt_token_ids, max_new_tokens)
    require(
        execution_mode in {"official", "stage1_product"},
        "generation execution mode is unsupported",
    )
    require(not output.exists(), f"generation output already exists: {public_path(output)}")
    output.mkdir(parents=True)
    start_record = {
        "process_start_count": 1,
        "started_monotonic": time.monotonic(),
        "execution_mode": execution_mode,
    }
    if execution_mode == "official":
        start_record["official_attempt_consumed"] = True
    write_json(output / "run.started.json", start_record)
    started = time.monotonic()
    run_cpu_started = process_cpu_seconds()
    snapshot_record = configure_snapshot(snapshot)
    state_owner = (
        carried_state
        if carried_state is not None
        else new_carried_generation_state(execution_mode)
    )
    retained_positions = validate_carried_generation_state(
        state_owner,
        prompt_token_ids,
        snapshot_record,
        execution_mode,
    )
    retained_context_token_ids = list(state_owner.context_token_ids)
    caches = state_owner.caches
    templates = state_owner.templates
    state_owner.valid = False
    generated: list[int] = []
    token_executions: list[dict[str, Any]] = []
    head_steps: list[dict[str, Any]] = []
    rank1_sidecar: Rank1SidecarSession | None = None
    try:
        if execution_mode == "stage1_product":
            rank1_sidecar = Rank1SidecarSession(output / "shared/rank1-sidecar")
        with torch.no_grad(), safe_open(canonical.MODEL, framework="pt", device="cpu") as weights, safe_open(
            canonical.ADAPTER, framework="pt", device="cpu"
        ) as adapter:
            embedding = weights.get_tensor("model.embed_tokens.weight").contiguous()
            norm_gain = weights.get_tensor("model.norm.weight").contiguous()
            shared_weights = derive_lm_head_weights(output / "shared", embedding, MODEL_OUTPUT_DOMAIN)
            rank1_compile_seconds = (
                rank1_sidecar.compile_wall_seconds if rank1_sidecar is not None else 0.0
            )
            setup_timing = timing_record(
                time.monotonic() - started,
                rank1_compile_seconds,
                process_cpu_seconds() - run_cpu_started,
            )
            input_tokens = list(prompt_token_ids)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_token_ids = (
                {int(eos_token_id)}
                if isinstance(eos_token_id, int) and not isinstance(eos_token_id, bool)
                else set()
            )
            position = retained_positions
            step_started = time.monotonic()
            step_cpu_started = process_cpu_seconds()
            step_icarus_wall_seconds = 0.0
            step_positions: list[int] = []
            while True:
                position_started = time.monotonic()
                position_cpu_started = process_cpu_seconds()
                token_id = input_tokens[position] if position < len(input_tokens) else generated[position - len(input_tokens)]
                state = embedding_state(weights, token_id, position)
                layer_records = []
                phase = "prefill" if position < len(prompt_token_ids) else "decode"
                for layer_id in range(LAYERS):
                    layer_dir = output / "tokens" / f"position-{position:02d}" / f"layer-{layer_id:02d}"
                    derived, state, templates[layer_id] = derive_layer_token(
                        layer_id,
                        state,
                        caches[layer_id],
                        templates[layer_id],
                        weights,
                        adapter,
                        rank1_sidecar=rank1_sidecar,
                        sidecar_working=(
                            output
                            / "rank1-sidecar-steps"
                            / f"position-{position:02d}"
                            / f"layer-{layer_id:02d}"
                        ),
                    )
                    rtl = run_layer_rtl(layer_dir, derived)
                    rtl_cache_append = consume_rtl_cache_append(
                        caches[layer_id],
                        rtl,
                        layer_id=layer_id,
                        absolute_position=position,
                    )
                    layer_records.append(
                        {
                            "layer_id": layer_id,
                            "rtl_execution": file_record(layer_dir / "rtl_execution.json"),
                            "cache_length_before": rtl["kv_cache"]["cache_length_before"],
                            "cache_length_after": rtl["kv_cache"]["cache_length_after"],
                            "kv_append_bytes": rtl["kv_cache"]["append_bytes"],
                            "integer_boundary_mismatches": rtl["integer_boundary_mismatches"],
                            "timing": rtl["timing"],
                            "compile_wall_seconds": rtl["compile_wall_seconds"],
                            "simulation_wall_seconds": rtl[
                                "simulation_wall_seconds"
                            ],
                            "cpu_wall_seconds": rtl["cpu_wall_seconds"],
                            "icarus_wall_seconds": rtl["icarus_wall_seconds"],
                            "rtl_cache_append": rtl_cache_append,
                        }
                    )
                position_icarus_wall_seconds = sum(
                    float(layer["icarus_wall_seconds"]) for layer in layer_records
                )
                position_timing = timing_record(
                    time.monotonic() - position_started,
                    position_icarus_wall_seconds,
                    process_cpu_seconds() - position_cpu_started,
                )
                token_executions.append(
                    {
                        "phase": phase,
                        "absolute_position": position,
                        "input_token_id": token_id,
                        "layers": layer_records,
                        "timing": position_timing,
                    }
                )
                step_icarus_wall_seconds += position_icarus_wall_seconds
                step_positions.append(position)

                should_emit = position >= len(prompt_token_ids) - 1
                if should_emit:
                    generation_index = len(generated)
                    head_dir = output / "heads" / f"step-{generation_index:02d}"
                    head = run_final_head_rtl(
                        head_dir, state, norm_gain, embedding, shared_weights, MODEL_OUTPUT_DOMAIN
                    )
                    selected = int(head["lm_head"]["top_token"])
                    domain = tokenizer_domain_decision(tokenizer, selected)
                    generated.append(selected)
                    eos_selected = selected in eos_token_ids
                    step_icarus_wall_seconds += float(head["icarus_wall_seconds"])
                    step_timing = timing_record(
                        time.monotonic() - step_started,
                        step_icarus_wall_seconds,
                        process_cpu_seconds() - step_cpu_started,
                    )
                    head_steps.append(
                        {
                            "generation_index": generation_index,
                            "source_absolute_position": position,
                            "selected_token_id": selected,
                            "selected_logit_s8": head["lm_head"]["top_logit_s8"],
                            "decoded_piece": domain["decoded_piece"],
                            "head_execution": file_record(head_dir / "head_execution.json"),
                            "full_vocabulary_outputs": MODEL_OUTPUT_DOMAIN,
                            "rtl_selected_token_agreement": True,
                            "eos_selected": eos_selected,
                            "feedback_required": (
                                not eos_selected
                                and generation_index + 1 < max_new_tokens
                            ),
                            "positions_included": list(step_positions),
                            "head_timing": head["timing"],
                            "compile_wall_seconds": head["compile_wall_seconds"],
                            "simulation_wall_seconds": head[
                                "simulation_wall_seconds"
                            ],
                            "timing": step_timing,
                        }
                    )
                    if eos_selected or len(generated) == max_new_tokens:
                        break
                    step_started = time.monotonic()
                    step_cpu_started = process_cpu_seconds()
                    step_icarus_wall_seconds = 0.0
                    step_positions = []
                position += 1

        detokenization_started = time.monotonic()
        decoded_with_special_tokens = tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decoded = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        detokenization_wall_seconds = time.monotonic() - detokenization_started
        require(bool(decoded.strip()), "generated RTL token sequence decodes to blank text")
        overall_icarus_wall_seconds = sum(
            float(step["timing"]["icarus_wall_seconds"]) for step in head_steps
        ) + (
            rank1_sidecar.compile_wall_seconds if rank1_sidecar is not None else 0.0
        )
        overall_timing = timing_record(
            time.monotonic() - started,
            overall_icarus_wall_seconds,
            process_cpu_seconds() - run_cpu_started,
        )
        compile_wall_seconds = sum(
            float(layer["compile_wall_seconds"])
            for execution in token_executions
            for layer in execution["layers"]
        ) + sum(float(step["compile_wall_seconds"]) for step in head_steps) + (
            rank1_sidecar.compile_wall_seconds if rank1_sidecar is not None else 0.0
        )
        simulation_wall_seconds = sum(
            float(layer["simulation_wall_seconds"])
            for execution in token_executions
            for layer in execution["layers"]
        ) + sum(float(step["simulation_wall_seconds"]) for step in head_steps)
        require(
            math.isclose(
                compile_wall_seconds + simulation_wall_seconds,
                overall_icarus_wall_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "session compile/simulation timing does not sum to Icarus wall time",
        )
        executed_positions = {
            int(execution["absolute_position"]) for execution in token_executions
        }
        corrected_v_provenance = []
        for execution in token_executions:
            record = dict(execution["layers"][-1]["rtl_cache_append"])
            record["downstream_consumed"] = (
                int(record["downstream_consumer_position"]) in executed_positions
            )
            corrected_v_provenance.append(record)
        require(
            any(record["downstream_consumed"] for record in corrected_v_provenance),
            "no RTL-emitted corrected-V bytes were consumed downstream",
        )
        executed_context_token_ids = [
            int(execution["input_token_id"]) for execution in token_executions
        ]
        require(
            [int(execution["absolute_position"]) for execution in token_executions]
            == list(
                range(
                    retained_positions,
                    retained_positions + len(token_executions),
                )
            ),
            "turn execution positions are not contiguous after carried state",
        )
        state_owner.context_token_ids.extend(executed_context_token_ids)
        state_owner.snapshot_model_sha256 = snapshot_record["model"]["sha256"]
        state_owner.snapshot_config_sha256 = snapshot_record["config"]["sha256"]
        state_owner.valid = True
        latency = {
            "compile_wall_seconds": compile_wall_seconds,
            "model_wall_seconds": max(
                0.0,
                float(overall_timing["total_wall_seconds"])
                - compile_wall_seconds
                - simulation_wall_seconds,
            ),
            "simulation_wall_seconds": simulation_wall_seconds,
            "total_wall_seconds": float(overall_timing["total_wall_seconds"]),
            "compile_definition": "sum of measured iverilog child wall intervals",
            "model_definition": (
                "host model/reference derivation and artifact orchestration wall time "
                "outside measured iverilog/vvp child intervals"
            ),
            "simulation_definition": "sum of measured vvp child wall intervals",
            "total_definition": "single-session backend wall time after output creation",
        }
        summary = {
            "schema_version": 1,
            "status": (
                "PASS_OFFICIAL_RTL_ARBITRARY_TEXT_GENERATION_UNCHECKED"
                if execution_mode == "official"
                else "PASS_STAGE1_PRODUCT_RTL_GENERATION_UNCHECKED"
            ),
            "execution_mode": execution_mode,
            "snapshot": snapshot_record,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated,
            "decoded_text": decoded,
            "decoded_text_with_special_tokens": decoded_with_special_tokens,
            "max_new_tokens": max_new_tokens,
            "termination": {
                "reason": (
                    "eos"
                    if bool(head_steps and head_steps[-1]["eos_selected"])
                    else "max_new_tokens"
                ),
                "eos_token_ids": sorted(eos_token_ids),
                "generated_token_count": len(generated),
            },
            "context_contract": context_contract,
            "carried_state": {
                "layer_count": LAYERS,
                "prefix_recomputed": False,
                "retained_context_token_ids": retained_context_token_ids,
                "retained_positions": retained_positions,
                "turn_executed_positions": len(token_executions),
                "turn_start_absolute_position": retained_positions,
                "output_context_token_ids": list(state_owner.context_token_ids),
                "output_positions": len(state_owner.context_token_ids),
            },
            "layers": LAYERS,
            "model_output_domain": MODEL_OUTPUT_DOMAIN,
            "token_executions": token_executions,
            "head_steps": head_steps,
            "per_generated_step_timing": [step["timing"] for step in head_steps],
            "timing_schema": {
                "per_generated_step": True,
                "cpu_wall_seconds": "host orchestration wall outside recorded Icarus child intervals",
                "icarus_wall_seconds": "iverilog compile plus vvp simulation wall intervals",
                "first_step_includes_uncached_prompt_suffix_positions": True,
            },
            "setup_timing": setup_timing,
            "overall_timing": overall_timing,
            "latency": latency,
            "detokenization_wall_seconds": detokenization_wall_seconds,
            "rtl_corrected_v_provenance": corrected_v_provenance,
            "rtl_emitted_corrected_v_consumed_downstream": True,
            "lm_head_weight": shared_weights["artifacts"],
            "process_start_count": 1,
            "software_transformer_or_logits_fallback": False,
            "elapsed_wall_seconds": overall_timing["total_wall_seconds"],
            "cpu_wall_seconds": overall_timing["cpu_wall_seconds"],
            "icarus_wall_seconds": overall_timing["icarus_wall_seconds"],
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        write_json(output / "run_summary.json", summary)
        shared_pruning = []
        rank1_shared_root = output / "shared/rank1-sidecar"
        if rank1_sidecar is not None and rank1_shared_root.is_dir():
            shared_pruning.append(
                prune_transient_execution_artifacts(rank1_shared_root)
            )
        shared_root = output / "shared"
        if shared_root.is_dir():
            shared_pruning.append(
                prune_transient_execution_artifacts(shared_root)
            )
        summary["storage_lifecycle"] = {
            "status": "TRANSIENT_SHARED_ARTIFACTS_PRUNED_AFTER_DURABLE_RESULT",
            "pruning": shared_pruning,
            "model_snapshot_copied": False,
            "tokenizer_cache_copied": False,
            "retained_classes": [
                "result_json",
                "simulator_logs",
                "rtl_observed_bytes",
            ],
        }
        write_json(output / "run_summary.json", summary)
        return summary, state_owner
    except BaseException as error:
        state_owner.valid = False
        failure = {
            "schema_version": 1,
            "status": (
                "SEALED_FAIL_OFFICIAL_RTL_GENERATION"
                if execution_mode == "official"
                else "FAIL_STAGE1_PRODUCT_RTL_GENERATION"
            ),
            "execution_mode": execution_mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "process_start_count": 1,
            "elapsed_wall_seconds": time.monotonic() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        if execution_mode == "official":
            failure["official_attempt_consumed"] = True
        write_json(output / "failure.json", failure)
        raise


def run_generation(
    output: Path,
    snapshot: Path,
    tokenizer: Any,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    *,
    execution_mode: str = "official",
) -> dict[str, Any]:
    """Run one fresh causal prefill/decode session through the Icarus backend."""
    summary, _state = run_generation_with_state(
        output,
        snapshot,
        tokenizer,
        prompt_token_ids,
        max_new_tokens,
        execution_mode=execution_mode,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ACE-2 RTL arbitrary-text generation backend utilities"
    )
    parser.add_argument(
        "--storage-lifecycle-check",
        action="store_true",
        help="run a construction-only repeated-storage check",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(
        args.storage_lifecycle_check,
        "--storage-lifecycle-check is required for direct backend invocation",
    )
    result = run_storage_lifecycle_construction_check(args.output)
    print(
        f"{result['status']} "
        f"post_prune_bytes={result['stable_post_prune_usage']['logical_bytes']} "
        "model_execution_count=0 rtl_execution_count=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
