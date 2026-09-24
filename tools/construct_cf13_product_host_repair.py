#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/ace2_chat_demo"
PACKAGE = (
    BUILD_ROOT
    / "stage1-w4a8-r6-generation-probe-product-host-repair-cf13-0001"
)
CF12_PACKAGE = (
    BUILD_ROOT / "stage1-w4a8-r6-generation-probe-cf12-attempt-0001"
)
CF12_STATE = BUILD_ROOT / f"{CF12_PACKAGE.name}-authority-state"
CF12_PREDECESSOR = (
    BUILD_ROOT / "stage1-w4a8-r6-reference-process-predecessor-cf12-0001"
)
CF12_SEAL = (
    BUILD_ROOT / "stage1-w4a8-r6-generation-probe-cf12-terminal-seal-0001"
)
CF12_TERMINAL_SEAL = CF12_SEAL / "terminal-seal.json"
CF12_TERMINAL_SEAL_SHA256 = (
    "dc3cb069ee1efe3cb210b41915ff36448670704b7e569c7d77200078c86d260a"
)
CF12_REVIEW_LATEST = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobeseal12/latest.json"
)
CF11 = BUILD_ROOT / "stage1-w4a8-r6-product-prep-cf11"
CF10 = BUILD_ROOT / "stage1-w4a8-r6-product-prep-cf10"
QUALIFICATION = (
    BUILD_ROOT / "stage1-w4a8-r6-chat-rtl-qualification-prep-cf07"
)
CONSTRAINT = ROOT / "constraints/ace2_rmsnorm_core.sdc"
BENCHMARK_INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
CF12_NAMESPACES = (
    CF12_PACKAGE,
    CF12_STATE,
    CF12_PREDECESSOR,
    CF12_SEAL,
)


HOST_RUNTIME_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


CAPTURE_CONTEXT_KEYS = frozenset(
    {
        "generation_budget",
        "input_token_count",
        "positions_persisted",
        "prefill_completed",
        "runtime_constructed",
        "runtime_module_loaded",
    }
)
CAPTURE_CONTEXT_MAX_BYTES = 1024


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class PhaseTracker:
    phase: str = "NOT_STARTED"
    values: dict[str, int | bool] = field(
        default_factory=lambda: {
            "generation_budget": 4,
            "input_token_count": 0,
            "positions_persisted": 0,
            "prefill_completed": False,
            "runtime_constructed": False,
            "runtime_module_loaded": False,
        }
    )

    def set(self, phase: str, **updates: int | bool) -> None:
        unknown = set(updates) - CAPTURE_CONTEXT_KEYS
        if unknown:
            raise ValueError(f"capture context keys are not allowlisted: {sorted(unknown)}")
        if not phase or len(phase) > 64:
            raise ValueError("phase must contain 1..64 characters")
        for key, value in updates.items():
            if type(value) not in (bool, int):
                raise TypeError(f"capture context {key} must be bool or int")
            self.values[key] = value
        self.phase = phase

    def bounded_context(self) -> dict[str, object]:
        context: dict[str, object] = {"phase": self.phase, **self.values}
        if len(canonical_bytes(context)) > CAPTURE_CONTEXT_MAX_BYTES:
            raise RuntimeError("allowlisted product-host context exceeds bound")
        return context


def _failure_capture(error: Exception, tracker: PhaseTracker) -> dict[str, object]:
    return {
        "schema": "ace2-r6-cf13-product-host-error-capture-v1",
        "status": "PRODUCT_HOST_EXCEPTION_FSYNCED_BEFORE_RTL_ENDPOINT",
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "local_context": tracker.bounded_context(),
        "privacy": {
            "allowlisted_context_only": True,
            "prompt_bytes": "EXCLUDED",
            "prompt_text": "EXCLUDED",
            "token_ids": "EXCLUDED",
            "traceback_locals": "NOT_CAPTURED",
        },
    }


def run_product_host_boundary(
    *,
    evidence_path: Path,
    capture_path: Path,
    host_runner: Callable[[PhaseTracker], dict[str, object]],
    endpoint_runner: Callable[[], dict[str, object]],
) -> int:
    tracker = PhaseTracker()
    evidence: dict[str, Any] = {
        "schema": "ace2-r6-cf13-product-rtl-evidence-v1",
        "status": "HOST_BOUNDARY_STARTED_NO_DECODE",
        "host": None,
        "endpoint": None,
        "failure_capture_sha256": None,
        "decode": "FORBIDDEN_UNTIL_FSYNCED_HOST_AND_RTL_EVIDENCE",
    }
    atomic_json(evidence_path, evidence, exclusive=True)
    host_failed = False
    try:
        evidence["host"] = host_runner(tracker)
        evidence["status"] = "HOST_EVIDENCE_FSYNCED_BEFORE_RTL_ENDPOINT"
        atomic_json(evidence_path, evidence)
    except Exception as error:
        host_failed = True
        capture = _failure_capture(error, tracker)
        atomic_json(capture_path, capture, exclusive=True)
        evidence["failure_capture_sha256"] = sha256_file(capture_path)
        evidence["status"] = "HOST_EXCEPTION_CAPTURE_FSYNCED_BEFORE_RTL_ENDPOINT"
        atomic_json(evidence_path, evidence)

    endpoint = endpoint_runner()
    evidence["endpoint"] = endpoint
    evidence["status"] = (
        "TERMINAL_HOST_FAILURE_WITH_ENDPOINT_RECEIPT_FSYNCED"
        if host_failed
        else "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE"
    )
    atomic_json(evidence_path, evidence)
    return 2 if host_failed or endpoint.get("exit_code") != 0 else 0
'''


PRODUCT_PROBE_SOURCE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from product_host_runtime import (
    PhaseTracker,
    canonical_bytes,
    run_product_host_boundary,
    sha256_file,
)


PACKAGE = Path(__file__).resolve().parent


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import pinned runtime: {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_bytes({"dtype": str(value.dtype), "shape": list(value.shape)}))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def top_k(logits: Any, count: int = 16) -> list[dict[str, object]]:
    import torch

    order = torch.argsort(logits, descending=True, stable=True)[:count]
    return [
        {"rank": rank, "token_id": int(token_id), "logit": float(logits[token_id])}
        for rank, token_id in enumerate(order.tolist(), 1)
    ]


def run_endpoint(argv: list[str], timeout: int) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise RuntimeError("RTL endpoint terminal timeout") from error
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rtl_positions(records: list[dict[str, Any]], prompt_tokens: int) -> list[dict[str, Any]]:
    positions = []
    for ordinal in range(4):
        token_step = prompt_tokens - 1 + ordinal
        selected = [item for item in records if item.get("token_step") == token_step]
        kv = [item for item in selected if item.get("operator") == "kv_write"]
        logits = [item for item in selected if item.get("operator") == "lm_head_tile"]
        if len(kv) != 24 or len(logits) != 4748:
            raise RuntimeError("RTL position lacks 24 KV writes or 4748 logit tiles")
        aggregate = hashlib.sha256()
        for item in logits:
            aggregate.update(
                f"{item['ordinal']}:{item['destination_sha256']}\n".encode("ascii")
            )
        positions.append(
            {
                "ordinal": ordinal,
                "absolute_position": token_step,
                "selected_token_id": logits[-1]["generated_token_after"],
                "per_layer_kv_digests": [item["destination_sha256"] for item in kv],
                "authenticated_logit_tile_count": 4748,
                "authenticated_logit_tile_aggregate_sha256": aggregate.hexdigest(),
            }
        )
    return positions


def verify_source_bindings(bindings: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(bindings["source_cf12_bindings"]["path"])
    if sha256_file(source_path) != bindings["source_cf12_bindings"]["sha256"]:
        raise RuntimeError("source CF12 bindings changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    prompt = source["prompt"]
    public_prompt = bindings["prompt_binding"]
    if (
        prompt["sha256"] != public_prompt["sha256"]
        or prompt["byte_count"] != public_prompt["byte_count"]
        or len(prompt["chat_template_token_ids"]) != public_prompt["token_count"]
        or prompt["chat_template_token_ids_sha256"]
        != public_prompt["chat_template_token_ids_sha256"]
    ):
        raise RuntimeError("source prompt binding changed")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    args = parser.parse_args()
    bindings = json.loads((PACKAGE / "bindings.json").read_text(encoding="utf-8"))
    source = verify_source_bindings(bindings)

    def host_runner(tracker: PhaseTracker) -> dict[str, object]:
        import torch

        input_ids = list(source["prompt"]["chat_template_token_ids"])
        tracker.set(
            "RUNTIME_MODULE_LOADING",
            generation_budget=4,
            input_token_count=len(input_ids),
        )
        runtime_module = load_module(
            Path(bindings["accepted_runtime"]["path"]), "_ace2_cf13_runtime"
        )
        tracker.set("RUNTIME_MODULE_LOADED", runtime_module_loaded=True)
        preflight = json.loads(
            Path(bindings["accepted_runtime"]["preflight_hashes_path"]).read_text(
                encoding="utf-8"
            )
        )
        tracker.set("RUNTIME_CONSTRUCTION")
        runtime = runtime_module.create_runtime(
            hardware_export_manifest=Path(
                bindings["accepted_runtime"]["hardware_export_manifest"]
            ),
            accepted_interface_hashes=preflight["accepted_interfaces"],
            accepted_tokenizer=preflight["accepted_tokenizer"],
        )
        tracker.set("RUNTIME_CONSTRUCTED", runtime_constructed=True)
        prompt = bytes.fromhex(source["prompt"]["bytes_hex"]).decode(
            "utf-8", errors="strict"
        )
        tracker.set("PREFILL")
        boundary = runtime.prefill(
            [{"role": "user", "content": prompt}], expected_input_ids=input_ids
        )
        tracker.set("PREFILL_COMPLETED", prefill_completed=True)
        sequence = list(input_ids)
        positions = []
        for ordinal in range(4):
            tracker.set(
                "POSITION_ZERO_PREPARATION"
                if ordinal == 0
                else "POSITION_PREPARATION",
                positions_persisted=len(positions),
            )
            cacheful_logits = torch.tensor(boundary["logits"], dtype=torch.float32)
            device = next(runtime.model.parameters()).device
            full_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            with torch.inference_mode():
                free = runtime.model(
                    input_ids=full_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            cache_free_logits = free.logits[0, -1].detach().cpu().to(torch.float32)
            if cache_free_logits.numel() != 151936 or cacheful_logits.numel() != 151936:
                raise RuntimeError("product logits do not cover 151936 rows")
            states = free.hidden_states
            if states is None or len(states[1:]) != 24:
                raise RuntimeError("product position lacks 24 layer states")
            if len(boundary["layer_state_digests"]) != 24:
                raise RuntimeError("product position lacks 24 KV digests")
            ranked = top_k(cacheful_logits)
            selected = ranked[0]["token_id"]
            free_ranked = top_k(cache_free_logits)
            position = {
                "ordinal": ordinal,
                "absolute_position": len(input_ids) - 1 + ordinal,
                "selected_token_id": selected,
                "full_logit_count": 151936,
                "top_k": ranked,
                "selected_rank": 1,
                "selected_margin": ranked[0]["logit"] - ranked[1]["logit"],
                "prompt_bytes_sha256": source["prompt"]["sha256"],
                "chat_template_token_ids_sha256": source["prompt"][
                    "chat_template_token_ids_sha256"
                ],
                "embedding_input_ids_sha256": hashlib.sha256(
                    canonical_bytes(sequence)
                ).hexdigest(),
                "per_layer_state_digests": [
                    tensor_digest(state[:, -1]) for state in states[1:]
                ],
                "per_layer_kv_digests": list(boundary["layer_state_digests"]),
                "logits_sha256": tensor_digest(cacheful_logits),
                "cache_comparison": {
                    "cache_free_selected_token_id": free_ranked[0]["token_id"],
                    "cacheful_selected_token_id": selected,
                    "cache_free_logits_sha256": tensor_digest(cache_free_logits),
                    "cacheful_logits_sha256": tensor_digest(cacheful_logits),
                },
            }
            positions.append(position)
            tracker.set("HOST_POSITION_PREPARED", positions_persisted=len(positions))
            sequence.append(selected)
            if ordinal < 3:
                tracker.set("CACHEFUL_DECODE", positions_persisted=len(positions))
                boundary = runtime.decode(selected)
        return {"positions": positions, "position_count": len(positions)}

    def endpoint_runner() -> dict[str, object]:
        endpoint = bindings["endpoint"]
        argv = list(endpoint["argv_prefix"]) + ["--output", str(args.endpoint_output)]
        completed = run_endpoint(argv, endpoint["timeout_seconds"])
        receipt: dict[str, object] = {
            "schema": "ace2-r6-cf13-endpoint-invocation-receipt-v1",
            "argv_sha256": hashlib.sha256(canonical_bytes(argv)).hexdigest(),
            "exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "rtl_positions": [],
        }
        if completed.returncode == 0:
            records = load_jsonl(args.endpoint_output / "commands.jsonl")
            receipt["rtl_positions"] = rtl_positions(
                records, bindings["prompt_binding"]["token_count"]
            )
        return receipt

    return run_product_host_boundary(
        evidence_path=args.output,
        capture_path=args.capture,
        host_runner=host_runner,
        endpoint_runner=endpoint_runner,
    )


if __name__ == "__main__":
    raise SystemExit(main())
'''


LOADER_CONTRACT_FIXTURE_SOURCE = r'''
from __future__ import annotations

import sys


FIXTURE_KIND = "DETERMINISTIC_NON_MODEL_PRE_POSITION_ZERO_LOADER_CONTRACT"
MODEL_LOADED = False
RTL_ENDPOINT_INVOKED = False


def position_zero_contract() -> dict[str, object]:
    return {
        "positions": [{"ordinal": 0}],
        "position_count": 1,
        "fixture_kind": FIXTURE_KIND,
        "model_loaded": MODEL_LOADED,
        "rtl_endpoint_invoked": RTL_ENDPOINT_INVOKED,
    }


registered_contract = getattr(
    sys.modules.get(__name__), "position_zero_contract", None
)
POSITION_ZERO_CONTRACT = registered_contract()
'''


TEST_SOURCE = r'''
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
LOADER_FIXTURE = PACKAGE / "loader_contract_fixture.py"


def load_runtime():
    specification = importlib.util.spec_from_file_location(
        "_cf13_product_host_runtime", PACKAGE / "product_host_runtime.py"
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_probe(runtime):
    name = "_cf13_product_probe_fixture"
    previous_runtime = sys.modules.get("product_host_runtime")
    sys.modules["product_host_runtime"] = runtime
    try:
        specification = importlib.util.spec_from_file_location(
            name, PACKAGE / "product_probe.py"
        )
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(name, None)
        if previous_runtime is None:
            sys.modules.pop("product_host_runtime", None)
        else:
            sys.modules["product_host_runtime"] = previous_runtime


def load_with_pre_repair_order(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import pinned runtime: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ProductHostBoundaryTests(unittest.TestCase):
    def test_loader_contract_failure_capture_and_repaired_success(self) -> None:
        runtime = load_runtime()
        probe = load_probe(runtime)
        sentinel = b"PRIVATE-PROMPT-SENTINEL-CF13"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            capture = root / "capture.json"
            order = []
            pre_repair_name = "_cf13_pre_repair_loader_contract"
            repaired_name = "_cf13_repaired_loader_contract"
            sys.modules.pop(pre_repair_name, None)
            sys.modules.pop(repaired_name, None)

            def failing_host(tracker):
                private_prompt = sentinel.decode("ascii")
                self.assertTrue(private_prompt)
                tracker.set(
                    "POSITION_ZERO_PREPARATION",
                    input_token_count=17,
                    runtime_module_loaded=True,
                    runtime_constructed=True,
                    prefill_completed=True,
                )
                module = load_with_pre_repair_order(
                    LOADER_FIXTURE, pre_repair_name
                )
                return module.POSITION_ZERO_CONTRACT

            def endpoint():
                self.assertTrue(capture.is_file())
                order.append("endpoint")
                return {"exit_code": 0, "fixture": True}

            original = runtime.atomic_json

            def recording_atomic(path, value, *, exclusive=False):
                original(path, value, exclusive=exclusive)
                if path == capture:
                    order.append("capture_fsynced")

            with mock.patch.object(runtime, "atomic_json", recording_atomic):
                result = runtime.run_product_host_boundary(
                    evidence_path=evidence,
                    capture_path=capture,
                    host_runner=failing_host,
                    endpoint_runner=endpoint,
                )
            self.assertEqual(result, 2)
            self.assertNotIn(pre_repair_name, sys.modules)
            self.assertEqual(order, ["capture_fsynced", "endpoint"])
            raw = capture.read_bytes()
            self.assertNotIn(sentinel, raw)
            payload = json.loads(raw)
            self.assertEqual(payload["exception"]["type"], "TypeError")
            self.assertEqual(
                payload["exception"]["message"],
                "'NoneType' object is not callable",
            )
            self.assertIn("Traceback (most recent call last):", payload["exception"]["traceback"])
            self.assertIn(
                "POSITION_ZERO_CONTRACT = registered_contract()",
                payload["exception"]["traceback"],
            )
            self.assertEqual(
                payload["local_context"]["phase"], "POSITION_ZERO_PREPARATION"
            )
            self.assertLessEqual(
                len(runtime.canonical_bytes(payload["local_context"])),
                runtime.CAPTURE_CONTEXT_MAX_BYTES,
            )
            self.assertEqual(
                set(payload["local_context"]) - {"phase"},
                runtime.CAPTURE_CONTEXT_KEYS,
            )

            def successful_host(tracker):
                tracker.set(
                    "POSITION_ZERO_PREPARATION",
                    input_token_count=17,
                    runtime_module_loaded=True,
                    runtime_constructed=True,
                    prefill_completed=True,
                )
                module = probe.load_module(LOADER_FIXTURE, repaired_name)
                tracker.set("HOST_POSITION_PREPARED", positions_persisted=1)
                return module.POSITION_ZERO_CONTRACT

            success_capture = root / "success-capture.json"
            try:
                success_result = runtime.run_product_host_boundary(
                    evidence_path=root / "success-evidence.json",
                    capture_path=success_capture,
                    host_runner=successful_host,
                    endpoint_runner=lambda: {"exit_code": 0, "fixture": True},
                )
            finally:
                sys.modules.pop(repaired_name, None)
            self.assertEqual(success_result, 0)
            self.assertFalse(success_capture.exists())
            success_payload = json.loads(
                (root / "success-evidence.json").read_text()
            )
            self.assertEqual(
                success_payload["status"],
                "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE",
            )
            self.assertEqual(success_payload["host"]["position_count"], 1)
            self.assertFalse(success_payload["host"]["model_loaded"])
            self.assertFalse(success_payload["host"]["rtl_endpoint_invoked"])

    def test_capture_persistence_error_fails_closed_before_endpoint(self) -> None:
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            capture = root / "capture.json"
            endpoint = mock.Mock(return_value={"exit_code": 0})
            original = runtime.atomic_json
            module_name = "_cf13_fail_closed_loader_contract"
            sys.modules.pop(module_name, None)

            def fail_capture(path, value, *, exclusive=False):
                if path == capture:
                    raise OSError("fixture fsync failure")
                original(path, value, exclusive=exclusive)

            def failing_host(tracker):
                tracker.set("POSITION_ZERO_PREPARATION")
                module = load_with_pre_repair_order(LOADER_FIXTURE, module_name)
                return module.POSITION_ZERO_CONTRACT

            with mock.patch.object(runtime, "atomic_json", fail_capture):
                with self.assertRaisesRegex(OSError, "fixture fsync failure"):
                    runtime.run_product_host_boundary(
                        evidence_path=evidence,
                        capture_path=capture,
                        host_runner=failing_host,
                        endpoint_runner=endpoint,
                    )
            endpoint.assert_not_called()
            self.assertNotIn(module_name, sys.modules)

    def test_repaired_loader_registers_identical_contract_during_exec(self) -> None:
        runtime = load_runtime()
        probe = load_probe(runtime)
        name = "_cf13_direct_repaired_loader_contract"
        sys.modules.pop(name, None)
        try:
            module = probe.load_module(LOADER_FIXTURE, name)
            self.assertIs(sys.modules[name], module)
            self.assertEqual(module.POSITION_ZERO_CONTRACT["position_count"], 1)
            self.assertFalse(module.POSITION_ZERO_CONTRACT["model_loaded"])
            self.assertFalse(module.POSITION_ZERO_CONTRACT["rtl_endpoint_invoked"])
        finally:
            sys.modules.pop(name, None)

    def test_atomic_json_fsyncs_file_then_directory(self) -> None:
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture.json"
            calls = []
            original = os.fsync

            def recording_fsync(descriptor):
                calls.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
                return original(descriptor)

            with mock.patch.object(runtime.os, "fsync", recording_fsync):
                runtime.atomic_json(output, {"status": "fixture"}, exclusive=True)
            self.assertEqual(calls, [False, True])


class PackageClosureTests(unittest.TestCase):
    def test_cf12_inventory_and_source_constraint_closure_match(self) -> None:
        runtime = load_runtime()
        inventory = json.loads(
            (PACKAGE / "cf12-read-only-inventory.json").read_text()
        )
        for relative, expected in inventory["entries"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(runtime.sha256_file(path), expected["sha256"], relative)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected["mode"], relative)
        closure = json.loads(
            (PACKAGE / "source-constraint-manifest.json").read_text()
        )
        for relative, expected in closure["files"].items():
            path = PACKAGE / relative if expected["scope"] == "package" else ROOT / relative
            self.assertEqual(runtime.sha256_file(path), expected["sha256"], relative)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected["mode"], relative)

    def test_contract_has_zero_authority_and_unchanged_stage_rtl_policy(self) -> None:
        contract = json.loads((PACKAGE / "contract.json").read_text())
        self.assertEqual(contract["source_terminal_seal_sha256"], "dc3cb069ee1efe3cb210b41915ff36448670704b7e569c7d77200078c86d260a")
        self.assertEqual(contract["execution_authority"], "NONE")
        self.assertEqual(contract["execution_limit_before_l2_acceptance"], 0)
        self.assertEqual(contract["cf12_retry_replay_resume_relaunch"], "PERMANENTLY_FORBIDDEN")
        self.assertEqual(contract["rtl_change"], "NONE")
        self.assertEqual(contract["public_rtl_contract"], "UNCHANGED_14_PARAMETERS_64_PORTS")
        self.assertEqual(contract["stage1"], "OPEN")
        self.assertEqual(contract["stage2"], "FORBIDDEN")
        for name in (
            "authority.json",
            "authorization-consumed.json",
            "attempt-0001",
            "execution-registry.json",
            "launch.sh",
            "product-result.json",
        ):
            self.assertFalse((PACKAGE / name).exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


VERIFIER_SOURCE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest import mock

import product_host_runtime as runtime
from product_probe import load_module


PACKAGE = Path(__file__).resolve().parent
LOADER_FIXTURE = PACKAGE / "loader_contract_fixture.py"
PRIVATE_SENTINEL = b"CF13-PRIVATE-PROMPT-BYTES-MUST-NOT-APPEAR"


def load_with_pre_repair_order(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import pinned runtime: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ace2-cf13-boundary-") as temporary:
        root = Path(temporary)
        capture = root / "capture.json"
        endpoint_observations = []
        persistence_endpoint_observations = []
        fsync_kinds = []
        fsync_sequences = {}
        pre_repair_name = "_cf13_verifier_pre_repair_loader_contract"
        repaired_name = "_cf13_verifier_repaired_loader_contract"
        persistence_name = "_cf13_verifier_fail_closed_loader_contract"
        for name in (pre_repair_name, repaired_name, persistence_name):
            sys.modules.pop(name, None)

        def failing_host(tracker):
            private_prompt = PRIVATE_SENTINEL.decode("ascii")
            if not private_prompt:
                raise AssertionError("unreachable")
            tracker.set(
                "POSITION_ZERO_PREPARATION",
                input_token_count=17,
                runtime_module_loaded=True,
                runtime_constructed=True,
                prefill_completed=True,
            )
            module = load_with_pre_repair_order(
                LOADER_FIXTURE, pre_repair_name
            )
            return module.POSITION_ZERO_CONTRACT

        def endpoint():
            endpoint_observations.append(capture.read_bytes())
            return {"exit_code": 0, "fixture": True}

        original_fsync = runtime.os.fsync
        original_atomic = runtime.atomic_json

        def recording_fsync(descriptor):
            fsync_kinds.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
            return original_fsync(descriptor)

        def recording_atomic(path, value, *, exclusive=False):
            start = len(fsync_kinds)
            original_atomic(path, value, exclusive=exclusive)
            fsync_sequences.setdefault(path, []).append(fsync_kinds[start:])

        with (
            mock.patch.object(runtime.os, "fsync", recording_fsync),
            mock.patch.object(runtime, "atomic_json", recording_atomic),
        ):
            failure_exit = runtime.run_product_host_boundary(
                evidence_path=root / "failure-evidence.json",
                capture_path=capture,
                host_runner=failing_host,
                endpoint_runner=endpoint,
            )
        payload = json.loads(capture.read_text(encoding="utf-8"))
        checks = {
            "failure_exit_is_terminal": failure_exit == 2,
            "pre_repair_loader_contract_failed": pre_repair_name not in sys.modules,
            "capture_precedes_endpoint": len(endpoint_observations) == 1,
            "capture_file_then_directory_fsync": fsync_sequences.get(capture)
            == [[False, True]],
            "exact_type": payload["exception"]["type"] == "TypeError",
            "exact_message": payload["exception"]["message"]
            == "'NoneType' object is not callable",
            "full_traceback": "Traceback (most recent call last):"
            in payload["exception"]["traceback"]
            and "POSITION_ZERO_CONTRACT = registered_contract()"
            in payload["exception"]["traceback"],
            "bounded_allowlisted_context": set(payload["local_context"])
            == runtime.CAPTURE_CONTEXT_KEYS | {"phase"}
            and len(runtime.canonical_bytes(payload["local_context"]))
            <= runtime.CAPTURE_CONTEXT_MAX_BYTES,
            "private_prompt_excluded": PRIVATE_SENTINEL not in capture.read_bytes(),
        }

        success_capture = root / "success-capture.json"

        def successful_host(tracker):
            tracker.set(
                "POSITION_ZERO_PREPARATION",
                input_token_count=17,
                runtime_module_loaded=True,
                runtime_constructed=True,
                prefill_completed=True,
            )
            module = load_module(LOADER_FIXTURE, repaired_name)
            tracker.set("HOST_POSITION_PREPARED", positions_persisted=1)
            return module.POSITION_ZERO_CONTRACT

        try:
            success_exit = runtime.run_product_host_boundary(
                evidence_path=root / "success-evidence.json",
                capture_path=success_capture,
                host_runner=successful_host,
                endpoint_runner=lambda: {"exit_code": 0, "fixture": True},
            )
            success = json.loads(
                (root / "success-evidence.json").read_text(encoding="utf-8")
            )
        finally:
            sys.modules.pop(repaired_name, None)
        checks["repaired_success_path"] = (
            success_exit == 0
            and not success_capture.exists()
            and success["host"]["position_count"] == 1
            and success["host"]["model_loaded"] is False
            and success["host"]["rtl_endpoint_invoked"] is False
        )

        fail_closed_capture = root / "fail-closed-capture.json"

        def fail_closed_atomic(path, value, *, exclusive=False):
            if path == fail_closed_capture:
                raise OSError("fixture fsync failure")
            original_atomic(path, value, exclusive=exclusive)

        def fail_closed_host(tracker):
            tracker.set("POSITION_ZERO_PREPARATION")
            module = load_with_pre_repair_order(
                LOADER_FIXTURE, persistence_name
            )
            return module.POSITION_ZERO_CONTRACT

        persistence_error = None
        try:
            with mock.patch.object(runtime, "atomic_json", fail_closed_atomic):
                runtime.run_product_host_boundary(
                    evidence_path=root / "fail-closed-evidence.json",
                    capture_path=fail_closed_capture,
                    host_runner=fail_closed_host,
                    endpoint_runner=lambda: persistence_endpoint_observations.append(
                        True
                    ),
                )
        except OSError as error:
            persistence_error = str(error)
        checks["persistence_failure_fails_closed"] = (
            persistence_error == "fixture fsync failure"
            and not persistence_endpoint_observations
            and persistence_name not in sys.modules
        )
        status = "PASS" if all(checks.values()) else "FAIL"
        print(
            json.dumps(
                {
                    "schema": "ace2-r6-cf13-product-host-boundary-verification-v1",
                    "status": status,
                    "model_loaded": False,
                    "rtl_endpoint_invoked": False,
                    "checks": checks,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


MAKEFILE_SOURCE = r'''
.PHONY: build test verify check

PYTHON ?= python3

build:
	$(PYTHON) -B -c 'from pathlib import Path; [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in (Path("product_host_runtime.py"), Path("product_probe.py"), Path("loader_contract_fixture.py"), Path("verify_product_host_boundary.py"), Path("tests/test_cf13_product_host.py"))]'

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v

verify:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B verify_product_host_boundary.py

check: build test verify
'''


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_bytes(path: Path, value: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(mode)


def write_text(path: Path, value: str, mode: int) -> None:
    write_bytes(path, textwrap.dedent(value).lstrip().encode("utf-8"), mode)


def write_json(path: Path, value: object, mode: int = 0o444) -> None:
    write_bytes(path, canonical_bytes(value), mode)


def namespace_files(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        result[item.relative_to(ROOT).as_posix()] = {
            "sha256": sha256_file(item),
            "mode": stat.S_IMODE(item.stat().st_mode),
            "size": item.stat().st_size,
        }
    return result


def package_manifest(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        result[item.relative_to(path).as_posix()] = {
            "sha256": sha256_file(item),
            "mode": stat.S_IMODE(item.stat().st_mode),
        }
    return result


def validate_cf12() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CF12_TERMINAL_SEAL) != CF12_TERMINAL_SEAL_SHA256:
        raise RuntimeError("CF12 terminal seal hash differs")
    seal = load_object(CF12_TERMINAL_SEAL)
    if (
        seal.get("classification", {}).get("classification")
        != "UNLOCALIZED_PRODUCT_HOST_TRAJECTORY_TYPE_ERROR_BEFORE_POSITION_0"
        or seal.get("authority", {}).get("retry_replay_resume_relaunch")
        != "PERMANENTLY_FORBIDDEN"
        or seal.get("fsynced_frontier", {}).get("product", {}).get("failure_type")
        != "TypeError"
    ):
        raise RuntimeError("CF12 terminal seal classification differs")
    latest = load_object(CF12_REVIEW_LATEST)
    if latest.get("kind") != "handoff_ref":
        raise RuntimeError("CF12 terminal-seal review handoff is unavailable")
    handoff_path = Path(latest["handoff"]["path"])
    handoff = load_object(handoff_path)
    if (
        handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("CF12 terminal seal lacks independent reviewer acceptance")
    return seal, {
        "latest_path": str(CF12_REVIEW_LATEST),
        "latest_sha256": sha256_file(CF12_REVIEW_LATEST),
        "handoff_path": str(handoff_path),
        "handoff_sha256": sha256_file(handoff_path),
        "producer_role": "reviewer",
        "status": "done",
    }, load_object(CF12_PACKAGE / "bindings.json")


def source_entry(path: Path, scope: str = "repository") -> dict[str, object]:
    return {
        "sha256": sha256_file(path),
        "mode": stat.S_IMODE(path.stat().st_mode),
        "scope": scope,
    }


def build_package(
    target: Path,
    *,
    seal: dict[str, Any],
    review: dict[str, Any],
    source_bindings: dict[str, Any],
    cf12_inventory: dict[str, dict[str, object]],
) -> None:
    target.mkdir()
    write_text(target / "product_host_runtime.py", HOST_RUNTIME_SOURCE, 0o444)
    write_text(target / "product_probe.py", PRODUCT_PROBE_SOURCE, 0o444)
    write_text(
        target / "loader_contract_fixture.py",
        LOADER_CONTRACT_FIXTURE_SOURCE,
        0o444,
    )
    write_text(target / "tests/test_cf13_product_host.py", TEST_SOURCE, 0o444)
    write_text(target / "verify_product_host_boundary.py", VERIFIER_SOURCE, 0o555)
    write_text(target / "Makefile", MAKEFILE_SOURCE, 0o444)
    (target / "tests").chmod(0o555)

    rtl_provenance = load_object(CF11 / "rtl-provenance.json")
    rtl_files = rtl_provenance["rtl_source_closure"]["files"]
    for relative, expected in rtl_files.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"accepted RTL source changed: {relative}")
    prompt = source_bindings["prompt"]
    endpoint_argv = list(source_bindings["endpoint"]["argv"])
    output_index = endpoint_argv.index("--output")
    if output_index + 1 >= len(endpoint_argv):
        raise RuntimeError("CF12 endpoint output argument is malformed")
    del endpoint_argv[output_index : output_index + 2]
    bindings = {
        "schema": "ace2-r6-cf13-product-host-bindings-v1",
        "source_cf12_bindings": {
            "path": str(CF12_PACKAGE / "bindings.json"),
            "sha256": sha256_file(CF12_PACKAGE / "bindings.json"),
        },
        "prompt_binding": {
            "sha256": prompt["sha256"],
            "byte_count": prompt["byte_count"],
            "token_count": len(prompt["chat_template_token_ids"]),
            "chat_template_token_ids_sha256": prompt[
                "chat_template_token_ids_sha256"
            ],
            "prompt_bytes_in_cf13": "NOT_COPIED",
        },
        "model": source_bindings["model"],
        "accepted_runtime": source_bindings["accepted_runtime"],
        "endpoint": {
            "argv_prefix": endpoint_argv,
            "timeout_seconds": 7200,
            "total_command_count": 1_306_104,
            "software_fallback": "FORBIDDEN",
        },
        "official_descriptor_sha256": source_bindings[
            "official_descriptor_sha256"
        ],
        "final_export_manifest_sha256": source_bindings[
            "final_export_manifest_sha256"
        ],
        "rtl_source_closure_sha256": source_bindings[
            "rtl_source_closure_sha256"
        ],
    }
    write_json(target / "bindings.json", bindings)

    inventory_payload = {
        "schema": "ace2-r6-cf13-cf12-read-only-inventory-v1",
        "status": "BOUND_FOR_BEFORE_AFTER_BYTE_MODE_COMPARISON",
        "source_terminal_seal_sha256": CF12_TERMINAL_SEAL_SHA256,
        "entries": cf12_inventory,
        "entry_count": len(cf12_inventory),
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
    }
    write_json(target / "cf12-read-only-inventory.json", inventory_payload)

    package_sources = (
        "Makefile",
        "bindings.json",
        "loader_contract_fixture.py",
        "product_host_runtime.py",
        "product_probe.py",
        "tests/test_cf13_product_host.py",
        "verify_product_host_boundary.py",
    )
    closure_files: dict[str, dict[str, object]] = {
        relative: source_entry(target / relative, "package")
        for relative in package_sources
    }
    repository_sources = (
        "constraints/ace2_rmsnorm_core.sdc",
        "design/BENCHMARK_INTERFACE.json",
        "tools/construct_cf13_product_host_repair.py",
        (
            "build/ace2_chat_demo/"
            "stage1-w4a8-r6-chat-rtl-qualification-prep-cf07/accepted_runtime.py"
        ),
        (
            "build/ace2_chat_demo/"
            "stage1-w4a8-r6-chat-rtl-qualification-prep-cf07/preflight-hashes.json"
        ),
        "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf10/product_driver.py",
        "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf11/rtl-provenance.json",
    )
    closure_files.update(
        {relative: source_entry(ROOT / relative) for relative in repository_sources}
    )
    closure_files.update(
        {relative: source_entry(ROOT / relative) for relative in rtl_files}
    )
    closure = {
        "schema": "ace2-r6-cf13-source-constraint-closure-v1",
        "status": "PASS_EXACT_HASH_AND_MODE_CLOSURE",
        "files": closure_files,
        "file_count": len(closure_files),
        "rtl_source_closure_sha256": rtl_provenance["rtl_source_closure"][
            "sha256"
        ],
        "constraint_sha256": sha256_file(CONSTRAINT),
        "benchmark_interface_sha256": sha256_file(BENCHMARK_INTERFACE),
        "rtl_change": "NONE",
        "canonical_sky130_synthesis_opensta": "NOT_RERUN_NO_RTL_OR_CONSTRAINT_CHANGE",
    }
    write_json(target / "source-constraint-manifest.json", closure)

    contract = {
        "schema": "ace2-r6-cf13-product-host-runtime-repair-contract-v1",
        "identity": PACKAGE.name,
        "status": "CONSTRUCTED_PENDING_CANONICAL_INDEPENDENT_L2",
        "source_terminal_seal_sha256": CF12_TERMINAL_SEAL_SHA256,
        "source_terminal_classification": seal["classification"]["classification"],
        "source_terminal_review": review,
        "cf12_inventory_sha256": sha256_bytes(canonical_bytes(inventory_payload)),
        "cf12_preservation": "BYTE_AND_MODE_UNCHANGED_READ_ONLY",
        "cf12_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "capture": {
            "boundary": "PRODUCT_HOST_BEFORE_RTL_ENDPOINT",
            "exception_type": "EXACT",
            "exception_message": "EXACT",
            "traceback": "FULL_WITHOUT_LOCALS",
            "local_context": "BOUNDED_ALLOWLIST_ONLY_MAX_1024_BYTES",
            "prompt_bytes": "EXCLUDED",
            "persistence": "ATOMIC_FILE_FSYNC_AND_PARENT_DIRECTORY_FSYNC",
            "persistence_failure": "FAIL_CLOSED_BEFORE_ENDPOINT",
        },
        "fixture": {
            "kind": "DETERMINISTIC_NON_MODEL_PRODUCT_HOST_BOUNDARY",
            "artifact": "loader_contract_fixture.py",
            "pre_repair_loader_order": "EXEC_MODULE_BEFORE_SYS_MODULES_REGISTRATION",
            "repaired_loader_order": "SYS_MODULES_REGISTRATION_BEFORE_EXEC_MODULE",
            "pre_repair_exception": {
                "type": "TypeError",
                "message": "'NoneType' object is not callable",
            },
            "model_loaded": False,
            "rtl_endpoint_invoked": False,
            "private_prompt_sentinel": "MUST_BE_ABSENT_FROM_CAPTURE",
            "verification": "verify_product_host_boundary.py",
        },
        "execution_authority": "NONE",
        "execution_limit_before_l2_acceptance": 0,
        "attempt_state": "NOT_CREATED",
        "registry": "NOT_CREATED",
        "model_trajectory": "NOT_EXECUTED",
        "rtl_endpoint": "NOT_EXECUTED",
        "official_model_tokenizer": "UNCHANGED",
        "w4a8_arithmetic": "UNCHANGED",
        "evaluator_policy": "UNCHANGED",
        "one_command_chat_goal": "UNCHANGED",
        "rtl_change": "NONE",
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "ordered_operator_frontier": "UNCHANGED",
        "non_sram_area_limit_mm2": 2.0,
        "minimum_frequency_mhz": 100,
        "source_constraint_manifest_sha256": sha256_file(
            target / "source-constraint-manifest.json"
        ),
        "tool_versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "make": subprocess.run(
                ["make", "--version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0],
        },
        "review": {
            "required_level": "INDEPENDENT_L2",
            "status": "PENDING",
            "authority_before_acceptance": "FORBIDDEN",
        },
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
    }
    write_json(target / "contract.json", contract)

    committed = seal["fsynced_frontier"]["rtl"]["committed_command_count"]
    total = seal["fsynced_frontier"]["rtl"]["total_commands"]
    timeout = 7200
    optimization_plan = {
        "schema": "ace2-r6-cf13-endpoint-optimization-plan-v1",
        "status": "PLAN_ONLY_NO_CAUSAL_BOTTLENECK_CLAIM",
        "observed_evidence": {
            "timeout_seconds": timeout,
            "committed_command_count": committed,
            "last_committed_ordinal": seal["fsynced_frontier"]["rtl"][
                "last_committed_ordinal"
            ],
            "total_command_count": total,
            "observed_commands_per_second": committed / timeout,
            "required_commands_per_second": total / timeout,
            "required_speedup_ratio": total / committed,
            "simulator_cycles": 3_262_958_766,
            "source_terminal_seal_sha256": CF12_TERMINAL_SEAL_SHA256,
        },
        "diagnosis": (
            "CAUSE_UNCLEAR_WITHOUT_PHASE_TIMING_PROFILE_OR_CONTROLLED_AB; "
            "PER_COMMAND_PERSISTENCE_AND_JSONL_VOLUME_ARE HYPOTHESES_ONLY"
        ),
        "controlled_future_experiments": [
            {
                "id": "PHASE_AND_WAIT_PROFILE",
                "method": (
                    "On a fresh non-authority deterministic command fixture, measure "
                    "simulator compute, command marshaling, JSON serialization, write, "
                    "fsync, and wait time separately."
                ),
                "acceptance": "ATTRIBUTES_A_MATERIAL_SHARE_OF_ELAPSED_TIME",
            },
            {
                "id": "SIMULATOR_HOT_PATH_AB",
                "method": (
                    "Profile a fixed command-prefix baseline, optimize only measured "
                    "Verilator/runtime hot paths, and compare command outputs, errors, "
                    "cycle counts, and destination hashes bit-for-bit."
                ),
                "acceptance": "BIT_EXACT_AND_MATERIAL_WALL_TIME_REDUCTION",
            },
            {
                "id": "PERSISTENCE_BATCHING_AB",
                "method": (
                    "Compare baseline persistence with bounded batching only if the "
                    "same crash-recovery frontier and required durable boundaries are "
                    "proven; reject any variant that loses an accepted durable command."
                ),
                "acceptance": "NO_DURABILITY_OR_COMPLETION_CRITERION_WEAKENING",
            },
            {
                "id": "FULL_ENDPOINT_CONFIRMATION",
                "method": (
                    "After separate review and authority, run the unchanged 1,306,104-"
                    "command endpoint with the same visible 7,200-second timeout."
                ),
                "acceptance": "HARDWARE_RTL_COMPLETION_WITHOUT_SOFTWARE_FALLBACK",
            },
        ],
        "timeout_policy": "UNCHANGED_VISIBLE_7200_SECONDS",
        "software_fallback": "FORBIDDEN_AS_COMPLETION",
        "completion_criteria": "UNCHANGED",
    }
    write_json(target / "optimization-plan.json", optimization_plan)

    construction = {
        "schema": "ace2-r6-cf13-construction-report-v1",
        "status": "PASS_FRESH_ADDITIVE_ZERO_AUTHORITY_ZERO_PRODUCT_RTL_EXECUTION",
        "source_terminal_seal_sha256": CF12_TERMINAL_SEAL_SHA256,
        "cf12_inventory_sha256": sha256_file(
            target / "cf12-read-only-inventory.json"
        ),
        "model_trajectory": "NOT_EXECUTED",
        "rtl_endpoint": "NOT_EXECUTED",
        "authority_count": 0,
        "attempt_count": 0,
        "registry_count": 0,
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    write_json(target / "construction-report.json", construction)

    review_targets = (
        "bindings.json",
        "cf12-read-only-inventory.json",
        "construction-report.json",
        "contract.json",
        "Makefile",
        "loader_contract_fixture.py",
        "optimization-plan.json",
        "product_host_runtime.py",
        "product_probe.py",
        "source-constraint-manifest.json",
        "tests/test_cf13_product_host.py",
        "verify_product_host_boundary.py",
    )
    write_json(
        target / "review-request.json",
        {
            "schema": "ace2-r6-cf13-canonical-independent-l2-review-request-v1",
            "status": "PENDING_INDEPENDENT_L2",
            "required_checks": [
                "CF12_BEFORE_AFTER_HASH_AND_MODE_EQUAL",
                "CAPTURE_COMPLETENESS_PRIVACY_FSYNC_ORDER",
                "DETERMINISTIC_PRE_POSITION_ZERO_LOADER_CONTRACT",
                "FAIL_CLOSED_PERSISTENCE_ERROR",
                "NON_MODEL_REPAIRED_SUCCESS_PATH",
                "SOURCE_CONSTRAINT_HASH_CLOSURE",
                "ZERO_AUTHORITY_ZERO_EXECUTION",
                "UNCHANGED_RTL_AND_STAGE_GATES",
                "OPTIMIZATION_PLAN_PRESERVES_TIMEOUT_DURABILITY_AND_NO_FALLBACK",
            ],
            "targets": {
                name: sha256_file(target / name) for name in review_targets
            },
            "activation_before_acceptance": "FORBIDDEN",
        },
    )


def main() -> int:
    if PACKAGE.exists():
        raise RuntimeError(f"fresh CF13 namespace already exists: {PACKAGE}")
    seal, review, source_bindings = validate_cf12()
    before: dict[str, dict[str, object]] = {}
    for namespace in CF12_NAMESPACES:
        before.update(namespace_files(namespace))
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cf13-construction-", dir=BUILD_ROOT) as tmp:
        temporary = Path(tmp)
        first = temporary / "build-a" / PACKAGE.name
        second = temporary / "build-b" / PACKAGE.name
        first.parent.mkdir()
        second.parent.mkdir()
        build_package(
            first,
            seal=seal,
            review=review,
            source_bindings=source_bindings,
            cf12_inventory=before,
        )
        build_package(
            second,
            seal=seal,
            review=review,
            source_bindings=source_bindings,
            cf12_inventory=before,
        )
        first_material = package_manifest(first)
        second_material = package_manifest(second)
        if first_material != second_material:
            raise RuntimeError("CF13 two-build byte/mode reproducibility differs")
        reproducibility = {
            "schema": "ace2-r6-cf13-two-build-reproducibility-v1",
            "status": "PASS_TWO_BUILD_BYTE_MODE_EQUAL",
            "build_count": 2,
            "material_entry_count": len(first_material),
            "material_manifest_sha256": sha256_bytes(
                canonical_bytes(first_material)
            ),
        }
        write_json(first / "reproducibility-report.json", reproducibility)
        write_json(second / "reproducibility-report.json", reproducibility)
        if package_manifest(first) != package_manifest(second):
            raise RuntimeError("CF13 complete namespace reproducibility differs")
        after: dict[str, dict[str, object]] = {}
        for namespace in CF12_NAMESPACES:
            after.update(namespace_files(namespace))
        if after != before:
            raise RuntimeError("CF12 byte or mode changed during CF13 construction")
        os.replace(first, PACKAGE)
        PACKAGE.chmod(0o555)
        directory = os.open(BUILD_ROOT, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    print(PACKAGE.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"CF13 construction refused: {error}", file=sys.stderr)
        raise SystemExit(2)
