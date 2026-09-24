#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import stat
import struct
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/ace2_chat_demo"
PACKAGE_NAME = "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
DEFAULT_PACKAGE = BUILD_ROOT / PACKAGE_NAME
CF12_PACKAGE = BUILD_ROOT / "stage1-w4a8-r6-generation-probe-cf12-attempt-0001"
CF12_STATE = BUILD_ROOT / f"{CF12_PACKAGE.name}-authority-state"
CF12_SEAL = BUILD_ROOT / "stage1-w4a8-r6-generation-probe-cf12-terminal-seal-0001"
CF13_PACKAGE = (
    BUILD_ROOT / "stage1-w4a8-r6-generation-probe-product-host-repair-cf13-0001"
)
CF14_PACKAGE = (
    BUILD_ROOT / "stage1-w4a8-r6-generation-probe-product-host-repair-cf14-0001"
)
CF14_STATE = BUILD_ROOT / f"{CF14_PACKAGE.name}-authority-state"
CF14_TERMINAL_SEAL = CF14_STATE / "attempt-0001/terminal-seal.json"
CF14_REVIEW_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobeexecute14"
)
CONSTRAINT = ROOT / "constraints/ace2_rmsnorm_core.sdc"
PREDECESSOR_ROOTS = (
    CF12_PACKAGE,
    CF12_STATE,
    CF12_SEAL,
    CF13_PACKAGE,
    CF14_PACKAGE,
    CF14_STATE,
)


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


def file_record(path: Path) -> dict[str, object]:
    return {
        "mode": stat.S_IMODE(path.stat().st_mode),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def write_file(path: Path, value: bytes | str, mode: int = 0o444) -> None:
    data = value.encode("utf-8") if isinstance(value, str) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def authenticate_cf14_terminal_review() -> dict[str, object]:
    seal = load_json(CF14_TERMINAL_SEAL)
    if (
        seal.get("classification") != "TIMEOUT_PROCESS_TREE_TERMINATED"
        or seal.get("status") != "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        or seal.get("retry_replay_resume_relaunch") != "PERMANENTLY_FORBIDDEN"
    ):
        raise RuntimeError("CF14 terminal seal classification or permanence differs")
    for artifact in seal.get("artifacts", []):
        path = Path(artifact["path"])
        actual = file_record(path)
        expected = {
            "mode": artifact["mode"],
            "sha256": artifact["sha256"],
            "size": artifact["size"],
        }
        if actual != expected:
            raise RuntimeError(f"CF14 terminal artifact changed: {path.name}")

    latest_path = CF14_REVIEW_ROOT / "latest.json"
    latest = load_json(latest_path)
    if latest.get("kind") != "handoff_ref":
        raise RuntimeError("CF14 terminal review index is not a handoff reference")
    handoff_path = Path(latest["handoff"]["path"])
    handoff = load_json(handoff_path)
    if (
        handoff.get("kind") != "round_reviewed_handoff"
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("CF14 terminal evidence lacks canonical Reviewer completion")
    checkpoint_path = Path(handoff["checkpoint"]["path"])
    checkpoint = checkpoint_path.read_text(encoding="utf-8")
    required = (
        "reached its first natural terminal",
        "permanently forbid retry, replay, resume, or relaunch",
        str(CF14_TERMINAL_SEAL),
        "Stage 1 remains OPEN",
        "Stage 2 remains FORBIDDEN",
    )
    if any(item not in checkpoint for item in required):
        raise RuntimeError("CF14 reviewed checkpoint omits a terminal invariant")
    return {
        "latest": {"path": str(latest_path), **file_record(latest_path)},
        "handoff": {"path": str(handoff_path), **file_record(handoff_path)},
        "checkpoint": {"path": str(checkpoint_path), **file_record(checkpoint_path)},
        "terminal_seal": {
            "path": str(CF14_TERMINAL_SEAL),
            **file_record(CF14_TERMINAL_SEAL),
        },
    }


def diagnose_qwen2_contract() -> dict[str, object]:
    transformers = importlib.import_module("transformers")
    qwen2 = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    model_forward = inspect.getsource(qwen2.Qwen2Model.forward)
    causal_forward = inspect.getsource(qwen2.Qwen2ForCausalLM.forward)
    model_signature = str(inspect.signature(qwen2.Qwen2Model.forward))
    if "output_hidden_states" in model_signature:
        raise RuntimeError("Qwen2Model now exposes an explicit hidden-state contract")
    if "all_hidden_states" in model_forward or "output_hidden_states" in model_forward:
        raise RuntimeError("Qwen2Model now captures hidden states; diagnosis is stale")
    if "hidden_states=outputs.hidden_states" not in causal_forward:
        raise RuntimeError("Qwen2ForCausalLM output propagation contract changed")
    return {
        "classification": "RUNTIME_OMITTED_LAYER_STATE_CAPTURE",
        "consumer_api": (
            "free.hidden_states is the standard causal-LM output field, but this "
            "producer leaves it None"
        ),
        "model_forward_signature": model_signature,
        "qwen2_causal_forward_sha256": sha256_bytes(causal_forward.encode("utf-8")),
        "qwen2_model_forward_sha256": sha256_bytes(model_forward.encode("utf-8")),
        "returned_different_field_or_shape": False,
        "transformers_version": transformers.__version__,
        "wrong_consumer_api": False,
    }


def predecessor_inventory() -> dict[str, object]:
    entries: dict[str, object] = {}
    for base in PREDECESSOR_ROOTS:
        if not base.is_dir():
            raise RuntimeError(f"required predecessor root is absent: {base.name}")
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            relative = path.relative_to(ROOT).as_posix()
            entries[relative] = file_record(path)
    return {
        "schema": "ace2-r6-cf15-cf12-through-cf14-read-only-inventory-v1",
        "entries": entries,
        "mutation": "FORBIDDEN",
    }


MODEL24_CONTRACT_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn


LAYER_COUNT = 24
EXECUTED_SOURCE = "FORWARD_HOOK_EXECUTED_LAYER_OUTPUT"
SUPPORTED_DTYPES = frozenset({"torch.bfloat16", "torch.float16", "torch.float32"})


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_bytes({"dtype": str(value.dtype), "shape": list(value.shape)}))
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LayerStateRecord:
    layer_index: int
    state: Tensor
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    source_kind: str


@dataclass(frozen=True)
class Model24Position:
    records: tuple[LayerStateRecord, ...]
    sha256: str


def _position_hash(records: Sequence[LayerStateRecord]) -> str:
    descriptor = [
        {
            "dtype": record.dtype,
            "layer_index": record.layer_index,
            "sha256": record.sha256,
            "shape": list(record.shape),
        }
        for record in records
    ]
    return hashlib.sha256(canonical_bytes(descriptor)).hexdigest()


def validate_layer_states(records: Sequence[LayerStateRecord]) -> Model24Position:
    if len(records) != LAYER_COUNT:
        raise ValueError(f"position requires exactly {LAYER_COUNT} layer states")
    if any(not isinstance(record, LayerStateRecord) for record in records):
        raise TypeError("position contains a missing or non-record layer state")
    indices = [record.layer_index for record in records]
    if indices != list(range(LAYER_COUNT)):
        raise ValueError("layer states are duplicated, missing, or reordered")

    expected_shape: tuple[int, ...] | None = None
    expected_dtype: str | None = None
    storage_pointers: set[int] = set()
    for record in records:
        if record.source_kind != EXECUTED_SOURCE:
            raise ValueError("synthetic placeholder layer state is forbidden")
        if not isinstance(record.state, Tensor):
            raise TypeError("layer state must be a tensor")
        if record.state.numel() == 0:
            raise ValueError("layer state must be nonempty")
        actual_shape = tuple(int(size) for size in record.state.shape)
        actual_dtype = str(record.state.dtype)
        if record.shape != actual_shape:
            raise ValueError("layer-state shape metadata differs from tensor")
        if record.dtype != actual_dtype:
            raise ValueError("layer-state dtype metadata differs from tensor")
        if record.dtype not in SUPPORTED_DTYPES:
            raise TypeError("layer state uses an unsupported dtype")
        if len(record.shape) != 2 or record.shape[0] != 1 or record.shape[1] <= 0:
            raise ValueError("layer state must have [1, hidden_size] position geometry")
        if expected_shape is None:
            expected_shape = record.shape
            expected_dtype = record.dtype
        elif record.shape != expected_shape or record.dtype != expected_dtype:
            raise ValueError("layer-state shape or dtype differs across layers")
        if record.sha256 != tensor_digest(record.state):
            raise ValueError("layer-state canonical hash differs")
        pointer = record.state.data_ptr()
        if pointer in storage_pointers:
            raise ValueError("layer states alias the same captured storage")
        storage_pointers.add(pointer)
    return Model24Position(tuple(records), _position_hash(records))


def _layer_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("decoder layer output does not expose a tensor state")


def capture_model24_position(model: nn.Module, **forward_kwargs: Any) -> tuple[Any, Model24Position]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != LAYER_COUNT:
        raise ValueError("model must expose exactly 24 decoder layers")
    if len({id(layer) for layer in layers}) != LAYER_COUNT:
        raise ValueError("model decoder layers must be distinct")

    records: list[LayerStateRecord] = []
    handles = []

    def hook_for(layer_index: int):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if layer_index != len(records):
                raise RuntimeError("decoder layers did not execute exactly once in order")
            tensor = _layer_tensor(output)
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] == 0:
                raise ValueError("decoder layer output lacks [1, sequence, hidden] geometry")
            state = tensor[:, -1].detach().cpu().contiguous().clone()
            records.append(
                LayerStateRecord(
                    layer_index=layer_index,
                    state=state,
                    shape=tuple(int(size) for size in state.shape),
                    dtype=str(state.dtype),
                    sha256=tensor_digest(state),
                    source_kind=EXECUTED_SOURCE,
                )
            )
        return capture

    try:
        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(hook_for(layer_index)))
        output = model(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return output, validate_layer_states(records)
'''


TEST_SOURCE = r'''
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE))
from model24_position_contract import (
    EXECUTED_SOURCE,
    LayerStateRecord,
    capture_model24_position,
    tensor_digest,
    validate_layer_states,
)

EXPECTED_FIXTURE_POSITION_SHA256 = "{expected_hash}"


def load_runtime():
    name = "_cf15_product_host_runtime"
    specification = importlib.util.spec_from_file_location(
        name, PACKAGE / "product_host_runtime.py"
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class FixtureLayer(nn.Module):
    def __init__(self, increment: int) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden_states):
        return hidden_states + self.increment


class FixtureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FixtureLayer(index + 1) for index in range(24))


class FixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FixtureBackbone()

    def forward(self, inputs):
        hidden_states = inputs
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(logits=hidden_states)


def fixture_position():
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    output, position = capture_model24_position(FixtureModel(), inputs=inputs)
    return output, position


class Model24ContractTests(unittest.TestCase):
    def test_non_model_fixture_uses_same_24_layer_path_and_stable_hash(self) -> None:
        output, position = fixture_position()
        self.assertEqual(len(position.records), 24)
        self.assertEqual([record.layer_index for record in position.records], list(range(24)))
        self.assertTrue(all(record.shape == (1, 8) for record in position.records))
        self.assertTrue(all(record.dtype == "torch.float32" for record in position.records))
        self.assertTrue(all(record.source_kind == EXECUTED_SOURCE for record in position.records))
        self.assertEqual(position.sha256, EXPECTED_FIXTURE_POSITION_SHA256)
        self.assertEqual(float(output.logits[0, 0, 0]), 300.0)

    def test_missing_duplicate_reordered_and_truncated_states_fail_closed(self) -> None:
        _, position = fixture_position()
        records = list(position.records)
        cases = {
            "missing": records[:5] + [None] + records[6:],
            "duplicate": records[:5] + [records[4]] + records[6:],
            "reordered": [records[1], records[0], *records[2:]],
            "truncated": records[:-1],
        }
        for label, malformed in cases.items():
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                validate_layer_states(malformed)

    def test_placeholder_shape_dtype_hash_and_alias_fail_closed(self) -> None:
        _, position = fixture_position()
        records = list(position.records)
        malformed = {
            "placeholder": dataclasses.replace(
                records[7], source_kind="SYNTHETIC_PLACEHOLDER"
            ),
            "shape": dataclasses.replace(records[7], shape=(1, 9)),
            "dtype": dataclasses.replace(records[7], dtype="torch.bfloat16"),
            "hash": dataclasses.replace(records[7], sha256="0" * 64),
            "alias": dataclasses.replace(
                records[7],
                state=records[6].state,
                shape=records[6].shape,
                dtype=records[6].dtype,
                sha256=records[6].sha256,
            ),
        }
        for label, replacement in malformed.items():
            candidate = list(records)
            candidate[7] = replacement
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                validate_layer_states(candidate)

    def test_failed_forward_removes_all_capture_hooks(self) -> None:
        model = FixtureModel()
        with mock.patch.object(model.model.layers[12], "forward", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                capture_model24_position(
                    model, inputs=torch.zeros((1, 1, 8), dtype=torch.float32)
                )
        self.assertTrue(all(not layer._forward_hooks for layer in model.model.layers))

    def test_product_probe_consumes_explicit_contract_not_hidden_states(self) -> None:
        source = (PACKAGE / "product_probe.py").read_text(encoding="utf-8")
        self.assertIn("capture_model24_position(", source)
        self.assertIn("position_states.records", source)
        self.assertNotIn("free.hidden_states", source)
        self.assertNotIn("output_hidden_states=True", source)


class DurableBoundaryTests(unittest.TestCase):
    def test_private_safe_capture_precedes_endpoint(self) -> None:
        runtime = load_runtime()
        sentinel = b"PRIVATE-PROMPT-SENTINEL-CF15"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            capture = root / "capture.json"
            endpoint = mock.Mock(return_value={"exit_code": 0, "fixture": True})

            def failing_host(tracker):
                private_prompt = sentinel.decode("ascii")
                self.assertTrue(private_prompt)
                tracker.set(
                    "POSITION_ZERO_PREPARATION",
                    input_token_count=37,
                    runtime_module_loaded=True,
                    runtime_constructed=True,
                    prefill_completed=True,
                    positions_persisted=0,
                )
                raise RuntimeError("fixture layer-state contract failure")

            self.assertEqual(
                runtime.run_product_host_boundary(
                    evidence_path=evidence,
                    capture_path=capture,
                    host_runner=failing_host,
                    endpoint_runner=endpoint,
                ),
                2,
            )
            endpoint.assert_called_once_with()
            raw = capture.read_bytes()
            self.assertNotIn(sentinel, raw)
            payload = json.loads(raw)
            self.assertEqual(payload["exception"]["type"], "RuntimeError")
            self.assertEqual(
                payload["exception"]["message"], "fixture layer-state contract failure"
            )
            self.assertEqual(payload["local_context"]["input_token_count"], 37)
            self.assertEqual(payload["local_context"]["positions_persisted"], 0)
            self.assertEqual(payload["privacy"]["traceback_locals"], "NOT_CAPTURED")
            self.assertLessEqual(
                len(runtime.canonical_bytes(payload["local_context"])),
                runtime.CAPTURE_CONTEXT_MAX_BYTES,
            )

    def test_capture_fsyncs_file_then_parent_directory(self) -> None:
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

    def test_capture_persistence_failure_prevents_endpoint(self) -> None:
        runtime = load_runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture.json"
            endpoint = mock.Mock(return_value={"exit_code": 0})
            original = runtime.atomic_json

            def fail_capture(path, value, *, exclusive=False):
                if path == capture:
                    raise OSError("fixture fsync failure")
                original(path, value, exclusive=exclusive)

            with mock.patch.object(runtime, "atomic_json", fail_capture):
                with self.assertRaisesRegex(OSError, "fixture fsync failure"):
                    runtime.run_product_host_boundary(
                        evidence_path=root / "evidence.json",
                        capture_path=capture,
                        host_runner=lambda _tracker: (_ for _ in ()).throw(
                            RuntimeError("fixture host failure")
                        ),
                        endpoint_runner=endpoint,
                    )
            endpoint.assert_not_called()


class PackageClosureTests(unittest.TestCase):
    def test_predecessor_inventory_is_read_only_and_exact(self) -> None:
        inventory = json.loads((PACKAGE / "predecessor-read-only-inventory.json").read_text())
        for relative, expected in inventory["entries"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(path.stat().st_size, expected["size"], relative)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected["mode"], relative)
            self.assertEqual(
                __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                expected["sha256"],
                relative,
            )

    def test_contract_has_no_execution_authority(self) -> None:
        contract = json.loads((PACKAGE / "contract.json").read_text())
        self.assertEqual(contract["execution_authority"], "NONE")
        self.assertEqual(contract["execution_limit"], 0)
        self.assertEqual(
            contract["diagnosis"]["classification"],
            "RUNTIME_OMITTED_LAYER_STATE_CAPTURE",
        )
        self.assertEqual(contract["prompt_change"], "NONE")
        self.assertEqual(contract["rtl_change"], "NONE")
        self.assertEqual(contract["software_fallback"], "FORBIDDEN")
        self.assertEqual(contract["stage1"], "OPEN")
        self.assertEqual(contract["stage2"], "FORBIDDEN")
        for name in (
            "authority.json",
            "authorization-consumed.json",
            "attempt-0001",
            "launch.sh",
            "product-result.json",
        ):
            self.assertFalse((PACKAGE / name).exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


VALIDATOR_SOURCE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: dict[str, object]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing bound file: {path}")
    if path.stat().st_size != expected["size"]:
        raise RuntimeError(f"bound size differs: {path}")
    if stat.S_IMODE(path.stat().st_mode) != expected["mode"]:
        raise RuntimeError(f"bound mode differs: {path}")
    if sha256_file(path) != expected["sha256"]:
        raise RuntimeError(f"bound hash differs: {path}")


def main() -> None:
    manifest = json.loads((PACKAGE / "package-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        verify(PACKAGE / relative, expected)
    inventory = json.loads((PACKAGE / "predecessor-read-only-inventory.json").read_text())
    for relative, expected in inventory["entries"].items():
        verify(ROOT / relative, expected)
    contract = json.loads((PACKAGE / "contract.json").read_text())
    if (
        contract["execution_authority"] != "NONE"
        or contract["execution_limit"] != 0
        or contract["stage1"] != "OPEN"
        or contract["stage2"] != "FORBIDDEN"
    ):
        raise RuntimeError("CF15 zero-authority stage gate differs")
    print("PASS_CF15_PRODUCT_HOST_CONTRACT_REPAIR_ZERO_AUTHORITY")


if __name__ == "__main__":
    main()
'''


def fixture_expected_hash() -> str:
    descriptors = []
    cumulative = 0
    for layer_index in range(24):
        cumulative += layer_index + 1
        values = [float(cumulative + value) for value in range(8)]
        state_bytes = struct.pack("<8f", *values)
        digest = hashlib.sha256()
        digest.update(
            canonical_bytes({"dtype": "torch.float32", "shape": [1, 8]})
        )
        digest.update(state_bytes)
        descriptors.append(
            {
                "dtype": "torch.float32",
                "layer_index": layer_index,
                "sha256": digest.hexdigest(),
                "shape": [1, 8],
            }
        )
    return sha256_bytes(canonical_bytes(descriptors))


def repaired_probe_source() -> str:
    source = (CF13_PACKAGE / "product_probe.py").read_text(encoding="utf-8")
    import_anchor = "from typing import Any\n\nfrom product_host_runtime import ("
    replacement_import = (
        "from typing import Any\n\n"
        "from model24_position_contract import capture_model24_position\n"
        "from product_host_runtime import ("
    )
    if source.count(import_anchor) != 1:
        raise RuntimeError("CF13 product probe import anchor changed")
    source = source.replace(import_anchor, replacement_import)
    old_call = '''            with torch.inference_mode():
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
'''
    new_call = '''            with torch.inference_mode():
                free, position_states = capture_model24_position(
                    runtime.model,
                    input_ids=full_ids,
                    use_cache=False,
                    return_dict=True,
                )
            cache_free_logits = free.logits[0, -1].detach().cpu().to(torch.float32)
            if cache_free_logits.numel() != 151936 or cacheful_logits.numel() != 151936:
                raise RuntimeError("product logits do not cover 151936 rows")
'''
    if source.count(old_call) != 1:
        raise RuntimeError("CF13 product probe model-call anchor changed")
    source = source.replace(old_call, new_call)
    old_digests = '''                "per_layer_state_digests": [
                    tensor_digest(state[:, -1]) for state in states[1:]
                ],
'''
    new_digests = '''                "per_layer_state_digests": [
                    record.sha256 for record in position_states.records
                ],
                "layer_state_contract_sha256": position_states.sha256,
'''
    if source.count(old_digests) != 1:
        raise RuntimeError("CF13 product probe digest anchor changed")
    return source.replace(old_digests, new_digests).replace(
        '"_ace2_cf13_runtime"', '"_ace2_cf15_runtime"'
    )


def materialize(package: Path) -> None:
    review = authenticate_cf14_terminal_review()
    diagnosis = diagnose_qwen2_contract()
    inventory = predecessor_inventory()
    package.mkdir(parents=True)

    write_file(package / "model24_position_contract.py", textwrap.dedent(MODEL24_CONTRACT_SOURCE).lstrip())
    write_file(package / "product_probe.py", repaired_probe_source(), 0o555)
    write_file(
        package / "product_host_runtime.py",
        (CF13_PACKAGE / "product_host_runtime.py").read_bytes(),
    )
    write_file(
        package / "tests/test_cf15_product_host_contract.py",
        textwrap.dedent(TEST_SOURCE)
        .lstrip()
        .replace("{expected_hash}", fixture_expected_hash()),
    )

    bindings = load_json(CF13_PACKAGE / "bindings.json")
    bindings["schema"] = "ace2-r6-cf15-product-host-contract-bindings-v1"
    bindings["source_cf13_product_probe"] = {
        "path": str(CF13_PACKAGE / "product_probe.py"),
        **file_record(CF13_PACKAGE / "product_probe.py"),
    }
    write_file(package / "bindings.json", canonical_bytes(bindings))
    write_file(
        package / "predecessor-read-only-inventory.json",
        canonical_bytes(inventory),
    )

    contract = {
        "schema": "ace2-r6-cf15-product-host-contract-repair-v1",
        "identity": PACKAGE_NAME,
        "diagnosis": diagnosis,
        "repair_surface": "ADDITIVE_EXPLICIT_MODEL24_DECODER_LAYER_OUTPUT_CAPTURE",
        "layer_count": 24,
        "layer_states": (
            "GENUINELY_EXECUTED_ORDERED_FORWARD_OUTPUTS_WITH_SHAPE_DTYPE_AND_HASH"
        ),
        "fabricated_states": "FORBIDDEN",
        "layer_bypass": "FORBIDDEN",
        "software_fallback": "FORBIDDEN",
        "prompt_change": "NONE",
        "model_tokenizer_w4a8_change": "NONE",
        "rtl_change": "NONE",
        "accelerator_evidence_change": "NONE",
        "execution_authority": "NONE",
        "execution_limit": 0,
        "official_model_trajectory": "NOT_EXECUTED",
        "rtl_endpoint": "NOT_EXECUTED",
        "simulator_synthesis_ppa_fpga": "NOT_EXECUTED",
        "cf14_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "cf14_terminal_review": review,
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    write_file(package / "contract.json", canonical_bytes(contract))

    source_files = {
        "model24_position_contract.py": {
            "scope": "package",
            **file_record(package / "model24_position_contract.py"),
        },
        "product_host_runtime.py": {
            "scope": "package",
            **file_record(package / "product_host_runtime.py"),
        },
        "product_probe.py": {
            "scope": "package",
            **file_record(package / "product_probe.py"),
        },
        "tests/test_cf15_product_host_contract.py": {
            "scope": "package",
            **file_record(package / "tests/test_cf15_product_host_contract.py"),
        },
        CONSTRAINT.relative_to(ROOT).as_posix(): {
            "scope": "repository",
            **file_record(CONSTRAINT),
        },
        (CF13_PACKAGE / "product_probe.py").relative_to(ROOT).as_posix(): {
            "scope": "repository",
            **file_record(CF13_PACKAGE / "product_probe.py"),
        },
    }
    source_manifest = {
        "schema": "ace2-r6-cf15-source-constraint-manifest-v1",
        "files": source_files,
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "non_sram_area_cap_mm2": 2.0,
        "frequency_floor_mhz": 100,
    }
    write_file(
        package / "source-constraint-manifest.json",
        canonical_bytes(source_manifest),
    )
    write_file(
        package / "source-constraint-manifest.sha256",
        f"{sha256_file(package / 'source-constraint-manifest.json')}  source-constraint-manifest.json\n",
    )
    write_file(
        package / "reproducibility-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf15-reproducibility-v1",
                "status": "PASS_TWO_INDEPENDENT_MATERIALIZATIONS_BYTE_MODE_SIZE_IDENTICAL",
                "official_execution": False,
            }
        ),
    )
    write_file(package / "validate_package.py", textwrap.dedent(VALIDATOR_SOURCE).lstrip(), 0o555)
    write_file(
        package / "Makefile",
        (
            "PYTHON ?= python3\n\n"
            ".PHONY: check\n"
            "check:\n"
            "\t$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v\n"
            "\t$(PYTHON) -B validate_package.py\n"
        ),
    )

    manifest_files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file():
            manifest_files[path.relative_to(package).as_posix()] = file_record(path)
    package_manifest = {
        "schema": "ace2-r6-cf15-package-manifest-v1",
        "files": manifest_files,
    }
    write_file(package / "package-manifest.json", canonical_bytes(package_manifest))
    write_file(
        package / "package-manifest.sha256",
        f"{sha256_file(package / 'package-manifest.json')}  package-manifest.json\n",
    )

    review_request = {
        "schema": "ace2-r6-cf15-canonical-independent-l2-review-request-v1",
        "status": "PENDING_INDEPENDENT_L2",
        "activation_before_acceptance": "FORBIDDEN",
        "execution_authority": "NONE",
        "required_checks": [
            "CF14_TERMINAL_SEAL_AND_CANONICAL_REVIEW_READ_ONLY_AUTHENTICATED",
            "CF12_THROUGH_CF14_PREDECESSOR_INVENTORY_UNCHANGED",
            "RUNTIME_OMITTED_LAYER_STATE_CAPTURE_DIAGNOSIS",
            "EXPLICIT_24_EXECUTED_LAYER_STATE_ORDER_SHAPE_DTYPE_HASH_CONTRACT",
            "NEGATIVE_MISSING_DUPLICATE_REORDERED_TRUNCATED_PLACEHOLDER_REJECTION",
            "PRIVATE_SAFE_ATOMIC_FAIL_CLOSED_EVIDENCE_PERSISTENCE",
            "TWO_BUILD_REPRODUCIBILITY",
            "ZERO_EXECUTION_AUTHORITY_STAGE1_OPEN_STAGE2_FORBIDDEN",
        ],
        "targets": {
            "package_manifest": file_record(package / "package-manifest.json"),
            "source_constraint_manifest": file_record(
                package / "source-constraint-manifest.json"
            ),
        },
    }
    write_file(package / "review-request.json", canonical_bytes(review_request))
    write_file(
        package / "review-request.sha256",
        f"{sha256_file(package / 'review-request.json')}  review-request.json\n",
    )


def snapshot(package: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(package).as_posix(): (
            stat.S_IMODE(path.stat().st_mode),
            path.stat().st_size,
            sha256_file(path),
        )
        for path in sorted(package.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to replace existing package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cf15-build-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        materialize(first)
        materialize(second)
        if snapshot(first) != snapshot(second):
            raise RuntimeError("CF15 independent materializations differ")
        first.rename(output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    print(f"BUILT_CF15_REPRODUCIBLY {output}")


if __name__ == "__main__":
    main()
