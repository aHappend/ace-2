#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import stat
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CF10 = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf10"
CF11 = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf11"
PREDECESSOR = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-reference-process-predecessor-cf12-0001"
)
PACKAGE = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-generation-probe-cf12-attempt-0001"
)
CF11_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationqualitydiagnosis11"
)
CF12_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/bfd265994985"
)
IDENTITY = "stage1-w4a8-r6-generation-probe-cf12-attempt-0001"
STATE = PACKAGE.parent / f"{IDENTITY}-authority-state"
TRANSACTION_NONCE = (
    "43b14235201a90ee8d520980e7cf6a5bbe7b57ecf4184977cfdc2bd0a0e38811"
)
DURABLE_TASK_ID = "ace2-r6-cf12-generation-probe-once-20260825-0001"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / MODEL_REVISION
)
MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
DEPENDENCIES = ("torch", "transformers", "tokenizers", "safetensors", "numpy")
PREDECESSORS = {
    f"cf{number:02d}": (
        ROOT
        / f"build/ace2_chat_demo/stage1-w4a8-r6-product-prep-cf{number:02d}"
    )
    for number in range(7, 12)
}


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
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: object, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    path.chmod(mode)


def write_text(path: Path, value: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(value).lstrip(), encoding="utf-8")
    path.chmod(mode)


def namespace_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"namespace is absent or linked: {path}")
    entries: dict[str, dict[str, object]] = {}
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"namespace contains a link: {entry}")
        mode = stat.S_IMODE(entry.stat().st_mode)
        if entry.is_dir():
            entries[relative] = {"kind": "directory", "mode": mode}
        elif entry.is_file():
            entries[relative] = {
                "kind": "file",
                "mode": mode,
                "size": entry.stat().st_size,
                "sha256": sha256_file(entry),
            }
        else:
            raise RuntimeError(f"unsupported namespace entry: {entry}")
    return entries


def material_manifest(path: Path) -> dict[str, dict[str, object]]:
    return {
        relative: record
        for relative, record in namespace_manifest(path).items()
        if relative != "reproducibility-report.json"
    }


def dependency_identity(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    record_paths = [
        distribution.locate_file(item)
        for item in distribution.files or ()
        if item.name == "RECORD" and ".dist-info" in item.as_posix()
    ]
    if len(record_paths) != 1 or not record_paths[0].is_file():
        raise RuntimeError(f"cannot pin installed dependency RECORD: {name}")
    return {
        "name": name,
        "version": distribution.version,
        "record_sha256": sha256_file(record_paths[0]),
    }


def runtime_identity() -> dict[str, object]:
    executable = Path(sys.executable).resolve()
    return {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "dependencies": [dependency_identity(name) for name in DEPENDENCIES],
    }


def validate_cf11_review() -> dict[str, Any]:
    latest = load_object(CF11_REVIEW / "latest.json")
    if latest.get("kind") != "handoff_ref":
        raise RuntimeError("CF11 latest index is not a reviewed handoff")
    handoff_path = Path(latest.get("handoff", {}).get("path", ""))
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != "r6generationqualitydiagnosis11"
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("CF11 lacks canonical independent acceptance")
    return {
        "mission_id": handoff["mission_id"],
        "latest_sha256": sha256_file(CF11_REVIEW / "latest.json"),
        "handoff_path": str(handoff_path),
        "handoff_sha256": sha256_file(handoff_path),
        "producer_role": "reviewer",
        "review_status": "done",
    }


def validate_frozen_inputs() -> tuple[bytes, list[int], dict[str, Any]]:
    prompt = (CF10 / "input.txt").read_bytes()
    descriptor = load_object(CF10 / "inputs/descriptor.json")
    token_ids = descriptor.get("request", {}).get("prompt_token_ids")
    if (
        sha256_bytes(prompt)
        != descriptor.get("request", {}).get("input_utf8_sha256")
        or not isinstance(token_ids, list)
        or not token_ids
        or not all(type(item) is int for item in token_ids)
    ):
        raise RuntimeError("CF10 exact prompt or template IDs differ")
    expected = {
        MODEL_ROOT / "model.safetensors": MODEL_SHA256,
        MODEL_ROOT / "tokenizer.json": TOKENIZER_JSON_SHA256,
        MODEL_ROOT / "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"official model/tokenizer binding differs: {path.name}")
    return prompt, token_ids, descriptor


RECEIPT_VALIDATOR_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class ReceiptError(RuntimeError):
    pass


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


def valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def receipt_binding(receipt: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in receipt.items() if key != "binding_sha256"}
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def validate_reference_artifact(
    artifact: dict[str, Any], contract: dict[str, Any]
) -> None:
    producer = contract["reference_producer"]
    positions = artifact.get("positions")
    if (
        artifact.get("schema") != contract["schemas"]["reference_artifact"]
        or artifact.get("status") != "SEALED_BEFORE_PRODUCT_EXECUTION"
        or artifact.get("producer") != producer["identity"]
        or artifact.get("model_sha256") != contract["model"]["model_sha256"]
        or artifact.get("tokenizer_json_sha256")
        != contract["model"]["tokenizer_json_sha256"]
        or artifact.get("full_logit_count") != 151936
        or not isinstance(positions, list)
        or len(positions) != 4
        or any(
            not isinstance(position, dict)
            or not valid_digest(position.get("logits_sha256"))
            or not isinstance(position.get("top_k"), list)
            or len(position["top_k"]) < 2
            or len(position.get("per_layer_state_digests", [])) != 24
            or len(position.get("per_layer_kv_digests", [])) != 24
            for position in positions
        )
    ):
        raise ReceiptError("independent reference artifact differs from the contract")


def validate_reference_receipt(
    receipt: dict[str, Any],
    artifact: dict[str, Any],
    artifact_path: Path,
    contract: dict[str, Any],
    product_evidence: dict[str, Any] | None = None,
) -> None:
    process = receipt.get("process")
    capture = receipt.get("capture")
    if (
        receipt.get("schema") != contract["schemas"]["reference_process_receipt"]
        or receipt.get("status") != "PASS_AUTHORITY_LAUNCHED_REFERENCE"
        or receipt.get("challenge_nonce") != contract["challenge_nonce"]
        or receipt.get("origin") != "AUTHORITY_RUNNER_DIRECT_SUBPROCESS"
        or receipt.get("authority_runner_sha256")
        != contract["authority_runner_sha256"]
        or receipt.get("producer_implementation_sha256")
        != contract["reference_producer"]["implementation_sha256"]
        or receipt.get("producer_manifest_sha256")
        != contract["reference_producer"]["manifest_sha256"]
        or receipt.get("argv_sha256")
        != contract["reference_invocation"]["argv_sha256"]
        or receipt.get("environment_sha256")
        != contract["reference_invocation"]["environment_sha256"]
        or receipt.get("isolated_output_namespace")
        != contract["isolated_output_namespace"]
        or not isinstance(process, dict)
        or type(process.get("pid")) is not int
        or process["pid"] <= 0
        or process.get("fresh_process") is not True
        or process.get("exit_code") != 0
        or process.get("started_before_product") is not True
        or not isinstance(capture, dict)
        or capture.get("reference_artifact_sha256") != sha256_file(artifact_path)
        or capture.get("full_logit_count") != 151936
        or capture.get("position_logits_sha256")
        != [position["logits_sha256"] for position in artifact.get("positions", [])]
        or not valid_digest(capture.get("stdout_sha256"))
        or not valid_digest(capture.get("stderr_sha256"))
        or receipt.get("binding_sha256") != receipt_binding(receipt)
    ):
        raise ReceiptError("authority-launched reference process receipt differs")
    validate_reference_artifact(artifact, contract)
    if product_evidence is not None:
        provenance = product_evidence.get("producer", {})
        if (
            product_evidence.get("source_artifact_sha256")
            == capture["reference_artifact_sha256"]
            or provenance.get("implementation_sha256")
            == contract["reference_producer"]["implementation_sha256"]
            or provenance.get("manifest_sha256")
            == contract["reference_producer"]["manifest_sha256"]
        ):
            raise ReceiptError("reference and product provenance are not independent")
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
import tempfile
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


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


def tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_bytes({"dtype": str(value.dtype), "shape": list(value.shape)}))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import pinned runtime: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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
        digest = hashlib.sha256()
        for item in logits:
            digest.update(
                f"{item['ordinal']}:{item['destination_sha256']}\n".encode("ascii")
            )
        positions.append(
            {
                "ordinal": ordinal,
                "absolute_position": token_step,
                "selected_token_id": logits[-1]["generated_token_after"],
                "per_layer_kv_digests": [item["destination_sha256"] for item in kv],
                "authenticated_logit_tile_count": 4748,
                "authenticated_logit_tile_aggregate_sha256": digest.hexdigest(),
            }
        )
    return positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    args = parser.parse_args()
    bindings = json.loads((PACKAGE / "bindings.json").read_text(encoding="utf-8"))
    evidence: dict[str, Any] = {
        "schema": "ace2-r6-cf12-product-rtl-evidence-v1",
        "status": "STARTED_NO_DECODE",
        "producer": bindings["product_producer"],
        "positions": [],
        "rtl_positions": [],
        "failures": [],
        "decode": "FORBIDDEN_UNTIL_AUTHORITY_COMBINES_FSYNCED_EVIDENCE",
    }
    atomic_json(args.output, evidence, exclusive=True)
    host_error: Exception | None = None
    try:
        import torch

        runtime_module = load_module(
            Path(bindings["accepted_runtime"]["path"]), "_ace2_cf12_runtime"
        )
        preflight = json.loads(
            Path(bindings["accepted_runtime"]["preflight_hashes_path"]).read_text(
                encoding="utf-8"
            )
        )
        runtime = runtime_module.create_runtime(
            hardware_export_manifest=Path(
                bindings["accepted_runtime"]["hardware_export_manifest"]
            ),
            accepted_interface_hashes=preflight["accepted_interfaces"],
            accepted_tokenizer=preflight["accepted_tokenizer"],
        )
        prompt = bytes.fromhex(bindings["prompt"]["bytes_hex"]).decode(
            "utf-8", errors="strict"
        )
        input_ids = list(bindings["prompt"]["chat_template_token_ids"])
        boundary = runtime.prefill(
            [{"role": "user", "content": prompt}], expected_input_ids=input_ids
        )
        sequence = list(input_ids)
        for ordinal in range(4):
            cacheful_logits = torch.tensor(boundary["logits"], dtype=torch.float32)
            full_ids = torch.tensor([sequence], dtype=torch.long)
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
            ranked = top_k(cacheful_logits)
            selected = ranked[0]["token_id"]
            free_ranked = top_k(cache_free_logits)
            states = free.hidden_states[1:]
            if len(states) != 24 or len(boundary["layer_state_digests"]) != 24:
                raise RuntimeError("product position lacks 24 layer/KV digests")
            position = {
                "ordinal": ordinal,
                "absolute_position": len(input_ids) - 1 + ordinal,
                "selected_token_id": selected,
                "full_logit_count": 151936,
                "top_k": ranked,
                "selected_rank": 1,
                "selected_margin": ranked[0]["logit"] - ranked[1]["logit"],
                "prompt_bytes_sha256": bindings["prompt"]["sha256"],
                "chat_template_token_ids_sha256": bindings["prompt"][
                    "chat_template_token_ids_sha256"
                ],
                "embedding_input_ids_sha256": hashlib.sha256(
                    canonical_bytes(sequence)
                ).hexdigest(),
                "per_layer_state_digests": [
                    tensor_digest(state[:, -1]) for state in states
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
            evidence["positions"].append(position)
            evidence["status"] = "HOST_POSITION_FSYNCED"
            atomic_json(args.output, evidence)
            sequence.append(selected)
            if ordinal < 3:
                boundary = runtime.decode(selected)
    except Exception as error:
        host_error = error
        evidence["failures"].append(
            {
                "boundary": "PRODUCT_HOST_TRAJECTORY",
                "error_type": type(error).__name__,
                "positions_persisted": len(evidence["positions"]),
            }
        )
        evidence["status"] = "HOST_PARTIAL_EVIDENCE_FSYNCED_ENDPOINT_STILL_REQUIRED"
        atomic_json(args.output, evidence)

    endpoint_argv = list(bindings["endpoint"]["argv"])
    completed = run_endpoint(endpoint_argv, bindings["endpoint"]["timeout_seconds"])
    endpoint_receipt = {
        "schema": "ace2-r6-cf12-endpoint-invocation-receipt-v1",
        "argv_sha256": hashlib.sha256(canonical_bytes(endpoint_argv)).hexdigest(),
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "output_namespace": str(args.endpoint_output),
    }
    evidence["endpoint_invocation_receipt"] = endpoint_receipt
    evidence["status"] = "ENDPOINT_RECEIPT_FSYNCED"
    atomic_json(args.output, evidence)
    if completed.returncode == 0:
        records = load_jsonl(args.endpoint_output / "commands.jsonl")
        evidence["rtl_positions"] = rtl_positions(
            records, len(bindings["prompt"]["chat_template_token_ids"])
        )
        evidence["status"] = "PRODUCT_AND_RTL_EVIDENCE_FSYNCED_NO_DECODE"
    else:
        evidence["failures"].append(
            {
                "boundary": "RTL_ENDPOINT",
                "error_type": "NonzeroExit",
                "positions_persisted": len(evidence["positions"]),
            }
        )
        evidence["status"] = "TERMINAL_PARTIAL_EVIDENCE_FSYNCED"
    atomic_json(args.output, evidence)
    return 0 if host_error is None and completed.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


AUTHORITY_RUNNER_SOURCE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
PREDECESSOR = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-reference-process-predecessor-cf12-0001"
)
STATE = PACKAGE.parent / f"{PACKAGE.name}-authority-state"
CONSUMED = STATE / "authorization-consumed.json"
ATTEMPT = STATE / "attempt-0001"
FORBIDDEN = (
    STATE,
    PACKAGE / "authorization-consumed.json",
    PACKAGE / "attempt-0001",
    PACKAGE / "execution-registry.json",
    PACKAGE / "completion-summary.json",
    PACKAGE / "product-result.json",
)


class AuthorityError(RuntimeError):
    pass


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


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuthorityError(f"JSON root is not an object: {path}")
    return value


def atomic_bytes(path: Path, raw: bytes, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
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


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    atomic_bytes(path, canonical_bytes(value), exclusive=exclusive)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_validator() -> Any:
    path = PREDECESSOR / "receipt_validator.py"
    specification = importlib.util.spec_from_file_location("_cf12_receipt_validator", path)
    if specification is None or specification.loader is None:
        raise AuthorityError("cannot import reviewed receipt validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_review() -> dict[str, Any]:
    authority = load_object(PACKAGE / "authority.json")
    review_root = Path(authority["activation_gate"]["review_root"])
    latest = load_object(review_root / "latest.json")
    if latest.get("kind") != "handoff_ref":
        raise AuthorityError("CF12 authority lacks canonical L2 handoff")
    handoff_path = Path(latest.get("handoff", {}).get("path", ""))
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != authority["activation_gate"]["mission_id"]
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise AuthorityError("CF12 authority is pending canonical L2 acceptance")
    request = load_object(PACKAGE / "review-request.json")
    for target in request["targets"]:
        path = ROOT / target["path"]
        if sha256_file(path) != target["sha256"]:
            raise AuthorityError("review target changed after construction")
    return authority


def claim(authority: dict[str, Any]) -> None:
    if (
        authority.get("consumption_evidence_namespace") != str(STATE)
        or authority.get("output_namespace") != str(ATTEMPT)
    ):
        raise AuthorityError("consumption/evidence namespace is not authority-bound")
    if any(os.path.lexists(path) for path in FORBIDDEN):
        raise AuthorityError("CF12 authority is already or ambiguously consumed")
    STATE.mkdir(mode=0o700)
    fsync_directory(STATE.parent)
    record = {
        "schema": "ace2-r6-cf12-authorization-consumption-v1",
        "status": "CONSUMED_BEFORE_REFERENCE_OR_PRODUCT_PROCESS",
        "authority_sha256": sha256_file(PACKAGE / "authority.json"),
        "identity": authority["identity"],
        "transaction_nonce": authority["transaction_nonce"],
        "durable_task_id": authority["durable_task_id"],
        "challenge_nonce": authority["challenge_nonce"],
        "retry": "PERMANENTLY_FORBIDDEN",
        "replay": "PERMANENTLY_FORBIDDEN",
        "resume": "PERMANENTLY_FORBIDDEN",
        "relaunch": "PERMANENTLY_FORBIDDEN",
    }
    atomic_json(CONSUMED, record, exclusive=True)
    CONSUMED.chmod(0o444)
    descriptor = os.open(CONSUMED, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(STATE)
    ATTEMPT.mkdir(mode=0o700)
    fsync_directory(STATE)


def journal(value: dict[str, Any]) -> None:
    atomic_json(ATTEMPT / "evidence-journal.json", value)


def run_process(
    argv: list[str], environment: dict[str, str], timeout: int
) -> tuple[int, int, bytes, bytes]:
    process = subprocess.Popen(
        argv,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return process.pid, 124, stdout, stderr
    return process.pid, process.returncode, stdout, stderr


def first_divergence(
    reference: list[dict[str, Any]],
    product: list[dict[str, Any]],
    rtl: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference) != 4 or len(product) != 4 or len(rtl) != 4:
        raise AuthorityError("four-position reference/product/RTL evidence is incomplete")
    for ordinal, (expected, observed, hardware) in enumerate(
        zip(reference, product, rtl, strict=True)
    ):
        comparisons = (
            (
                "PROMPT_CHAT_TEMPLATE_SERIALIZATION",
                expected.get("prompt_bytes_sha256"),
                observed.get("prompt_bytes_sha256"),
            ),
            (
                "PROMPT_CHAT_TEMPLATE_SERIALIZATION",
                expected.get("chat_template_token_ids_sha256"),
                observed.get("chat_template_token_ids_sha256"),
            ),
            (
                "EMBEDDING_INPUT_IDS",
                expected.get("embedding_input_ids_sha256"),
                observed.get("embedding_input_ids_sha256"),
            ),
        )
        for stage, left, right in comparisons:
            if left != right:
                return {
                    "status": "DIVERGED",
                    "ordinal": ordinal,
                    "stage": stage,
                    "classification": f"{stage}_DIGEST_MISMATCH",
                }
        for stage, field in (
            ("PER_LAYER_STATE", "per_layer_state_digests"),
            ("KV_STATE", "per_layer_kv_digests"),
        ):
            left_values = expected.get(field)
            right_values = observed.get(field)
            if (
                not isinstance(left_values, list)
                or not isinstance(right_values, list)
                or len(left_values) != 24
                or len(right_values) != 24
            ):
                raise AuthorityError(f"{stage} lacks 24 digests")
            for layer, (left, right) in enumerate(
                zip(left_values, right_values, strict=True)
            ):
                if left != right:
                    return {
                        "status": "DIVERGED",
                        "ordinal": ordinal,
                        "layer": layer,
                        "stage": stage,
                        "classification": f"{stage}_DIGEST_MISMATCH",
                    }
        if expected.get("logits_sha256") != observed.get("logits_sha256"):
            return {
                "status": "DIVERGED",
                "ordinal": ordinal,
                "stage": "TIED_LM_HEAD_LOGITS",
                "classification": "INDEPENDENT_REFERENCE_PRODUCT_LOGITS_MISMATCH",
            }
        if expected.get("expected_argmax") != observed.get("selected_token_id"):
            return {
                "status": "DIVERGED",
                "ordinal": ordinal,
                "stage": "RANK_SELECTION",
                "classification": (
                    "SELECTED_TOKEN_DISAGREES_WITH_INDEPENDENT_ARGMAX"
                ),
            }
        cache = observed.get("cache_comparison", {})
        if (
            cache.get("cache_free_logits_sha256")
            != cache.get("cacheful_logits_sha256")
            or cache.get("cache_free_selected_token_id")
            != cache.get("cacheful_selected_token_id")
        ):
            return {
                "status": "DIVERGED",
                "ordinal": ordinal,
                "stage": "CACHE_PLUMBING",
                "classification": "CACHE_FREE_CACHEFUL_MISMATCH",
            }
        if hardware.get("selected_token_id") != observed.get("selected_token_id"):
            return {
                "status": "DIVERGED",
                "ordinal": ordinal,
                "stage": "RTL_LOGIT_TILES",
                "classification": "RTL_PRODUCT_SELECTED_TOKEN_MISMATCH",
            }
    return {
        "status": "NO_DIVERGENCE_AT_RECORDED_BOUNDARIES",
        "ordinal": None,
        "stage": None,
        "classification": "BOUNDARIES_AGREE",
    }


def main() -> int:
    authority = validate_review()
    contract = load_object(PREDECESSOR / "contract.json")
    claim(authority)
    evidence: dict[str, Any] = {
        "schema": "ace2-r6-cf12-execution-journal-v1",
        "status": "AUTHORITY_CONSUMED",
        "reference": "NOT_STARTED",
        "product": "NOT_STARTED",
        "decode": "NOT_STARTED",
        "failures": [],
    }
    journal(evidence)

    reference_dir = ATTEMPT / "reference"
    reference_dir.mkdir()
    reference_argv = list(authority["reference_invocation"]["argv"])
    environment = dict(authority["environment"])
    pid, exit_code, stdout, stderr = run_process(
        reference_argv, environment, authority["timeout_policy"]["reference_seconds"]
    )
    atomic_bytes(reference_dir / "stdout.bin", stdout, exclusive=True)
    atomic_bytes(reference_dir / "stderr.bin", stderr, exclusive=True)
    artifact_path = Path(authority["reference_invocation"]["artifact_path"])
    artifact = load_object(artifact_path) if artifact_path.is_file() else None
    artifact_positions = (
        artifact.get("positions", []) if isinstance(artifact, dict) else []
    )
    receipt: dict[str, Any] = {
        "schema": contract["schemas"]["reference_process_receipt"],
        "status": (
            "PASS_AUTHORITY_LAUNCHED_REFERENCE"
            if exit_code == 0 and artifact is not None
            else "FAIL_REFERENCE_PROCESS"
        ),
        "challenge_nonce": authority["challenge_nonce"],
        "origin": "AUTHORITY_RUNNER_DIRECT_SUBPROCESS",
        "authority_runner_sha256": sha256_file(PACKAGE / "authority_runner.py"),
        "producer_implementation_sha256": contract["reference_producer"][
            "implementation_sha256"
        ],
        "producer_manifest_sha256": contract["reference_producer"]["manifest_sha256"],
        "argv_sha256": hashlib.sha256(canonical_bytes(reference_argv)).hexdigest(),
        "environment_sha256": hashlib.sha256(canonical_bytes(environment)).hexdigest(),
        "isolated_output_namespace": contract["isolated_output_namespace"],
        "process": {
            "pid": pid,
            "fresh_process": True,
            "exit_code": exit_code,
            "started_before_product": True,
        },
        "capture": {
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "reference_artifact_sha256": (
                sha256_file(artifact_path) if artifact is not None else None
            ),
            "full_logit_count": (
                artifact.get("full_logit_count") if artifact is not None else None
            ),
            "position_logits_sha256": (
                [
                    item.get("logits_sha256")
                    for item in artifact_positions
                    if isinstance(item, dict)
                ]
                if artifact is not None
                else []
            ),
        },
    }
    receipt["binding_sha256"] = load_validator().receipt_binding(receipt)
    atomic_json(reference_dir / "process-receipt.json", receipt, exclusive=True)
    evidence["reference"] = "RECEIPT_FSYNCED"
    journal(evidence)

    product_dir = ATTEMPT / "product"
    product_dir.mkdir()
    product_argv = list(authority["product_invocation"]["argv"])
    product_pid, product_exit, product_stdout, product_stderr = run_process(
        product_argv, environment, authority["timeout_policy"]["product_seconds"]
    )
    atomic_bytes(product_dir / "stdout.bin", product_stdout, exclusive=True)
    atomic_bytes(product_dir / "stderr.bin", product_stderr, exclusive=True)
    product_receipt = {
        "schema": "ace2-r6-cf12-product-process-receipt-v1",
        "pid": product_pid,
        "fresh_separate_process": product_pid != pid,
        "started_after_reference_receipt_fsync": True,
        "exit_code": product_exit,
        "argv_sha256": hashlib.sha256(canonical_bytes(product_argv)).hexdigest(),
        "stdout_sha256": hashlib.sha256(product_stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(product_stderr).hexdigest(),
    }
    atomic_json(product_dir / "process-receipt.json", product_receipt, exclusive=True)
    evidence["product"] = "RECEIPT_FSYNCED"
    journal(evidence)

    product_path = Path(authority["product_invocation"]["artifact_path"])
    if artifact is None or exit_code != 0:
        evidence["failures"].append(
            {
                "boundary": "INDEPENDENT_REFERENCE",
                "classification": "REFERENCE_PROCESS_FAILED",
                "product_rtl_still_launched": True,
            }
        )
    if not product_path.is_file() or product_exit != 0:
        evidence["failures"].append(
            {
                "boundary": "PRODUCT_RTL",
                "classification": "PRODUCT_OR_RTL_PROCESS_FAILED",
            }
        )
    if evidence["failures"]:
        evidence["status"] = "TERMINAL_PARTIAL_EVIDENCE_FSYNCED_NO_RETRY"
        journal(evidence)
        return 2

    product = load_object(product_path)
    validator = load_validator()
    validator.validate_reference_receipt(
        receipt,
        artifact,
        artifact_path,
        contract,
        {
            "source_artifact_sha256": sha256_file(product_path),
            "producer": product["producer"],
        },
    )
    combined = {
        "schema": "ace2-r6-cf12-combined-predecode-evidence-v1",
        "status": "ALL_REFERENCE_PRODUCT_RTL_RECEIPTS_FSYNCED_BEFORE_DECODE",
        "reference_receipt_sha256": sha256_file(
            reference_dir / "process-receipt.json"
        ),
        "product_receipt_sha256": sha256_file(product_dir / "process-receipt.json"),
        "reference_artifact_sha256": sha256_file(artifact_path),
        "product_artifact_sha256": sha256_file(product_path),
        "reference_positions": artifact["positions"],
        "product_positions": product["positions"],
        "rtl_positions": product["rtl_positions"],
        "first_divergence": first_divergence(
            artifact["positions"], product["positions"], product["rtl_positions"]
        ),
        "decode": {"status": "READY_FOR_EXACTLY_ONE_OFFICIAL_WHOLE_SEQUENCE_CALL"},
    }
    atomic_json(ATTEMPT / "combined-predecode-evidence.json", combined, exclusive=True)
    evidence["status"] = "PREDECODE_BARRIER_FSYNCED"
    evidence["decode"] = "AUTHORIZED_EXACTLY_ONCE"
    journal(evidence)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        authority["model"]["root"], local_files_only=True, trust_remote_code=False
    )
    token_ids = [item["selected_token_id"] for item in product["positions"]]
    decoded = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    readable = (
        isinstance(decoded, str)
        and bool(decoded.strip())
        and "\ufffd" not in decoded
        and any(character.isalnum() for character in decoded)
    )
    combined["decode"] = {
        "status": "PASS" if readable else "FAIL_CLOSED",
        "call_count": 1,
        "classification": (
            "PASS_READABLE_STRICT_UTF8"
            if readable
            else "WHOLE_SEQUENCE_NOT_READABLE_STRICT_UTF8"
        ),
    }
    combined["status"] = "TERMINAL_DIAGNOSTIC_EVIDENCE"
    atomic_json(ATTEMPT / "combined-predecode-evidence.json", combined)
    evidence["status"] = "TERMINAL_NO_RETRY"
    evidence["decode"] = "COMPLETED_EXACTLY_ONCE"
    journal(evidence)
    return 0 if readable else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorityError, OSError, ValueError, RuntimeError) as error:
        print(f"CF12 authority execution refused: {error}", file=os.sys.stderr)
        raise SystemExit(2)
'''


PREDECESSOR_TEST_SOURCE = r'''
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import receipt_validator as validator


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def artifact(contract: dict[str, object]) -> dict[str, object]:
    positions = []
    for ordinal in range(4):
        positions.append(
            {
                "expected_argmax": 7,
                "top_k": [
                    {"rank": 1, "token_id": 7, "logit": 2.0},
                    {"rank": 2, "token_id": 9, "logit": 1.0},
                ],
                "prompt_bytes_sha256": contract["prompt"]["sha256"],
                "chat_template_token_ids_sha256": contract["prompt"][
                    "chat_template_token_ids_sha256"
                ],
                "embedding_input_ids_sha256": digest(f"input-{ordinal}"),
                "per_layer_state_digests": [
                    digest(f"state-{ordinal}-{layer}") for layer in range(24)
                ],
                "per_layer_kv_digests": [
                    digest(f"kv-{ordinal}-{layer}") for layer in range(24)
                ],
                "logits_sha256": digest(f"logits-{ordinal}"),
            }
        )
    return {
        "schema": contract["schemas"]["reference_artifact"],
        "status": "SEALED_BEFORE_PRODUCT_EXECUTION",
        "producer": contract["reference_producer"]["identity"],
        "probe_spec_sha256": contract["cf11"]["future_probe_spec_sha256"],
        "model_sha256": contract["model"]["model_sha256"],
        "tokenizer_json_sha256": contract["model"]["tokenizer_json_sha256"],
        "full_logit_count": 151936,
        "positions": positions,
    }


class ReceiptValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads((PACKAGE / "contract.json").read_text())

    def valid_receipt(self, path: Path, value: dict[str, object]) -> dict[str, object]:
        receipt = {
            "schema": self.contract["schemas"]["reference_process_receipt"],
            "status": "PASS_AUTHORITY_LAUNCHED_REFERENCE",
            "challenge_nonce": self.contract["challenge_nonce"],
            "origin": "AUTHORITY_RUNNER_DIRECT_SUBPROCESS",
            "authority_runner_sha256": self.contract["authority_runner_sha256"],
            "producer_implementation_sha256": self.contract["reference_producer"][
                "implementation_sha256"
            ],
            "producer_manifest_sha256": self.contract["reference_producer"][
                "manifest_sha256"
            ],
            "argv_sha256": self.contract["reference_invocation"]["argv_sha256"],
            "environment_sha256": self.contract["reference_invocation"][
                "environment_sha256"
            ],
            "isolated_output_namespace": self.contract["isolated_output_namespace"],
            "process": {
                "pid": 1234,
                "fresh_process": True,
                "exit_code": 0,
                "started_before_product": True,
            },
            "capture": {
                "stdout_sha256": digest("stdout"),
                "stderr_sha256": digest("stderr"),
                "reference_artifact_sha256": validator.sha256_file(path),
                "full_logit_count": 151936,
                "position_logits_sha256": [
                    item["logits_sha256"] for item in value["positions"]
                ],
            },
        }
        receipt["binding_sha256"] = validator.receipt_binding(receipt)
        return receipt

    def test_valid_authority_receipt_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            value = artifact(self.contract)
            path.write_bytes(validator.canonical_bytes(value))
            receipt = self.valid_receipt(path, value)
            validator.validate_reference_receipt(receipt, value, path, self.contract)

    def test_challenge_and_receipt_mutation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            value = artifact(self.contract)
            path.write_bytes(validator.canonical_bytes(value))
            receipt = self.valid_receipt(path, value)
            receipt["challenge_nonce"] = digest("wrong-challenge")
            with self.assertRaisesRegex(validator.ReceiptError, "receipt differs"):
                validator.validate_reference_receipt(receipt, value, path, self.contract)

    def test_copied_product_claiming_exact_producer_still_needs_receipt(self) -> None:
        value = artifact(self.contract)
        copied = copy.deepcopy(value)
        copied["producer"] = copy.deepcopy(
            self.contract["reference_producer"]["identity"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "copied-product.json"
            path.write_bytes(validator.canonical_bytes(copied))
            with self.assertRaises((validator.ReceiptError, AttributeError, TypeError)):
                validator.validate_reference_receipt(
                    {}, copied, path, self.contract, copied
                )

    def test_shared_product_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            value = artifact(self.contract)
            path.write_bytes(validator.canonical_bytes(value))
            receipt = self.valid_receipt(path, value)
            product = {
                "source_artifact_sha256": digest("product"),
                "producer": {
                    "implementation_sha256": self.contract["reference_producer"][
                        "implementation_sha256"
                    ],
                    "manifest_sha256": digest("product-manifest"),
                },
            }
            with self.assertRaisesRegex(validator.ReceiptError, "not independent"):
                validator.validate_reference_receipt(
                    receipt, value, path, self.contract, product
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


AUTHORITY_TEST_SOURCE = r'''
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
STATE = PACKAGE.parent / f"{PACKAGE.name}-authority-state"
ROOT = PACKAGE.parents[2]
PREDECESSOR = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-reference-process-predecessor-cf12-0001"
)


def load_runner():
    specification = importlib.util.spec_from_file_location(
        "_cf12_authority_runner", PACKAGE / "authority_runner.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class Cf12AuthorityTests(unittest.TestCase):
    def test_pending_review_refuses_before_consumption(self) -> None:
        runner = load_runner()
        with self.assertRaisesRegex(runner.AuthorityError, "canonical L2"):
            runner.validate_review()
        self.assertFalse(STATE.exists())

    def test_contract_preserves_exactly_once_and_stage_gates(self) -> None:
        authority = json.loads((PACKAGE / "authority.json").read_text())
        self.assertEqual(authority["identity"], PACKAGE.name)
        self.assertEqual(authority["execution_limit"], 1)
        self.assertEqual(authority["authority_consumption"], "BEFORE_ANY_PROCESS")
        self.assertEqual(authority["retry_replay_resume_relaunch"], "FORBIDDEN")
        self.assertEqual(authority["stage1_state"], "OPEN")
        self.assertEqual(authority["stage2"], "FORBIDDEN")
        self.assertEqual(
            authority["status"], "GRANTED_PENDING_CANONICAL_L2_ACCEPTANCE"
        )

    def test_zero_execution_and_predecessor_immutability_record(self) -> None:
        report = json.loads((PACKAGE / "construction-report.json").read_text())
        self.assertEqual(
            report["status"],
            "PASS_ZERO_REFERENCE_PRODUCT_RTL_ATTEMPT_CONSUMPTION_RESULT",
        )
        for artifact_path in report["forbidden_execution_artifacts"]:
            self.assertFalse(Path(artifact_path).exists(), artifact_path)
        predecessor = json.loads((PREDECESSOR / "construction-report.json").read_text())
        self.assertEqual(
            predecessor["status"], "PASS_CF07_CF08_CF09_CF10_CF11_BYTE_MODE_UNCHANGED"
        )

    def test_claim_creates_bound_writable_state_and_attempt_without_execution(
        self,
    ) -> None:
        runner = load_runner()
        authority = json.loads((PACKAGE / "authority.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / STATE.name
            consumed = state / "authorization-consumed.json"
            attempt = state / "attempt-0001"
            authority["consumption_evidence_namespace"] = str(state)
            authority["output_namespace"] = str(attempt)
            with (
                mock.patch.multiple(
                    runner,
                    STATE=state,
                    CONSUMED=consumed,
                    ATTEMPT=attempt,
                    FORBIDDEN=(state,),
                ),
                mock.patch.object(
                    runner,
                    "run_process",
                    side_effect=AssertionError("claim must not execute a process"),
                ),
            ):
                runner.claim(authority)
            self.assertTrue(consumed.is_file())
            self.assertEqual(
                json.loads(consumed.read_text())["status"],
                "CONSUMED_BEFORE_REFERENCE_OR_PRODUCT_PROCESS",
            )
            self.assertTrue(attempt.is_dir())
            self.assertTrue(stat.S_IMODE(state.stat().st_mode) & stat.S_IWUSR)
            self.assertFalse(stat.S_IMODE(consumed.stat().st_mode) & stat.S_IWUSR)

    def test_main_claims_once_and_reaches_mocked_reference_launch(self) -> None:
        runner = load_runner()
        authority = json.loads((PACKAGE / "authority.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / STATE.name
            consumed = state / "authorization-consumed.json"
            attempt = state / "attempt-0001"
            authority["consumption_evidence_namespace"] = str(state)
            authority["output_namespace"] = str(attempt)
            reference_launch = mock.Mock(
                side_effect=RuntimeError("mocked reference launch reached")
            )
            with (
                mock.patch.multiple(
                    runner,
                    STATE=state,
                    CONSUMED=consumed,
                    ATTEMPT=attempt,
                    FORBIDDEN=(state,),
                ),
                mock.patch.object(
                    runner, "validate_review", return_value=authority
                ),
                mock.patch.object(runner, "run_process", reference_launch),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "mocked reference launch reached"
                ):
                    runner.main()
            reference_launch.assert_called_once_with(
                authority["reference_invocation"]["argv"],
                authority["environment"],
                authority["timeout_policy"]["reference_seconds"],
            )
            self.assertTrue(consumed.is_file())
            self.assertTrue(attempt.is_dir())
            self.assertTrue((attempt / "evidence-journal.json").is_file())
            self.assertTrue((attempt / "reference").is_dir())

    def test_bound_runner_requires_reference_before_product(self) -> None:
        source = (PACKAGE / "authority_runner.py").read_text()
        reference = source.index("reference_argv =")
        reference_receipt = source.index('"process-receipt.json", receipt')
        product = source.index("product_argv =")
        self.assertLess(reference, reference_receipt)
        self.assertLess(reference_receipt, product)
        self.assertIn("def first_divergence(", source)
        self.assertNotIn("VALIDATOR_RECOMPUTATION_REQUIRED", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


LAUNCH_SOURCE = """#!/bin/sh
set -eu
exec python3 "$(dirname "$0")/authority_runner.py" "$@"
"""


REJECT_SOURCE = """#!/bin/sh
printf '%s\\n' 'The CF12 replacement predecessor is a reviewed contract, not an execution surface.' >&2
exit 2
"""


def build_predecessor(
    target: Path,
    *,
    challenge: str,
    runtime: dict[str, object],
    prompt: bytes,
    token_ids: list[int],
    review: dict[str, Any],
    before: dict[str, object],
    authority_runner_sha256: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    target.mkdir()
    write_text(target / "receipt_validator.py", RECEIPT_VALIDATOR_SOURCE, 0o444)
    write_text(target / "tests/test_receipt_validator.py", PREDECESSOR_TEST_SOURCE, 0o444)
    write_text(target / "launch.sh", REJECT_SOURCE, 0o555)
    producer = load_object(CF11 / "future-probe-spec.json")[
        "independent_reference_requirements"
    ]["authorized_reference_producer"]
    reference_artifact = (
        STATE / "attempt-0001/reference/independent-reference.json"
    )
    reference_argv = [
        runtime["python_executable"],
        str(CF11 / "independent_reference_producer.py"),
        "--model",
        str(MODEL_ROOT),
        "--prompt-file",
        str(PACKAGE / "input.txt"),
        "--output",
        str(reference_artifact),
    ]
    contract = {
        "schema": "ace2-r6-cf12-authority-launched-reference-predecessor-v1",
        "status": "CONSTRUCTED_CANONICAL_L2_REVIEW_REQUIRED",
        "supersedes_only": "CF11_EXTERNAL_ED25519_EXECUTION_ATTESTATION_PREREQUISITE",
        "does_not_modify_cf11": True,
        "trust_root": "AUTHORITY_CONTROLLED_DIRECT_PROCESS_LAUNCH_AND_CAPTURE",
        "challenge_nonce": challenge,
        "authority_runner_sha256": authority_runner_sha256,
        "reference_producer": {
            "implementation_path": str(CF11 / "independent_reference_producer.py"),
            "implementation_sha256": sha256_file(
                CF11 / "independent_reference_producer.py"
            ),
            "manifest_path": str(CF11 / "reference-producer-manifest.json"),
            "manifest_sha256": sha256_file(CF11 / "reference-producer-manifest.json"),
            "identity": producer,
            "product_source_imports": "FORBIDDEN",
            "product_position_inputs": "FORBIDDEN",
        },
        "runtime": runtime,
        "model": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": MODEL_REVISION,
            "root": str(MODEL_ROOT),
            "model_sha256": MODEL_SHA256,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        },
        "prompt": {
            "bytes_hex": prompt.hex(),
            "byte_count": len(prompt),
            "sha256": sha256_bytes(prompt),
            "chat_template_token_ids": token_ids,
            "chat_template_token_ids_sha256": sha256_bytes(
                canonical_bytes(token_ids)
            ),
            "template_id": "QWEN25_INSTRUCT_APPLY_CHAT_TEMPLATE_ADD_GENERATION_PROMPT",
        },
        "reference_invocation": {
            "argv": reference_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(reference_argv)),
            "environment": environment,
            "environment_sha256": sha256_bytes(canonical_bytes(environment)),
            "fresh_separate_process": True,
            "must_complete_before_product_start": True,
        },
        "isolated_output_namespace": str(STATE / "attempt-0001/reference"),
        "schemas": {
            "reference_artifact": "ace2-r6-cf11-independent-reference-artifact-v1",
            "reference_process_receipt": (
                "ace2-r6-cf12-authority-reference-process-receipt-v1"
            ),
            "combined_evidence": "ace2-r6-cf12-combined-predecode-evidence-v1",
        },
        "receipt_requirements": {
            "authority_runner_recomputes_binding": True,
            "challenge_bound": True,
            "argv_environment_bound": True,
            "stdout_stderr_hashes": True,
            "artifact_and_four_position_full_logit_digests": True,
            "full_logit_count": 151936,
            "reject_artifact_supplied_identity_without_process_receipt": True,
            "reject_product_shared_or_copied_provenance": True,
        },
        "atomicity": {
            "file_fsync_before_publication": True,
            "directory_fsync_after_publication": True,
            "partial_evidence_at_every_boundary": True,
            "all_reference_product_rtl_evidence_before_decode": True,
        },
        "cf11": {
            "review_acceptance": review,
            "package_contract_sha256": sha256_file(CF11 / "package-contract.json"),
            "future_probe_spec_sha256": sha256_file(CF11 / "future-probe-spec.json"),
            "future_probe_implementation_sha256": sha256_file(
                CF11 / "future_probe.py"
            ),
            "independent_reference_producer_sha256": sha256_file(
                CF11 / "independent_reference_producer.py"
            ),
        },
        "independence": "GENUINE_SEPARATE_HOST_REFERENCE_COMPUTATION_REQUIRED",
        "construction_execution_limit": 0,
        "retry_replay_resume_relaunch": "FORBIDDEN",
        "stage1_state": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
    }
    write_json(target / "contract.json", contract)
    write_json(
        target / "construction-report.json",
        {
            "schema": "ace2-r6-cf12-predecessor-construction-report-v1",
            "status": "PASS_CF07_CF08_CF09_CF10_CF11_BYTE_MODE_UNCHANGED",
            "predecessor_namespaces": before,
            "reference_execution": "NOT_EXECUTED",
            "product_execution": "NOT_EXECUTED",
            "rtl_execution": "NOT_EXECUTED",
            "attempt_count": 0,
        },
    )
    (target / "tests").chmod(0o555)
    return contract


def build_package(
    target: Path,
    *,
    challenge: str,
    runtime: dict[str, object],
    prompt: bytes,
    token_ids: list[int],
    descriptor: dict[str, Any],
    environment: dict[str, str],
    predecessor_contract: dict[str, Any],
) -> None:
    target.mkdir()
    write_text(target / "authority_runner.py", AUTHORITY_RUNNER_SOURCE, 0o444)
    write_text(target / "product_probe.py", PRODUCT_PROBE_SOURCE, 0o444)
    write_text(target / "tests/test_cf12_authority.py", AUTHORITY_TEST_SOURCE, 0o444)
    write_text(target / "launch.sh", LAUNCH_SOURCE, 0o555)
    (target / "input.txt").write_bytes(prompt)
    (target / "input.txt").chmod(0o444)
    product_artifact = STATE / "attempt-0001/product/product-rtl-evidence.json"
    endpoint_output = STATE / "attempt-0001/product/rtl-completion"
    product_argv = [
        runtime["python_executable"],
        str(PACKAGE / "product_probe.py"),
        "--output",
        str(product_artifact),
        "--endpoint-output",
        str(endpoint_output),
    ]
    endpoint_argv = [
        str(CF10 / "Vace2_shell_runtime_harness"),
        "--package",
        str(CF10 / "inputs/runtime-package.bin"),
        "--image",
        str(CF10 / "inputs/memory-image.bin"),
        "--model",
        str(MODEL_ROOT / "model.safetensors"),
        "--output",
        str(endpoint_output),
        "--identity-profile",
        "r6-final-export-product-cf10",
    ]
    product_manifest = {
        "schema": "ace2-r6-cf12-product-probe-manifest-v1",
        "implementation_sha256": sha256_file(target / "product_probe.py"),
        "accepted_runtime_sha256": sha256_file(
            ROOT
            / "build/ace2_chat_demo"
            / "stage1-w4a8-r6-chat-rtl-qualification-prep-cf07"
            / "accepted_runtime.py"
        ),
        "product_driver_source_sha256": sha256_file(CF10 / "product_driver.py"),
        "endpoint_binary_sha256": sha256_file(
            CF10 / "Vace2_shell_runtime_harness"
        ),
        "independent_reference_imports": "FORBIDDEN",
    }
    write_json(target / "product-probe-manifest.json", product_manifest)
    bindings = {
        "schema": "ace2-r6-cf12-probe-bindings-v1",
        "identity": IDENTITY,
        "transaction_nonce": TRANSACTION_NONCE,
        "challenge_nonce": challenge,
        "prompt": predecessor_contract["prompt"],
        "model": predecessor_contract["model"],
        "accepted_runtime": {
            "path": str(
                ROOT
                / "build/ace2_chat_demo"
                / "stage1-w4a8-r6-chat-rtl-qualification-prep-cf07"
                / "accepted_runtime.py"
            ),
            "preflight_hashes_path": str(
                ROOT
                / "build/ace2_chat_demo"
                / "stage1-w4a8-r6-chat-rtl-qualification-prep-cf07"
                / "preflight-hashes.json"
            ),
            "hardware_export_manifest": str(
                ROOT
                / "build/ace2_chat_demo"
                / "stage1-w4a8-capture-aware-successor-cf01"
                / "future-runs/stage1-w4a8-qat-capture-aware-r6-cf01"
                / "hardware-export/manifest.json"
            ),
            "implementation_sha256": product_manifest["accepted_runtime_sha256"],
        },
        "product_producer": {
            "kind": "ACE2_W4A8_PRODUCT_DIAGNOSTIC",
            "implementation_sha256": product_manifest["implementation_sha256"],
            "manifest_sha256": sha256_bytes(canonical_bytes(product_manifest)),
        },
        "endpoint": {
            "argv": endpoint_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(endpoint_argv)),
            "timeout_seconds": 7200,
            "output_namespace": str(endpoint_output),
            "must_run_after_host_failure": True,
            "logit_evidence": "AUTHENTICATED_4748_TILE_AGGREGATE_PER_POSITION",
            "kv_digest_count_per_position": 24,
        },
        "cf11": predecessor_contract["cf11"],
        "official_descriptor_sha256": sha256_file(
            CF10 / "inputs/descriptor.json"
        ),
        "final_export_manifest_sha256": sha256_file(
            CF10 / "inputs/final-export-manifest.json"
        ),
        "rtl_provenance_sha256": sha256_file(CF11 / "rtl-provenance.json"),
        "rtl_source_closure_sha256": load_object(CF11 / "future-probe-spec.json")[
            "bindings"
        ]["rtl_source_closure_sha256"],
        "full_logit_count": 151936,
        "required_positions": 4,
    }
    write_json(target / "bindings.json", bindings)
    reference_artifact = (
        STATE / "attempt-0001/reference/independent-reference.json"
    )
    reference_argv = predecessor_contract["reference_invocation"]["argv"]
    authority = {
        "schema": "ace2-r6-cf12-exactly-once-diagnostic-authority-v1",
        "status": "GRANTED_PENDING_CANONICAL_L2_ACCEPTANCE",
        "authorized_action": "EXECUTE_DIAGNOSTIC_PROBE_EXACTLY_ONCE",
        "not_authorized": "STAGE1_PRODUCT_COMPLETION",
        "identity": IDENTITY,
        "transaction_nonce": TRANSACTION_NONCE,
        "challenge_nonce": challenge,
        "durable_task_id": DURABLE_TASK_ID,
        "authority_cardinality": 1,
        "execution_limit": 1,
        "authority_consumption": "BEFORE_ANY_PROCESS",
        "reference_invocation": {
            "argv": reference_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(reference_argv)),
            "artifact_path": str(reference_artifact),
            "must_launch_first_in_fresh_process": True,
        },
        "product_invocation": {
            "argv": product_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(product_argv)),
            "artifact_path": str(product_artifact),
            "must_launch_in_separate_process_after_reference_receipt_fsync": True,
            "must_launch_even_after_reference_failure": True,
        },
        "environment": environment,
        "environment_sha256": sha256_bytes(canonical_bytes(environment)),
        "runtime": runtime,
        "model": predecessor_contract["model"],
        "prompt": predecessor_contract["prompt"],
        "bindings_sha256": sha256_file(target / "bindings.json"),
        "predecessor_contract_sha256": sha256_bytes(
            canonical_bytes(predecessor_contract)
        ),
        "timeout_policy": {
            "reference_seconds": 7200,
            "product_seconds": 7200,
            "signal_grace_seconds": 5,
            "timeout_is_terminal": True,
        },
        "consumption_evidence_namespace": str(STATE),
        "output_namespace": str(STATE / "attempt-0001"),
        "output_namespace_must_not_exist_before_consumption": True,
        "evidence_policy": {
            "atomic_file_and_directory_fsync": True,
            "partial_evidence_at_every_boundary": True,
            "reference_stdout_stderr_artifact_and_process_receipt": True,
            "product_stdout_stderr_artifact_and_process_receipt": True,
            "four_positions_before_decode": True,
            "reference_full_logits_digest_count": 151936,
            "product_full_logits_digest_count": 151936,
            "rtl_authenticated_logit_tile_aggregate": True,
            "twenty_four_layer_and_kv_digests": True,
            "cache_free_cacheful_ids_and_digests": True,
            "selected_id_rank_margin": True,
            "first_divergence_classification": True,
            "formatting_only_success": "REJECTED",
            "self_derived_expected_tokens": "FORBIDDEN",
        },
        "terminal_failure_semantics": {
            "reference_failure": "PRODUCT_RTL_STILL_RUN_THEN_TERMINAL_NO_RETRY",
            "product_failure": "PERSIST_PARTIAL_THEN_TERMINAL_NO_RETRY",
            "rtl_failure": "PERSIST_PARTIAL_THEN_TERMINAL_NO_RETRY",
            "decode_failure": "TERMINAL_AFTER_RTL_NO_RETRY",
            "unknown_failure": "TERMINAL_NO_RETRY",
        },
        "retry_replay_resume_relaunch": "FORBIDDEN",
        "activation_gate": {
            "mission_id": "bfd265994985",
            "review_root": str(CF12_REVIEW),
            "producer_role": "reviewer",
            "review_status": "done",
            "review_scope": (
                "EXACT_PREDECESSOR_CONTRACT_AND_CF12_AUTHORITY_PACKAGE_HASHES"
            ),
        },
        "stage1_state": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
    }
    write_json(target / "authority.json", authority)
    targets = []
    for base, names in (
        (
            PREDECESSOR,
            (
                "contract.json",
                "receipt_validator.py",
                "tests/test_receipt_validator.py",
                "construction-report.json",
            ),
        ),
        (
            PACKAGE,
            (
                "authority.json",
                "authority_runner.py",
                "product_probe.py",
                "product-probe-manifest.json",
                "bindings.json",
                "tests/test_cf12_authority.py",
                "input.txt",
            ),
        ),
    ):
        actual_base = target if base == PACKAGE else target.parent / PREDECESSOR.name
        for name in names:
            path = actual_base / name
            targets.append(
                {
                    "path": (base / name).relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        target / "review-request.json",
        {
            "schema": "ace2-r6-cf12-canonical-l2-review-request-v1",
            "status": "PENDING_INDEPENDENT_REVIEW",
            "mission_id": "bfd265994985",
            "targets": targets,
            "predecessor_contract_sha256": authority[
                "predecessor_contract_sha256"
            ],
            "authority_sha256": sha256_file(target / "authority.json"),
            "execution_before_acceptance": "FORBIDDEN",
            "stage_transition": "DISABLED",
        },
    )
    write_json(
        target / "construction-report.json",
        {
            "schema": "ace2-r6-cf12-construction-report-v1",
            "status": "PASS_ZERO_REFERENCE_PRODUCT_RTL_ATTEMPT_CONSUMPTION_RESULT",
            "forbidden_execution_artifacts": [
                str(STATE),
                str(PACKAGE / "authorization-consumed.json"),
                str(PACKAGE / "attempt-0001"),
                str(PACKAGE / "execution-registry.json"),
                str(PACKAGE / "completion-summary.json"),
                str(PACKAGE / "product-result.json"),
            ],
            "consumption_evidence_namespace": str(STATE),
            "static_package_mode": "0555",
            "reference_execution": "NOT_EXECUTED",
            "product_execution": "NOT_EXECUTED",
            "rtl_execution": "NOT_EXECUTED",
            "decode_execution": "NOT_EXECUTED",
            "consumption_count": 0,
            "attempt_count": 0,
            "stage1_state": "OPEN",
            "stage2": "FORBIDDEN",
        },
    )
    (target / "tests").chmod(0o555)


def add_reproducibility_report(first: Path, second: Path, schema: str) -> None:
    first_manifest = material_manifest(first)
    second_manifest = material_manifest(second)
    if first_manifest != second_manifest:
        raise RuntimeError(f"two-build byte/mode reproducibility differs: {first.name}")
    report = {
        "schema": schema,
        "status": "PASS_TWO_BUILD_BYTE_MODE_EQUAL",
        "build_count": 2,
        "material_entry_count": len(first_manifest),
        "material_manifest_sha256": sha256_bytes(canonical_bytes(first_manifest)),
    }
    write_json(first / "reproducibility-report.json", report)
    write_json(second / "reproducibility-report.json", report)
    if namespace_manifest(first) != namespace_manifest(second):
        raise RuntimeError(f"complete two-build namespace differs: {first.name}")


def main() -> int:
    if PREDECESSOR.exists() or PACKAGE.exists():
        raise RuntimeError("fresh CF12 predecessor or authority namespace already exists")
    review = validate_cf11_review()
    prompt, token_ids, descriptor = validate_frozen_inputs()
    runtime = runtime_identity()
    before = {
        name: namespace_manifest(path) for name, path in PREDECESSORS.items()
    }
    challenge = secrets.token_hex(32)
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    authority_runner_sha256 = sha256_bytes(
        textwrap.dedent(AUTHORITY_RUNNER_SOURCE).lstrip().encode("utf-8")
    )

    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cf12-construction-", dir=PACKAGE.parent
    ) as temporary:
        root = Path(temporary)
        for build_name in ("build-a", "build-b"):
            build_root = root / build_name
            build_root.mkdir()
            predecessor_target = build_root / PREDECESSOR.name
            package_target = build_root / PACKAGE.name
            contract = build_predecessor(
                predecessor_target,
                challenge=challenge,
                runtime=runtime,
                prompt=prompt,
                token_ids=token_ids,
                review=review,
                before=before,
                authority_runner_sha256=authority_runner_sha256,
                environment=environment,
            )
            build_package(
                package_target,
                challenge=challenge,
                runtime=runtime,
                prompt=prompt,
                token_ids=token_ids,
                descriptor=descriptor,
                environment=environment,
                predecessor_contract=contract,
            )
        first_predecessor = root / "build-a" / PREDECESSOR.name
        second_predecessor = root / "build-b" / PREDECESSOR.name
        first_package = root / "build-a" / PACKAGE.name
        second_package = root / "build-b" / PACKAGE.name
        add_reproducibility_report(
            first_predecessor,
            second_predecessor,
            "ace2-r6-cf12-predecessor-two-build-reproducibility-v1",
        )
        add_reproducibility_report(
            first_package,
            second_package,
            "ace2-r6-cf12-authority-two-build-reproducibility-v1",
        )
        after = {
            name: namespace_manifest(path) for name, path in PREDECESSORS.items()
        }
        if after != before:
            raise RuntimeError("CF07-CF11 changed during CF12 construction")
        os.replace(first_predecessor, PREDECESSOR)
        os.replace(first_package, PACKAGE)
        PREDECESSOR.chmod(0o555)
        PACKAGE.chmod(0o555)
        directory = os.open(PACKAGE.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    print(PREDECESSOR)
    print(PACKAGE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"CF12 construction refused: {error}", file=sys.stderr)
        raise SystemExit(2)
