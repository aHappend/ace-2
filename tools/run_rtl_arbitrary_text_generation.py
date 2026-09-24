#!/usr/bin/env python3
"""Preflight and exactly-once checkpoint-176 arbitrary-text RTL generation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import rtl_arbitrary_text_generation_backend as backend


REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MISSION_ID = "rtl-arbitrary-text-generation-integration-v1"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
TOKENIZER_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
TOKENIZER_SNAPSHOT = (
    ROOT
    / "build/ace2_chat_demo/qwen25-05b-instruct-awq-software-baseline-cf01/official"
)
TOKENIZER_SHA256 = {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
CANONICAL_HELLO_CHAT_TOKEN_IDS = (
    151644,
    8948,
    198,
    2610,
    525,
    1207,
    16948,
    11,
    3465,
    553,
    54364,
    14817,
    13,
    1446,
    525,
    264,
    10950,
    17847,
    13,
    151645,
    198,
    151644,
    872,
    198,
    9707,
    151645,
    198,
    151644,
    77091,
    198,
)
ATTEMPT_24L = (
    ROOT
    / "evidence/verification/lora-v4-two-token-24layer-rtl-v1/attempt-0003"
)
ATTEMPT_LM_HEAD = (
    ROOT
    / "evidence/verification/lora-v4-lm-head-top-token-rtl-v1/attempt-0001"
)
EXPECTED_24L_SUMS_SHA256 = (
    "0d3cc8064f947183c83ae24add4631cade31b164a7331a1507b86d27d5a29c85"
)
EXPECTED_LM_HEAD_SUMS_SHA256 = (
    "3fea254a92d55bfae2e6fd8bf043771410b08b93cc662f75632a2a652b46dfce"
)
EXPECTED_24L_STATUS = (
    "PASS_24LAYER_EXACT_INTEGER_CYCLE_REPAIR_CHECK_AWAITING_INDEPENDENT_REVIEW"
)
EXPECTED_LM_HEAD_STATUS = "PASS_LM_HEAD_TOP_TOKEN_EXACT_CHECK_AWAITING_FRESH_L2"
EXPECTED_LM_HEAD_CHECK_STATUS = "PASS_LM_HEAD_TOP_TOKEN_EXACT_CHECK_AWAITING_FRESH_L2"
EXPECTED_MODEL_SHA256 = (
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
)
EXPECTED_ADAPTER_SHA256 = (
    "c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28"
)
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "8cee9b9343e49b66ee2ea427578f87335ffa259919356cdd119cd8712ff7eeee"
)
PREFLIGHT_0004 = ROOT / "build/rtl-arbitrary-text-generation-preflight-v1/preflight-0004"
EXPECTED_PREFLIGHT_0004_SHA256 = (
    "c866285310fe06f7fb7b87e91a003550caa7e40f2822de5a8276f1d2b356f0c2"
)
EXPECTED_PREFLIGHT_0004_SUMS_SHA256 = (
    "9abd2fa6f5c0943bbeef446815d78f35e243aeabfcb5b50be6d8ab72b178f19d"
)
EXPECTED_PREFLIGHT_0004_RUNTIME_SHA256 = (
    "a8a33b74a4faecfc4ef63d0135a42c02f20b0776fb8eec1179879bb3db98018f"
)
EXPECTED_PREFLIGHT_0004_PROTECTED_SHA256 = (
    "bc9e2fe839cd907f32b2bd61ce6fa62a654a0651d688de8a2d3657a5e3d5a775"
)
TOKENIZER_BEHAVIOR_PROBES = (
    ("ascii", "Hello, RTL!"),
    ("unicode", "Grüße, 世界"),
    ("composed", "café"),
    ("decomposed", "cafe\u0301"),
    ("emoji", "logic ⚙️"),
    ("multiline", "line one\nline two\tend"),
)
MIN_NEW_TOKENS = 2
MAX_NEW_TOKENS = 4
DEFAULT_MAX_PROMPT_TOKENS = len(CANONICAL_HELLO_CHAT_TOKEN_IDS)
MAX_CONTEXT_TOKENS = backend.MAX_CONTEXT_TOKENS
MODEL_OUTPUT_DOMAIN = backend.MODEL_OUTPUT_DOMAIN
PROTECTED_FILES = (
    ROOT / "design/SPEC.md",
    ROOT / "design/BENCHMARK_INTERFACE.json",
    ROOT / "research/PIPELINE_STATE.json",
)
PYTHON_SOURCE_ENTRYPOINTS = (Path(__file__).resolve(),)
RTL_SOURCE_SUFFIXES = frozenset({".sv", ".svh", ".v", ".vh"})
TEXT_ARTIFACT_SUFFIXES = frozenset({".json", ".log", ".sv", ".svh", ".txt"})


class PreflightError(RuntimeError):
    """A fail-closed preflight rejection."""


class OfficialAuthorizationError(RuntimeError):
    """The official exactly-once execution lacks a valid Fresh-L2 authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def fail(message: str) -> NoReturn:
    raise PreflightError(message)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


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


def file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"required file is missing: {public_path(path)}")
    observed = sha256_file(path)
    if expected_sha256 is not None:
        require(
            observed == expected_sha256,
            f"SHA-256 mismatch for {public_path(path)}: {observed}",
        )
    return {
        "path": public_path(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def identity_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def directory_tree_record(root: Path) -> dict[str, Any]:
    require(root.is_dir(), f"required directory is missing: {public_path(root)}")
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        record = {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.is_symlink():
            record["symlink_target"] = os.readlink(path)
        records.append(record)
    require(bool(records), f"required directory is empty: {public_path(root)}")
    return {
        "path": public_path(root),
        "member_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "sha256": identity_sha256(records),
        "records": records,
    }


def verify_directory_identity(identity_path: Path) -> dict[str, Any]:
    identity = load_json(identity_path)
    require(identity.get("schema_version") == 1, "checkpoint identity schema differs")
    files = identity.get("files")
    require(isinstance(files, dict) and bool(files), "checkpoint identity files are missing")
    raw_path = Path(str(identity.get("path", "")))
    require(bool(str(raw_path)), "checkpoint identity path is missing")
    root = raw_path.resolve() if raw_path.is_absolute() else (ROOT / raw_path).resolve()
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        observed[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    require(observed == files, f"checkpoint identity file tree differs: {public_path(root)}")
    digest = hashlib.sha256()
    for name, metadata in observed.items():
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(str(metadata["sha256"])))
        digest.update(int(metadata["bytes"]).to_bytes(8, "big"))
    observed_tree = digest.hexdigest()
    require(observed_tree == identity.get("tree_sha256"), "checkpoint identity tree hash differs")
    return {
        "path": public_path(root),
        "identity": file_record(identity_path),
        "tree_sha256": observed_tree,
        "member_count": len(observed),
        "total_bytes": sum(int(item["bytes"]) for item in observed.values()),
        "files": observed,
    }


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON is missing: {public_path(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {public_path(path)}")
    return value


def validate_max_new_tokens(value: int) -> int:
    if not MIN_NEW_TOKENS <= value <= MAX_NEW_TOKENS:
        fail(
            f"--max-new-tokens must be between {MIN_NEW_TOKENS} and "
            f"{MAX_NEW_TOKENS} for bounded preflight"
        )
    return value


def validate_generation_bounds(
    prompt_token_count: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, int]:
    validate_max_new_tokens(max_new_tokens)
    require(max_prompt_tokens >= 1, "--max-prompt-tokens must be positive")
    require(prompt_token_count >= 1, "prompt tokenizer output must be non-empty")
    require(
        prompt_token_count <= max_prompt_tokens,
        f"prompt token count {prompt_token_count} exceeds bounded limit {max_prompt_tokens}",
    )
    total_context_tokens = prompt_token_count + max_new_tokens
    processed_positions = total_context_tokens - 1
    require(
        total_context_tokens <= MAX_CONTEXT_TOKENS,
        (
            f"prompt plus generation requires {total_context_tokens} total tokens; "
            f"accepted host/backend context bound is {MAX_CONTEXT_TOKENS}"
        ),
    )
    return {
        "prompt_token_count": prompt_token_count,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "total_context_tokens": total_context_tokens,
        "processed_positions": processed_positions,
        "rtl_context_bound": MAX_CONTEXT_TOKENS,
    }


def canonical_chat_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    chat_template = getattr(tokenizer, "chat_template", None)
    require(isinstance(chat_template, str), "canonical tokenizer has no chat template")
    require(
        sha256_bytes(chat_template.encode("utf-8")) == CHAT_TEMPLATE_SHA256,
        "canonical Qwen chat template identity differs",
    )
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    require(isinstance(token_ids, list), "chat template tokenizer output must be a list")
    return [int(value) for value in token_ids]


def snapshot_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("ACE2_QWEN25_SNAPSHOT")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(
        ROOT / "build/ace2_chat_demo/cf20-bf16-portable-snapshot" / REVISION
    )
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        candidates.append(
            Path(hub_cache).expanduser()
            / "models--Qwen--Qwen2.5-0.5B-Instruct"
            / "snapshots"
            / REVISION
        )
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()
    candidates.append(
        hf_home
        / "hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
        / REVISION
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_snapshot(explicit: Path | None = None) -> Path:
    candidates = [explicit.expanduser().resolve()] if explicit is not None else snapshot_candidates()
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")):
            require(
                candidate.name == REVISION,
                f"snapshot revision directory must be {REVISION}: {public_path(candidate)}",
            )
            return candidate
    searched = ", ".join(public_path(path) for path in candidates)
    fail(f"canonical local snapshot is unavailable; searched: {searched}")


def resolve_tokenizer_snapshot() -> Path:
    for name, expected_sha256 in TOKENIZER_SHA256.items():
        path = TOKENIZER_SNAPSHOT / name
        require(path.is_file(), f"official AWQ tokenizer file is absent: {public_path(path)}")
        require(
            sha256_file(path) == expected_sha256,
            f"official AWQ tokenizer file hash differs: {name}",
        )
    return TOKENIZER_SNAPSHOT.resolve()


def read_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt_file is not None:
        raw = prompt_file.read_bytes()
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PreflightError("prompt file is not valid UTF-8") from error
    elif prompt is not None:
        value = prompt
    else:
        fail("one prompt source is required")
    require(bool(value.strip()), "prompt must contain non-whitespace text")
    return value


def parse_sha256_manifest(root: Path, manifest: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        fields = line.split("  ", 1)
        require(len(fields) == 2, f"malformed SHA256SUMS line {line_number}")
        expected, relative = fields
        require(
            len(expected) == 64 and all(char in "0123456789abcdef" for char in expected),
            f"invalid SHA-256 on SHA256SUMS line {line_number}",
        )
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe SHA256SUMS path on line {line_number}",
        )
        normalized = relative_path.as_posix()
        require(normalized not in seen, f"duplicate SHA256SUMS path: {normalized}")
        seen.add(normalized)
        records.append((expected, root / relative_path))
    require(bool(records), f"empty SHA256SUMS: {public_path(manifest)}")
    return records


def verify_sealed_attempt(
    root: Path,
    *,
    expected_sums_sha256: str,
    status_path: Path,
    expected_status: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    sums = root / "SHA256SUMS"
    sums_record = file_record(sums, expected_sha256=expected_sums_sha256)
    records = parse_sha256_manifest(root, sums)
    total_bytes = 0
    for expected, path in records:
        record = file_record(path, expected_sha256=expected)
        total_bytes += int(record["bytes"])
    status = load_json(status_path)
    require(
        status.get("status") == expected_status,
        f"sealed attempt status changed: {public_path(status_path)}",
    )
    return {
        "path": public_path(root),
        "status": status["status"],
        "sha256s": sums_record,
        "members_verified": len(records),
        "member_bytes_verified": total_bytes,
        "verification_wall_seconds": time.perf_counter() - started,
        "mutation": False,
        "replay": False,
    }


def module_identity(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    require(
        spec is not None and spec.origin not in {None, "built-in", "frozen"},
        f"required module has no file identity: {module_name}",
    )
    return file_record(Path(str(spec.origin)))


def package_identity(
    distribution_name: str,
    module_name: str,
    *,
    native_modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    metadata_records: dict[str, Any] = {}
    for entry in distribution.files or []:
        if entry.name not in {"METADATA", "RECORD"} or ".dist-info" not in entry.as_posix():
            continue
        path = Path(distribution.locate_file(entry))
        if path.is_file():
            metadata_records[entry.name] = file_record(path)
    require("METADATA" in metadata_records, f"{distribution_name} METADATA is missing")
    record = {
        "distribution": distribution_name,
        "version": distribution.version,
        "module": module_name,
        "module_file": module_identity(module_name),
        "native_module_files": {
            native: module_identity(native) for native in native_modules
        },
        "distribution_metadata": metadata_records,
    }
    return {**record, "identity_sha256": identity_sha256(record)}


def command_identity(program: str, version_args: list[str]) -> dict[str, Any]:
    executable = shutil.which(program)
    require(executable is not None, f"required tool is unavailable on PATH: {program}")
    completed, _child_record = backend.tracked_run(
        [executable, *version_args],
        cwd=ROOT,
        timeout=15,
    )
    combined = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    require(completed.returncode == 0, f"{program} version command failed")
    require(bool(combined), f"{program} returned no version identity")
    record = {
        "argv": [program, *version_args],
        "invoked_executable": public_path(Path(executable)),
        "resolved_executable": public_path(Path(executable).resolve()),
        "executable_sha256": sha256_file(Path(executable)),
        "version_output": combined,
    }
    return {**record, "identity_sha256": identity_sha256(record)}


def toolchain_identity() -> dict[str, Any]:
    python_record = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable": public_path(Path(sys.executable)),
        "resolved_executable": public_path(Path(sys.executable).resolve()),
        "executable_sha256": sha256_file(Path(sys.executable)),
        "version_output": subprocess.run(
            [sys.executable, "--version"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
    }
    python_record["identity_sha256"] = identity_sha256(python_record)
    record = {
        "python": {
            **python_record,
        },
        "packages": {
            "torch": package_identity("torch", "torch", native_modules=("torch._C",)),
            "transformers": package_identity("transformers", "transformers"),
            "safetensors": package_identity(
                "safetensors",
                "safetensors",
                native_modules=("safetensors._safetensors_rust",),
            ),
            "tokenizers": package_identity(
                "tokenizers",
                "tokenizers",
                native_modules=("tokenizers.tokenizers",),
            ),
        },
        "iverilog": command_identity("iverilog", ["-V"]),
        "vvp": command_identity("vvp", ["-V"]),
    }
    return {**record, "identity_sha256": identity_sha256(record)}


def tokenizer_record(
    prompt: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    snapshot: Path,
) -> tuple[dict[str, Any], Any]:
    required = ("tokenizer.json", "tokenizer_config.json")
    tokenizer_snapshot = resolve_tokenizer_snapshot()
    source_files = {
        name: file_record(tokenizer_snapshot / name)
        for name in required
    }
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    canonical_hello_ids = canonical_chat_token_ids(tokenizer, "Hello")
    require(
        canonical_hello_ids == list(CANONICAL_HELLO_CHAT_TOKEN_IDS),
        "canonical Qwen Hello chat-template token sequence differs",
    )
    token_ids = canonical_chat_token_ids(tokenizer, prompt)
    bounds = validate_generation_bounds(len(token_ids), max_prompt_tokens, max_new_tokens)
    require(
        all(0 <= token < len(tokenizer) for token in token_ids),
        "canonical tokenizer produced an out-of-range token id",
    )
    behavior_records: list[dict[str, Any]] = []
    for label, probe in TOKENIZER_BEHAVIOR_PROBES:
        probe_ids = canonical_chat_token_ids(tokenizer, probe)
        decoded = tokenizer.decode(
            probe_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        behavior_records.append(
            {
                "label": label,
                "input_utf8_sha256": sha256_bytes(probe.encode("utf-8")),
                "input_utf8_bytes": len(probe.encode("utf-8")),
                "token_ids": probe_ids,
                "token_ids_sha256": identity_sha256(probe_ids),
                "decoded_utf8_sha256": sha256_bytes(decoded.encode("utf-8")),
                "decoded_utf8_bytes": len(decoded.encode("utf-8")),
            }
        )
    behavior = {
        "mode": "apply_chat_template(user_message, tokenize=True, add_generation_prompt=True)",
        "decode_mode": (
            "decode(skip_special_tokens=False, clean_up_tokenization_spaces=False)"
        ),
        "probe_count": len(behavior_records),
        "records": behavior_records,
    }
    behavior["sha256"] = identity_sha256(behavior_records)
    domain = {
        "input_encoding": "strict UTF-8",
        "accepted_text": "non-empty after Unicode whitespace stripping",
        "normalization_by_wrapper": False,
        "chat_template_applied": True,
        "generation_prompt_added": True,
        "default_system_message_applied": True,
        "truncation": False,
        "accepted_prompt_token_count": [1, max_prompt_tokens],
        "accepted_prompt_token_ids": [0, len(tokenizer) - 1],
        "feedback_model_output_domain": [0, MODEL_OUTPUT_DOMAIN - 1],
        "feedback_tokenizer_domain": [0, len(tokenizer) - 1],
        "feedback_rejection": (
            "fail closed for padded/unmapped rows, empty/whitespace-only decode, "
            "or unreadable decoded text"
        ),
    }
    domain["sha256"] = identity_sha256(domain)
    record = {
        "repository": TOKENIZER_REPOSITORY,
        "revision": TOKENIZER_REVISION,
        "snapshot": public_path(tokenizer_snapshot),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": REVISION,
        "model_snapshot": public_path(snapshot),
        "class": tokenizer.__class__.__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "tokenizer_length": len(tokenizer),
        "model_output_domain": MODEL_OUTPUT_DOMAIN,
        "padded_unmapped_model_rows": [len(tokenizer), MODEL_OUTPUT_DOMAIN - 1],
        "source_files": source_files,
        "source_files_sha256": identity_sha256(source_files),
        "mode": "apply_chat_template(user_message, tokenize=True, add_generation_prompt=True)",
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "canonical_hello_chat_token_ids": canonical_hello_ids,
        "canonical_hello_chat_token_count": len(canonical_hello_ids),
        "canonical_hello_chat_token_ids_sha256": identity_sha256(canonical_hello_ids),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "max_prompt_tokens": max_prompt_tokens,
        "generation_bounds": bounds,
        "rtl_argmax_policy": (
            "execute all 151936 model output rows without masking; fail closed rather "
            "than feed back an argmax outside tokenizer length or an empty/unreadable decode"
        ),
        "prompt_text_persisted": False,
        "behavior": behavior,
        "domain": domain,
    }
    return {**record, "identity_sha256": identity_sha256(record)}, tokenizer


def validate_checkpoint_bindings(snapshot: Path) -> dict[str, Any]:
    contract_path = ATTEMPT_LM_HEAD / "frozen_contract.json"
    contract = load_json(contract_path)
    bindings = contract.get("model_bindings")
    require(isinstance(bindings, dict), "LM-head frozen contract lacks model bindings")
    require(
        bindings.get("checkpoint_tree_sha256") == EXPECTED_CHECKPOINT_TREE_SHA256,
        "checkpoint-176 tree binding changed",
    )
    adapter = bindings.get("adapter")
    require(isinstance(bindings.get("model"), dict) and isinstance(adapter, dict), "model binding is incomplete")
    model_path = snapshot / "model.safetensors"
    adapter_path = ROOT / str(adapter["path"])
    adapter_config = bindings.get("adapter_config")
    checkpoint_identity = bindings.get("checkpoint_identity")
    require(
        isinstance(adapter_config, dict) and isinstance(checkpoint_identity, dict),
        "adapter/checkpoint identity binding is incomplete",
    )
    adapter_config_path = ROOT / str(adapter_config["path"])
    checkpoint_identity_path = ROOT / str(checkpoint_identity["path"])
    adapter_tree = verify_directory_identity(checkpoint_identity_path)
    require(
        adapter_tree["tree_sha256"] == EXPECTED_CHECKPOINT_TREE_SHA256,
        "live adapter checkpoint tree binding changed",
    )
    snapshot_tree = directory_tree_record(snapshot)
    record = {
        "contract": file_record(contract_path),
        "checkpoint_tree_sha256": EXPECTED_CHECKPOINT_TREE_SHA256,
        "model": file_record(model_path, expected_sha256=EXPECTED_MODEL_SHA256),
        "model_config": file_record(snapshot / "config.json"),
        "parent_model_tree": snapshot_tree,
        "adapter": file_record(adapter_path, expected_sha256=EXPECTED_ADAPTER_SHA256),
        "adapter_config": file_record(
            adapter_config_path, expected_sha256=str(adapter_config["sha256"])
        ),
        "checkpoint_identity": file_record(
            checkpoint_identity_path,
            expected_sha256=str(checkpoint_identity["sha256"]),
        ),
        "adapter_checkpoint_tree": adapter_tree,
        "final_rmsnorm": bindings["final_rmsnorm_source"],
        "tied_lm_head": bindings["lm_head_source"],
    }
    return {**record, "identity_sha256": identity_sha256(record)}


def source_records(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    return [file_record(path) for path in paths]


def records_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = sorted(records, key=lambda item: str(item["path"]))
    paths = [str(item["path"]) for item in normalized]
    require(len(paths) == len(set(paths)), "aggregate source paths are not unique")
    return {
        "sha256": sha256_bytes(canonical_bytes(normalized)),
        "member_count": len(normalized),
        "total_bytes": sum(int(item["bytes"]) for item in normalized),
        "records": normalized,
    }


def resolve_local_python_module(module_name: str) -> list[Path]:
    if not module_name:
        return []
    relative = Path(*module_name.split("."))
    candidates = [
        ROOT / relative.with_suffix(".py"),
        ROOT / relative / "__init__.py",
    ]
    if relative.parts[0] != "tools":
        candidates += [
            ROOT / "tools" / relative.with_suffix(".py"),
            ROOT / "tools" / relative / "__init__.py",
        ]
    return [path.resolve() for path in candidates if path.is_file()]


def python_runtime_source_paths(
    entrypoints: tuple[Path, ...] = PYTHON_SOURCE_ENTRYPOINTS,
) -> tuple[Path, ...]:
    pending = [path.resolve() for path in entrypoints]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        require(path.is_file(), f"runtime Python source is missing: {public_path(path)}")
        discovered.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise PreflightError(
                f"cannot parse runtime Python source {public_path(path)}: {error}"
            ) from error
        imported: set[Path] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.update(resolve_local_python_module(alias.name))
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                imported.update(resolve_local_python_module(module))
                for alias in node.names:
                    if alias.name != "*":
                        imported.update(
                            resolve_local_python_module(
                                f"{module}.{alias.name}" if module else alias.name
                            )
                        )
        pending.extend(sorted(imported - discovered))
    return tuple(sorted(discovered))


def rtl_runtime_source_paths() -> tuple[Path, ...]:
    paths = {
        path.resolve()
        for path in (ROOT / "rtl").rglob("*")
        if path.is_file() and path.suffix.lower() in RTL_SOURCE_SUFFIXES
    }
    paths.update(
        path.resolve()
        for path in (
            backend.layer_runner.PROJECTION_TB,
            backend.layer_runner.RMS_TB,
            backend.layer_runner.ROPE_TB,
            backend.layer_runner.SCORE_TB,
            backend.layer_runner.SOFTMAX_TB,
            backend.layer_runner.COMPOSE_TB,
            backend.layer_runner.SILU_TB,
            backend.layer_runner.RESIDUAL_TB,
            backend.lm_head.RMS_TB,
        )
    )
    for path in paths:
        require(path.is_file(), f"runtime RTL/testbench source is missing: {public_path(path)}")
    return tuple(sorted(paths))


def runtime_source_aggregate() -> dict[str, Any]:
    python = source_records(python_runtime_source_paths())
    rtl = source_records(rtl_runtime_source_paths())
    aggregate = records_aggregate(python + rtl)
    return {
        "sha256": aggregate["sha256"],
        "member_count": aggregate["member_count"],
        "total_bytes": aggregate["total_bytes"],
        "python_member_count": len(python),
        "rtl_testbench_member_count": len(rtl),
        "python": python,
        "rtl_testbench": rtl,
        "records": aggregate["records"],
    }


def require_aggregate_match(expected: str, observed: str, label: str) -> None:
    require(expected == observed, f"{label} aggregate differs: {observed}")


def verify_predecessor_preflight() -> dict[str, Any]:
    preflight_path = PREFLIGHT_0004 / "preflight.json"
    sums_path = PREFLIGHT_0004 / "SHA256SUMS"
    preflight_record = file_record(
        preflight_path, expected_sha256=EXPECTED_PREFLIGHT_0004_SHA256
    )
    sums_record = file_record(
        sums_path, expected_sha256=EXPECTED_PREFLIGHT_0004_SUMS_SHA256
    )
    verified = verify_preflight_members(PREFLIGHT_0004)
    predecessor = load_json(preflight_path)
    runtime = predecessor.get("runtime_source_aggregate")
    protected = predecessor.get("protected_files_aggregate")
    require(isinstance(runtime, dict), "preflight-0004 runtime aggregate is missing")
    require(isinstance(protected, dict), "preflight-0004 protected aggregate is missing")
    require_aggregate_match(
        EXPECTED_PREFLIGHT_0004_RUNTIME_SHA256,
        str(runtime.get("sha256")),
        "preflight-0004 runtime source",
    )
    require_aggregate_match(
        EXPECTED_PREFLIGHT_0004_PROTECTED_SHA256,
        str(protected.get("sha256")),
        "preflight-0004 protected file",
    )
    writable = [
        public_path(path)
        for path in (PREFLIGHT_0004, *PREFLIGHT_0004.rglob("*"))
        if path.stat().st_mode & 0o222
    ]
    require(not writable, f"preflight-0004 is writable: {writable[0] if writable else ''}")
    record = {
        "path": public_path(PREFLIGHT_0004),
        "preflight": preflight_record,
        "sha256s": sums_record,
        "runtime_source_aggregate_sha256": EXPECTED_PREFLIGHT_0004_RUNTIME_SHA256,
        "protected_files_aggregate_sha256": EXPECTED_PREFLIGHT_0004_PROTECTED_SHA256,
        "members_verified": verified["members_verified"],
        "member_bytes_verified": verified["member_bytes_verified"],
        "immutable": True,
        "mutation": False,
        "replay": False,
    }
    return {**record, "identity_sha256": identity_sha256(record)}


def collect_authority_bindings(request: dict[str, Any]) -> dict[str, Any]:
    checkpoint = validate_checkpoint_bindings(request["snapshot"])
    tools = toolchain_identity()
    sources = runtime_source_aggregate()
    protected = records_aggregate(source_records(PROTECTED_FILES))
    predecessor = verify_predecessor_preflight()
    tokenization = request["tokenization"]
    record = {
        "preflight_0004": predecessor,
        "checkpoint_176": checkpoint,
        "tokenizer": tokenization,
        "toolchain": tools,
        "runtime_source_aggregate": sources,
        "protected_files_aggregate": protected,
    }
    return {**record, "identity_sha256": identity_sha256(record)}


def fresh_l2_authorization_contract(
    *,
    args: argparse.Namespace,
    output: Path,
    official_output: Path,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = bindings["checkpoint_176"]
    tokenizer = bindings["tokenizer"]
    toolchain = bindings["toolchain"]
    predecessor = bindings["preflight_0004"]
    return {
        "schema_version": 3,
        "status": "AUTHORIZED_FRESH_L2_OFFICIAL_ATTEMPT",
        "mission_id": MISSION_ID,
        "preflight_path": public_path(output),
        "preflight_sha256s_sha256": "<sha256 of sealed preflight SHA256SUMS>",
        "official_output": public_path(official_output),
        "prompt_sha256": tokenizer["prompt_sha256"],
        "prompt_token_ids_sha256": identity_sha256(tokenizer["prompt_token_ids"]),
        "max_new_tokens": args.max_new_tokens,
        "preflight_0004_preflight_sha256": predecessor["preflight"]["sha256"],
        "preflight_0004_sha256s_sha256": predecessor["sha256s"]["sha256"],
        "preflight_0004_identity_sha256": predecessor["identity_sha256"],
        "parent_model_file_sha256": checkpoint["model"]["sha256"],
        "parent_model_tree_sha256": checkpoint["parent_model_tree"]["sha256"],
        "adapter_file_sha256": checkpoint["adapter"]["sha256"],
        "adapter_tree_sha256": checkpoint["adapter_checkpoint_tree"]["tree_sha256"],
        "checkpoint_identity_file_sha256": checkpoint["checkpoint_identity"]["sha256"],
        "checkpoint_binding_identity_sha256": checkpoint["identity_sha256"],
        "tokenizer_files_sha256": tokenizer["source_files_sha256"],
        "tokenizer_behavior_sha256": tokenizer["behavior"]["sha256"],
        "tokenizer_domain_sha256": tokenizer["domain"]["sha256"],
        "tokenizer_identity_sha256": tokenizer["identity_sha256"],
        "toolchain_identity_sha256": toolchain["identity_sha256"],
        "python_identity_sha256": toolchain["python"]["identity_sha256"],
        "torch_identity_sha256": toolchain["packages"]["torch"]["identity_sha256"],
        "transformers_identity_sha256": toolchain["packages"]["transformers"]["identity_sha256"],
        "safetensors_identity_sha256": toolchain["packages"]["safetensors"]["identity_sha256"],
        "iverilog_identity_sha256": toolchain["iverilog"]["identity_sha256"],
        "vvp_identity_sha256": toolchain["vvp"]["identity_sha256"],
        "runtime_source_aggregate_sha256": bindings["runtime_source_aggregate"]["sha256"],
        "protected_files_aggregate_sha256": bindings["protected_files_aggregate"]["sha256"],
        "authority_bindings_identity_sha256": bindings["identity_sha256"],
    }


def seal_directory(output: Path) -> None:
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    output.chmod(0o555)


def write_sha256_manifest(output: Path) -> Path:
    sums_path = output / "SHA256SUMS"
    members = sorted(
        path for path in output.rglob("*") if path.is_file() and path != sums_path
    )
    sums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in members
        ),
        encoding="ascii",
    )
    return sums_path


def assert_no_machine_private_paths(output: Path) -> dict[str, Any]:
    private_prefixes = tuple(
        dict.fromkeys((str(Path.home().resolve()), str(ROOT.resolve())))
    )
    scanned_files = 0
    scanned_bytes = 0
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        scanned_files += 1
        scanned_bytes += len(text.encode("utf-8"))
        for prefix in private_prefixes:
            require(
                prefix not in text,
                f"generated artifact embeds a machine-private absolute path: {public_path(path)}",
            )
    return {
        "status": "PASS_NO_MACHINE_PRIVATE_ABSOLUTE_PATHS",
        "text_artifact_count": scanned_files,
        "text_artifact_bytes": scanned_bytes,
        "machine_private_path_matches": 0,
        "repository_artifact_paths": "root_relative",
        "external_process_executables": "basename_only",
    }


def persist_path_hygiene_evidence(output: Path) -> dict[str, Any]:
    scan = assert_no_machine_private_paths(output)
    manifest_path = output / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["path_hygiene"] = scan
    manifest_path.write_bytes(canonical_bytes(manifest))
    assert_no_machine_private_paths(output)
    return scan


def write_preflight(output: Path, record: dict[str, Any]) -> None:
    preflight_path = output / "preflight.json"
    preflight_path.write_bytes(canonical_bytes(record))
    assert_no_machine_private_paths(output)
    write_sha256_manifest(output)
    seal_directory(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight the bounded canonical checkpoint-176 W4A8 arbitrary-text "
            "RTL generation command and run it only with Fresh-L2 authority"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="arbitrary non-empty UTF-8 prompt text")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="canonical revision snapshot; otherwise resolve from ACE2_QWEN25_SNAPSHOT/HF cache",
    )
    parser.add_argument("--output", type=Path, required=True, help="fresh official attempt path")
    parser.add_argument(
        "--preflight-output",
        type=Path,
        help="fresh immutable preflight path; required with --preflight-only",
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        help="Fresh-L2 authorization JSON required for an official execution",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def prepare_request(args: argparse.Namespace) -> dict[str, Any]:
    prompt = read_prompt(args.prompt, args.prompt_file)
    snapshot = resolve_snapshot(args.snapshot)
    tokenization, tokenizer = tokenizer_record(
        prompt,
        args.max_prompt_tokens,
        args.max_new_tokens,
        snapshot,
    )
    return {
        "prompt": prompt,
        "snapshot": snapshot,
        "tokenization": tokenization,
        "tokenizer": tokenizer,
    }


def command_record(
    args: argparse.Namespace,
    tokenization: dict[str, Any],
    *,
    preflight: bool,
) -> dict[str, Any]:
    argv = [
        "python3",
        "tools/run_rtl_arbitrary_text_generation.py",
    ]
    if preflight:
        argv.append("--preflight-only")
    argv += [
        "--prompt",
        f"<redacted sha256={tokenization['prompt_sha256']}>",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--output",
        public_path(args.output.resolve()),
    ]
    if args.snapshot is not None:
        argv += ["--snapshot", public_path(args.snapshot.resolve())]
    if preflight:
        argv += ["--preflight-output", public_path(args.preflight_output.resolve())]
    else:
        argv += ["--authorization", public_path(args.authorization.resolve())]
    return {"argv": argv, "cwd": "."}


def run_preflight(args: argparse.Namespace, request: dict[str, Any]) -> int:
    started = time.perf_counter()
    require(args.preflight_output is not None, "--preflight-output is required")
    require(args.authorization is None, "--authorization is not legal with --preflight-only")
    output = args.preflight_output.resolve()
    official_output = args.output.resolve()
    require(output != official_output, "preflight and official output paths must differ")
    require(not output.exists(), f"preflight output already exists: {public_path(output)}")
    require(
        not official_output.exists(),
        f"official output already exists: {public_path(official_output)}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    try:
        with backend.process_tree_rss_tracking(interval_seconds=0.01) as rss_tracker:
            tokenization = request["tokenization"]
            snapshot = request["snapshot"]
            bindings = collect_authority_bindings(request)
            checkpoint = bindings["checkpoint_176"]
            tools = bindings["toolchain"]
            protected = bindings["protected_files_aggregate"]
            sources = bindings["runtime_source_aggregate"]
            repair_regressions = backend.run_stage1_repair_regressions(
                output / "stage1-repair-regressions"
            )
            attempt_24l = verify_sealed_attempt(
                ATTEMPT_24L,
                expected_sums_sha256=EXPECTED_24L_SUMS_SHA256,
                status_path=ATTEMPT_24L / "manifest.json",
                expected_status=EXPECTED_24L_STATUS,
            )
            manifest_24l = load_json(ATTEMPT_24L / "manifest.json")
            require(manifest_24l.get("immutable_after_check") is True, "24-layer attempt is not immutable")
            require(manifest_24l.get("official_attempt_consumed") is True, "24-layer attempt is not sealed")
            attempt_lm_head = verify_sealed_attempt(
                ATTEMPT_LM_HEAD,
                expected_sums_sha256=EXPECTED_LM_HEAD_SUMS_SHA256,
                status_path=ATTEMPT_LM_HEAD / "status.json",
                expected_status=EXPECTED_LM_HEAD_STATUS,
            )
            lm_head_check = load_json(ATTEMPT_LM_HEAD / "check.json")
            require(
                lm_head_check.get("status") == EXPECTED_LM_HEAD_CHECK_STATUS,
                "LM-head exact check status changed",
            )
            require(int(lm_head_check.get("output_count", 0)) == 151936, "LM-head is not full vocabulary")
            smoke = backend.run_preflight_smoke(
                output / "backend-smoke",
                snapshot,
                request["tokenizer"],
                tokenization["prompt_token_ids"][0],
                lm_head_outputs=32,
            )
        process_tree_rss = rss_tracker.summary(require_complete=True)
        elapsed = time.perf_counter() - started
        authorization_contract = fresh_l2_authorization_contract(
            args=args,
            output=output,
            official_output=official_output,
            bindings=bindings,
        )
        record = {
            "schema_version": 3,
            "generated_at_utc": utc_now(),
            "status": "PASS_REAL_ICARUS_BACKEND_PREFLIGHT_AWAITING_FRESH_L2_AUTHORIZATION",
            "classification": "bounded_immutable_engineer_preflight",
            "mission_id": MISSION_ID,
            "command": command_record(args, tokenization, preflight=True),
            "frozen_generation_contract": {
                "layers": 24,
                "hidden": 896,
                "kv_heads": 2,
                "head_dim": 64,
                "model_output_domain": MODEL_OUTPUT_DOMAIN,
                "tokenizer_base_vocab_size": tokenization["vocab_size"],
                "tokenizer_length": tokenization["tokenizer_length"],
                "max_new_tokens": args.max_new_tokens,
                "prompt_token_bound": [1, args.max_prompt_tokens],
                "total_context_tokens": tokenization["generation_bounds"]["total_context_tokens"],
                "processed_position_bound": tokenization["generation_bounds"]["processed_positions"],
                "accepted_rtl_context_bound": MAX_CONTEXT_TOKENS,
                "selection": (
                    "unmasked greedy argmax over all 151936 signed-int8 RTL logits, "
                    "lowest-token-id tie break; fail closed on unmapped/empty/unreadable token"
                ),
                "prefill": "stream every prompt token through all 24 RTL layers without intermediate LM-head",
                "per_generated_step": [
                    "24-layer RTL decode",
                    "distinct per-layer K/V append",
                    "final RMSNorm RTL",
                    "4748 tied LM-head RTL tiles covering 151936 tokens",
                    "selected token feedback into next RTL decode position",
                ],
                "public_module_port_changes": False,
            },
            "tokenization": tokenization,
            "checkpoint_176": checkpoint,
            "preflight_0004": bindings["preflight_0004"],
            "implementation_smoke": {
                "summary": file_record(output / "backend-smoke/smoke_summary.json"),
                "status": smoke["status"],
                "real_icarus_prefill": True,
                "real_icarus_selected_logit_feedback": True,
                "real_icarus_decode": True,
                "cache_growth": smoke["cache_growth"],
                "partial_lm_head_outputs": smoke["first_head"]["output_count"],
                "full_vocabulary_claim": False,
            },
            "stage1_repair_regressions": {
                "summary": file_record(output / "stage1-repair-regressions/summary.json"),
                "status": repair_regressions["status"],
                "softmax_context_executed": repair_regressions[
                    "softmax_context_executed"
                ],
                "retained_residual_observation_count": repair_regressions[
                    "retained_residual_observation_count"
                ],
                "retained_residual_xz_count": repair_regressions[
                    "retained_residual_xz_count"
                ],
                "relevant_out_of_bounds_warning_count": repair_regressions[
                    "relevant_out_of_bounds_warning_count"
                ],
            },
            "accepted_immutable_bases": {
                "transformer_24layer": attempt_24l,
                "final_rmsnorm_full_vocab_lm_head": attempt_lm_head,
                "lm_head_check": file_record(ATTEMPT_LM_HEAD / "check.json"),
            },
            "toolchain": tools,
            "authority_bindings": bindings,
            "runtime_source_aggregate": sources,
            "source_bindings": sources["records"],
            "protected_files_aggregate": protected,
            "protected_files": protected["records"],
            "official_attempt": {
                "output": public_path(official_output),
                "output_absent": True,
                "process_start_count": 0,
                "authorized": False,
                "authorization_status": "AWAITING_FRESH_L2",
                "reason": "implementation preflight passed; Fresh-L2 must issue a separate hash-bound authority before process start",
            },
            "fresh_l2_authorization_contract": authorization_contract,
            "measurements": {
                "wall_seconds": elapsed,
                "peak_rss_kib": process_tree_rss["peak_rss_kib"],
                "process_tree_rss": process_tree_rss,
                "scope": "model/adapter/tokenizer/tool hashing plus context-40 softmax, retained residual warning/X regression, and real layer-0 Icarus prefill, partial-logit feedback, and decode smoke; RSS covers the Python root and every descendant",
                "rtl_executed": True,
                "smoke_icarus_wall_seconds": (
                    smoke["prefill"]["icarus_wall_seconds"]
                    + smoke["first_head"]["icarus_wall_seconds"]
                    + smoke["decode"]["icarus_wall_seconds"]
                    + smoke["second_head"]["icarus_wall_seconds"]
                ),
                "official_attempt_consumed": False,
            },
            "prohibited_activity": {
                "sealed_attempts_mutated": False,
                "sealed_attempts_replayed": False,
                "u280_or_vendor_tools": False,
                "software_logits_fallback": False,
                "protected_files_modified": False,
            },
        }
    except Exception as error:
        elapsed = time.perf_counter() - started
        process_tree_rss = (
            rss_tracker.summary(require_complete=False)
            if "rss_tracker" in locals()
            else None
        )
        failure_classification = getattr(
            error, "classification", "AUTHORITY_OR_PREFLIGHT_INTERLOCK_FAILURE"
        )
        record = {
            "schema_version": 3,
            "generated_at_utc": utc_now(),
            "status": "FAIL_PREFLIGHT",
            "failure_taxonomy": failure_classification,
            "root_cause_hypothesis": str(error),
            "regression": "verification/test_rtl_arbitrary_text_generation.py",
            "error_type": type(error).__name__,
            "error": str(error),
            "official_attempt": {
                "output": public_path(official_output),
                "process_start_count": 0,
                "authorized": False,
            },
            "measurements": {
                "wall_seconds": elapsed,
                "peak_rss_kib": (
                    process_tree_rss["peak_rss_kib"]
                    if process_tree_rss is not None
                    else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                ),
                "process_tree_rss": process_tree_rss,
                "rtl_executed": bool(getattr(error, "rtl_executed", False)),
                "official_attempt_consumed": False,
            },
        }
        write_preflight(output, record)
        raise

    write_preflight(output, record)
    print(
        "ACE2_RTL_GENERATION_PREFLIGHT_PASS "
        f"output={public_path(output)} prompt_tokens={tokenization['prompt_token_count']} "
        f"max_new_tokens={args.max_new_tokens} process_start_count=0 "
        "authorization=awaiting_fresh_l2"
    )
    return 0


def resolve_authority_preflight(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_preflight_members(preflight: Path) -> dict[str, Any]:
    sums = preflight / "SHA256SUMS"
    records = parse_sha256_manifest(preflight, sums)
    manifested_paths = {path.resolve() for _expected, path in records}
    actual_paths = {
        path.resolve()
        for path in preflight.rglob("*")
        if path.is_file() and path != sums
    }
    require(manifested_paths == actual_paths, "authorized preflight file set differs from SHA256SUMS")
    verified = []
    for expected, path in records:
        verified.append(file_record(path, expected_sha256=expected))
    require(
        any(record["path"] == public_path(preflight / "preflight.json") for record in verified),
        "authorized preflight aggregate omits preflight.json",
    )
    return {
        "path": public_path(preflight),
        "sha256s": file_record(sums),
        "members_verified": len(verified),
        "member_bytes_verified": sum(int(record["bytes"]) for record in verified),
    }


def validate_authorization(
    args: argparse.Namespace,
    request: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if args.authorization is None:
        raise OfficialAuthorizationError("--authorization is required for official execution")
    authority_path = args.authorization.resolve()
    authority = load_json(authority_path)
    minimum_required = {
        "schema_version",
        "status",
        "mission_id",
        "preflight_path",
        "preflight_sha256s_sha256",
        "official_output",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "max_new_tokens",
    }
    require(
        minimum_required <= set(authority),
        "Fresh-L2 authorization lacks required preflight selection keys",
    )
    require(authority["schema_version"] == 3, "Fresh-L2 authorization schema differs")
    require(authority["status"] == "AUTHORIZED_FRESH_L2_OFFICIAL_ATTEMPT", "Fresh-L2 status is not authorized")
    require(authority["mission_id"] == MISSION_ID, "Fresh-L2 mission binding differs")
    preflight = resolve_authority_preflight(str(authority["preflight_path"]))
    sums = preflight / "SHA256SUMS"
    require(sums.is_file(), "authorized preflight SHA256SUMS is absent")
    require(sha256_file(sums) == authority["preflight_sha256s_sha256"], "authorized preflight aggregate differs")
    preflight_verification = verify_preflight_members(preflight)
    preflight_record = load_json(preflight / "preflight.json")
    require(
        preflight_record.get("status")
        == "PASS_REAL_ICARUS_BACKEND_PREFLIGHT_AWAITING_FRESH_L2_AUTHORIZATION",
        "authorized preflight is not passing",
    )
    require(preflight_record.get("schema_version") == 3, "authorized preflight schema differs")
    frozen_contract = preflight_record.get("fresh_l2_authorization_contract")
    require(isinstance(frozen_contract, dict), "authorized preflight lacks Fresh-L2 contract")
    require(
        set(authority) == set(frozen_contract),
        "Fresh-L2 authorization keys differ from frozen contract",
    )
    for key, value in frozen_contract.items():
        if key != "preflight_sha256s_sha256":
            require(authority.get(key) == value, f"Fresh-L2 authority differs from preflight contract: {key}")
    tokenization = request["tokenization"]
    require(authority["official_output"] == public_path(args.output.resolve()), "authorized official output differs")
    require(authority["prompt_sha256"] == tokenization["prompt_sha256"], "authorized prompt digest differs")
    require(
        authority["prompt_token_ids_sha256"]
        == sha256_bytes(canonical_bytes(tokenization["prompt_token_ids"])),
        "authorized prompt token IDs differ",
    )
    require(authority["max_new_tokens"] == args.max_new_tokens, "authorized generation count differs")
    preflight_rss = preflight_record.get("measurements", {}).get("process_tree_rss")
    require(isinstance(preflight_rss, dict), "authorized preflight lacks process-tree RSS")
    require(preflight_rss.get("complete") is True, "authorized preflight RSS accounting is incomplete")
    require(
        preflight_rss.get("all_icarus_children_observed") is True,
        "authorized preflight missed an iverilog/vvp child",
    )
    current_bindings = collect_authority_bindings(request)
    current_contract = fresh_l2_authorization_contract(
        args=args,
        output=preflight,
        official_output=args.output.resolve(),
        bindings=current_bindings,
    )
    for key, value in current_contract.items():
        if key != "preflight_sha256s_sha256":
            require(
                authority.get(key) == value,
                f"current pre-start authority binding differs: {key}",
            )
    return authority, authority_path, {
        "preflight_verification": preflight_verification,
        "authority_bindings": current_bindings,
        "runtime_source_aggregate": current_bindings["runtime_source_aggregate"],
        "protected_files_aggregate": current_bindings["protected_files_aggregate"],
    }


def persist_official_process_tree_evidence(
    official_output: Path,
    process_tree_rss: dict[str, Any],
    *,
    success: bool,
    error: BaseException | None = None,
) -> dict[str, Any]:
    children = process_tree_rss.get("children")
    require(isinstance(children, list), "official process-tree RSS child records are missing")
    require(
        isinstance(process_tree_rss.get("all_icarus_children_observed"), bool),
        "official process-tree RSS Icarus completeness flag is missing",
    )
    fields = {
        "rss_authority": "full_official_backend_process_tree",
        "peak_rss_kib": int(process_tree_rss["peak_rss_kib"]),
        "process_tree_rss": process_tree_rss,
        "process_tree_children": children,
        "all_icarus_children_observed": process_tree_rss[
            "all_icarus_children_observed"
        ],
    }
    run_summary_path = official_output / "run_summary.json"
    if success:
        require(run_summary_path.is_file(), "official backend omitted run_summary.json")
        require(
            process_tree_rss.get("complete") is True,
            "official success cannot use incomplete process-tree RSS",
        )
        require(
            process_tree_rss["all_icarus_children_observed"] is True,
            "official success missed an iverilog/vvp child",
        )
        record = load_json(run_summary_path)
        record.update(fields)
        record["official_result_valid"] = True
        run_summary_path.write_bytes(canonical_bytes(record))
        return record

    failure_path = official_output / "failure.json"
    record = load_json(failure_path) if failure_path.is_file() else {}
    record.update(
        {
            "schema_version": max(1, int(record.get("schema_version", 1))),
            "status": "SEALED_FAIL_OFFICIAL_RTL_GENERATION",
            "official_attempt_consumed": True,
            "official_result_valid": False,
            **fields,
        }
    )
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    failure_path.write_bytes(canonical_bytes(record))

    if run_summary_path.is_file():
        invalidated = load_json(run_summary_path)
        invalidated.update(fields)
        invalidated["status"] = "INVALIDATED_OFFICIAL_RTL_GENERATION_PROCESS_TREE_RSS"
        invalidated["official_result_valid"] = False
        run_summary_path.write_bytes(canonical_bytes(invalidated))
    return record


def persist_official_provenance(
    official_output: Path,
    *,
    args: argparse.Namespace,
    request: dict[str, Any],
    authority_bytes: bytes,
    launch: dict[str, Any],
    status: str,
    process_tree_rss: dict[str, Any],
    summary: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    authorization_path = official_output / "authorization.json"
    authorization_path.write_bytes(authority_bytes)
    require(
        sha256_file(authorization_path) == launch["authorization_source"]["sha256"],
        "preserved authorization bytes differ from pre-start capture",
    )
    launch_path = official_output / "launch_provenance.json"
    launch_path.write_bytes(canonical_bytes(launch))
    manifest = {
        "schema_version": 2,
        "status": status,
        "mission_id": MISSION_ID,
        "command": launch["command"],
        "tokenization": request["tokenization"],
        "authorization": file_record(authorization_path),
        "launch_provenance": file_record(launch_path),
        "runtime_source_aggregate": launch["runtime_source_aggregate"],
        "source_bindings": launch["runtime_source_aggregate"]["records"],
        "protected_files_aggregate": launch["protected_files_aggregate"],
        "protected_files": launch["protected_files_aggregate"]["records"],
        "prestart_authority_bindings": launch.get("authority_bindings"),
        "process_start_count": 1,
        "official_attempt_consumed": True,
        "rss_authority": "full_official_backend_process_tree",
        "peak_rss_kib": int(process_tree_rss["peak_rss_kib"]),
        "process_tree_rss": process_tree_rss,
        "process_tree_children": process_tree_rss["children"],
        "all_icarus_children_observed": process_tree_rss[
            "all_icarus_children_observed"
        ],
        "full_vocabulary_per_step": summary is not None,
        "software_transformer_or_logits_fallback": False,
    }
    if summary is not None:
        manifest.update(
            {
                "run_summary": file_record(official_output / "run_summary.json"),
                "generated_token_ids": summary["generated_token_ids"],
                "decoded_text": summary["decoded_text"],
            }
        )
    if error is not None:
        manifest.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "failure": file_record(official_output / "failure.json")
                if (official_output / "failure.json").is_file()
                else None,
            }
        )
    (official_output / "manifest.json").write_bytes(canonical_bytes(manifest))
    return manifest


def run_official(args: argparse.Namespace, request: dict[str, Any]) -> int:
    require(args.preflight_output is None, "--preflight-output is only legal with --preflight-only")
    official_output = args.output.resolve()
    require(not official_output.exists(), f"official output already exists: {public_path(official_output)}")
    _authority, authority_path, launch_bindings = validate_authorization(args, request)
    authority_bytes = authority_path.read_bytes()
    launch = {
        "schema_version": 1,
        "status": "PRESTART_OFFICIAL_LAUNCH_PROVENANCE",
        "mission_id": MISSION_ID,
        "command": command_record(args, request["tokenization"], preflight=False),
        "tokenization": request["tokenization"],
        "authorization_source": {
            "path": public_path(authority_path),
            "bytes": len(authority_bytes),
            "sha256": sha256_bytes(authority_bytes),
        },
        "runtime_source_aggregate": launch_bindings["runtime_source_aggregate"],
        "protected_files_aggregate": launch_bindings["protected_files_aggregate"],
        "authority_bindings": launch_bindings.get("authority_bindings"),
        "preflight_verification": launch_bindings["preflight_verification"],
        "process_start_count": 1,
    }
    rss_tracker: backend.ProcessTreeRssTracker | None = None
    process_tree_rss: dict[str, Any] | None = None
    try:
        with backend.process_tree_rss_tracking(interval_seconds=0.01) as rss_tracker:
            summary = backend.run_generation(
                official_output,
                request["snapshot"],
                request["tokenizer"],
                request["tokenization"]["prompt_token_ids"],
                args.max_new_tokens,
            )
        process_tree_rss = rss_tracker.summary(require_complete=True)
        summary = persist_official_process_tree_evidence(
            official_output,
            process_tree_rss,
            success=True,
        )
        manifest = persist_official_provenance(
            official_output,
            args=args,
            request=request,
            authority_bytes=authority_bytes,
            launch=launch,
            status="PASS_OFFICIAL_RTL_ARBITRARY_TEXT_GENERATION_UNCHECKED_AWAITING_FRESH_L2",
            process_tree_rss=process_tree_rss,
            summary=summary,
        )
        (official_output / "status.json").write_bytes(
            canonical_bytes({"status": manifest["status"], "sealed": True})
        )
        persist_path_hygiene_evidence(official_output)
        write_sha256_manifest(official_output)
        seal_directory(official_output)
    except BaseException as error:
        if official_output.is_dir():
            if process_tree_rss is None and rss_tracker is not None:
                process_tree_rss = rss_tracker.summary(require_complete=False)
            require(
                process_tree_rss is not None,
                "official failure lacks process-tree RSS evidence",
            )
            persist_official_process_tree_evidence(
                official_output,
                process_tree_rss,
                success=False,
                error=error,
            )
            manifest = persist_official_provenance(
                official_output,
                args=args,
                request=request,
                authority_bytes=authority_bytes,
                launch=launch,
                status="SEALED_FAIL_OFFICIAL_RTL_GENERATION",
                process_tree_rss=process_tree_rss,
                error=error,
            )
            (official_output / "status.json").write_bytes(
                canonical_bytes({"status": manifest["status"], "sealed": True})
            )
            persist_path_hygiene_evidence(official_output)
            write_sha256_manifest(official_output)
            seal_directory(official_output)
        raise
    print(
        "ACE2_RTL_GENERATION_OFFICIAL_UNCHECKED "
        f"output={public_path(official_output)} generated_tokens={len(summary['generated_token_ids'])} "
        "process_start_count=1"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = prepare_request(args)
    if args.preflight_only:
        return run_preflight(args, request)
    return run_official(args, request)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreflightError, OfficialAuthorizationError, backend.BackendError) as error:
        print(f"ACE2_RTL_GENERATION_SETUP_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
