#!/usr/bin/env python3
"""Shared immutable identities for the Manager-authorized Option-B artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    / REVISION
)
SOURCE_HASHES = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
}
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
SOURCE_AUDIT = (
    ROOT
    / "build/ace2_chat_diagnostics/qwen2.5-0.5b-instruct-local-audit-20260806.json"
)
SOURCE_AUDIT_SHA256 = "5ea75b61105ff5ce30803c5c7b33e2bde36acb0511ffddc2da55303909cf657d"
SOURCE_CONTRACT = ROOT / "design/QWEN_INSTRUCT_OPTION_B_SOURCE_CONTRACT.json"
CALIBRATION_ID = "qwen2.5-0.5b-instruct-w4a8-calibration-v1"
CALIBRATION_DIR = ROOT / "evidence/verification" / CALIBRATION_ID
IMAGE_ID = "build-qwen2.5-0.5b-instruct-w4a8-image-v1"
IMAGE_DIR = ROOT / "evidence/verification" / IMAGE_ID
TERMINATION_TOKEN_IDS = (151643, 151645)
EXPECTED_VERSIONS = {
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0",
    "transformers": "4.57.6",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        label = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_source_snapshot() -> dict[str, Any]:
    require(SNAPSHOT.is_dir(), "pinned Instruct snapshot is missing")
    records: dict[str, Any] = {}
    for name, expected in SOURCE_HASHES.items():
        path = SNAPSHOT / name
        require(path.is_file(), f"pinned Instruct source is missing: {name}")
        actual = sha256_file(path)
        require(actual == expected, f"pinned Instruct source differs: {name}")
        records[name] = {
            "path": f"hf-cache://{REPOSITORY}@{REVISION}/{name}",
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    tokenizer_config = load_json(SNAPSHOT / "tokenizer_config.json")
    template = tokenizer_config.get("chat_template")
    require(isinstance(template, str), "pinned Instruct chat template is missing")
    template_sha = sha256_bytes(template.encode())
    require(template_sha == CHAT_TEMPLATE_SHA256, "pinned Instruct chat template differs")
    require(SOURCE_AUDIT.is_file(), "pinned Instruct source audit is missing")
    require(sha256_file(SOURCE_AUDIT) == SOURCE_AUDIT_SHA256, "source audit differs")
    audit = load_json(SOURCE_AUDIT)
    require(audit.get("status") == "PASS_LOCAL_INSTRUCT_ARTIFACT_AVAILABLE", "source audit did not pass")
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": records,
        "chat_template_sha256": template_sha,
        "source_audit": file_record(SOURCE_AUDIT),
    }


def verify_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if package == "torch" and "+" in actual:
            actual = actual.split("+", 1)[0]
        require(actual == expected, f"package version differs: {package}={actual}")
        observed[package] = importlib.metadata.version(package)
    return observed


def verify_source_contract() -> dict[str, Any]:
    require(SOURCE_CONTRACT.is_file(), "Option-B source contract is missing")
    contract = load_json(SOURCE_CONTRACT)
    require(contract.get("contract_id") == "qwen2.5-0.5b-instruct-option-b-source-v1", "Option-B source contract differs")
    require(contract.get("status") == "FROZEN_OPTION_B_SOURCE_CONTRACT", "Option-B source contract is not frozen")
    model = contract.get("source_model", {})
    require(model.get("repository") == REPOSITORY and model.get("revision") == REVISION, "Option-B source model differs")
    require(model.get("files") == SOURCE_HASHES, "Option-B source file identities differ")
    require(model.get("chat_template_sha256") == CHAT_TEMPLATE_SHA256, "Option-B chat-template identity differs")
    for name, record in contract.get("tools", {}).items():
        require(isinstance(record, dict) and set(record) == {"path", "sha256"}, f"Option-B tool record differs: {name}")
        path = ROOT / record["path"]
        require(path.is_file(), f"Option-B tool is missing: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"Option-B tool differs: {record['path']}")
    return contract


def verify_wrapped_contract(path: Path) -> dict[str, Any]:
    wrapper = load_json(path)
    require(set(wrapper) == {"contract", "contract_sha256"}, f"contract wrapper differs: {path}")
    contract = wrapper["contract"]
    require(isinstance(contract, dict), f"contract body differs: {path}")
    digest = sha256_bytes(canonical_bytes(contract))
    require(digest == wrapper["contract_sha256"], f"contract digest differs: {path}")
    return contract
