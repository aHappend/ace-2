#!/home/argustest/miniconda3/bin/python3.13
"""Shared read-only path/hash/mode-bound V21 CORE_MANIFEST preflight."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


class CoreManifestPreflightError(RuntimeError):
    def __init__(self, message: str, stage: str) -> None:
        super().__init__(message)
        self.stage = stage


def require(condition: bool, message: str, stage: str) -> None:
    if not condition:
        raise CoreManifestPreflightError(message, stage)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0))
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def verify_core_manifest_closure(closure: dict[str, Any], *, expected_core_manifest_path: Path) -> dict[str, Any]:
    required = {
        "core_manifest_mode", "core_manifest_path", "core_manifest_semantic_sha256", "core_manifest_sha256",
        "core_manifest_self_sha256", "deferred_hash_paths", "directory_count", "entries", "entries_sha256",
        "file_count", "full_hash_file_count", "official_evaluator_sha256", "official_parser_sha256",
        "package_key", "path_hash_mode_complete", "schema_version",
    }
    require(type(closure) is dict and set(closure) == required, "closure keys", "CLOSURE_MALFORMED")
    require(closure["schema_version"] == 1 and closure["package_key"] == "core_manifest_closure", "closure identity", "CLOSURE_MALFORMED")
    require(closure["path_hash_mode_complete"] is True, "closure completeness claim", "CLOSURE_MALFORMED")
    entries = closure["entries"]
    require(type(entries) is list and entries == sorted(entries, key=lambda item: item["path"]), "closure entry order", "CLOSURE_MALFORMED")
    require(len({entry["path"] for entry in entries}) == len(entries), "closure duplicate path", "CLOSURE_MALFORMED")
    require(sha256_bytes(compact_bytes(entries)) == closure["entries_sha256"], "closure digest", "CLOSURE_INCONSISTENT")
    require(closure["core_manifest_path"] == str(expected_core_manifest_path), "bound core manifest path", "CORE_MANIFEST_PATH")
    files = directories = full_hash_files = 0
    deferred: list[str] = []
    core_value: dict[str, Any] | None = None
    for entry in entries:
        require(type(entry) is dict and set(entry) == {"canonical_json", "hash_policy", "kind", "mode", "path", "semantic_sha256", "sha256", "size", "sources"}, "entry keys", "CLOSURE_MALFORMED")
        path = Path(entry["path"])
        require(os.path.lexists(path), f"path absent: {path}", "PATH_ABSENT")
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"symlink: {path}", "PATH_KIND")
        require(f"{stat.S_IMODE(info.st_mode):04o}" == entry["mode"], f"mode: {path}", "PATH_MODE")
        if entry["kind"] == "directory":
            directories += 1
            require(stat.S_ISDIR(info.st_mode), f"directory kind: {path}", "PATH_KIND")
            require(entry["sha256"] is None and entry["size"] is None and entry["hash_policy"] == "NOT_APPLICABLE", f"directory fields: {path}", "CLOSURE_MALFORMED")
            continue
        files += 1
        require(entry["kind"] == "file" and stat.S_ISREG(info.st_mode), f"file kind: {path}", "PATH_KIND")
        require(info.st_size == entry["size"], f"size: {path}", "PATH_SIZE")
        if entry["hash_policy"] == "FULL_READ_ONLY_PREFLIGHT":
            full_hash_files += 1
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == entry["sha256"], f"hash: {path}", "PATH_HASH")
            if entry["canonical_json"]:
                try:
                    value = json.loads(raw.decode("ascii", "strict"))
                except Exception as error:
                    raise CoreManifestPreflightError(f"JSON: {path}: {type(error).__name__}", "JSON_MALFORMED") from error
                require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}", "JSON_MALFORMED")
                require(sha256_bytes(compact_bytes(value)) == entry["semantic_sha256"], f"semantic JSON: {path}", "MANIFEST_INCONSISTENT")
                if path == expected_core_manifest_path:
                    core_value = value
        else:
            require(entry["hash_policy"] == "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN", f"hash policy: {path}", "CLOSURE_MALFORMED")
            deferred.append(str(path))
    require(files == closure["file_count"] and directories == closure["directory_count"], "closure cardinality", "CLOSURE_INCONSISTENT")
    require(full_hash_files == closure["full_hash_file_count"], "full hash cardinality", "CLOSURE_INCONSISTENT")
    require(sorted(deferred) == closure["deferred_hash_paths"], "deferred paths", "CLOSURE_INCONSISTENT")
    require(core_value is not None, "core manifest not verified", "CORE_MANIFEST_PATH")
    require(closure["core_manifest_sha256"] == sha256_file(expected_core_manifest_path), "core manifest closure hash", "PATH_HASH")
    require(closure["core_manifest_mode"] == f"{stat.S_IMODE(os.lstat(expected_core_manifest_path).st_mode):04o}", "core manifest closure mode", "PATH_MODE")
    require(sha256_bytes(compact_bytes(core_value)) == closure["core_manifest_semantic_sha256"], "core manifest semantic binding", "MANIFEST_INCONSISTENT")
    require(core_value.get("package_content_sha256") == closure["core_manifest_self_sha256"], "core manifest self binding", "MANIFEST_INCONSISTENT")
    return {
        "deferred_hash_file_count": len(deferred),
        "directory_count": directories,
        "file_count": files,
        "full_hash_file_count": full_hash_files,
        "status": "PASS_COMPLETE_BOUND_CORE_MANIFEST_CLOSURE",
    }
