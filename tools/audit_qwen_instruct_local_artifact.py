#!/usr/bin/env python3
"""Audit an exact Qwen Instruct revision without downloading model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE


REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIRECTORY_NAME = "models--Qwen--Qwen2.5-0.5B-Instruct"
REVISION_SOURCE = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def file_record(path: Path, *, snapshot_root: Path | None = None) -> dict[str, Any]:
    record = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if snapshot_root is not None:
        record["snapshot_relative_path"] = path.relative_to(snapshot_root).as_posix()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.revision
    ):
        raise RuntimeError("--revision must be a full lowercase 40-hex commit")

    cache_root = Path(HF_HUB_CACHE)
    repository_cache = cache_root / CACHE_DIRECTORY_NAME
    snapshot = repository_cache / "snapshots" / args.revision
    main_ref = repository_cache / "refs" / "main"
    observed_main_ref = (
        main_ref.read_text(encoding="utf-8").strip()
        if main_ref.is_file()
        else None
    )
    snapshot_files = (
        sorted(path for path in snapshot.rglob("*") if path.is_file())
        if snapshot.is_dir()
        else []
    )
    weight_files = [
        path
        for path in snapshot_files
        if path.name == "model.safetensors"
        or path.name == "model.safetensors.index.json"
        or (path.name.startswith("model-") and path.suffix == ".safetensors")
    ]
    required_metadata = [
        snapshot / "config.json",
        snapshot / "tokenizer.json",
        snapshot / "tokenizer_config.json",
    ]
    complete = bool(
        snapshot.is_dir()
        and weight_files
        and all(path.is_file() for path in required_metadata)
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "classification": "qwen_instruct_exact_revision_local_artifact_audit",
        "network_access_performed": False,
        "base_model_substitution_performed": False,
        "repository": REPOSITORY,
        "required_revision": args.revision,
        "revision_resolution": {
            "observed_date": "2026-08-05",
            "source": REVISION_SOURCE,
            "source_classification": "official_huggingface_repository_main_tree",
        },
        "cache_root": str(cache_root.resolve()),
        "repository_cache": str(repository_cache.resolve()),
        "repository_cache_exists": repository_cache.is_dir(),
        "main_ref": {
            "path": str(main_ref.resolve()),
            "exists": main_ref.is_file(),
            "observed_revision": observed_main_ref,
            "matches_required_revision": observed_main_ref == args.revision,
        },
        "snapshot": {
            "path": str(snapshot.resolve()),
            "exists": snapshot.is_dir(),
            "files": [
                file_record(path, snapshot_root=snapshot) for path in snapshot_files
            ],
            "weight_files": [
                file_record(path, snapshot_root=snapshot) for path in weight_files
            ],
            "required_metadata": [
                {
                    "path": str(path.resolve()),
                    "exists": path.is_file(),
                }
                for path in required_metadata
            ],
            "complete_for_local_bf16_generation": complete,
        },
        "status": (
            "PASS_LOCAL_INSTRUCT_ARTIFACT_AVAILABLE"
            if complete
            else "BLOCKED_MISSING_LOCAL_INSTRUCT_SNAPSHOT"
        ),
        "precise_prerequisite": (
            None
            if complete
            else {
                "action": (
                    "Provide an already-downloaded Hugging Face snapshot for "
                    f"{REPOSITORY} at exact revision {args.revision}; do not provide "
                    "Qwen/Qwen2.5-0.5B Base as a substitute."
                ),
                "required_snapshot_path": str(snapshot.resolve()),
                "required_contents": [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors or model.safetensors.index.json plus every shard",
                ],
            }
        ),
    }
    output = args.output.resolve()
    write_atomic(output, canonical_bytes(result))
    print(
        "QWEN_INSTRUCT_LOCAL_AUDIT "
        f"status={result['status']} "
        f"revision={args.revision} "
        f"snapshot={snapshot} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"QWEN_INSTRUCT_LOCAL_AUDIT_FAIL detail={error}", file=sys.stderr)
        raise SystemExit(3)
