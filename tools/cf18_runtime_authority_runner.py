#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
CANDIDATE = PACKAGE / "authority-candidate.json"
AUTHORITY = PACKAGE.parent / f"{PACKAGE.name}-execution-authority.json"


class AuthorityError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuthorityError(f"JSON root is not an object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_json(path: Path, value: Any) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        raw = canonical_bytes(value)
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def mkdir_exclusive(path: Path) -> None:
    path.mkdir(mode=0o700)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or path.is_symlink():
        raise AuthorityError(f"created namespace is not a directory: {path}")
    fsync_directory(path.parent)


def validate_external_authority(path: Path = AUTHORITY) -> dict[str, Any]:
    if path != AUTHORITY or not path.is_file() or path.is_symlink():
        raise AuthorityError("separate CF18 execution authority is absent")
    candidate = load_object(CANDIDATE)
    authority = load_object(path)
    if (
        authority.get("schema") != "ace2-cf18-separate-execution-authority-v1"
        or authority.get("status") != "GRANTED_AFTER_SEPARATE_OPERATOR_TASK"
        or authority.get("identity") != candidate["identity"]
        or authority.get("nonce") != candidate["nonce"]
        or authority.get("authority_cardinality") != 1
        or authority.get("execution_limit") != 1
        or authority.get("candidate_sha256") != sha256_file(CANDIDATE)
        or authority.get("exact_launch_argv") != candidate["exact_future_launch_argv"]
        or authority.get("retry_replay_resume_relaunch")
        != "PERMANENTLY_FORBIDDEN"
    ):
        raise AuthorityError("separate CF18 execution authority does not match candidate")
    return authority


def claim(authority: dict[str, Any], state: Path) -> None:
    candidate = load_object(CANDIDATE)
    if state != Path(candidate["state_namespace"]) or os.path.lexists(state):
        raise AuthorityError("CF18 authority is already or ambiguously consumed")
    mkdir_exclusive(state)
    create_json(
        state / "authorization-consumed.json",
        {
            "schema": "ace2-cf18-authorization-consumption-v1",
            "status": "CONSUMED_BEFORE_ANY_PROCESS",
            "identity": authority["identity"],
            "nonce": authority["nonce"],
            "authority_sha256": sha256_file(AUTHORITY),
            "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        },
    )


def run_process(argv: list[str], timeout_seconds: int) -> dict[str, Any]:
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "HOME": str(Path.home()),
            "OMP_NUM_THREADS": "1",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        natural_terminal = True
    except subprocess.TimeoutExpired:
        natural_terminal = False
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return {
        "pid": process.pid,
        "exit_code": process.returncode,
        "natural_terminal": natural_terminal,
        "stdout": stdout,
        "stderr": stderr,
    }


def validate_and_decode(evidence_path: Path) -> dict[str, Any]:
    evidence = load_object(evidence_path)
    host = evidence.get("host")
    endpoint = evidence.get("endpoint")
    host_positions = host.get("positions") if isinstance(host, dict) else None
    rtl_positions = endpoint.get("rtl_positions") if isinstance(endpoint, dict) else None
    if (
        evidence.get("status") != "HOST_AND_ENDPOINT_RECEIPTS_FSYNCED_NO_DECODE"
        or not isinstance(host_positions, list)
        or not isinstance(rtl_positions, list)
        or len(host_positions) != 4
        or len(rtl_positions) != 4
    ):
        raise AuthorityError("CF18 host/RTL evidence is incomplete")
    generated = []
    comparisons = []
    for host_position, rtl_position in zip(
        host_positions, rtl_positions, strict=True
    ):
        selected_match = (
            host_position["selected_token_id"] == rtl_position["selected_token_id"]
        )
        kv_match = (
            host_position["per_layer_kv_digests"]
            == rtl_position["per_layer_kv_digests"]
        )
        comparisons.append(
            {
                "ordinal": host_position["ordinal"],
                "selected_token_id_match": selected_match,
                "per_layer_kv_digests_match": kv_match,
            }
        )
        generated.append(host_position["selected_token_id"])
    if not all(
        item["selected_token_id_match"] and item["per_layer_kv_digests_match"]
        for item in comparisons
    ):
        raise AuthorityError("CF18 host/RTL semantic comparison failed")
    bindings = load_object(PACKAGE / "bindings.json")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        bindings["model"]["root"],
        local_files_only=True,
        trust_remote_code=False,
    )
    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "schema": "ace2-cf18-product-result-v1",
        "status": "HOST_RTL_AGREEMENT_AND_OFFICIAL_DECODE_COMPLETE",
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "comparisons": comparisons,
        "software_fallback": "FORBIDDEN",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }


def execute(authority: dict[str, Any]) -> int:
    candidate = load_object(CANDIDATE)
    state = Path(candidate["state_namespace"])
    attempt = Path(candidate["output_namespace"])
    claim(authority, state)
    mkdir_exclusive(attempt)
    product = candidate["exact_product_invocation"]
    process = run_process(list(product["argv"]), product["timeout_seconds"])
    create_json(
        attempt / "process-receipt.json",
        {
            "schema": "ace2-cf18-process-receipt-v1",
            "exit_code": process["exit_code"],
            "natural_terminal": process["natural_terminal"],
            "stdout_sha256": hashlib.sha256(process["stdout"]).hexdigest(),
            "stderr_sha256": hashlib.sha256(process["stderr"]).hexdigest(),
        },
    )
    classification = "PRODUCT_PROCESS_FAILED"
    try:
        if process["exit_code"] != 0 or not process["natural_terminal"]:
            raise AuthorityError("CF18 product process did not terminate successfully")
        result = validate_and_decode(Path(product["evidence_path"]))
        create_json(attempt / "product-result.json", result)
        classification = result["status"]
        return_code = 0
    except BaseException as error:
        create_json(
            attempt / "failure.json",
            {
                "schema": "ace2-cf18-failure-v1",
                "classification": classification,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "generated_token_claim": "NONE",
            },
        )
        return_code = 2
    outcome = (
        attempt / "product-result.json"
        if return_code == 0
        else attempt / "failure.json"
    )
    create_json(
        state / "terminal-seal.json",
        {
            "schema": "ace2-cf18-terminal-seal-v1",
            "status": "SEALED_TERMINAL_NO_RETRY",
            "classification": classification,
            "outcome_sha256": sha256_file(outcome),
            "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
        },
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args()
    if args.authority.resolve() != AUTHORITY.resolve():
        raise AuthorityError("CF18 exact authority path changed")
    return execute(validate_external_authority(args.authority.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
