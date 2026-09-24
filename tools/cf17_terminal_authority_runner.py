#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = "stage1-w4a8-r6-generation-probe-terminal-accountability-cf17-0002"
STATE = PACKAGE.parent / f"{IDENTITY}-authority-state-0001"
ATTEMPT = STATE / "attempt-0001"
CF16_STATE = (
    PACKAGE.parent
    / "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0002-authority-state-0001"
)
HOST_EVIDENCE = CF16_STATE / "attempt-0001/product/product-rtl-evidence.json"
REVIEW_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/5387edbfef70"
)
PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")


class AuthorityError(RuntimeError):
    pass


Publisher = Callable[[Path, object], None]
ProcessRunner = Callable[[list[str], dict[str, str], int], dict[str, Any]]


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
        raise AuthorityError(f"JSON root is not an object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_bytes(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short create-only publication write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        created = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    published = path.lstat()
    if (
        not stat.S_ISREG(published.st_mode)
        or (created.st_dev, created.st_ino) != (published.st_dev, published.st_ino)
    ):
        raise AuthorityError(f"create-only publication identity changed: {path}")
    fsync_directory(path.parent)


def create_json(path: Path, value: object) -> None:
    create_bytes(path, canonical_bytes(value))


def mkdir_create_only(path: Path, mode: int = 0o700) -> None:
    path.mkdir(mode=mode)
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or path.is_symlink():
        raise AuthorityError(f"created namespace is not a regular directory: {path}")
    fsync_directory(path.parent)


def file_record(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.is_symlink():
        return None
    observed = path.stat()
    return {
        "path": str(path),
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def load_validator() -> Any:
    specification = importlib.util.spec_from_file_location(
        "_cf17_validator", PACKAGE / "validate_package.py"
    )
    if specification is None or specification.loader is None:
        raise AuthorityError("cannot import CF17 validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_review() -> dict[str, Any]:
    load_validator().validate_package(require_zero_state=True)
    authority = load_object(PACKAGE / "authority.json")
    gate = authority["activation_gate"]
    if Path(gate["review_root"]) != REVIEW_ROOT:
        raise AuthorityError("canonical CF17 review root changed")
    latest = load_object(REVIEW_ROOT / "latest.json")
    if latest.get("kind") != "handoff_ref":
        raise AuthorityError("CF17 authority lacks independent Reviewer acceptance")
    handoff_path = Path(latest.get("handoff", {}).get("path", ""))
    try:
        handoff_path.resolve().relative_to(REVIEW_ROOT.resolve())
    except ValueError as error:
        raise AuthorityError("canonical CF17 handoff escapes review root") from error
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != gate["mission_id"]
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise AuthorityError("CF17 authority is pending independent Reviewer acceptance")
    if sha256_file(Path(gate["mission_path"])) != gate["mission_sha256"]:
        raise AuthorityError("CF17 immutable mission changed")
    return authority


def validate_host_evidence(authority: dict[str, Any]) -> list[dict[str, Any]]:
    binding = authority["cf16_host_capture"]
    if Path(binding["path"]) != HOST_EVIDENCE:
        raise AuthorityError("CF16 host-capture path changed")
    if sha256_file(HOST_EVIDENCE) != binding["sha256"]:
        raise AuthorityError("CF16 host-capture bytes changed")
    evidence = load_object(HOST_EVIDENCE)
    if (
        evidence.get("schema") != "ace2-r6-cf13-product-rtl-evidence-v1"
        or evidence.get("status") != "HOST_EVIDENCE_FSYNCED_BEFORE_RTL_ENDPOINT"
        or evidence.get("endpoint") is not None
        or evidence.get("decode") != "FORBIDDEN_UNTIL_FSYNCED_HOST_AND_RTL_EVIDENCE"
    ):
        raise AuthorityError("CF16 durable host-capture envelope changed")
    host = evidence.get("host")
    positions = host.get("positions") if isinstance(host, dict) else None
    if (
        not isinstance(host, dict)
        or host.get("position_count") != 4
        or not isinstance(positions, list)
        or len(positions) != 4
    ):
        raise AuthorityError("CF16 durable host capture lacks four positions")
    expected = binding["positions"]
    for ordinal, position in enumerate(positions):
        if (
            not isinstance(position, dict)
            or position.get("ordinal") != ordinal
            or position.get("absolute_position") != expected[ordinal]["absolute_position"]
            or position.get("selected_token_id") != expected[ordinal]["selected_token_id"]
            or position.get("layer_state_contract_sha256")
            != expected[ordinal]["layer_state_contract_sha256"]
            or len(position.get("per_layer_state_digests", [])) != 24
            or len(position.get("per_layer_kv_digests", [])) != 24
        ):
            raise AuthorityError("CF16 durable host positions changed or are malformed")
    return positions


def claim(
    authority: dict[str, Any], *, state: Path = STATE, attempt: Path = ATTEMPT
) -> None:
    if (
        authority.get("consumption_evidence_namespace") != str(state)
        or authority.get("output_namespace") != str(attempt)
        or attempt != state / "attempt-0001"
    ):
        raise AuthorityError("CF17 consumption and output namespaces are not bound")
    if os.path.lexists(state):
        raise AuthorityError("CF17 authority is already or ambiguously consumed")
    mkdir_create_only(state)
    create_json(
        state / "authorization-consumed.json",
        {
            "schema": "ace2-r6-cf17-authorization-consumption-v1",
            "status": "CONSUMED_BEFORE_ENDPOINT_PROCESS",
            "identity": authority["identity"],
            "authority_sha256": sha256_file(PACKAGE / "authority.json"),
            "cf16_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        },
    )
    create_json(
        state / "execution-claim.json",
        {
            "schema": "ace2-r6-cf17-execution-claim-v1",
            "status": "CLAIMED_AFTER_DURABLE_CONSUMPTION",
            "authorization_consumed_sha256": sha256_file(
                state / "authorization-consumed.json"
            ),
            "output_namespace": str(attempt),
        },
    )
    mkdir_create_only(attempt)
    mkdir_create_only(attempt / "endpoint-output")
    mkdir_create_only(attempt / "fallback")
    mkdir_create_only(state / "fallback")
    create_json(
        state / "consumption-receipt.json",
        {
            "schema": "ace2-r6-cf17-consumption-receipt-v1",
            "status": "DURABLE_BEFORE_ENDPOINT_PROCESS",
            "authorization_consumed_sha256": sha256_file(
                state / "authorization-consumed.json"
            ),
            "execution_claim_sha256": sha256_file(state / "execution-claim.json"),
        },
    )


def run_endpoint(
    argv: list[str], environment: dict[str, str], timeout_seconds: int
) -> dict[str, Any]:
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {
            "pid": process.pid,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "natural_terminal": True,
        }
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return {
            "pid": process.pid,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "natural_terminal": False,
            "failure_type": type(error).__name__,
        }


def parse_endpoint_positions(
    commands_path: Path, host_positions: list[dict[str, Any]]
) -> list[dict[str, object]]:
    selected: dict[int, dict[str, list[dict[str, Any]]]] = {
        position["absolute_position"]: {"kv": [], "logits": []}
        for position in host_positions
    }
    with commands_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            bucket = selected.get(record.get("token_step"))
            if bucket is None:
                continue
            if record.get("operator") == "kv_write":
                bucket["kv"].append(record)
            elif record.get("operator") == "lm_head_tile":
                bucket["logits"].append(record)
    positions: list[dict[str, object]] = []
    for ordinal, host in enumerate(host_positions):
        bucket = selected[host["absolute_position"]]
        kv = bucket["kv"]
        logits = bucket["logits"]
        if len(kv) != 24 or len(logits) != 4748:
            raise AuthorityError("RTL endpoint lacks 24 KV writes or 4748 logit tiles")
        aggregate = hashlib.sha256()
        for item in logits:
            aggregate.update(
                f"{item['ordinal']}:{item['destination_sha256']}\n".encode("ascii")
            )
        positions.append(
            {
                "ordinal": ordinal,
                "absolute_position": host["absolute_position"],
                "selected_token_id": logits[-1]["generated_token_after"],
                "per_layer_kv_digests": [
                    item["destination_sha256"] for item in kv
                ],
                "authenticated_logit_tile_count": 4748,
                "authenticated_logit_tile_aggregate_sha256": aggregate.hexdigest(),
            }
        )
    return positions


def agreement_record(
    host_positions: list[dict[str, Any]], rtl_positions: list[dict[str, object]]
) -> dict[str, object]:
    comparisons = []
    for host, rtl in zip(host_positions, rtl_positions, strict=True):
        selected_match = host["selected_token_id"] == rtl["selected_token_id"]
        kv_match = host["per_layer_kv_digests"] == rtl["per_layer_kv_digests"]
        comparisons.append(
            {
                "ordinal": host["ordinal"],
                "selected_token_id_match": selected_match,
                "per_layer_kv_digests_match": kv_match,
            }
        )
    passed = all(
        item["selected_token_id_match"] and item["per_layer_kv_digests_match"]
        for item in comparisons
    )
    return {
        "schema": "ace2-r6-cf17-host-rtl-agreement-v1",
        "status": "PASS" if passed else "FAIL",
        "positions": comparisons,
        "software_fallback": "FORBIDDEN",
    }


def publish_failure(
    *,
    publisher: Publisher,
    attempt: Path,
    classification: str,
    error: BaseException | None,
    process: dict[str, Any] | None,
) -> Path:
    payload = {
        "schema": "ace2-r6-cf17-endpoint-failure-capture-v1",
        "status": "RTL_ENDPOINT_FAILED",
        "classification": classification,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
        "process_exit_code": process.get("exit_code") if process else None,
        "natural_terminal": process.get("natural_terminal") if process else None,
        "software_fallback": "FORBIDDEN",
    }
    primary = attempt / "endpoint-failure-capture.json"
    try:
        publisher(primary, payload)
        return primary
    except (AuthorityError, OSError, RuntimeError, ValueError, TypeError) as failure:
        payload["primary_publication_failure"] = type(failure).__name__
        fallback = attempt / "fallback/endpoint-failure-capture.json"
        create_json(fallback, payload)
        return fallback


def publish_terminal(
    *,
    publisher: Publisher,
    authority: dict[str, Any],
    state: Path,
    attempt: Path,
    classification: str,
    outcome_path: Path,
) -> Path:
    artifacts = []
    for path in (
        state / "authorization-consumed.json",
        state / "execution-claim.json",
        state / "consumption-receipt.json",
        attempt / "host-input-receipt.json",
        attempt / "stdout.bin",
        attempt / "stderr.bin",
        attempt / "endpoint-process-receipt.json",
        attempt / "endpoint-receipt.json",
        attempt / "endpoint-failure-capture.json",
        attempt / "fallback/endpoint-failure-capture.json",
        attempt / "host-rtl-agreement.json",
    ):
        record = file_record(path)
        if record is not None:
            artifacts.append(record)
    payload = {
        "schema": "ace2-r6-cf17-terminal-seal-v1",
        "status": "SEALED_TERMINAL_NO_RETRY",
        "classification": classification,
        "identity": authority["identity"],
        "authority_sha256": sha256_file(PACKAGE / "authority.json"),
        "outcome_sha256": sha256_file(outcome_path),
        "artifacts": artifacts,
        "cf16_retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    primary = state / "terminal-seal.json"
    try:
        publisher(primary, payload)
        return primary
    except (AuthorityError, OSError, RuntimeError, ValueError, TypeError) as failure:
        payload["primary_publication_failure"] = type(failure).__name__
        fallback = state / "fallback/terminal-seal.json"
        create_json(fallback, payload)
        return fallback


def execute_authorized(
    authority: dict[str, Any],
    *,
    state: Path = STATE,
    attempt: Path = ATTEMPT,
    process_runner: ProcessRunner = run_endpoint,
    publisher: Publisher = create_json,
) -> int:
    host_positions = validate_host_evidence(authority)
    claim(authority, state=state, attempt=attempt)
    create_json(
        attempt / "host-input-receipt.json",
        {
            "schema": "ace2-r6-cf17-host-input-receipt-v1",
            "status": "CF16_HOST_CAPTURE_AUTHENTICATED_READ_ONLY_BEFORE_ENDPOINT",
            "path": str(HOST_EVIDENCE),
            "sha256": sha256_file(HOST_EVIDENCE),
            "position_count": 4,
            "layer_state_count_per_position": 24,
            "cf16_authority_state_mutation": "FORBIDDEN",
        },
    )
    process: dict[str, Any] | None = None
    outcome: Path | None = None
    classification = "CF17_RUNNER_FAILURE"
    try:
        endpoint = authority["rtl_endpoint_invocation"]
        process = process_runner(
            list(endpoint["argv"]),
            dict(authority["environment"]),
            endpoint["timeout_seconds"],
        )
        stdout = process.get("stdout")
        stderr = process.get("stderr")
        if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
            raise AuthorityError("endpoint process returned non-byte output")
        create_bytes(attempt / "stdout.bin", stdout)
        create_bytes(attempt / "stderr.bin", stderr)
        create_json(
            attempt / "endpoint-process-receipt.json",
            {
                "schema": "ace2-r6-cf17-endpoint-process-receipt-v1",
                "status": "PROCESS_TERMINAL_CAPTURE_FSYNCED",
                "pid": process.get("pid"),
                "exit_code": process.get("exit_code"),
                "natural_terminal": process.get("natural_terminal"),
                "argv_sha256": sha256_bytes(canonical_bytes(endpoint["argv"])),
                "stdout_sha256": sha256_bytes(stdout),
                "stderr_sha256": sha256_bytes(stderr),
            },
        )
        if process.get("exit_code") != 0 or process.get("natural_terminal") is not True:
            classification = "RTL_ENDPOINT_PROCESS_FAILURE"
            outcome = publish_failure(
                publisher=publisher,
                attempt=attempt,
                classification=classification,
                error=None,
                process=process,
            )
        else:
            rtl_positions = parse_endpoint_positions(
                attempt / "endpoint-output/commands.jsonl", host_positions
            )
            receipt = {
                "schema": "ace2-r6-cf17-endpoint-receipt-v1",
                "status": "RTL_ENDPOINT_RECEIPT_FSYNCED",
                "argv_sha256": sha256_bytes(canonical_bytes(endpoint["argv"])),
                "exit_code": 0,
                "stdout_sha256": sha256_bytes(stdout),
                "stderr_sha256": sha256_bytes(stderr),
                "rtl_positions": rtl_positions,
                "software_fallback": "FORBIDDEN",
            }
            outcome = attempt / "endpoint-receipt.json"
            publisher(outcome, receipt)
            agreement = agreement_record(host_positions, rtl_positions)
            create_json(attempt / "host-rtl-agreement.json", agreement)
            classification = (
                "RTL_ENDPOINT_AND_HOST_AGREEMENT_COMPLETED"
                if agreement["status"] == "PASS"
                else "RTL_ENDPOINT_HOST_AGREEMENT_MISMATCH"
            )
    except BaseException as error:
        classification = f"RTL_ENDPOINT_EXCEPTION_{type(error).__name__}"
        outcome = publish_failure(
            publisher=publisher,
            attempt=attempt,
            classification=classification,
            error=error,
            process=process,
        )
    if outcome is None:
        raise AuthorityError("post-consumption outcome was not published")
    publish_terminal(
        publisher=publisher,
        authority=authority,
        state=state,
        attempt=attempt,
        classification=classification,
        outcome_path=outcome,
    )
    return 0 if classification == "RTL_ENDPOINT_AND_HOST_AGREEMENT_COMPLETED" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.authority.resolve() != (PACKAGE / "authority.json").resolve():
        raise AuthorityError("only the canonical CF17 authority is accepted")
    if Path(sys.executable).resolve() != PYTHON.resolve() or not sys.dont_write_bytecode:
        raise AuthorityError("frozen Python executable or -B policy changed")
    return execute_authorized(validate_review())


if __name__ == "__main__":
    raise SystemExit(main())
