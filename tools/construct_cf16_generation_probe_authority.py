#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/ace2_chat_demo"
IDENTITY = "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0001"
PACKAGE = BUILD_ROOT / IDENTITY
STATE = BUILD_ROOT / f"{IDENTITY}-authority-state-0001"
ATTEMPT = STATE / "attempt-0001"
CF15_IDENTITY = (
    "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
)
CF15 = BUILD_ROOT / CF15_IDENTITY
CF15_SUCCESSOR = BUILD_ROOT / (
    "stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0002"
)
CF14_STATE = BUILD_ROOT / (
    "stage1-w4a8-r6-generation-probe-product-host-repair-cf14-0001-authority-state"
)
ENDPOINT_CANDIDATE = BUILD_ROOT / "stage1-w4a8-r6-rtl-endpoint-perf-candidate-0001"
CF15_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationproberepair15/round-0002.json"
)
CF16_REVIEW_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobeauthority16"
)
CF16_MISSION = CF16_REVIEW_ROOT / "mission.json"
PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")
CONSTRUCTOR = Path(__file__).resolve()

EXPECTED_CF15 = {
    "contract.json": "ca54aebc53f0858fb09b851f5b3e1a34b9617897233e671d32fa821d806fd36c",
    "package-manifest.json": "c2e08aaa88051446748d9038d8147719c4d55b1d624e7b35633a87392a655fb4",
    "source-constraint-manifest.json": "f48e30bcf4a892efce37b3f15849a7f57bfabf43900f61524482d9a2546df620",
    "review-request.json": "dc0ddb527867cbf01c0bbd8e205b68678b5b0cada8b3bb286432eb5234d500d1",
}
EXPECTED_CF15_REVIEW = "137957f3da7fbc113870115e6f5997e69f0fbeb26459b13f78586e3ec4256148"


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


def file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular file: {path}")
    observed = path.stat()
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def write_file(path: Path, value: bytes | str, mode: int = 0o444) -> None:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def tree_records(root: Path) -> dict[str, dict[str, object]]:
    if not root.is_dir():
        raise RuntimeError(f"required immutable tree is absent: {root}")
    records: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in immutable tree: {relative}")
        if path.is_file():
            records[relative] = file_record(path)
    return records


def authenticate_sources() -> None:
    for relative, expected in EXPECTED_CF15.items():
        if sha256_file(CF15 / relative) != expected:
            raise RuntimeError(f"accepted CF15 anchor changed: {relative}")
    if sha256_file(CF15_REVIEW) != EXPECTED_CF15_REVIEW:
        raise RuntimeError("canonical CF15 independent L2 handoff changed")
    review = load_object(CF15_REVIEW)
    if (
        review.get("producer_role") != "reviewer"
        or review.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("canonical CF15 independent L2 review is not done")
    if not CF16_MISSION.is_file():
        raise RuntimeError("CF16 immutable mission is absent")


def exact_process_matches(argv_options: tuple[tuple[str, ...], ...]) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = tuple(part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part)
        if argv in argv_options:
            matches.append(int(entry.name))
    return sorted(matches)


def authority_values() -> tuple[dict[str, object], list[str], list[str]]:
    launch_argv = [
        str(PYTHON),
        "-B",
        str(PACKAGE / "authority_runner.py"),
        "--authority",
        str(PACKAGE / "authority.json"),
    ]
    product_argv = [
        str(PYTHON),
        "-B",
        str(CF15 / "product_probe.py"),
        "--output",
        str(ATTEMPT / "product/product-rtl-evidence.json"),
        "--capture",
        str(ATTEMPT / "product/product-host-error-capture.json"),
        "--endpoint-output",
        str(ATTEMPT / "product/rtl-completion"),
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/argustest",
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    authority = {
        "schema": "ace2-r6-cf16-exactly-once-local-diagnostic-authority-v1",
        "identity": IDENTITY,
        "status": "GRANTED_PENDING_CANONICAL_INDEPENDENT_L2",
        "authority_cardinality": 1,
        "execution_limit": 1,
        "authorized_action": "EXECUTE_ACCEPTED_CF15_LOCAL_STAGE1_DIAGNOSTIC_EXACTLY_ONCE",
        "accepted_cf15_identity": CF15_IDENTITY,
        "accepted_cf15_anchors": EXPECTED_CF15,
        "accepted_cf15_reviewer_sha256": EXPECTED_CF15_REVIEW,
        "activation_gate": {
            "activation_before_reviewer_done_handoff": "FORBIDDEN",
            "mission_id": "r6generationprobeauthority16",
            "mission_path": str(CF16_MISSION),
            "mission_sha256": sha256_file(CF16_MISSION),
            "required_level": "INDEPENDENT_L2",
            "review_root": str(CF16_REVIEW_ROOT),
            "status": "PENDING",
        },
        "consumption_evidence_namespace": str(STATE),
        "output_namespace": str(ATTEMPT),
        "output_namespace_must_not_exist_before_consumption": True,
        "authority_consumption": "DURABLE_BEFORE_ANY_PROCESS_CREATION",
        "launch_invocation": {
            "argv": launch_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(launch_argv)),
            "local_only": True,
        },
        "product_host_invocation": {
            "argv": product_argv,
            "argv_sha256": sha256_bytes(canonical_bytes(product_argv)),
            "working_directory": str(ROOT),
            "natural_terminal_only": True,
        },
        "environment": environment,
        "environment_sha256": sha256_bytes(canonical_bytes(environment)),
        "executed_state_contract": {
            "source": "CF15_MODEL24_FORWARD_HOOK_EXECUTED_LAYER_OUTPUT",
            "layer_count": 24,
            "layer_order": "EXACT_0_THROUGH_23",
            "shape": "ONE_POSITION_BY_COMMON_NONZERO_HIDDEN_SIZE",
            "dtype": "COMMON_SUPPORTED_FLOAT_DTYPE",
            "per_layer_hash": "REQUIRED",
            "aggregate_hash": "REQUIRED",
            "distinct_storage": "REQUIRED",
            "synthetic_or_missing_state": "FORBIDDEN",
            "position_count": 4,
        },
        "persistence_before_rtl_endpoint": {
            "host_state_validation": "REQUIRED_BEFORE_HOST_RESULT",
            "host_evidence_atomic_file_fsync_parent_fsync": "REQUIRED",
            "order": "VALIDATED_CF15_HOST_EVIDENCE_DURABLE_BEFORE_RTL_ENDPOINT",
            "persistence_failure": "FAIL_CLOSED_BEFORE_RTL_ENDPOINT",
        },
        "rtl_endpoint_binding": "CF15_BASELINE_BINDINGS_ONLY",
        "endpoint_optimization_candidate": {
            "identity": ENDPOINT_CANDIDATE.name,
            "integration": "EXCLUDED",
            "future_integration_requirement": (
                "SEPARATE_ADDITIVE_PACKAGE_WITH_EXACT_HASH_AND_SEMANTIC_EQUIVALENCE_REAUTHENTICATION"
            ),
        },
        "natural_terminal": {
            "runner_timeout": "NONE",
            "runner_signal_or_kill": "FORBIDDEN",
            "one_product_process": True,
        },
        "retry": "PERMANENTLY_FORBIDDEN",
        "replay": "PERMANENTLY_FORBIDDEN",
        "resume": "PERMANENTLY_FORBIDDEN",
        "relaunch": "PERMANENTLY_FORBIDDEN",
        "terminal_sealing": {
            "seal_path": str(STATE / "terminal-seal.json"),
            "seal_after_consumption_on_every_runner_controlled_outcome": True,
            "partial_evidence_hashes": True,
            "natural_process_receipt_durable_before_seal": True,
        },
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "non_sram_area_cap_mm2": 2.0,
        "minimum_frequency_mhz": 100,
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
        "not_authorized": [
            "CONSTRUCTION_OR_REVIEW_TIME_EXECUTION",
            "TOKENIZER_OR_MODEL_GENERATION",
            "PRODUCT_HOST_OR_RTL_ENDPOINT_BEFORE_CF16_ACCEPTANCE",
            "SIMULATOR_SYNTHESIS_TIMING_PPA_FPGA",
            "REMOTE_GPU_OR_STAGE2",
            "ENDPOINT_OPTIMIZATION_CANDIDATE_INTEGRATION",
        ],
    }
    return authority, launch_argv, product_argv


RUNNER_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0001"
STATE = PACKAGE.parent / f"{IDENTITY}-authority-state-0001"
ATTEMPT = STATE / "attempt-0001"
TERMINAL = STATE / "terminal-seal.json"
REVIEW_ROOT = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/r6generationprobeauthority16")
PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")


class AuthorityError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


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
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    atomic_bytes(path, canonical_bytes(value), exclusive=exclusive)


def load_validator() -> Any:
    specification = importlib.util.spec_from_file_location("_cf16_validator", PACKAGE / "validate_package.py")
    if specification is None or specification.loader is None:
        raise AuthorityError("cannot import CF16 validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_review() -> dict[str, Any]:
    validator = load_validator()
    validator.validate_package(require_zero_state=True)
    authority = load_object(PACKAGE / "authority.json")
    gate = authority["activation_gate"]
    if Path(gate["review_root"]) != REVIEW_ROOT:
        raise AuthorityError("canonical CF16 review root changed")
    latest = load_object(REVIEW_ROOT / "latest.json")
    if latest.get("kind") != "handoff_ref":
        raise AuthorityError("CF16 authority lacks canonical independent L2 acceptance")
    handoff_path = Path(latest.get("handoff", {}).get("path", ""))
    try:
        handoff_path.resolve().relative_to(REVIEW_ROOT.resolve())
    except ValueError as error:
        raise AuthorityError("canonical CF16 handoff escapes review root") from error
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != gate["mission_id"]
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise AuthorityError("CF16 authority is pending canonical independent L2 acceptance")
    if sha256_file(Path(gate["mission_path"])) != gate["mission_sha256"]:
        raise AuthorityError("CF16 immutable mission changed")
    return authority


def package_state_paths() -> tuple[Path, ...]:
    return (
        PACKAGE / "execution-registry.json",
        PACKAGE / "execution-claim.json",
        PACKAGE / "authorization-consumed.json",
        PACKAGE / "consumption-receipt.json",
        PACKAGE / "attempt-0001",
        PACKAGE / "output",
        PACKAGE / "terminal-seal.json",
    )


def claim(authority: dict[str, Any], *, state: Path = STATE, attempt: Path = ATTEMPT) -> None:
    if (
        authority.get("consumption_evidence_namespace") != str(state)
        or authority.get("output_namespace") != str(attempt)
        or attempt != state / "attempt-0001"
    ):
        raise AuthorityError("consumption and output namespaces are not authority-bound")
    if os.path.lexists(state) or any(os.path.lexists(path) for path in package_state_paths()):
        raise AuthorityError("CF16 authority is already or ambiguously consumed")
    state.mkdir(mode=0o700)
    fsync_directory(state.parent)
    consumed = state / "authorization-consumed.json"
    consumed_record = {
        "schema": "ace2-r6-cf16-authorization-consumption-v1",
        "status": "CONSUMED_BEFORE_ANY_PROCESS",
        "identity": authority["identity"],
        "authority_sha256": sha256_file(PACKAGE / "authority.json"),
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
    }
    atomic_json(consumed, consumed_record, exclusive=True)
    consumed.chmod(0o444)
    claim_path = state / "execution-claim.json"
    atomic_json(
        claim_path,
        {
            "schema": "ace2-r6-cf16-execution-claim-v1",
            "status": "CLAIMED_AFTER_DURABLE_CONSUMPTION_BEFORE_PROCESS",
            "authorization_consumed_sha256": sha256_file(consumed),
            "output_namespace": str(attempt),
        },
        exclusive=True,
    )
    claim_path.chmod(0o444)
    receipt = state / "consumption-receipt.json"
    atomic_json(
        receipt,
        {
            "schema": "ace2-r6-cf16-consumption-receipt-v1",
            "status": "DURABLE_BEFORE_PROCESS_CREATION",
            "authorization_consumed_sha256": sha256_file(consumed),
            "execution_claim_sha256": sha256_file(claim_path),
        },
        exclusive=True,
    )
    receipt.chmod(0o444)
    attempt.mkdir(mode=0o700)
    fsync_directory(state)


def run_process(argv: list[str], environment: dict[str, str]) -> dict[str, Any]:
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = process.communicate()
    return {
        "pid": process.pid,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "natural_terminal": True,
    }


def artifact_record(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.is_symlink():
        return None
    return {
        "path": str(path),
        "mode": stat.S_IMODE(path.stat().st_mode),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def validate_product_evidence(path: Path) -> None:
    evidence = load_object(path)
    if evidence.get("status") != "COMPLETE":
        raise AuthorityError("CF15 product/RTL evidence is not complete")
    host = evidence.get("host")
    if not isinstance(host, dict) or host.get("position_count") != 4:
        raise AuthorityError("CF15 host evidence lacks four positions")
    positions = host.get("positions")
    if not isinstance(positions, list) or len(positions) != 4:
        raise AuthorityError("CF15 host position list is malformed")
    for ordinal, position in enumerate(positions):
        if not isinstance(position, dict) or position.get("ordinal") != ordinal:
            raise AuthorityError("CF15 host positions are missing or reordered")
        layer_hashes = position.get("per_layer_state_digests")
        aggregate = position.get("layer_state_contract_sha256")
        if (
            not isinstance(layer_hashes, list)
            or len(layer_hashes) != 24
            or any(not isinstance(value, str) or len(value) != 64 for value in layer_hashes)
            or not isinstance(aggregate, str)
            or len(aggregate) != 64
        ):
            raise AuthorityError("CF15 executed 24-layer state hashes are malformed")


def seal_terminal(
    authority: dict[str, Any],
    *,
    state: Path,
    attempt: Path,
    classification: str,
    status: str,
) -> None:
    terminal = state / "terminal-seal.json"
    if terminal.exists():
        return
    candidates = (
        state / "authorization-consumed.json",
        state / "execution-claim.json",
        state / "consumption-receipt.json",
        attempt / "execution-journal.json",
        attempt / "product/stdout.bin",
        attempt / "product/stderr.bin",
        attempt / "product/process-receipt.json",
        attempt / "product/product-rtl-evidence.json",
        attempt / "product/product-host-error-capture.json",
    )
    artifacts = [record for path in candidates if (record := artifact_record(path)) is not None]
    atomic_json(
        terminal,
        {
            "schema": "ace2-r6-cf16-terminal-seal-v1",
            "status": status,
            "classification": classification,
            "identity": authority["identity"],
            "authority_sha256": sha256_file(PACKAGE / "authority.json"),
            "artifacts": artifacts,
            "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
            "stage1": "OPEN",
            "stage2": "FORBIDDEN",
            "stage_transition": "DISABLED",
        },
        exclusive=True,
    )
    terminal.chmod(0o444)
    fsync_directory(state)


def execute_authorized(
    authority: dict[str, Any],
    *,
    state: Path = STATE,
    attempt: Path = ATTEMPT,
    process_runner: Callable[[list[str], dict[str, str]], dict[str, Any]] = run_process,
) -> int:
    try:
        claim(authority, state=state, attempt=attempt)
        journal_path = attempt / "execution-journal.json"
        atomic_json(
            journal_path,
            {
                "schema": "ace2-r6-cf16-execution-journal-v1",
                "status": "AUTHORITY_CONSUMED_PRODUCT_NOT_STARTED",
                "product_process_count": 0,
                "reference_process": "FORBIDDEN",
                "retry_replay_resume_relaunch": "FORBIDDEN",
            },
            exclusive=True,
        )
        product = attempt / "product"
        product.mkdir(mode=0o700)
        fsync_directory(attempt)
        result = process_runner(
            list(authority["product_host_invocation"]["argv"]),
            dict(authority["environment"]),
        )
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
            raise AuthorityError("process runner returned non-byte output")
        if result.get("natural_terminal") is not True:
            raise AuthorityError("product process did not report one natural terminal")
        atomic_bytes(product / "stdout.bin", stdout, exclusive=True)
        atomic_bytes(product / "stderr.bin", stderr, exclusive=True)
        receipt_path = product / "process-receipt.json"
        atomic_json(
            receipt_path,
            {
                "schema": "ace2-r6-cf16-product-process-receipt-v1",
                "status": "ONE_NATURAL_TERMINAL_CAPTURE_FSYNCED",
                "pid": result.get("pid"),
                "exit_code": result.get("exit_code"),
                "natural_terminal": True,
                "argv_sha256": hashlib.sha256(canonical_bytes(authority["product_host_invocation"]["argv"])).hexdigest(),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            },
            exclusive=True,
        )
        journal = load_object(journal_path)
        journal["product_process_count"] = 1
        journal["status"] = "ONE_NATURAL_TERMINAL_RECEIPT_FSYNCED"
        atomic_json(journal_path, journal)
        evidence_path = product / "product-rtl-evidence.json"
        if result.get("exit_code") == 0:
            validate_product_evidence(evidence_path)
            classification = "CF15_LOCAL_DIAGNOSTIC_COMPLETED"
            status = "SEALED_TERMINAL_DIAGNOSTIC_COMPLETED_NO_RETRY"
        elif (product / "product-host-error-capture.json").is_file():
            classification = "CF15_PRODUCT_HOST_EXCEPTION_CAPTURED"
            status = "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        else:
            classification = "CF15_PRODUCT_OR_RTL_PROCESS_FAILED"
            status = "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        seal_terminal(
            authority,
            state=state,
            attempt=attempt,
            classification=classification,
            status=status,
        )
        return 0 if status == "SEALED_TERMINAL_DIAGNOSTIC_COMPLETED_NO_RETRY" else 2
    except (AuthorityError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        if (state / "authorization-consumed.json").is_file():
            seal_terminal(
                authority,
                state=state,
                attempt=attempt,
                classification=f"RUNNER_EXCEPTION_{type(error).__name__}",
                status="SEALED_TERMINAL_RUNNER_FAILURE_NO_RETRY",
            )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.authority.resolve() != (PACKAGE / "authority.json").resolve():
        raise AuthorityError("only the canonical CF16 authority is accepted")
    if Path(sys.executable).resolve() != PYTHON.resolve() or not sys.dont_write_bytecode:
        raise AuthorityError("frozen Python executable or -B policy changed")
    return execute_authorized(validate_review())


if __name__ == "__main__":
    raise SystemExit(main())
'''


VALIDATOR_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = "stage1-w4a8-r6-generation-probe-diagnostic-authority-cf16-0001"
STATE = PACKAGE.parent / f"{IDENTITY}-authority-state-0001"
CF15 = ROOT / "build/ace2_chat_demo/stage1-w4a8-r6-generation-probe-product-host-contract-repair-cf15-0001"
CF15_REVIEW = Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/r6generationproberepair15/round-0002.json")
EXPECTED_CF15 = {
    "contract.json": "ca54aebc53f0858fb09b851f5b3e1a34b9617897233e671d32fa821d806fd36c",
    "package-manifest.json": "c2e08aaa88051446748d9038d8147719c4d55b1d624e7b35633a87392a655fb4",
    "source-constraint-manifest.json": "f48e30bcf4a892efce37b3f15849a7f57bfabf43900f61524482d9a2546df620",
    "review-request.json": "dc0ddb527867cbf01c0bbd8e205b68678b5b0cada8b3bb286432eb5234d500d1",
}
EXPECTED_CF15_REVIEW = "137957f3da7fbc113870115e6f5997e69f0fbeb26459b13f78586e3ec4256148"


class ValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value


def file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"expected regular file: {path}")
    observed = path.stat()
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": sha256_file(path),
        "size": observed.st_size,
    }


def verify_digest(path: Path, target_name: str) -> None:
    expected = f"{sha256_file(path.parent / target_name)}  {target_name}\n"
    if path.read_text(encoding="ascii") != expected:
        raise ValidationError(f"detached digest mismatch: {path.name}")


def verify_package_manifest(package: Path = PACKAGE) -> None:
    manifest = load_object(package / "package-manifest.json")
    if manifest.get("schema") != "ace2-r6-cf16-package-manifest-v1":
        raise ValidationError("unexpected package manifest schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValidationError("package manifest files are absent")
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    actual: dict[str, dict[str, object]] = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).as_posix()
        if path.is_symlink():
            raise ValidationError(f"package symlink is forbidden: {relative}")
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode) != 0o555:
                raise ValidationError(f"package directory mode changed: {relative}")
        elif relative not in excluded:
            actual[relative] = file_record(path)
    if actual != expected:
        raise ValidationError("package hash/mode/size closure changed")
    verify_digest(package / "package-manifest.sha256", "package-manifest.json")
    request = load_object(package / "review-request.json")
    if request.get("status") != "PENDING_INDEPENDENT_L2":
        raise ValidationError("review request status changed")
    for relative, expected_record in request.get("targets", {}).items():
        if file_record(package / relative) != expected_record:
            raise ValidationError(f"review target changed: {relative}")
    verify_digest(package / "review-request.sha256", "review-request.json")


def resolve_record_path(relative: str) -> Path:
    path = ROOT / relative
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValidationError("inventory path escapes repository") from error
    return path


def verify_inventory(name: str) -> None:
    inventory = load_object(PACKAGE / name)
    entries = inventory.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValidationError(f"{name} is empty")
    for relative, expected in entries.items():
        if file_record(resolve_record_path(relative)) != expected:
            raise ValidationError(f"immutable inventory changed: {relative}")


def verify_accepted_cf15() -> None:
    for relative, expected in EXPECTED_CF15.items():
        if sha256_file(CF15 / relative) != expected:
            raise ValidationError(f"accepted CF15 anchor changed: {relative}")
    if sha256_file(CF15_REVIEW) != EXPECTED_CF15_REVIEW:
        raise ValidationError("canonical CF15 reviewer handoff changed")
    review = load_object(CF15_REVIEW)
    if (
        review.get("producer_role") != "reviewer"
        or review.get("review", {}).get("status") != "done"
    ):
        raise ValidationError("canonical CF15 independent L2 review is not done")


def exact_process_matches(authority: dict[str, Any]) -> list[int]:
    expected = {
        tuple(authority["launch_invocation"]["argv"]),
        tuple(authority["product_host_invocation"]["argv"]),
    }
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = tuple(part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part)
        if argv in expected:
            matches.append(int(entry.name))
    return sorted(matches)


def verify_zero_state(authority: dict[str, Any]) -> None:
    forbidden = (
        STATE,
        PACKAGE / "execution-registry.json",
        PACKAGE / "execution-claim.json",
        PACKAGE / "authorization-consumed.json",
        PACKAGE / "consumption-receipt.json",
        PACKAGE / "attempt-0001",
        PACKAGE / "output",
        PACKAGE / "terminal-seal.json",
    )
    if any(os.path.lexists(path) for path in forbidden):
        raise ValidationError("CF16 namespace has registry, claim, attempt, output, terminal, or receipt state")
    if exact_process_matches(authority):
        raise ValidationError("CF16 launch or product process already exists")


def verify_contract(authority: dict[str, Any]) -> None:
    contract = load_object(PACKAGE / "contract.json")
    if (
        contract.get("execution_limit") != 1
        or contract.get("authority_state") != "UNCONSUMED"
        or contract.get("stage1") != "OPEN"
        or contract.get("stage2") != "FORBIDDEN"
        or contract.get("public_rtl_contract") != "UNCHANGED_14_PARAMETERS_64_PORTS"
    ):
        raise ValidationError("CF16 contract gates changed")
    if (
        authority.get("authority_cardinality") != 1
        or authority.get("authority_consumption") != "DURABLE_BEFORE_ANY_PROCESS_CREATION"
        or authority.get("natural_terminal", {}).get("runner_timeout") != "NONE"
        or authority.get("endpoint_optimization_candidate", {}).get("integration") != "EXCLUDED"
    ):
        raise ValidationError("CF16 exactly-once authority policy changed")
    for key in ("retry", "replay", "resume", "relaunch"):
        if authority.get(key) != "PERMANENTLY_FORBIDDEN":
            raise ValidationError(f"CF16 {key} policy changed")


def validate_package(*, require_zero_state: bool = True) -> None:
    verify_package_manifest()
    verify_digest(PACKAGE / "source-constraint-manifest.sha256", "source-constraint-manifest.json")
    verify_accepted_cf15()
    verify_inventory("predecessor-read-only-inventory.json")
    verify_inventory("endpoint-candidate-exclusion-inventory.json")
    authority = load_object(PACKAGE / "authority.json")
    verify_contract(authority)
    if require_zero_state:
        verify_zero_state(authority)


def main() -> None:
    validate_package()
    print("PASS_CF16_EXACTLY_ONCE_AUTHORITY_ZERO_STATE")


if __name__ == "__main__":
    main()
'''


TEST_SOURCE = r'''from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load_module("_cf16_runner_test", PACKAGE / "authority_runner.py")
validator = load_module("_cf16_validator_test", PACKAGE / "validate_package.py")


def authority_for(state: Path) -> dict[str, object]:
    authority = json.loads((PACKAGE / "authority.json").read_text(encoding="utf-8"))
    authority["consumption_evidence_namespace"] = str(state)
    authority["output_namespace"] = str(state / "attempt-0001")
    return authority


class PackageTests(unittest.TestCase):
    def test_package_closure_and_zero_state(self) -> None:
        validator.validate_package(require_zero_state=True)

    def test_tampered_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "package"
            shutil.copytree(PACKAGE, copy)
            target = copy / "contract.json"
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaisesRegex(validator.ValidationError, "closure changed"):
                validator.verify_package_manifest(copy)

    def test_endpoint_candidate_is_inventory_only(self) -> None:
        authority = json.loads((PACKAGE / "authority.json").read_text(encoding="utf-8"))
        argv = authority["product_host_invocation"]["argv"]
        self.assertFalse(any("rtl-endpoint-perf-candidate" in value for value in argv))
        self.assertEqual(authority["endpoint_optimization_candidate"]["integration"], "EXCLUDED")


class ExactlyOnceTests(unittest.TestCase):
    def test_duplicate_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            authority = authority_for(state)
            runner.claim(authority, state=state, attempt=state / "attempt-0001")
            with self.assertRaisesRegex(runner.AuthorityError, "already or ambiguously consumed"):
                runner.claim(authority, state=state, attempt=state / "attempt-0001")

    def test_consumption_is_durable_before_process_and_natural_terminal_seals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def process_runner(_argv, _environment):
                self.assertTrue((state / "authorization-consumed.json").is_file())
                self.assertTrue((state / "execution-claim.json").is_file())
                self.assertTrue((state / "consumption-receipt.json").is_file())
                product = attempt / "product"
                layer_hashes = ["0" * 64 for _ in range(24)]
                positions = [
                    {
                        "ordinal": ordinal,
                        "per_layer_state_digests": layer_hashes,
                        "layer_state_contract_sha256": "1" * 64,
                    }
                    for ordinal in range(4)
                ]
                runner.atomic_json(
                    product / "product-rtl-evidence.json",
                    {"status": "COMPLETE", "host": {"position_count": 4, "positions": positions}},
                    exclusive=True,
                )
                return {
                    "pid": 123,
                    "exit_code": 0,
                    "stdout": b"",
                    "stderr": b"",
                    "natural_terminal": True,
                }

            self.assertEqual(
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=process_runner,
                ),
                0,
            )
            seal = json.loads((state / "terminal-seal.json").read_text(encoding="utf-8"))
            self.assertEqual(seal["classification"], "CF15_LOCAL_DIAGNOSTIC_COMPLETED")
            receipt = json.loads(
                (attempt / "product/process-receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(receipt["natural_terminal"])

    def test_post_consumption_runner_failure_is_terminally_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def failing_runner(_argv, _environment):
                raise RuntimeError("mocked process boundary failure")

            with self.assertRaisesRegex(RuntimeError, "mocked process boundary failure"):
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=failing_runner,
                )
            seal = json.loads((state / "terminal-seal.json").read_text(encoding="utf-8"))
            self.assertEqual(seal["status"], "SEALED_TERMINAL_RUNNER_FAILURE_NO_RETRY")

    def test_non_natural_terminal_is_rejected_and_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            attempt = state / "attempt-0001"
            authority = authority_for(state)

            def bad_runner(_argv, _environment):
                return {
                    "pid": 123,
                    "exit_code": 0,
                    "stdout": b"",
                    "stderr": b"",
                    "natural_terminal": False,
                }

            with self.assertRaisesRegex(runner.AuthorityError, "natural terminal"):
                runner.execute_authorized(
                    authority,
                    state=state,
                    attempt=attempt,
                    process_runner=bad_runner,
                )
            self.assertTrue((state / "terminal-seal.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


def predecessor_inventory() -> dict[str, object]:
    source = load_object(CF15_SUCCESSOR / "predecessor-read-only-inventory.json")
    entries = source.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError("CF15 predecessor inventory is malformed")
    combined = dict(entries)
    combined.update(tree_records(CF15_SUCCESSOR))
    return {
        "schema": "ace2-r6-cf16-cf12-through-cf15-read-only-inventory-v1",
        "entries": dict(sorted(combined.items())),
        "consumed_cf14_state": str(CF14_STATE),
        "accepted_cf15_identity": CF15_IDENTITY,
        "cf15_successor_preserved": CF15_SUCCESSOR.name,
        "certified_baselines": "PRESERVED_TRANSITIVELY_BY_PREDECESSOR_CLOSURE",
    }


def endpoint_inventory() -> dict[str, object]:
    request = load_object(ENDPOINT_CANDIDATE / "review-request.json")
    return {
        "schema": "ace2-r6-cf16-endpoint-candidate-exclusion-inventory-v1",
        "identity": ENDPOINT_CANDIDATE.name,
        "integration": "EXCLUDED",
        "observed_review_status": request.get("status"),
        "required_before_future_integration": (
            "SEPARATE_ADDITIVE_PACKAGE_REAUTHENTICATES_ALL_FILES_AND_SEMANTIC_EQUIVALENCE_EVIDENCE"
        ),
        "entries": tree_records(ENDPOINT_CANDIDATE),
    }


def source_manifest() -> dict[str, object]:
    sources = {
        CONSTRUCTOR.relative_to(ROOT).as_posix(): {
            "scope": "repository",
            **file_record(CONSTRUCTOR),
        },
        CF15_REVIEW.as_posix(): {
            "scope": "external_canonical",
            **file_record(CF15_REVIEW),
        },
        CF16_MISSION.as_posix(): {
            "scope": "external_canonical",
            **file_record(CF16_MISSION),
        },
    }
    for relative in EXPECTED_CF15:
        path = CF15 / relative
        sources[path.relative_to(ROOT).as_posix()] = {
            "scope": "repository",
            **file_record(path),
        }
    return {
        "schema": "ace2-r6-cf16-source-constraint-manifest-v1",
        "files": dict(sorted(sources.items())),
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "non_sram_area_cap_mm2": 2.0,
        "minimum_frequency_mhz": 100,
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "rtl_change": "NONE",
        "constraint_change": "NONE",
    }


def zero_state_report(
    launch_argv: list[str], product_argv: list[str]
) -> dict[str, object]:
    forbidden = {
        "state_namespace": STATE,
        "registry": PACKAGE / "execution-registry.json",
        "claim": PACKAGE / "execution-claim.json",
        "attempt": PACKAGE / "attempt-0001",
        "output": PACKAGE / "output",
        "terminal": PACKAGE / "terminal-seal.json",
        "consumption_receipt": PACKAGE / "consumption-receipt.json",
        "authorization_consumed": PACKAGE / "authorization-consumed.json",
    }
    observed = {
        name: os.path.lexists(path)
        for name, path in forbidden.items()
    }
    process_matches = exact_process_matches((tuple(launch_argv), tuple(product_argv)))
    if any(observed.values()) or process_matches:
        raise RuntimeError("CF16 namespace is not virgin")
    predecessor_states = [
        CF14_STATE,
        BUILD_ROOT / "stage1-w4a8-r6-generation-probe-cf12-attempt-0001-authority-state",
    ]
    if any(STATE.resolve() == path.resolve() for path in predecessor_states):
        raise RuntimeError("CF16 state namespace is not path-distinct")
    return {
        "schema": "ace2-r6-cf16-preconstruction-zero-state-v1",
        "identity": IDENTITY,
        "state_namespace": str(STATE),
        "path_distinct_from_predecessor_states": True,
        "observed_paths_exist": observed,
        "exact_launch_or_product_process_matches": process_matches,
        "status": "PASS_ZERO_REGISTRY_CLAIM_ATTEMPT_OUTPUT_TERMINAL_RECEIPT_PROCESS_STATE",
    }


def materialize(package: Path) -> None:
    authority, launch_argv, product_argv = authority_values()
    contract = {
        "schema": "ace2-r6-cf16-local-diagnostic-authority-contract-v1",
        "identity": IDENTITY,
        "authority_state": "UNCONSUMED",
        "execution_limit": 1,
        "accepted_cf15_identity": CF15_IDENTITY,
        "accepted_cf15_anchors": EXPECTED_CF15,
        "accepted_cf15_reviewer_sha256": EXPECTED_CF15_REVIEW,
        "future_invocation_sha256": authority["launch_invocation"]["argv_sha256"],
        "product_invocation_sha256": authority["product_host_invocation"]["argv_sha256"],
        "genuine_executed_layer_states": 24,
        "durable_validation_before_rtl_endpoint": True,
        "natural_terminal_count": 1,
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "terminal_seal_every_post_consumption_outcome": True,
        "endpoint_optimization_candidate": "EXCLUDED",
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "non_sram_area_cap_mm2": 2.0,
        "minimum_frequency_mhz": 100,
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "official_execution": False,
    }
    write_file(package / "contract.json", canonical_bytes(contract))
    write_file(package / "authority.json", canonical_bytes(authority))
    write_file(package / "authority_runner.py", RUNNER_SOURCE, 0o555)
    write_file(package / "validate_package.py", VALIDATOR_SOURCE, 0o555)
    write_file(package / "tests/test_cf16_authority.py", TEST_SOURCE)
    write_file(
        package / "preconstruction-zero-state.json",
        canonical_bytes(zero_state_report(launch_argv, product_argv)),
    )
    write_file(
        package / "predecessor-read-only-inventory.json",
        canonical_bytes(predecessor_inventory()),
    )
    write_file(
        package / "endpoint-candidate-exclusion-inventory.json",
        canonical_bytes(endpoint_inventory()),
    )
    source = source_manifest()
    write_file(package / "source-constraint-manifest.json", canonical_bytes(source))
    write_file(
        package / "source-constraint-manifest.sha256",
        f"{sha256_file(package / 'source-constraint-manifest.json')}  source-constraint-manifest.json\n",
    )
    write_file(
        package / "execution-backlog.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-execution-backlog-v1",
                "identity": IDENTITY,
                "action": "EXECUTE_ACCEPTED_CF15_LOCAL_STAGE1_DIAGNOSTIC_EXACTLY_ONCE",
                "status": "BLOCKED_PENDING_CANONICAL_INDEPENDENT_L2_ACCEPTANCE",
                "separate_from_construction_and_review": True,
            }
        ),
    )
    write_file(
        package / "construction-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-construction-report-v1",
                "status": "CONSTRUCTED_NOT_EXECUTED",
                "model_tokenizer_product_host_rtl_endpoint_simulator_remote_gpu_stage2": "NOT_EXECUTED",
                "deterministic_non_model_checks": "REQUIRED_AFTER_PUBLICATION",
                "cf12_through_cf15_and_endpoint_candidate": "READ_ONLY",
            }
        ),
    )
    write_file(
        package / "construction-repair-record.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-construction-repair-v1",
                "failure_taxonomy": "PACKAGE_PUBLICATION_MODE_CONFLICT",
                "root_cause": (
                    "THE_TEMPORARY_PACKAGE_ROOT_WAS_MODE_0555_BEFORE_CROSS_PARENT_RENAME"
                ),
                "scope": "PUBLICATION_ONLY_NO_AUTHORITY_STATE_OR_EXECUTION_CREATED",
                "regression": (
                    "ONLY_THE_TEMPORARY_ROOT_IS_MODE_0755_DURING_RENAME_AND_THE_PUBLISHED_ROOT_IS_IMMEDIATELY_MODE_0555"
                ),
            }
        ),
    )
    write_file(
        package / "Makefile",
        (
            ".PHONY: check\n"
            "check:\n"
            "\t/home/argustest/miniconda3/bin/python3.13 -B validate_package.py\n"
            "\t/home/argustest/miniconda3/bin/python3.13 -B -m unittest discover -s tests -v\n"
        ),
    )

    core = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file()
    }
    write_file(
        package / "reproducibility-report.json",
        canonical_bytes(
            {
                "schema": "ace2-r6-cf16-reproducibility-v1",
                "status": "PASS_TWO_INDEPENDENT_MATERIALIZATIONS_DIRECT_BYTES_MODE_SIZE_IDENTICAL",
                "comparison": "DIRECT_FILE_SET_BYTES_MODE_SIZE",
                "core_files": core,
                "official_execution": False,
            }
        ),
    )
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    files = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.relative_to(package).as_posix() not in excluded
    }
    manifest = {
        "schema": "ace2-r6-cf16-package-manifest-v1",
        "identity": IDENTITY,
        "files": files,
    }
    write_file(package / "package-manifest.json", canonical_bytes(manifest))
    write_file(
        package / "package-manifest.sha256",
        f"{sha256_file(package / 'package-manifest.json')}  package-manifest.json\n",
    )
    targets = {
        relative: file_record(package / relative)
        for relative in (
            "authority.json",
            "contract.json",
            "endpoint-candidate-exclusion-inventory.json",
            "package-manifest.json",
            "preconstruction-zero-state.json",
            "predecessor-read-only-inventory.json",
            "reproducibility-report.json",
            "source-constraint-manifest.json",
        )
    }
    request = {
        "schema": "ace2-r6-cf16-canonical-independent-l2-review-request-v1",
        "identity": IDENTITY,
        "status": "PENDING_INDEPENDENT_L2",
        "activation_before_acceptance": "FORBIDDEN",
        "execution_during_review": "FORBIDDEN",
        "requested_level": "CANONICAL_INDEPENDENT_L2",
        "required_checks": [
            "CF15_EXACT_ANCHORS_AND_CANONICAL_REVIEW_AUTHENTICATED",
            "CF12_THROUGH_CF15_AND_CONSUMED_CF14_CLOSURE_UNCHANGED",
            "VIRGIN_PATH_DISTINCT_AUTHORITY_STATE_AND_ZERO_PROCESS",
            "AUTHORITY_CONSUMED_DURABLY_BEFORE_PROCESS_CREATION",
            "EXACT_CF15_INVOCATION_AND_GENUINE_ORDERED_24_LAYER_STATE_CONTRACT",
            "DURABLE_VALIDATION_AND_PERSISTENCE_BEFORE_RTL_ENDPOINT",
            "ONE_NATURAL_TERMINAL_NO_RETRY_REPLAY_RESUME_RELAUNCH",
            "TERMINAL_SEAL_ON_EVERY_POST_CONSUMPTION_RUNNER_OUTCOME",
            "ENDPOINT_PERFORMANCE_CANDIDATE_EXCLUDED",
            "NON_MODEL_TAMPER_DUPLICATE_ZERO_STATE_TESTS",
        ],
        "targets": targets,
    }
    write_file(package / "review-request.json", canonical_bytes(request))
    write_file(
        package / "review-request.sha256",
        f"{sha256_file(package / 'review-request.json')}  review-request.json\n",
    )

    for directory in sorted(
        (path for path in package.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    package.chmod(0o555)


def compare_materializations(first: Path, second: Path) -> None:
    first_paths = [
        path.relative_to(first).as_posix()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    ]
    second_paths = [
        path.relative_to(second).as_posix()
        for path in sorted(second.rglob("*"))
        if path.is_file()
    ]
    if first_paths != second_paths:
        raise RuntimeError("CF16 materialization file sets differ")
    for relative in first_paths:
        first_path = first / relative
        second_path = second / relative
        if (
            first_path.read_bytes() != second_path.read_bytes()
            or stat.S_IMODE(first_path.stat().st_mode)
            != stat.S_IMODE(second_path.stat().st_mode)
            or first_path.stat().st_size != second_path.stat().st_size
        ):
            raise RuntimeError(f"CF16 materialization differs: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PACKAGE)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to replace existing package: {output}")
    if STATE.exists():
        raise RuntimeError(f"refusing construction with existing state: {STATE}")
    authenticate_sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cf16-build-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        materialize(first)
        materialize(second)
        compare_materializations(first, second)
        first.chmod(0o755)
        first.rename(output)
        output.chmod(0o555)
    print(f"CONSTRUCTED_CF16_NOT_EXECUTED {output}")


if __name__ == "__main__":
    main()
