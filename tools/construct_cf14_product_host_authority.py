#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build/ace2_chat_demo"
IDENTITY = "stage1-w4a8-r6-generation-probe-product-host-repair-cf14-0001"
PACKAGE = BUILD_ROOT / IDENTITY
STATE = BUILD_ROOT / f"{IDENTITY}-authority-state"
ATTEMPT = STATE / "attempt-0001"
CF13 = (
    BUILD_ROOT
    / "stage1-w4a8-r6-generation-probe-product-host-repair-cf13-0001"
)
CF12_SEAL = (
    BUILD_ROOT
    / "stage1-w4a8-r6-generation-probe-cf12-terminal-seal-0001"
)
CF13_REVIEW = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobereview13/round-0002.json"
)
CF14_REVIEW_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobeauthority14"
)
CF14_MISSION = CF14_REVIEW_ROOT / "mission.json"
PYTHON = Path("/home/argustest/miniconda3/bin/python3.13")

EXPECTED = {
    "cf13_reviewer": "ce9c56070e19c278fb473104085e938ea02188e985fd90516f93b5bd8a824fb7",
    "cf13_review_request": "fcc080062a720037d45d4a9cf3f73a1bca7c0442034deed7ca26ffbc71464c67",
    "cf13_contract": "509a2ca3d44995d5aae8df90b426fe95122ad58f91533f512695e7925c4ec094",
    "cf13_source_manifest": "242b572b64a35118219f22fe3fa9529bbaa3a1c64b275d29cd372e75077081ce",
    "cf12_terminal_seal": "dc3cb069ee1efe3cb210b41915ff36448670704b7e569c7d77200078c86d260a",
}


VALIDATOR_SOURCE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = "stage1-w4a8-r6-generation-probe-product-host-repair-cf14-0001"
CANONICAL_PACKAGE = (
    ROOT / "build/ace2_chat_demo" / IDENTITY
)
STATE = CANONICAL_PACKAGE.parent / f"{IDENTITY}-authority-state"
CF13 = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-generation-probe-product-host-repair-cf13-0001"
)
EXPECTED = {
    "cf13_reviewer": "ce9c56070e19c278fb473104085e938ea02188e985fd90516f93b5bd8a824fb7",
    "cf13_review_request": "fcc080062a720037d45d4a9cf3f73a1bca7c0442034deed7ca26ffbc71464c67",
    "cf13_contract": "509a2ca3d44995d5aae8df90b426fe95122ad58f91533f512695e7925c4ec094",
    "cf13_source_manifest": "242b572b64a35118219f22fe3fa9529bbaa3a1c64b275d29cd372e75077081ce",
    "cf12_terminal_seal": "dc3cb069ee1efe3cb210b41915ff36448670704b7e569c7d77200078c86d260a",
}


class ValidationError(RuntimeError):
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot load canonical JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root is not an object: {path}")
    return value


def file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"expected regular file: {path}")
    return {
        "mode": stat.S_IMODE(path.stat().st_mode),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def verify_digest_file(path: Path, target_name: str) -> None:
    expected = f"{sha256_file(path.parent / target_name)}  {target_name}\n"
    if path.read_text(encoding="ascii") != expected:
        raise ValidationError(f"detached digest mismatch: {path.name}")


def verify_package_manifest(package: Path) -> None:
    manifest = load_object(package / "package-manifest.json")
    if manifest.get("schema") != "ace2-r6-cf14-package-content-manifest-v1":
        raise ValidationError("unexpected package manifest schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValidationError("package manifest files are missing")
    actual: dict[str, dict[str, object]] = {}
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
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
        raise ValidationError("package content hash/mode/size closure changed")
    verify_digest_file(package / "package-manifest.sha256", "package-manifest.json")

    request = load_object(package / "review-request.json")
    targets = request.get("targets")
    if not isinstance(targets, dict):
        raise ValidationError("review request targets are missing")
    actual_targets = {
        path.relative_to(package).as_posix(): file_record(path)
        for path in sorted(package.rglob("*"))
        if path.is_file()
        and path.name not in {"review-request.json", "review-request.sha256"}
    }
    if targets != actual_targets:
        raise ValidationError("review-request target closure changed")
    verify_digest_file(package / "review-request.sha256", "review-request.json")


def resolve_reference(record: dict[str, Any]) -> Path:
    scope = record.get("scope")
    raw = record.get("path")
    if not isinstance(raw, str):
        raise ValidationError("source reference path is invalid")
    if scope == "repository":
        path = ROOT / raw
        try:
            path.resolve().relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValidationError("repository source reference escapes root") from error
        return path
    if scope == "external_canonical":
        path = Path(raw)
        if not path.is_absolute():
            raise ValidationError("external canonical reference is not absolute")
        return path
    raise ValidationError(f"unknown source-reference scope: {scope}")


def verify_source_references(package: Path) -> None:
    manifest = load_object(package / "source-reference-manifest.json")
    if manifest.get("schema") != "ace2-r6-cf14-source-reference-closure-v1":
        raise ValidationError("unexpected source-reference manifest schema")
    references = manifest.get("files")
    if not isinstance(references, list) or not references:
        raise ValidationError("source-reference closure is empty")
    seen: set[str] = set()
    for record in references:
        if not isinstance(record, dict):
            raise ValidationError("source-reference record is invalid")
        path = resolve_reference(record)
        key = str(path)
        if key in seen:
            raise ValidationError(f"duplicate source reference: {path}")
        seen.add(key)
        symlink_target = record.get("symlink_target")
        if symlink_target is not None:
            if (
                not path.is_symlink()
                or os.readlink(path) != symlink_target
                or str(path.resolve(strict=True)) != record.get("resolved_path")
            ):
                raise ValidationError(f"source-reference symlink changed: {path}")
            observed = file_record(path.resolve(strict=True))
        else:
            observed = file_record(path)
        if observed != {
            "mode": record.get("mode"),
            "sha256": record.get("sha256"),
            "size": record.get("size"),
        }:
            raise ValidationError(f"source reference changed: {path}")

    anchors = manifest.get("anchors")
    if anchors != EXPECTED:
        raise ValidationError("canonical source anchors changed")
    anchor_paths = {
        "cf13_reviewer": Path(manifest["anchor_paths"]["cf13_reviewer"]),
        "cf13_review_request": CF13 / "review-request.json",
        "cf13_contract": CF13 / "contract.json",
        "cf13_source_manifest": CF13 / "source-constraint-manifest.json",
        "cf12_terminal_seal": (
            ROOT
            / "build/ace2_chat_demo"
            / "stage1-w4a8-r6-generation-probe-cf12-terminal-seal-0001"
            / "terminal-seal.json"
        ),
    }
    for name, path in anchor_paths.items():
        if sha256_file(path) != EXPECTED[name]:
            raise ValidationError(f"canonical source anchor changed: {name}")

    cf13_request = load_object(CF13 / "review-request.json")
    for relative, digest in cf13_request["targets"].items():
        if sha256_file(CF13 / relative) != digest:
            raise ValidationError(f"CF13 reviewed target changed: {relative}")
    cf13_closure = load_object(CF13 / "source-constraint-manifest.json")
    for relative, record in cf13_closure["files"].items():
        path = CF13 / relative if record["scope"] == "package" else ROOT / relative
        observed = file_record(path)
        if (
            observed["sha256"] != record["sha256"]
            or observed["mode"] != record["mode"]
        ):
            raise ValidationError(f"CF13 transitive source changed: {relative}")
    cf12_inventory = load_object(CF13 / "cf12-read-only-inventory.json")
    for relative, record in cf12_inventory["entries"].items():
        observed = file_record(ROOT / relative)
        if (
            observed["sha256"] != record["sha256"]
            or observed["mode"] != record["mode"]
            or observed["size"] != record["size"]
        ):
            raise ValidationError(f"accepted CF12 closure changed: {relative}")


def process_cmdlines() -> list[list[str]]:
    cmdlines: list[list[str]] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if raw:
            cmdlines.append(
                [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
            )
    return cmdlines


def verify_semantics(package: Path, *, check_freshness: bool) -> None:
    contract = load_object(package / "contract.json")
    authority = load_object(package / "authority.json")
    bindings = load_object(CF13 / "bindings.json")
    prompt = (package / "input.txt").read_bytes()

    if contract.get("identity") != IDENTITY or authority.get("identity") != IDENTITY:
        raise ValidationError("CF14 identity changed")
    if contract.get("canonical_sources") != EXPECTED:
        raise ValidationError("CF14 canonical source bindings changed")
    if contract.get("stage1") != "OPEN" or contract.get("stage2") != "FORBIDDEN":
        raise ValidationError("Stage gate changed")
    if contract.get("rtl_change") != "NONE":
        raise ValidationError("CF14 may not change RTL")
    if contract.get("execution_during_construction") != "FORBIDDEN":
        raise ValidationError("construction execution policy changed")
    if authority.get("authority_cardinality") != 1 or authority.get("execution_limit") != 1:
        raise ValidationError("exactly-once authority cardinality changed")
    if authority.get("status") != "GRANTED_PENDING_CANONICAL_INDEPENDENT_L2":
        raise ValidationError("authority activation status changed")
    if authority.get("authority_consumption") != "BEFORE_ANY_PROCESS":
        raise ValidationError("authority consumption order changed")
    if authority.get("retry_replay_resume_relaunch") != "PERMANENTLY_FORBIDDEN":
        raise ValidationError("terminal no-retry policy changed")
    if authority.get("output_namespace") != str(STATE / "attempt-0001"):
        raise ValidationError("output namespace changed")
    if authority.get("consumption_evidence_namespace") != str(STATE):
        raise ValidationError("authority state namespace changed")
    if contract.get("reference_behavior") != {
        "new_reference_process": "FORBIDDEN",
        "accepted_cf12_reference_evidence": "READ_ONLY_CONTEXT_NOT_EXPECTED_OUTPUT",
        "golden_or_hidden_output": "FORBIDDEN",
        "self_derived_expected_tokens": "FORBIDDEN",
    }:
        raise ValidationError("reference behavior changed")

    prompt_binding = authority.get("prompt")
    if not isinstance(prompt_binding, dict):
        raise ValidationError("prompt binding is missing")
    if (
        hashlib.sha256(prompt).hexdigest() != prompt_binding.get("sha256")
        or len(prompt) != prompt_binding.get("byte_count")
        or prompt.hex() != prompt_binding.get("bytes_hex")
        or prompt_binding.get("sha256") != bindings["prompt_binding"]["sha256"]
    ):
        raise ValidationError("prompt bytes or digest changed")
    if authority.get("model_tokenizer_w4a8") != contract.get("model_tokenizer_w4a8"):
        raise ValidationError("model/tokenizer/W4A8 binding split")

    launch = authority["launch_invocation"]["argv"]
    expected_launch = [
        "/home/argustest/miniconda3/bin/python3.13",
        "-B",
        str(CANONICAL_PACKAGE / "authority_runner.py"),
        "--authority",
        str(CANONICAL_PACKAGE / "authority.json"),
    ]
    if launch != expected_launch:
        raise ValidationError("local launch argv changed")
    if hashlib.sha256(canonical_bytes(launch)).hexdigest() != authority["launch_invocation"]["argv_sha256"]:
        raise ValidationError("local launch argv digest changed")

    product = authority["product_host_invocation"]["argv"]
    expected_product = [
        "/home/argustest/miniconda3/bin/python3.13",
        "-B",
        str(CF13 / "product_probe.py"),
        "--output",
        str(STATE / "attempt-0001/product/product-rtl-evidence.json"),
        "--capture",
        str(STATE / "attempt-0001/product/product-host-error-capture.json"),
        "--endpoint-output",
        str(STATE / "attempt-0001/product/rtl-completion"),
    ]
    if product != expected_product:
        raise ValidationError("product-host argv changed")
    if hashlib.sha256(canonical_bytes(product)).hexdigest() != authority["product_host_invocation"]["argv_sha256"]:
        raise ValidationError("product-host argv digest changed")

    endpoint = list(bindings["endpoint"]["argv_prefix"]) + [
        "--output",
        str(STATE / "attempt-0001/product/rtl-completion"),
    ]
    if authority["rtl_endpoint_invocation"]["argv"] != endpoint:
        raise ValidationError("RTL endpoint argv changed")
    if hashlib.sha256(canonical_bytes(endpoint)).hexdigest() != authority["rtl_endpoint_invocation"]["argv_sha256"]:
        raise ValidationError("RTL endpoint argv digest changed")
    if authority["rtl_endpoint_invocation"]["software_fallback"] != "FORBIDDEN":
        raise ValidationError("software fallback was enabled")

    if check_freshness:
        forbidden = (
            STATE,
            CANONICAL_PACKAGE / "authorization-consumed.json",
            CANONICAL_PACKAGE / "execution-claim.json",
            CANONICAL_PACKAGE / "execution-registry.json",
            CANONICAL_PACKAGE / "attempt-0001",
            CANONICAL_PACKAGE / "evidence",
            CANONICAL_PACKAGE / "product-result.json",
            CANONICAL_PACKAGE / "terminal-seal.json",
        )
        if any(os.path.lexists(path) for path in forbidden):
            raise ValidationError("CF14 authority is already or ambiguously consumed")
        signatures = [launch, product, endpoint]
        if any(cmdline in signatures for cmdline in process_cmdlines()):
            raise ValidationError("matching CF14 launch/product/endpoint process exists")


def validate_package(
    package: Path = PACKAGE,
    *,
    check_freshness: bool = True,
    verify_sources: bool = True,
) -> dict[str, object]:
    package = package.resolve()
    if not package.is_dir() or package.is_symlink():
        raise ValidationError("CF14 package is not a regular directory")
    if stat.S_IMODE(package.stat().st_mode) != 0o555:
        raise ValidationError("CF14 package directory is not immutable")
    verify_package_manifest(package)
    if verify_sources:
        verify_source_references(package)
    verify_semantics(package, check_freshness=check_freshness)
    return {
        "status": "PASS_CF14_STATIC_AUTHORITY_UNCONSUMED",
        "identity": IDENTITY,
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }


def main() -> int:
    print(json.dumps(validate_package(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"CF14 validation refused: {error}", file=os.sys.stderr)
        raise SystemExit(2)
'''


RUNNER_SOURCE = r'''
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
import tempfile
from pathlib import Path
from typing import Any, Callable


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
IDENTITY = "stage1-w4a8-r6-generation-probe-product-host-repair-cf14-0001"
STATE = PACKAGE.parent / f"{IDENTITY}-authority-state"
CONSUMED = STATE / "authorization-consumed.json"
ATTEMPT = STATE / "attempt-0001"
REVIEW_ROOT = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/r6generationprobeauthority14"
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
        fsync_directory(path.parent)
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
    path = PACKAGE / "validate_package.py"
    specification = importlib.util.spec_from_file_location("_cf14_validator", path)
    if specification is None or specification.loader is None:
        raise AuthorityError("cannot import CF14 package validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_review(authority: dict[str, Any] | None = None) -> dict[str, Any]:
    validator = load_validator()
    validator.validate_package()
    authority = authority or load_object(PACKAGE / "authority.json")
    gate = authority["activation_gate"]
    if Path(gate["review_root"]) != REVIEW_ROOT:
        raise AuthorityError("canonical L2 review root changed")
    latest = load_object(REVIEW_ROOT / "latest.json")
    if latest.get("kind") != "handoff_ref":
        raise AuthorityError("CF14 authority lacks canonical independent L2 acceptance")
    handoff_path = Path(latest.get("handoff", {}).get("path", ""))
    try:
        handoff_path.resolve().relative_to(REVIEW_ROOT.resolve())
    except ValueError as error:
        raise AuthorityError("canonical L2 handoff escapes its review root") from error
    handoff = load_object(handoff_path)
    if (
        handoff.get("mission_id") != gate["mission_id"]
        or handoff.get("producer_role") != "reviewer"
        or handoff.get("review", {}).get("status") != "done"
    ):
        raise AuthorityError("CF14 authority is pending canonical independent L2 acceptance")
    mission = Path(gate["mission_path"])
    if sha256_file(mission) != gate["mission_sha256"]:
        raise AuthorityError("CF14 immutable review mission changed")
    return authority


def forbidden_paths(state: Path, attempt: Path) -> tuple[Path, ...]:
    return (
        state,
        PACKAGE / "authorization-consumed.json",
        PACKAGE / "execution-claim.json",
        PACKAGE / "execution-registry.json",
        PACKAGE / "attempt-0001",
        PACKAGE / "evidence",
        PACKAGE / "product-result.json",
        PACKAGE / "terminal-seal.json",
        attempt,
    )


def claim(
    authority: dict[str, Any],
    *,
    state: Path = STATE,
    attempt: Path = ATTEMPT,
) -> None:
    if (
        authority.get("consumption_evidence_namespace") != str(state)
        or authority.get("output_namespace") != str(attempt)
        or attempt != state / "attempt-0001"
    ):
        raise AuthorityError("consumption/evidence namespace is not authority-bound")
    if any(os.path.lexists(path) for path in forbidden_paths(state, attempt)):
        raise AuthorityError("CF14 authority is already or ambiguously consumed")
    state.mkdir(mode=0o700)
    fsync_directory(state.parent)
    consumed = state / "authorization-consumed.json"
    record = {
        "schema": "ace2-r6-cf14-authorization-consumption-v1",
        "status": "CONSUMED_BEFORE_ANY_PROCESS",
        "authority_sha256": sha256_file(PACKAGE / "authority.json"),
        "identity": authority["identity"],
        "durable_task_id": authority["durable_task_id"],
        "transaction_nonce": authority["transaction_nonce"],
        "retry": "PERMANENTLY_FORBIDDEN",
        "replay": "PERMANENTLY_FORBIDDEN",
        "resume": "PERMANENTLY_FORBIDDEN",
        "relaunch": "PERMANENTLY_FORBIDDEN",
    }
    atomic_json(consumed, record, exclusive=True)
    consumed.chmod(0o444)
    descriptor = os.open(consumed, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(state)
    attempt.mkdir(mode=0o700)
    fsync_directory(state)


def descendant_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            lines = (entry / "status").read_text(encoding="ascii").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        ppid_line = next((line for line in lines if line.startswith("PPid:")), None)
        if ppid_line is not None:
            parents[int(entry.name)] = int(ppid_line.split()[1])
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in descendants and (parent == root_pid or parent in descendants):
                descendants.add(pid)
                changed = True
    return descendants


def signal_process_tree(root_pid: int, process_signal: signal.Signals) -> list[int]:
    pids = {root_pid, *descendant_pids(root_pid)}
    groups: set[int] = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, process_signal)
        except ProcessLookupError:
            continue
    return sorted(pids)


def run_process(
    argv: list[str],
    environment: dict[str, str],
    timeout_seconds: int,
    signal_grace_seconds: int,
) -> dict[str, Any]:
    process = subprocess.Popen(
        argv,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    termination: list[dict[str, object]] = []
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        terminated = signal_process_tree(process.pid, signal.SIGTERM)
        termination.append({"signal": "SIGTERM", "pids": terminated})
        try:
            stdout, stderr = process.communicate(timeout=signal_grace_seconds)
        except subprocess.TimeoutExpired:
            killed = signal_process_tree(process.pid, signal.SIGKILL)
            termination.append({"signal": "SIGKILL", "pids": killed})
            stdout, stderr = process.communicate()
        exit_code = 124
    return {
        "pid": process.pid,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "termination": termination,
    }


def artifact_record(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.is_symlink():
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "mode": stat.S_IMODE(path.stat().st_mode),
        "size": path.stat().st_size,
    }


def seal_terminal(
    authority: dict[str, Any],
    *,
    state: Path,
    attempt: Path,
    process_receipt: Path | None,
    classification: str,
    status: str,
) -> None:
    product = attempt / "product"
    candidates = (
        product / "product-rtl-evidence.json",
        product / "product-host-error-capture.json",
        product / "process-receipt.json",
        attempt / "execution-journal.json",
    )
    artifacts = [
        record for path in candidates if (record := artifact_record(path)) is not None
    ]
    seal = {
        "schema": "ace2-r6-cf14-terminal-seal-v1",
        "status": status,
        "classification": classification,
        "identity": authority["identity"],
        "authority_sha256": sha256_file(PACKAGE / "authority.json"),
        "authorization_consumption_sha256": sha256_file(
            state / "authorization-consumed.json"
        ),
        "process_receipt_sha256": (
            sha256_file(process_receipt)
            if process_receipt is not None and process_receipt.is_file()
            else None
        ),
        "artifacts": artifacts,
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
    }
    path = attempt / "terminal-seal.json"
    atomic_json(path, seal, exclusive=True)
    path.chmod(0o444)
    fsync_directory(attempt)


def execute_authorized(
    authority: dict[str, Any],
    *,
    state: Path = STATE,
    attempt: Path = ATTEMPT,
    process_runner: Callable[[list[str], dict[str, str], int, int], dict[str, Any]] = run_process,
) -> int:
    claim(authority, state=state, attempt=attempt)
    journal_path = attempt / "execution-journal.json"
    journal = {
        "schema": "ace2-r6-cf14-execution-journal-v1",
        "status": "AUTHORITY_CONSUMED_PRODUCT_NOT_STARTED",
        "reference_process": "FORBIDDEN",
        "product_host": "NOT_STARTED",
        "rtl_endpoint": "NOT_STARTED",
        "decode": "NOT_STARTED",
    }
    atomic_json(journal_path, journal, exclusive=True)
    product_dir = attempt / "product"
    product_dir.mkdir(mode=0o700)
    fsync_directory(attempt)
    product_argv = list(authority["product_host_invocation"]["argv"])
    process_receipt_path = product_dir / "process-receipt.json"
    try:
        result = process_runner(
            product_argv,
            dict(authority["environment"]),
            authority["timeout_policy"]["product_host_process_seconds"],
            authority["timeout_policy"]["signal_grace_seconds"],
        )
        stdout = result["stdout"]
        stderr = result["stderr"]
        if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
            raise AuthorityError("process runner returned non-byte capture")
        atomic_bytes(product_dir / "stdout.bin", stdout, exclusive=True)
        atomic_bytes(product_dir / "stderr.bin", stderr, exclusive=True)
        receipt = {
            "schema": "ace2-r6-cf14-product-host-process-receipt-v1",
            "status": "PROCESS_CAPTURE_FSYNCED",
            "pid": result["pid"],
            "exit_code": result["exit_code"],
            "timed_out": result["timed_out"],
            "termination": result["termination"],
            "argv_sha256": hashlib.sha256(canonical_bytes(product_argv)).hexdigest(),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "reference_process": "NOT_LAUNCHED",
        }
        atomic_json(process_receipt_path, receipt, exclusive=True)
        journal["product_host"] = "PROCESS_RECEIPT_FSYNCED"
        journal["rtl_endpoint"] = "CF13_CAPTURE_ORDER_GOVERNS"
        journal["status"] = "PROCESS_RECEIPT_FSYNCED_TERMINAL_SEAL_PENDING"
        atomic_json(journal_path, journal)

        if result["timed_out"]:
            classification = "TIMEOUT_PROCESS_TREE_TERMINATED"
            status = "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        elif (product_dir / "product-host-error-capture.json").is_file():
            classification = "PRODUCT_HOST_EXCEPTION_CAPTURED_BEFORE_RTL_ENDPOINT"
            status = "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        elif result["exit_code"] != 0:
            classification = "PRODUCT_OR_RTL_PROCESS_FAILED"
            status = "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        elif not (product_dir / "product-rtl-evidence.json").is_file():
            classification = "REQUIRED_PRODUCT_RTL_EVIDENCE_MISSING"
            status = "SEALED_TERMINAL_DIAGNOSTIC_FAILURE_NO_RETRY"
        else:
            classification = "LOCAL_CF13_SUCCESSOR_DIAGNOSTIC_COMPLETED"
            status = "SEALED_TERMINAL_DIAGNOSTIC_COMPLETED_NO_RETRY"
        journal["status"] = "TERMINAL_SEAL_READY_NO_RETRY"
        atomic_json(journal_path, journal)
        seal_terminal(
            authority,
            state=state,
            attempt=attempt,
            process_receipt=process_receipt_path,
            classification=classification,
            status=status,
        )
        return 0 if status == "SEALED_TERMINAL_DIAGNOSTIC_COMPLETED_NO_RETRY" else 2
    except (AuthorityError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        seal_terminal(
            authority,
            state=state,
            attempt=attempt,
            process_receipt=(
                process_receipt_path if process_receipt_path.is_file() else None
            ),
            classification=f"RUNNER_EXCEPTION_{type(error).__name__}",
            status="SEALED_TERMINAL_RUNNER_FAILURE_NO_RETRY",
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.authority.resolve() != (PACKAGE / "authority.json").resolve():
        raise AuthorityError("only the canonical CF14 authority is accepted")
    if Path(sys.executable).resolve() != Path(
        "/home/argustest/miniconda3/bin/python3.13"
    ).resolve() or not sys.dont_write_bytecode:
        raise AuthorityError("frozen Python executable or -B policy changed")
    authority = validate_review()
    return execute_authorized(authority)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AuthorityError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"CF14 authority execution refused: {error}", file=os.sys.stderr)
        raise SystemExit(2)
'''


TEST_SOURCE = r'''
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
CF13 = (
    ROOT
    / "build/ace2_chat_demo"
    / "stage1-w4a8-r6-generation-probe-product-host-repair-cf13-0001"
)
STATE = PACKAGE.parent / f"{PACKAGE.name}-authority-state"


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class Cf14AuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_module(PACKAGE / "validate_package.py", "_cf14_validator_test")
        cls.runner = load_module(PACKAGE / "authority_runner.py", "_cf14_runner_test")
        cls.authority = json.loads((PACKAGE / "authority.json").read_text())

    def test_package_and_transitive_source_closure(self) -> None:
        result = self.validator.validate_package()
        self.assertEqual(result["status"], "PASS_CF14_STATIC_AUTHORITY_UNCONSUMED")
        self.assertFalse(STATE.exists())

    def test_pending_review_refuses_without_consumption(self) -> None:
        with self.assertRaisesRegex(self.runner.AuthorityError, "canonical independent L2"):
            self.runner.validate_review(self.authority)
        self.assertFalse(STATE.exists())

    def test_exactly_once_stage_and_isolation_contract(self) -> None:
        contract = json.loads((PACKAGE / "contract.json").read_text())
        self.assertEqual(self.authority["authority_cardinality"], 1)
        self.assertEqual(self.authority["execution_limit"], 1)
        self.assertEqual(self.authority["authority_consumption"], "BEFORE_ANY_PROCESS")
        self.assertEqual(
            self.authority["retry_replay_resume_relaunch"],
            "PERMANENTLY_FORBIDDEN",
        )
        self.assertEqual(contract["scope"], "LOCAL_CF13_SUCCESSOR_DIAGNOSTIC_ONLY")
        self.assertEqual(contract["stage1"], "OPEN")
        self.assertEqual(contract["stage2"], "FORBIDDEN")
        self.assertEqual(contract["rtl_change"], "NONE")
        self.assertEqual(contract["remote_infrastructure"], "FORBIDDEN")
        self.assertEqual(contract["cf12_identity_reuse"], "FORBIDDEN")
        self.assertEqual(contract["consumed_s7_amlt_reservation"], "OUT_OF_SCOPE_UNTOUCHED")
        self.assertEqual(
            contract["reference_behavior"]["new_reference_process"], "FORBIDDEN"
        )

    def test_tamper_and_mode_changes_are_rejected(self) -> None:
        mutations = (
            ("authority.json", lambda path: path.write_bytes(path.read_bytes() + b" ")),
            ("input.txt", lambda path: path.write_bytes(path.read_bytes() + b"x")),
            ("contract.json", lambda path: path.chmod(0o644)),
            (
                "source-reference-manifest.json",
                lambda path: path.write_bytes(path.read_bytes().replace(b"cf13", b"xf13", 1)),
            ),
        )
        for relative, mutate in mutations:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                clone = Path(temporary) / PACKAGE.name
                shutil.copytree(PACKAGE, clone, copy_function=shutil.copy2)
                clone.chmod(0o555)
                target = clone / relative
                target.chmod(0o644)
                mutate(target)
                if relative != "contract.json":
                    target.chmod(0o444)
                with self.assertRaises(self.validator.ValidationError):
                    self.validator.validate_package(
                        clone, check_freshness=False, verify_sources=False
                    )

    def test_claim_is_exclusive_and_rejects_unauthorized_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "cf14-state"
            attempt = state / "attempt-0001"
            authority = dict(self.authority)
            authority["consumption_evidence_namespace"] = str(state)
            authority["output_namespace"] = str(attempt)
            self.runner.claim(authority, state=state, attempt=attempt)
            self.assertTrue((state / "authorization-consumed.json").is_file())
            self.assertFalse(
                stat.S_IMODE((state / "authorization-consumed.json").stat().st_mode)
                & stat.S_IWUSR
            )
            with self.assertRaisesRegex(self.runner.AuthorityError, "already"):
                self.runner.claim(authority, state=state, attempt=attempt)
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "cf14-state"
            attempt = state / "attempt-0001"
            with self.assertRaisesRegex(self.runner.AuthorityError, "not authority-bound"):
                self.runner.claim(self.authority, state=state, attempt=attempt)

    def test_mocked_exact_argv_claim_before_process_and_terminal_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "cf14-state"
            attempt = state / "attempt-0001"
            authority = json.loads(json.dumps(self.authority))
            authority["consumption_evidence_namespace"] = str(state)
            authority["output_namespace"] = str(attempt)
            expected = list(authority["product_host_invocation"]["argv"])
            product = attempt / "product"
            expected[4] = str(product / "product-rtl-evidence.json")
            expected[6] = str(product / "product-host-error-capture.json")
            expected[8] = str(product / "rtl-completion")
            authority["product_host_invocation"]["argv"] = expected
            observed = []

            def mocked_process(argv, environment, timeout, grace):
                self.assertTrue((state / "authorization-consumed.json").is_file())
                self.assertTrue(attempt.is_dir())
                self.assertEqual(argv, expected)
                self.assertEqual(environment, authority["environment"])
                self.assertEqual(timeout, 7265)
                self.assertEqual(grace, 5)
                observed.append(argv)
                (product / "product-rtl-evidence.json").write_text(
                    '{"status":"MOCK_ONLY"}\n', encoding="utf-8"
                )
                return {
                    "pid": 4242,
                    "exit_code": 0,
                    "stdout": b"",
                    "stderr": b"",
                    "timed_out": False,
                    "termination": [],
                }

            result = self.runner.execute_authorized(
                authority,
                state=state,
                attempt=attempt,
                process_runner=mocked_process,
            )
            self.assertEqual(result, 0)
            self.assertEqual(observed, [expected])
            seal = json.loads((attempt / "terminal-seal.json").read_text())
            self.assertEqual(
                seal["status"], "SEALED_TERMINAL_DIAGNOSTIC_COMPLETED_NO_RETRY"
            )

    def test_cf13_capture_is_fsynced_before_mocked_endpoint(self) -> None:
        runtime = load_module(
            CF13 / "product_host_runtime.py", "_cf13_runtime_for_cf14_test"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            capture = root / "capture.json"
            order = []
            original = runtime.atomic_json

            def recording_atomic(path, value, *, exclusive=False):
                original(path, value, exclusive=exclusive)
                if path == capture:
                    order.append("capture_fsynced")

            def failing_host(tracker):
                tracker.set("POSITION_ZERO_PREPARATION")
                raise TypeError("mocked CF14 host failure")

            def endpoint():
                self.assertTrue(capture.is_file())
                order.append("endpoint")
                return {"exit_code": 0, "mocked": True}

            with mock.patch.object(runtime, "atomic_json", recording_atomic):
                result = runtime.run_product_host_boundary(
                    evidence_path=evidence,
                    capture_path=capture,
                    host_runner=failing_host,
                    endpoint_runner=endpoint,
                )
            self.assertEqual(result, 2)
            self.assertEqual(order, ["capture_fsynced", "endpoint"])

    def test_timeout_terminates_process_tree_then_returns_terminal_code(self) -> None:
        process = mock.Mock(pid=9001, returncode=None)
        process.communicate.side_effect = (
            subprocess.TimeoutExpired(["mock"], 1),
            subprocess.TimeoutExpired(["mock"], 1),
            (b"out", b"err"),
        )
        with (
            mock.patch.object(self.runner.subprocess, "Popen", return_value=process),
            mock.patch.object(
                self.runner,
                "signal_process_tree",
                side_effect=([9001, 9002], [9001, 9002]),
            ) as terminate,
        ):
            result = self.runner.run_process(["mock"], {}, 1, 1)
        self.assertEqual(result["exit_code"], 124)
        self.assertTrue(result["timed_out"])
        self.assertEqual(
            terminate.call_args_list,
            [mock.call(9001, signal.SIGTERM), mock.call(9001, signal.SIGKILL)],
        )
        self.assertEqual(
            result["termination"],
            [
                {"signal": "SIGTERM", "pids": [9001, 9002]},
                {"signal": "SIGKILL", "pids": [9001, 9002]},
            ],
        )

    def test_validation_never_invokes_real_process_or_creates_state(self) -> None:
        with mock.patch.object(
            self.runner.subprocess,
            "Popen",
            side_effect=AssertionError("real process invocation is forbidden"),
        ):
            self.validator.validate_package()
        self.assertFalse(STATE.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


MAKEFILE_SOURCE = r'''
.PHONY: build test verify check

PYTHON ?= python3

build:
	$(PYTHON) -B -c 'from pathlib import Path; [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in (Path("authority_runner.py"), Path("validate_package.py"), Path("tests/test_cf14_authority.py"))]'

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v

verify:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B validate_package.py

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
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_bytes(path: Path, value: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    path.chmod(mode)


def write_text(path: Path, value: str, mode: int) -> None:
    write_bytes(path, textwrap.dedent(value).lstrip().encode("utf-8"), mode)


def write_json(path: Path, value: object, mode: int = 0o444) -> None:
    write_bytes(path, canonical_bytes(value), mode)


def file_record(path: Path) -> dict[str, object]:
    return {
        "mode": stat.S_IMODE(path.stat().st_mode),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def namespace_manifest(
    path: Path, *, excluded: set[str] | None = None
) -> dict[str, dict[str, object]]:
    excluded = excluded or set()
    return {
        item.relative_to(path).as_posix(): file_record(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.relative_to(path).as_posix() not in excluded
    }


def process_cmdlines() -> list[list[str]]:
    result: list[list[str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if raw:
            result.append(
                [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
            )
    return result


def source_entry(path: Path, relation: str) -> dict[str, object]:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
        scope = "repository"
        stored_path = relative.as_posix()
    except ValueError:
        scope = "external_canonical"
        stored_path = str(path)
    resolved = path.resolve(strict=True)
    record = {
        "path": stored_path,
        "scope": scope,
        "relation": relation,
        **file_record(resolved),
    }
    if path.is_symlink():
        record["symlink_target"] = os.readlink(path)
        record["resolved_path"] = str(resolved)
    return record


def validate_anchors() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    anchors = {
        "cf13_reviewer": CF13_REVIEW,
        "cf13_review_request": CF13 / "review-request.json",
        "cf13_contract": CF13 / "contract.json",
        "cf13_source_manifest": CF13 / "source-constraint-manifest.json",
        "cf12_terminal_seal": CF12_SEAL / "terminal-seal.json",
    }
    for name, path in anchors.items():
        if sha256_file(path) != EXPECTED[name]:
            raise RuntimeError(f"canonical source anchor mismatch: {name}")
    reviewer = load_object(CF13_REVIEW)
    if (
        reviewer.get("producer_role") != "reviewer"
        or reviewer.get("round") != 2
        or reviewer.get("review", {}).get("status") != "done"
    ):
        raise RuntimeError("CF13 canonical Reviewer round-0002 is not accepted")
    request = load_object(CF13 / "review-request.json")
    for relative, digest in request["targets"].items():
        if sha256_file(CF13 / relative) != digest:
            raise RuntimeError(f"CF13 reviewed target changed: {relative}")
    closure = load_object(CF13 / "source-constraint-manifest.json")
    for relative, record in closure["files"].items():
        path = CF13 / relative if record["scope"] == "package" else ROOT / relative
        observed = file_record(path)
        if observed["sha256"] != record["sha256"] or observed["mode"] != record["mode"]:
            raise RuntimeError(f"CF13 source/constraint closure changed: {relative}")
    inventory = load_object(CF13 / "cf12-read-only-inventory.json")
    for relative, record in inventory["entries"].items():
        if file_record(ROOT / relative) != {
            "mode": record["mode"],
            "sha256": record["sha256"],
            "size": record["size"],
        }:
            raise RuntimeError(f"accepted CF12 byte/mode closure changed: {relative}")
    return request, closure, inventory


def collect_source_references(
    request: dict[str, Any],
    closure: dict[str, Any],
    inventory: dict[str, Any],
) -> list[dict[str, object]]:
    references: dict[str, dict[str, object]] = {}

    def add(path: Path, relation: str) -> None:
        key = str(path.resolve())
        if key not in references:
            references[key] = source_entry(path, relation)

    add(Path(__file__).resolve(), "cf14_deterministic_constructor")
    add(CF13_REVIEW, "canonical_cf13_reviewer_round_0002")
    reviewer = load_object(CF13_REVIEW)
    add(Path(reviewer["mission_context"]), "cf13_review_mission")
    add(Path(reviewer["checkpoint"]["path"]), "cf13_reviewer_checkpoint")
    add(Path(reviewer["frontier"]["path"]), "cf13_reviewer_frontier")
    add(CF13 / "review-request.json", "canonical_cf13_review_request")
    add(CF13 / "contract.json", "canonical_cf13_contract")
    add(CF13 / "source-constraint-manifest.json", "canonical_cf13_source_constraint_manifest")
    add(CF13 / "cf12-read-only-inventory.json", "cf13_bound_cf12_inventory")
    for relative in request["targets"]:
        add(CF13 / relative, "cf13_reviewed_package_target")
    for relative, record in closure["files"].items():
        add(
            CF13 / relative if record["scope"] == "package" else ROOT / relative,
            "cf13_transitive_source_constraint_closure",
        )
    for relative in inventory["entries"]:
        add(ROOT / relative, "accepted_cf12_read_only_closure")

    contract = load_object(CF13 / "contract.json")
    add(
        Path(contract["source_terminal_review"]["handoff_path"]),
        "accepted_cf12_terminal_reviewer_handoff",
    )
    add(
        Path(contract["source_terminal_review"]["latest_path"]),
        "accepted_cf12_terminal_reviewer_index",
    )
    add(CF12_SEAL / "terminal-seal.json", "accepted_cf12_terminal_seal")
    terminal = load_object(CF12_SEAL / "terminal-seal.json")
    if sha256_file(CF12_SEAL / "hash-inventory.json") != terminal["hash_inventory_sha256"]:
        raise RuntimeError("CF12 terminal hash inventory changed")
    if sha256_file(CF12_SEAL / "process-death-evidence.json") != terminal["process_death_evidence_sha256"]:
        raise RuntimeError("CF12 process-death evidence changed")
    add(CF12_SEAL / "hash-inventory.json", "cf12_terminal_hash_inventory")
    add(CF12_SEAL / "process-death-evidence.json", "cf12_terminal_process_death_evidence")

    bindings = load_object(CF13 / "bindings.json")
    add(Path(bindings["source_cf12_bindings"]["path"]), "cf13_prompt_model_binding_source")
    add(Path(bindings["accepted_runtime"]["path"]), "cf13_accepted_runtime")
    add(
        Path(bindings["accepted_runtime"]["preflight_hashes_path"]),
        "cf13_accepted_runtime_preflight",
    )
    add(
        Path(bindings["accepted_runtime"]["hardware_export_manifest"]),
        "cf13_hardware_export_manifest",
    )
    endpoint_argv = bindings["endpoint"]["argv_prefix"]
    for index in (0, 2, 4, 6):
        add(Path(endpoint_argv[index]), "cf13_rtl_endpoint_bound_input")
    model_root = Path(bindings["model"]["root"])
    add(model_root / "tokenizer.json", "official_tokenizer")
    add(model_root / "tokenizer_config.json", "official_tokenizer_config")
    final_export = (
        BUILD_ROOT
        / "stage1-w4a8-r6-product-prep-cf10/inputs/final-export-manifest.json"
    )
    if sha256_file(final_export) != bindings["final_export_manifest_sha256"]:
        raise RuntimeError("CF13 final export manifest changed")
    add(final_export, "cf13_final_export_manifest")
    add(CF14_MISSION, "immutable_cf14_review_mission")
    return [references[key] for key in sorted(references)]


def frozen_values() -> dict[str, Any]:
    bindings = load_object(CF13 / "bindings.json")
    source = load_object(Path(bindings["source_cf12_bindings"]["path"]))
    prompt = source["prompt"]
    prompt_bytes = bytes.fromhex(prompt["bytes_hex"])
    if (
        sha256_bytes(prompt_bytes) != prompt["sha256"]
        or prompt["sha256"] != bindings["prompt_binding"]["sha256"]
    ):
        raise RuntimeError("CF13 prompt binding changed")
    launch_argv = [
        str(PYTHON),
        "-B",
        str(PACKAGE / "authority_runner.py"),
        "--authority",
        str(PACKAGE / "authority.json"),
    ]
    product_dir = ATTEMPT / "product"
    product_argv = [
        str(PYTHON),
        "-B",
        str(CF13 / "product_probe.py"),
        "--output",
        str(product_dir / "product-rtl-evidence.json"),
        "--capture",
        str(product_dir / "product-host-error-capture.json"),
        "--endpoint-output",
        str(product_dir / "rtl-completion"),
    ]
    endpoint_argv = list(bindings["endpoint"]["argv_prefix"]) + [
        "--output",
        str(product_dir / "rtl-completion"),
    ]
    return {
        "bindings": bindings,
        "prompt": prompt,
        "prompt_bytes": prompt_bytes,
        "launch_argv": launch_argv,
        "product_argv": product_argv,
        "endpoint_argv": endpoint_argv,
    }


def verify_freshness(values: dict[str, Any]) -> dict[str, object]:
    cf14_names = sorted(
        item.name for item in BUILD_ROOT.iterdir() if "cf14" in item.name.lower()
    )
    if cf14_names or os.path.lexists(PACKAGE) or os.path.lexists(STATE):
        raise RuntimeError(f"prior CF14 namespace exists: {cf14_names}")
    signatures = [
        values["launch_argv"],
        values["product_argv"],
        values["endpoint_argv"],
    ]
    if any(cmdline in signatures for cmdline in process_cmdlines()):
        raise RuntimeError("matching CF14 launch/product/endpoint process already exists")
    return {
        "schema": "ace2-r6-cf14-preconstruction-freshness-inventory-v1",
        "status": "PASS_NO_PRIOR_CF14_AUTHORITY_CLAIM_REGISTRY_ATTEMPT_EVIDENCE_OUTPUT_OR_PROCESS",
        "observed_before_construction": {
            "cf14_build_names": [],
            "authority_namespace": "ABSENT",
            "consumption_claim": "ABSENT",
            "execution_registry": "ABSENT",
            "attempt_namespace": "ABSENT",
            "evidence_namespace": "ABSENT",
            "output_namespace": "ABSENT",
            "matching_launch_process": "ABSENT",
            "matching_product_host_process": "ABSENT",
            "matching_rtl_endpoint_process": "ABSENT",
        },
        "process_match_policy": "EXACT_NUL_SEPARATED_ARGV_EQUALITY",
        "remote_or_reserved_resources": {
            "consumed_s7_amlt_reservation": "NOT_QUERIED_NOT_DEPENDENCY",
            "ssh": "NOT_INVOKED_NOT_DEPENDENCY",
            "remote_infrastructure": "NOT_INVOKED_NOT_DEPENDENCY",
            "cf12_identity": "READ_ONLY_TERMINAL_SEAL_BINDING_ONLY_NOT_REUSED",
            "stage2": "FORBIDDEN",
        },
    }


def build_base(
    target: Path,
    source_references: list[dict[str, object]],
    values: dict[str, Any],
    freshness: dict[str, object],
) -> None:
    target.mkdir(parents=True)
    write_text(target / "authority_runner.py", RUNNER_SOURCE, 0o444)
    write_text(target / "validate_package.py", VALIDATOR_SOURCE, 0o555)
    write_text(target / "tests/test_cf14_authority.py", TEST_SOURCE, 0o444)
    write_text(target / "Makefile", MAKEFILE_SOURCE, 0o444)
    write_bytes(target / "input.txt", values["prompt_bytes"], 0o444)
    (target / "tests").chmod(0o555)

    source_manifest = {
        "schema": "ace2-r6-cf14-source-reference-closure-v1",
        "status": "PASS_AUTHENTICATED_TRANSITIVE_HASH_MODE_SIZE_CLOSURE",
        "anchors": EXPECTED,
        "anchor_paths": {"cf13_reviewer": str(CF13_REVIEW)},
        "files": source_references,
    }
    write_json(target / "source-reference-manifest.json", source_manifest)
    write_json(target / "preconstruction-inventory.json", freshness)

    bindings = values["bindings"]
    model_binding = {
        "model_id": bindings["model"]["model_id"],
        "model_revision": bindings["model"]["revision"],
        "model_root": bindings["model"]["root"],
        "model_sha256": bindings["model"]["model_sha256"],
        "tokenizer_json_sha256": bindings["model"]["tokenizer_json_sha256"],
        "tokenizer_config_sha256": bindings["model"]["tokenizer_config_sha256"],
        "chat_template_token_ids_sha256": bindings["prompt_binding"][
            "chat_template_token_ids_sha256"
        ],
        "final_export_manifest_sha256": bindings["final_export_manifest_sha256"],
        "official_descriptor_sha256": bindings["official_descriptor_sha256"],
        "rtl_source_closure_sha256": bindings["rtl_source_closure_sha256"],
        "w4a8_arithmetic": "UNCHANGED_FROM_ACCEPTED_CF13",
        "public_rtl_contract": "UNCHANGED_14_PARAMETERS_64_PORTS",
        "streaming_memory_boundary": "ABSTRACT_UNCHANGED",
    }
    reference_behavior = {
        "new_reference_process": "FORBIDDEN",
        "accepted_cf12_reference_evidence": "READ_ONLY_CONTEXT_NOT_EXPECTED_OUTPUT",
        "golden_or_hidden_output": "FORBIDDEN",
        "self_derived_expected_tokens": "FORBIDDEN",
    }
    capture_policy = {
        "order": "PRODUCT_HOST_CAPTURE_ATOMIC_FILE_FSYNC_PARENT_FSYNC_BEFORE_RTL_ENDPOINT",
        "capture_persistence_failure": "FAIL_CLOSED_BEFORE_RTL_ENDPOINT",
        "exception_type_message": "EXACT",
        "traceback": "FULL_WITHOUT_LOCALS",
        "context": "BOUNDED_ALLOWLIST_ONLY_MAX_1024_BYTES",
        "prompt_bytes": "EXCLUDED_FROM_CAPTURE",
    }
    timeout_policy = {
        "product_host_process_seconds": 7265,
        "rtl_endpoint_seconds": 7200,
        "signal_grace_seconds": 5,
        "process_tree": "START_NEW_SESSION_SIGTERM_ALL_DISCOVERED_GROUPS_THEN_SIGKILL",
        "timeout_is_terminal": True,
    }
    terminal_policy = {
        "seal_required_after_consumption": True,
        "process_receipt_stdout_stderr_fsync_before_seal": True,
        "partial_evidence_hashes": True,
        "terminal_failure_no_retry": True,
        "seal_path": str(ATTEMPT / "terminal-seal.json"),
    }
    contract = {
        "schema": "ace2-r6-cf14-local-exactly-once-diagnostic-contract-v1",
        "identity": IDENTITY,
        "scope": "LOCAL_CF13_SUCCESSOR_DIAGNOSTIC_ONLY",
        "canonical_sources": EXPECTED,
        "source_reference_manifest_sha256": sha256_file(
            target / "source-reference-manifest.json"
        ),
        "authority_state": "UNCONSUMED",
        "execution_during_construction": "FORBIDDEN",
        "execution_limit_after_new_l2_acceptance": 1,
        "model_tokenizer_w4a8": model_binding,
        "capture_before_endpoint": capture_policy,
        "reference_behavior": reference_behavior,
        "timeout_policy": timeout_policy,
        "terminal_sealing": terminal_policy,
        "no_fallback": "ABSOLUTE",
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN_AFTER_CONSUMPTION",
        "rtl_change": "NONE",
        "consumed_s7_amlt_reservation": "OUT_OF_SCOPE_UNTOUCHED",
        "ssh": "FORBIDDEN",
        "remote_infrastructure": "FORBIDDEN",
        "cf12_identity_reuse": "FORBIDDEN",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
        "non_sram_area_limit_mm2": 2.0,
        "minimum_frequency_mhz": 100,
    }
    write_json(target / "contract.json", contract)

    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    authority = {
        "schema": "ace2-r6-cf14-exactly-once-local-diagnostic-authority-v1",
        "identity": IDENTITY,
        "status": "GRANTED_PENDING_CANONICAL_INDEPENDENT_L2",
        "authorized_action": "EXECUTE_CF13_SUCCESSOR_LOCAL_STAGE1_DIAGNOSTIC_EXACTLY_ONCE",
        "not_authorized": [
            "CONSTRUCTION_TIME_EXECUTION",
            "REFERENCE_REEXECUTION",
            "REMOTE_OR_SSH_EXECUTION",
            "STAGE1_COMPLETION_CLAIM",
            "STAGE2",
        ],
        "authority_cardinality": 1,
        "execution_limit": 1,
        "authority_consumption": "BEFORE_ANY_PROCESS",
        "consumption_evidence_namespace": str(STATE),
        "output_namespace": str(ATTEMPT),
        "output_namespace_must_not_exist_before_consumption": True,
        "transaction_nonce": sha256_bytes(
            f"{IDENTITY}:transaction-v1".encode("ascii")
        ),
        "durable_task_id": "ace2-r6-cf14-product-host-diagnostic-once-0001",
        "activation_gate": {
            "required_level": "INDEPENDENT_L2",
            "status": "PENDING",
            "mission_id": "r6generationprobeauthority14",
            "mission_path": str(CF14_MISSION),
            "mission_sha256": sha256_file(CF14_MISSION),
            "review_root": str(CF14_REVIEW_ROOT),
            "activation_before_reviewer_done_handoff": "FORBIDDEN",
        },
        "canonical_sources": EXPECTED,
        "contract_sha256": sha256_file(target / "contract.json"),
        "source_reference_manifest_sha256": sha256_file(
            target / "source-reference-manifest.json"
        ),
        "launch_invocation": {
            "argv": values["launch_argv"],
            "argv_sha256": sha256_bytes(canonical_bytes(values["launch_argv"])),
            "local_only": True,
        },
        "prompt": {
            "bytes_hex": values["prompt_bytes"].hex(),
            "byte_count": len(values["prompt_bytes"]),
            "sha256": sha256_bytes(values["prompt_bytes"]),
            "chat_template_token_ids_sha256": values["prompt"][
                "chat_template_token_ids_sha256"
            ],
            "token_count": len(values["prompt"]["chat_template_token_ids"]),
        },
        "model_tokenizer_w4a8": model_binding,
        "product_host_invocation": {
            "argv": values["product_argv"],
            "argv_sha256": sha256_bytes(canonical_bytes(values["product_argv"])),
            "capture_before_endpoint": capture_policy,
        },
        "rtl_endpoint_invocation": {
            "argv": values["endpoint_argv"],
            "argv_sha256": sha256_bytes(canonical_bytes(values["endpoint_argv"])),
            "software_fallback": "FORBIDDEN",
        },
        "reference_behavior": reference_behavior,
        "environment": environment,
        "environment_sha256": sha256_bytes(canonical_bytes(environment)),
        "timeout_policy": timeout_policy,
        "terminal_sealing": terminal_policy,
        "retry_replay_resume_relaunch": "PERMANENTLY_FORBIDDEN",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "stage_transition": "DISABLED",
    }
    write_json(target / "authority.json", authority)
    construction = {
        "schema": "ace2-r6-cf14-authority-construction-report-v1",
        "status": "PASS_FRESH_ADDITIVE_UNCONSUMED_AUTHORITY_ONLY_ZERO_EXECUTION",
        "authority_count": 1,
        "claim_count": 0,
        "registry_count": 0,
        "attempt_count": 0,
        "evidence_namespace_count": 0,
        "matching_process_count": 0,
        "model_invocation_count": 0,
        "tokenizer_generation_count": 0,
        "product_host_invocation_count": 0,
        "rtl_endpoint_invocation_count": 0,
        "simulator_invocation_count": 0,
        "submit_invocation_count": 0,
        "preconstruction_inventory_sha256": sha256_file(
            target / "preconstruction-inventory.json"
        ),
        "source_reference_manifest_sha256": sha256_file(
            target / "source-reference-manifest.json"
        ),
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
    }
    write_json(target / "construction-report.json", construction)
    write_json(
        target / "construction-repair-record.json",
        {
            "schema": "ace2-r6-cf14-construction-repair-record-v1",
            "attempts": [
                {
                    "attempt": 1,
                    "failure_taxonomy": "SOURCE_REFERENCE_SYMLINK_POLICY_MISMATCH",
                    "root_cause_hypothesis": "SNAPSHOT_SYMLINK_TARGET_BYTES_WERE_HASHED_BUT_LINK_IDENTITY_WAS_NOT_MANIFESTED",
                    "regression": "SOURCE_REFERENCES_BIND_LINK_TEXT_RESOLVED_TARGET_AND_TARGET_BYTES_MODE_SIZE",
                    "real_model_product_rtl_simulator_submit_execution": 0,
                }
            ],
        },
    )


def finalize_package(target: Path, reproducibility: dict[str, object]) -> None:
    write_json(target / "reproducibility-report.json", reproducibility)
    excluded = {
        "package-manifest.json",
        "package-manifest.sha256",
        "review-request.json",
        "review-request.sha256",
    }
    package_manifest = {
        "schema": "ace2-r6-cf14-package-content-manifest-v1",
        "status": "PASS_STATIC_BYTE_MODE_SIZE_CLOSURE",
        "files": namespace_manifest(target, excluded=excluded),
    }
    write_json(target / "package-manifest.json", package_manifest)
    write_text(
        target / "package-manifest.sha256",
        f"{sha256_file(target / 'package-manifest.json')}  package-manifest.json\n",
        0o444,
    )
    request_targets = namespace_manifest(
        target, excluded={"review-request.json", "review-request.sha256"}
    )
    review_request = {
        "schema": "ace2-r6-cf14-canonical-independent-l2-review-request-v1",
        "status": "PENDING_INDEPENDENT_L2",
        "mission_id": "r6generationprobeauthority14",
        "activation_before_acceptance": "FORBIDDEN",
        "authority_consumption_during_review": "FORBIDDEN",
        "targets": request_targets,
        "required_checks": [
            "CANONICAL_CF13_REVIEW_AND_TRANSITIVE_CLOSURE",
            "CF12_TERMINAL_SEAL_READ_ONLY_BINDING",
            "FRESH_NO_CF14_STATE_OR_MATCHING_PROCESS",
            "EXACT_LOCAL_LAUNCH_PRODUCT_HOST_AND_RTL_ENDPOINT_ARGV",
            "PROMPT_MODEL_TOKENIZER_W4A8_HASH_BINDING",
            "CAPTURE_FSYNC_BEFORE_ENDPOINT_AND_PERSISTENCE_FAIL_CLOSED",
            "PROCESS_TREE_TIMEOUT_SIGTERM_THEN_SIGKILL",
            "NO_REFERENCE_FALLBACK_RETRY_REPLAY_RESUME_RELAUNCH",
            "MOCK_ONLY_SAFETY_PROOFS_ZERO_REAL_EXECUTION",
            "TERMINAL_SEAL_AFTER_CONSUMPTION",
            "STAGE1_OPEN_STAGE2_FORBIDDEN",
        ],
    }
    write_json(target / "review-request.json", review_request)
    write_text(
        target / "review-request.sha256",
        f"{sha256_file(target / 'review-request.json')}  review-request.json\n",
        0o444,
    )
    for directory in sorted(
        (path for path in target.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    target.chmod(0o555)


def main() -> int:
    request, closure, inventory = validate_anchors()
    if sha256_file(PYTHON) != "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad":
        raise RuntimeError("frozen local Python executable changed")
    values = frozen_values()
    freshness = verify_freshness(values)
    references = collect_source_references(request, closure, inventory)

    with tempfile.TemporaryDirectory(prefix=".cf14-build-", dir=BUILD_ROOT) as temporary:
        root = Path(temporary)
        first = root / "first" / IDENTITY
        second = root / "second" / IDENTITY
        build_base(first, references, values, freshness)
        build_base(second, references, values, freshness)
        first_base = namespace_manifest(first)
        second_base = namespace_manifest(second)
        if first_base != second_base:
            raise RuntimeError("CF14 isolated base builds differ")
        reproducibility = {
            "schema": "ace2-r6-cf14-two-build-reproducibility-v1",
            "status": "PASS_TWO_ISOLATED_BUILDS_BYTE_MODE_SIZE_EQUAL",
            "build_count": 2,
            "base_entry_count": len(first_base),
            "base_manifest_sha256": sha256_bytes(canonical_bytes(first_base)),
        }
        finalize_package(first, reproducibility)
        finalize_package(second, reproducibility)
        if namespace_manifest(first) != namespace_manifest(second):
            raise RuntimeError("CF14 complete isolated builds differ")
        shutil.copytree(first, PACKAGE, copy_function=shutil.copy2)
    PACKAGE.chmod(0o555)
    print(
        json.dumps(
            {
                "identity": IDENTITY,
                "package": str(PACKAGE),
                "review_request_sha256": sha256_file(PACKAGE / "review-request.json"),
                "source_reference_count": len(references),
                "status": "CONSTRUCTED_UNCONSUMED_PENDING_INDEPENDENT_L2",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
