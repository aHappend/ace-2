#!/usr/bin/env python3
"""Candidate-independent, synthetic-only exactly-once validation harness.

The harness intentionally has no arbitrary command or subprocess execution
surface.  It accepts one versioned synthetic JSON operation, checks exact input
and authority provenance before reserving a run, durably records execution, and
atomically publishes an artifact/hash bundle as one renamed directory.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
ENTRYPOINT_KIND = "synthetic_fixture_v1"
OPERATION = "canonical_json_digest_v1"
REQUIRED_STAGE = "architecture"
ARTIFACT_NAME = "ARTIFACT.json"
COMPANION_NAME = "ARTIFACT.sha256"
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
ALLOWED_SPEC_KEYS = {
    "schema_version",
    "run_id",
    "entrypoint_kind",
    "operation",
    "fixture",
    "authority",
    "provenance",
}
ALLOWED_FIXTURE_KEYS = {"path", "bytes", "sha256"}
ALLOWED_AUTHORITY_KEYS = {"path", "bytes", "sha256", "required_stage"}
ALLOWED_PROVENANCE_KEYS = {"contract_id", "harness_sha256"}
FORBIDDEN_PATH_PARTS = {
    "rtl",
    "reference",
    "references",
    "verification",
    "vector",
    "vectors",
    "model",
    "models",
    "candidate",
    "candidates",
}
FORBIDDEN_NAME_FRAGMENTS = ("candidate", "model", "precheck")
FAILPOINTS = {
    "after_reservation",
    "after_execution_mark",
    "before_bundle_publish",
    "after_bundle_publish",
    "after_ledger_commit",
}


class HarnessError(RuntimeError):
    """A fail-closed harness rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot load JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_durable(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def exact_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    require(set(value) == allowed, f"{label} keys must be exactly {sorted(allowed)}")


class SyntheticBaselineHarness:
    """Durable synthetic runner with no candidate/model execution surface."""

    def __init__(
        self,
        *,
        synthetic_root: Path,
        ledger_path: Path,
        output_root: Path,
    ) -> None:
        self.synthetic_root = synthetic_root.resolve()
        self.ledger_path = ledger_path.resolve()
        self.output_root = output_root.resolve()
        self.lock_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        self.harness_path = Path(__file__).resolve()

    def _resolve_synthetic_path(self, relative: str, label: str) -> Path:
        require(isinstance(relative, str) and relative, f"{label} path is required")
        candidate = Path(relative)
        require(not candidate.is_absolute(), f"{label} path must be relative")
        require(".." not in candidate.parts, f"{label} path traversal is forbidden")
        lowered_parts = {part.lower() for part in candidate.parts}
        require(
            not lowered_parts.intersection(FORBIDDEN_PATH_PARTS),
            f"{label} path names a protected/candidate surface",
        )
        lowered_name = candidate.name.lower()
        require(
            not any(fragment in lowered_name for fragment in FORBIDDEN_NAME_FRAGMENTS),
            f"{label} filename names a protected/candidate surface",
        )
        resolved = (self.synthetic_root / candidate).resolve()
        require(
            resolved == self.synthetic_root or self.synthetic_root in resolved.parents,
            f"{label} escapes the synthetic root",
        )
        return resolved

    def _validate_bound_file(
        self,
        binding: dict[str, Any],
        allowed_keys: set[str],
        label: str,
    ) -> tuple[Path, dict[str, Any]]:
        require(isinstance(binding, dict), f"{label} binding must be an object")
        exact_keys(binding, allowed_keys, f"{label} binding")
        path = self._resolve_synthetic_path(binding["path"], label)
        require(path.is_file(), f"missing {label}: {path}")
        require(path.suffix == ".json", f"{label} must be JSON")
        require(path.stat().st_size == binding["bytes"], f"{label} byte count mismatch")
        require(sha256_file(path) == binding["sha256"], f"{label} SHA-256 mismatch")
        return path, load_json(path)

    def validate_spec(self, spec_path: Path) -> dict[str, Any]:
        spec = load_json(spec_path)
        exact_keys(spec, ALLOWED_SPEC_KEYS, "run spec")
        require(spec["schema_version"] == SCHEMA_VERSION, "unsupported run spec schema")
        require(
            isinstance(spec["run_id"], str) and RUN_ID_RE.fullmatch(spec["run_id"]),
            "invalid run_id",
        )
        require(spec["entrypoint_kind"] == ENTRYPOINT_KIND, "candidate/unknown entrypoint rejected")
        require(spec["operation"] == OPERATION, "unknown synthetic operation rejected")

        provenance = spec["provenance"]
        require(isinstance(provenance, dict), "provenance must be an object")
        exact_keys(provenance, ALLOWED_PROVENANCE_KEYS, "provenance")
        require(
            isinstance(provenance["contract_id"], str) and provenance["contract_id"],
            "contract_id is required",
        )
        require(
            provenance["harness_sha256"] == sha256_file(self.harness_path),
            "harness source provenance mismatch",
        )

        fixture_path, fixture = self._validate_bound_file(
            spec["fixture"], ALLOWED_FIXTURE_KEYS, "fixture"
        )
        authority_path, authority = self._validate_bound_file(
            spec["authority"], ALLOWED_AUTHORITY_KEYS, "authority"
        )
        require(
            spec["authority"]["required_stage"] == REQUIRED_STAGE,
            "only architecture-stage synthetic hardening is authorized",
        )
        require(authority.get("current_stage") == REQUIRED_STAGE, "stage authorization denied")
        require(
            authority.get("synthetic_harness_execution_authorized") is True,
            "synthetic harness execution is not authorized",
        )
        require(authority.get("dry_fixture_only") is True, "authority is not dry-fixture-only")
        require(
            authority.get("candidate_or_model_execution_permitted") is False,
            "authority permits candidate/model execution",
        )
        require(
            set(fixture) == {"schema_version", "payload"}
            and fixture.get("schema_version") == SCHEMA_VERSION,
            "fixture schema must be exactly synthetic payload v1",
        )

        return {
            "spec": spec,
            "spec_path": spec_path.resolve(),
            "spec_sha256": sha256_file(spec_path),
            "fixture_path": fixture_path,
            "fixture": fixture,
            "fixture_sha256": spec["fixture"]["sha256"],
            "authority_path": authority_path,
            "authority": authority,
            "authority_sha256": spec["authority"]["sha256"],
            "harness_sha256": provenance["harness_sha256"],
        }

    @contextlib.contextmanager
    def _locked_ledger(self) -> Iterator[dict[str, Any]]:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.ledger_path.exists():
                ledger = load_json(self.ledger_path)
                require(ledger.get("schema_version") == SCHEMA_VERSION, "ledger schema mismatch")
                require(isinstance(ledger.get("runs"), dict), "ledger runs must be an object")
            else:
                ledger = {"schema_version": SCHEMA_VERSION, "runs": {}}
            yield ledger

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        atomic_write_json(self.ledger_path, ledger)

    def _failpoint(self, requested: str | None, point: str) -> None:
        if requested != point:
            return
        require(
            os.environ.get("ACE2_SYNTHETIC_HARNESS_TEST_MODE") == "1",
            "failpoints require explicit synthetic test mode",
        )
        os._exit(97)

    def _bundle_path(self, run_id: str) -> Path:
        return self.output_root / f"{run_id}.bundle"

    def _execute_synthetic(self, validated: dict[str, Any]) -> dict[str, Any]:
        payload = validated["fixture"]["payload"]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": validated["spec"]["run_id"],
            "entrypoint_kind": ENTRYPOINT_KIND,
            "operation": OPERATION,
            "result": {
                "canonical_payload_bytes": len(canonical_json_bytes(payload)),
                "canonical_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            "provenance": {
                "contract_id": validated["spec"]["provenance"]["contract_id"],
                "spec_sha256": validated["spec_sha256"],
                "fixture_sha256": validated["fixture_sha256"],
                "authority_sha256": validated["authority_sha256"],
                "harness_sha256": validated["harness_sha256"],
                "authorized_stage": REQUIRED_STAGE,
            },
        }

    def _publish_bundle(
        self,
        artifact: dict[str, Any],
        *,
        failpoint: str | None,
    ) -> tuple[Path, str]:
        run_id = artifact["run_id"]
        self.output_root.mkdir(parents=True, exist_ok=True)
        fsync_directory(self.output_root.parent)
        final = self._bundle_path(run_id)
        require(not final.exists(), f"published bundle already exists: {final}")
        staging = self.output_root / f".staging-{run_id}-{os.getpid()}"
        require(not staging.exists(), f"staging collision: {staging}")
        staging.mkdir()
        artifact_path = staging / ARTIFACT_NAME
        artifact_bytes = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        artifact_sha256 = sha256_bytes(artifact_bytes)
        write_durable(artifact_path, artifact_bytes)
        write_durable(
            staging / COMPANION_NAME,
            f"{artifact_sha256}  {ARTIFACT_NAME}\n".encode("ascii"),
        )
        fsync_directory(staging)
        self._failpoint(failpoint, "before_bundle_publish")
        os.replace(staging, final)
        fsync_directory(self.output_root)
        return final, artifact_sha256

    def _verify_bundle(
        self,
        record: dict[str, Any],
        *,
        require_ledger_hash: bool,
    ) -> dict[str, Any]:
        final = self._bundle_path(record["run_id"])
        require(final.is_dir(), f"published bundle missing: {final}")
        require(
            {entry.name for entry in final.iterdir()} == {ARTIFACT_NAME, COMPANION_NAME},
            "bundle contents are incomplete or unexpected",
        )
        artifact_path = final / ARTIFACT_NAME
        companion_path = final / COMPANION_NAME
        fields = companion_path.read_text(encoding="ascii").strip().split()
        require(len(fields) == 2, "invalid artifact companion hash")
        require(fields[1] == ARTIFACT_NAME, "companion names the wrong artifact")
        actual_sha256 = sha256_file(artifact_path)
        require(fields[0] == actual_sha256, "artifact companion hash mismatch")
        if require_ledger_hash:
            require(record.get("artifact_sha256") == actual_sha256, "ledger artifact hash mismatch")
        artifact = load_json(artifact_path)
        require(artifact.get("run_id") == record["run_id"], "artifact run_id mismatch")
        provenance = artifact.get("provenance", {})
        require(provenance.get("spec_sha256") == record["spec_sha256"], "artifact spec mismatch")
        require(
            provenance.get("fixture_sha256") == record["fixture_sha256"],
            "artifact fixture mismatch",
        )
        require(
            provenance.get("authority_sha256") == record["authority_sha256"],
            "artifact authority mismatch",
        )
        require(
            provenance.get("harness_sha256") == record["harness_sha256"],
            "artifact harness mismatch",
        )
        return {
            "bundle": final,
            "artifact": artifact_path,
            "artifact_sha256": actual_sha256,
        }

    def _new_record(self, validated: dict[str, Any]) -> dict[str, Any]:
        spec = validated["spec"]
        return {
            "run_id": spec["run_id"],
            "state": "reserved",
            "reserved_at_utc": now_utc(),
            "execution_count": 0,
            "entrypoint_kind": ENTRYPOINT_KIND,
            "operation": OPERATION,
            "contract_id": spec["provenance"]["contract_id"],
            "spec_sha256": validated["spec_sha256"],
            "fixture_sha256": validated["fixture_sha256"],
            "authority_sha256": validated["authority_sha256"],
            "harness_sha256": validated["harness_sha256"],
        }

    def _execute_record_locked(
        self,
        ledger: dict[str, Any],
        validated: dict[str, Any],
        *,
        failpoint: str | None,
    ) -> dict[str, Any]:
        run_id = validated["spec"]["run_id"]
        record = ledger["runs"][run_id]
        require(record["state"] == "reserved", "run is not safely resumable")
        require(record["execution_count"] == 0, "run execution was already attempted")
        require(record["spec_sha256"] == validated["spec_sha256"], "reservation spec mismatch")
        record["state"] = "running"
        record["execution_count"] = 1
        record["execution_started_at_utc"] = now_utc()
        self._save_ledger(ledger)
        self._failpoint(failpoint, "after_execution_mark")

        artifact = self._execute_synthetic(validated)
        final, artifact_sha256 = self._publish_bundle(artifact, failpoint=failpoint)
        self._failpoint(failpoint, "after_bundle_publish")
        record["state"] = "committed"
        record["committed_at_utc"] = now_utc()
        record["artifact_sha256"] = artifact_sha256
        record["bundle_name"] = final.name
        self._save_ledger(ledger)
        self._failpoint(failpoint, "after_ledger_commit")
        return self._verify_bundle(record, require_ledger_hash=True)

    def run(self, spec_path: Path, *, failpoint: str | None = None) -> dict[str, Any]:
        require(failpoint in FAILPOINTS or failpoint is None, "unknown failpoint")
        validated = self.validate_spec(spec_path)
        run_id = validated["spec"]["run_id"]
        with self._locked_ledger() as ledger:
            require(run_id not in ledger["runs"], "run_id already has a durable reservation")
            ledger["runs"][run_id] = self._new_record(validated)
            self._save_ledger(ledger)
            self._failpoint(failpoint, "after_reservation")
            return self._execute_record_locked(ledger, validated, failpoint=failpoint)

    def recover(self, spec_path: Path, *, failpoint: str | None = None) -> dict[str, Any]:
        require(failpoint in FAILPOINTS or failpoint is None, "unknown failpoint")
        spec = load_json(spec_path)
        require(isinstance(spec.get("run_id"), str), "recovery spec lacks run_id")
        run_id = spec["run_id"]
        with self._locked_ledger() as ledger:
            require(run_id in ledger["runs"], "no durable reservation to recover")
            record = ledger["runs"][run_id]
            require(record["spec_sha256"] == sha256_file(spec_path), "recovery spec mismatch")
            if record["state"] == "committed":
                return self._verify_bundle(record, require_ledger_hash=True)
            if record["state"] == "reserved":
                validated = self.validate_spec(spec_path)
                return self._execute_record_locked(ledger, validated, failpoint=failpoint)
            if record["state"] == "running":
                try:
                    verified = self._verify_bundle(record, require_ledger_hash=False)
                except HarnessError as exc:
                    record["state"] = "interrupted_fail_closed"
                    record["recovery_error"] = str(exc)
                    record["recovery_checked_at_utc"] = now_utc()
                    self._save_ledger(ledger)
                    raise HarnessError(
                        "execution had started without a valid published bundle; rerun forbidden"
                    ) from exc
                record["state"] = "committed"
                record["committed_at_utc"] = now_utc()
                record["recovered_after_crash"] = True
                record["artifact_sha256"] = verified["artifact_sha256"]
                record["bundle_name"] = verified["bundle"].name
                self._save_ledger(ledger)
                return self._verify_bundle(record, require_ledger_hash=True)
            raise HarnessError(f"run state is permanently fail-closed: {record['state']}")

    def verify(self, run_id: str) -> dict[str, Any]:
        require(RUN_ID_RE.fullmatch(run_id) is not None, "invalid run_id")
        with self._locked_ledger() as ledger:
            require(run_id in ledger["runs"], "run_id is not in the ledger")
            record = ledger["runs"][run_id]
            require(record["state"] == "committed", "run is not committed")
            return self._verify_bundle(record, require_ledger_hash=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "recover"):
        child = subparsers.add_parser(command)
        child.add_argument("--spec", type=Path, required=True)
        child.add_argument("--failpoint", choices=sorted(FAILPOINTS))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    harness = SyntheticBaselineHarness(
        synthetic_root=args.synthetic_root,
        ledger_path=args.ledger,
        output_root=args.output_root,
    )
    try:
        if args.command == "run":
            result = harness.run(args.spec, failpoint=args.failpoint)
        elif args.command == "recover":
            result = harness.recover(args.spec, failpoint=args.failpoint)
        else:
            result = harness.verify(args.run_id)
    except HarnessError as exc:
        print(f"SYNTHETIC_BASELINE_HARNESS_FAIL {exc}", file=sys.stderr)
        return 2
    print(
        "SYNTHETIC_BASELINE_HARNESS_PASS "
        f"artifact_sha256={result['artifact_sha256']} bundle={result['bundle']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
