#!/usr/bin/env python3
"""Build the immutable Base V11 canonical-byte read-only-preflight successor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
V9_ROOT = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root"
V11_ROOT = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v11_shellfree_action_root"
V8_STATIC_ROOT = ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root"
VERIFIER_SOURCE = ROOT / "tools/v11_canonical_preflight_verify_independent.py"
ACTION_ID = "ace2:qk-gbfp8-base-v11:execute-once:bb6a7653:20260814T075633Z"
V9_ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
OLD_V8_ACTION_IDS = [
    "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1610Z",
    "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1614Z",
    "ace2:qk-gbfp8-base-v8:execute-once:2254dd91:20260813T1724Z",
]
PACKAGE_PATH = V11_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_PACKAGE.json"
LAUNCHER_PATH = V11_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v11.py"
TRANSPORT_PATH = V11_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v11.py"
STATIC_VERIFIER_PATH = V11_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v11_shellfree.py"
REPORT_PATH = V11_ROOT / "V9_CANONICAL_BYTE_FAILURE_REPORT.md"
CANONICAL_PACKAGE = V11_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
CANONICAL_RESULT_SCHEMA = V11_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json"
SOURCE_PACKAGE = V8_STATIC_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json"
SOURCE_RESULT_SCHEMA = V8_STATIC_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json"
V9_TERMINAL = V9_ROOT / "live/authority/base/first-terminal.json"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_object(raw: bytes) -> dict[str, Any]:
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(raw.decode("ascii", "strict"), object_pairs_hook=pairs)
    require(type(value) is dict and not duplicate, "source JSON shape")
    return value


def replace_v9(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("V9", "V11").replace("v9", "v11").replace(
            "ace2:qk-gbfp8-base-v11:execute-once:2254dd91:20260813T2013Z",
            ACTION_ID,
        )
    if isinstance(value, list):
        return [replace_v9(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_v9(item) for key, item in value.items()}
    return value


def write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def invocation(argv: list[str]) -> dict[str, Any]:
    base = {
        "argv": argv,
        "command_representation": "ARGV_VECTOR_ONLY",
        "cwd": str(V11_ROOT),
        "environment": EXACT_ENVIRONMENT,
        "shell": False,
    }
    return base | {"invocation_sha256": sha256_bytes(compact_bytes(base))}


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def canonical_binding(binding_id: str, path: Path) -> dict[str, Any]:
    return {"id": binding_id, "path": str(path), "sha256": sha256_file(path)}


def local_binding(binding_id: str, path: Path) -> dict[str, Any]:
    return canonical_binding(binding_id, path)


def main() -> int:
    require(not V11_ROOT.exists(), f"refusing to overwrite successor root: {V11_ROOT}")
    require(VERIFIER_SOURCE.is_file(), "independent verifier source is absent")
    require(V9_TERMINAL.is_file(), "retired V9 terminal is absent")

    source_package_raw = SOURCE_PACKAGE.read_bytes()
    source_schema_raw = SOURCE_RESULT_SCHEMA.read_bytes()
    source_package = strict_object(source_package_raw)
    source_schema = strict_object(source_schema_raw)
    canonical_package_raw = compact_bytes(source_package)
    canonical_schema_raw = compact_bytes(source_schema)
    require(canonical_package_raw != source_package_raw, "V8 package unexpectedly already canonical")
    require(canonical_schema_raw != source_schema_raw, "V8 schema unexpectedly already canonical")
    write_bytes(CANONICAL_PACKAGE, canonical_package_raw)
    write_bytes(CANONICAL_RESULT_SCHEMA, canonical_schema_raw)

    v9_schema_dir = V9_ROOT / "reference"
    v11_schema_dir = V11_ROOT / "reference"
    schema_names = (
        "AUTHORITY_SCHEMA",
        "CREDENTIAL_SCHEMA",
        "FIRST_TERMINAL_SCHEMA",
        "FRESH_L2_ACCEPTANCE_SCHEMA",
        "LEDGER_SCHEMA",
    )
    schema_paths: dict[str, Path] = {}
    for name in schema_names:
        source = v9_schema_dir / f"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_{name}.json"
        target = v11_schema_dir / f"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_{name}.json"
        write_bytes(target, compact_bytes(replace_v9(strict_object(source.read_bytes()))))
        schema_paths[name] = target

    report = f"""# Base V9 canonical-byte preflight failure and V11 additive repair

Date: 2026-08-14

The retired action `{V9_ACTION_ID}` is sealed by its immutable
`PREFLIGHT_FAILED_TERMINAL/READ_ONLY_PREFLIGHT_FAILED` record.  Its reproduced
failure detail is `ExecutionError:accepted V8 package: noncanonical bytes`.
No controller, evaluator, C02 parser, tensor payload, or G8/G4/G2/G1 run
occurred.  V9 must never be retried, replayed, resumed, repaired, or replaced
in place.

Measured byte cause:

- accepted V8 package predecessor: {len(source_package_raw)} bytes, SHA-256 `{sha256_bytes(source_package_raw)}`;
- exact production-canonical package mirror: {len(canonical_package_raw)} bytes, SHA-256 `{sha256_bytes(canonical_package_raw)}`;
- accepted V8 result-schema predecessor: {len(source_schema_raw)} bytes, SHA-256 `{sha256_bytes(source_schema_raw)}`;
- exact production-canonical result-schema mirror: {len(canonical_schema_raw)} bytes, SHA-256 `{sha256_bytes(canonical_schema_raw)}`.

The predecessor objects are semantically equal to their mirrors.  Only JSON
serialization changes: sorted keys, compact separators, strict ASCII, and one
terminal newline.  All predecessor roots and terminal evidence remain
byte-unchanged.

The additive successor action is `{ACTION_ID}`.  Its production launcher keeps
the exact V9 canonical reader and points the accepted package/schema bindings
to the new canonical mirrors.  The independent verifier extracts those exact
reader function bodies by AST, runs the exact file read on both bound mirrors,
checks the read-only-preflight call occurs before any durable live-state create,
and rejects adversarial noncanonical encodings.  Static review grants no
execution authority; a later separately authorized mission is required for any
transport or launcher invocation.
"""
    write_bytes(REPORT_PATH, report.encode("utf-8"))

    v9_launcher = (V9_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v9.py").read_text(encoding="utf-8")
    launcher = v9_launcher.replace("V9", "V11").replace("v9", "v11")
    launcher = launcher.replace(
        'ACTION_ID = "ace2:qk-gbfp8-base-v11:execute-once:2254dd91:20260813T2013Z"',
        f'ACTION_ID = "{ACTION_ID}"',
    )
    launcher = launcher.replace(
        'ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v11_shellfree_action_root")\n',
        'ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v11_shellfree_action_root")\n'
        'V9_ROOT = Path("/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root")\n',
    )
    launcher = launcher.replace(
        f'ACTION_ID = "{ACTION_ID}"\n',
        f'ACTION_ID = "{ACTION_ID}"\nV9_ACTION_ID = "{V9_ACTION_ID}"\n',
    )

    accepted_paths = f'''ACCEPTED_PATHS = {{
    "accepted_v8_handoff": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json"),
    "accepted_r2_handoff": Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b8v8execpkg02/round-0003.json"),
    "accepted_v8_manifest": ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json",
    "runtime_package": ACCEPTED_V8_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json",
    "package_predecessor_raw": ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.json",
    "result_schema_predecessor_raw": ACCEPTED_V8_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.json",
    "package": ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json",
    "result_schema": ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json",
    "controller": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py",
    "evaluator": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py",
    "c02_parser": ACCEPTED_V8_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py",
    "decisive_verifier": ACCEPTED_V8_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py",
    "v9_first_terminal": V9_ROOT / "live/authority/base/first-terminal.json",
    **V8_PATHS,
    **V7_PATHS,
}}
'''
    launcher = replace_block(launcher, "ACCEPTED_PATHS = {", "EXPECTED_BINDING_HASHES = {", accepted_paths)

    expected_paths = [
        ("accepted_v8_handoff", Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/f0fa5c269681/round-0007.json")),
        ("accepted_r2_handoff", Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/b8v8execpkg02/round-0003.json")),
        ("accepted_v8_manifest", V8_STATIC_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_MANIFEST.json"),
        ("runtime_package", V8_STATIC_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RUNTIME_PACKAGE.json"),
        ("package_predecessor_raw", SOURCE_PACKAGE),
        ("result_schema_predecessor_raw", SOURCE_RESULT_SCHEMA),
        ("package", CANONICAL_PACKAGE),
        ("result_schema", CANONICAL_RESULT_SCHEMA),
        ("controller", V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_controller_static_v8.py"),
        ("evaluator", V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"),
        ("c02_parser", V8_STATIC_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"),
        ("decisive_verifier", V8_STATIC_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_static_v8.py"),
        ("v9_first_terminal", V9_TERMINAL),
        ("v8_authority", ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root/live/authority/base/authority.json"),
        ("v8_consumed_ledger", ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root/live/authority/base/authority-ledger.json"),
        ("v8_first_terminal", ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_shellfree_action_root/live/authority/base/first-terminal.json"),
        ("v7_consumed_ledger", ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root/live/authority/base/authority-ledger.json"),
        ("v7_first_terminal", ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v8_action_root/live/authority/base/first-terminal.json"),
    ]
    expected_hashes = "EXPECTED_BINDING_HASHES = {\n" + "".join(
        f'    "{binding_id}": "{sha256_file(path)}",\n' for binding_id, path in expected_paths
    ) + "}\n"
    launcher = replace_block(launcher, "EXPECTED_BINDING_HASHES = {", "EXPECTED_BINDING_ORDER =", expected_hashes)
    launcher = launcher.replace(
        '    "forensic_report": ROOT / "V8_FORENSIC_REPORT.md",',
        '    "forensic_report": ROOT / "V9_CANONICAL_BYTE_FAILURE_REPORT.md",',
    )

    old_identity = '''    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": V8_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, V7_ACTION_ID, V8_ACTION_ID],
        "reuse_permitted": False,
    }, "action identity")'''
    new_identity = '''    require(package["action_identity"] == {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": V9_ACTION_ID,
        "retired_action_ids": [RETIRED_PREPACKAGE_ACTION_ID, V7_ACTION_ID, V8_ACTION_ID, V9_ACTION_ID],
        "reuse_permitted": False,
    }, "action identity")'''
    require(old_identity in launcher, "launcher action identity block not found")
    launcher = launcher.replace(old_identity, new_identity)
    old_prior = '''    require(package["prior_terminal_state"] == {
        "action_id": V8_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "result_file_sha256": None,
        "status": "CONSUMED_ORPHAN",
    }, "prior terminal declaration")'''
    new_prior = '''    require(package["prior_terminal_state"] == {
        "action_id": V9_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "reason_code": "READ_ONLY_PREFLIGHT_FAILED",
        "result_file_sha256": None,
        "status": "PREFLIGHT_FAILED_TERMINAL",
    }, "prior terminal declaration")'''
    require(old_prior in launcher, "launcher prior terminal block not found")
    launcher = launcher.replace(old_prior, new_prior)
    retained_marker = "def _verify_retained_terminals() -> None:\n"
    retained_prefix = '''def _verify_retained_terminals() -> None:
    v9_terminal, _ = read_canonical_json(ACCEPTED_PATHS["v9_first_terminal"], "V9 first terminal")
    verify_self_checksum(v9_terminal, "first_terminal_sha256", "V9 first terminal")
    require(v9_terminal["action_id"] == V9_ACTION_ID, "V9 terminal action")
    require(v9_terminal["status"] == "PREFLIGHT_FAILED_TERMINAL" and v9_terminal["reason_code"] == "READ_ONLY_PREFLIGHT_FAILED", "V9 terminal status")
    require(v9_terminal["failure_detail_sha256"] == "7fe1602a1cd89139fc712070f5961dce30d1bc5be2f13153130498de18661373", "V9 failure detail")
    require(v9_terminal["invocation_count_performed"] == 0 and v9_terminal["payload_open_count"] == 0, "V9 no-execution counters")
    require(v9_terminal["result_file_sha256"] is None and v9_terminal["retry_replay_resume_repair_replacement_permitted"] is False, "V9 no-replay terminal")
    for relative in ("live/authority/base/authority.json", "live/authority/base/credential.json", "live/authority/base/authority-ledger.json", "live/result/base/result.json"):
        require(not os.path.lexists(V9_ROOT / relative), f"unexpected V9 state: {relative}")
'''
    require(retained_marker in launcher, "retained terminal function not found")
    launcher = launcher.replace(retained_marker, retained_prefix, 1)
    write_bytes(LAUNCHER_PATH, launcher.encode("utf-8"))

    v9_transport = (V9_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v9.py").read_text(encoding="utf-8")
    transport = v9_transport.replace("V9", "V11").replace("v9", "v11")
    transport = transport.replace(
        'ACTION_ID = "ace2:qk-gbfp8-base-v11:execute-once:2254dd91:20260813T2013Z"',
        f'ACTION_ID = "{ACTION_ID}"',
    )
    write_bytes(TRANSPORT_PATH, transport.encode("utf-8"))
    write_bytes(STATIC_VERIFIER_PATH, VERIFIER_SOURCE.read_bytes())

    old_package = strict_object((V9_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_PACKAGE.json").read_bytes())
    package = replace_v9(old_package)
    package.pop("package_content_sha256", None)
    package["action_identity"] = {
        "future_action_id": ACTION_ID,
        "prior_terminal_action_id": V9_ACTION_ID,
        "retired_action_ids": [*OLD_V8_ACTION_IDS, V9_ACTION_ID],
        "reuse_permitted": False,
    }
    package["canonical_bindings"] = [canonical_binding(binding_id, path) for binding_id, path in expected_paths]
    local_paths = [
        ("authority_schema", schema_paths["AUTHORITY_SCHEMA"]),
        ("credential_schema", schema_paths["CREDENTIAL_SCHEMA"]),
        ("ledger_schema", schema_paths["LEDGER_SCHEMA"]),
        ("first_terminal_schema", schema_paths["FIRST_TERMINAL_SCHEMA"]),
        ("fresh_l2_acceptance_schema", schema_paths["FRESH_L2_ACCEPTANCE_SCHEMA"]),
        ("transport", TRANSPORT_PATH),
        ("launcher", LAUNCHER_PATH),
        ("static_verifier", STATIC_VERIFIER_PATH),
        ("forensic_report", REPORT_PATH),
    ]
    package["local_artifact_bindings"] = [local_binding(binding_id, path) for binding_id, path in local_paths]
    transport_argv = [
        str(INTERPRETER),
        str(TRANSPORT_PATH),
        "--package",
        str(PACKAGE_PATH),
        "--acceptance",
        str(V11_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
        "--irreversible-action-id",
        ACTION_ID,
    ]
    launcher_argv = [
        str(INTERPRETER),
        str(LAUNCHER_PATH),
        "--package",
        str(PACKAGE_PATH),
        "--acceptance",
        str(V11_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
        "--irreversible-action-id",
        ACTION_ID,
    ]
    package["future_invocation"] = invocation(transport_argv)
    package["launcher_invocation"] = invocation(launcher_argv)
    attestation = {
        "api": "os.posix_spawn",
        "immediate_parent_argv": transport_argv,
        "immediate_parent_environment": EXACT_ENVIRONMENT,
        "immediate_parent_executable_path": str(INTERPRETER),
        "immediate_parent_executable_sha256": sha256_file(INTERPRETER),
        "immediate_parent_transport_path": str(TRANSPORT_PATH),
        "immediate_parent_transport_sha256": sha256_file(TRANSPORT_PATH),
        "launcher_argv": launcher_argv,
        "launcher_environment": EXACT_ENVIRONMENT,
        "launcher_sha256": sha256_file(LAUNCHER_PATH),
        "shell": False,
    }
    package["transport_contract"]["attestation_sha256"] = sha256_bytes(compact_bytes(attestation))
    package["prior_terminal_state"] = {
        "action_id": V9_ACTION_ID,
        "invocation_count_performed": 0,
        "no_replay_retry_resume_repair_replacement": True,
        "payload_open_count": 0,
        "reason_code": "READ_ONLY_PREFLIGHT_FAILED",
        "result_file_sha256": None,
        "status": "PREFLIGHT_FAILED_TERMINAL",
    }
    package["accepted_byte_derivation"] = {
        "package": {
            "canonical_byte_count": len(canonical_package_raw),
            "canonical_path": str(CANONICAL_PACKAGE),
            "canonical_sha256": sha256_bytes(canonical_package_raw),
            "production_reader_context": "accepted V8 package",
            "semantic_json_equal": True,
            "source_byte_count": len(source_package_raw),
            "source_path": str(SOURCE_PACKAGE),
            "source_raw_sha256": sha256_bytes(source_package_raw),
        },
        "result_schema": {
            "canonical_byte_count": len(canonical_schema_raw),
            "canonical_path": str(CANONICAL_RESULT_SCHEMA),
            "canonical_sha256": sha256_bytes(canonical_schema_raw),
            "production_reader_context": "accepted V8 result schema",
            "semantic_json_equal": True,
            "source_byte_count": len(source_schema_raw),
            "source_path": str(SOURCE_RESULT_SCHEMA),
            "source_raw_sha256": sha256_bytes(source_schema_raw),
        },
    }
    package["production_canonical_preflight_proof"] = {
        "accepted_reader_function": "_read_accepted_package_and_result_schema",
        "independent_verifier": str(STATIC_VERIFIER_PATH),
        "live_creation_after_preflight": True,
        "preflight_function": "_preflight",
        "reader_functions": ["decode_canonical_json", "read_canonical_json"],
        "target_process_starts_required": 0,
    }
    package["static_file_policy"] = {
        "allowed_relative_files": [
            "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_PACKAGE.json",
            "V9_CANONICAL_BYTE_FAILURE_REPORT.md",
            "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json",
            "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_RESULT_SCHEMA.canonical.json",
            "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_AUTHORITY_SCHEMA.json",
            "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_CREDENTIAL_SCHEMA.json",
            "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_FIRST_TERMINAL_SCHEMA.json",
            "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
            "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V11_SHELLFREE_LEDGER_SCHEMA.json",
            "review/FRESH_L2_STATIC_ACCEPTANCE.json",
            "tools/qk_gbfp8_head64_granularity_sweep_execute_once_shellfree_v11.py",
            "tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v11.py",
            "tools/verify_qk_gbfp8_head64_granularity_sweep_execution_v11_shellfree.py",
        ],
        "forbidden_directories": ["build", "live", "__pycache__"],
        "generated_python_artifacts_permitted": False,
        "static_files_only": True,
    }
    package["claim_boundary"] = {
        "acceptance_materialized": False,
        "authority_materialized": False,
        "controller_or_evaluator_invoked": False,
        "credential_materialized": False,
        "execution_authorized": False,
        "first_terminal_materialized": False,
        "ledger_materialized": False,
        "result_materialized": False,
        "sealed_tensor_access": "NONE",
        "transport_or_launcher_invoked": False,
    }
    package["package_content_sha256"] = sha256_bytes(compact_bytes(package))
    write_bytes(PACKAGE_PATH, compact_bytes(package))

    for path in sorted(V11_ROOT.rglob("*"), reverse=True):
        if path.is_file():
            os.chmod(path, 0o444)
    for path in sorted((item for item in V11_ROOT.rglob("*") if item.is_dir()), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(V11_ROOT, 0o555)
    print(json.dumps({
        "action_id": ACTION_ID,
        "canonical_package_sha256": sha256_bytes(canonical_package_raw),
        "canonical_result_schema_sha256": sha256_bytes(canonical_schema_raw),
        "manifest_raw_sha256": sha256_file(PACKAGE_PATH),
        "package_content_sha256": package["package_content_sha256"],
        "root": str(V11_ROOT),
        "status": "BUILT_IMMUTABLE_V11_CANONICAL_PREFLIGHT_SUCCESSOR_NO_AUTHORITY",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
