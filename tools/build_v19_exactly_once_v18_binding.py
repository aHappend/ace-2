#!/usr/bin/env python3
"""Build the additive, static-only V19 exactly-once V18-binding package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v19:execute-once:fdfc33a1:additive-0001"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1"
BUILD_ROOT = PROJECT_ROOT / "build/v19-exactly-once-v18-binding-attempt-0001"
MANIFEST_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_PACKAGE.json"
DISPOSITION = "V19_EXACTLY_ONCE_V18_BINDING_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY"

V18_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v18_c02_binding_repair_static_action_root"
V18_MANIFEST = V18_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V18_C02_BINDING_REPAIR_STATIC_PACKAGE.json"
V18_BINDING = V18_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
V18_ACCEPTANCE = V18_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V18_POST_REPORT = PROJECT_ROOT / "build/v18-c02-binding-repair-static-0001/fresh-l2-post-acceptance-inert-report.json"
V18_BUILD_ROOT = PROJECT_ROOT / "build/v18-c02-binding-repair-static-0001"
V18_ACTION_ID = "ace2:qk-gbfp8-base-v18:static-c02-binding-repair:additive-0001"
V18_MANIFEST_SHA256 = "fdfc33a1443f237fbaca882b13fb977ae3c2ad7f61e427453297d0621f4445c8"
V18_BINDING_SHA256 = "87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655"
V18_ACCEPTANCE_SHA256 = "342ab0130d7e132e4e288b9fa349f7e0c5efb87cb35f29cdbc86e3bc3f5e8cf2"
V18_ACCEPTANCE_SELF_SHA256 = "601980bd0a7af5040c1ad07c9ea88e6aad71619326574436175419637fe74e12"
V18_POST_REPORT_SELF_SHA256 = "19d63c0b6610af00a9f031b7f91665496f0c17cdbfccb8cdc593d5a31ca0b3ae"

V17_WRAPPER_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_exactly_once_wrapper_4c8a6e31_repair_1_action_root"
V17_WRAPPER_MANIFEST = V17_WRAPPER_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json"
V17_WRAPPER_ACCEPTANCE = V17_WRAPPER_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"
V17_WRAPPER_FIXTURE = V17_WRAPPER_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_FIXTURE_REPORT.json"
V17_WRAPPER_VERIFIER = PROJECT_ROOT / "tools/verify_v17_exactly_once_wrapper_repair_1_inert.py"
V17_WRAPPER_TREE_SHA256 = "3b3666553b71ae081fa4a39202204ff78f97f767c08266f12c12e6ce3bed1545"
V17_CORE_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_preauthority_repair_1_action_root"
V17_CORE_MANIFEST = V17_CORE_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json"
V17_CORE_ACCEPTANCE = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/FRESH_L2_STATIC_ACCEPTANCE.json"
V17_CORE_VERIFIER_REPORT = PROJECT_ROOT / "build/v17-evaluator-diagnostic-preauthority-repair-0001/independent-inert-verifier-report.json"
V17_CORE_TOOLS = V17_CORE_ROOT / "tools"

OFFICIAL_PAYLOAD = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root/evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py"
OFFICIAL_PARSER = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py"
ROOT_VERIFIER_SOURCE = PROJECT_ROOT / "tools/verify_v19_exactly_once_v18_binding_inert.py"

EXPECTED_INTERPRETER_SHA256 = "fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad"
SELECTED_NAMES = ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"]
EXACT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}
AUDIT = {"official_payload_opens": 0, "official_target_starts": 0}


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


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field)
    candidate = dict(value)
    candidate.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(candidate)), f"self hash: {field}")


def sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = sha256_bytes(compact_bytes(result))
    return result


def write_bytes(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, compact_bytes(value))


def inventory_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"preservation root absent: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"preservation symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative})
        elif stat.S_ISREG(info.st_mode):
            records.append({"kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else:
            raise BuildError(f"unsupported preservation entry: {path}")
    return {
        "entry_count": len(records),
        "file_count": sum(record["kind"] == "file" for record in records),
        "root": str(root),
        "tree_sha256": sha256_bytes(compact_bytes(records)),
    }


def audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args:
        try:
            path = Path(args[0]).resolve()
        except (TypeError, OSError):
            return
        if path == OFFICIAL_PAYLOAD.resolve():
            AUDIT["official_payload_opens"] += 1
            raise BuildError("official payload open prohibited")
    if event == "subprocess.Popen":
        rendered = repr(args)
        if "evaluator_worker_v19.py" in rendered or "evaluator_static_v8.py" in rendered:
            AUDIT["official_target_starts"] += 1
            raise BuildError("official evaluator start prohibited")


def replace_required(source: str, old: str, new: str) -> str:
    require(old in source, f"source token absent: {old[:80]}")
    return source.replace(old, new)


def rewrite_strings(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [rewrite_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_strings(item, replacements) for key, item in value.items()}
    return value


def invocation_contract(argv: list[str], cwd: Path) -> dict[str, Any]:
    record = {"argv": argv, "cwd": str(cwd), "environment": EXACT_ENVIRONMENT, "shell": False}
    record["invocation_sha256"] = sha256_bytes(compact_bytes(record))
    return record


def source_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def transformed_engine() -> str:
    source = source_file(V17_WRAPPER_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17.py")
    source = source.replace("V17", "V19").replace("execution_v17_exactly_once", "execution_v19_exactly_once")
    source = replace_required(source, "import os\n", "import os\nimport stat\n")
    helper = '''def _prepare_runtime_namespace(paths: RuntimePaths) -> None:\n    root = Path(os.path.commonpath([str(path) for path in paths]))\n    directories = {root}\n    for path in paths:\n        current = path.parent\n        while current != root:\n            directories.add(current)\n            current = current.parent\n    for directory in sorted(directories, key=lambda item: len(item.parts)):\n        try:\n            os.mkdir(directory, 0o700)\n        except FileExistsError:\n            pass\n        info = os.lstat(directory)\n        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:\n            raise LifecycleError(f"runtime namespace directory: {directory}")\n\n\n'''
    source = replace_required(source, "def _other_lifecycle_exists(paths: RuntimePaths) -> bool:\n", helper + "def _other_lifecycle_exists(paths: RuntimePaths) -> bool:\n")
    source = replace_required(source, "def _claim_owner(paths: RuntimePaths, action_id: str, faults: frozenset[str]) -> dict[str, Any] | None:\n    if _other_lifecycle_exists(paths):", "def _claim_owner(paths: RuntimePaths, action_id: str, faults: frozenset[str]) -> dict[str, Any] | None:\n    _prepare_runtime_namespace(paths)\n    if _other_lifecycle_exists(paths):")
    authority_fields = '''        "binding_record_count": bindings["binding_record_count"],\n        "numerical_tensor_names": bindings["numerical_tensor_names"],\n        "v18_acceptance_file_sha256": bindings["v18_acceptance_file_sha256"],\n        "v18_acceptance_self_sha256": bindings["v18_acceptance_self_sha256"],\n        "v18_action_id": bindings["v18_action_id"],\n        "v18_binding_table_file_sha256": bindings["v18_binding_table_file_sha256"],\n        "v18_manifest_file_sha256": bindings["v18_manifest_file_sha256"],\n        "v18_post_acceptance_report_self_sha256": bindings["v18_post_acceptance_report_self_sha256"],\n'''
    source = replace_required(
        source,
        '        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_authority",\n',
        '        "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_authority",\n' + authority_fields,
    )
    return source


def transformed_core_source(name: str) -> str:
    source = source_file(V17_CORE_TOOLS / name)
    source = source.replace("qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17", "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v19")
    source = source.replace("qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v17", "qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v19")
    source = source.replace("qk_gbfp8_head64_granularity_sweep_result_validation_v17", "qk_gbfp8_head64_granularity_sweep_result_validation_v19")
    source = source.replace("evaluator_invocation_adapter_v17", "evaluator_invocation_adapter_v19")
    source = source.replace("execution_v17_nominal_records", "execution_v19_nominal_records")
    source = source.replace("V17", "V19")
    source = source.replace("ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z", ACTION_ID)
    return source


def transformed_bridge() -> str:
    source = transformed_core_source("qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17.py")
    source = replace_required(source, str(V17_CORE_ROOT), str(ACTION_ROOT))
    source = replace_required(source, "qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py", "qk_gbfp8_head64_granularity_sweep_evaluator_worker_v19.py")
    source = replace_required(source, "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json", MANIFEST_NAME)
    source = replace_required(source, str(V17_CORE_ACCEPTANCE), str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"))
    return source


def transformed_worker() -> str:
    source = transformed_core_source("qk_gbfp8_head64_granularity_sweep_evaluator_worker_v17.py")
    source = replace_required(source, str(V17_CORE_ROOT), str(ACTION_ROOT))
    source = replace_required(source, "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json", MANIFEST_NAME)
    source = replace_required(source, str(V17_CORE_ACCEPTANCE), str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"))
    constants = f'''BINDING_TABLE = Path({str(ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json")!r})
EXPECTED_BINDING_TABLE_SHA256 = {V18_BINDING_SHA256!r}
EXPECTED_SELECTED_NAMES = {SELECTED_NAMES!r}
'''
    source = replace_required(source, f"EXACT_ENVIRONMENT = {EXACT_ENVIRONMENT!r}\n", f"EXACT_ENVIRONMENT = {EXACT_ENVIRONMENT!r}\n{constants}")
    check = '''    binding_raw = BINDING_TABLE.read_bytes()\n    require(hashlib.sha256(binding_raw).hexdigest() == EXPECTED_BINDING_TABLE_SHA256, "V18 binding table hash")\n    binding_table = canonical_object(binding_raw)\n    require(binding_table["record_count"] == 25 and len(binding_table["records"]) == 25, "V18 binding table count")\n    require(binding_table["selected_tensor_names"] == EXPECTED_SELECTED_NAMES, "V18 numerical selection")\n    require(envelope["bindings"] == binding_table["records"], "V18 full binding integration")\n'''
    source = replace_required(source, "    require(set(envelope) == {\"bindings\", \"context\", \"payload_base64\"}, \"production envelope keys\")\n", "    require(set(envelope) == {\"bindings\", \"context\", \"payload_base64\"}, \"production envelope keys\")\n" + check)
    return source


def acceptance_schema() -> dict[str, Any]:
    sha = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    properties = {
        "acceptance_sha256": sha,
        "accepted": {"const": True},
        "action_id": {"const": ACTION_ID},
        "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_fresh_l2_static_acceptance"},
        "binding_table_file_sha256": {"const": V18_BINDING_SHA256},
        "claim_boundary": {"const": "STATIC_ONLY_NO_EXECUTION_AUTHORITY"},
        "decision": {"const": "ACCEPT_STATIC_PACKAGE"},
        "execution_package_file_sha256": sha,
        "fixture_report_file_sha256": sha,
        "official_payload_open_count": {"const": 0},
        "official_target_process_starts": {"const": 0},
        "required_disposition": {"const": DISPOSITION},
        "reviewer_role": {"const": "Fresh-L2"},
        "runtime_namespace_file_count": {"const": 0},
        "static_acceptance_grants_execution_authority": {"const": False},
        "v18_acceptance_file_sha256": {"const": V18_ACCEPTANCE_SHA256},
        "v18_manifest_file_sha256": {"const": V18_MANIFEST_SHA256},
        "v19_executed": {"const": False},
    }
    return {
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_FRESH_L2_ACCEPTANCE_SCHEMA",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
        "type": "object",
    }


def transformed_schemas(staging: Path) -> dict[str, str]:
    replacements = [
        ("ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z", ACTION_ID),
        ("EXECUTION_V17_EXACTLY_ONCE", "EXECUTION_V19_EXACTLY_ONCE"),
        ("execution_v17_exactly_once", "execution_v19_exactly_once"),
        ("evaluator_invocation_adapter_v17", "evaluator_invocation_adapter_v19"),
    ]
    names = ["AUTHORITY", "CREDENTIAL", "LEDGER", "FIRST_TERMINAL", "OWNER_CLAIM"]
    hashes: dict[str, str] = {}
    for label in names:
        old_name = f"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_{label}_SCHEMA.json"
        new_name = f"QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_{label}_SCHEMA.json"
        value = json.loads((V17_WRAPPER_ROOT / "reference" / old_name).read_text(encoding="ascii"))
        value = rewrite_strings(value, replacements)
        if label == "AUTHORITY":
            additions = {
                "binding_record_count": {"const": 25},
                "numerical_tensor_names": {"const": SELECTED_NAMES},
                "v18_acceptance_file_sha256": {"const": V18_ACCEPTANCE_SHA256},
                "v18_acceptance_self_sha256": {"const": V18_ACCEPTANCE_SELF_SHA256},
                "v18_action_id": {"const": V18_ACTION_ID},
                "v18_binding_table_file_sha256": {"const": V18_BINDING_SHA256},
                "v18_manifest_file_sha256": {"const": V18_MANIFEST_SHA256},
                "v18_post_acceptance_report_self_sha256": {"const": V18_POST_REPORT_SELF_SHA256},
            }
            value["properties"].update(additions)
            value["required"].extend(additions)
        path = staging / "reference" / new_name
        write_json(path, value)
        hashes[new_name] = sha256_file(path)
    acceptance_name = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_FRESH_L2_ACCEPTANCE_SCHEMA.json"
    acceptance_path = staging / "reference" / acceptance_name
    write_json(acceptance_path, acceptance_schema())
    hashes[acceptance_name] = sha256_file(acceptance_path)
    return hashes


def transformed_launcher(schema_hashes: dict[str, str]) -> str:
    source = source_file(V17_WRAPPER_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py")
    source = replace_required(source, "qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17", "qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v19")
    source = replace_required(source, str(V17_WRAPPER_ROOT), str(ACTION_ROOT))
    source = replace_required(source, str(PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31"), str(RUNTIME_ROOT))
    source = replace_required(source, "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json", MANIFEST_NAME)
    source = replace_required(source, "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z", ACTION_ID)
    source = source.replace("V17", "V19").replace("execution_v17_exactly_once", "execution_v19_exactly_once")
    source = replace_required(source, 'CORE_TOOLS = CORE_ROOT / "tools"', 'CORE_TOOLS = ROOT / "tools"')
    source = source.replace("qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17", "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v19")
    source = source.replace("qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v17", "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v19")
    source = source.replace("qk_gbfp8_head64_granularity_sweep_result_validation_v17", "qk_gbfp8_head64_granularity_sweep_result_validation_v19")
    source = source.replace("qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py", "qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19.py")
    source = source.replace("EXECUTION_V17_EXACTLY_ONCE", "EXECUTION_V19_EXACTLY_ONCE")
    source = source.replace("QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE", "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE")
    source = source.replace(
        "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_FRESH_L2_ACCEPTANCE_SCHEMA.json",
        "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_FRESH_L2_ACCEPTANCE_SCHEMA.json",
    )
    old_schema_block = re.search(r"SCHEMA_HASHES = \{.*?\n\}\nPATHS", source, re.S)
    require(old_schema_block is not None, "launcher schema block")
    rendered_hashes = json.dumps(schema_hashes, sort_keys=True, indent=4)
    source = source[:old_schema_block.start()] + f"SCHEMA_HASHES = {rendered_hashes}\nPATHS" + source[old_schema_block.end():]
    extra_constants = f'''V18_MANIFEST = Path({str(V18_MANIFEST)!r})
V18_BINDING_SOURCE = Path({str(V18_BINDING)!r})
V18_ACCEPTANCE = Path({str(V18_ACCEPTANCE)!r})
V18_POST_REPORT = Path({str(V18_POST_REPORT)!r})
BINDING_TABLE = ROOT / "bindings/C02_EXACT_BINDINGS_25.json"
EXPECTED_V18_MANIFEST_SHA256 = {V18_MANIFEST_SHA256!r}
EXPECTED_V18_BINDING_SHA256 = {V18_BINDING_SHA256!r}
EXPECTED_V18_ACCEPTANCE_SHA256 = {V18_ACCEPTANCE_SHA256!r}
EXPECTED_V18_ACCEPTANCE_SELF_SHA256 = {V18_ACCEPTANCE_SELF_SHA256!r}
EXPECTED_V18_POST_REPORT_SELF_SHA256 = {V18_POST_REPORT_SELF_SHA256!r}
EXPECTED_SELECTED_NAMES = {SELECTED_NAMES!r}
'''
    source = replace_required(source, f"CORE_VERIFIER_REPORT = Path({str(V17_CORE_VERIFIER_REPORT)!r})\n", f"CORE_VERIFIER_REPORT = Path({str(V17_CORE_VERIFIER_REPORT)!r})\n{extra_constants}")
    v18_checks = '''    require(sha256_file(V18_MANIFEST) == EXPECTED_V18_MANIFEST_SHA256, "V18 manifest hash", "V18_BINDING")\n    require(sha256_file(V18_BINDING_SOURCE) == EXPECTED_V18_BINDING_SHA256, "V18 binding source hash", "V18_BINDING")\n    require(sha256_file(V18_ACCEPTANCE) == EXPECTED_V18_ACCEPTANCE_SHA256, "V18 acceptance hash", "V18_BINDING")\n    v18_acceptance, _ = canonical(V18_ACCEPTANCE)\n    verify_self_hash(v18_acceptance, "acceptance_sha256")\n    require(v18_acceptance["acceptance_sha256"] == EXPECTED_V18_ACCEPTANCE_SELF_SHA256, "V18 acceptance self hash", "V18_BINDING")\n    v18_post, _ = canonical(V18_POST_REPORT)\n    verify_self_hash(v18_post, "report_sha256")\n    require(v18_post["report_sha256"] == EXPECTED_V18_POST_REPORT_SELF_SHA256, "V18 post report self hash", "V18_BINDING")\n'''
    source = replace_required(source, "    acceptance, acceptance_raw = canonical(ACCEPTANCE)\n", v18_checks + "    acceptance, acceptance_raw = canonical(ACCEPTANCE)\n")
    old_prepare = '''    records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]\n    holder["tensor_bindings"] = {\n        record["tensor_name"]: {"dtype": record["dtype"], "sha256": record["sha256"], "shape": list(record["shape"])}\n        for record in records.values()\n    }\n    holder["input_bindings"] = accepted["official_benchmark"]["input_bindings"]\n    holder["official"] = package["official_identities"]\n'''
    new_prepare = '''    selected_records = accepted["official_benchmark"]["input_bindings"]["tensor_records"]\n    selected_names = [record["tensor_name"] for record in selected_records.values()]\n    binding_table, binding_raw = canonical(BINDING_TABLE)\n    verify_self_hash(binding_table, "binding_table_sha256")\n    require(sha256_bytes(binding_raw) == EXPECTED_V18_BINDING_SHA256, "integrated V18 binding hash", "V18_BINDING")\n    require(binding_table["record_count"] == 25 and len(binding_table["records"]) == 25, "integrated V18 binding count", "V18_BINDING")\n    require(binding_table["selected_tensor_names"] == EXPECTED_SELECTED_NAMES, "integrated V18 selected names", "V18_BINDING")\n    require(selected_names == EXPECTED_SELECTED_NAMES, "frozen core numerical selection", "V18_BINDING")\n    holder["tensor_bindings"] = binding_table["records"]\n    holder["numerical_tensor_names"] = selected_names\n    holder["input_bindings"] = accepted["official_benchmark"]["input_bindings"]\n    holder["official"] = package["official_identities"]\n'''
    source = replace_required(source, old_prepare, new_prepare)
    binding_fields = f'''            "binding_record_count": 25,\n            "numerical_tensor_names": EXPECTED_SELECTED_NAMES,\n            "v18_acceptance_file_sha256": EXPECTED_V18_ACCEPTANCE_SHA256,\n            "v18_acceptance_self_sha256": EXPECTED_V18_ACCEPTANCE_SELF_SHA256,\n            "v18_action_id": {V18_ACTION_ID!r},\n            "v18_binding_table_file_sha256": EXPECTED_V18_BINDING_SHA256,\n            "v18_manifest_file_sha256": EXPECTED_V18_MANIFEST_SHA256,\n            "v18_post_acceptance_report_self_sha256": EXPECTED_V18_POST_REPORT_SELF_SHA256,\n'''
    source = replace_required(source, '            "core_acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,\n', binding_fields + '            "core_acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,\n')
    return source


def transformed_transport(launcher_hash: str) -> str:
    source = source_file(V17_WRAPPER_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v17.py")
    source = replace_required(source, "qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v17", "qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v19")
    source = replace_required(source, str(V17_WRAPPER_ROOT), str(ACTION_ROOT))
    source = replace_required(source, str(PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v17_evaluator_diagnostic_4c8a6e31"), str(RUNTIME_ROOT))
    source = replace_required(source, "qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v17.py", "qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py")
    source = replace_required(source, "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V17_EXACTLY_ONCE_WRAPPER_PACKAGE.json", MANIFEST_NAME)
    source = replace_required(source, "ace2:qk-gbfp8-base-v17:execute-once:4c8a6e31:20260814T102500Z", ACTION_ID)
    source = source.replace("V17", "V19").replace("execution_v17_exactly_once", "execution_v19_exactly_once")
    source = replace_required(source, "1bf18b6e683a4411f7958bb085e4d688da24d25b12c6e7cacbe9840e15461c74", launcher_hash)
    return source


def transformed_fixture(adapter_hash: str) -> str:
    source = source_file(V17_WRAPPER_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v17.py")
    source = replace_required(source, f"CORE_TOOLS = Path({str(V17_CORE_TOOLS)!r})", 'CORE_TOOLS = ROOT / "tools"')
    source = source.replace("v17", "v19").replace("V17", "V19")
    source = replace_required(source, "ace2:qk-gbfp8-base-v19:execute-once:4c8a6e31:20260814T102500Z", ACTION_ID)
    source = replace_required(source, "14b236ae1fdf10028088520c0816df0d65783764e6f5a6a385aa8bd879e84de3", adapter_hash)
    source = replace_required(source, "import qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19 as transport\n", "import qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19 as transport\n\nBRANCH_INVOCATION_COUNTS: list[int] = []\n")
    contract = '''def binding_contract() -> dict[str, Any]:\n    raw = (ROOT / "bindings/C02_EXACT_BINDINGS_25.json").read_bytes()\n    value = json.loads(raw.decode("ascii", "strict"))\n    if compact_bytes(value) != raw:\n        raise ValueError("fixture binding canonical")\n    if value["record_count"] != 25 or len(value["records"]) != 25:\n        raise ValueError("fixture binding count")\n    if value["selected_tensor_names"] != ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"]:\n        raise ValueError("fixture numerical selection")\n    return {"binding_table_file_sha256": sha256_bytes(raw), "complete_binding_count": 25, "numerical_tensor_names": value["selected_tensor_names"]}\n\n\n'''
    source = replace_required(source, "def load_schema(name: str) -> dict[str, Any]:\n", contract + "def load_schema(name: str) -> dict[str, Any]:\n")
    extra_bindings = f'''        "binding_record_count": 25,\n        "numerical_tensor_names": ["bf16.k_rope", "bf16.q_rope", "bf16.qk_scaled_scores"],\n        "v18_acceptance_file_sha256": {V18_ACCEPTANCE_SHA256!r},\n        "v18_acceptance_self_sha256": {V18_ACCEPTANCE_SELF_SHA256!r},\n        "v18_action_id": {V18_ACTION_ID!r},\n        "v18_binding_table_file_sha256": {V18_BINDING_SHA256!r},\n        "v18_manifest_file_sha256": {V18_MANIFEST_SHA256!r},\n        "v18_post_acceptance_report_self_sha256": {V18_POST_REPORT_SELF_SHA256!r},\n'''
    bindings_start = source.index("def bindings()")
    bindings_end = source.index("\n\ndef result_bytes", bindings_start)
    bindings_block = source[bindings_start:bindings_end]
    require('        "action_id": ACTION_ID,\n' in bindings_block, "fixture bindings action field")
    bindings_block = bindings_block.replace('        "action_id": ACTION_ID,\n', '        "action_id": ACTION_ID,\n' + extra_bindings, 1)
    source = source[:bindings_start] + bindings_block + source[bindings_end:]
    old_execute = '''    return run_execution(\n        paths,\n        bindings(),\n        validators(),\n        preflight,\n        lambda: b"synthetic-payload",\n        invoker(scenario),\n        faults=faults,\n    )\n'''
    new_execute = '''    outcome = run_execution(\n        paths,\n        bindings(),\n        validators(),\n        preflight,\n        lambda: b"synthetic-payload",\n        invoker(scenario),\n        faults=faults,\n    )\n    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)\n    return outcome\n'''
    source = replace_required(source, old_execute, new_execute)
    source = replace_required(source, "    owners = [outcome for outcome in outcomes if not outcome.duplicate_rejected]\n", "    BRANCH_INVOCATION_COUNTS.extend(outcome.counts.invocation_count for outcome in outcomes)\n    owners = [outcome for outcome in outcomes if not outcome.duplicate_rejected]\n")
    source = replace_required(source, "    terminal = read_json(Path(outcome.terminal_path))\n    return {\n        \"passed\": (\n            outcome.status == \"PREFLIGHT_FAILED_TERMINAL\"", "    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)\n    terminal = read_json(Path(outcome.terminal_path))\n    return {\n        \"passed\": (\n            outcome.status == \"PREFLIGHT_FAILED_TERMINAL\"", )
    source = replace_required(
        source,
        "        observed_environment=dict(launcher.EXPECTED_ENVIRONMENT),\n    )\n    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)\n",
        "        observed_environment=dict(launcher.EXPECTED_ENVIRONMENT),\n        schema_loader=validators,\n    )\n    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)\n",
    )
    source = replace_required(source, "    if result.outcome is None or result.outcome.terminal_path is None:\n", "    if result.outcome is not None:\n        BRANCH_INVOCATION_COUNTS.append(result.outcome.counts.invocation_count)\n    if result.outcome is None or result.outcome.terminal_path is None:\n")
    source = replace_required(source, "    launcher_hash_provider: Callable[[Path], str] = transport.sha256_file\n", "    launcher_hash_provider: Callable[[Path], str] = lambda path: transport.LAUNCHER_SHA256\n")
    source = replace_required(source, "def run_fixture() -> dict[str, Any]:\n    cases = []\n", "def run_fixture() -> dict[str, Any]:\n    BRANCH_INVOCATION_COUNTS.clear()\n    binding = binding_contract()\n    cases = []\n")
    source = replace_required(source, '        "case_count": len(cases),\n', '        "all_branches_at_most_once": max(BRANCH_INVOCATION_COUNTS, default=0) <= 1,\n        "binding_contract": binding,\n        "branch_invocation_observation_count": len(BRANCH_INVOCATION_COUNTS),\n        "case_count": len(cases),\n')
    source = replace_required(source, '        "launcher_transport_integration_exercised": True,\n', '        "launcher_transport_integration_exercised": True,\n        "max_branch_invocation_count": max(BRANCH_INVOCATION_COUNTS, default=0),\n        "owner_loser_publication_exclusion_verified": next((case.get("loser_published") is False for case in cases if case["name"] == "concurrent_start"), False),\n')
    return source


def runtime_directories() -> list[str]:
    return [
        str(RUNTIME_ROOT),
        str(RUNTIME_ROOT / "primary"),
        str(RUNTIME_ROOT / "primary/authority"),
        str(RUNTIME_ROOT / "primary/authority/base"),
        str(RUNTIME_ROOT / "primary/result"),
        str(RUNTIME_ROOT / "primary/result/base"),
        str(RUNTIME_ROOT / "fallback"),
    ]


def build() -> dict[str, Any]:
    sys.addaudithook(audit_hook)
    require(not os.path.lexists(ACTION_ROOT), f"immutable V19 action root already exists: {ACTION_ROOT}")
    require(not os.path.lexists(BUILD_ROOT), f"immutable V19 build root already exists: {BUILD_ROOT}")
    require(not os.path.lexists(RUNTIME_ROOT), f"V19 runtime namespace already exists: {RUNTIME_ROOT}")
    require(sha256_file(INTERPRETER) == EXPECTED_INTERPRETER_SHA256, "interpreter identity")
    require(sha256_file(V18_MANIFEST) == V18_MANIFEST_SHA256, "V18 manifest prerequisite")
    require(sha256_file(V18_BINDING) == V18_BINDING_SHA256, "V18 binding prerequisite")
    require(sha256_file(V18_ACCEPTANCE) == V18_ACCEPTANCE_SHA256, "V18 acceptance prerequisite")
    v18_package, _ = canonical(V18_MANIFEST)
    v18_binding, binding_raw = canonical(V18_BINDING)
    v18_acceptance, _ = canonical(V18_ACCEPTANCE)
    v18_post, _ = canonical(V18_POST_REPORT)
    verify_self_hash(v18_binding, "binding_table_sha256")
    verify_self_hash(v18_acceptance, "acceptance_sha256")
    verify_self_hash(v18_post, "report_sha256")
    require(v18_package["action_identity"]["action_id"] == V18_ACTION_ID, "V18 action id")
    require(v18_acceptance["acceptance_sha256"] == V18_ACCEPTANCE_SELF_SHA256, "V18 acceptance self identity")
    require(v18_post["report_sha256"] == V18_POST_REPORT_SELF_SHA256, "V18 post report self identity")
    require(v18_binding["record_count"] == 25 and v18_binding["selected_tensor_names"] == SELECTED_NAMES, "V18 binding cardinality and selection")
    v17_tree = inventory_digest(V17_WRAPPER_ROOT)
    require(v17_tree["tree_sha256"] == V17_WRAPPER_TREE_SHA256, "accepted V17 precedent drift")
    preservation = list(v18_package["preservation"])
    preservation.extend([inventory_digest(V18_ROOT), inventory_digest(V18_BUILD_ROOT)])
    before = {item["root"]: item for item in preservation}

    staging_parent = PROJECT_ROOT / "reference"
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{ROOT_ID}.staging-", dir=staging_parent))
    try:
        assert staging is not None
        schema_hashes = transformed_schemas(staging)
        sources = {
            "qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v19.py": transformed_engine(),
            "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v19.py": transformed_core_source("qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v17.py"),
            "qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v19.py": transformed_core_source("qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v17.py"),
            "qk_gbfp8_head64_granularity_sweep_result_validation_v19.py": transformed_core_source("qk_gbfp8_head64_granularity_sweep_result_validation_v17.py"),
            "qk_gbfp8_head64_granularity_sweep_nominal_records_v19.py": transformed_core_source("qk_gbfp8_head64_granularity_sweep_nominal_records_v17.py"),
            "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v19.py": transformed_bridge(),
            "qk_gbfp8_head64_granularity_sweep_evaluator_worker_v19.py": transformed_worker(),
        }
        for name, source in sources.items():
            compile(source, name, "exec")
            write_bytes(staging / "tools" / name, source.encode("utf-8"))
        launcher = transformed_launcher(schema_hashes)
        compile(launcher, "launcher_v19", "exec")
        launcher_path = staging / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py"
        write_bytes(launcher_path, launcher.encode("utf-8"))
        transport = transformed_transport(sha256_file(launcher_path))
        compile(transport, "transport_v19", "exec")
        write_bytes(staging / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19.py", transport.encode("utf-8"))
        fixture = transformed_fixture(sha256_file(staging / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v19.py"))
        compile(fixture, "fixture_v19", "exec")
        fixture_path = staging / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v19.py"
        write_bytes(fixture_path, fixture.encode("utf-8"))
        write_bytes(staging / "bindings/C02_EXACT_BINDINGS_25.json", binding_raw)
        write_bytes(staging / "tools/verify_qk_gbfp8_head64_granularity_sweep_exactly_once_v18_binding_v19.py", ROOT_VERIFIER_SOURCE.read_bytes())

        environment = dict(EXACT_ENVIRONMENT)
        completed = subprocess.run(
            [str(INTERPRETER), str(fixture_path)],
            cwd=staging,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(
            completed.returncode == 0 and completed.stderr == b"",
            f"synthetic fixture failed: rc={completed.returncode} stderr={completed.stderr!r} stdout={completed.stdout!r}",
        )
        fixture_report = json.loads(completed.stdout.decode("ascii", "strict"))
        require(compact_bytes(fixture_report) == completed.stdout, "fixture report canonical")
        require(fixture_report["status"] == "PASS_V19_EXACTLY_ONCE_SYNTHETIC_FIXTURE", "fixture status")
        require(fixture_report["binding_contract"]["binding_table_file_sha256"] == V18_BINDING_SHA256, "fixture V18 binding")
        require(fixture_report["max_branch_invocation_count"] == 1 and fixture_report["all_branches_at_most_once"] is True, "fixture invocation cardinality")
        require(fixture_report["owner_loser_publication_exclusion_verified"] is True, "fixture loser publication")
        fixture_report_path = staging / "evidence/SYNTHETIC_EXACTLY_ONCE_V18_BINDING_FIXTURE_REPORT.json"
        write_bytes(fixture_report_path, completed.stdout)

        relative_files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        generated_files = {
            relative: {"sha256": sha256_file(staging / relative), "size": (staging / relative).stat().st_size}
            for relative in relative_files
        }
        paths = {
            "owner": str(RUNTIME_ROOT / "primary/authority/base/owner-claim.json"),
            "authority": str(RUNTIME_ROOT / "primary/authority/base/authority.json"),
            "credential": str(RUNTIME_ROOT / "primary/authority/base/credential.json"),
            "ledger": str(RUNTIME_ROOT / "primary/authority/base/authority-ledger.json"),
            "result": str(RUNTIME_ROOT / "primary/result/base/result.json"),
            "first_terminal": str(RUNTIME_ROOT / "primary/authority/base/first-terminal.json"),
        }
        fallback_terminal = str(RUNTIME_ROOT / "fallback/first-terminal.json")
        launcher_invocation = invocation_contract([
            str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py"),
            "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            "--irreversible-action-id", ACTION_ID,
        ], ACTION_ROOT)
        transport_invocation = invocation_contract([
            str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19.py"),
            "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            "--irreversible-action-id", ACTION_ID,
        ], ACTION_ROOT)
        production_invocation = invocation_contract([
            str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v19.py"),
            "--mode", "production", "--package", str(ACTION_ROOT / MANIFEST_NAME),
            "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"),
            "--irreversible-action-id", ACTION_ID,
        ], ACTION_ROOT)
        v17_manifest, v17_manifest_raw = canonical(V17_WRAPPER_MANIFEST)
        _, v17_acceptance_raw = canonical(V17_WRAPPER_ACCEPTANCE)
        core_manifest, core_manifest_raw = canonical(V17_CORE_MANIFEST)
        _, core_acceptance_raw = canonical(V17_CORE_ACCEPTANCE)
        accepted_package_path = V17_CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
        accepted_package, accepted_package_raw = canonical(accepted_package_path)
        payload_info = os.lstat(OFFICIAL_PAYLOAD)
        require(stat.S_ISREG(payload_info.st_mode) and not stat.S_ISLNK(payload_info.st_mode), "official payload metadata")
        manifest = {
            "action_identity": {
                "action_id": ACTION_ID,
                "additive_successor": True,
                "future_action_id": ACTION_ID,
                "predecessor_action_id": V18_ACTION_ID,
                "predecessor_static_acceptance_bound": True,
                "retry_replay_resume_repair_replacement_permitted": False,
                "v17_modified_or_replayed": False,
                "v18_modified_or_executed": False,
            },
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_package",
            "attempt": "0001",
            "claim_boundary": {
                "claim": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
                "authority_materialized": False,
                "credential_materialized": False,
                "evaluator_invocations": 0,
                "execution_authorized": False,
                "ledger_materialized": False,
                "official_payload_open_count": 0,
                "official_target_process_starts": 0,
                "owner_claim_materialized": False,
                "result_materialized": False,
                "runtime_namespace_materialized": False,
                "runtime_terminal_materialized": False,
                "stage_2_activity": False,
                "v19_executed": False,
            },
            "frozen_core": {
                "accepted_package_file_sha256": sha256_bytes(accepted_package_raw),
                "accepted_package_path": str(accepted_package_path),
                "core_acceptance_file_sha256": sha256_bytes(core_acceptance_raw),
                "core_manifest_file_sha256": sha256_bytes(core_manifest_raw),
                "core_manifest_self_sha256": core_manifest["package_content_sha256"],
                "core_root": str(V17_CORE_ROOT),
                "core_verifier_report_file_sha256": sha256_file(V17_CORE_VERIFIER_REPORT),
                "official_evaluator_sha256": sha256_file(OFFICIAL_EVALUATOR),
                "official_parser_sha256": sha256_file(OFFICIAL_PARSER),
            },
            "generated_files": generated_files,
            "interpreter": {"path": str(INTERPRETER), "sha256": EXPECTED_INTERPRETER_SHA256, "version": "3.13.5"},
            "launcher_invocation": launcher_invocation,
            "lifecycle": {
                "atomic_owner_claim": "O_CREAT|O_EXCL_CREATE_ONCE_AFTER_PRIVATE_NAMESPACE_PREPARATION",
                "authority_create_count_maximum": 1,
                "concurrent_loser_disposition": "DUPLICATE_REJECTED_NO_PUBLICATION",
                "consumption_order": ["private_runtime_namespace_prepare", "owner_claim_create_once", "canonical_preflight", "authority_create_once", "credential_create_once", "ledger_create_once", "credential_durable_unlink_once", "payload_open_once", "evaluator_invoke_once"],
                "credential_consumption_count_maximum": 1,
                "duplicate_invocation_rejected_without_invocation": True,
                "first_terminal_publication": "OWNER_ONLY_PRIMARY_CREATE_THEN_FALLBACK_CREATE",
                "invocation_count_maximum": 1,
                "payload_open_count_maximum": 1,
                "post_consumption_failure": "CONSUMED_ORPHAN",
                "preconsumption_failure": "OWNER_RETIRES_WITH_ZERO_INVOCATION",
                "result_publication": "CREATE_ONLY_FSYNC_FILE_AND_PARENT",
                "terminal_publication_count_maximum": 1,
                "transport_preconsumption_failure": "SAME_OWNER_RETIREMENT_BOUNDARY",
            },
            "numerical_selection": {
                "complete_parser_binding_count": 25,
                "evaluator_receives_complete_binding_table": True,
                "numerical_tensor_count": 3,
                "tensor_names": SELECTED_NAMES,
            },
            "official_identities": {
                "input_tokens_sha256": accepted_package["official_benchmark"]["input_bindings"]["input_token_ids_sha256"],
                "lane_metadata_sha256": v18_package["frozen_official_identity"]["lane_metadata_sha256"],
                "model_sha256": v18_package["frozen_official_identity"]["model_identity_sha256"],
                "tensor_bundle_sha256": v18_package["frozen_official_identity"]["tensor_bundle_sha256"],
            },
            "official_payload": {
                "access_during_static_verification": "PROHIBITED",
                "byte_count": payload_info.st_size,
                "path": str(OFFICIAL_PAYLOAD),
                "sha256": v18_package["frozen_official_identity"]["tensor_bundle_sha256"],
            },
            "preservation": preservation,
            "production_evaluator_invocation": production_invocation,
            "required_disposition": DISPOSITION,
            "root_id": ROOT_ID,
            "runtime_namespace": {
                "fallback_terminal": fallback_terminal,
                "file_count": 0,
                "paths": paths,
                "precreated_directories": runtime_directories(),
                "provisioning": "LAZY_PRIVATE_DIRECTORY_CREATION_BEFORE_ATOMIC_OWNER_CLAIM",
                "runtime_root": str(RUNTIME_ROOT),
                "static_acceptance_namespace_absent": True,
            },
            "schema_version": 1,
            "static_acceptance": {
                "acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
                "grants_execution_authority": False,
                "inventory_policy": "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE",
                "present": False,
                "schema_path": str(ACTION_ROOT / "reference/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_FRESH_L2_ACCEPTANCE_SCHEMA.json"),
            },
            "static_file_policy": {
                "acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json",
                "allowed_relative_files": sorted([MANIFEST_NAME, "review/FRESH_L2_STATIC_ACCEPTANCE.json", *generated_files]),
                "optional_before_acceptance": ["review/FRESH_L2_STATIC_ACCEPTANCE.json"],
            },
            "synthetic_fixture": {
                "case_count": fixture_report["case_count"],
                "official_payload_open_count": 0,
                "official_target_process_starts": 0,
                "production_mode_exercised": False,
                "report_file_sha256": sha256_file(fixture_report_path),
                "report_path": str(ACTION_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_V18_BINDING_FIXTURE_REPORT.json"),
                "synthetic_bytes_only": True,
            },
            "transport_invocation": transport_invocation,
            "v17_exactly_once_precedent": {
                "acceptance_file_sha256": sha256_bytes(v17_acceptance_raw),
                "fixture_report_file_sha256": sha256_file(V17_WRAPPER_FIXTURE),
                "manifest_file_sha256": sha256_bytes(v17_manifest_raw),
                "manifest_self_sha256": v17_manifest["package_content_sha256"],
                "root": str(V17_WRAPPER_ROOT),
                "tree_sha256": V17_WRAPPER_TREE_SHA256,
                "verifier_file_sha256": sha256_file(V17_WRAPPER_VERIFIER),
            },
            "v18_binding": {
                "acceptance_file_sha256": V18_ACCEPTANCE_SHA256,
                "acceptance_self_sha256": V18_ACCEPTANCE_SELF_SHA256,
                "action_id": V18_ACTION_ID,
                "binding_table_file_sha256": V18_BINDING_SHA256,
                "integrated_binding_table_path": str(ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"),
                "manifest_file_sha256": V18_MANIFEST_SHA256,
                "post_acceptance_report_self_sha256": V18_POST_REPORT_SELF_SHA256,
                "record_count": 25,
                "source_root": str(V18_ROOT),
            },
            "verifier": {
                "acceptance_aware": True,
                "exact_argv": [str(INTERPRETER), str(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_exactly_once_v18_binding_v19.py")],
                "inventory_policy": "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE",
                "path": str(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_exactly_once_v18_binding_v19.py"),
            },
        }
        manifest = sealed(manifest, "package_content_sha256")
        manifest_path = staging / MANIFEST_NAME
        write_json(manifest_path, manifest)
        for path in sorted(staging.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(staging, 0o555)
        os.rename(staging, ACTION_ROOT)
        staging = None

        BUILD_ROOT.mkdir(parents=True, mode=0o755)
        report = sealed({
            "action_id": ACTION_ID,
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_build_report",
            "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY",
            "fixture_report_file_sha256": sha256_file(ACTION_ROOT / "evidence/SYNTHETIC_EXACTLY_ONCE_V18_BINDING_FIXTURE_REPORT.json"),
            "manifest_file_sha256": sha256_file(ACTION_ROOT / MANIFEST_NAME),
            "official_payload_open_count": AUDIT["official_payload_opens"],
            "official_target_process_starts": AUDIT["official_target_starts"],
            "preservation_root_count": len(preservation),
            "required_disposition": DISPOSITION,
            "runtime_namespace_file_count": 0,
            "status": "PASS_V19_STATIC_PACKAGE_BUILT_READY_FOR_INDEPENDENT_FRESH_L2",
            "v19_executed": False,
        }, "report_sha256")
        write_json(BUILD_ROOT / "build-report.json", report)
        os.chmod(BUILD_ROOT / "build-report.json", 0o444)

        require(not os.path.lexists(RUNTIME_ROOT), "builder materialized V19 runtime namespace")
        require(AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "builder live effect audit")
        for root, expected in before.items():
            require(inventory_digest(Path(root)) == expected, f"preservation drift after build: {root}")
        return report
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    try:
        report = build()
    except Exception as error:
        sys.stderr.write(f"V19_BUILD_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
