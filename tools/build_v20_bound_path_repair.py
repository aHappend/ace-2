#!/usr/bin/env python3
"""Build the additive V20 bound-path CORE_MANIFEST repair candidate."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v19_exactly_once_v18_binding as base


PROJECT_ROOT = Path("/home/argustest/ace-2")
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ACTION_ID = "ace2:qk-gbfp8-base-v20:execute-once:a81f2916:additive-0001"
ROOT_ID = "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_action_root"
ACTION_ROOT = PROJECT_ROOT / "reference" / ROOT_ID
RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_a81f2916"
BUILD_ROOT = PROJECT_ROOT / "build/v20-bound-path-repair-attempt-0001"
MANIFEST_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_PACKAGE.json"
DISPOSITION = "V20_BOUND_PATH_REPAIR_STATIC_PACKAGE_READY_FOR_FRESH_L2_NO_EXECUTION_AUTHORITY"

V19_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_v18_binding_action_root"
V19_MANIFEST = V19_ROOT / "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V19_EXACTLY_ONCE_V18_BINDING_PACKAGE.json"
V19_BUILD_ROOT = PROJECT_ROOT / "build/v19-exactly-once-v18-binding-attempt-0001"
V19_RUNTIME_ROOT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v19_exactly_once_fdfc33a1"
V19_ACTION_ID = "ace2:qk-gbfp8-base-v19:execute-once:fdfc33a1:additive-0001"
V19_TERMINAL = V19_RUNTIME_ROOT / "primary/authority/base/first-terminal.json"
V19_TERMINAL_SHA256 = "a81f2916fe3e5f2aa0489caee7d7a52ada1a8760b32ea7faf3a94bad6f4cb862"

CORE_ROOT = base.V17_CORE_ROOT
CORE_MANIFEST = base.V17_CORE_MANIFEST
CORE_ACCEPTANCE = base.V17_CORE_ACCEPTANCE
CORE_VERIFIER_REPORT = base.V17_CORE_VERIFIER_REPORT
STATIC_V8_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root"
G16_ROOT = PROJECT_ROOT / "reference/qk_gbfp8_g16_base_cardinality_one_authority_static_v6_action_root"
OFFICIAL_LANE_METADATA = G16_ROOT / "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json"
OFFICIAL_PAYLOAD = base.OFFICIAL_PAYLOAD
OFFICIAL_EVALUATOR = base.OFFICIAL_EVALUATOR
OFFICIAL_PARSER = base.OFFICIAL_PARSER
V18_ROOT = base.V18_ROOT
V18_MANIFEST = base.V18_MANIFEST
V18_BINDING = base.V18_BINDING
V18_ACCEPTANCE = base.V18_ACCEPTANCE
V18_POST_REPORT = base.V18_POST_REPORT
V18_BUILD_ROOT = base.V18_BUILD_ROOT
ROOT_VERIFIER_SOURCE = PROJECT_ROOT / "tools/verify_v20_bound_path_repair_inert.py"
EXPECTED_INTERPRETER_SHA256 = base.EXPECTED_INTERPRETER_SHA256
SELECTED_NAMES = base.SELECTED_NAMES
EXACT_ENVIRONMENT = base.EXACT_ENVIRONMENT

OLD_MANIFEST_NAME = V19_MANIFEST.name
OLD_ROOT = str(V19_ROOT)
OLD_RUNTIME = str(V19_RUNTIME_ROOT)
PREFLIGHT_NAME = "qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v20.py"
FIXTURE_REPORT_NAME = "SYNTHETIC_BOUND_CORE_MANIFEST_PREFLIGHT_FIXTURE_REPORT.json"
ACCEPTANCE_SCHEMA_NAME = "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA.json"


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


compact_bytes = base.compact_bytes
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
canonical = base.canonical
verify_self_hash = base.verify_self_hash
sealed = base.sealed
write_bytes = base.write_bytes
write_json = base.write_json
inventory_digest = base.inventory_digest
invocation_contract = base.invocation_contract


def v20ize(source: str) -> str:
    replacements = [
        (OLD_ROOT, str(ACTION_ROOT)),
        (OLD_RUNTIME, str(RUNTIME_ROOT)),
        (V19_ACTION_ID, ACTION_ID),
        (OLD_MANIFEST_NAME, MANIFEST_NAME),
        ("EXECUTION_V19_EXACTLY_ONCE_V18_BINDING", "EXECUTION_V20_BOUND_PATH_REPAIR"),
        ("execution_v19_exactly_once_v18_binding", "execution_v20_bound_path_repair"),
        ("V19_EXACTLY_ONCE_V18_BINDING", "V20_BOUND_PATH_REPAIR"),
        ("v19_exactly_once_v18_binding", "v20_bound_path_repair"),
        ("V19", "V20"),
        ("v19", "v20"),
    ]
    for old, new in replacements:
        source = source.replace(old, new)
    return source


def transformed_source(relative: str) -> str:
    return v20ize((V19_ROOT / relative).read_text(encoding="utf-8"))


def acceptance_schema() -> dict[str, Any]:
    sha = {"pattern": "^[0-9a-f]{64}$", "type": "string"}
    properties = {
        "acceptance_sha256": sha,
        "accepted": {"const": True},
        "action_id": {"const": ACTION_ID},
        "artifact_kind": {"const": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_fresh_l2_static_acceptance"},
        "claim_boundary": {"const": "STATIC_ONLY_NO_EXECUTION_AUTHORITY"},
        "core_manifest_closure_entries_sha256": sha,
        "decision": {"const": "ACCEPT_STATIC_PACKAGE"},
        "execution_package_file_sha256": sha,
        "fixture_report_file_sha256": sha,
        "official_payload_open_count": {"const": 0},
        "official_target_process_starts": {"const": 0},
        "required_disposition": {"const": DISPOSITION},
        "reviewer_role": {"const": "Fresh-L2"},
        "runtime_namespace_file_count": {"const": 0},
        "static_acceptance_grants_execution_authority": {"const": False},
        "v18_binding_table_file_sha256": {"const": base.V18_BINDING_SHA256},
        "v19_terminal_file_sha256": {"const": V19_TERMINAL_SHA256},
        "v20_executed": {"const": False},
    }
    return {
        "$id": "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_BOUND_PATH_REPAIR_FRESH_L2_ACCEPTANCE_SCHEMA",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
        "type": "object",
    }


def transformed_schemas(staging: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for source_path in sorted((V19_ROOT / "reference").glob("*.json")):
        if source_path.name.endswith("FRESH_L2_ACCEPTANCE_SCHEMA.json"):
            continue
        name = v20ize(source_path.name)
        value = json.loads(source_path.read_text(encoding="ascii"))
        value = base.rewrite_strings(value, [
            (V19_ACTION_ID, ACTION_ID),
            ("EXECUTION_V19_EXACTLY_ONCE", "EXECUTION_V20_EXACTLY_ONCE"),
            ("execution_v19_exactly_once", "execution_v20_exactly_once"),
            ("V19", "V20"),
            ("v19", "v20"),
        ])
        path = staging / "reference" / name
        write_json(path, value)
        hashes[name] = sha256_file(path)
    acceptance_path = staging / "reference" / ACCEPTANCE_SCHEMA_NAME
    write_json(acceptance_path, acceptance_schema())
    hashes[ACCEPTANCE_SCHEMA_NAME] = sha256_file(acceptance_path)
    return hashes


def preflight_source() -> str:
    return r'''#!/home/argustest/miniconda3/bin/python3.13
"""Shared read-only path/hash/mode-bound V20 CORE_MANIFEST preflight."""

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
'''


def patch_launcher(source: str, schema_hashes: dict[str, str]) -> str:
    source = v20ize(source)
    source = source.replace(
        "QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V20_EVALUATOR_DIAGNOSTIC_PREAUTHORITY_PACKAGE.json",
        CORE_MANIFEST.name,
    )
    source = source.replace(
        "from qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v20 import (",
        "from qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v20 import verify_core_manifest_closure\nfrom qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v20 import (",
    )
    old_schema = re.search(r"SCHEMA_HASHES = \{.*?\n\}\nPATHS", source, re.S)
    require(old_schema is not None, "V20 launcher schema block")
    source = source[:old_schema.start()] + f"SCHEMA_HASHES = {json.dumps(schema_hashes, sort_keys=True, indent=4)}\nPATHS" + source[old_schema.end():]
    source = source.replace(
        '''    owner, _ = canonical(paths.owner)\n    validators["owner"](owner)\n    verify_self_hash(owner, "owner_claim_sha256")\n    require(not any(os.path.lexists(path) for path in paths if path != paths.owner), "runtime lifecycle already materialized", "RUNTIME_PREFLIGHT")\n''',
        '''    require(not any(os.path.lexists(path) for path in paths), "runtime lifecycle already materialized", "RUNTIME_PREFLIGHT")\n''',
    )
    source = source.replace(
        '''    require(package["runtime_namespace"]["paths"]["owner"] == str(paths.owner), "owner path binding", "PACKAGE_BINDING")\n''',
        '''    require(package["runtime_namespace"]["paths"]["owner"] == str(paths.owner), "owner path binding", "PACKAGE_BINDING")\n    holder_closure = package["core_manifest_closure"]\n    verify_core_manifest_closure(holder_closure, expected_core_manifest_path=CORE_MANIFEST)\n''',
    )
    runtime_loop = '''    for directory in package["runtime_namespace"]["precreated_directories"]:\n        info = os.lstat(directory)\n        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"runtime directory: {directory}", "RUNTIME_PREFLIGHT")\n        require(stat.S_IMODE(info.st_mode) == 0o700, f"runtime mode: {directory}", "RUNTIME_PREFLIGHT")\n'''
    source = source.replace(runtime_loop, '''    require(not os.path.lexists(RUNTIME_ROOT), "runtime root absent before owner", "RUNTIME_PREFLIGHT")\n''')
    source = source.replace('package["frozen_core_imports"]["official_evaluator_sha256"]', 'package["core_manifest_closure"]["official_evaluator_sha256"]')
    start = source.index("def run_launcher(\n")
    end = source.index("\n\ndef main()", start)
    new_run = '''def run_launcher(\n    *,\n    paths: RuntimePaths = PATHS,\n    argv: list[str] | None = None,\n    observed_cwd: Path | None = None,\n    observed_environment: dict[str, str] | None = None,\n    schema_loader: Callable[[], dict[str, Any]] = load_schemas,\n    module_loader: Callable[[str, Path], Any] = load_module,\n) -> Outcome:\n    raw_argv = list(sys.argv[1:] if argv is None else argv)\n    arguments = parse_exact_arguments(raw_argv)\n    validators = schema_loader()\n    holder = read_only_preflight(\n        arguments, validators, paths, observed_cwd=observed_cwd, observed_environment=observed_environment\n    )\n    prepare_execution(holder, module_loader)\n    package = holder["package"]\n    bindings = {\n        "action_id": ACTION_ID,\n        "binding_record_count": 25,\n        "numerical_tensor_names": EXPECTED_SELECTED_NAMES,\n        "v18_acceptance_file_sha256": EXPECTED_V18_ACCEPTANCE_SHA256,\n        "v18_acceptance_self_sha256": EXPECTED_V18_ACCEPTANCE_SELF_SHA256,\n        "v18_action_id": ''' + repr(base.V18_ACTION_ID) + ''',\n        "v18_binding_table_file_sha256": EXPECTED_V18_BINDING_SHA256,\n        "v18_manifest_file_sha256": EXPECTED_V18_MANIFEST_SHA256,\n        "v18_post_acceptance_report_self_sha256": EXPECTED_V18_POST_REPORT_SELF_SHA256,\n        "core_acceptance_file_sha256": EXPECTED_CORE_ACCEPTANCE_SHA256,\n        "core_manifest_file_sha256": EXPECTED_CORE_MANIFEST_SHA256,\n        "execution_package_file_sha256": holder["package_sha256"],\n        "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],\n        "interpreter_sha256": EXPECTED_INTERPRETER_SHA256,\n        "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],\n        "launcher_invocation_sha256": package["launcher_invocation"]["invocation_sha256"],\n        "official_identities": holder["official"],\n        "transport_attestation": holder["transport_attestation"],\n    }\n\n    def payload_reader() -> bytes:\n        return read_payload_once(holder["official"]["tensor_bundle_sha256"], package["official_payload"]["byte_count"])\n\n    def invoke_once(payload: bytes, authority_sha256: str, ledger_sha256: str, publish_result: Any) -> Any:\n        official = holder["official"]\n        result_validation = holder["result_validation"]\n        evaluator_sha256 = package["core_manifest_closure"]["official_evaluator_sha256"]\n        result_bindings = result_validation.ResultBindings(\n            acceptance_sha256=holder["acceptance_sha256"], authority_sha256=authority_sha256,\n            evaluator_sha256=evaluator_sha256, invocation_sha256=package["production_evaluator_invocation"]["invocation_sha256"],\n            ledger_sha256=ledger_sha256, model_identity_sha256=official["model_sha256"],\n            package_sha256=holder["core_package_sha256"], tensor_bundle_sha256=official["tensor_bundle_sha256"],\n        )\n        bound = result_validation.bind_result_validators(holder["result_validator"], result_bindings, ACTION_ID)\n        context = {\n            "authority_sha256": authority_sha256, "consumed_ledger_sha256": ledger_sha256,\n            "evaluator_sha256": evaluator_sha256, "fresh_l2_acceptance_sha256": holder["acceptance_sha256"],\n            "input_bindings": holder["input_bindings"],\n            "invocation_sha256": package["production_evaluator_invocation"]["invocation_sha256"],\n            "irreversible_action_id": ACTION_ID, "model_identity_sha256": official["model_sha256"],\n            "package_sha256": holder["core_package_sha256"],\n        }\n        return holder["bridge"].invoke_production_evaluator(\n            payload, holder["tensor_bindings"], context, bound.validate_result_schema,\n            bound.validate_result_record, publish_result,\n        )\n\n    return run_execution(paths, bindings, validators, lambda: None, payload_reader, invoke_once)\n'''
    return source[:start] + new_run + source[end:]


def patch_transport(source: str, launcher_hash: str) -> str:
    source = v20ize(source)
    old_hash = re.search(r"LAUNCHER_SHA256 = '[0-9a-f]{64}'", source)
    require(old_hash is not None, "V20 transport launcher hash")
    return source[:old_hash.start()] + f"LAUNCHER_SHA256 = {launcher_hash!r}" + source[old_hash.end():]


def synthetic_preflight_helpers() -> str:
    return r'''

def _seal_synthetic_closure(closure: dict[str, Any]) -> None:
    closure["entries"] = sorted(closure["entries"], key=lambda item: item["path"])
    closure["entries_sha256"] = sha256_bytes(compact_bytes(closure["entries"]))


def _synthetic_closure(root: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = root / "CORE_MANIFEST.json"
    artifact_path = root / "artifact.bin"
    payload_path = root / "payload.bin"
    manifest = {"artifact_kind": "synthetic_core_manifest", "generated_files": {"artifact.bin": {"sha256": sha256_bytes(b"artifact"), "size": 8}}, "package_content_sha256": "1" * 64}
    manifest_raw = compact_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    artifact_path.write_bytes(b"artifact")
    payload_path.write_bytes(b"synthetic-payload")
    for path in (manifest_path, artifact_path, payload_path):
        os.chmod(path, 0o444)
    def record(path: Path, *, canonical_json: bool = False, deferred: bool = False) -> dict[str, Any]:
        raw = path.read_bytes()
        semantic = None
        if canonical_json:
            semantic = sha256_bytes(compact_bytes(json.loads(raw.decode("ascii"))))
        return {"canonical_json": canonical_json, "hash_policy": "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN" if deferred else "FULL_READ_ONLY_PREFLIGHT", "kind": "file", "mode": "0444", "path": str(path), "semantic_sha256": semantic, "sha256": sha256_bytes(raw), "size": len(raw), "sources": ["synthetic"]}
    entries = [
        {"canonical_json": False, "hash_policy": "NOT_APPLICABLE", "kind": "directory", "mode": f"{os.lstat(root).st_mode & 0o7777:04o}", "path": str(root), "semantic_sha256": None, "sha256": None, "size": None, "sources": ["synthetic"]},
        record(manifest_path, canonical_json=True), record(artifact_path), record(payload_path, deferred=True),
    ]
    closure = {
        "core_manifest_mode": "0444", "core_manifest_path": str(manifest_path),
        "core_manifest_semantic_sha256": sha256_bytes(manifest_raw), "core_manifest_sha256": sha256_bytes(manifest_raw),
        "core_manifest_self_sha256": "1" * 64, "deferred_hash_paths": [str(payload_path)],
        "directory_count": 1, "entries": entries, "entries_sha256": "", "file_count": 3,
        "full_hash_file_count": 2, "official_evaluator_sha256": "2" * 64,
        "official_parser_sha256": "3" * 64, "package_key": "core_manifest_closure",
        "path_hash_mode_complete": True, "schema_version": 1,
    }
    _seal_synthetic_closure(closure)
    return closure, manifest_path, artifact_path


def _real_preflight_case(name: str, mutation: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v20-real-preflight-") as temporary:
        root = Path(temporary)
        closure, manifest_path, artifact_path = _synthetic_closure(root)
        expected_stage = None
        if mutation == "absent":
            manifest_path.unlink()
            expected_stage = "PATH_ABSENT"
        elif mutation == "wrong_path":
            wrong = root / "WRONG_CORE_MANIFEST.json"
            wrong.write_bytes(manifest_path.read_bytes())
            os.chmod(wrong, 0o444)
            entry = next(item for item in closure["entries"] if item["path"] == str(manifest_path))
            entry["path"] = str(wrong)
            closure["core_manifest_path"] = str(wrong)
            _seal_synthetic_closure(closure)
            expected_stage = "CORE_MANIFEST_PATH"
        elif mutation == "wrong_hash":
            os.chmod(artifact_path, 0o644)
            artifact_path.write_bytes(b"mutated!")
            os.chmod(artifact_path, 0o444)
            expected_stage = "PATH_HASH"
        elif mutation == "wrong_mode":
            os.chmod(artifact_path, 0o644)
            expected_stage = "PATH_MODE"
        elif mutation == "malformed":
            os.chmod(manifest_path, 0o644)
            raw = b"{malformed\n"
            manifest_path.write_bytes(raw)
            os.chmod(manifest_path, 0o444)
            entry = next(item for item in closure["entries"] if item["path"] == str(manifest_path))
            entry["sha256"] = sha256_bytes(raw)
            entry["size"] = len(raw)
            _seal_synthetic_closure(closure)
            expected_stage = "JSON_MALFORMED"
        elif mutation == "inconsistent":
            os.chmod(manifest_path, 0o644)
            value = json.loads(manifest_path.read_text(encoding="ascii"))
            value["generated_files"]["artifact.bin"]["size"] = 9
            raw = compact_bytes(value)
            manifest_path.write_bytes(raw)
            os.chmod(manifest_path, 0o444)
            entry = next(item for item in closure["entries"] if item["path"] == str(manifest_path))
            entry["sha256"] = sha256_bytes(raw)
            entry["size"] = len(raw)
            entry["semantic_sha256"] = sha256_bytes(raw)
            closure["core_manifest_sha256"] = sha256_bytes(raw)
            _seal_synthetic_closure(closure)
            expected_stage = "MANIFEST_INCONSISTENT"
        runtime = root / "runtime"
        try:
            observed = core_preflight.verify_core_manifest_closure(closure, expected_core_manifest_path=manifest_path)
        except core_preflight.CoreManifestPreflightError as error:
            return {"name": name, "passed": mutation is not None and error.stage == expected_stage and not runtime.exists(), "failure_stage": error.stage, "invocation_calls": 0, "owner_created": runtime.exists()}
        if mutation is not None:
            return {"name": name, "passed": False, "failure_stage": "NOT_REJECTED", "invocation_calls": 0, "owner_created": runtime.exists()}
        paths = make_paths(runtime)
        outcome = run_execution(paths, bindings(), validators(), lambda: None, lambda: b"synthetic-payload", invoker("nominal"))
        BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)
        return {"name": name, "passed": observed["status"].startswith("PASS_") and outcome.status == "SUCCEEDED_TERMINAL" and outcome.counts.invocation_count == 1, "failure_stage": None, "invocation_calls": outcome.counts.invocation_count, "owner_created": paths.owner.exists()}


def _real_preflight_concurrent() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v20-real-preflight-race-") as temporary:
        root = Path(temporary)
        closure, manifest_path, _ = _synthetic_closure(root)
        paths = make_paths(root / "runtime")
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        outcomes = []
        invocation_calls = 0
        def worker() -> None:
            nonlocal invocation_calls
            core_preflight.verify_core_manifest_closure(closure, expected_core_manifest_path=manifest_path)
            barrier.wait(timeout=5)
            base_invoker = invoker("nominal")
            def invoke_once(*args: Any) -> Any:
                nonlocal invocation_calls
                with lock:
                    invocation_calls += 1
                return base_invoker(*args)
            outcome = run_execution(paths, bindings(), validators(), lambda: None, lambda: b"synthetic-payload", invoke_once)
            with lock:
                outcomes.append(outcome)
        threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
        for thread in threads: thread.start()
        barrier.wait(timeout=5)
        for thread in threads: thread.join(timeout=10)
        owners = [item for item in outcomes if not item.duplicate_rejected]
        losers = [item for item in outcomes if item.duplicate_rejected]
        loser_published = any(item.terminal_path is not None or item.terminal_publication_failed for item in losers)
        BRANCH_INVOCATION_COUNTS.extend(item.counts.invocation_count for item in outcomes)
        return {"name": "real_preflight_concurrent", "passed": len(owners) == 1 and len(losers) == 1 and invocation_calls == 1 and not loser_published, "invocation_calls": invocation_calls, "loser_published": loser_published}


def real_preflight_cases() -> list[dict[str, Any]]:
    cases = [_real_preflight_case("real_preflight_positive", None)]
    for mutation in ("absent", "wrong_path", "wrong_hash", "wrong_mode", "malformed", "inconsistent"):
        cases.append(_real_preflight_case(f"real_preflight_{mutation}", mutation))
    cases.append(_real_preflight_concurrent())
    return cases
'''


def patch_fixture(source: str) -> str:
    source = v20ize(source)
    source = source.replace(
        "import qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v20 as adapter\n",
        "import qk_gbfp8_head64_granularity_sweep_core_manifest_preflight_v20 as core_preflight\nimport qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v20 as adapter\n",
    )
    source = source.replace("\ndef load_schema(name: str) -> dict[str, Any]:\n", synthetic_preflight_helpers() + "\n\ndef load_schema(name: str) -> dict[str, Any]:\n")
    source = source.replace(
        '''def launcher_cwd_failure(paths: RuntimePaths) -> dict[str, Any]:\n    outcome = launcher.run_launcher(\n        paths=paths,\n        argv=launcher.expected_launcher_arguments(),\n        observed_cwd=Path("/synthetic-wrong-cwd"),\n        observed_environment=dict(launcher.EXPECTED_ENVIRONMENT),\n        schema_loader=validators,\n    )\n    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)\n    terminal = read_json(Path(outcome.terminal_path))\n    return {\n        "passed": (\n            outcome.status == "PREFLIGHT_FAILED_TERMINAL"\n            and outcome.counts.invocation_count == 0\n            and terminal["wrapper_diagnostic"]["failure_stage"] == "LAUNCHER_CWD"\n            and paths.owner.exists()\n            and not paths.authority.exists()\n        ),\n        "failure_stage": terminal["wrapper_diagnostic"]["failure_stage"],\n        "status": outcome.status,\n    }\n''',
        '''def launcher_cwd_failure(paths: RuntimePaths) -> dict[str, Any]:\n    try:\n        launcher.run_launcher(paths=paths, argv=launcher.expected_launcher_arguments(), observed_cwd=Path("/synthetic-wrong-cwd"), observed_environment=dict(launcher.EXPECTED_ENVIRONMENT), schema_loader=validators)\n    except launcher.LauncherError as error:\n        return {"passed": error.failure_stage == "LAUNCHER_CWD" and not paths.owner.exists() and not paths.authority.exists(), "failure_stage": error.failure_stage, "status": "REJECTED_BEFORE_OWNER"}\n    return {"passed": False, "failure_stage": "NOT_REJECTED", "status": "INVALID"}\n''',
    )
    source = source.replace(
        '''def launcher_setup_failure(paths: RuntimePaths) -> dict[str, Any]:\n    def fail_schema_setup() -> dict[str, Any]:\n        raise RuntimeError("synthetic schema setup")\n\n    outcome = launcher.run_launcher(\n        paths=paths,\n        argv=launcher.expected_launcher_arguments(),\n        observed_cwd=launcher.ROOT,\n        observed_environment=dict(launcher.EXPECTED_ENVIRONMENT),\n        schema_loader=fail_schema_setup,\n    )\n    BRANCH_INVOCATION_COUNTS.append(outcome.counts.invocation_count)\n    terminal = read_json(Path(outcome.terminal_path))\n    return {\n        "passed": (\n            outcome.status == "PREFLIGHT_FAILED_TERMINAL"\n            and outcome.counts.invocation_count == 0\n            and terminal["wrapper_diagnostic"]["failure_stage"] == "PREFLIGHT"\n            and paths.owner.exists()\n            and not paths.authority.exists()\n        ),\n        "failure_stage": terminal["wrapper_diagnostic"]["failure_stage"],\n        "status": outcome.status,\n    }\n''',
        '''def launcher_setup_failure(paths: RuntimePaths) -> dict[str, Any]:\n    def fail_schema_setup() -> dict[str, Any]:\n        raise RuntimeError("synthetic schema setup")\n    try:\n        launcher.run_launcher(paths=paths, argv=launcher.expected_launcher_arguments(), observed_cwd=launcher.ROOT, observed_environment=dict(launcher.EXPECTED_ENVIRONMENT), schema_loader=fail_schema_setup)\n    except RuntimeError:\n        return {"passed": not paths.owner.exists() and not paths.authority.exists(), "failure_stage": "SCHEMA_SETUP", "status": "REJECTED_BEFORE_OWNER"}\n    return {"passed": False, "failure_stage": "NOT_REJECTED", "status": "INVALID"}\n''',
    )
    source = source.replace("    passed = all(case[\"passed\"] for case in cases)\n", "    preflight_cases = real_preflight_cases()\n    cases.extend(preflight_cases)\n    passed = all(case[\"passed\"] for case in cases)\n")
    source = source.replace('        "official_payload_open_count": 0,\n', '        "official_payload_open_count": 0,\n        "real_preflight_case_count": len(preflight_cases),\n        "real_preflight_required_mutations": ["absent", "wrong_path", "wrong_hash", "wrong_mode", "malformed", "inconsistent"],\n')
    source = source.replace("PASS_V20_EXACTLY_ONCE_SYNTHETIC_FIXTURE", "PASS_V20_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE")
    source = source.replace("FAIL_V20_EXACTLY_ONCE_SYNTHETIC_FIXTURE", "FAIL_V20_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE")
    return source


def add_record(records: dict[str, dict[str, Any]], path: Path, source: str, *, expected_sha256: str | None = None, expected_size: int | None = None, deferred: bool = False) -> None:
    path = path.resolve()
    require(os.path.lexists(path), f"closure path absent: {path}")
    info = os.lstat(path)
    require(not stat.S_ISLNK(info.st_mode), f"closure symlink: {path}")
    key = str(path)
    if stat.S_ISDIR(info.st_mode):
        record = {"canonical_json": False, "hash_policy": "NOT_APPLICABLE", "kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": key, "semantic_sha256": None, "sha256": None, "size": None, "sources": [source]}
    else:
        require(stat.S_ISREG(info.st_mode), f"closure non-regular: {path}")
        if deferred:
            observed_sha = expected_sha256
            require(type(observed_sha) is str and len(observed_sha) == 64, f"deferred hash absent: {path}")
            canonical_json = False
            semantic_sha = None
        else:
            raw = path.read_bytes()
            observed_sha = hashlib.sha256(raw).hexdigest()
            if expected_sha256 is not None:
                require(observed_sha == expected_sha256, f"closure expected hash: {path}")
            canonical_json = False
            semantic_sha = None
            if path.suffix == ".json":
                try:
                    value = json.loads(raw.decode("ascii", "strict"))
                except Exception:
                    value = None
                if type(value) is dict and compact_bytes(value) == raw:
                    canonical_json = True
                    semantic_sha = sha256_bytes(compact_bytes(value))
        if expected_size is not None:
            require(info.st_size == expected_size, f"closure expected size: {path}")
        record = {"canonical_json": canonical_json, "hash_policy": "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN" if deferred else "FULL_READ_ONLY_PREFLIGHT", "kind": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": key, "semantic_sha256": semantic_sha, "sha256": observed_sha, "size": info.st_size, "sources": [source]}
    if key in records:
        records[key]["sources"] = sorted(set(records[key]["sources"] + [source]))
        require({k: v for k, v in records[key].items() if k != "sources"} == {k: v for k, v in record.items() if k != "sources"}, f"closure inconsistent duplicate: {path}")
    else:
        records[key] = record


def add_parent_directories(records: dict[str, dict[str, Any]], path: Path, root: Path, source: str) -> None:
    current = path.parent
    root = root.resolve()
    while current == root or root in current.parents:
        add_record(records, current, source)
        if current == root:
            break
        current = current.parent


def build_core_closure() -> dict[str, Any]:
    core, core_raw = canonical(CORE_MANIFEST)
    accepted_path = CORE_ROOT / "accepted/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_STATIC_V8_PACKAGE.canonical.json"
    accepted, _ = canonical(accepted_path)
    records: dict[str, dict[str, Any]] = {}
    add_record(records, CORE_ROOT, "core.root")
    add_record(records, CORE_MANIFEST, "core.manifest", expected_sha256=sha256_bytes(core_raw))
    for relative, binding in sorted(core["generated_files"].items()):
        path = CORE_ROOT / relative
        add_record(records, path, f"core.generated_files.{relative}", expected_sha256=binding["sha256"], expected_size=binding["size"])
        add_parent_directories(records, path, CORE_ROOT, "core.generated_parent")
    for label, preserved in sorted(core["preservation"].items()):
        root = Path(preserved["root"])
        add_record(records, root, f"core.preservation.{label}.root")
        for relative, binding in sorted(preserved.get("files", {}).items()):
            path = root / relative
            add_record(records, path, f"core.preservation.{label}.files.{relative}", expected_sha256=binding["sha256"], expected_size=binding["size"])
            require(records[str(path.resolve())]["mode"] == binding["mode"], f"closure preserved mode: {path}")
            add_parent_directories(records, path, root, f"core.preservation.{label}.parent")
    def absolute_strings(value: Any, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                absolute_strings(item, trail + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                absolute_strings(item, trail + (str(index),))
        elif isinstance(value, str) and value.startswith("/"):
            path = Path(value)
            if os.path.lexists(path):
                add_record(records, path, "core.absolute." + ".".join(trail), deferred=path.resolve() == OFFICIAL_PAYLOAD.resolve(), expected_sha256=base.V18_BINDING_SHA256 if False else (accepted["official_benchmark"]["input_bindings"]["tensor_bundle"]["sha256"] if path.resolve() == OFFICIAL_PAYLOAD.resolve() else None))
    absolute_strings(core)
    add_record(records, CORE_ACCEPTANCE, "launcher.core_acceptance")
    add_record(records, CORE_VERIFIER_REPORT, "launcher.core_verifier_report")
    add_record(records, V18_MANIFEST, "launcher.v18_manifest", expected_sha256=base.V18_MANIFEST_SHA256)
    add_record(records, V18_BINDING, "launcher.v18_binding", expected_sha256=base.V18_BINDING_SHA256)
    add_record(records, V18_ACCEPTANCE, "launcher.v18_acceptance", expected_sha256=base.V18_ACCEPTANCE_SHA256)
    add_record(records, V18_POST_REPORT, "launcher.v18_post_report")
    add_record(records, INTERPRETER, "launcher.interpreter", expected_sha256=EXPECTED_INTERPRETER_SHA256)
    add_record(records, accepted_path, "core.accepted_package")
    static_bindings = accepted["static_bindings"]
    for label in ("c02_parser", "controller", "evaluator", "result_schema", "verifier"):
        binding = static_bindings[label]
        path = STATIC_V8_ROOT / binding["path"]
        add_record(records, path, f"accepted.static_bindings.{label}", expected_sha256=binding["sha256"])
        add_parent_directories(records, path, STATIC_V8_ROOT, "accepted.static_parent")
    add_record(records, OFFICIAL_LANE_METADATA, "accepted.official_benchmark.lane_metadata", expected_sha256=accepted["official_benchmark"]["input_bindings"]["lane_metadata"]["sha256"], expected_size=accepted["official_benchmark"]["input_bindings"]["lane_metadata"]["byte_count"])
    add_record(records, OFFICIAL_PAYLOAD, "accepted.official_benchmark.tensor_bundle", expected_sha256=accepted["official_benchmark"]["input_bindings"]["tensor_bundle"]["sha256"], expected_size=accepted["official_benchmark"]["input_bindings"]["tensor_bundle"]["byte_count"], deferred=True)
    add_parent_directories(records, OFFICIAL_PAYLOAD, G16_ROOT, "accepted.official_payload_parent")
    entries = sorted(records.values(), key=lambda item: item["path"])
    deferred = sorted(item["path"] for item in entries if item["hash_policy"] == "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN")
    return {
        "core_manifest_mode": records[str(CORE_MANIFEST.resolve())]["mode"],
        "core_manifest_path": str(CORE_MANIFEST),
        "core_manifest_semantic_sha256": sha256_bytes(compact_bytes(core)),
        "core_manifest_sha256": sha256_bytes(core_raw),
        "core_manifest_self_sha256": core["package_content_sha256"],
        "deferred_hash_paths": deferred,
        "directory_count": sum(item["kind"] == "directory" for item in entries),
        "entries": entries,
        "entries_sha256": sha256_bytes(compact_bytes(entries)),
        "file_count": sum(item["kind"] == "file" for item in entries),
        "full_hash_file_count": sum(item["hash_policy"] == "FULL_READ_ONLY_PREFLIGHT" for item in entries),
        "official_evaluator_sha256": sha256_file(OFFICIAL_EVALUATOR),
        "official_parser_sha256": sha256_file(OFFICIAL_PARSER),
        "package_key": "core_manifest_closure",
        "path_hash_mode_complete": True,
        "schema_version": 1,
    }


def runtime_directories() -> list[str]:
    return [
        str(RUNTIME_ROOT), str(RUNTIME_ROOT / "primary"), str(RUNTIME_ROOT / "primary/authority"),
        str(RUNTIME_ROOT / "primary/authority/base"), str(RUNTIME_ROOT / "primary/result"),
        str(RUNTIME_ROOT / "primary/result/base"), str(RUNTIME_ROOT / "fallback"),
    ]


def build() -> dict[str, Any]:
    sys.addaudithook(base.audit_hook)
    require(not os.path.lexists(ACTION_ROOT), f"immutable V20 action root already exists: {ACTION_ROOT}")
    require(not os.path.lexists(BUILD_ROOT), f"immutable V20 build root already exists: {BUILD_ROOT}")
    require(not os.path.lexists(RUNTIME_ROOT), f"V20 runtime namespace already exists: {RUNTIME_ROOT}")
    require(sha256_file(V19_TERMINAL) == V19_TERMINAL_SHA256, "retired V19 terminal identity")
    v19_package, v19_raw = canonical(V19_MANIFEST)
    require(v19_package["action_identity"]["action_id"] == V19_ACTION_ID, "V19 action identity")
    v18_package, _ = canonical(V18_MANIFEST)
    v18_binding, binding_raw = canonical(V18_BINDING)
    require(v18_binding["record_count"] == 25 and v18_binding["selected_tensor_names"] == SELECTED_NAMES, "V18 binding contract")
    closure = build_core_closure()
    preservation = list(v19_package["preservation"])
    preservation.extend([inventory_digest(V19_ROOT), inventory_digest(V19_BUILD_ROOT), inventory_digest(V19_RUNTIME_ROOT)])
    before = {item["root"]: item for item in preservation}

    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{ROOT_ID}.staging-", dir=PROJECT_ROOT / "reference"))
    try:
        assert staging is not None
        schema_hashes = transformed_schemas(staging)
        preflight_path = staging / "tools" / PREFLIGHT_NAME
        write_bytes(preflight_path, preflight_source().encode("utf-8"))
        source_names = [
            "qk_gbfp8_head64_granularity_sweep_exactly_once_engine_v19.py",
            "qk_gbfp8_head64_granularity_sweep_evaluator_adapter_v19.py",
            "qk_gbfp8_head64_granularity_sweep_evaluator_return_path_v19.py",
            "qk_gbfp8_head64_granularity_sweep_result_validation_v19.py",
            "qk_gbfp8_head64_granularity_sweep_nominal_records_v19.py",
            "qk_gbfp8_head64_granularity_sweep_production_adapter_bridge_v19.py",
            "qk_gbfp8_head64_granularity_sweep_evaluator_worker_v19.py",
        ]
        for old_name in source_names:
            source = transformed_source("tools/" + old_name)
            new_name = old_name.replace("v19", "v20")
            compile(source, new_name, "exec")
            write_bytes(staging / "tools" / new_name, source.encode("utf-8"))
        launcher = patch_launcher((V19_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v19.py").read_text(encoding="utf-8"), schema_hashes)
        compile(launcher, "launcher_v20", "exec")
        launcher_path = staging / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py"
        write_bytes(launcher_path, launcher.encode("utf-8"))
        transport = patch_transport((V19_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v19.py").read_text(encoding="utf-8"), sha256_file(launcher_path))
        compile(transport, "transport_v20", "exec")
        write_bytes(staging / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py", transport.encode("utf-8"))
        fixture = patch_fixture((V19_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v19.py").read_text(encoding="utf-8"))
        compile(fixture, "fixture_v20", "exec")
        fixture_path = staging / "tools/qk_gbfp8_head64_granularity_sweep_exactly_once_fixture_v20.py"
        write_bytes(fixture_path, fixture.encode("utf-8"))
        write_bytes(staging / "bindings/C02_EXACT_BINDINGS_25.json", binding_raw)
        write_bytes(staging / "tools/verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py", ROOT_VERIFIER_SOURCE.read_bytes())

        completed = subprocess.run([str(INTERPRETER), str(fixture_path)], cwd=staging, env=dict(EXACT_ENVIRONMENT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        require(completed.returncode == 0 and completed.stderr == b"", f"V20 fixture failed: rc={completed.returncode} stderr={completed.stderr!r} stdout={completed.stdout!r}")
        fixture_report = json.loads(completed.stdout.decode("ascii", "strict"))
        require(compact_bytes(fixture_report) == completed.stdout, "V20 fixture canonical")
        require(fixture_report["status"] == "PASS_V20_BOUND_CORE_MANIFEST_SYNTHETIC_FIXTURE", "V20 fixture status")
        require(fixture_report["real_preflight_case_count"] == 8, "V20 real preflight case count")
        fixture_report_path = staging / "evidence" / FIXTURE_REPORT_NAME
        write_bytes(fixture_report_path, completed.stdout)

        relative_files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        generated_files = {relative: {"sha256": sha256_file(staging / relative), "size": (staging / relative).stat().st_size} for relative in relative_files}
        paths = {
            "owner": str(RUNTIME_ROOT / "primary/authority/base/owner-claim.json"),
            "authority": str(RUNTIME_ROOT / "primary/authority/base/authority.json"),
            "credential": str(RUNTIME_ROOT / "primary/authority/base/credential.json"),
            "ledger": str(RUNTIME_ROOT / "primary/authority/base/authority-ledger.json"),
            "result": str(RUNTIME_ROOT / "primary/result/base/result.json"),
            "first_terminal": str(RUNTIME_ROOT / "primary/authority/base/first-terminal.json"),
        }
        launcher_invocation = invocation_contract([str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_execute_once_wrapper_v20.py"), "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"), "--irreversible-action-id", ACTION_ID], ACTION_ROOT)
        transport_invocation = invocation_contract([str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_transport_wrapper_v20.py"), "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"), "--irreversible-action-id", ACTION_ID], ACTION_ROOT)
        production_invocation = invocation_contract([str(INTERPRETER), str(ACTION_ROOT / "tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v20.py"), "--mode", "production", "--package", str(ACTION_ROOT / MANIFEST_NAME), "--acceptance", str(ACTION_ROOT / "review/FRESH_L2_STATIC_ACCEPTANCE.json"), "--irreversible-action-id", ACTION_ID], ACTION_ROOT)
        payload_info = os.lstat(OFFICIAL_PAYLOAD)
        manifest = {
            "action_identity": {"action_id": ACTION_ID, "additive_successor": True, "future_action_id": ACTION_ID, "predecessor_action_id": V19_ACTION_ID, "retired_predecessor_terminal_sha256": V19_TERMINAL_SHA256, "retry_replay_resume_repair_replacement_permitted": False, "v18_modified_or_executed": False, "v19_modified_or_replayed": False},
            "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_package",
            "attempt": "0001",
            "claim_boundary": {"claim": "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "authority_materialized": False, "credential_materialized": False, "evaluator_invocations": 0, "execution_authorized": False, "ledger_materialized": False, "official_payload_open_count": 0, "official_target_process_starts": 0, "owner_claim_materialized": False, "result_materialized": False, "runtime_namespace_materialized": False, "runtime_terminal_materialized": False, "stage_2_activity": False, "v20_executed": False},
            "core_manifest_closure": closure,
            "generated_files": generated_files,
            "interpreter": {"path": str(INTERPRETER), "sha256": EXPECTED_INTERPRETER_SHA256, "version": "3.13.5"},
            "launcher_invocation": launcher_invocation,
            "lifecycle": {"atomic_owner_claim": "O_CREAT|O_EXCL_CREATE_ONCE_AFTER_SHARED_READ_ONLY_PREFLIGHT", "authority_create_count_maximum": 1, "concurrent_loser_disposition": "DUPLICATE_REJECTED_NO_PUBLICATION", "consumption_order": ["shared_complete_core_manifest_preflight", "private_runtime_namespace_prepare", "owner_claim_create_once", "authority_create_once", "credential_create_once", "ledger_create_once", "credential_durable_unlink_once", "payload_open_once", "evaluator_invoke_once"], "credential_consumption_count_maximum": 1, "duplicate_invocation_rejected_without_invocation": True, "first_terminal_publication": "OWNER_ONLY_PRIMARY_CREATE_THEN_FALLBACK_CREATE", "invocation_count_maximum": 1, "payload_open_count_maximum": 1, "post_consumption_failure": "CONSUMED_ORPHAN", "pre_owner_preflight_failure": "NO_RUNTIME_OR_LIFECYCLE_PUBLICATION", "result_publication": "CREATE_ONLY_FSYNC_FILE_AND_PARENT", "terminal_publication_count_maximum": 1},
            "numerical_selection": {"complete_parser_binding_count": 25, "evaluator_receives_complete_binding_table": True, "numerical_tensor_count": 3, "tensor_names": SELECTED_NAMES},
            "official_identities": {"input_tokens_sha256": v18_package["frozen_official_identity"]["input_token_ids_sha256"], "lane_metadata_sha256": v18_package["frozen_official_identity"]["lane_metadata_sha256"], "model_sha256": v18_package["frozen_official_identity"]["model_identity_sha256"], "tensor_bundle_sha256": v18_package["frozen_official_identity"]["tensor_bundle_sha256"]},
            "official_payload": {"access_during_static_verification": "PROHIBITED", "byte_count": payload_info.st_size, "hash_verification": "DEFERRED_TO_SINGLE_CONSUMING_PAYLOAD_OPEN", "path": str(OFFICIAL_PAYLOAD), "sha256": v18_package["frozen_official_identity"]["tensor_bundle_sha256"]},
            "preservation": preservation,
            "production_evaluator_invocation": production_invocation,
            "required_disposition": DISPOSITION,
            "root_id": ROOT_ID,
            "runtime_namespace": {"fallback_terminal": str(RUNTIME_ROOT / "fallback/first-terminal.json"), "file_count": 0, "paths": paths, "precreated_directories": runtime_directories(), "provisioning": "LAZY_PRIVATE_DIRECTORY_CREATION_AFTER_SHARED_PREFLIGHT", "runtime_root": str(RUNTIME_ROOT), "static_acceptance_namespace_absent": True},
            "schema_version": 1,
            "static_acceptance": {"acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json", "grants_execution_authority": False, "inventory_policy": "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE", "present": False, "schema_path": str(ACTION_ROOT / "reference" / ACCEPTANCE_SCHEMA_NAME)},
            "static_file_policy": {"acceptance_relative_path": "review/FRESH_L2_STATIC_ACCEPTANCE.json", "allowed_relative_files": sorted([MANIFEST_NAME, "review/FRESH_L2_STATIC_ACCEPTANCE.json", *generated_files]), "optional_before_acceptance": ["review/FRESH_L2_STATIC_ACCEPTANCE.json"]},
            "synthetic_fixture": {"case_count": fixture_report["case_count"], "official_payload_open_count": 0, "official_target_process_starts": 0, "production_mode_exercised": False, "real_preflight_case_count": fixture_report["real_preflight_case_count"], "report_file_sha256": sha256_file(fixture_report_path), "report_path": str(ACTION_ROOT / "evidence" / FIXTURE_REPORT_NAME), "synthetic_bytes_only": True},
            "transport_invocation": transport_invocation,
            "v18_binding": {"acceptance_file_sha256": base.V18_ACCEPTANCE_SHA256, "acceptance_self_sha256": base.V18_ACCEPTANCE_SELF_SHA256, "action_id": base.V18_ACTION_ID, "binding_table_file_sha256": base.V18_BINDING_SHA256, "integrated_binding_table_path": str(ACTION_ROOT / "bindings/C02_EXACT_BINDINGS_25.json"), "manifest_file_sha256": base.V18_MANIFEST_SHA256, "post_acceptance_report_self_sha256": base.V18_POST_REPORT_SELF_SHA256, "record_count": 25, "source_root": str(V18_ROOT)},
            "v19_retired_predecessor": {"action_id": V19_ACTION_ID, "manifest_file_sha256": sha256_bytes(v19_raw), "root": str(V19_ROOT), "runtime_root": str(V19_RUNTIME_ROOT), "terminal_file_sha256": V19_TERMINAL_SHA256},
            "verifier": {"acceptance_aware": True, "exact_argv": [str(INTERPRETER), str(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py")], "inventory_policy": "EXACT_STATIC_FILES_PLUS_ZERO_OR_ONE_BOUND_FRESH_L2_ACCEPTANCE", "path": str(ACTION_ROOT / "tools/verify_qk_gbfp8_head64_granularity_sweep_bound_path_repair_v20.py")},
        }
        manifest = sealed(manifest, "package_content_sha256")
        write_json(staging / MANIFEST_NAME, manifest)
        for path in sorted(staging.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(staging, 0o555)
        os.rename(staging, ACTION_ROOT)
        staging = None

        BUILD_ROOT.mkdir(parents=True, mode=0o755)
        report = sealed({"action_id": ACTION_ID, "artifact_kind": "qk_gbfp8_head64_granularity_sweep_execution_v20_bound_path_repair_build_report", "claim_boundary": "STATIC_ONLY_NO_EXECUTION_AUTHORITY", "core_manifest_closure_entries_sha256": closure["entries_sha256"], "core_manifest_closure_file_count": closure["file_count"], "fixture_report_file_sha256": sha256_file(ACTION_ROOT / "evidence" / FIXTURE_REPORT_NAME), "manifest_file_sha256": sha256_file(ACTION_ROOT / MANIFEST_NAME), "official_payload_open_count": base.AUDIT["official_payload_opens"], "official_target_process_starts": base.AUDIT["official_target_starts"], "preservation_root_count": len(preservation), "required_disposition": DISPOSITION, "runtime_namespace_file_count": 0, "status": "PASS_V20_STATIC_PACKAGE_BUILT_READY_FOR_INDEPENDENT_FRESH_L2", "v20_executed": False}, "report_sha256")
        write_json(BUILD_ROOT / "build-report.json", report)
        os.chmod(BUILD_ROOT / "build-report.json", 0o444)
        require(not os.path.lexists(RUNTIME_ROOT), "builder materialized V20 runtime namespace")
        require(base.AUDIT == {"official_payload_opens": 0, "official_target_starts": 0}, "builder official effect audit")
        for root, expected in before.items():
            require(inventory_digest(Path(root)) == expected, f"preservation drift after V20 build: {root}")
        return report
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    try:
        report = build()
    except Exception as error:
        sys.stderr.write(f"V20_BUILD_FAIL:{type(error).__name__}:{error}\n")
        return 1
    sys.stdout.buffer.write(compact_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
