#!/usr/bin/env python3
"""Static, zero-authority verifier for the diagnostic-Instruct v3 package."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/argustest/ace-2")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import ace2_exact_once_runtime_supervisor as supervisor  # noqa: E402


MISSION_ID = "freeze-instruct-rtl-diagnostic-exact-once-package-v3"
PACKAGE_ID = "freeze-instruct-rtl-diagnostic-exact-once-package-v3-20260809"
PACKAGE = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.json"
)
COMPANION = PACKAGE.with_name(
    "instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.provenance.json"
)
CHECKSUM_MANIFEST = PACKAGE.with_name(
    "instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.SHA256SUMS"
)
VERIFIER = Path(__file__).resolve()
TEST = TOOLS / "test_verify_instruct_rtl_diagnostic_exact_once_package_v3.py"

RUNTIME_OUTPUT = ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809-runtime-output"
STDOUT_PATH = ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.stdout.log"
STDERR_PATH = ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.stderr.log"
TERMINAL_PATH = ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.terminal.json"
STATE_DIR = ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.state"
AUTHORITY_RECORD_PATH = ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.authority.json"

V2_OUTPUT = ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001"
V2_EVIDENCE = (
    ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-terminal-evidence"
)
V2_AUTHORITY_ID = "mgr-exec-instruct-v2-ad1cd918-20260809T2227Z"


def _file(path: Path | str, byte_count: int, sha256: str) -> dict[str, Any]:
    return {"path": str(path), "byte_count": byte_count, "sha256": sha256}


EXPECTED_BOUND_FILES = [
    _file(ROOT / "tools/ace2_exact_once_runtime_supervisor.py", 34546, "635aceaa4e9a0907c3c1402360c531caf10b67da59a0f52955100e60ed45996d"),
    _file(ROOT / "tools/test_ace2_exact_once_runtime_supervisor.py", 12470, "37bbc0e2107e24cf5ee63a5230ae600ce33077f17cd3c7de1ede61feb63fff1c"),
    _file(ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness", 388296, "7f90ac671eabcefce7f95fdabba75de103298d3c3e0bd1465a357cf71fb1161d"),
    _file(ROOT / "Makefile", 45408, "5a8e08aa8d7773809ffaf32292ab008fb97ca9a10f258a1ad16c172ee6cdee78"),
    _file(ROOT / "verification/verilator/ace2_runtime_identity_profile.h", 2569, "773d2efcd209a60fd7a372beb1599a98d1bad888af47878f730fb30ab5cd68d5"),
    _file(ROOT / "verification/verilator/ace2_runtime_identity_profile_test.cpp", 3759, "e6933639193bdea418fb8085aba8c479fa439dd39875ec07945ccca851c1bfb6"),
    _file(ROOT / "verification/verilator/ace2_shell_runtime_harness.sv", 4986, "18d75af5c8818f81443543da44938fff68d19d777e6fc3111ea41724d4cda1f1"),
    _file(ROOT / "verification/verilator/ace2_shell_runtime_main.cpp", 128971, "89b0d16a9ca5a47e12c6483892c1987b116ea6e24b5e3106da445fbb4c2105a8"),
    _file(ROOT / "tools/run_full_qwen_command_schedule_runtime.py", 17906, "61b94d528fa256cba53faa398419b2c118b06cca32dbb8531be1bd0dc4c8ebb3"),
    _file(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/runtime_package.bin", 92917676, "3e3eae19d69d0b24b7bdfd0b5eb0f35b1cf663d22554f988f6035b99f8191295"),
    _file(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/ds32_prompt_applicability.json", 26986, "bdaa2b15200f09268c454d3daf92797fe85ccb97f729ec2a2d6cb8773eda9ed5"),
    _file(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/product_contract_preflight.json", 3709, "b1c0c6342ec77ce61bfba0626aca6146aabdc680a92440a497f45543fbc5d9d1"),
    _file(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/provenance.json", 21846, "e8105f946895a25d89ee2f1eb6dfc02f7f8e81befb65e0ea6595aff13671d9d7"),
    _file(ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/full_model_image.bin", 254421520, "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42"),
    _file(ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/image_contract_v2.json", 7347, "c8aa96b6fe660ff78f659eadfca7bf1785d391fb8b3c00ee4a940813604f466e"),
    _file(ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/validation_report.json", 5623, "d29779283c32a6093f833c6df3074daacfcf532b0ea0f4bf9d1d738f2d3d8a79"),
    _file(Path("/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"), 988097824, "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
    _file(Path("/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f"), 659, "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"),
    _file(Path("/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/443909a61d429dff23010e5bddd28ff530edda00"), 7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    _file(Path("/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/07bfe0640cb5a0037f9322287fbfc682806cf672"), 7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    _file(ROOT / "research/PIPELINE_STATE.json", 183412, "5a941e816d114438cd551e0405052e9686513931349c026728b47df715edbcba"),
    _file(ROOT / "constraints/ace2_rmsnorm_core.sdc", 250, "b21f8da39a0bdc5bf4db70b82013efb3d06a686347749c0070f37f37eca9e9ea"),
]

EXPECTED_SOURCE_TREE = {
    "path": str(ROOT / "rtl"),
    "algorithm": supervisor.SOURCE_TREE_ALGORITHM,
    "file_count": 26,
    "byte_count": 685737,
    "sha256": "01bd731570bf149d6173a3705b288718b18a5eeb718596a5da741b38162876d4",
}
EXPECTED_CONSTRAINT_TREE = {
    "path": str(ROOT / "constraints"),
    "algorithm": supervisor.SOURCE_TREE_ALGORITHM,
    "file_count": 1,
    "byte_count": 250,
    "sha256": "324ed614b2153730a56d6f2ce1c1d1035e2ca16be0d7a107ce3ac2ce3ee74c96",
}
EXPECTED_RUNTIME_SOURCE_TREE = {
    "path": str(ROOT / "verification/verilator"),
    "algorithm": supervisor.SOURCE_TREE_ALGORITHM,
    "file_count": 6,
    "byte_count": 163158,
    "sha256": "7743776bda2e0af16ccc0d2fbf62d22961996388cea61edb543568695790ec0c",
}
EXPECTED_V2_EVIDENCE_TREE = {
    "path": str(V2_EVIDENCE),
    "algorithm": supervisor.SOURCE_TREE_ALGORITHM,
    "file_count": 12,
    "byte_count": 22018,
    "sha256": "23965a2ff7bd05ce4d4782a18c2a935e3769ffce482cc0a5b671a458b6ac9662",
}

EXPECTED_V2_FILES = [
    _file(ROOT / "research/raw/specification/instruct-rtl-diagnostic-future-execution-package-20260809-v2.json", 17306, "ad1cd918378de3a39c9638443f392ed30dfb23e62799a33970ea0ce7b18e4913"),
    _file(ROOT / "research/raw/specification/instruct-rtl-diagnostic-future-execution-package-20260809-v2.json.sha256", 159, "7983a97769d06c12517cd7596e5ced08efdf29b67f170164efba7aef8ae27064"),
    _file(ROOT / "research/raw/specification/instruct-rtl-diagnostic-v2-package-fresh-review-binding-20260809.json", 1800, "07600adcc5a4f47ff95823ef2a83ac4e08de20626d603573d92c252418fc1e3b"),
    _file(ROOT / "research/raw/specification/instruct-rtl-diagnostic-v2-package-fresh-review-binding-20260809.json.sha256", 136, "d111173b387853c4e013950a19556d0ad35439e1a8d37c2e4db21b9c41edaf06"),
    _file(ROOT / "research/raw/specification/instruct-rtl-diagnostic-terminal-fresh-review-binding-20260809.json", 2722, "8565d07cb26d3b9d9d9c8052c970eb5626d60933f9787af6e92e8ce03ab51e03"),
    _file(ROOT / "research/raw/specification/instruct-rtl-diagnostic-terminal-fresh-review-binding-20260809.json.sha256", 134, "a0ab9541f7b3cc91c9936621eefdcc8a68c3012f50ca14c1b2aeac145ffadec3"),
    _file(ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-authority-preflight.json", 11156, "bc24bc81752e307899405e31c4068fe4f1ee5d9fe8a9f0e0685047445a36eccf"),
    _file(ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-executor.py", 22647, "8d8b7bcc5726e88f3911fef7fb6d56fcaf829072e38023aff1539b3c5fd95b79"),
]

SUPERVISOR_ACCEPTANCE = _file(
    Path("/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/implement-tested-exactly-once-runtime-supervisor-v1/CHECKPOINT.md"),
    3851,
    "4407cf3e4c2cdfcf83fefdd7a88b25026ecdabca15f8b02e46c043b6a4590cbf",
)

EXPECTED_ARGV = [
    str(ROOT / "build/verilator_full_qwen_runtime/Vace2_shell_runtime_harness"),
    "--identity-profile",
    "diagnostic-instruct",
    "--package",
    str(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/runtime_package.bin"),
    "--image",
    str(ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/full_model_image.bin"),
    "--model",
    "/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "--output",
    str(RUNTIME_OUTPUT),
    "--stop-after",
    "0",
    "--timeout-cycles",
    "100000000",
]
EXPECTED_ENVIRONMENT = {
    "HOME": "/home/argustest",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "TZ": "UTC",
}
EXPECTED_OUTPUTS = [
    str(RUNTIME_OUTPUT),
    str(STDOUT_PATH),
    str(STDERR_PATH),
    str(TERMINAL_PATH),
]
EXPECTED_ROLE_PATHS = {
    "accepted_non_hi_ace2rt2_package": EXPECTED_ARGV[4],
    "constraint_source": str(ROOT / "constraints/ace2_rmsnorm_core.sdc"),
    "image": EXPECTED_ARGV[6],
    "image_contract": str(ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/image_contract_v2.json"),
    "image_validation": str(ROOT / "evidence/verification/build-qwen2.5-0.5b-instruct-w4a8-image-v1/validation_report.json"),
    "makefile": str(ROOT / "Makefile"),
    "model": EXPECTED_ARGV[8],
    "model_config": "/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f",
    "pipeline_state": str(ROOT / "research/PIPELINE_STATE.json"),
    "prepare_applicability": str(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/ds32_prompt_applicability.json"),
    "prepare_preflight": str(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/product_contract_preflight.json"),
    "prepare_provenance": str(ROOT / "build/audit-arbitrary-text-chat-product-gap-v1/instruct-prepare/provenance.json"),
    "runtime_binary": EXPECTED_ARGV[0],
    "runtime_harness": str(ROOT / "verification/verilator/ace2_shell_runtime_harness.sv"),
    "runtime_identity_header": str(ROOT / "verification/verilator/ace2_runtime_identity_profile.h"),
    "runtime_identity_test": str(ROOT / "verification/verilator/ace2_runtime_identity_profile_test.cpp"),
    "runtime_main": str(ROOT / "verification/verilator/ace2_shell_runtime_main.cpp"),
    "runtime_schedule_source": str(ROOT / "tools/run_full_qwen_command_schedule_runtime.py"),
    "supervisor_source": str(ROOT / "tools/ace2_exact_once_runtime_supervisor.py"),
    "supervisor_tests": str(ROOT / "tools/test_ace2_exact_once_runtime_supervisor.py"),
    "tokenizer": "/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/443909a61d429dff23010e5bddd28ff530edda00",
    "tokenizer_config": "/home/argustest/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/07bfe0640cb5a0037f9322287fbfc682806cf672",
}

COMPANION_KEYS = {
    "schema",
    "mission_id",
    "generated_at_utc",
    "package",
    "supervisor_contract",
    "execution_binding",
    "input_roles",
    "tree_bindings",
    "v3_namespace",
    "sealed_v2",
    "authority_and_process_policy",
    "prohibitions",
    "review",
}
PROHIBITIONS = [
    "authority creation",
    "child process start",
    "GPU, S7, FPGA, scheduler, or network activity",
    "RTL or constraints edit",
    "runtime or supervisor invocation",
    "stage transition or product, latency, generation, decode, or RTL-agreement claim",
    "v2 package, authority identifier, output, executor, preflight, or terminal-evidence reuse",
    "v2 replay, continuation, repair, resume, or retry",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise supervisor.SupervisorError(message)


def _binding_by_path(bindings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {binding["path"]: binding for binding in bindings}


def _verify_file(binding: dict[str, Any], label: str) -> None:
    supervisor._validate_file_binding(binding, label)


def _verify_tree(binding: dict[str, Any], label: str) -> None:
    _require(binding["algorithm"] == supervisor.SOURCE_TREE_ALGORITHM, f"{label} algorithm")
    measured = supervisor.measure_source_tree(Path(binding["path"]))
    _require(measured.file_count == binding["file_count"], f"{label} file count")
    _require(measured.byte_count == binding["byte_count"], f"{label} byte count")
    _require(measured.sha256 == binding["sha256"], f"{label} SHA-256")


def _require_read_only(path: Path, *, recursive: bool = False) -> None:
    paths = [path]
    if recursive:
        paths.extend(sorted(path.rglob("*")))
    for child in paths:
        mode = stat.S_IMODE(os.lstat(child).st_mode)
        _require(mode & 0o222 == 0, f"protected path is writable: {child}")


def _nul_argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(b"".join(item.encode("utf-8") + b"\0" for item in argv)).hexdigest()


def _verify_no_execution_calls() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"run_once", "validate_run", "_reserve_pre_start_intent", "Popen", "run", "call"}
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    _require(not (called & forbidden), f"verifier contains execution call(s): {sorted(called & forbidden)}")


def _verify_checksum_manifest() -> dict[str, str]:
    measurement, content = supervisor._read_regular_file(
        CHECKSUM_MANIFEST, "v3 checksum manifest", capture_content=True
    )
    _require(content is not None, "checksum manifest content missing")
    expected_paths = [PACKAGE, COMPANION, VERIFIER, TEST]
    parsed: dict[str, str] = {}
    for line in content.decode("utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        _require(bool(separator), "malformed checksum manifest line")
        _require(relative not in parsed, "duplicate checksum manifest path")
        parsed[relative] = digest
    expected_relative = {str(path.relative_to(ROOT)) for path in expected_paths}
    _require(set(parsed) == expected_relative, "checksum manifest path set mismatch")
    for path in expected_paths:
        relative = str(path.relative_to(ROOT))
        _require(supervisor.measure_file(path).sha256 == parsed[relative], f"checksum mismatch: {relative}")
    _require(measurement.byte_count > 0, "empty checksum manifest")
    return parsed


def verify() -> dict[str, Any]:
    _verify_no_execution_calls()

    package, package_measurement = supervisor._load_json_measurement(PACKAGE, "v3 package")
    supervisor._exact_keys(package, supervisor.PACKAGE_KEYS, "v3 package")
    _require(package["schema"] == supervisor.PACKAGE_SCHEMA, "v3 package schema mismatch")
    _require(package["package_id"] == PACKAGE_ID, "v3 package_id mismatch")
    source_root, outputs = supervisor._validate_bindings(package["bindings"], "v3 bindings")
    _require(package["bindings"]["bound_files"] == EXPECTED_BOUND_FILES, "bound file set/order mismatch")
    _require(package["bindings"]["source_tree"] == EXPECTED_SOURCE_TREE, "RTL source tree mismatch")
    _require(package["bindings"]["cwd"] == str(ROOT), "cwd mismatch")
    _require(package["bindings"]["argv"] == EXPECTED_ARGV, "argv mismatch")
    _require(package["bindings"]["argv_sha256"] == supervisor.canonical_value_sha256(EXPECTED_ARGV), "argv digest mismatch")
    _require(package["bindings"]["environment"] == EXPECTED_ENVIRONMENT, "environment mismatch")
    _require(package["bindings"]["required_absent_outputs"] == EXPECTED_OUTPUTS, "required outputs mismatch")
    supervisor._check_outputs_absent(outputs)
    _require(os.access(EXPECTED_ARGV[0], os.X_OK), "runtime binary is not executable")

    companion, companion_measurement = supervisor._load_json_measurement(
        COMPANION, "v3 provenance companion"
    )
    supervisor._exact_keys(companion, COMPANION_KEYS, "v3 provenance companion")
    _require(companion["schema"] == "ace2-exact-once-execution-package-provenance-review-v1", "companion schema mismatch")
    _require(companion["mission_id"] == MISSION_ID, "companion mission mismatch")
    expected_package_record = {
        "path": str(PACKAGE),
        "byte_count": package_measurement.byte_count,
        "sha256": package_measurement.sha256,
        "package_id": PACKAGE_ID,
        "schema": supervisor.PACKAGE_SCHEMA,
    }
    _require(companion["package"] == expected_package_record, "companion package binding mismatch")

    contract = companion["supervisor_contract"]
    _require(contract["source"] == EXPECTED_BOUND_FILES[0], "supervisor source provenance mismatch")
    _require(contract["tests"] == EXPECTED_BOUND_FILES[1], "supervisor test provenance mismatch")
    _require(contract["acceptance_checkpoint"] == SUPERVISOR_ACCEPTANCE, "supervisor acceptance provenance mismatch")
    _require(contract["fresh_l2_status"] == "ACCEPTED", "supervisor Fresh L2 status mismatch")
    _require(contract["package_schema"] == supervisor.PACKAGE_SCHEMA, "supervisor package schema mismatch")
    _require(contract["source_tree_algorithm"] == supervisor.SOURCE_TREE_ALGORITHM, "source-tree algorithm mismatch")
    _verify_file(SUPERVISOR_ACCEPTANCE, "supervisor acceptance checkpoint")

    bound_by_path = _binding_by_path(package["bindings"]["bound_files"])
    expected_roles = {role: bound_by_path[path] for role, path in EXPECTED_ROLE_PATHS.items()}
    _require(companion["input_roles"] == expected_roles, "input role provenance mismatch")
    _require(set(bound_by_path) == set(EXPECTED_ROLE_PATHS.values()), "unclassified bound file")

    expected_trees = {
        "package_rtl_source_tree": EXPECTED_SOURCE_TREE,
        "constraints_tree": EXPECTED_CONSTRAINT_TREE,
        "runtime_source_tree": EXPECTED_RUNTIME_SOURCE_TREE,
    }
    _require(companion["tree_bindings"] == expected_trees, "companion tree bindings mismatch")
    _verify_tree(EXPECTED_CONSTRAINT_TREE, "constraints tree")
    _verify_tree(EXPECTED_RUNTIME_SOURCE_TREE, "runtime source tree")

    expected_execution = {
        "cwd": str(ROOT),
        "argv": EXPECTED_ARGV,
        "argv_sha256": supervisor.canonical_value_sha256(EXPECTED_ARGV),
        "nul_separated_argv_sha256": _nul_argv_sha256(EXPECTED_ARGV),
        "environment": EXPECTED_ENVIRONMENT,
        "identity_profile_option_count": 1,
        "identity_profile": "diagnostic-instruct",
        "stop_after": 0,
        "timeout_cycles": 100000000,
        "maximum_generated_token_count": 8,
        "format": "ACE2RT2",
        "prompt_class": "non-Hi diagnostic Instruct",
    }
    _require(companion["execution_binding"] == expected_execution, "execution binding mismatch")
    _require(EXPECTED_ARGV.count("--identity-profile") == 1, "identity option cardinality")
    profile_index = EXPECTED_ARGV.index("--identity-profile")
    _require(EXPECTED_ARGV[profile_index + 1] == "diagnostic-instruct", "identity profile mismatch")
    _require(EXPECTED_ARGV[EXPECTED_ARGV.index("--stop-after") + 1] == "0", "stop-after mismatch")
    _require(EXPECTED_ARGV[EXPECTED_ARGV.index("--timeout-cycles") + 1] == "100000000", "timeout mismatch")
    forbidden_argv = {"--resume", "--inspect-package", "--inspect-journal", "base", "python", "python3"}
    _require(not (set(EXPECTED_ARGV) & forbidden_argv), "Base/software/replay fallback in argv")

    namespace = companion["v3_namespace"]
    expected_runtime = {
        "state_dir": str(STATE_DIR),
        "stdout_path": str(STDOUT_PATH),
        "stderr_path": str(STDERR_PATH),
        "terminal_record_path": str(TERMINAL_PATH),
    }
    expected_absent = [str(STATE_DIR), *EXPECTED_OUTPUTS, str(AUTHORITY_RECORD_PATH)]
    _require(namespace["classification"] == "independent new v3 namespace; not replay or continuation", "v3 namespace classification")
    _require(namespace["supervisor_runtime"] == expected_runtime, "future supervisor runtime mismatch")
    _require(namespace["runtime_output_path"] == str(RUNTIME_OUTPUT), "runtime output mismatch")
    _require(namespace["external_authority_record_path"] == str(AUTHORITY_RECORD_PATH), "authority path mismatch")
    _require(namespace["all_paths_required_absent"] == expected_absent, "absent namespace path set mismatch")
    _require(namespace["v2_namespace_reused"] is False, "v2 namespace reuse")
    supervisor._validate_runtime(expected_runtime, outputs, source_root)
    supervisor._check_state_available(STATE_DIR)
    _require(all(not os.path.lexists(path) for path in expected_absent), "v3 namespace path exists")

    sealed_v2 = companion["sealed_v2"]
    _require(sealed_v2["status"] == "TERMINALLY_CONSUMED_NO_REUSE", "v2 terminal status")
    _require(sealed_v2["attempt"] == "attempt-0001", "v2 attempt mismatch")
    _require(sealed_v2["authority_id"] == V2_AUTHORITY_ID, "v2 authority mismatch")
    _require(sealed_v2["protected_files"] == EXPECTED_V2_FILES, "v2 protected file set mismatch")
    _require(sealed_v2["terminal_evidence_tree"] == EXPECTED_V2_EVIDENCE_TREE, "v2 evidence tree mismatch")
    _require(sealed_v2["runtime_output_path"] == str(V2_OUTPUT), "v2 output path mismatch")
    _require(sealed_v2["runtime_output_expected_absent"] is True, "v2 output absence policy")
    _require(sealed_v2["read_only_enforced"] is True, "v2 read-only policy")
    _require(set(sealed_v2["prohibited_reuse"]) == {binding["path"] for binding in EXPECTED_V2_FILES} | {str(V2_EVIDENCE), str(V2_OUTPUT), V2_AUTHORITY_ID}, "v2 prohibited-reuse set mismatch")
    for index, binding in enumerate(EXPECTED_V2_FILES):
        _verify_file(binding, f"sealed v2 file {index}")
        _require_read_only(Path(binding["path"]))
    _verify_tree(EXPECTED_V2_EVIDENCE_TREE, "sealed v2 terminal evidence")
    _require_read_only(V2_EVIDENCE, recursive=True)
    _require(not os.path.lexists(V2_OUTPUT), "sealed v2 output path changed")

    expected_policy = {
        "external_authority_record_present": False,
        "execution_authority_granted": False,
        "fresh_l2_package_acceptance_grants_execution_authority": False,
        "permitted_process_starts": 0,
        "state_reservation_permitted": False,
        "supervisor_cli_permitted": False,
        "run_once_permitted": False,
        "runtime_invocation_permitted": False,
    }
    _require(companion["authority_and_process_policy"] == expected_policy, "zero-authority policy mismatch")
    _require(companion["prohibitions"] == PROHIBITIONS, "prohibition set mismatch")

    review = companion["review"]
    _require(review["status"] == "PENDING_FRESH_L2", "Fresh L2 review status must remain pending")
    _require(review["checksum_manifest_path"] == str(CHECKSUM_MANIFEST), "review checksum path mismatch")
    _require(review["required_acceptance"] == "exact package SHA-256 and exact provenance companion SHA-256", "review acceptance binding mismatch")
    _require(review["acceptance_effect"] == "package review only; zero execution and zero stage authority", "review authority boundary mismatch")

    checksums = _verify_checksum_manifest()
    for path in (PACKAGE, COMPANION, CHECKSUM_MANIFEST, VERIFIER, TEST):
        _require_read_only(path)

    return {
        "status": "PASS_STATIC_ZERO_AUTHORITY",
        "mission_id": MISSION_ID,
        "package_sha256": package_measurement.sha256,
        "companion_sha256": companion_measurement.sha256,
        "checksum_manifest_entries": len(checksums),
        "bound_file_count": len(EXPECTED_BOUND_FILES),
        "rtl_source_tree_sha256": EXPECTED_SOURCE_TREE["sha256"],
        "constraint_tree_sha256": EXPECTED_CONSTRAINT_TREE["sha256"],
        "runtime_source_tree_sha256": EXPECTED_RUNTIME_SOURCE_TREE["sha256"],
        "sealed_v2_evidence_tree_sha256": EXPECTED_V2_EVIDENCE_TREE["sha256"],
        "v3_absent_path_count": len(expected_absent),
        "authority_records": 0,
        "state_reservations": 0,
        "process_starts": 0,
        "runtime_executed": False,
        "rtl_or_constraints_edited": False,
        "stage_transition": False,
    }


def main() -> int:
    try:
        report = verify()
    except (OSError, ValueError, supervisor.SupervisorError) as exc:
        print(json.dumps({"status": "FAIL_STATIC", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
