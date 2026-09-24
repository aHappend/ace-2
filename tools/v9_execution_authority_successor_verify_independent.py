#!/usr/bin/env python3
"""Independent inert verifier for the static V9 execution-authority successor package."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/argustest/ace-2")
ROOT = PROJECT_ROOT / "build/v9-single-request-execution-authority-successor-package-v1-candidate-0004"
MANIFEST = ROOT / "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE.json"
SIDECAR = ROOT / "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE.json.sha256"
DISPOSITION = "V9_SINGLE_REQUEST_EXECUTION_AUTHORITY_SUCCESSOR_PACKAGE_READY_NO_AUTHORITY_CONSUMED"
PACKAGE_ID = "v9-single-request-execution-authority-successor-package-v1-candidate-0004"
BASE_IDENTITY = "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7"
ACTION_ID = "ace2:qk-gbfp8-base-v9:execute-once:2254dd91:20260813T2013Z"
INVOCATION_SHA256 = "e329556cae9d7903705c03ded4e3b347ed52007fa6e2c175887af820041e1c83"
LAUNCHER_INVOCATION_SHA256 = "55d685e3ca5420529c131e50fffcee276e15e96ff8726ea5e5dbcf40ee085635"
EXPECTED_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
EXPECTED_ARGV = [
    "/home/argustest/miniconda3/bin/python3.13",
    "/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root/tools/qk_gbfp8_head64_granularity_sweep_transport_shellfree_v9.py",
    "--package",
    "/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V9_SHELLFREE_PACKAGE.json",
    "--acceptance",
    "/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json",
    "--irreversible-action-id",
    ACTION_ID,
]
EXPECTED_BINDINGS = {
    "execution_domain": {
        "manifest_raw_sha256": "1b7d73ffef214dffa20f0b2a9a70c65ac04f75b152fdcabd378970260b1e071b",
        "package_content_sha256": "9066b2d9a91c94d464be3598a24977fef947adcfa5ae68ba7e222e457e17c9fa",
        "accepted_disposition": "V9_KERNEL_VERIFIABLE_EXECUTION_DOMAIN_PACKAGE_READY_NO_AUTHORITY_CONSUMED",
        "review_raw_sha256": "4f1252d4e8cb17ba8bbb8e70816975b1d90b61f27cd42435a8ae2d501ea5be49",
        "review_round": 7,
    },
    "direct_spawn": {
        "manifest_raw_sha256": "93c352e24120336203afb009301fa3b45f8ffec0ed32e65a80baca50630784ce",
        "package_content_sha256": "7c4923e7862167bd3b2fd52d5e8ee8d58d0288d79a5b6ed209f3cf6278fdb22b",
        "accepted_disposition": "PACKAGE_READY_NO_EXECUTION_AUTHORITY",
        "review_raw_sha256": "7368814cff8a0052e5fd73891551c9604d5778325c8ec7be0256266d6f5ad0b4",
        "review_round": 3,
    },
    "activation": {
        "manifest_raw_sha256": "db5a6ecaf9af67eb575cc413067842d39d78796834cddcc2adab34f4e158b1cf",
        "package_content_sha256": "dd18531f1047b0170ca2c1947701e55ca6ad8b2804d77ec5de9c5f7bab5ecbc8",
        "accepted_disposition": "V9_ONE_SHOT_ACTIVATION_V2_PACKAGE_READY_NO_EXECUTION_AUTHORITY",
        "review_raw_sha256": "193fdf5099ec2989b63cebbd31e78e04ea9ef0061464eda4c49fc90be6a3fcd7",
        "review_round": 3,
    },
    "broker": {
        "manifest_raw_sha256": "5e0a4798cf710ca7d25ad630db011022eab651dbd7283fb592d08209aed11ccf",
        "package_content_sha256": "3717d6969abf049a162c99ec474d200c91222baf6a18e30daaa06dad59a0ac68",
        "accepted_disposition": "BROKER_START_AUTHORITY_PACKAGE_READY_NO_AUTHORITY_CONSUMED",
        "review_raw_sha256": "ab58c972f8dfec59fe863693df69f9b383e8b62f079d82ac3df378ba0fd42341",
        "review_round": 2,
    },
    "canonical_v9": {
        "manifest_raw_sha256": "14fae526ce7c8c65e898eaa5400166855f214fb28dd676693632d3a85178727f",
        "package_content_sha256": "099be430faf60888b92ca5f358a21ee16f860360843a98f0f0944aeace47305f",
        "acceptance_raw_sha256": "8b969b27120243f8ce5787a6bd598a1d2f18da8b3e0728434177fcbbd89a42cb",
        "launcher_sha256": "0857368ec09ab6c69ea5b22482c34a214298e6f8bcf41c19a9bcf5b2494de513",
        "transport_sha256": "7e03a35881642bd5a93b7047ece920122534c7afab4bbf773bbc22d9ef69faf7",
        "static_verifier_sha256": "8e687659b4f8eb3f8262adf0e791ae2db662b8f2a9367d456ad48b6ee60552cc",
    },
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
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
    require(type(value) is dict and not duplicate and compact_bytes(value) == raw, f"noncanonical JSON: {path}")
    return value, raw


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    expected = value[field]
    require(type(expected) is str and len(expected) == 64, f"invalid self hash: {field}")
    observed = sha256_bytes(compact_bytes({key: item for key, item in value.items() if key != field}))
    require(observed == expected, f"self hash mismatch: {field}")


def verify_domain_content_hash(value: dict[str, Any], expected: str) -> None:
    require(value["package_content_sha256"] == expected, "domain content hash field")
    payload = {key: item for key, item in value.items() if key != "package_content_sha256"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    require(sha256_bytes(raw) == expected, "domain content hash")


def mode_octal(path: Path) -> str:
    return f"{stat.S_IMODE(os.lstat(path).st_mode):04o}"


def inventory(root: Path) -> dict[str, Any]:
    observed_root = os.lstat(root)
    require(stat.S_ISDIR(observed_root.st_mode) and not stat.S_ISLNK(observed_root.st_mode), f"invalid root: {root}")
    files: dict[str, Any] = {}
    directories: dict[str, Any] = {".": {"mode_octal": mode_octal(root)}}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        observed = os.lstat(path)
        require(not stat.S_ISLNK(observed.st_mode), f"symlink in inventory: {path}")
        if stat.S_ISDIR(observed.st_mode):
            directories[relative] = {"mode_octal": mode_octal(path)}
        else:
            require(stat.S_ISREG(observed.st_mode), f"non-regular inventory entry: {path}")
            files[relative] = {
                "byte_count": observed.st_size,
                "mode_octal": mode_octal(path),
                "sha256": sha256_file(path),
            }
    return {"directories": directories, "files": files}


def verify_review(path: Path, raw_sha256: str, disposition: str, round_index: int) -> None:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == raw_sha256, f"review hash mismatch: {path}")
    value = json.loads(raw.decode("utf-8"))
    require(value.get("producer_role") == "reviewer" and value.get("round") == round_index, f"review identity mismatch: {path}")
    review = value.get("review", {})
    require(review.get("status") == "done" and disposition in str(review.get("reason", "")), f"review disposition mismatch: {path}")


def verify_binding(label: str, binding: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_BINDINGS[label]
    for key, value in expected.items():
        require(binding.get(key) == value, f"{label} binding drift: {key}")
    root = Path(binding["root"])
    require(inventory(root) == binding["inventory"], f"{label} inventory drift")
    manifest_path = Path(binding["manifest_path"])
    manifest, raw = read_canonical(manifest_path)
    require(sha256_bytes(raw) == expected["manifest_raw_sha256"], f"{label} manifest raw hash")
    if label == "execution_domain":
        verify_domain_content_hash(manifest, expected["package_content_sha256"])
    else:
        verify_self_hash(manifest, "package_content_sha256")
        require(manifest["package_content_sha256"] == expected["package_content_sha256"], f"{label} content hash")
    if label != "canonical_v9":
        require(manifest.get("acceptance_boundary") == expected["accepted_disposition"], f"{label} acceptance boundary")
        verify_review(Path(binding["review_path"]), expected["review_raw_sha256"], expected["accepted_disposition"], expected["review_round"])
    return manifest


def literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                matches.append(ast.literal_eval(node.value))
    require(len(matches) == 1, f"literal assignment {name}: {path}")
    return matches[0]


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def call_argument(node: ast.Call, position: int, *keywords: str) -> ast.AST | None:
    if position < len(node.args):
        return node.args[position]
    for keyword in node.keywords:
        if keyword.arg in keywords:
            return keyword.value
    return None


def static_string_resolver(tree: ast.AST):
    assignments: dict[str, str] = {}

    def static_string(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and type(node.value) is str:
            return node.value
        if isinstance(node, ast.Name):
            return assignments.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = static_string(node.left)
            right = static_string(node.right)
            return None if left is None or right is None else left + right
        return None

    pending = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(pending) + 1):
        changed = False
        for node in pending:
            if node.value is None:
                continue
            value = static_string(node.value)
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and assignments.get(target.id) != value:
                    assignments[target.id] = value
                    changed = True
        if not changed:
            break
    return static_string


def dynamic_callable_oracle(tree: ast.AST):
    dynamic_symbols: set[str] = set()
    resolver_names = {"__import__", "get", "getattr", "globals", "locals", "vars"}

    def value_is_dynamic(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in dynamic_symbols
        if isinstance(node, ast.Starred):
            return value_is_dynamic(node.value)
        if isinstance(node, (ast.Tuple, ast.List)):
            return any(value_is_dynamic(item) for item in node.elts)
        if isinstance(node, ast.Subscript):
            return True
        if isinstance(node, ast.Attribute):
            return value_is_dynamic(node.value)
        if isinstance(node, ast.Call):
            return call_name(node) in resolver_names or callable_is_dynamic(node.func)
        if isinstance(node, ast.IfExp):
            return value_is_dynamic(node.body) or value_is_dynamic(node.orelse)
        return False

    def callable_is_dynamic(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in dynamic_symbols
        if isinstance(node, ast.Attribute):
            return value_is_dynamic(node.value)
        if isinstance(node, (ast.Call, ast.Subscript, ast.Lambda)):
            return True
        if isinstance(node, ast.IfExp):
            return value_is_dynamic(node)
        return True

    def assigned_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, ast.Starred):
            return assigned_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(name for item in target.elts for name in assigned_names(item))
        return ()

    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assignments) + 1):
        changed = False
        for assignment in assignments:
            if assignment.value is None or not value_is_dynamic(assignment.value):
                continue
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            for target in targets:
                for name in assigned_names(target):
                    if name not in dynamic_symbols:
                        dynamic_symbols.add(name)
                        changed = True
        if not changed:
            break
    return callable_is_dynamic


def target_alias(argument: str, marker: str) -> bool:
    if marker in argument:
        return True
    basename = Path(marker).name
    filename_characters = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    offset = 0
    while True:
        index = argument.find(basename, offset)
        if index < 0:
            return False
        before = argument[index - 1] if index else ""
        end = index + len(basename)
        after = argument[end] if end < len(argument) else ""
        if (not before or before not in filename_characters) and (not after or after not in filename_characters):
            return True
        offset = index + 1


def source_is_hazardous(source: str, markers: tuple[str, ...], depth: int = 0) -> bool:
    if depth > 4:
        return True
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError):
        return True
    static_string = static_string_resolver(tree)
    callable_is_dynamic = dynamic_callable_oracle(tree)
    for node in ast.walk(tree):
        value = static_string(node)
        if value is not None and any(target_alias(value, marker) for marker in markers):
            return True

    def code_is_hazardous(expression: ast.AST) -> bool:
        value = static_string(expression)
        if value is not None:
            return source_is_hazardous(value, markers, depth + 1)
        if isinstance(expression, ast.Call) and call_name(expression) == "compile":
            nested = call_argument(expression, 0, "source")
            return nested is None or code_is_hazardous(nested)
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute):
            if expression.func.attr in {"read", "read_bytes", "read_text"}:
                receiver = expression.func.value
                if isinstance(receiver, ast.Call) and call_name(receiver) in {"open", "Path"}:
                    value = static_string(call_argument(receiver, 0, "file"))
                    return value is None or any(target_alias(value, marker) for marker in markers)
        return True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if callable_is_dynamic(node.func):
            return True
        name = call_name(node)
        if name in {"exec", "eval"}:
            code = call_argument(node, 0, "source", "object")
            if code is None or code_is_hazardous(code):
                return True
        elif name in {"exec_module", "load_module"}:
            return True
        elif name == "spec_from_file_location":
            location = call_argument(node, 1, "location")
            value = static_string(location)
            if value is None or any(target_alias(value, marker) for marker in markers):
                return True
        elif name == "run_path":
            target = call_argument(node, 0, "path_name")
            value = static_string(target)
            if value is None or any(target_alias(value, marker) for marker in markers):
                return True
    return False


def verify_live_processes(markers: tuple[str, ...]) -> None:
    marker_paths = tuple(Path(marker).resolve(strict=True) for marker in markers)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]
        try:
            cwd = Path(os.readlink(entry / "cwd"))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            cwd = None
        try:
            executable = os.readlink(entry / "exe")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            executable = argv[0] if argv else ""
        for argument in argv:
            require(not any(target_alias(argument, marker) for marker in markers), f"target process marker: {argument}")
            candidate = Path(argument)
            if not candidate.is_absolute() and cwd is not None:
                candidate = cwd / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                resolved = None
            require(resolved not in marker_paths, f"resolved target process marker: {argument}")
        if Path(executable).name.lower().startswith(("python", "pypy")):
            inline = None
            for index, argument in enumerate(argv[1:], start=1):
                if argument == "-c":
                    inline = argv[index + 1] if index + 1 < len(argv) else ""
                    break
                if argument.startswith("-c") and len(argument) > 2:
                    inline = argument[2:]
                    break
                if argument == "--" or not argument.startswith("-"):
                    break
            if inline is not None:
                require(not source_is_hazardous(inline, markers), "hazardous inline target process")


def validate_manifest(manifest: dict[str, Any], *, check_external: bool) -> None:
    require(manifest["package_id"] == PACKAGE_ID, "package id")
    require(manifest["acceptance_boundary"] == DISPOSITION, "acceptance boundary")
    require(manifest["review_status"] == "PENDING_FRESH_L2", "review status")
    require(manifest["base_identity_sha256"] == BASE_IDENTITY, "Base identity")
    require(all(value is False for value in manifest["claim_boundary"].values()), "claim boundary")
    verify_self_hash(manifest, "package_content_sha256")

    request = manifest["request_descriptor"]
    verify_self_hash(request, "descriptor_sha256")
    require(request["action_id"] == ACTION_ID and request["argv"] == EXPECTED_ARGV, "request identity")
    require(request["cwd"] == str(PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v9_shellfree_action_root"), "request cwd")
    require(request["environment"] == EXPECTED_ENVIRONMENT and request["shell"] is False, "request environment/shell")
    require(request["invocation_sha256"] == INVOCATION_SHA256, "transport invocation hash")
    require(request["launcher_invocation_sha256"] == LAUNCHER_INVOCATION_SHA256, "launcher invocation hash")
    require(request["candidate_order"] == ["G8", "G4", "G2", "G1"], "candidate order")
    require(request["selection_policy"] == "FIRST_ALL_HARD_GATES_PASS_IN_FIXED_ORDER", "selection policy")
    require(request["request_cardinality"] == 1, "request cardinality")

    outer = manifest["execution_domain_publication_descriptor"]
    verify_self_hash(outer, "descriptor_sha256")
    require(outer["format"] == "V9ED-PUBLICATION-V3", "outer publication format")
    require(outer["status_values"] == ["CRASH_TERMINAL_CONSUMED", "REJECTED_TERMINAL_CONSUMED", "SUCCEEDED_TERMINAL", "UNKNOWN_TERMINAL_CONSUMED"], "outer status values")
    require(outer["request_consumption"] == "CONSUMED_ONCE" and outer["request_consumption_count"] == "1", "outer consumption identity")
    require(outer["request_queue_observation"] == "EMPTY_AFTER_CONSUMPTION", "outer queue identity")
    require(outer["authority_consumed"] == "false" and outer["production_state_created"] == "false", "outer authority/state identity")
    required_publication = {"request_id", "run_anchor_sha256", "domain_admission_sha256", "request_consumption", "request_consumption_count", "request_queue_observation", "status", "authority_consumed", "production_state_created", "record_sha256", "signature_ed25519"}
    require(required_publication <= set(outer["publication_fields"]), "outer publication identity fields")

    inner = manifest["canonical_v9_terminal_descriptor"]
    verify_self_hash(inner, "descriptor_sha256")
    require(inner["status_values"] == ["CONSUMED_ORPHAN", "FAILED_TERMINAL", "PREFLIGHT_FAILED_TERMINAL", "SUCCEEDED_TERMINAL"], "inner status values")
    require(inner["reason_code_values"] == ["HARD_THRESHOLD_FAILED", "HARD_THRESHOLDS_PASSED", "LIVE_MATERIALIZATION_FAILED", "POST_CONSUMPTION_AMBIGUITY", "READ_ONLY_PREFLIGHT_FAILED", "TRANSPORT_ATTESTATION_FAILED"], "inner reason codes")
    require(inner["schema_sha256"] == "54955aec6fcc9af52bb7c18468f16f1edfc8e2d407783d1d7b409edc2f55fa9b", "inner terminal schema")
    require(inner["first_terminal_create_only"] is True and inner["replacement_permitted"] is False, "inner terminal replacement")
    require(inner["result_terminal_status_values"] == ["FAILED_TERMINAL", "SUCCEEDED_TERMINAL"], "inner result statuses")

    cross_layer = manifest["cross_layer_terminal_contract"]
    require(cross_layer["mapping_policy"] == "NO_TOTAL_STATUS_MAPPING_CLAIMED_VALIDATE_LAYERS_INDEPENDENTLY", "cross-layer mapping policy")
    require(cross_layer["status_renaming_or_unification_permitted"] is False, "cross-layer status conflation")
    require(cross_layer["execution_domain_and_canonical_vocabularies_distinct"] is True, "cross-layer vocabulary separation")
    require(cross_layer["outer_success_requires_canonical_success_terminal"] is True, "cross-layer success implication")

    lifecycle = manifest["one_consumption_contract"]
    require(lifecycle["consumption_cardinality_maximum"] == 1, "consumption cardinality")
    require(lifecycle["request_replacement_permitted"] is False, "request replacement")
    require(lifecycle["terminal_publication_required_after_consumption"] is True, "terminal publication requirement")
    require(lifecycle["prohibited_operations"] == ["RETRY", "REPLAY", "RESUME", "REPAIR", "REPLACEMENT"], "prohibited operations")
    require(lifecycle["execution_domain_post_consumption_unknown"] == "UNKNOWN_TERMINAL_CONSUMED", "outer post-consumption ambiguity")
    require(lifecycle["canonical_v9_post_ledger_exception_terminal"] == "CONSUMED_ORPHAN", "inner post-ledger ambiguity")

    fd = manifest["interpreter_input_fd_binding"]
    verify_self_hash(fd, "binding_sha256")
    require(fd["source_or_ast_inference"] == "PROHIBITED", "source inference")
    require(fd["input_delivery"] == "SEALED_MEMFD_READ_ONLY_SECCOMP_IOCTL_NOTIF_ADDFD_NO_PATHNAME_CONTINUE", "input delivery")
    require(fd["exact_and_dynamic_bindings_required"] is True, "exact/dynamic binding")
    require(fd["missing_input_fd_bound_receipt_rejects"] is True, "missing FD receipt")
    require(fd["replacement_interpreter_input_inodes_consumed"] == 0, "replacement input inode")

    provenance = manifest["provenance_chain"]
    for label, expected in EXPECTED_BINDINGS.items():
        binding = provenance[label]
        for key, value in expected.items():
            require(binding.get(key) == value, f"{label} provenance: {key}")

    historical = manifest["canonical_historical_incompatibility"]
    require(historical["old_verifier_execution_permitted"] is False, "old verifier execution policy")
    require(historical["expected_failure"] == "FAIL V9 exact static file set", "historical failure")
    require(historical["current_acceptance_path"] == "review/FRESH_L2_STATIC_ACCEPTANCE.json", "acceptance path")
    require(historical["old_policy_omits_current_acceptance"] is True, "historical omission")

    if check_external:
        for label in EXPECTED_BINDINGS:
            verify_binding(label, provenance[label])
        for predecessor in manifest["preserved_predecessors"]:
            require(inventory(Path(predecessor["root"])) == predecessor["inventory"], f"predecessor drift: {predecessor['root']}")
        for path_text in manifest["forbidden_actual_state"]["paths"]:
            require(not os.path.lexists(path_text), f"forbidden state exists: {path_text}")


def expect_reject(manifest: dict[str, Any], mutate) -> None:
    candidate = copy.deepcopy(manifest)
    mutate(candidate)
    try:
        validate_manifest(candidate, check_external=False)
    except (KeyError, TypeError, VerificationError):
        return
    raise VerificationError("adversarial mutation accepted")


def adversarial_checks(manifest: dict[str, Any]) -> int:
    mutations = [
        lambda value: value["provenance_chain"]["execution_domain"].__setitem__("manifest_raw_sha256", "0" * 64),
        lambda value: value["provenance_chain"]["direct_spawn"].__setitem__("manifest_raw_sha256", "0" * 64),
        lambda value: value["request_descriptor"]["argv"].__setitem__(1, "/tmp/replacement.py"),
        lambda value: value["request_descriptor"].__setitem__("descriptor_sha256", "0" * 64),
        lambda value: value["execution_domain_publication_descriptor"]["status_values"].remove("CRASH_TERMINAL_CONSUMED"),
        lambda value: value["execution_domain_publication_descriptor"]["publication_fields"].remove("request_consumption_count"),
        lambda value: value["execution_domain_publication_descriptor"].__setitem__("request_consumption_count", "2"),
        lambda value: value["canonical_v9_terminal_descriptor"]["status_values"].remove("CONSUMED_ORPHAN"),
        lambda value: value["canonical_v9_terminal_descriptor"]["required_fields"].remove("first_terminal_sha256"),
        lambda value: value["canonical_v9_terminal_descriptor"].__setitem__("schema_sha256", "0" * 64),
        lambda value: value["canonical_v9_terminal_descriptor"].__setitem__("first_terminal_path", "/tmp/replacement-terminal.json"),
        lambda value: value["cross_layer_terminal_contract"].__setitem__("status_renaming_or_unification_permitted", True),
        lambda value: value["cross_layer_terminal_contract"].__setitem__("mapping_policy", "MERGED_STATUS_ENUM"),
        lambda value: value["one_consumption_contract"].__setitem__("consumption_cardinality_maximum", 2),
        lambda value: value["one_consumption_contract"].__setitem__("request_replacement_permitted", True),
        lambda value: value["one_consumption_contract"]["prohibited_operations"].remove("REPLAY"),
        lambda value: value["one_consumption_contract"].__setitem__("execution_domain_post_consumption_unknown", "CONSUMED_ORPHAN"),
        lambda value: value["one_consumption_contract"].__setitem__("canonical_v9_post_ledger_exception_terminal", "UNKNOWN_TERMINAL_CONSUMED"),
        lambda value: value["interpreter_input_fd_binding"].__setitem__("input_delivery", "PATHNAME_REOPEN"),
        lambda value: value["interpreter_input_fd_binding"].__setitem__("exact_and_dynamic_bindings_required", False),
        lambda value: value["interpreter_input_fd_binding"].__setitem__("replacement_interpreter_input_inodes_consumed", 1),
        lambda value: value["canonical_historical_incompatibility"].__setitem__("old_verifier_execution_permitted", True),
        lambda value: value["claim_boundary"].__setitem__("authority_consumed", True),
    ]
    for mutation in mutations:
        expect_reject(manifest, mutation)
    return len(mutations)


def main() -> int:
    observed_root = os.lstat(ROOT)
    require(stat.S_ISDIR(observed_root.st_mode) and not stat.S_ISLNK(observed_root.st_mode), "package root type")
    require(stat.S_IMODE(observed_root.st_mode) == 0o555, "package root mode")
    expected_files = {
        MANIFEST.name,
        SIDECAR.name,
        "REVIEW_CONTRACT.md",
        "verify_package_independent.py",
    }
    actual_files = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file() or path.is_symlink()}
    actual_directories = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_dir() and not path.is_symlink()}
    require(actual_files == expected_files and not actual_directories, "successor exact file set")
    for name in expected_files:
        observed = os.lstat(ROOT / name)
        require(stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode), f"package file type: {name}")
        require(stat.S_IMODE(observed.st_mode) == 0o444, f"package file mode: {name}")

    manifest, raw = read_canonical(MANIFEST)
    require(SIDECAR.read_text(encoding="ascii") == f"{sha256_bytes(raw)}  {MANIFEST.name}\n", "manifest sidecar")
    validate_manifest(manifest, check_external=True)
    artifact_paths = [item["path"] for item in manifest["artifact_bindings"]]
    require(artifact_paths == ["REVIEW_CONTRACT.md", "verify_package_independent.py"], "artifact binding order")
    for item in manifest["artifact_bindings"]:
        path = ROOT / item["path"]
        require(item == {"byte_count": path.stat().st_size, "mode_octal": mode_octal(path), "path": item["path"], "sha256": sha256_file(path)}, f"artifact binding: {path}")

    canonical_binding = manifest["provenance_chain"]["canonical_v9"]
    canonical = verify_binding("canonical_v9", canonical_binding)
    require(canonical["official_identities"]["model_sha256"] == BASE_IDENTITY, "canonical Base identity")
    require(canonical["future_invocation"] == {key: manifest["request_descriptor"][key] for key in ("argv", "command_representation", "cwd", "environment", "invocation_sha256", "shell")}, "canonical future invocation")
    require(canonical["launcher_invocation"]["invocation_sha256"] == LAUNCHER_INVOCATION_SHA256, "canonical launcher invocation")
    static_policy = canonical["static_file_policy"]
    require("review/FRESH_L2_STATIC_ACCEPTANCE.json" not in static_policy["allowed_relative_files"] and "review" in static_policy["forbidden_directories"], "historical static policy evidence")

    outer = manifest["execution_domain_publication_descriptor"]
    outer_source = Path(outer["source_path"])
    require(sha256_file(outer_source) == outer["source_sha256"], "outer publication verifier source hash")
    require(literal_assignment(outer_source, "PUBLICATION_FIELDS") == outer["publication_fields"], "outer publication fields vs source")
    outer_source_text = outer_source.read_text(encoding="utf-8")
    require(all(status_value in outer_source_text for status_value in outer["status_values"]), "outer status values vs source")
    require('publication["status"] == "SUCCEEDED_TERMINAL"' in outer_source_text, "outer success status source")
    require('publication["supervisor_wait_kind"] == "exit" and publication["supervisor_wait_value"] == "0"' in outer_source_text, "outer success exit source")

    inner = manifest["canonical_v9_terminal_descriptor"]
    inner_schema_path = Path(inner["schema_path"])
    require(sha256_file(inner_schema_path) == inner["schema_sha256"], "inner terminal schema hash")
    schema = json.loads(inner_schema_path.read_text(encoding="ascii"))
    require(schema["properties"]["status"]["enum"] == inner["status_values"], "inner statuses vs schema")
    require(schema["properties"]["reason_code"]["enum"] == inner["reason_code_values"], "inner reasons vs schema")
    require(schema["required"] == inner["required_fields"], "inner required fields vs schema")
    require(set(canonical["lifecycle"]["terminal_outcomes"]) == set(inner["status_values"]), "inner statuses vs canonical lifecycle")
    require(canonical["lifecycle"]["post_ledger_exception_terminal"] == "CONSUMED_ORPHAN", "canonical ambiguity status")
    require(Path(inner["first_terminal_path"]) == Path(canonical["live_namespace"]["paths"]["first_terminal"]), "inner terminal path vs canonical")
    require(Path(inner["result_path"]) == Path(canonical["live_namespace"]["paths"]["result"]), "inner result path vs canonical")
    require(set(outer["status_values"]) != set(inner["status_values"]), "outer/inner vocabularies are conflated")
    launcher_source = Path(canonical_binding["launcher_path"]).read_text(encoding="utf-8")
    require('return 0 if terminal["status"] == "SUCCEEDED_TERMINAL" else 1' in launcher_source, "canonical success return source")

    for label, binding in manifest["protected_policy_bindings"].items():
        require(sha256_file(Path(binding["path"])) == binding["sha256"], f"protected policy binding: {label}")

    execution_domain = verify_binding("execution_domain", manifest["provenance_chain"]["execution_domain"])
    contract = execution_domain["execution_domain_contract"]
    require(contract["request_consumption"] == "PEEK_VALIDATED_SINGLE_SOCK_SEQPACKET_PRIVATE_SEED_PACKET_ATOMIC_UNCLAIMED_TO_CONSUMED_ONCE_WITH_DUPLICATE_QUEUE_OBSERVER", "domain consumption semantics")
    require(contract["terminal_witness"] == "PRECONSUMPTION_REGISTERED_PIDFD_WITNESS_SEALS_ISSUER_RETAINED_TERMINAL_ON_EXECUTOR_EXIT_OR_CONTROL_LOSS", "domain terminal witness")
    require(contract["interpreter_input_identity"] == "VERIFIED_SOURCE_FD_TO_FULLY_SEALED_MEMFD_SNAPSHOT_THEN_SECCOMP_IOCTL_NOTIF_ADDFD_READ_ONLY_INJECTION_AT_ACTUAL_OPEN_WITHOUT_PATHNAME_CONTINUE", "domain input FD binding")
    require(contract["production_interpreter_binding_coverage"] == "EVERY_PRODUCTION_EXACT_OR_DYNAMIC_BINDING_PATH_APPEARS_IN_BOUND_ARGV_AND_MUST_EMIT_INPUT_FD_BOUND", "domain exact/dynamic coverage")
    require(contract["source_or_ast_inference"] == "PROHIBITED", "domain source inference")
    fd = manifest["interpreter_input_fd_binding"]
    domain_files = manifest["provenance_chain"]["execution_domain"]["inventory"]["files"]
    require(fd["production_contract_sha256"] == domain_files["production.contract"]["sha256"], "production contract FD binding hash")
    require(fd["verify_interpreter_fd_binding_sha256"] == domain_files["verify_interpreter_fd_binding_independent.py"]["sha256"], "exact FD verifier hash")
    require(fd["verify_dynamic_interpreter_fd_binding_sha256"] == domain_files["verify_dynamic_interpreter_fd_binding_independent.py"]["sha256"], "dynamic FD verifier hash")
    require(fd["verify_receipts_sha256"] == domain_files["verify_receipts_independent.py"]["sha256"], "receipt verifier hash")

    markers = tuple(manifest["forbidden_actual_state"]["target_argv_markers"])
    synthetic_bypasses = (
        "import os; exec(open(os.environ['TARGET'], encoding='utf-8').read())",
        "import builtins, os; (runner,) = (getattr(builtins, 'exec'),); runner(open(os.environ['TARGET'], encoding='utf-8').read())",
        "import importlib.util, os; spec = importlib.util.spec_from_file_location('worker', os.environ['TARGET']); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)",
    )
    require(all(source_is_hazardous(source, markers) for source in synthetic_bypasses), "process guard bypass oracle")
    verify_live_processes(markers)
    adversarial_count = adversarial_checks(manifest)
    print(json.dumps({
        "adversarial_cases": adversarial_count,
        "authority_consumed": False,
        "disposition": DISPOSITION,
        "independent_verifier_imports_candidate_code": False,
        "package_content_sha256": manifest["package_content_sha256"],
        "production_and_live_state_absent": True,
        "status": "PASS_V9_SUCCESSOR_INDEPENDENT_STATIC_VERIFICATION",
        "target_process_starts": 0,
    }, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
