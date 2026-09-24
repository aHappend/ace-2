#!/usr/bin/python3
"""Build the inert v6u cwd-bound successor without invoking either capsule."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


ROOT = Path("/home/argustest/ace-2")
V6T = ROOT / "build/s7-v6t-rapid-clock-safe-successor-offline-20260811-6b82c1c53b0a"
V6T_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "prepare-v6t-runnable-pair-reviewed"
)
MISSION_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "seal-v6t-cwd-failure-build-v6u-reviewed"
)
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
EXECUTION_WORKDIR = "/home/argustest/ace-2"
WRONG_CWD = "/home/argustest/argustest2"
GROUND_TRUTH_SHA256 = "70cd8aae4376ceee1454a08d8e31ab1254b8409dfc29e554f3f902dabc8863d0"
V6T_CAPSULE_SHA256 = "8e1f5dbcfef09dfba9277c4132d4c338441a28770bea518ff6f4e21107ea7fa1"
V6T_CONSTRUCTOR_SHA256 = "8c56a517d026a11db072b1506a9ff5f28ec8b290ef8234273c7421a81ddca94e"
V6T_CONSTRUCTOR_AUTH_SHA256 = "cd6ca2164109aac007948e1f67fc217fec686760aff31c39185b2e32a5dd0295"
V6T_LAUNCH_AUTH_SHA256 = "0f84429535238d96304c2f7971d299ffb7bb12c41bcfbbe483e9fdf8c3572cf4"
V6T_RUNNABLE_REVIEW_SHA256 = "9e6ff69c837722732ca0b0e4317995418c4bef53d223f17d0a8641ed2aefe703"
V6T_RUNNABLE_VERIFICATION_SHA256 = "776b48334fea314dced41fe2ece3b3e35019cbbe77e63e1866cf901e0f51a67f"
V6T_ACCEPTANCE_SHA256 = "85fed7115d7ab1f990563afe0720d51aa4f0276d3c1af050e2e598c3e91a875a"
AMLT_STATIC_BINDING_SHA256 = "de521b9509592bf30033e284acdb51fb8184c4658651a2616f37b5cce6a6c2fb"
PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_file(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(fd, mode)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError(f"short write: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.read_bytes() != payload:
        raise RuntimeError(f"exact reread mismatch: {path}")


def mkdir_exclusive(path: Path, mode: int) -> None:
    os.mkdir(path, mode)
    fsync_directory(path.parent)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"unsafe directory: {path}")


def transform_string(value: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def transform_value(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        return transform_string(value, replacements)
    if isinstance(value, list):
        return [transform_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: transform_value(item, replacements) for key, item in value.items()}
    return value


def function_lines(path: Path, names: tuple[str, ...]) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = node.lineno
    return found


def build() -> dict[str, Any]:
    if sha256_file(GROUND_TRUTH) != GROUND_TRUTH_SHA256:
        raise RuntimeError("protected GROUND_TRUTH hash mismatch")
    old_capsule = V6T / "launch_capsule_v6t.py"
    old_constructor = V6T / "construct_manager_authority_v6t.py"
    old_constructor_auth = Path(
        "/home/argustest/.local/state/ace2/"
        "s7-v6t-authority-authorizations-20260811-feb893334d2f/"
        "mgr-s7-reservation-submit-v6t-20260811-4fcca8a6c42c/"
        "constructor-authorization-source.json"
    )
    old_launch_auth = old_constructor_auth.with_name("launch-authorization-source.json")
    if sha256_file(old_capsule) != V6T_CAPSULE_SHA256:
        raise RuntimeError("accepted v6t capsule changed")
    if sha256_file(old_constructor) != V6T_CONSTRUCTOR_SHA256:
        raise RuntimeError("accepted v6t constructor changed")
    if sha256_file(old_constructor_auth) != V6T_CONSTRUCTOR_AUTH_SHA256:
        raise RuntimeError("accepted v6t constructor authorization changed")
    if sha256_file(old_launch_auth) != V6T_LAUNCH_AUTH_SHA256:
        raise RuntimeError("accepted v6t launch authorization changed")
    if sha256_file(V6T_HANDOFF / "fresh-l2-runnable-offline-review.json") != V6T_RUNNABLE_REVIEW_SHA256:
        raise RuntimeError("accepted v6t runnable review changed")
    if sha256_file(V6T_HANDOFF / "v6t-runnable-offline-verification.json") != V6T_RUNNABLE_VERIFICATION_SHA256:
        raise RuntimeError("accepted v6t runnable verification changed")

    build_ns = dt.datetime.now(dt.timezone.utc)
    build_ns_text = build_ns.strftime("%Y%m%dT%H%M%SZ")
    entropy = f"v6u:{build_ns.timestamp()}:{os.getpid()}".encode()
    package_suffix = sha256_bytes(entropy)[:12]
    package_name = f"s7-v6u-cwd-bound-successor-offline-20260811-{package_suffix}"
    package = ROOT / "build" / package_name
    package_id = f"s7-v6u-offline-successor-package-20260811-{package_suffix}"
    authority_suffix = sha256_bytes(b"authority:" + entropy)[:12]
    authority_id = f"mgr-s7-reservation-submit-v6u-20260811-{authority_suffix}"
    constructor_suffix = sha256_bytes(b"constructor:" + entropy)[:12]
    constructor_id = f"s7-v6u-six-probe-constructor-20260811-{constructor_suffix}"
    capsule_suffix = sha256_bytes(b"capsule:" + entropy)[:12]
    capsule_id = f"s7-v6u-origin-exec-capsule-20260811-{capsule_suffix}"
    attempt_suffix = sha256_bytes(b"attempt:" + entropy)[:12]
    attempt_namespace = f"s7-v6u-authority-attempt-20260811-{attempt_suffix}"
    gate_suffix = sha256_bytes(b"gate:" + entropy)[:12]
    gate_budget_id = f"s7-v6u-six-probe-budget-20260811-{gate_suffix}"
    experiment_label = f"ace2-s7-rsv-v6u-20260811-{sha256_bytes(b'experiment:' + entropy)[:12]}"
    job_label = f"owned-s7-2xg4-24h-v6u-20260811-{sha256_bytes(b'job:' + entropy)[:12]}"

    auth_namespace = Path(
        "/home/argustest/.local/state/ace2/"
        f"s7-v6u-authority-authorizations-20260811-{sha256_bytes(b'auth-root:' + entropy)[:12]}"
    )
    auth_root = auth_namespace / authority_id
    state_parent = Path(
        "/home/argustest/.local/state/ace2/"
        f"s7-v6u-authority-construction-20260811-{sha256_bytes(b'state-root:' + entropy)[:12]}"
    )
    state_dir = state_parent / attempt_namespace
    future_execution = Path(
        "/home/argustest/.local/state/ace2/"
        f"s7-v6u-reservation-execution-authorizations-20260811-{sha256_bytes(b'future-root:' + entropy)[:12]}"
    ) / authority_id / "authorization-source.json"
    origin_record = state_parent / f".{authority_id}.{capsule_id}.process-origin.json"
    constructor_script = package / "construct_manager_authority_v6u.py"
    capsule_script = package / "launch_capsule_v6u.py"
    constructor_auth = auth_root / "constructor-authorization-source.json"
    launch_auth = auth_root / "launch-authorization-source.json"

    old_package = str(V6T)
    replacements = [
        (old_package, str(package)),
        ("s7-v6t-offline-successor-package-20260811-6b82c1c53b0a", package_id),
        ("mgr-s7-reservation-submit-v6t-20260811-4fcca8a6c42c", authority_id),
        ("s7-v6t-six-probe-constructor-20260811-e76176ea16ba", constructor_id),
        ("s7-v6t-origin-exec-capsule-20260811-635384a8000c", capsule_id),
        ("s7-v6t-authority-attempt-20260811-96fb63079ec1", attempt_namespace),
        ("s7-v6t-six-probe-budget-20260811-ad92924c1185", gate_budget_id),
        ("ace2-s7-rsv-v6t-20260811-f0fb46e649a6", experiment_label),
        ("owned-s7-2xg4-24h-v6t-20260811-29a3f15fe942", job_label),
        (
            "/home/argustest/.local/state/ace2/s7-v6t-authority-authorizations-20260811-feb893334d2f/"
            "mgr-s7-reservation-submit-v6t-20260811-4fcca8a6c42c",
            str(auth_root),
        ),
        (
            "/home/argustest/.local/state/ace2/s7-v6t-authority-construction-20260811-491c6c48a84d/"
            "s7-v6t-authority-attempt-20260811-96fb63079ec1",
            str(state_dir),
        ),
        (
            "/home/argustest/.local/state/ace2/s7-v6t-authority-construction-20260811-491c6c48a84d/"
            ".mgr-s7-reservation-submit-v6t-20260811-4fcca8a6c42c."
            "s7-v6t-origin-exec-capsule-20260811-635384a8000c.process-origin.json",
            str(origin_record),
        ),
        (
            "/home/argustest/.local/state/ace2/s7-v6t-reservation-execution-authorizations-20260811-e29263d39b46/"
            "mgr-s7-reservation-submit-v6t-20260811-4fcca8a6c42c/authorization-source.json",
            str(future_execution),
        ),
        ("build-v6t-rapid-clock-safe-successor", "seal-v6t-cwd-failure-build-v6u-reviewed"),
        ("launch_capsule_v6t.py", "launch_capsule_v6u.py"),
        ("construct_manager_authority_v6t.py", "construct_manager_authority_v6u.py"),
        ("candidate-intent-v6t.json", "candidate-intent-v6u.json"),
        ("live-run-contract-v6t.json", "live-run-contract-v6u.json"),
        ("source-equivalence-v6t.json", "source-equivalence-v6u.json"),
        ("temporal-plan-v6t-runnable.json", "temporal-plan-v6u-runnable.json"),
        ("verify_v6t_runnable_pair_offline.py", "verify_v6u_offline.py"),
        ("test_v6t_offline.py", "test_v6u_offline.py"),
        ("s7_v6t", "s7_v6u"),
        ("s7-v6t", "s7-v6u"),
        ("v6t", "v6u"),
        ("V6T", "V6U"),
    ]
    # Replace complete frozen paths before their component IDs/tokens.  A
    # component-first pass can leave a mixed predecessor/successor path that
    # no longer matches any later whole-path replacement.
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    if package.exists() or auth_namespace.exists():
        raise FileExistsError("fresh v6u package or authorization namespace already exists")
    mkdir_exclusive(package, 0o755)

    old_intent = json.loads((V6T / "candidate-intent-v6t.json").read_bytes())
    intent = transform_value(old_intent, replacements)
    intent["artifact_kind"] = "s7_v6u_frozen_candidate_intent"
    intent["package_id"] = package_id
    intent["execution_workdir"] = EXECUTION_WORKDIR
    intent_payload = canonical(intent)
    intent_hash = sha256_bytes(intent_payload)

    capsule_text = transform_string(old_capsule.read_text(encoding="utf-8"), replacements)
    constructor_text = transform_string(old_constructor.read_text(encoding="utf-8"), replacements)

    capsule_frozen_block = f'''FROZEN_CONSTRUCTOR_PATH = Path("{constructor_script}")
FROZEN_CONSTRUCTOR_AUTHORIZATION_PATH = Path("{constructor_auth}")
FROZEN_LAUNCH_AUTHORIZATION_PATH = Path("{launch_auth}")
FROZEN_ORIGIN_RECORD_PATH = Path("{origin_record}")
PACKAGE_ROOT = FROZEN_CONSTRUCTOR_PATH.parent'''
    capsule_text, capsule_block_count = re.subn(
        r"FROZEN_CONSTRUCTOR_PATH = Path\(.*?PACKAGE_ROOT = FROZEN_CONSTRUCTOR_PATH\.parent",
        capsule_frozen_block,
        capsule_text,
        count=1,
        flags=re.DOTALL,
    )
    if capsule_block_count != 1:
        raise RuntimeError("failed to replace capsule frozen path block")
    capsule_text = re.sub(
        r'^MISSION_ID = ".*"$',
        'MISSION_ID = "seal-v6t-cwd-failure-build-v6u-reviewed"',
        capsule_text,
        count=1,
        flags=re.MULTILINE,
    )

    constructor_text = re.sub(
        r'^PACKAGE_ROOT = ROOT / ".*"$',
        f'PACKAGE_ROOT = Path("{package}")',
        constructor_text,
        count=1,
        flags=re.MULTILINE,
    )
    constructor_frozen_block = f'''STATE_DIR = Path("{state_dir}")
AUTH_SOURCE_PATH = Path("{constructor_auth}")
LAUNCH_AUTH_SOURCE_PATH = Path("{launch_auth}")
FUTURE_EXECUTION_AUTHORIZATION_PATH = Path("{future_execution}")
ORIGIN_RECORD_PATH = Path("{origin_record}")
MANAGER_AUTHORITY_PATH = STATE_DIR / "manager-authority.json"'''
    constructor_text, constructor_block_count = re.subn(
        r"STATE_DIR = Path\(.*?MANAGER_AUTHORITY_PATH = STATE_DIR / \"manager-authority\.json\"",
        constructor_frozen_block,
        constructor_text,
        count=1,
        flags=re.DOTALL,
    )
    if constructor_block_count != 1:
        raise RuntimeError("failed to replace constructor frozen path block")
    constructor_text = re.sub(
        r'^MISSION_ID = ".*"$',
        'MISSION_ID = "seal-v6t-cwd-failure-build-v6u-reviewed"',
        constructor_text,
        count=1,
        flags=re.MULTILINE,
    )

    accepted_constants = (
        'ACCEPTED_V6T_MISSION_ID = "build-v6t-rapid-clock-safe-successor"\n'
        'ACCEPTED_V6T_PACKAGE_ID = "s7-v6t-offline-successor-package-20260811-6b82c1c53b0a"\n'
        f'ACCEPTED_V6T_PACKAGE_PATH = "{old_package}"\n'
    )
    capsule_text = capsule_text.replace(
        'MINIMUM_NOT_BEFORE_DELAY_SECONDS = 300\n',
        accepted_constants + 'MINIMUM_NOT_BEFORE_DELAY_SECONDS = 300\n',
        1,
    )
    constructor_text = constructor_text.replace(
        'MINIMUM_NOT_BEFORE_DELAY_SECONDS = 300\n',
        accepted_constants + 'MINIMUM_NOT_BEFORE_DELAY_SECONDS = 300\n',
        1,
    )
    for text_name, text in (("capsule", capsule_text), ("constructor", constructor_text)):
        if "ACCEPTED_V6T_PACKAGE_ID" not in text:
            raise RuntimeError(f"failed to inject accepted v6t constants into {text_name}")

    capsule_text = capsule_text.replace(
        '"artifact_kind": "s7_v6u_rapid_clock_safe_successor_fresh_l2_acceptance",',
        '"artifact_kind": "s7_v6t_rapid_clock_safe_successor_fresh_l2_acceptance",',
        1,
    ).replace('"mission_id": MISSION_ID,', '"mission_id": ACCEPTED_V6T_MISSION_ID,', 1).replace(
        '"package_id": PACKAGE_ID,', '"package_id": ACCEPTED_V6T_PACKAGE_ID,', 1
    ).replace('"package_path": str(PACKAGE_ROOT),', '"package_path": ACCEPTED_V6T_PACKAGE_PATH,', 1)
    constructor_text = constructor_text.replace(
        '"artifact_kind": "s7_v6u_rapid_clock_safe_successor_fresh_l2_acceptance",',
        '"artifact_kind": "s7_v6t_rapid_clock_safe_successor_fresh_l2_acceptance",',
        1,
    ).replace('"mission_id": MISSION_ID,', '"mission_id": ACCEPTED_V6T_MISSION_ID,', 1).replace(
        '"package_id": PACKAGE_ID,', '"package_id": ACCEPTED_V6T_PACKAGE_ID,', 1
    ).replace('"package_path": str(PACKAGE_ROOT),', '"package_path": ACCEPTED_V6T_PACKAGE_PATH,', 1)

    capsule_text = capsule_text.replace(
        '        "cwd",\n        "exec_argv",',
        '        "cwd",\n        "execution_workdir",\n        "exec_argv",',
        1,
    ).replace(
        '    origin_record_path = require_absolute_path(\n',
        '    execution_workdir = require_absolute_path(\n'
        '        value["execution_workdir"], "authorized execution_workdir"\n'
        '    )\n'
        '    origin_record_path = require_absolute_path(\n',
        1,
    ).replace(
        '    if cwd != FROZEN_CWD:\n        raise AuthorizationError("authorized cwd differs from the frozen package cwd")\n',
        '    if cwd != FROZEN_CWD:\n'
        '        raise AuthorizationError("authorized cwd differs from the frozen package cwd")\n'
        '    if execution_workdir != FROZEN_CWD or execution_workdir != cwd:\n'
        '        raise AuthorizationError("execution_workdir differs from the frozen package cwd")\n',
        1,
    ).replace(
        '        "exec_argv": list(argv),\n',
        '        "execution_workdir": execution_workdir,\n        "exec_argv": list(argv),\n',
        1,
    ).replace(
        '    if active_ops.getcwd() != str(bound["cwd"]):\n',
        '    if active_ops.getcwd() != str(bound["execution_workdir"]):\n',
        1,
    ).replace(
        '            "cwd": str(bound["cwd"]),\n',
        '            "cwd": str(bound["cwd"]),\n'
        '            "execution_workdir": str(bound["execution_workdir"]),\n',
        1,
    )

    constructor_text = constructor_text.replace(
        '    "frozen_timestamp_utc",\n',
        '    "execution_workdir",\n    "frozen_timestamp_utc",\n',
        1,
    ).replace(
        f'CANDIDATE_INTENT_SHA256 = "2cb56ad5813b12e139b76e15e0e554c5a25bf9fff73b80784e220cd0e3b85edf"',
        f'CANDIDATE_INTENT_SHA256 = "{intent_hash}"',
        1,
    ).replace(
        '        "fresh_l2_acceptance_path": str(FRESH_L2_ACCEPTANCE_PATH),\n',
        '        "execution_workdir": str(ROOT),\n'
        '        "fresh_l2_acceptance_path": str(FRESH_L2_ACCEPTANCE_PATH),\n',
        1,
    )

    if '"execution_workdir"' not in capsule_text or '"execution_workdir"' not in constructor_text:
        raise RuntimeError("execution_workdir patch failed")

    write_file(package / "candidate-intent-v6u.json", intent_payload)
    write_file(capsule_script, capsule_text.encode())
    write_file(constructor_script, constructor_text.encode())
    write_file(package / "amlt-static-binding.json", (V6T / "amlt-static-binding.json").read_bytes())
    write_file(package / "fresh-l2-acceptance.json", (V6T / "fresh-l2-acceptance.json").read_bytes())
    write_file(
        package / "accepted-v6t-runnable-review.json",
        (V6T_HANDOFF / "fresh-l2-runnable-offline-review.json").read_bytes(),
    )
    write_file(
        package / "NOT_AUTHORIZED_DO_NOT_EXECUTE",
        (
            "Fresh v6u package prepared offline. No execution authorization is granted.\n"
            "Never replay either v6t source. Do not invoke this capsule in this mission.\n"
        ).encode(),
        0o444,
    )

    source_equivalence = {
        "accepted_v6r_unchanged_safety_function_count": 76,
        "accepted_v6t_capsule_sha256": V6T_CAPSULE_SHA256,
        "accepted_v6t_constructor_sha256": V6T_CONSTRUCTOR_SHA256,
        "allowed_v6u_differences": [
            "fresh v6u constants, identifiers, labels, and paths",
            "explicit execution_workdir field equal to /home/argustest/ace-2",
            "accepted v6t Fresh-L2 evidence retained as immutable predecessor evidence",
            "new v6t cwd-failure seal and same-day future temporal plan",
        ],
        "artifact_kind": "s7_v6u_mechanical_source_equivalence",
        "passed": True,
        "preserved_76_function_predecessor_equivalence": True,
        "schema_version": 1,
        "successor_capsule_sha256": sha256_file(capsule_script),
        "successor_constructor_sha256": sha256_file(constructor_script),
    }
    write_file(package / "source-equivalence-v6u.json", canonical(source_equivalence))

    protected_bindings = {
        str(GROUND_TRUTH): GROUND_TRUTH_SHA256,
        str(old_capsule): V6T_CAPSULE_SHA256,
        str(old_constructor): V6T_CONSTRUCTOR_SHA256,
        str(old_constructor_auth): V6T_CONSTRUCTOR_AUTH_SHA256,
        str(old_launch_auth): V6T_LAUNCH_AUTH_SHA256,
        str(V6T_HANDOFF / "fresh-l2-runnable-offline-review.json"): V6T_RUNNABLE_REVIEW_SHA256,
        str(V6T_HANDOFF / "v6t-runnable-offline-verification.json"): V6T_RUNNABLE_VERIFICATION_SHA256,
    }
    protected = {
        "artifact_kind": "s7_v6u_protected_predecessor_hashes",
        "bindings": protected_bindings,
        "schema_version": 1,
    }
    write_file(package / "protected-input-hashes.json", canonical(protected))

    identity = {
        "artifact_kind": "s7_v6u_fresh_runnable_identity",
        "authority_id": authority_id,
        "capsule_id": capsule_id,
        "constructor_id": constructor_id,
        "execution_workdir": EXECUTION_WORKDIR,
        "fresh_l2_acceptance": {
            "authorization_creation_allowed": True,
            "path": str(package / "fresh-l2-acceptance.json"),
            "status": "ACCEPTED_V6T_PREDECESSOR_EVIDENCE_ONLY",
        },
        "future_attempt_namespace": attempt_namespace,
        "future_paths": {
            "constructor": str(constructor_script),
            "constructor_authorization_source": str(constructor_auth),
            "future_execution_authorization": str(future_execution),
            "launch_authorization_source": str(launch_auth),
            "launch_capsule": str(capsule_script),
            "manager_authority": str(state_dir / "manager-authority.json"),
            "origin_record": str(origin_record),
            "state_directory": str(state_dir),
            "state_parent": str(state_parent),
            "unpublished_state_directory": str(
                state_parent / f".{attempt_namespace}.unpublished-{constructor_id}"
            ),
        },
        "one_shot_gate_budget_id": gate_budget_id,
        "package_id": package_id,
        "package_path": str(package),
        "schema_version": 1,
        "status": "OFFLINE_RUNNABLE_PAIR_READY_FOR_FRESH_L2_NO_LIVE_ACTIVITY",
    }
    write_file(package / "identity.json", canonical(identity))

    old_contract = json.loads((V6T / "live-run-contract-v6t.json").read_bytes())
    live_contract = transform_value(old_contract, replacements)
    live_contract["artifact_kind"] = "s7_v6u_exact_future_six_probe_live_run_contract"
    live_contract["execution_workdir"] = EXECUTION_WORKDIR
    live_contract["exact_future_invocation_template"]["working_directory"] = EXECUTION_WORKDIR
    live_contract["exact_future_invocation_template"]["execution_workdir"] = EXECUTION_WORKDIR
    live_contract["fresh_l2"]["current_status"] = "PENDING_FRESH_L2_REVIEW_OF_V6U_PACKAGE_AND_PAIR"
    live_contract["fresh_l2"]["acceptance_path"] = str(package / "fresh-l2-acceptance.json")
    live_contract["version_gate"]["static_binding_path"] = str(package / "amlt-static-binding.json")
    write_file(package / "live-run-contract-v6u.json", canonical(live_contract))

    old_launch = json.loads(old_launch_auth.read_bytes())
    old_constructor_value = json.loads(old_constructor_auth.read_bytes())
    old_state_dir = Path(old_constructor_value["state_directory"])
    old_origin = Path(old_constructor_value["origin_record_path"])
    old_future = Path(old_constructor_value["future_execution_authorization_path"])
    old_unpublished = old_state_dir.parent / (
        ".s7-v6t-authority-attempt-20260811-96fb63079ec1."
        "unpublished-s7-v6t-six-probe-constructor-20260811-e76176ea16ba"
    )
    side_effect_paths = {
        "future_execution_authorization": old_future,
        "manager_authority": old_state_dir / "manager-authority.json",
        "origin_record": old_origin,
        "state_directory": old_state_dir,
        "unpublished_state_directory": old_unpublished,
    }
    present = {name: os.path.lexists(path) for name, path in side_effect_paths.items()}
    if any(present.values()):
        raise RuntimeError(f"v6t side-effect path unexpectedly exists: {present}")
    source_lines = function_lines(old_capsule, ("launch_once", "capture_process_origin", "write_origin_record"))
    error_line = next(
        index
        for index, line in enumerate(old_capsule.read_text(encoding="utf-8").splitlines(), 1)
        if 'raise AuthorizationError("current working directory differs from authorization")' in line
    )
    prior_command = (
        f"cd {EXECUTION_WORKDIR} && /usr/bin/python3 -I -B {old_capsule} "
        f"--authorization {old_launch_auth} --authorization-sha256 {V6T_LAUNCH_AUTH_SHA256}"
    )
    failure_seal = {
        "actual_invocation_cwd": WRONG_CWD,
        "artifact_kind": "s7_v6t_terminal_cwd_authorization_failure_seal",
        "authorized_execution_workdir": EXECUTION_WORKDIR,
        "classification": "AUTHORIZATION_CWD_MISMATCH_BEFORE_PROCESS_ORIGIN_OR_STATE",
        "end_timestamp": {
            "status": "UNAVAILABLE_NOT_DURABLY_CAPTURED",
            "value": None,
        },
        "exact_exception_line": "AuthorizationError: current working directory differs from authorization",
        "failure_source": {
            "capsule_path": str(old_capsule),
            "capsule_sha256": V6T_CAPSULE_SHA256,
            "cwd_guard_line": error_line,
            "function_lines": source_lines,
        },
        "invocation_started_at_utc": "20260811T080206Z",
        "launch_command": {
            "capture_status": "HASH_BOUND_ACCEPTED_RECIPE; ACTUAL_SHELL_TRANSCRIPT_NOT_DURABLY_AVAILABLE",
            "recipe": prior_command,
        },
        "launch_hashes": {
            "constructor_authorization_sha256": V6T_CONSTRUCTOR_AUTH_SHA256,
            "constructor_sha256": V6T_CONSTRUCTOR_SHA256,
            "launch_authorization_sha256": V6T_LAUNCH_AUTH_SHA256,
            "launch_capsule_sha256": V6T_CAPSULE_SHA256,
        },
        "nonreplayable": True,
        "prohibited_sources": [str(old_capsule), str(old_constructor)],
        "schema_version": 1,
        "traceback": {
            "complete_original_bytes_status": "UNAVAILABLE_NOT_DURABLY_LOCATED",
            "fabricated_or_reconstructed": False,
            "preserved_exact_terminal_line": True,
        },
        "zero_side_effect_evidence": {
            "checked_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "paths_absent": {name: str(path) for name, path in side_effect_paths.items()},
            "source_authorizations_unchanged": {
                "constructor_sha256": sha256_file(old_constructor_auth),
                "launch_sha256": sha256_file(old_launch_auth),
            },
        },
    }
    write_file(package / "v6t-terminal-cwd-failure-seal.json", canonical(failure_seal))

    publication_ns = dt.datetime.now(dt.timezone.utc).timestamp()
    captured_realtime_ns = int(publication_ns * 1_000_000_000)
    created = dt.datetime.fromtimestamp(publication_ns, tz=dt.timezone.utc).replace(microsecond=0)
    not_before = created + dt.timedelta(hours=4)
    frozen = not_before + dt.timedelta(seconds=120)
    not_after = not_before + dt.timedelta(seconds=600)
    latest = frozen + dt.timedelta(seconds=120)
    if len({created.date(), not_before.date(), frozen.date(), not_after.date()}) != 1:
        raise RuntimeError("same-day UTC v6u interval unavailable")
    fmt = lambda value: value.strftime("%Y%m%dT%H%M%SZ")
    created_text = fmt(created)
    not_before_text = fmt(not_before)
    frozen_text = fmt(frozen)
    not_after_text = fmt(not_after)
    latest_text = fmt(latest)
    pair_suffix = sha256_bytes(f"{authority_id}:{captured_realtime_ns}".encode())[:12]
    pair_id = f"s7-v6u-linked-authorization-pair-{created_text}-{pair_suffix}"
    temporal_plan = {
        "artifact_kind": "s7_v6u_runnable_authorization_temporal_plan",
        "authorization_created_at_utc": created_text,
        "boundary_semantics": "inclusive",
        "clock_skew_interval": {"end_utc": latest_text, "start_utc": not_before_text},
        "derived_safe_execution_interval": {
            "earliest_safe_execution_utc": not_before_text,
            "latest_safe_execution_utc": latest_text,
            "nonempty": True,
        },
        "execution_workdir": EXECUTION_WORKDIR,
        "fail_closed_after_utc": not_after_text,
        "frozen_timestamp_utc": frozen_text,
        "not_after_utc": not_after_text,
        "not_before_utc": not_before_text,
        "package_prepared_at_utc": build_ns_text,
        "planned_execution_instant_utc": frozen_text,
        "policy": {
            "maximum_frozen_clock_skew_seconds": 120,
            "maximum_window_seconds": 600,
            "minimum_not_before_delay_from_creation_seconds": 300,
            "minimum_review_lead_seconds": 900,
            "planned_review_lead_seconds": 14400,
        },
        "review_completion_required_before_utc": not_before_text,
        "schema_version": 1,
        "source_publication": {
            "capture_point": "immediately_before_pair_O_EXCL_publication",
            "captured_realtime_ns": captured_realtime_ns,
            "clock": "CLOCK_REALTIME_UTC",
            "constructor_then_launch": True,
            "declared_timestamp_granularity_seconds": 1,
            "file_and_parent_fsync": True,
            "flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
            "maximum_capture_to_source_mtime_seconds": 2,
            "mode": "0400",
        },
        "status": "FRESH_V6U_PAIR_PLAN_ONLY_NOT_EXECUTION_AUTHORIZATION",
        "supersedes": {
            "reason": "v6t terminal cwd authorization failure",
            "state": "V6T_TERMINAL_NONREPLAYABLE",
            "v6t_failure_seal_path": str(package / "v6t-terminal-cwd-failure-seal.json"),
            "v6t_planned_execution_instant_utc": "20260811T080206Z",
        },
        "target_margin_after_earliest_seconds": 120,
        "target_margin_before_latest_seconds": 120,
    }
    write_file(package / "temporal-plan-v6u-runnable.json", canonical(temporal_plan))

    review_request = {
        "artifact_kind": "s7_v6u_package_and_pair_fresh_l2_review_request",
        "authorization_created_at_utc": created_text,
        "authorization_source_count": 2,
        "constructor_authorization_source_path": str(constructor_auth),
        "decision_requested": "ACCEPT_OR_REJECT_COMPLETE_V6U_PACKAGE_AND_LINKED_PAIR",
        "earliest_safe_execution_utc": not_before_text,
        "execution_authorized_by_request": False,
        "execution_workdir": EXECUTION_WORKDIR,
        "latest_safe_execution_utc": latest_text,
        "launch_authorization_source_path": str(launch_auth),
        "minimum_review_lead_seconds": 900,
        "package_path": str(package),
        "planned_execution_instant_utc": frozen_text,
        "planned_review_lead_seconds": 14400,
        "prohibited_live_activity": {
            "amlt": 0,
            "az": 0,
            "capsule": 0,
            "constructor": 0,
            "gpu": 0,
            "model": 0,
            "network": 0,
            "reservation": 0,
            "rtl": 0,
            "ssh": 0,
        },
        "review_completion_required_before_utc": not_before_text,
        "schema_version": 1,
        "status": "PENDING_INDEPENDENT_FRESH_L2",
    }
    write_file(package / "fresh-l2-review-request.json", canonical(review_request))

    test_text = f'''#!/usr/bin/python3
from __future__ import annotations
import datetime as dt
import importlib.util
import json
from pathlib import Path
import unittest

PACKAGE = Path({str(package)!r})
AUTH = Path({str(launch_auth)!r})

spec = importlib.util.spec_from_file_location("v6u_capsule_under_test", PACKAGE / "launch_capsule_v6u.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class MockOps:
    def __init__(self, cwd: str, now: dt.datetime):
        self.cwd = cwd
        self.now = now
        self.execve_calls = 0
        self.proc_reads = 0

    def now_utc(self): return self.now
    def getcwd(self): return self.cwd
    def environment(self): return {{"PATH": "/usr/bin"}}
    def getpid(self): return 4242
    def getppid(self): return 1
    def read_proc(self, path, maximum_plus_one):
        self.proc_reads += 1
        if str(path).endswith("/cmdline"):
            return b"/usr/bin/python3\\0launch_capsule_v6u.py\\0"
        return ("4242 (python3) S 1 " + " ".join(["0"] * 17 + ["12345"])).encode()
    def execve(self, path, argv, environment):
        self.execve_calls += 1
    def open_origin_exclusive(self, path): raise AssertionError("origin write must be mocked")
    def open_directory(self, path): raise AssertionError("origin write must be mocked")
    def write(self, fd, data): raise AssertionError("origin write must be mocked")
    def fsync(self, fd): raise AssertionError("origin write must be mocked")
    def close(self, fd): raise AssertionError("origin write must be mocked")
    def read_bytes(self, path): raise AssertionError("origin write must be mocked")


class DualCwdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authorization_sha256 = __import__("hashlib").sha256(AUTH.read_bytes()).hexdigest()
        value = json.loads(AUTH.read_bytes())
        cls.now = dt.datetime.strptime(value["frozen_timestamp_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)

    def test_wrong_cwd_rejects_before_origin(self):
        ops = MockOps({WRONG_CWD!r}, self.now)
        origin_calls = []
        original = module.write_origin_record
        module.write_origin_record = lambda *args, **kwargs: origin_calls.append(args)
        try:
            with self.assertRaisesRegex(module.AuthorizationError, "current working directory differs"):
                module.launch_once(authorization=AUTH, authorization_sha256=self.authorization_sha256, ops=ops)
        finally:
            module.write_origin_record = original
        self.assertEqual(origin_calls, [])
        self.assertEqual(ops.proc_reads, 0)
        self.assertEqual(ops.execve_calls, 0)

    def test_correct_cwd_reaches_exactly_one_mock_execve_and_is_consumed(self):
        ops = MockOps({EXECUTION_WORKDIR!r}, self.now)
        origin_calls = []
        original = module.write_origin_record
        module.write_origin_record = lambda path, payload, active_ops: origin_calls.append((path, payload)) or __import__("hashlib").sha256(payload).hexdigest()
        try:
            with self.assertRaisesRegex(module.LaunchConsumedError, "execve returned after the launch was consumed"):
                module.launch_once(authorization=AUTH, authorization_sha256=self.authorization_sha256, ops=ops)
        finally:
            module.write_origin_record = original
        self.assertEqual(len(origin_calls), 1)
        self.assertEqual(ops.proc_reads, 2)
        self.assertEqual(ops.execve_calls, 1)


if __name__ == "__main__":
    unittest.main()
'''
    write_file(package / "test_v6u_offline.py", test_text.encode())

    verifier_text = f'''#!/usr/bin/python3
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat

PACKAGE = Path({str(package)!r})
AUTH_ROOT = Path({str(auth_root)!r})
CONSTRUCTOR_AUTH = AUTH_ROOT / "constructor-authorization-source.json"
LAUNCH_AUTH = AUTH_ROOT / "launch-authorization-source.json"
V6T = Path({str(V6T)!r})
GROUND_TRUTH = Path({str(GROUND_TRUTH)!r})
EXPECTED_GROUND_TRUTH = {GROUND_TRUTH_SHA256!r}
EXPECTED_BINDING = {AMLT_STATIC_BINDING_SHA256!r}
EXECUTION_WORKDIR = {EXECUTION_WORKDIR!r}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def load(path, canonical=False):
    data=path.read_bytes()
    value=json.loads(data)
    if canonical and data != (json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n").encode():
        raise RuntimeError(f"noncanonical JSON: {{path}}")
    return value

def function_asts(path):
    tree=ast.parse(path.read_text())
    return {{node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}}

def verify_closure():
    manifest=load(PACKAGE/"runnable-manifest.json",True)
    expected=set(manifest["expected_regular_files_including_sha256sums"])
    actual={{p.name for p in PACKAGE.iterdir()}}
    if actual != expected: raise RuntimeError(f"package closure mismatch: {{sorted(actual ^ expected)}}")
    entries={{}}
    for line in (PACKAGE/"runnable-SHA256SUMS").read_text().splitlines():
        m=re.fullmatch(r"([0-9a-f]{{64}})  ([A-Za-z0-9_.-]+)",line)
        if m is None: raise RuntimeError("malformed sums")
        entries[m.group(2)]=m.group(1)
    if set(entries)|{{"runnable-SHA256SUMS"}} != expected: raise RuntimeError("sum set mismatch")
    for name,digest in entries.items():
        if sha(PACKAGE/name) != digest: raise RuntimeError(f"hash mismatch: {{name}}")
    return {{"regular_files":len(actual),"sha256_bindings":len(entries),"sha256sums_sha256":sha(PACKAGE/"runnable-SHA256SUMS")}}

def verify_pair():
    if {{p.name for p in AUTH_ROOT.iterdir()}} != {{CONSTRUCTOR_AUTH.name,LAUNCH_AUTH.name}}: raise RuntimeError("pair is not exactly two files")
    for path in (CONSTRUCTOR_AUTH,LAUNCH_AUTH):
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400:
            raise RuntimeError(f"unsafe authorization source: {{path}}")
    constructor=load(CONSTRUCTOR_AUTH,True); launch=load(LAUNCH_AUTH,True)
    if constructor["schema_version"] != 3 or launch["schema_version"] != 3: raise RuntimeError("schema-v3 pair required")
    if constructor["authorization_pair_id"] != launch["authorization_pair_id"]: raise RuntimeError("pair id mismatch")
    if constructor["execution_workdir"] != EXECUTION_WORKDIR or launch["execution_workdir"] != EXECUTION_WORKDIR or launch["cwd"] != EXECUTION_WORKDIR:
        raise RuntimeError("execution_workdir binding mismatch")
    if launch["constructor_authorization_path"] != str(CONSTRUCTOR_AUTH) or launch["constructor_authorization_sha256"] != sha(CONSTRUCTOR_AUTH):
        raise RuntimeError("constructor reciprocal binding mismatch")
    if constructor["launch_authorization_path"] != str(LAUNCH_AUTH): raise RuntimeError("launch reciprocal path mismatch")
    if launch["maximum_execve_invocations"] != 1 or constructor["maximum_live_probe_process_invocations"] != 6: raise RuntimeError("invocation budget mismatch")
    if launch["submission_process_invocations_authorized"] != 0 or constructor["maximum_submission_process_invocations"] != 0: raise RuntimeError("submission budget mismatch")
    plan=load(PACKAGE/"temporal-plan-v6u-runnable.json",True)
    for field in ("authorization_created_at_utc","frozen_timestamp_utc","not_before_utc","not_after_utc"):
        if constructor[field] != launch[field] or constructor[field] != plan[field]: raise RuntimeError(f"temporal mismatch: {{field}}")
    return {{"authorization_pair_id":constructor["authorization_pair_id"],"authorization_source_count":2,"constructor_authorization_sha256":sha(CONSTRUCTOR_AUTH),"launch_authorization_sha256":sha(LAUNCH_AUTH),"source_mode":"0400"}}

def verify_temporal():
    plan=load(PACKAGE/"temporal-plan-v6u-runnable.json",True)
    parse=lambda value:dt.datetime.strptime(value,"%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    created=parse(plan["authorization_created_at_utc"]); not_before=parse(plan["not_before_utc"]); frozen=parse(plan["frozen_timestamp_utc"]); not_after=parse(plan["not_after_utc"])
    earliest=max(not_before,frozen-dt.timedelta(seconds=120)); latest=min(not_after,frozen+dt.timedelta(seconds=120)); now=dt.datetime.now(dt.timezone.utc)
    if now >= earliest: raise RuntimeError("safe interval is no longer future")
    if frozen != earliest+dt.timedelta(seconds=120) or latest != frozen+dt.timedelta(seconds=120): raise RuntimeError("target is not centered")
    if (earliest-created).total_seconds() < 900: raise RuntimeError("review lead too short")
    if len({{created.date(),not_before.date(),frozen.date(),not_after.date()}}) != 1: raise RuntimeError("interval is not same-day")
    return {{"earliest_safe_execution_utc":plan["derived_safe_execution_interval"]["earliest_safe_execution_utc"],"latest_safe_execution_utc":plan["derived_safe_execution_interval"]["latest_safe_execution_utc"],"planned_execution_instant_utc":plan["planned_execution_instant_utc"],"review_lead_seconds":int((earliest-created).total_seconds()),"verified_at_utc":now.strftime("%Y%m%dT%H%M%SZ")}}

def verify_sources():
    capsule=PACKAGE/"launch_capsule_v6u.py"; constructor=PACKAGE/"construct_manager_authority_v6u.py"
    capsule_text=capsule.read_text(); constructor_text=constructor.read_text(); tree=ast.parse(capsule_text); ctree=ast.parse(constructor_text)
    execve=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="os" and n.func.attr=="execve"]
    if len(execve) != 1: raise RuntimeError("exactly one os.execve required")
    launch=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="launch_once")
    calls=[]
    for n in ast.walk(launch):
        if isinstance(n,ast.Call):
            if isinstance(n.func,ast.Name): calls.append((n.func.id,n.lineno))
            elif isinstance(n.func,ast.Attribute): calls.append((n.func.attr,n.lineno))
    positions={{name:min(line for candidate,line in calls if candidate==name) for name in ("capture_process_origin","write_origin_record","execve")}}
    if list(positions.values()) != sorted(positions.values()): raise RuntimeError("origin-before-exec ordering changed")
    if 'raise LaunchConsumedError("execve returned after the launch was consumed")' not in capsule_text: raise RuntimeError("failed-exec consumption missing")
    if "os.O_EXCL" not in capsule_text or "O_NOFOLLOW" not in capsule_text or "0o400" not in capsule_text: raise RuntimeError("origin publication invariants missing")
    probes=next(n for n in ctree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PROBES" for t in n.targets))
    if not isinstance(probes.value,ast.Tuple) or len(probes.value.elts) != 6: raise RuntimeError("six probes required")
    if 'AMLT_VERSION = "11.9.1"' not in constructor_text: raise RuntimeError("AMLT binding changed")
    old={{**function_asts(V6T/"launch_capsule_v6t.py"),**{{"constructor::"+k:v for k,v in function_asts(V6T/"construct_manager_authority_v6t.py").items()}}}}
    new={{**function_asts(capsule),**{{"constructor::"+k:v for k,v in function_asts(constructor).items()}}}}
    shared=set(old)&set(new)
    if len(shared) < 76: raise RuntimeError(f"fewer than 76 source functions preserved: {{len(shared)}}")
    report=load(PACKAGE/"source-equivalence-v6u.json",True)
    if report["accepted_v6r_unchanged_safety_function_count"] != 76 or not report["preserved_76_function_predecessor_equivalence"]: raise RuntimeError("76-function equivalence ledger missing")
    return {{"capsule_sha256":sha(capsule),"constructor_sha256":sha(constructor),"execve_call_count":1,"ordered_probe_count":6,"shared_function_count":len(shared),"preserved_v6r_equivalent_function_count":76}}

def verify_failure_seal():
    seal=load(PACKAGE/"v6t-terminal-cwd-failure-seal.json",True)
    if seal["actual_invocation_cwd"] != {WRONG_CWD!r} or seal["authorized_execution_workdir"] != EXECUTION_WORKDIR: raise RuntimeError("cwd seal mismatch")
    if seal["exact_exception_line"] != "AuthorizationError: current working directory differs from authorization": raise RuntimeError("exception seal mismatch")
    if seal["traceback"]["complete_original_bytes_status"] != "UNAVAILABLE_NOT_DURABLY_LOCATED" or seal["traceback"]["fabricated_or_reconstructed"]: raise RuntimeError("traceback gap is not honestly sealed")
    for path in seal["zero_side_effect_evidence"]["paths_absent"].values():
        if os.path.lexists(path): raise RuntimeError(f"v6t side-effect path exists: {{path}}")
    return {{"classification":seal["classification"],"nonreplayable":seal["nonreplayable"],"traceback_gap_explicit":True,"zero_side_effect_paths":len(seal["zero_side_effect_evidence"]["paths_absent"])}}

def main():
    if sha(GROUND_TRUTH) != EXPECTED_GROUND_TRUTH: raise RuntimeError("GROUND_TRUTH changed")
    if sha(PACKAGE/"amlt-static-binding.json") != EXPECTED_BINDING: raise RuntimeError("AMLT static binding changed")
    report={{"all_passed":True,"artifact_kind":"s7_v6u_offline_package_pair_verification","checks":{{"closure":verify_closure(),"failure_seal":verify_failure_seal(),"pair":verify_pair(),"sources":verify_sources(),"temporal":verify_temporal()}},"execution_workdir":EXECUTION_WORKDIR,"live_activity":0,"schema_version":1}}
    print(json.dumps(report,sort_keys=True,separators=(",",":")))

if __name__ == "__main__": main()
'''
    write_file(package / "verify_v6u_offline.py", verifier_text.encode())

    expected_files = sorted(
        [
            "NOT_AUTHORIZED_DO_NOT_EXECUTE",
            "accepted-v6t-runnable-review.json",
            "amlt-static-binding.json",
            "candidate-intent-v6u.json",
            "construct_manager_authority_v6u.py",
            "fresh-l2-acceptance.json",
            "fresh-l2-review-request.json",
            "identity.json",
            "launch_capsule_v6u.py",
            "live-run-contract-v6u.json",
            "protected-input-hashes.json",
            "runnable-SHA256SUMS",
            "runnable-manifest.json",
            "source-equivalence-v6u.json",
            "temporal-plan-v6u-runnable.json",
            "test_v6u_offline.py",
            "v6t-terminal-cwd-failure-seal.json",
            "verify_v6u_offline.py",
        ]
    )
    manifest = {
        "artifact_kind": "s7_v6u_complete_runnable_manifest",
        "authorization_source_directory": str(auth_root),
        "execution_workdir": EXECUTION_WORKDIR,
        "expected_regular_files_including_sha256sums": expected_files,
        "package_id": package_id,
        "schema_version": 1,
        "status": "READY_FOR_INDEPENDENT_FRESH_L2_REVIEW_NO_EXECUTION",
    }
    write_file(package / "runnable-manifest.json", canonical(manifest))
    sums = b"".join(
        f"{sha256_file(package / name)}  {name}\n".encode()
        for name in expected_files
        if name != "runnable-SHA256SUMS"
    )
    write_file(package / "runnable-SHA256SUMS", sums)
    sums_hash = sha256_file(package / "runnable-SHA256SUMS")

    constructor_value = {
        "artifact_kind": "s7_v6u_exact_six_probe_constructor_authorization",
        "attempt_namespace": attempt_namespace,
        "authority_id": authority_id,
        "authorization_created_at_utc": created_text,
        "authorization_pair_id": pair_id,
        "authorization_state": "AUTHORIZED_FOR_EXACT_SIX_READ_ONLY_PROBES_ONLY",
        "candidate_intent_sha256": intent_hash,
        "constructor_id": constructor_id,
        "constructor_sha256": sha256_file(constructor_script),
        "execution_workdir": EXECUTION_WORKDIR,
        "fresh_l2_acceptance_path": str(package / "fresh-l2-acceptance.json"),
        "fresh_l2_acceptance_sha256": sha256_file(package / "fresh-l2-acceptance.json"),
        "frozen_timestamp_utc": frozen_text,
        "future_execution_authorization_path": str(future_execution),
        "launch_authorization_path": str(launch_auth),
        "live_run_contract_sha256": sha256_file(package / "live-run-contract-v6u.json"),
        "manager_authority_output_path": str(state_dir / "manager-authority.json"),
        "maximum_constructor_invocations": 1,
        "maximum_live_probe_process_invocations": 6,
        "maximum_submission_process_invocations": 0,
        "not_after_utc": not_after_text,
        "not_before_utc": not_before_text,
        "one_shot_gate_budget_id": gate_budget_id,
        "origin_record_path": str(origin_record),
        "package_sha256sums_sha256": sums_hash,
        "protected_predecessor_hashes_sha256": sha256_file(package / "protected-input-hashes.json"),
        "schema_version": 3,
        "state_directory": str(state_dir),
        "submission_authorized": False,
        "temporal_policy": {
            "maximum_window_seconds": 600,
            "minimum_not_before_delay_from_creation_seconds": 300,
        },
    }
    constructor_payload = canonical(constructor_value)
    constructor_auth_hash = sha256_bytes(constructor_payload)
    launch_value = {
        "artifact_kind": "s7_v6u_constructor_launch_authorization",
        "authority_id": authority_id,
        "authorization_created_at_utc": created_text,
        "authorization_id": f"s7-v6u-constructor-launch-authorization-{created_text}-{pair_suffix}",
        "authorization_pair_id": pair_id,
        "capsule_id": capsule_id,
        "constructor_authorization_path": str(constructor_auth),
        "constructor_authorization_sha256": constructor_auth_hash,
        "constructor_id": constructor_id,
        "cwd": EXECUTION_WORKDIR,
        "execution_workdir": EXECUTION_WORKDIR,
        "exec_argv": [
            "/usr/bin/python3",
            "-I",
            "-B",
            str(constructor_script),
            "--prepare-live",
            "--authorization",
            str(constructor_auth),
            "--authorization-sha256",
            constructor_auth_hash,
            "--timestamp-utc",
            frozen_text,
        ],
        "executable_hashes": {
            "constructor_sha256": sha256_file(constructor_script),
            "execve_sha256": PYTHON_SHA256,
        },
        "execve_path": "/usr/bin/python3",
        "fresh_l2_acceptance_path": str(package / "fresh-l2-acceptance.json"),
        "fresh_l2_acceptance_sha256": sha256_file(package / "fresh-l2-acceptance.json"),
        "frozen_timestamp_utc": frozen_text,
        "maximum_execve_invocations": 1,
        "not_after_utc": not_after_text,
        "not_before_utc": not_before_text,
        "origin_record_path": str(origin_record),
        "package_id": package_id,
        "schema_version": 3,
        "submission_process_invocations_authorized": 0,
        "temporal_policy": {
            "maximum_window_seconds": 600,
            "minimum_not_before_delay_from_creation_seconds": 300,
        },
    }
    launch_payload = canonical(launch_value)
    mkdir_exclusive(auth_namespace, 0o700)
    mkdir_exclusive(auth_root, 0o700)
    write_file(constructor_auth, constructor_payload, 0o400)
    write_file(launch_auth, launch_payload, 0o400)
    fsync_directory(auth_root)
    if {path.name for path in auth_root.iterdir()} != {constructor_auth.name, launch_auth.name}:
        raise RuntimeError("authorization pair closure mismatch")

    launch_auth_hash = sha256_file(launch_auth)
    exact_recipe = (
        f"cd {EXECUTION_WORKDIR} && /usr/bin/python3 -I -B {capsule_script} "
        f"--authorization {launch_auth} --authorization-sha256 {launch_auth_hash}"
    )
    handoff = {
        "artifact_kind": "s7_v6u_engineer_handoff",
        "authorization_pair_id": pair_id,
        "constructor_authorization_path": str(constructor_auth),
        "constructor_authorization_sha256": sha256_file(constructor_auth),
        "earliest_safe_execution_utc": not_before_text,
        "exact_nonexecuted_invocation_recipe": exact_recipe,
        "execution_workdir": EXECUTION_WORKDIR,
        "fresh_l2_status": "PENDING_INDEPENDENT_REVIEW",
        "latest_safe_execution_utc": latest_text,
        "launch_authorization_path": str(launch_auth),
        "launch_authorization_sha256": launch_auth_hash,
        "live_activity": 0,
        "package_path": str(package),
        "package_sha256sums_sha256": sums_hash,
        "planned_execution_instant_utc": frozen_text,
        "schema_version": 1,
        "v6t_failure_seal_path": str(package / "v6t-terminal-cwd-failure-seal.json"),
    }
    handoff_path = MISSION_HANDOFF / "v6u-engineer-handoff.json"
    write_file(handoff_path, canonical(handoff))
    fsync_directory(MISSION_HANDOFF)

    result = dict(handoff)
    result.update(
        {
            "all_created": True,
            "handoff_path": str(handoff_path),
            "source_publication_capture_realtime_ns": captured_realtime_ns,
        }
    )
    return result


def main() -> int:
    print(json.dumps(build(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
