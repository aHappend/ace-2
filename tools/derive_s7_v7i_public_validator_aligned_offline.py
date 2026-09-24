#!/usr/bin/python3
"""Build the inert v7i public-validator-aligned successor from sealed v7h."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
from typing import Any


ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / (
    "build/s7-v7h-authorization-schema-parity-successor-offline-"
    "20260811-15a4b0f6f6ff"
)
DESTINATION = ROOT / (
    "build/s7-v7i-public-validator-aligned-successor-offline-20260827"
)
EVIDENCE = ROOT / "evidence/s7-v7i-public-validator-aligned-20260827"
SOURCE_FAILURE = ROOT / "evidence/s7-v7h-public-validator-failure-20260811T142545Z.json"
PACKAGED_FAILURE_NAME = "v7h-public-validator-failure.json"

PACKAGE_ID = "s7-v7i-public-validator-aligned-successor-package-20260827"
AUTHORITY_ID = "mgr-s7-reservation-submit-v7i-20260827"
CONSTRUCTOR_ID = "s7-v7i-six-probe-constructor-20260827"
CAPSULE_ID = "s7-v7i-origin-exec-capsule-20260827"
ATTEMPT = "s7-v7i-authority-attempt-20260827"
GATE_BUDGET = "s7-v7i-six-probe-budget-20260827"
PAIR_SUFFIX = "s7v7i20260827"
AUTH_TOP = EVIDENCE / "authorization-sources"
AUTH_ROOT = AUTH_TOP / AUTHORITY_ID
CONSTRUCTOR_AUTH = AUTH_ROOT / "constructor-authorization-source.json"
LAUNCH_AUTH = AUTH_ROOT / "launch-authorization-source.json"
STATE_PARENT = Path(
    "/home/argustest/.local/state/ace2/s7-v7i-authority-construction-20260827"
)
STATE_DIR = STATE_PARENT / ATTEMPT
ORIGIN = STATE_PARENT / f".{AUTHORITY_ID}.{CAPSULE_ID}.process-origin.json"
FUTURE_EXEC_AUTH = (
    Path(
        "/home/argustest/.local/state/ace2/"
        "s7-v7i-reservation-execution-authorizations-20260827"
    )
    / AUTHORITY_ID
    / "authorization-source.json"
)
CREATED = dt.datetime(2026, 8, 27, 8, 22, 43, tzinfo=dt.timezone.utc)

EXPECTED_V7H_SUMS_SHA256 = (
    "bde4573a43ebc33297a3af1131cf49dbd1b3eeb60856a407623e1ae601efa9cf"
)
EXPECTED_V7H_FAILURE_SHA256 = (
    "3842a1cab03924ca88709e3d3f7e9daf5cffcc5b12dd04118bfc3a9315aa31dc"
)
EXPECTED_V7G_SUMS_SHA256 = (
    "98c05c4f7a8be5ec022d976db40ae8a152fbd3af5daff49b607682b2f832f276"
)
EXPECTED_V7G_ACCEPTANCE_SHA256 = (
    "9929f8db5af85a05008b6234326b21b9413170c0def875fe1de522dabea7f983"
)
PUBLICATION_REGRESSION_COUNT = 12
LIFECYCLE_REGRESSION_COUNT = 18

EXECUTABLE_LINEAGE_POLICY = {
    "all_observed_descendants_require_allowed_executable_identity": True,
    "identity_unavailable_descendant_policy":
        "fail_closed_unless_immutable_allowed_identity_proof",
    "unexpected_executable_decision_scope":
        "every_observed_process_across_complete_lifecycle",
}
PROCESS_GROUP_CLEANUP_POLICY = {
    "child_subreaper_required": True,
    "escape_filter_denied_syscalls": ["setns", "setpgid", "setsid", "unshare"],
    "exclusive_child_process_ownership_required": True,
    "kill_grace_seconds": 3,
    "start_new_session": True,
    "term_grace_seconds": 2,
}
CHANGED_SCOPE = [
    (
        "constructor public validator requires the accepted six-key fail-closed "
        "process_group_cleanup value"
    ),
    (
        "project-local canonical-source validation rejects legacy, missing, "
        "extra, and mutated cleanup values"
    ),
    (
        "v7g and v7h executable-lineage and lifecycle protections remain "
        "hash-bound"
    ),
    "fresh v7i identities, paths, package closure, and two schema-v3 sources",
]


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def fmt(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def function_hash(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return hashlib.sha256(
                ast.dump(node, include_attributes=False).encode("utf-8")
            ).hexdigest()
    raise RuntimeError(f"missing function: {name}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label} replacement count is {text.count(old)}, expected 1")
    return text.replace(old, new)


def verify_checksum_closure(package: Path) -> None:
    for line in (package / "runnable-SHA256SUMS").read_text(
        encoding="ascii"
    ).splitlines():
        digest, name = line.split("  ", 1)
        if sha(package / name) != digest:
            raise RuntimeError(f"checksum mismatch: {package / name}")


def assert_sealed_inputs() -> None:
    if sha(SOURCE / "runnable-SHA256SUMS") != EXPECTED_V7H_SUMS_SHA256:
        raise RuntimeError("sealed v7h package checksum changed")
    verify_checksum_closure(SOURCE)
    if sha(SOURCE_FAILURE) != EXPECTED_V7H_FAILURE_SHA256:
        raise RuntimeError("sealed v7h public-validator failure evidence changed")
    failure = load(SOURCE_FAILURE)
    if (
        failure.get("classification") != "PUBLIC_VALIDATOR_SOURCE_CONTRACT_MISMATCH"
        or failure.get("v7h_constructor_invocations") != 0
        or failure.get("v7h_capsule_invocations") != 0
        or failure.get("submission_process_invocations") != 0
    ):
        raise RuntimeError("v7h failure evidence contract changed")


def rewrite_tree(stage: Path) -> None:
    shutil.copytree(SOURCE, stage)
    paths = sorted(stage.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if "v7h" in path.name:
            path.rename(path.with_name(path.name.replace("v7h", "v7i")))

    replacements = [
        (
            str(SOURCE),
            str(DESTINATION),
        ),
        (
            SOURCE.name,
            DESTINATION.name,
        ),
        (
            "/home/argustest/.local/state/ace2/"
            "s7-v7h-authority-authorizations-20260811-815cfa247859/"
            "mgr-s7-reservation-submit-v7h-20260811-f9f25b9364d1",
            str(AUTH_ROOT),
        ),
        (
            "/home/argustest/.local/state/ace2/"
            "s7-v7h-authority-authorizations-20260811-815cfa247859",
            str(AUTH_TOP),
        ),
        (
            "s7-v7h-authorization-schema-parity-successor-package-"
            "20260811-15a4b0f6f6ff",
            PACKAGE_ID,
        ),
        (
            "mgr-s7-reservation-submit-v7h-20260811-f9f25b9364d1",
            AUTHORITY_ID,
        ),
        (
            "s7-v7h-six-probe-constructor-20260811-ee7877e7f66b",
            CONSTRUCTOR_ID,
        ),
        (
            "s7-v7h-origin-exec-capsule-20260811-14209d0a3e25",
            CAPSULE_ID,
        ),
        (
            "s7-v7h-authority-attempt-20260811-c09746c58180",
            ATTEMPT,
        ),
        (
            "s7-v7h-six-probe-budget-20260811-6d38b02967db",
            GATE_BUDGET,
        ),
        (
            "s7-v7h-authority-construction-20260811-9692cb00d847",
            "s7-v7i-authority-construction-20260827",
        ),
        (
            "s7-v7h-reservation-execution-authorizations-20260811-b5ea020b6cd6",
            "s7-v7i-reservation-execution-authorizations-20260827",
        ),
        (
            "ace2-s7-rsv-v7h-20260811-2480dffe1217",
            "ace2-s7-rsv-v7i-20260827",
        ),
        (
            "owned-s7-2xg4-24h-v7h-20260811-6d38b02967db",
            "owned-s7-2xg4-24h-v7i-20260827",
        ),
        (
            "s7-v7h-candidate-intent-20260811-b5ea020b6cd6",
            "s7-v7i-candidate-intent-20260827",
        ),
        ("2480dffe1217", PAIR_SUFFIX),
        ("v7h", "v7i"),
        ("V7H", "V7I"),
    ]
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for old, new in replacements:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def repair_constructor(stage: Path) -> None:
    constructor = stage / "construct_manager_authority_v7i.py"
    candidate = stage / "candidate-intent-v7i.json"
    text = constructor.read_text(encoding="utf-8")

    text, count = re.subn(
        r'CANDIDATE_INTENT_SHA256 = "[0-9a-f]{64}"',
        f'CANDIDATE_INTENT_SHA256 = "{sha(candidate)}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("candidate-intent hash replacement failed")

    legacy_cleanup = '''        "process_group_cleanup": {
            "kill_grace_seconds": PROCESS_GROUP_KILL_GRACE_SECONDS,
            "start_new_session": True,
            "term_grace_seconds": PROCESS_GROUP_TERM_GRACE_SECONDS,
        },'''
    aligned_cleanup = '''        "process_group_cleanup": {
            "child_subreaper_required": True,
            "escape_filter_denied_syscalls": ["setns", "setpgid", "setsid", "unshare"],
            "exclusive_child_process_ownership_required": True,
            "kill_grace_seconds": PROCESS_GROUP_KILL_GRACE_SECONDS,
            "start_new_session": True,
            "term_grace_seconds": PROCESS_GROUP_TERM_GRACE_SECONDS,
        },'''
    text = replace_once(
        text,
        legacy_cleanup,
        aligned_cleanup,
        "six-key public-validator cleanup",
    )

    old_scope = '''    if report.get("changed_scope") != [
        "inherited seccomp denies descendant session and process-group escape before descendant code runs",
        "verified child subreaper adopts surviving descendants that leave the root ancestry before observation",
        "a post-root snapshot is mandatory before natural quiescence can pass",
        "every observed descendant is rejected when executable identity is unexpected or unavailable regardless of root-exit timing",
        "interrupt and observer failures enter bounded TERM/KILL cleanup and require certain final absence",
        "fresh v7i identities, paths, package closure, and exactly two schema-v3 sources",
    ]:'''
    scope_lines = "\n".join(f"        {item!r}," for item in CHANGED_SCOPE)
    new_scope = f'''    if report.get("changed_scope") != [
{scope_lines}
    ]:'''
    text = replace_once(text, old_scope, new_scope, "source-equivalence scope")
    text = replace_once(
        text,
        '    if count != 7:\n'
        '        raise ContractError("protected predecessor ledger is incomplete")\n',
        '    if count != 8:\n'
        '        raise ContractError("protected predecessor ledger is incomplete")\n',
        "protected predecessor count",
    )
    constructor.write_text(text, encoding="utf-8")


def strengthen_verifier(stage: Path) -> None:
    verifier = stage / "verify_v7i_offline.py"
    text = verifier.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'V7G_PACKAGE = ROOT / "build/s7-v7g-complete-lineage-fail-closed-successor-offline-20260811-d6d478c6d320"',
        (
            'V7H_PACKAGE = ROOT / "build/s7-v7h-authorization-schema-parity-'
            'successor-offline-20260811-15a4b0f6f6ff"\n'
            'V7H_FAILURE = ROOT / "evidence/s7-v7h-public-validator-failure-'
            '20260811T142545Z.json"\n'
            'V7G_PACKAGE = ROOT / "build/s7-v7g-complete-lineage-fail-closed-'
            'successor-offline-20260811-d6d478c6d320"'
        ),
        "v7h predecessor constants",
    )
    text = replace_once(
        text,
        "EXPECTED_V7G = {",
        f"""PROCESS_GROUP_CLEANUP_POLICY = {PROCESS_GROUP_CLEANUP_POLICY!r}
EXPECTED_V7G = {{""",
        "cleanup policy constant",
    )
    text = replace_once(
        text,
        '    if sha(ROOT / "research/GROUND_TRUTH.md") != "70cd8aae4376ceee1454a08d8e31ab1254b8409dfc29e554f3f902dabc8863d0":\n'
        '        raise RuntimeError("GROUND_TRUTH changed")\n',
        f'    if sha(V7H_PACKAGE / "runnable-SHA256SUMS") != "{EXPECTED_V7H_SUMS_SHA256}":\n'
        '        raise RuntimeError("sealed v7h package changed")\n'
        '    verify_checksum_closure(V7H_PACKAGE)\n'
        f'    if sha(V7H_FAILURE) != "{EXPECTED_V7H_FAILURE_SHA256}":\n'
        '        raise RuntimeError("sealed v7h failure evidence changed")\n'
        f'    if (PACKAGE / "{PACKAGED_FAILURE_NAME}").read_bytes() != V7H_FAILURE.read_bytes():\n'
        '        raise RuntimeError("packaged v7h failure evidence differs")\n'
        f'    failure = load(PACKAGE / "{PACKAGED_FAILURE_NAME}")\n'
        '    if failure.get("classification") != "PUBLIC_VALIDATOR_SOURCE_CONTRACT_MISMATCH":\n'
        '        raise RuntimeError("v7h failure classification changed")\n',
        "v7h predecessor verification",
    )
    text = text.replace(
        'regressions.get("publication_validator_checks") != 8',
        f'regressions.get("publication_validator_checks") != {PUBLICATION_REGRESSION_COUNT}',
    )
    text = replace_once(
        text,
        '    request = load(PACKAGE / "fresh-l2-review-request.json")\n'
        '    if request.get("authorization_pair_id") != constructor.get("authorization_pair_id"):\n'
        '        raise RuntimeError("review request pair mismatch")\n'
        '    if request.get("authorization_source_count") != 2 or request.get("review_execution_forbidden") is not True:\n'
        '        raise RuntimeError("review request scope mismatch")\n',
        '    request = load(PACKAGE / "fresh-l2-review-request.json")\n'
        '    if request.get("authorization_pair_id") != constructor.get("authorization_pair_id"):\n'
        '        raise RuntimeError("review request pair mismatch")\n'
        '    if request.get("authorization_source_count") != 2 or request.get("review_execution_forbidden") is not True:\n'
        '        raise RuntimeError("review request scope mismatch")\n'
        '    durable_request = load(AUTH_TOP.parent / "fresh-l2-review-request.json")\n'
        '    if durable_request.get("authorization_pair_id") != constructor.get("authorization_pair_id"):\n'
        '        raise RuntimeError("durable review request pair mismatch")\n'
        '    if durable_request.get("authorization_source_hashes") != {\n'
        '        "constructor": sha(files[0]), "launch": sha(files[1])\n'
        '    }:\n'
        '        raise RuntimeError("durable review request source binding mismatch")\n'
        '    if durable_request.get("package_sha256sums_sha256") != sha(PACKAGE / "runnable-SHA256SUMS"):\n'
        '        raise RuntimeError("durable review request package binding mismatch")\n',
        "durable review request verification",
    )
    text = replace_once(
        text,
        "    constructor_missing = copy.deepcopy(constructor)\n",
        '    if constructor.get("process_group_cleanup") != PROCESS_GROUP_CLEANUP_POLICY:\n'
        '        raise RuntimeError("constructor cleanup policy is not the six-key contract")\n'
        '    legacy_cleanup = copy.deepcopy(constructor)\n'
        '    legacy_cleanup["process_group_cleanup"] = {\n'
        '        "kill_grace_seconds": 3,\n'
        '        "start_new_session": True,\n'
        '        "term_grace_seconds": 2,\n'
        '    }\n'
        '    cleanup_missing = copy.deepcopy(constructor)\n'
        '    cleanup_missing["process_group_cleanup"].pop("child_subreaper_required")\n'
        '    cleanup_extra = copy.deepcopy(constructor)\n'
        '    cleanup_extra["process_group_cleanup"]["unexpected_field"] = False\n'
        '    cleanup_mutated = copy.deepcopy(constructor)\n'
        '    cleanup_mutated["process_group_cleanup"]["escape_filter_denied_syscalls"] = ["setsid"]\n'
        "    constructor_missing = copy.deepcopy(constructor)\n",
        "cleanup negative fixtures",
    )
    text = replace_once(
        text,
        '    expect_constructor_rejection(constructor_missing, "missing lineage policy")\n',
        '    expect_constructor_rejection(legacy_cleanup, "legacy three-key cleanup")\n'
        '    expect_constructor_rejection(cleanup_missing, "missing cleanup key")\n'
        '    expect_constructor_rejection(cleanup_extra, "extra cleanup key")\n'
        '    expect_constructor_rejection(cleanup_mutated, "mutated cleanup value")\n'
        '    expect_constructor_rejection(constructor_missing, "missing lineage policy")\n',
        "cleanup rejection calls",
    )
    text = replace_once(
        text,
        "    return 8\n",
        f"    return {PUBLICATION_REGRESSION_COUNT}\n",
        "publication check count",
    )
    verifier.write_text(text, encoding="utf-8")


def update_package_metadata(stage: Path) -> tuple[dt.datetime, str]:
    stage.joinpath(PACKAGED_FAILURE_NAME).write_bytes(SOURCE_FAILURE.read_bytes())

    ledger_path = stage / "protected-input-hashes.json"
    ledger = load(ledger_path)
    ledger["artifact_kind"] = "s7_v7i_protected_predecessor_hashes"
    ledger.setdefault("bindings", {}).pop(
        str(ROOT / "research/GROUND_TRUTH.md"), None
    )
    ledger.setdefault("bindings", {}).update(
        {
            str(SOURCE / "runnable-SHA256SUMS"): EXPECTED_V7H_SUMS_SHA256,
            str(SOURCE_FAILURE): EXPECTED_V7H_FAILURE_SHA256,
        }
    )
    ledger["v7h_terminal_public_validator_failure_sha256"] = (
        EXPECTED_V7H_FAILURE_SHA256
    )
    ledger["terminal_failure_seal_sha256"] = sha(
        stage / "v7g-terminal-pre-origin-failure-seal.json"
    )
    write_json(ledger_path, ledger)

    not_before = CREATED + dt.timedelta(minutes=30)
    frozen = not_before + dt.timedelta(minutes=2)
    not_after = not_before + dt.timedelta(minutes=10)
    pair_id = f"s7-v7i-linked-authorization-pair-{fmt(CREATED)}-{PAIR_SUFFIX}"

    identity_path = stage / "identity.json"
    identity = load(identity_path)
    identity["artifact_kind"] = "s7_v7i_fresh_runnable_identity"
    identity["package_id"] = PACKAGE_ID
    identity["package_path"] = str(DESTINATION)
    identity["status"] = (
        "OFFLINE_PUBLIC_VALIDATOR_ALIGNED_READY_FOR_FRESH_L2_NO_LIVE_ACTIVITY"
    )
    identity["failed_v7h_predecessor"] = {
        "failure_evidence_path": str(SOURCE_FAILURE),
        "failure_evidence_sha256": EXPECTED_V7H_FAILURE_SHA256,
        "package_sha256sums_sha256": EXPECTED_V7H_SUMS_SHA256,
        "status": "SEALED_PUBLIC_VALIDATOR_SOURCE_CONTRACT_MISMATCH",
    }
    write_json(identity_path, identity)

    plan_path = stage / "temporal-plan-v7i-runnable.json"
    plan = load(plan_path)
    plan.update(
        {
            "artifact_kind": "s7_v7i_runnable_authorization_temporal_plan",
            "authorization_created_at_utc": fmt(CREATED),
            "derived_safe_execution_interval": {
                "earliest_safe_execution_utc": fmt(not_before),
                "latest_safe_execution_utc": fmt(frozen + dt.timedelta(seconds=120)),
                "nonempty": True,
            },
            "frozen_timestamp_utc": fmt(frozen),
            "not_after_utc": fmt(not_after),
            "not_before_utc": fmt(not_before),
            "status": "FRESH_WINDOW_PAIR_CREATED_NOT_AUTHORIZED_TO_EXECUTE",
        }
    )
    write_json(plan_path, plan)

    request_path = stage / "fresh-l2-review-request.json"
    request = load(request_path)
    request.update(
        {
            "artifact_kind":
                "s7_v7i_package_and_pair_fresh_l2_review_request",
            "authorization_created_at_utc": fmt(CREATED),
            "authorization_pair_id": pair_id,
            "authorization_source_count": 2,
            "decision_requested":
                "ACCEPT_OR_REJECT_V7I_PUBLIC_VALIDATOR_ALIGNED_PACKAGE",
            "durable_review_request_path":
                str(EVIDENCE / "fresh-l2-review-request.json"),
            "earliest_safe_execution_utc": fmt(not_before),
            "offline_regression_count":
                LIFECYCLE_REGRESSION_COUNT + PUBLICATION_REGRESSION_COUNT,
            "package_path": str(DESTINATION),
            "planned_execution_instant_utc": fmt(frozen),
            "publication_validator_regression_count":
                PUBLICATION_REGRESSION_COUNT,
            "review_completion_required_before_utc": fmt(not_before),
            "review_execution_forbidden": True,
            "submission_process_invocations": 0,
            "v7h_failure_evidence_sha256": EXPECTED_V7H_FAILURE_SHA256,
            "v7h_package_sha256sums_sha256": EXPECTED_V7H_SUMS_SHA256,
        }
    )
    write_json(request_path, request)

    finalizer = stage / "finalize_v7i_offline.py"
    finalizer_text = finalizer.read_text(encoding="utf-8").replace(
        "PUBLICATION_REGRESSION_COUNT = 8",
        f"PUBLICATION_REGRESSION_COUNT = {PUBLICATION_REGRESSION_COUNT}",
    )
    finalizer.write_text(finalizer_text, encoding="utf-8")

    evaluator_names = {
        "amlt_version": "_evaluate_version",
        "project_binding": "_evaluate_project",
        "principal_account": "_evaluate_account",
        "principal_management_token": "_evaluate_token",
        "frozen_label_absence": "_evaluate_labels",
    }
    constructor = stage / "construct_manager_authority_v7i.py"
    v7d_constructor = ROOT / (
        "build/s7-v7d-cwd-bound-successor-offline-20260811-e2cbdd7db801/"
        "construct_manager_authority_v7d.py"
    )
    evaluator_hashes = {
        gate: function_hash(constructor, name)
        for gate, name in evaluator_names.items()
    }
    if evaluator_hashes != {
        gate: function_hash(v7d_constructor, name)
        for gate, name in evaluator_names.items()
    }:
        raise RuntimeError("trusted gate evaluator body changed")
    v7g_package = ROOT / (
        "build/s7-v7g-complete-lineage-fail-closed-successor-offline-"
        "20260811-d6d478c6d320"
    )
    if function_hash(
        stage / "process_lifecycle_v7i.py", "run_bounded_process"
    ) != function_hash(v7g_package / "process_lifecycle_v7g.py", "run_bounded_process"):
        raise RuntimeError("v7g lifecycle implementation changed")

    equivalence = {
        "accepted_v7g": {
            "fresh_l2_review_sha256": EXPECTED_V7G_ACCEPTANCE_SHA256,
            "package_sha256sums_sha256": EXPECTED_V7G_SUMS_SHA256,
        },
        "artifact_kind": "s7_v7i_source_equivalence_and_validator_delta",
        "changed_scope": CHANGED_SCOPE,
        "passed": True,
        "preserved_scope": [
            "v7g run_bounded_process implementation and all 18 lifecycle tests",
            "v7h executable_lineage_policy field sets and exact-value validators",
            "trusted AMLT gate evaluators, order, timeouts, and no-submission policy",
            "sealed v7g and v7h evidence",
        ],
        "regressions": {
            "fixture_count": LIFECYCLE_REGRESSION_COUNT,
            "immediate_escape_iterations": 10,
            "interrupt_cleanup": True,
            "observer_failure_cleanup": True,
            "publication_validator_checks": PUBLICATION_REGRESSION_COUNT,
            "six_key_process_group_cleanup_positive": True,
            "three_key_process_group_cleanup_rejected": True,
            "transient_pre_root_identity_unavailable": True,
            "transient_pre_root_unexpected_executable": True,
        },
        "schema_version": 3,
        "successor": {
            "capsule_sha256": sha(stage / "launch_capsule_v7i.py"),
            "constructor_sha256": sha(constructor),
            "process_lifecycle_sha256": sha(stage / "process_lifecycle_v7i.py"),
        },
        "target_evaluator_policy_sha256": sha(
            stage / "target_output_policy_v7i.py"
        ),
        "unchanged_gate_evaluator_body_sha256": evaluator_hashes,
        "v7h_terminal_failure": {
            "failure_evidence_sha256": EXPECTED_V7H_FAILURE_SHA256,
            "package_sha256sums_sha256": EXPECTED_V7H_SUMS_SHA256,
        },
    }
    write_json(stage / "source-equivalence-v7i.json", equivalence)
    return frozen, pair_id


def finalize_package(stage: Path) -> str:
    manifest_path = stage / "runnable-manifest.json"
    manifest = load(manifest_path)
    expected_files = sorted(path.name for path in stage.iterdir() if path.is_file())
    manifest.update(
        {
            "artifact_kind": "s7_v7i_complete_runnable_manifest",
            "authorization_source_directory": str(AUTH_ROOT),
            "execution_workdir": str(ROOT),
            "expected_regular_files_including_sha256sums": expected_files,
            "package_id": PACKAGE_ID,
            "schema_version": 3,
            "status": "READY_FOR_INDEPENDENT_FRESH_L2_REVIEW_NO_EXECUTION",
        }
    )
    write_json(manifest_path, manifest)
    sums = "".join(
        f"{sha(stage / name)}  {name}\n"
        for name in expected_files
        if name != "runnable-SHA256SUMS"
    )
    (stage / "runnable-SHA256SUMS").write_text(sums, encoding="ascii")
    return sha(stage / "runnable-SHA256SUMS")


def build_sources(
    evidence_stage: Path,
    package_stage: Path,
    frozen: dt.datetime,
    pair_id: str,
    sums_sha: str,
) -> dict[str, str]:
    not_before = CREATED + dt.timedelta(minutes=30)
    not_after = not_before + dt.timedelta(minutes=10)
    constructor = package_stage / "construct_manager_authority_v7i.py"
    acceptance = package_stage / "accepted-v7g-fresh-l2-acceptance.json"
    constructor_sha = sha(constructor)
    constructor_auth = {
        "artifact_kind": "s7_v7i_exact_six_probe_constructor_authorization",
        "attempt_namespace": ATTEMPT,
        "authority_id": AUTHORITY_ID,
        "authorization_created_at_utc": fmt(CREATED),
        "authorization_pair_id": pair_id,
        "authorization_state": "AUTHORIZED_FOR_EXACT_SIX_READ_ONLY_PROBES_ONLY",
        "candidate_intent_sha256": sha(
            package_stage / "candidate-intent-v7i.json"
        ),
        "constructor_id": CONSTRUCTOR_ID,
        "constructor_sha256": constructor_sha,
        "executable_lineage_policy": EXECUTABLE_LINEAGE_POLICY,
        "execution_workdir": str(ROOT),
        "fresh_l2_acceptance_path": str(
            DESTINATION / "accepted-v7g-fresh-l2-acceptance.json"
        ),
        "fresh_l2_acceptance_sha256": sha(acceptance),
        "frozen_timestamp_utc": fmt(frozen),
        "future_execution_authorization_path": str(FUTURE_EXEC_AUTH),
        "gate_timeouts_seconds": {
            "amlt_version": 30,
            "frozen_label_absence": 120,
            "principal_account": 30,
            "principal_management_token": 30,
            "project_binding": 30,
            "target_binding": 30,
        },
        "launch_authorization_path": str(LAUNCH_AUTH),
        "live_run_contract_sha256": sha(
            package_stage / "live-run-contract-v7i.json"
        ),
        "manager_authority_output_path": str(STATE_DIR / "manager-authority.json"),
        "maximum_constructor_invocations": 1,
        "maximum_live_probe_process_invocations": 6,
        "maximum_submission_process_invocations": 0,
        "not_after_utc": fmt(not_after),
        "not_before_utc": fmt(not_before),
        "one_shot_gate_budget_id": GATE_BUDGET,
        "origin_record_path": str(ORIGIN),
        "package_sha256sums_sha256": sums_sha,
        "process_group_cleanup": PROCESS_GROUP_CLEANUP_POLICY,
        "protected_predecessor_hashes_sha256": sha(
            package_stage / "protected-input-hashes.json"
        ),
        "schema_version": 3,
        "state_directory": str(STATE_DIR),
        "submission_authorized": False,
        "temporal_policy": {
            "fixed_reserve_seconds": 30,
            "maximum_window_seconds": 600,
            "minimum_not_before_delay_from_creation_seconds": 300,
        },
        "total_temporal_budget_seconds": 305,
    }
    constructor_bytes = canonical(constructor_auth)
    constructor_auth_sha = hashlib.sha256(constructor_bytes).hexdigest()
    launch_auth = {
        "artifact_kind": "s7_v7i_constructor_launch_authorization",
        "authority_id": AUTHORITY_ID,
        "authorization_created_at_utc": fmt(CREATED),
        "authorization_id":
            f"s7-v7i-constructor-launch-authorization-{fmt(CREATED)}-{PAIR_SUFFIX}",
        "authorization_pair_id": pair_id,
        "capsule_id": CAPSULE_ID,
        "constructor_authorization_path": str(CONSTRUCTOR_AUTH),
        "constructor_authorization_sha256": constructor_auth_sha,
        "constructor_id": CONSTRUCTOR_ID,
        "cwd": str(ROOT),
        "exec_argv": [
            "/usr/bin/python3",
            "-I",
            "-B",
            str(DESTINATION / "construct_manager_authority_v7i.py"),
            "--prepare-live",
            "--authorization",
            str(CONSTRUCTOR_AUTH),
            "--authorization-sha256",
            constructor_auth_sha,
            "--timestamp-utc",
            fmt(frozen),
        ],
        "executable_hashes": {
            "constructor_sha256": constructor_sha,
            "execve_sha256":
                "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86",
        },
        "executable_lineage_policy": EXECUTABLE_LINEAGE_POLICY,
        "execution_workdir": str(ROOT),
        "execve_path": "/usr/bin/python3",
        "fresh_l2_acceptance_path": str(
            DESTINATION / "accepted-v7g-fresh-l2-acceptance.json"
        ),
        "fresh_l2_acceptance_sha256": sha(acceptance),
        "frozen_timestamp_utc": fmt(frozen),
        "maximum_execve_invocations": 1,
        "not_after_utc": fmt(not_after),
        "not_before_utc": fmt(not_before),
        "origin_record_path": str(ORIGIN),
        "package_id": PACKAGE_ID,
        "schema_version": 3,
        "submission_process_invocations_authorized": 0,
        "temporal_policy": {
            "maximum_window_seconds": 600,
            "minimum_not_before_delay_from_creation_seconds": 300,
        },
    }

    source_stage = evidence_stage / "authorization-sources" / AUTHORITY_ID
    source_stage.mkdir(parents=True, mode=0o700)
    source_payloads = {
        "constructor-authorization-source.json": constructor_bytes,
        "launch-authorization-source.json": canonical(launch_auth),
    }
    hashes: dict[str, str] = {}
    for name, payload in source_payloads.items():
        path = source_stage / name
        path.write_bytes(payload)
        path.chmod(0o400)
        hashes["constructor" if name.startswith("constructor") else "launch"] = sha(path)
    return hashes


def write_evidence(
    evidence_stage: Path,
    pair_id: str,
    sums_sha: str,
    source_hashes: dict[str, str],
) -> None:
    review_request = {
        "artifact_kind": "s7_v7i_durable_fresh_l2_review_request",
        "authorization_pair_id": pair_id,
        "authorization_source_count": 2,
        "authorization_source_hashes": source_hashes,
        "decision_requested":
            "ACCEPT_OR_REJECT_V7I_PUBLIC_VALIDATOR_ALIGNED_PACKAGE",
        "package_path": str(DESTINATION),
        "package_sha256sums_sha256": sums_sha,
        "review_execution_forbidden": True,
        "schema_version": 3,
        "submission_process_invocations": 0,
        "v7h_failure_evidence_sha256": EXPECTED_V7H_FAILURE_SHA256,
        "v7h_package_sha256sums_sha256": EXPECTED_V7H_SUMS_SHA256,
    }
    write_json(evidence_stage / "fresh-l2-review-request.json", review_request)
    build_receipt = {
        "artifact_kind": "s7_v7i_offline_package_construction_receipt",
        "authorization_pair_id": pair_id,
        "authorization_source_count": 2,
        "authorization_source_hashes": source_hashes,
        "constructor_invocations": 0,
        "created_at_utc": fmt(CREATED),
        "live_s7_process_invocations": 0,
        "package_path": str(DESTINATION),
        "package_sha256sums_sha256": sums_sha,
        "schema_version": 3,
        "ssh_invocations": 0,
        "submission_process_invocations": 0,
        "v7h_failure_evidence_sha256": EXPECTED_V7H_FAILURE_SHA256,
        "v7h_package_sha256sums_sha256": EXPECTED_V7H_SUMS_SHA256,
    }
    write_json(evidence_stage / "package-build.json", build_receipt)


def main() -> None:
    assert_sealed_inputs()
    package_stage = DESTINATION.with_name(f".{DESTINATION.name}.stage")
    evidence_stage = EVIDENCE.with_name(f".{EVIDENCE.name}.stage")
    forbidden = (
        DESTINATION,
        EVIDENCE,
        package_stage,
        evidence_stage,
        STATE_PARENT,
        FUTURE_EXEC_AUTH.parents[1],
    )
    for path in forbidden:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh v7i path already exists: {path}")

    rewrite_tree(package_stage)
    repair_constructor(package_stage)
    strengthen_verifier(package_stage)
    frozen, pair_id = update_package_metadata(package_stage)
    sums_sha = finalize_package(package_stage)
    source_hashes = build_sources(
        evidence_stage,
        package_stage,
        frozen,
        pair_id,
        sums_sha,
    )
    write_evidence(evidence_stage, pair_id, sums_sha, source_hashes)
    package_stage.rename(DESTINATION)
    evidence_stage.rename(EVIDENCE)
    print(
        json.dumps(
            {
                "authorization_pair_id": pair_id,
                "authorization_source_count": 2,
                "package_path": str(DESTINATION),
                "package_sha256sums_sha256": sums_sha,
                "submission_process_invocations": 0,
                "v7i_capsule_invocations": 0,
                "v7i_constructor_invocations": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
