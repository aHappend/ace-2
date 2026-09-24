#!/usr/bin/env python3
"""Create the corrected layer-16 authority package sources exclusively."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path("/home/argustest/ace-2")
SOURCE = ROOT / "reports/ace2-layer16-runtime-pass-authority-0001"
TARGET = ROOT / "reports/ace2-layer16-runtime-pass-authority-0002"


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"source section markers changed: {start!r}, {end!r}")
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def write_exclusive(path: Path, text: str) -> None:
    raw = text.encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def corrected_executor() -> str:
    text = (SOURCE / "execute_layer16.py").read_text(encoding="utf-8")
    replacements = {
        "ace2-layer16-runtime-pass-authority-0001": (
            "ace2-layer16-runtime-pass-authority-0002"
        ),
        "ace2-layer16-runtime-pass-authority-review-0001": (
            "ace2-layer16-runtime-pass-authority-review-0002"
        ),
        "ace2-layer16-runtime-pass-manager-grant-0001": (
            "ace2-layer16-runtime-pass-manager-grant-0002"
        ),
        "ace2-layer16-continuation-0010-consumption-state-0001": (
            "ace2-layer16-continuation-0010-consumption-state-0002"
        ),
        "runtime-pass-authority-0001": "runtime-pass-authority-0002",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    constants = '''LAYER15_ROOT = ROOT / (
    "reports/ace2-layer15-continuation-0009-consumption-state-0006"
)
LIVE_RUNTIME = LAYER15_ROOT / "runtime"
CHECKPOINT = LIVE_RUNTIME / "checkpoint-manifest.json"
CHECKPOINT_SHA256 = "ed01b51090f835d00dc2ddc19ee43c081b34ed2a06eee8f1fb0450fc7fdf0e5e"
CHECKPOINT_STATUS = (
    "PUBLISHED_AUTHENTICATED_THROUGH_LAYER15_AWAITING_LAYER16_AUTHORITY"
)
LAYER15_OUTPUT_SHA256 = (
    "761a964eb145933c64935bb108f97daa9db89203a31a1b00ba168371887ecc02"
)
LAYER15_OUTPUT_IDENTITY = (
    "ace2:stage1:r5:position-00:layer-15:runtime-pass-authority-0006"
)

MANAGER_DIRECTIVE = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "active_manager_directive.json"
)
ACTIVE_DIRECTIVE_REVISION = "d2baec27ffa54f43bc1f46a4bf762dc9"
ACTIVE_DIRECTIVE_SHA256 = (
    "c25b614ce996e05171c96a48303b8b3f9cb0bfe4626ea77f31285c783e8472cf"
)
ACTIVE_DIRECTIVE_OBJECTIVE_SHA256 = (
    "43fd081a2ec0196f5100f18980c9c58017888abf17b7cfcc77444e31ebd8ad39"
)

LIVE_PREDICATE_POLICY = {
    "allowed": [
        "MANAGER_CORRECTED_DIRECTIVE",
        "CORRECTED_CHECKPOINT",
        "POSITION00_LAYER16",
    ],
    "excluded": [
        "HISTORICAL_ATTEMPT_STATE",
        "PRE_CORRECTION_ADJUDICATION",
    ],
}

'''
    text = replace_section(
        text,
        "LAYER15_PACKAGE = ROOT /",
        "SOURCE_PACKAGE = ROOT /",
        constants,
    )

    frontier = '''def _frontier_binding() -> dict[str, Any]:
    return {
        "corrected_checkpoint": {
            "path": CHECKPOINT.relative_to(ROOT).as_posix(),
            "sha256": CHECKPOINT_SHA256,
            "status": CHECKPOINT_STATUS,
        },
        "unit_key": UNIT,
    }


'''
    text = replace_section(
        text,
        "def _frontier_binding() -> dict[str, Any]:",
        "def _exact_invocation() -> dict[str, Any]:",
        frontier,
    )

    authority = '''def authority_record() -> dict[str, Any]:
    return {
        "acceptance_artifact_targets": [
            REVIEW.relative_to(ROOT).as_posix(),
            GRANT.relative_to(ROOT).as_posix(),
        ],
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "authorization": {
            "authority_cardinality_before_manager_grant": 0,
            "creator_role": "engineer",
            "fresh_review_effect": "PACKAGE_ACCEPTANCE_ONLY",
            "requested_execution_cardinality": 1,
            "state": "SEALED_CANDIDATE_AWAITING_SOURCE_DISJOINT_REVIEW_AND_MANAGER_GRANT",
        },
        "bindings": {
            "acceptance_contract": {
                "path": ACCEPTANCE.relative_to(ROOT).as_posix(),
                "sealing": "MEMBER_OF_PACKAGE_SHA256SUMS_AND_TREE_ROOT",
            },
            "corrected_checkpoint": _frontier_binding()["corrected_checkpoint"],
            "manager_corrected_directive": _manager_requirement()[
                "active_layer16_directive"
            ],
        },
        "cardinality": 1,
        "consumption": {
            "cardinality": 1,
            "initial_state": "ABSENT",
            "overwrite_policy": "CREATE_EXCLUSIVE",
            "path": CONSUMPTION.relative_to(ROOT).as_posix(),
        },
        "exact_invocation": _exact_invocation(),
        "failure_recovery": {
            "after_successful_computation_publication_failure": (
                "SEPARATE_PUBLICATION_ONLY_RECOVERY"
            ),
            "pre_computation_failure": "FRESH_EXECUTION_SUCCESSOR_REQUIRED",
        },
        "identity": IDENTITY,
        "implementation_integrity": {
            "source_execution_package": {
                "path": SOURCE_PACKAGE.relative_to(ROOT).as_posix(),
                "tree_root_sha256": SOURCE_PACKAGE_ROOT,
            },
        },
        "live_predicate_policy": LIVE_PREDICATE_POLICY,
        "manager_requirements": _manager_requirement(),
        "package_creation": {
            "semantics": "CREATE_EXCLUSIVE",
            "superseded_failed_packet_mutation": "FORBIDDEN",
        },
        "permitted_effect": (
            "ONE_POSITION00_LAYER16_PRODUCTION_UNIT_ONLY_AFTER_SOURCE_DISJOINT_"
            "REVIEW_PASS_AND_MANAGER_API_CARDINALITY_ONE_GRANT"
        ),
        "prohibitions": [
            "REPLAY_LAYERS00_15",
            "EXECUTE_LAYER17_OR_LATER",
            "GENERATE_TOKENS",
            "RUN_PPA",
            "RUN_STAGE2",
            "RUN_U280",
            "MUTATE_SEALED_PRIOR_EVIDENCE",
        ],
        "schema": "ace2-stage1-layer16-runtime-pass-authority-v2",
        "stage1": "OPEN",
        "stage2": "FORBIDDEN",
        "u280": "FORBIDDEN",
        "unit_key": UNIT,
        "writable_output_targets": [
            (CONSUMPTION / "consumed.json").relative_to(ROOT).as_posix(),
            (CONSUMPTION / "runtime").relative_to(ROOT).as_posix(),
            (CONSUMPTION / "terminal.json").relative_to(ROOT).as_posix(),
        ],
    }


'''
    text = replace_section(
        text,
        "def authority_record() -> dict[str, Any]:",
        "def acceptance_contract_record() -> dict[str, Any]:",
        authority,
    )

    acceptance = '''def acceptance_contract_record() -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "authority_identity": IDENTITY,
        "create_exclusive": {
            "consumption": True,
            "manager_grant": True,
            "package": True,
        },
        "fresh_reviewer_acceptance": {
            "effect": "PACKAGE_ACCEPTANCE_ONLY",
            "path": REVIEW.relative_to(ROOT).as_posix(),
            "required_decision": "PASS",
            "required_level": "L2",
            "required_producer_role": "reviewer",
            "required_source_disjoint_from_creator": True,
            "schema": "ace2-stage1-layer16-runtime-pass-review-adjudication-v2",
        },
        "live_predicate_policy": LIVE_PREDICATE_POLICY,
        "manager_acceptance": {
            "fresh_review_sha256_binding": "REQUIRED",
            "origin": "IMMUTABLE_FRAMEWORK_MANAGER_HANDOFF",
            "path": GRANT.relative_to(ROOT).as_posix(),
            "required_authority_cardinality": 1,
            "required_decision": "AUTHORIZE_EXACTLY_ONE_LAYER16_UNIT",
            "required_execution_limit": 1,
            "required_producer_role": "manager",
            "requirements": _manager_requirement(),
            "schema": "ace2-stage1-layer16-framework-manager-grant-v2",
        },
        "ordering": [
            "CREATE_EXCLUSIVE_AND_SEAL_LAYER16_PACKAGE",
            "SOURCE_DISJOINT_REVIEWER_ACCEPTS_EXACT_LAYER16_PACKAGE",
            "MANAGER_API_CREATE_EXCLUSIVE_GRANT_BINDS_EXACT_REVIEW_AND_DIRECTIVE",
            "CONSUME_ONCE_BEFORE_LAYER16_EXECUTION",
        ],
        "package_cardinality": 1,
        "package_tree_binding": {
            "algorithm": "SHA256",
            "external_records_must_equal_tree_root": True,
            "package_path": PACKAGE.relative_to(ROOT).as_posix(),
            "tree_root_sidecar": (
                PACKAGE / "TREE_ROOT.sha256"
            ).relative_to(ROOT).as_posix(),
        },
        "position00_layer16_frontier": _frontier_binding(),
        "schema": "ace2-stage1-layer16-acceptance-contract-v2",
        "unit_key": UNIT,
    }


'''
    text = replace_section(
        text,
        "def acceptance_contract_record() -> dict[str, Any]:",
        "def review_request_record() -> dict[str, Any]:",
        acceptance,
    )

    review_request = '''def review_request_record() -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "decision_requested": "PASS",
        "package_path": PACKAGE.relative_to(ROOT).as_posix(),
        "required_checks": [
            "cardinality-one-layer16-specific-create-exclusive-package",
            "sealed-package-member-set-digests-modes-and-root",
            "exact-live-predicate-set-manager-directive-checkpoint-and-unit",
            "no-historical-attempt-or-pre-correction-adjudication-live-predicates",
            "corrected-checkpoint-position00-layer16-frontier",
            "source-disjoint-review-before-manager-api-grant",
            "exact-layer16-only-sealed-invocation",
            "package-native-hostile-controls-pass-with-zero-workload",
            "zero-authority-consumption-and-zero-model-reference-rtl-unit-token-ppa-u280",
        ],
        "review_effect": (
            "ACCEPT_EXACT_LAYER16_PACKAGE_WITHOUT_CREATING_OR_CONSUMING_"
            "EXECUTION_AUTHORITY"
        ),
        "review_level": "L2",
        "review_type": "FRESH_SOURCE_DISJOINT_LAYER16_RUNTIME_PASS_PACKAGE_REVIEW",
        "schema": "ace2-stage1-layer16-runtime-pass-review-request-v2",
        "subject": {
            "authority_identity": IDENTITY,
            "unit_key": UNIT,
        },
    }


'''
    text = replace_section(
        text,
        "def review_request_record() -> dict[str, Any]:",
        "def _external_subject(package_root: str) -> dict[str, Any]:",
        review_request,
    )

    subject = '''def _external_subject(package_root: str) -> dict[str, Any]:
    return {
        "authority_identity": IDENTITY,
        "corrected_checkpoint": _frontier_binding()["corrected_checkpoint"],
        "package_path": PACKAGE.relative_to(ROOT).as_posix(),
        "package_tree_root_sha256": package_root,
        "unit_key": UNIT,
    }


'''
    text = replace_section(
        text,
        "def _external_subject(package_root: str) -> dict[str, Any]:",
        "def expected_review_record(",
        subject,
    )

    expected_review = '''def expected_review_record(
    package_root: str,
    hostile_controls_sha256: str,
) -> dict[str, Any]:
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **ZERO_WORKLOAD,
        },
        "control_receipts": {
            "hostile_controls": {
                "path": (PACKAGE / "hostile-controls.json").relative_to(ROOT).as_posix(),
                "sha256": hostile_controls_sha256,
            },
        },
        "decision": "PASS",
        "execution_performed": False,
        "level": "L2",
        "producer_role": "reviewer",
        "schema": "ace2-stage1-layer16-runtime-pass-review-adjudication-v2",
        "source_disjoint_from_package_creator": True,
        "subject": _external_subject(package_root),
    }


'''
    text = replace_section(
        text,
        "def expected_review_record(",
        "def _directive_binding(record: dict[str, Any]) -> dict[str, Any]:",
        expected_review,
    )

    text = text.replace(
        '"schema": "ace2-stage1-layer16-framework-manager-grant-v1",',
        '"schema": "ace2-stage1-layer16-framework-manager-grant-v2",',
    )

    evidence = '''def _validate_bound_evidence(*, current_directive_required: bool) -> None:
    validate_sealed_tree(SOURCE_PACKAGE, SOURCE_PACKAGE_ROOT)
    _validate_file(CHECKPOINT, CHECKPOINT_SHA256)
    if current_directive_required:
        _validate_file(MANAGER_DIRECTIVE, ACTIVE_DIRECTIVE_SHA256)
        directive = load_json(MANAGER_DIRECTIVE)
        text = str(directive.get("text", ""))
        require(
            directive.get("revision") == ACTIVE_DIRECTIVE_REVISION
            and directive.get("objective_sha256")
            == ACTIVE_DIRECTIVE_OBJECTIVE_SHA256
            and str(directive.get("source", "")).startswith("manager.")
            and CHECKPOINT_SHA256 in text
            and CHECKPOINT_STATUS in text
            and UNIT in text,
            "live corrected layer-16 Manager directive changed",
        )

    checkpoint = load_json(CHECKPOINT)
    units = checkpoint.get("units")
    require(
        checkpoint.get("status") == CHECKPOINT_STATUS
        and checkpoint.get("completed_unit_keys")
        == [f"position-00/layer-{index:02d}" for index in range(16)]
        and checkpoint.get("next_incomplete_unit") == UNIT
        and isinstance(units, list)
        and len(units) == 24
        and units[15].get("output", {}).get("sha256") == LAYER15_OUTPUT_SHA256
        and units[16].get("input_sha256") == LAYER15_OUTPUT_SHA256
        and units[16].get("validation_status") == "INCOMPLETE",
        "corrected position-00/layer-16 checkpoint frontier changed",
    )


'''
    text = replace_section(
        text,
        "def _validate_bound_evidence(*, current_directive_required: bool) -> None:",
        "def live_snapshot() -> dict[str, Any]:",
        evidence,
    )

    snapshot = '''def live_snapshot() -> dict[str, Any]:
    records = {
        str(path): sha256_file(path)
        for path in (
            MANAGER_DIRECTIVE,
            CHECKPOINT,
        )
    }
    return {
        "bound_records": records,
        "consumption_exists": os.path.lexists(CONSUMPTION),
        "layer16_pending": os.path.lexists(LIVE_RUNTIME / ".pending" / UNIT),
        "layer16_published": os.path.lexists(LIVE_RUNTIME / "published" / UNIT),
        "layer16_work": os.path.lexists(LIVE_RUNTIME / "work" / UNIT),
    }


'''
    text = replace_section(
        text,
        "def live_snapshot() -> dict[str, Any]:",
        "def validate_preflight(",
        snapshot,
    )

    preflight = '''def validate_preflight(
    authority: dict[str, Any],
    *,
    current_directive_required: bool = True,
) -> dict[str, str]:
    require(authority == authority_record(), "authority contract changed")
    require(
        set(authority["bindings"])
        == {
            "acceptance_contract",
            "corrected_checkpoint",
            "manager_corrected_directive",
        },
        "live authorization predicate surface changed",
    )
    require(
        authority["live_predicate_policy"] == LIVE_PREDICATE_POLICY,
        "live predicate policy changed",
    )
    require(
        load_json(ACCEPTANCE) == acceptance_contract_record(),
        "acceptance contract changed",
    )
    require(
        load_json(PACKAGE / "review-request.json") == review_request_record(),
        "review request changed",
    )
    _validate_namespace(REVIEW_NAMESPACE, REVIEW)
    _validate_namespace(GRANT_NAMESPACE, GRANT)
    _validate_bound_evidence(
        current_directive_required=current_directive_required,
    )
    require(not os.path.lexists(CONSUMPTION), "layer-16 authority already consumed")
    for candidate in (
        LIVE_RUNTIME / ".pending" / UNIT,
        LIVE_RUNTIME / "published" / UNIT,
        LIVE_RUNTIME / "work" / UNIT,
    ):
        require(not os.path.lexists(candidate), f"stale layer-16 path exists: {candidate}")
    return {
        "cardinality_one": "PASS",
        "corrected_checkpoint": "PASS",
        "create_exclusive": "PASS",
        "exact_invocation": "PASS",
        "historical_attempt_live_predicates": "PASS_ABSENT",
        "manager_corrected_directive": "PASS",
        "pre_correction_adjudication_live_predicates": "PASS_ABSENT",
        "source_execution_package": "PASS",
        "unit_scope": "PASS_POSITION00_LAYER16_ONLY",
    }


'''
    text = replace_section(
        text,
        "def validate_preflight(",
        "def _validate_review(",
        preflight,
    )

    if "ATTEMPT_0012" in text or "TERMINAL_ADJUDICATION" in text:
        raise RuntimeError("forbidden historical predicate survived executor correction")
    if "ace2-layer16-runtime-pass-authority-0001" in text:
        raise RuntimeError("failed package identity survived executor correction")
    return text


def corrected_hostile_controls() -> str:
    return '''#!/usr/bin/env python3
"""Zero-workload hostile controls for corrected layer-16 authority."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Callable


def _load_layer16() -> object:
    path = Path(__file__).with_name("execute_layer16.py")
    spec = importlib.util.spec_from_file_location("execute_layer16_hostile_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load layer-16 executor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layer16 = _load_layer16()


def expect_rejection(name: str, operation: Callable[[], None]) -> dict[str, str]:
    try:
        operation()
    except (layer16.Layer16Error, FileNotFoundError):
        return {"name": name, "result": "PASS_REJECTED_WITHOUT_CONSUMPTION"}
    raise RuntimeError(f"hostile control unexpectedly accepted: {name}")


def run_controls() -> dict[str, object]:
    authority = layer16.load_json(layer16.PACKAGE / "authority.json")
    before = layer16.live_snapshot()
    layer16.validate_preflight(authority)
    cases: list[dict[str, str]] = []

    for name, mutate in (
        (
            "unit-scope",
            lambda value: value.__setitem__("unit_key", "position-00/layer-17"),
        ),
        (
            "cardinality",
            lambda value: value.__setitem__("cardinality", 2),
        ),
        (
            "corrected-checkpoint",
            lambda value: value["bindings"]["corrected_checkpoint"].__setitem__(
                "sha256", "0" * 64
            ),
        ),
        (
            "manager-corrected-directive",
            lambda value: value["bindings"]["manager_corrected_directive"].__setitem__(
                "sha256", "0" * 64
            ),
        ),
        (
            "live-predicate-set",
            lambda value: value["live_predicate_policy"]["allowed"].append(
                "HISTORICAL_STATE"
            ),
        ),
        (
            "sealed-invocation",
            lambda value: value["exact_invocation"]["argv"].append(
                "--alternate-route"
            ),
        ),
    ):
        candidate = copy.deepcopy(authority)
        mutate(candidate)
        cases.append(
            expect_rejection(
                name,
                lambda candidate=candidate: layer16.validate_preflight(candidate),
            )
        )

    after = layer16.live_snapshot()
    layer16.require(before == after, "hostile controls changed the live frontier")
    layer16.require(
        before["consumption_exists"] is False
        and before["layer16_pending"] is False
        and before["layer16_published"] is False
        and before["layer16_work"] is False,
        "hostile controls observed layer-16 execution residue",
    )
    return {
        "activity": {
            "authority_consumed": 0,
            "authority_created": 0,
            **layer16.ZERO_WORKLOAD,
        },
        "cases": cases,
        "execution_performed": False,
        "live_snapshot_sha256_after": layer16.hashlib.sha256(
            layer16.canonical_bytes(after)
        ).hexdigest(),
        "live_snapshot_sha256_before": layer16.hashlib.sha256(
            layer16.canonical_bytes(before)
        ).hexdigest(),
        "schema": "ace2-stage1-layer16-hostile-controls-v2",
        "status": "PASS_ZERO_WORKLOAD_ZERO_CONSUMPTION",
    }


def main() -> int:
    sys.stdout.buffer.write(layer16.canonical_bytes(run_controls()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    if TARGET.exists():
        raise RuntimeError(f"create-exclusive target already exists: {TARGET}")
    TARGET.mkdir(mode=0o755)
    write_exclusive(TARGET / "execute_layer16.py", corrected_executor())
    write_exclusive(TARGET / "hostile_controls.py", corrected_hostile_controls())

    validate = (SOURCE / "validate_nonexecuting.py").read_text(encoding="utf-8")
    validate = validate.replace(
        "ace2-stage1-layer16-runtime-pass-preflight-v1",
        "ace2-stage1-layer16-runtime-pass-preflight-v2",
    )
    write_exclusive(TARGET / "validate_nonexecuting.py", validate)

    seal = (SOURCE / "seal_package.py").read_text(encoding="utf-8")
    seal = seal.replace("package 0001", "package 0002")
    stale_preflight = """    layer16.validate_sealed_tree(
        layer16.LAYER15_PACKAGE,
        layer16.LAYER15_PACKAGE_ROOT,
        require_read_only=True,
    )
"""
    if seal.count(stale_preflight) != 1:
        raise RuntimeError("source sealer layer-15 preflight changed")
    seal = seal.replace(stale_preflight, "")
    import_boundary = "from typing import Any\n\n\nPACKAGE ="
    if seal.count(import_boundary) != 1:
        raise RuntimeError("source sealer import boundary changed")
    seal = seal.replace(
        import_boundary,
        "from typing import Any\n\n\nsys.dont_write_bytecode = True\n\nPACKAGE =",
    )
    write_exclusive(TARGET / "seal_package.py", seal)

    descriptor = os.open(TARGET, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(TARGET.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
