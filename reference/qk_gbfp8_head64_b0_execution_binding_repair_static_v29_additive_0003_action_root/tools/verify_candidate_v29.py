#!/usr/bin/env python3
"""Protected-data-free V29 candidate verifier and counterexample matrix."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema


ACTION_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load_module(ACTION_ROOT / "tools/v29_contract.py", "v29_candidate_contract")
W = load_module(ACTION_ROOT / "tools/future_execute_once_wrapper_v29.py", "v29_candidate_wrapper")


class FakeInfo:
    st_dev = 101
    st_ino = 202


def _rehash(value: dict[str, Any], field: str) -> dict[str, Any]:
    changed = deepcopy(value)
    changed.pop(field, None)
    changed[field] = C.sha256_bytes(C.compact_bytes(changed))
    return changed


def _source_for(records: list[dict[str, Any]], raw: bytes, index: int, role: str) -> dict[str, Any]:
    record = records[index]
    source = {
        "event_end_byte": record["end"],
        "event_hash": C.sha256_bytes(record["line"]),
        "event_line_number": record["line_number"],
        "event_start_byte": record["start"],
        "event_type": "life.manager.intent.started" if role == "MANAGER" else "ui.operator",
        "frontier_byte_count": len(raw),
        "frontier_line_count": len(records),
        "frontier_sha256": C.sha256_bytes(raw),
        "origin_agent_layer": "manager" if role == "MANAGER" else "operator",
        "source_device": FakeInfo.st_dev,
        "source_id": "ARGUS_PROJECT_EVENTS_JSONL",
        "source_inode": FakeInfo.st_ino,
        "source_path": str(C.EVENT_LOG),
    }
    if role == "MANAGER":
        source.update({"intent_id": "intent-v29-static-fixture", "origin_source": "user"})
    else:
        source.update({"message_id": "message-v29-static-fixture"})
    return source


def _capsules(package: dict[str, Any], now_ts: float = 1000.0) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    nonce = "a" * 64
    manager_claim = C.expected_claim(package, "MANAGER", nonce, now_ts, now_ts + 300.0)
    operator_claim = C.expected_claim(package, "OPERATOR", nonce, now_ts, now_ts + 300.0)
    manager_event = {
        "agent_layer": "manager",
        "event_schema_version": 1,
        "intent_id": "intent-v29-static-fixture",
        "objective": C.authority_event_text(manager_claim),
        "source": "user",
        "text": C.authority_event_text(manager_claim),
        "ts": now_ts,
        "type": "life.manager.intent.started",
    }
    operator_event = {
        "agent_layer": "operator",
        "event_schema_version": 1,
        "message_id": "message-v29-static-fixture",
        "text": C.authority_event_text(operator_claim),
        "ts": now_ts,
        "type": "ui.operator",
    }
    raw = C.compact_bytes(manager_event) + C.compact_bytes(operator_event)
    records = C._event_records(raw)
    manager = _rehash(
        {
            "artifact_kind": "qk_gbfp8_head64_b0_v29_external_manager_admission_capsule",
            "claim": manager_claim,
            "role": "MANAGER",
            "schema_version": 1,
            "source": _source_for(records, raw, 0, "MANAGER"),
        },
        "capsule_sha256",
    )
    operator = _rehash(
        {
            "artifact_kind": "qk_gbfp8_head64_b0_v29_external_operator_authority_capsule",
            "claim": operator_claim,
            "role": "OPERATOR",
            "schema_version": 1,
            "source": _source_for(records, raw, 1, "OPERATOR"),
        },
        "capsule_sha256",
    )
    return raw, manager, operator


def _expect_rejected(case_id: str, operation: Any, rejected: list[str]) -> None:
    try:
        operation()
    except Exception:
        rejected.append(case_id)
    else:
        raise C.ContractError("NEGATIVE_CASE_ACCEPTED", case_id)


def _expect_contract_code(case_id: str, code: str, operation: Any, rejected: list[str]) -> None:
    try:
        operation()
    except Exception as error:
        observed = getattr(error, "code", None)
        C.require(observed == code, "NEGATIVE_CASE_WRONG_CODE", f"{case_id}:{observed}")
        rejected.append(case_id)
    else:
        raise C.ContractError("NEGATIVE_CASE_ACCEPTED", case_id)


def _mutate_capsule(capsule: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    changed = deepcopy(capsule)
    parent: Any = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    if path[0] == "claim":
        changed["claim"] = _rehash(changed["claim"], "claim_sha256")
    return _rehash(changed, "capsule_sha256")


def _reviewer_fixture(
    package: dict[str, Any],
    report: dict[str, Any],
    report_raw: bytes,
    *,
    completed: bool = False,
    wrong_command: bool = False,
    wrong_output: bool = False,
) -> bytes:
    call_id = "1" * 32
    commitment = C.reviewer_preflight_commitment(package, report, report_raw, call_id)
    if wrong_output:
        commitment["candidate_action_tree_sha256"] = "9" * 64
        commitment = _rehash(commitment, "commitment_sha256")
    payload = "true" if wrong_command else C.reviewer_preflight_command()
    lines = [
        {"call_id": call_id, "run_label": "reviewer", "ts": 10.0, "type": "provider.request.started"},
        {
            "call_id": call_id,
            "prompt": f"Fresh-L2 mission {C.MISSION_ID}; generated successor identity is bound after scheduler start",
            "run_label": "reviewer",
            "ts": 10.1,
            "type": "agent.io.start",
            "working_dir": str(C.PROJECT_ROOT),
        },
        {
            "actor": "reviewer",
            "agent_layer": "reviewer",
            "exit_code": 0,
            "kind": "command_execution",
            "output_excerpt": C.compact_bytes(commitment).decode("ascii").strip(),
            "status": "completed",
            "text": f"/bin/bash -lc {shlex.quote(payload)}",
            "ts": 10.2,
            "type": "engineer.progress",
        },
    ]
    if completed:
        lines.append({"call_id": call_id, "ts": 10.3, "type": "agent.io.complete"})
    return b"".join(C.compact_bytes(item) for item in lines)


class _PreOpenSwapPath:
    def __init__(self, target: Path, replacement: Path) -> None:
        self.target = target
        self.replacement = replacement
        self.calls = 0

    def __fspath__(self) -> str:
        self.calls += 1
        if self.calls == 2:
            os.replace(self.replacement, self.target)
        return str(self.target)

    def __str__(self) -> str:
        return str(self.target)


def _run_real_pathname_races(rejected: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="v29-additive-0003-races-", dir="/tmp") as temporary:
        root = Path(temporary)

        capsule = root / "capsule.json"
        capsule_replacement = root / "capsule-replacement.json"
        capsule.write_bytes(C.compact_bytes({"generation": "original"}))
        capsule_replacement.write_bytes(C.compact_bytes({"generation": "replacement"}))
        _expect_contract_code(
            "capsule_substitution_between_lstat_and_open_real_race",
            "CAPSULE_REPLACED_BEFORE_OPEN",
            lambda: W._read_stable_capsule(_PreOpenSwapPath(capsule, capsule_replacement)),
            rejected,
        )

        event_log = root / "events.jsonl"
        event_log_replacement = root / "events-replacement.jsonl"
        event_log.write_bytes(C.compact_bytes({"type": "fixture.original"}))
        event_log_replacement.write_bytes(C.compact_bytes({"type": "fixture.replacement"}))
        original_read = W.C.os.read
        replaced = False

        def read_after_replacement(descriptor: int, count: int) -> bytes:
            nonlocal replaced
            if not replaced:
                replaced = True
                os.replace(event_log_replacement, event_log)
            return original_read(descriptor, count)

        W.C.os.read = read_after_replacement
        try:
            _expect_contract_code(
                "event_log_replacement_after_open_before_read_completion_real_race",
                "EVENT_LOG_REPLACED_AFTER_READ",
                lambda: W.C._read_event_log_stable(event_log),
                rejected,
            )
        finally:
            W.C.os.read = original_read

        move_source = root / "move-source.json"
        move_replacement = root / "move-replacement.json"
        move_destination = root / "consumed.json"
        move_source.write_bytes(C.compact_bytes({"generation": "opened"}))
        move_replacement.write_bytes(C.compact_bytes({"generation": "substituted-before-move"}))
        opened = os.lstat(move_source)
        original_rename = W.os.rename

        def rename_after_substitution(source: Any, destination: Any) -> None:
            os.replace(move_replacement, move_source)
            original_rename(source, destination)

        W.os.rename = rename_after_substitution
        try:
            _expect_contract_code(
                "capsule_substitution_before_final_move_real_race",
                "CAPSULE_RENAME_SUBSTITUTION",
                lambda: W._move_consumed_inode(move_source, move_destination, opened, "OPERATOR"),
                rejected,
            )
        finally:
            W.os.rename = original_rename


def _run_real_atomic_duplicate_start(rejected: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="v29-additive-0003-duplicate-start-", dir="/tmp") as temporary:
        root = Path(temporary)
        runtime = root / "runtime"
        failure_seal = root / "runtime.failed-start"
        original_runtime = W.C.RUNTIME_NAMESPACE
        original_failure_seal = W.C.FAILURE_SEAL_NAMESPACE
        original_run = W.subprocess.run
        process_starts = 0

        def forbidden_process_start(*args: Any, **kwargs: Any) -> Any:
            nonlocal process_starts
            process_starts += 1
            raise AssertionError("protected process start reached after duplicate terminal")

        W.C.RUNTIME_NAMESPACE = runtime
        W.C.FAILURE_SEAL_NAMESPACE = failure_seal
        W.subprocess.run = forbidden_process_start
        claim_token = "b" * 64
        try:
            W._claim_absent_runtime(runtime)
            W._initialize_claimed_runtime(runtime, claim_token)
            _expect_contract_code(
                "duplicate_start_real_atomic_claim_loser_terminalizes_runtime",
                "DUPLICATE_START_OR_RUNTIME_PREEXISTENCE",
                lambda: W._claim_absent_runtime(runtime),
                rejected,
            )
            W._seal_preclaim_failure("DUPLICATE_START_OR_RUNTIME_PREEXISTENCE", str(runtime))
            C.require((runtime / "terminal/first-terminal.json").is_file(), "DUPLICATE_RUNTIME_TERMINAL_ABSENT", str(runtime))
            C.require((failure_seal / "first-terminal.json").is_file(), "DUPLICATE_SIBLING_TERMINAL_ABSENT", str(failure_seal))
            _expect_contract_code(
                "duplicate_start_real_first_claimant_blocked_before_protected_boundary",
                "DUPLICATE_START_TERMINAL",
                lambda: W._protected_boundary(runtime, claim_token, root / "absent-envelope.json", {}),
                rejected,
            )
            C.require(process_starts == 0, "DUPLICATE_START_PROTECTED_PROCESS_STARTED", str(process_starts))
        finally:
            W.subprocess.run = original_run
            W.C.RUNTIME_NAMESPACE = original_runtime
            W.C.FAILURE_SEAL_NAMESPACE = original_failure_seal
            for path in (runtime / "terminal", runtime, failure_seal):
                if os.path.lexists(path):
                    os.chmod(path, 0o700)


def run_regressions(package: dict[str, Any]) -> dict[str, Any]:
    manager_schema, _ = C.load_canonical(ACTION_ROOT / package["authority_contract"]["manager_capsule_schema_path"])
    operator_schema, _ = C.load_canonical(ACTION_ROOT / package["authority_contract"]["operator_capsule_schema_path"])
    raw, manager, operator = _capsules(package)
    jsonschema.Draft202012Validator(manager_schema).validate(manager)
    jsonschema.Draft202012Validator(operator_schema).validate(operator)
    C.validate_authority_capsule(manager, "MANAGER", package, 1100.0, raw_override=raw, info_override=FakeInfo())
    C.validate_authority_capsule(operator, "OPERATOR", package, 1100.0, raw_override=raw, info_override=FakeInfo())

    dummy_report = {
        "preacceptance_tree_sha256": "1" * 64,
        "regression_case_count": 30,
        "report_sha256": "2" * 64,
    }
    dummy_report_raw = C.compact_bytes(dummy_report)
    reviewer_raw = _reviewer_fixture(package, dummy_report, dummy_report_raw)
    reviewer_records = C._event_records(reviewer_raw)
    C.require(C.STATIC_ACTION_ID not in reviewer_records[1]["event"]["prompt"], "REVIEWER_FIXTURE_START_PREBINDS_ACTION", C.STATIC_ACTION_ID)
    reviewer_identity = C.derive_reviewer_provenance(reviewer_raw, FakeInfo(), package, dummy_report, dummy_report_raw)
    C.require(
        reviewer_identity["candidate_action_tree_sha256"] == dummy_report["preacceptance_tree_sha256"],
        "REVIEWER_PREFLIGHT_CANDIDATE_BINDING",
        "tree",
    )

    rejected: list[str] = []
    _expect_rejected(
        "v28_counterexample_arbitrary_synthetic_operator_digest",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("source", "event_hash"), "f" * 64),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    _expect_rejected(
        "wrong_event_source_id",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("source", "source_id"), "ROLE_AUTHORED_JSON"),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    _expect_rejected(
        "wrong_event_source_path",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("source", "source_path"), "/tmp/fake-events.jsonl"),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    _expect_rejected(
        "wrong_event_frontier_hash",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("source", "frontier_sha256"), "e" * 64),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    unrelated_event = C.compact_bytes({"ts": 1001.0, "type": "authority.unrelated", "x": "x"})
    C.require(len(unrelated_event) == 51, "AUTHORITY_FRONTIER_COUNTEREXAMPLE_SIZE", str(len(unrelated_event)))
    _expect_contract_code(
        "authority_exact_live_frontier_rejects_unrelated_51_byte_append",
        "EVENT_LIVE_FRONTIER_DRIFT",
        lambda: C.validate_authority_capsule(
            operator,
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw + unrelated_event,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    _expect_rejected(
        "wrong_event_frontier_line",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("source", "event_line_number"), 1),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    _expect_rejected(
        "wrong_event_type",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("source", "event_type"), "engineer.progress"),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    _expect_rejected(
        "stale_operator_authority",
        lambda: C.validate_authority_capsule(operator, "OPERATOR", package, 1400.0, raw_override=raw, info_override=FakeInfo()),
        rejected,
    )
    _expect_rejected(
        "authority_replay_consumed_nonce",
        lambda: C.validate_authority_capsule(operator, "OPERATOR", package, 1100.0, {"a" * 64}, raw, FakeInfo()),
        rejected,
    )
    _expect_rejected(
        "capsule_role_substitution",
        lambda: C.validate_authority_capsule(manager, "OPERATOR", package, 1100.0, raw_override=raw, info_override=FakeInfo()),
        rejected,
    )
    _expect_rejected(
        "capsule_package_hash_substitution",
        lambda: C.validate_authority_capsule(
            _mutate_capsule(operator, ("claim", "package_file_sha256"), "d" * 64),
            "OPERATOR",
            package,
            1100.0,
            raw_override=raw,
            info_override=FakeInfo(),
        ),
        rejected,
    )
    completion = {"nonce": "a" * 64, "ts": 1001.0, "type": "life.operator.execution.completed"}
    replay_raw = raw + C.compact_bytes(completion)
    _expect_rejected(
        "completed_call_authority_replay",
        lambda: C.validate_authority_capsule(operator, "OPERATOR", package, 1100.0, raw_override=replay_raw, info_override=FakeInfo()),
        rejected,
    )
    _expect_rejected(
        "reviewer_completed_call_replay",
        lambda: C.derive_reviewer_provenance(
            _reviewer_fixture(package, dummy_report, dummy_report_raw, completed=True),
            FakeInfo(),
            package,
            dummy_report,
            dummy_report_raw,
        ),
        rejected,
    )
    _expect_rejected(
        "reviewer_wrong_preflight_command",
        lambda: C.derive_reviewer_provenance(
            _reviewer_fixture(package, dummy_report, dummy_report_raw, wrong_command=True),
            FakeInfo(),
            package,
            dummy_report,
            dummy_report_raw,
        ),
        rejected,
    )
    _expect_rejected(
        "reviewer_wrong_preflight_output_binding",
        lambda: C.derive_reviewer_provenance(
            _reviewer_fixture(package, dummy_report, dummy_report_raw, wrong_output=True),
            FakeInfo(),
            package,
            dummy_report,
            dummy_report_raw,
        ),
        rejected,
    )
    _run_real_pathname_races(rejected)
    _run_real_atomic_duplicate_start(rejected)

    fake_provenance = {"provenance_identity_sha256": "3" * 64}
    baseline_acceptance = C.add_self_hash(C.acceptance_unhashed(package, dummy_report, dummy_report_raw, fake_provenance), "acceptance_sha256")
    wrong_tree = deepcopy(baseline_acceptance)
    wrong_tree["candidate_action_tree_sha256"] = "4" * 64
    wrong_tree = _rehash(wrong_tree, "acceptance_sha256")
    _expect_rejected(
        "v28_counterexample_wrong_candidate_tree",
        lambda: C.verify_acceptance(package, wrong_tree, C.compact_bytes(wrong_tree), dummy_report),
        rejected,
    )
    missing_reviewer = deepcopy(baseline_acceptance)
    del missing_reviewer["creator_provenance"]
    missing_reviewer = _rehash(missing_reviewer, "acceptance_sha256")
    _expect_rejected(
        "v28_counterexample_missing_reviewer_provenance",
        lambda: C.verify_acceptance(package, missing_reviewer, C.compact_bytes(missing_reviewer), dummy_report),
        rejected,
    )
    wrong_reviewer = deepcopy(baseline_acceptance)
    wrong_reviewer["creator_provenance"] = {"actor": "engineer", "provenance_identity_sha256": "5" * 64}
    wrong_reviewer = _rehash(wrong_reviewer, "acceptance_sha256")
    _expect_rejected(
        "wrong_reviewer_self_asserted_role",
        lambda: C.verify_acceptance(package, wrong_reviewer, C.compact_bytes(wrong_reviewer), dummy_report),
        rejected,
    )

    transitions = {
        "runtime_preexistence": C.transition_model(True, False, 1, True),
        "prior_failure_seal": C.transition_model(False, True, 1, True),
        "duplicate_starts": C.transition_model(False, False, 2, True),
        "capsule_failure_after_claim": C.transition_model(False, False, 1, False),
        "nominal_order": C.transition_model(False, False, 1, True),
    }
    C.require(transitions["runtime_preexistence"] == "REJECT_RUNTIME_PREEXISTENCE_AND_SEAL_SIBLING", "TRANSITION_RUNTIME_PREEXISTENCE", "model")
    C.require(transitions["duplicate_starts"] == "REJECT_DUPLICATE_START", "TRANSITION_DUPLICATE_START", "model")
    C.require(transitions["nominal_order"] == "CLAIM_CONSUME_FREEZE_BEFORE_PROTECTED_BOUNDARY", "TRANSITION_ORDER", "model")

    adapter_cases = ["spawn", "abi", "return_code", "stdout", "stderr", "decode", "result_schema", "result_record", "exception", "publication"]
    adapter_boundaries = {case: C.adapter_boundary_model(case) for case in adapter_cases}

    oracle = C.synthetic_rows()
    exact = C.classify_rows(deepcopy(oracle), oracle)
    C.require(exact["classification"] == "SOURCE_ORACLE_MATCH", "SOURCE_ORACLE_MATCH_REGRESSION", "exact")
    mismatch_digests: set[str] = set()
    for index in range(574):
        source = deepcopy(oracle)
        row = source[index]
        source[index] = (row[0], row[1] + 1, row[2])
        outcome = C.classify_rows(source, oracle)
        C.require(outcome["classification"] == "SOURCE_ORACLE_MISMATCH" and outcome["mismatch_count"] == 1, "SOURCE_ORACLE_MISMATCH_REGRESSION", str(index))
        mismatch_digests.add(outcome["mismatch_membership_sha256"])
    C.require(len(mismatch_digests) == 574, "SOURCE_ORACLE_POSITION_COVERAGE", str(len(mismatch_digests)))

    wrapper_source = (ACTION_ROOT / "tools/future_execute_once_wrapper_v29.py").read_text(encoding="ascii")
    order_tokens = ["_claim_absent_runtime(", "_consume_capsule(", "_write_frozen_envelope(", "_protected_boundary("]
    positions = [wrapper_source.index(token) for token in order_tokens]
    C.require(positions == sorted(positions), "WRAPPER_ORDERING", str(positions))
    C.require("shell=False" in wrapper_source and "shell=True" not in wrapper_source, "WRAPPER_SHELL_FALSE", "source")

    passed = [
        "manager_authority_positive",
        "operator_authority_positive",
        "reviewer_scheduler_mission_start_without_generated_action_id",
        "reviewer_authoritative_preflight_command_candidate_binding",
        "source_oracle_match_574_rows",
        "source_oracle_mismatch_each_of_574_rows",
        "safe_aggregate_only_output",
        "runtime_preexistence_model",
        "claim_consume_freeze_order",
        *[f"adapter_boundary_{case}" for case in adapter_cases],
    ]
    all_cases = passed + rejected
    return {
        "adapter_boundaries": adapter_boundaries,
        "case_ids": all_cases,
        "classification_position_test_count": 575,
        "rejected_case_ids": rejected,
        "regression_case_count": len(all_cases),
        "safe_output_fields": sorted(exact),
        "transition_results": transitions,
        "unique_mismatch_membership_digest_count": len(mismatch_digests),
    }


def verify_candidate(allow_acceptance: bool = False, allow_live_execution_recheck: bool = False) -> dict[str, Any]:
    package, package_raw = C.load_canonical(C.PACKAGE_PATH)
    C.verify_package(package, package_raw)
    for relative in (
        package["authority_contract"]["manager_capsule_schema_path"],
        package["authority_contract"]["operator_capsule_schema_path"],
        package["fresh_l2_review"]["acceptance_schema_path"],
    ):
        schema, _ = C.load_canonical(ACTION_ROOT / relative)
        jsonschema.Draft202012Validator.check_schema(schema)
    C.require(C.sha256_file(C.INTERPRETER) == C.INTERPRETER_SHA256, "INTERPRETER_HASH", str(C.INTERPRETER))
    if allow_live_execution_recheck:
        C.require(allow_acceptance, "LIVE_RECHECK_REQUIRES_ACCEPTANCE", "allow_acceptance")
        C.require(C.RUNTIME_NAMESPACE.is_dir(), "LIVE_RECHECK_RUNTIME_ABSENT", str(C.RUNTIME_NAMESPACE))
    else:
        C.require(not os.path.lexists(C.RUNTIME_NAMESPACE), "OFFICIAL_RUNTIME_MATERIALIZED", str(C.RUNTIME_NAMESPACE))
        C.require(not os.path.lexists(C.FAILURE_SEAL_NAMESPACE), "OFFICIAL_FAILURE_SEAL_MATERIALIZED", str(C.FAILURE_SEAL_NAMESPACE))
        C.require(not os.path.lexists(C.AUTHORITY_NAMESPACE), "OFFICIAL_AUTHORITY_NAMESPACE_MATERIALIZED", str(C.AUTHORITY_NAMESPACE))
    inventory = C.verify_exact_inventory(package, allow_acceptance)
    regressions = run_regressions(package)
    report = {
        "action_id": C.STATIC_ACTION_ID,
        "artifact_kind": "qk_gbfp8_head64_b0_execution_binding_repair_static_v29_candidate_report",
        "claim_boundary": C.CLAIM_BOUNDARY,
        "execution_binding_sha256": package["future_execution_binding"]["execution_binding_sha256"],
        "generated_files": C.verify_generated_files(package),
        "inventory": inventory,
        "official_evaluator_invocation_count": 0,
        "official_payload_open_count": 0,
        "official_target_process_starts": 0,
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": C.sha256_bytes(package_raw),
        "preacceptance_tree_sha256": inventory["preacceptance_tree_sha256"],
        "predecessor_preservation": C.verify_predecessor_preservation(),
        "regression_case_count": regressions["regression_case_count"],
        "regressions": regressions,
        "runtime_namespace_materialized": False,
        "static_acceptance_grants_execution_authority": False,
        "status": "PASS_V29_STATIC_CANDIDATE",
    }
    report["report_sha256"] = C.sha256_bytes(C.compact_bytes(report))
    return report


def main() -> int:
    C.install_audit_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-acceptance", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify_candidate(args.allow_acceptance)
    raw = C.compact_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
