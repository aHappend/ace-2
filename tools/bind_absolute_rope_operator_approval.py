#!/usr/bin/env python3
"""Bind the real Manager-inbox approval for the absolute-RoPE RTL attempt.

This tool records authority that already exists in the active Manager inbox.  It
does not create authority and it never edits the Manager-owned pipeline stage.
All project authorization surfaces are staged and committed under the active
Manager session and pipeline locks; the approval artifact is the commit marker.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "layer0_absolute_rope_online_attention_v1"
PROPOSAL_SHA256 = "b14489f8c882ff53f27db87612238efa06ef8d102930da15b2916ef314b683de"
APPROVED_AT = "2026-08-01T05:43:00.142529Z"
INBOX_TS = 1785562980.1425285
INBOX_LINE_SHA256 = "647b677166eb72dd945a99d3ffcdbfabd70ee6d821531790d6b79deada7bba0a"
APPROVAL_REL = Path("evidence/authorization") / CONTRACT / "operator_approval.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(relative: str | Path) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {relative}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approval_binding(transaction_id: str, inbox: Path) -> dict[str, Any]:
    return {
        "authority": "operator",
        "approved_at_utc": APPROVED_AT,
        "contract_id": CONTRACT,
        "manager_inbox": {
            "entry_sha256": INBOX_LINE_SHA256,
            "entry_timestamp": INBOX_TS,
            "session_id": os.environ["ARGUS_SKILL_SESSION_ID"],
            "source": "active_manager_inbox",
        },
        "proposal_sha256": PROPOSAL_SHA256,
        "scope": "one_bounded_non_stage_closing_rtl_implementation_attempt_then_focused_software_rtl_tests_and_unique_paired_smoke",
        "stage_closing": False,
        "stop_rule": "both_wikitext2_and_c4_must_strictly_improve_or_seal_bounded_no_go",
        "transaction_id": transaction_id,
        "forbidden": [
            "wholesale_quarantine_restore",
            "architecture_contract_change",
            "historical_frontier_replacement",
            "full_shell_before_paired_smoke_pass",
            "canonical_or_noncanonical_ppa_before_paired_smoke_pass",
            "official_14_item_evaluation",
            "continuation_after_paired_smoke_failure",
        ],
        "manager_inbox_path_privacy": "resolved_from_ARGUS_SKILL_SESSION_ROOT_not_published",
        "manager_inbox_sha256_at_binding": sha256(inbox),
    }


def set_authorized(record: dict[str, Any], binding: dict[str, Any], status: str) -> None:
    require(record.get("contract_id") == CONTRACT, "active contract mismatch")
    record["authority"] = "operator"
    record["approved_at_utc"] = APPROVED_AT
    record["implementation_authorized"] = True
    record["implementation_approval"] = binding
    record["required_operator_action"] = "satisfied_by_manager_inbox_entry_43"
    record["required_manager_action"] = "none_current_stage_remains_rtl"
    record["stage_closing"] = False
    record["status"] = status


def build_payloads(binding: dict[str, Any]) -> dict[Path, bytes]:
    payloads: dict[Path, bytes] = {}

    policy = load_json("design/FAST_LOOP_POLICY.json")
    set_authorized(
        policy["active_repair_authorization"], binding,
        "operator_approved_bounded_rtl_attempt_not_started",
    )
    active = policy["active_repair_authorization"]
    active["approval_evidence"] = binding
    active["execution_status"] = "authorized_not_started"
    active["task_count"] = 1
    active["updated_at_utc"] = APPROVED_AT
    set_authorized(
        policy["architecture_proposal_authorization"], binding,
        "independently_reviewed_environment_complete_operator_approved",
    )
    policy["architecture_proposal_authorization"]["task_count"] = 1
    policy["architecture_proposal_authorization"]["updated_at_utc"] = APPROVED_AT
    set_authorized(
        policy["selected_replacement_contract"], binding,
        "operator_approved_implementation_not_started",
    )
    policy["selected_replacement_contract"]["execution_result"] = {
        "rtl_change_started": False,
        "focused_software_rtl_tests_run": False,
        "paired_smoke_run": False,
        "full_shell_regression_run": False,
        "canonical_sky130_ppa_run": False,
        "official_14_item_evaluation_run": False,
    }
    policy["selected_replacement_contract"]["updated_at_utc"] = APPROVED_AT
    payloads[Path("design/FAST_LOOP_POLICY.json")] = encode_json(policy)

    target = load_json("design/TARGET.json")
    target["generated_at_utc"] = APPROVED_AT
    fast = target["fast_loop_contract"]
    fast["manager_recommendation"] = "remain_in_rtl_execute_one_operator_approved_bounded_attempt"
    for name in ("active_repair_authorization", "architecture_proposal_authorization"):
        set_authorized(
            fast[name], binding,
            "operator_approved_bounded_rtl_attempt_not_started",
        )
        fast[name]["approval_scope"] = binding["scope"]
        fast[name]["first_gate"] = "focused_software_and_rtl_tests_then_unique_paired_smoke"
    payloads[Path("design/TARGET.json")] = encode_json(target)

    chip = load_json("design/CHIP_SCOPE.json")
    override = chip["authority_override"]
    require(override.get("active_replacement_contract") == CONTRACT, "CHIP_SCOPE contract mismatch")
    override.update({
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
        "required_next_action": "execute_one_bounded_rtl_attempt_in_current_rtl_stage",
    })
    rope = chip["numerical_behavior"]["rope"]["absolute_rope_online_attention_contract"]
    require(rope.get("contract_id") == CONTRACT, "CHIP_SCOPE RoPE contract mismatch")
    rope.update({
        "architecture_review_status": "independently_accepted_environment_complete",
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
    })
    proposal = chip["operator_owned_execution_policy"]["sixth_mechanism_architecture_proposal"]
    require(proposal.get("contract_id") == CONTRACT, "CHIP_SCOPE proposal mismatch")
    proposal.update({
        "authority": "operator",
        "approval_scope": binding["scope"],
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
        "task_count": 1,
        "status": "operator_approved_bounded_rtl_attempt_not_started",
        "required_operator_action": "satisfied_by_manager_inbox_entry_43",
        "required_manager_action": "none_current_stage_remains_rtl",
        "proposed_first_gate": "focused_software_and_rtl_tests_then_unique_paired_smoke",
        "proposed_after_gate": "seal_no_go_on_failure_otherwise_stop_for_new_authority_before_full_shell_or_ppa",
        "observed_result": "not_run_operator_approval_bound",
    })
    chip["stage"]["current_stage_status"] = "operator_approved_bounded_rtl_attempt_not_started"
    payloads[Path("design/CHIP_SCOPE.json")] = encode_json(chip)

    memory = load_json("design/MEMORY_MODEL.json")
    override = memory["authority_override"]
    require(override.get("active_replacement_contract") == CONTRACT, "MEMORY_MODEL contract mismatch")
    override.update({
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
        "required_next_action": "execute_one_bounded_rtl_attempt_in_current_rtl_stage",
    })
    memory["claim_status"] = "operator_authorized_rtl_implementation_not_yet_tested_or_accepted"
    payloads[Path("design/MEMORY_MODEL.json")] = encode_json(memory)

    oracle = load_json("reference/ORACLE_MANIFEST.json")
    proposed = oracle["proposed_architecture_contract"]
    set_authorized(proposed, binding, "operator_approved_focused_vectors_required")
    proposed["existing_vectors_status"] = "historical_vectors_plus_new_contract_vectors_pending_generation"
    oracle["generated_at_utc"] = APPROVED_AT
    oracle["claim_boundary"] = (
        "Reference/vector and bounded RTL implementation work is operator-authorized for the active contract; "
        "no quality, accepted capability, PPA, stage-closure, prototype, benchmark, signoff, or silicon claim exists."
    )
    payloads[Path("reference/ORACLE_MANIFEST.json")] = encode_json(oracle)

    manifest = load_json("design/RTL_MANIFEST.json")
    replacement = manifest["proposed_replacement_contract"]
    set_authorized(replacement, binding, "operator_approved_implementation_not_started")
    replacement["implementation_result"] = {
        "accepted_rtl_change": False,
        "accepted_verification_evidence": False,
        "accepted_ppa_evidence": False,
    }
    manifest["generated_at_utc"] = APPROVED_AT
    manifest["architecture_contract_status"] = (
        "layer0_absolute_rope_online_attention_v1_independently_reviewed_environment_complete_"
        "operator_approved_implementation_not_started"
    )
    manifest["traceability"]["architecture_contract_gap"] = {
        "status": "implementation_authorized_not_yet_implemented",
        "resolution_owner": "Planner executes the single bounded current-stage attempt",
    }
    payloads[Path("design/RTL_MANIFEST.json")] = encode_json(manifest)

    status = load_json("research/PUBLIC_STATUS.json")
    status["architecture_proposal_gate"].update({
        "implementation_authorized": True,
        "implementation_approval": binding,
        "status": "rtl_entry_complete_operator_approved_implementation_not_started",
    })
    status["selected_replacement_contract"].update({
        "implementation_authorized": True,
        "implementation_approval": binding,
        "status": "rtl_entry_complete_operator_approved_implementation_not_started",
    })
    dashboard = status["dashboard_fields"]
    dashboard["candidate_mechanism"].update({
        "implementation_authorized": True,
        "implementation_approval": binding,
        "status": "operator_approved_bounded_rtl_attempt_not_started",
    })
    dashboard["current_architecture_performance_model"]["implementation_authorized"] = True
    dashboard["current_architecture_performance_model"]["status"] = (
        "operator_approved_architecture_estimate_pending_rtl_and_test_evidence"
    )
    dashboard["operator_policy"]["active_repair_authorization"] = policy["active_repair_authorization"]
    dashboard["operator_policy"]["selected_replacement_contract"] = policy["selected_replacement_contract"]
    dashboard["latest_decision"] = "operator_approved_bounded_absolute_rope_rtl_attempt_not_started"
    dashboard["routing_status"] = "rtl_operator_approved_bounded_attempt"
    status["implementation_frontier"].update({
        "latest_decision": "operator_approved_bounded_absolute_rope_rtl_attempt_not_started",
        "required_manager_action": "none_current_stage_remains_rtl",
        "required_operator_action": "satisfied_for_exactly_one_bounded_attempt",
        "routing_status": "rtl_operator_approved_bounded_attempt",
    })
    status["latest_decision"] = "operator_approved_bounded_absolute_rope_rtl_attempt_not_started"
    status["stage"]["current_stage_status"] = "operator_approved_bounded_rtl_attempt_not_started"
    payloads[Path("research/PUBLIC_STATUS.json")] = encode_json(status)

    mission = (ROOT / "MISSION.md").read_text(encoding="utf-8")
    old = (
        "Independent architecture review accepted the structurally distinct sixth\n"
        "contract `layer0_absolute_rope_online_attention_v1` for environment entry on\n"
        "2026-07-31 at 20:12:25 UTC. Independent environment recertification completed,\n"
        "and the Manager advanced the project to `rtl` at 22:40:46 UTC while explicitly\n"
        "preserving `implementation_authorized=false`. No fresh explicit operator\n"
        "implementation approval exists for this sixth contract, so no RTL edit,\n"
        "focused quality discriminator, or downstream executable run is authorized.\n"
        "The 1.05x quality limit, 2.0 mm^2 non-SRAM cap, and 100 MHz floor remain\n"
        "unchanged."
    )
    new = (
        "Independent architecture review accepted the structurally distinct sixth\n"
        "contract `layer0_absolute_rope_online_attention_v1` for environment entry on\n"
        "2026-07-31 at 20:12:25 UTC. Independent environment recertification completed,\n"
        "and the Manager advanced the project to `rtl` at 22:40:46 UTC. The operator\n"
        "then explicitly approved exactly one bounded, non-stage-closing implementation\n"
        "attempt at 2026-08-01 05:43:00 UTC, bound to proposal SHA-256\n"
        f"`{PROPOSAL_SHA256}`. The attempt may run focused software/RTL tests and the\n"
        "unique WikiText-2/C4 paired smoke only; failure on either dataset seals an\n"
        "immediate bounded no-go before full shell, PPA, or official-14 evaluation.\n"
        "The 1.05x quality limit, 2.0 mm^2 non-SRAM cap, and 100 MHz floor remain\n"
        "unchanged."
    )
    require(old in mission, "MISSION authorization paragraph changed")
    payloads[Path("MISSION.md")] = mission.replace(old, new).encode("utf-8")

    trace = (ROOT / "design/RTL_TRACEABILITY.md").read_text(encoding="utf-8")
    trace = trace.replace(
        "- Status: architecture and environment gates complete; the Manager-owned stage\n"
        "  is `rtl`, pending fresh explicit operator implementation approval.\n"
        "- Stage closing: false.\n- RTL modified: no.\n- Implementation authorized: no.",
        "- Status: architecture and environment gates complete; the Manager-owned stage\n"
        "  is `rtl`, and the exact hash-bound bounded implementation attempt is operator-approved.\n"
        "- Stage closing: false.\n- RTL modified: no.\n- Implementation authorized: yes, exactly one bounded attempt.",
    )
    trace = trace.replace(
        "`rtl.contract-traceability` remains false because no approved or implemented\n"
        "RTL supplies the proposed layer-0 operation. The proposal itself introduces no\n"
        "capability claim. Architecture and environment acceptance are complete; fresh\n"
        "operator approval is the remaining prerequisite before the bounded RTL task\n"
        "may begin.",
        "`rtl.contract-traceability` remains false until the approved RTL is implemented\n"
        "and bound to the manifest. Architecture, environment, and exact operator\n"
        "authorization gates are complete; implementation and focused evidence remain."
    )
    payloads[Path("design/RTL_TRACEABILITY.md")] = trace.encode("utf-8")

    checkpoint = (ROOT / "CHECKPOINT.md").read_text(encoding="utf-8")
    checkpoint = checkpoint.replace(
        "one bounded non-milestone RTL task only after its required fresh explicit\noperator implementation approval is recorded.",
        "one bounded non-milestone RTL task under the exact operator approval recorded\nat 2026-08-01T05:43:00Z."
    )
    checkpoint += (
        "\n## Fresh Operator Approval\n\n"
        f"The active Manager inbox records exact implementation approval for `{CONTRACT}`\n"
        f"at `{APPROVED_AT}`, bound to proposal SHA-256 `{PROPOSAL_SHA256}`. The approval\n"
        "permits exactly one bounded non-stage-closing RTL attempt, focused software/RTL\n"
        "tests, and the unique paired smoke. It forbids wholesale quarantine restoration,\n"
        "frontier replacement, full shell/PPA/official-14 before the gate, and continuation\n"
        "after failure on either dataset.\n"
    )
    payloads[Path("CHECKPOINT.md")] = checkpoint.encode("utf-8")

    payloads[APPROVAL_REL] = encode_json(binding)
    return payloads


def encode_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_external_lock(path: Path) -> None:
    """Require the active Manager/Planner parent to own a serialization lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            raise RuntimeError(f"required active Manager lock is not held: {path.name}")


def validate_manager_approval() -> Path:
    session_root_raw = os.environ.get("ARGUS_SKILL_SESSION_ROOT", "").strip()
    session_id = os.environ.get("ARGUS_SKILL_SESSION_ID", "").strip()
    require(session_root_raw and session_id, "active Argus session environment missing")
    session_root = Path(session_root_raw).expanduser().resolve()
    require(session_root.name == session_id, "Argus session root/id mismatch")
    inbox = session_root / "inbox.jsonl"
    require(inbox.is_file(), "active Manager inbox missing")
    matches = []
    for raw in inbox.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if item.get("ts") == INBOX_TS:
            matches.append((raw, item))
    require(len(matches) == 1, "exact Manager approval entry not unique")
    raw, item = matches[0]
    require(hashlib.sha256(raw.encode("utf-8")).hexdigest() == INBOX_LINE_SHA256,
            "Manager approval entry hash changed")
    text = str(item.get("text") or "")
    require(CONTRACT in text and PROPOSAL_SHA256 in text, "Manager approval contract/hash mismatch")
    require("明确授权并立即推进" in text, "operator selection missing")
    require("Both WikiText-2 and C4 must strictly improve" in text, "paired-smoke stop rule missing")
    return inbox


def atomic_commit(payloads: dict[Path, bytes]) -> None:
    originals: dict[Path, bytes | None] = {}
    staged: dict[Path, Path] = {}
    order = [path for path in payloads if path != APPROVAL_REL] + [APPROVAL_REL]
    try:
        for relative in order:
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            originals[relative] = target.read_bytes() if target.exists() else None
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                             suffix=".tmp", delete=False) as handle:
                handle.write(payloads[relative])
                handle.flush()
                os.fsync(handle.fileno())
                staged[relative] = Path(handle.name)
        for relative in order:
            os.replace(staged[relative], ROOT / relative)
            staged.pop(relative, None)
    except Exception:
        for relative, content in originals.items():
            target = ROOT / relative
            if content is None:
                target.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.rollback.",
                                                 suffix=".tmp", delete=False) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                    rollback = Path(handle.name)
                os.replace(rollback, target)
        raise
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)


def main() -> None:
    inbox = validate_manager_approval()
    require(load_json("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "Manager-owned current stage is not rtl")
    require(sha256(ROOT / "design/NUMERICAL_REPLACEMENT_PROPOSAL.md") == PROPOSAL_SHA256,
            "active architecture proposal hash changed")
    require(not (ROOT / APPROVAL_REL).exists(), "approval is already bound")

    session_root = Path(os.environ["ARGUS_SKILL_SESSION_ROOT"]).expanduser().resolve()
    require_external_lock(session_root / ".manager_session.lock")
    require_external_lock(session_root / ".manager_pipeline.lock")
    with file_lock(ROOT / ".manager_session.lock"):
        transaction_id = f"absolute-rope-approval-{uuid.uuid4().hex}"
        binding = approval_binding(transaction_id, inbox)
        payloads = build_payloads(binding)
        atomic_commit(payloads)

    require(load_json("design/FAST_LOOP_POLICY.json")["active_repair_authorization"]
            ["implementation_authorized"] is True, "authorization commit did not stick")
    require(load_json(APPROVAL_REL).get("transaction_id") == transaction_id,
            "approval commit marker mismatch")
    require(load_json("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "pipeline stage changed unexpectedly")
    print(
        "ACE2_ABSOLUTE_ROPE_OPERATOR_APPROVAL_BOUND "
        f"contract={CONTRACT} approved_at={APPROVED_AT} transaction_id={transaction_id} stage=rtl"
    )


if __name__ == "__main__":
    main()
