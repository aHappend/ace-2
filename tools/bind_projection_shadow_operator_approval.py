#!/usr/bin/env python3
"""Bind the live operator approval for the projection-shadow RTL attempt.

The script records authority already present in the active Argus Manager inbox.
It never edits the Manager-owned pipeline stage.  Project authorization views
are committed atomically, with the approval artifact written last as the commit
marker.
"""

from __future__ import annotations

import copy
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
CONTRACT = "layer0_projection_shadow_staged_attention_v1"
PROPOSAL_SHA256 = "bb686fdb1a8e886ba80e8578db25a5db39eb58a757fc053986fc4538d6c36d92"
APPROVED_AT = "2026-08-01T10:01:44.695971Z"
INBOX_TS = 1785578504.695971
INBOX_LINE_SHA256 = "aeccc3b92e6e8e8597f4abc97fd0e6002447e0bf71c5efea7e0e06516ca063fb"
APPROVAL_REL = Path("evidence/authorization") / CONTRACT / "operator_approval.json"
SCOPE = (
    "one_bounded_non_stage_closing_projection_shadow_rtl_implementation_"
    "then_focused_checks_and_unique_wikitext2_c4_paired_smoke"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(relative: str | Path) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {relative}")
    return value


def encode_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = copy.deepcopy(value)
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    return sha256_bytes(
        (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def validate_manager_approval() -> Path:
    session_root_raw = os.environ.get("ARGUS_SKILL_SESSION_ROOT", "").strip()
    session_id = os.environ.get("ARGUS_SKILL_SESSION_ID", "").strip()
    require(session_root_raw and session_id, "active Argus session environment missing")
    session_root = Path(session_root_raw).expanduser().resolve()
    require(session_root.name == session_id, "Argus session root/id mismatch")
    inbox = session_root / "inbox.jsonl"
    require(inbox.is_file(), "active Manager inbox missing")
    matches: list[tuple[str, dict[str, Any]]] = []
    for raw in inbox.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if item.get("ts") == INBOX_TS:
            matches.append((raw, item))
    require(len(matches) == 1, "exact Manager approval entry is not unique")
    raw, item = matches[0]
    require(sha256_bytes(raw.encode("utf-8")) == INBOX_LINE_SHA256,
            "Manager approval entry hash changed")
    text = str(item.get("text") or "")
    for fragment in (
        "FRESH EXPLICIT OPERATOR AUTHORIZATION",
        CONTRACT,
        "set implementation_authorized=true",
        "run focused checks",
        "unique WikiText-2/C4 paired smoke",
        "Do not fabricate stage transitions or edit PIPELINE_STATE outside Manager ownership",
    ):
        require(fragment in text, f"Manager approval is missing: {fragment}")
    return inbox


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
        "scope": SCOPE,
        "stage_closing": False,
        "stop_rule": (
            "both_wikitext2_and_c4_ratios_must_strictly_improve_or_seal_bounded_no_go"
        ),
        "thresholds": {
            "wikitext2_ratio_strictly_below": 13549.939049967887,
            "c4_en_512_ratio_strictly_below": 4477.990517308544,
        },
        "transaction_id": transaction_id,
        "forbidden": [
            "architecture_contract_change",
            "historical_frontier_replacement",
            "pipeline_stage_edit_by_planner",
            "full_shell_before_paired_smoke_pass",
            "canonical_or_noncanonical_ppa_before_paired_smoke_pass",
            "official_14_item_evaluation",
            "prototype_full_benchmark_or_signoff",
            "continuation_after_paired_smoke_failure",
        ],
        "manager_inbox_path_privacy": "resolved_from_ARGUS_SKILL_SESSION_ROOT_not_published",
        "manager_inbox_sha256_at_binding": sha256(inbox),
    }


def authorized_record(status: str, binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "authority": "operator",
        "approved_at_utc": APPROVED_AT,
        "contract_id": CONTRACT,
        "implementation_authorized": True,
        "implementation_approval": binding,
        "proposal_sha256": PROPOSAL_SHA256,
        "required_manager_action": "none_current_stage_remains_rtl",
        "required_operator_action": "satisfied_for_exactly_one_bounded_attempt",
        "stage_closing": False,
        "status": status,
    }


def artifact(relative: str, payloads: dict[Path, bytes]) -> dict[str, Any]:
    rel = Path(relative)
    if rel in payloads:
        data = payloads[rel]
    else:
        data = (ROOT / rel).read_bytes()
    return {"bytes": len(data), "path": relative, "sha256": sha256_bytes(data)}


def build_payloads(binding: dict[str, Any]) -> dict[Path, bytes]:
    payloads: dict[Path, bytes] = {}

    policy = load_json("design/FAST_LOOP_POLICY.json")
    policy["active_repair_authorization"] = {
        **authorized_record("operator_approved_bounded_rtl_attempt_not_started", binding),
        "approval_evidence": binding,
        "execution_status": "authorized_not_started",
        "task_count": 1,
        "updated_at_utc": APPROVED_AT,
        "predecessor_bounded_no_go": {
            "contract_id": "layer0_absolute_rope_online_attention_v1",
            "evidence": "evidence/layer0_absolute_rope_online_attention_v1/latest/BOUNDED_NO_GO.json",
            "evidence_sha256": "ea985c5e11f38a432adef93ed31b4039b1c03cfff470e169e8aa3b21973747a4",
        },
    }
    policy["architecture_proposal_authorization"] = {
        **authorized_record("independently_reviewed_environment_complete_operator_approved", binding),
        "approval_scope": SCOPE,
        "earliest_measured_divergence": "model.layers.0.score",
        "excluded_directions": [
            "static_per_head_qk_scale_metadata",
            "fused_qk_basis_rotation",
            "dynamic_post_rope_per_head_requantization",
            "fixed_q7_rope_to_score_transport",
            "relative_rope_score_fusion_or_global_valid_key_centering",
            "absolute_rope_online_attention_v1",
        ],
        "task_count": 1,
        "updated_at_utc": APPROVED_AT,
    }
    policy["selected_replacement_contract"] = {
        **authorized_record("operator_approved_implementation_not_started", binding),
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "execution_result": {
            "rtl_change_started": False,
            "focused_software_rtl_tests_run": False,
            "paired_smoke_run": False,
            "full_shell_regression_run": False,
            "canonical_sky130_ppa_run": False,
            "official_14_item_evaluation_run": False,
        },
        "updated_at_utc": APPROVED_AT,
    }
    payloads[Path("design/FAST_LOOP_POLICY.json")] = encode_json(policy)

    target = load_json("design/TARGET.json")
    target["generated_at_utc"] = APPROVED_AT
    target["current_architecture_contract"] = {
        "contract_id": CONTRACT,
        "proposal": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "proposal_sha256": PROPOSAL_SHA256,
        "implementation_authorized": True,
        "implementation_approval": binding,
        "stage_closing": False,
    }
    fast = target["fast_loop_contract"]
    fast["manager_recommendation"] = "remain_in_rtl_execute_one_operator_approved_bounded_attempt"
    fast["active_repair_authorization"] = {
        **authorized_record("operator_approved_bounded_rtl_attempt_not_started", binding),
        "approval_scope": SCOPE,
        "first_gate": "focused_software_and_rtl_tests_then_unique_paired_smoke",
        "official_14_item_evaluation": "forbidden",
    }
    fast["architecture_proposal_authorization"] = {
        **authorized_record("operator_approved_bounded_rtl_attempt_not_started", binding),
        "approval_scope": SCOPE,
        "first_gate": "focused_software_and_rtl_tests_then_unique_paired_smoke",
        "official_14_item_evaluation": "forbidden",
    }
    payloads[Path("design/TARGET.json")] = encode_json(target)

    chip = load_json("design/CHIP_SCOPE.json")
    override = chip["authority_override"]
    require(override.get("active_replacement_contract") == CONTRACT,
            "CHIP_SCOPE active contract mismatch")
    override.update({
        "architecture_review_status": "independently_accepted_environment_complete",
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
        "required_next_action": "execute_one_bounded_rtl_attempt_in_current_rtl_stage",
    })
    chip["numerical_behavior"]["rope"]["projection_shadow_staged_attention_contract"] = {
        "contract_id": CONTRACT,
        "architecture_review_status": "independently_accepted_environment_complete",
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
        "projection_shadow_format": "signed_q15_16_in_s32",
        "score_format": "materialized_signed_q6_9_in_s16",
        "status": "operator_approved_bounded_rtl_attempt_not_started",
    }
    chip["operator_owned_execution_policy"]["active_successor_contract"] = {
        **authorized_record("operator_approved_bounded_rtl_attempt_not_started", binding),
        "approval_scope": SCOPE,
        "task_count": 1,
        "first_gate": "focused_software_and_rtl_tests_then_unique_paired_smoke",
        "observed_result": "not_run_operator_approval_bound",
    }
    chip["stage"]["current_stage"] = "rtl"
    chip["stage"]["current_stage_status"] = "operator_approved_bounded_rtl_attempt_not_started"
    payloads[Path("design/CHIP_SCOPE.json")] = encode_json(chip)

    memory = load_json("design/MEMORY_MODEL.json")
    require(memory["authority_override"].get("active_replacement_contract") == CONTRACT,
            "MEMORY_MODEL active contract mismatch")
    memory["authority_override"].update({
        "architecture_review_status": "independently_accepted_environment_complete",
        "implementation_authorized": True,
        "operator_implementation_approval": binding,
        "required_next_action": "execute_one_bounded_rtl_attempt_in_current_rtl_stage",
    })
    memory["stage"] = "rtl"
    memory["claim_status"] = "operator_authorized_rtl_implementation_not_yet_tested_or_accepted"
    payloads[Path("design/MEMORY_MODEL.json")] = encode_json(memory)

    oracle = load_json("reference/ORACLE_MANIFEST.json")
    oracle["proposed_architecture_contract"].update({
        **authorized_record("operator_approved_focused_vectors_required", binding),
        "architecture_review_status": "independently_accepted_environment_complete",
        "existing_vectors_status": "historical_vectors_plus_new_contract_vectors_pending_generation",
    })
    oracle["generated_at_utc"] = APPROVED_AT
    oracle["claim_boundary"] = (
        "Reference, vector, and bounded RTL work is operator-authorized for the exact active "
        "contract. No accepted capability, PPA, stage closure, prototype, benchmark, signoff, "
        "tapeout, or silicon claim exists."
    )
    payloads[Path("reference/ORACLE_MANIFEST.json")] = encode_json(oracle)

    manifest = load_json("design/RTL_MANIFEST.json")
    manifest["proposed_replacement_contract"] = {
        **authorized_record("operator_approved_implementation_not_started", binding),
        "source": "design/NUMERICAL_REPLACEMENT_PROPOSAL.md",
        "implementation_result": {
            "rtl_change_started": False,
            "accepted_rtl_change": False,
            "accepted_verification_evidence": False,
            "accepted_ppa_evidence": False,
        },
    }
    manifest["generated_at_utc"] = APPROVED_AT
    manifest["current_stage"] = "rtl"
    manifest["architecture_contract_status"] = (
        "layer0_projection_shadow_staged_attention_v1_independently_reviewed_"
        "environment_complete_operator_approved_implementation_not_started"
    )
    manifest["traceability"].update({
        "selected_mechanism": CONTRACT,
        "architecture_contract_gap": {
            "status": "implementation_authorized_not_yet_implemented",
            "resolution_owner": "Planner executes the single bounded current-stage attempt",
        },
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    })
    manifest["traceability"]["stage_checklist"] = {
        "rtl.contract-traceability": False,
        "rtl.hardware-discipline": True,
        "rtl.ip-provenance": True,
    }
    payloads[Path("design/RTL_MANIFEST.json")] = encode_json(manifest)

    mission_path = Path("MISSION.md")
    mission = (ROOT / mission_path).read_text(encoding="utf-8")
    old = (
        "It moves the layer-0 Q/K quantization boundary to signed Q15.16 projection\n"
        "shadows and restores conventional absolute RoPE, materialized Q6.9 score,\n"
        "softmax, and attention-value stages. It is pending independent L2 architecture\n"
        "review and fresh explicit operator approval. `implementation_authorized=false`;\n"
        "no reference, RTL, verification-harness, quality, or EDA run is authorized.\n"
    )
    new = (
        "It moves the layer-0 Q/K quantization boundary to signed Q15.16 projection\n"
        "shadows and restores conventional absolute RoPE, materialized Q6.9 score,\n"
        "softmax, and attention-value stages. Independent architecture and environment\n"
        "review are complete. At 2026-08-01T10:01:44.695971Z the operator explicitly\n"
        "authorized exactly one bounded implementation, focused checks, and the unique\n"
        "WikiText-2/C4 paired smoke. `implementation_authorized=true`; failure or equality\n"
        "on either dataset stops before full-shell regression or PPA.\n"
    )
    require(old in mission, "MISSION active authorization paragraph changed")
    payloads[mission_path] = mission.replace(old, new).encode("utf-8")

    trace_path = Path("design/RTL_TRACEABILITY.md")
    trace = (ROOT / trace_path).read_text(encoding="utf-8")
    preface = f"""# ACE-2 architecture/RTL traceability notes

## Current architecture proposal

- Contract: `{CONTRACT}`.
- Proposal: `design/NUMERICAL_REPLACEMENT_PROPOSAL.md`, SHA-256
  `{PROPOSAL_SHA256}`.
- Status: architecture and environment reviews complete; one exact bounded RTL
  implementation is operator-authorized and not yet started.
- Stage closing: false.
- RTL modified for this contract: no.
- Implementation authorized: yes, exactly one bounded attempt.

The contract moves layer-0 Q/K from the lossy signed-int8 boundary to checked
signed-Q15.16 projection shadows, then restores absolute split-half RoPE,
materialized Q6.9 scores, the accepted staged softmax, and accepted staged
attention-value arithmetic. The accepted prefix and historical PPA frontier are
unchanged until focused checks and the paired-smoke gate succeed.

## Preserved frontier
"""
    marker = "## Preserved frontier\n"
    require(marker in trace, "RTL traceability frontier marker missing")
    trace = preface + trace.split(marker, 1)[1]
    payloads[trace_path] = trace.encode("utf-8")

    checkpoint_path = Path("CHECKPOINT.md")
    checkpoint = f"""# Goal

Execute the operator-authorized bounded RTL attempt for
`{CONTRACT}` while preserving the Manager-owned `rtl` stage, immutable targets,
accepted prefix, and historical PPA frontier.

# Current State

- Independent architecture and environment review are complete.
- The Manager-owned stage remains `rtl`; this task does not edit
  `research/PIPELINE_STATE.json`.
- The active proposal SHA-256 is `{PROPOSAL_SHA256}`.
- Fresh explicit operator approval was received at `{APPROVED_AT}` and is bound
  to `evidence/authorization/{CONTRACT}/operator_approval.json`.
- Exactly one bounded implementation, focused software/RTL checks, and the
  unique WikiText-2/C4 paired smoke are authorized.
- Full-shell regression, PPA, official evaluation, prototype, full benchmark,
  and signoff remain gated until the paired smoke passes.

# Immutable Boundaries

- Accepted prefix: through `layer_0.v_proj`; first unsupported: `layer_0.rope_q`.
- Mode: `ADVANCE`.
- Historical frontier: 62,199 cells, 0.6108746272 mm2 non-SRAM, +0.1502 ns at
  100 MHz.
- Targets: at most 2.0 mm2 non-SRAM and at least 100 MHz; quality limit 1.05x.
- Failure or equality on either paired-smoke dataset seals a bounded no-go and
  forbids continuation of this direction.

# Next Action

Implement the frozen projection-shadow arithmetic and focused RTL, run the
approved focused checks, bind source hashes, then execute the unique paired
smoke exactly once.
"""
    payloads[checkpoint_path] = checkpoint.encode("utf-8")

    payloads[APPROVAL_REL] = encode_json(binding)

    status = load_json("research/PUBLIC_STATUS.json")
    for key in ("architecture_proposal_gate", "selected_replacement_contract"):
        status[key].update({
            "implementation_authorized": True,
            "implementation_approval": binding,
            "required_manager_action": "none_current_stage_remains_rtl",
            "required_operator_action": "satisfied_for_exactly_one_bounded_attempt",
            "status": "rtl_entry_complete_operator_approved_implementation_not_started",
        })
    dashboard = status["dashboard_fields"]
    dashboard["candidate_mechanism"].update({
        "implementation_authorized": True,
        "implementation_approval": binding,
        "required_manager_action": "none_current_stage_remains_rtl",
        "required_operator_action": "satisfied_for_exactly_one_bounded_attempt",
        "status": "operator_approved_bounded_rtl_attempt_not_started",
    })
    dashboard["operator_policy"]["active_repair_authorization"] = policy["active_repair_authorization"]
    dashboard["operator_policy"]["selected_replacement_contract"] = policy["selected_replacement_contract"]
    dashboard["latest_decision"] = "operator_approved_projection_shadow_bounded_rtl_attempt_not_started"
    dashboard["routing_status"] = "rtl_operator_approved_bounded_attempt"
    frontier = status["implementation_frontier"]
    frontier.update({
        "latest_decision": "operator_approved_projection_shadow_bounded_rtl_attempt_not_started",
        "required_manager_action": "none_current_stage_remains_rtl",
        "required_operator_action": "satisfied_for_exactly_one_bounded_attempt",
        "routing_status": "rtl_operator_approved_bounded_attempt",
        "rtl_contract_traceability": False,
    })
    status["latest_decision"] = "operator_approved_projection_shadow_bounded_rtl_attempt_not_started"
    status["stage"]["current_stage_status"] = "operator_approved_bounded_rtl_attempt_not_started"
    status["generated_at_utc"] = APPROVED_AT
    status["last_updated_utc"] = APPROVED_AT

    changed_paths = {
        *[path.as_posix() for path in payloads],
        "tools/bind_projection_shadow_operator_approval.py",
    }
    prior_paths = {
        str(item.get("path"))
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and item.get("path")
    }
    paths = sorted(prior_paths | changed_paths)
    status["artifact_hashes"] = [
        artifact(path, payloads) for path in paths
        if path != "research/PUBLIC_STATUS.json" and (Path(path) in payloads or (ROOT / path).is_file())
    ]
    status["integrity"] = {
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, "
            "with integrity.canonical_sha256 set to null"
        ),
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    payloads[Path("research/PUBLIC_STATUS.json")] = encode_json(status)
    return payloads


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_commit(payloads: dict[Path, bytes]) -> None:
    originals: dict[Path, bytes | None] = {}
    staged: dict[Path, Path] = {}
    order = [path for path in payloads if path != APPROVAL_REL] + [APPROVAL_REL]
    try:
        for relative in order:
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            originals[relative] = target.read_bytes() if target.exists() else None
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
            ) as handle:
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
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.rollback.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
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
    review = load_json(
        "evidence/review/environment_stage_closing_layer0_projection_shadow_staged_attention_v1/decision.json"
    )
    require(review.get("contract_id") == CONTRACT, "environment review contract mismatch")
    require(review.get("decision", {}).get("status") == "done", "environment review is not done")
    require(not (ROOT / APPROVAL_REL).exists(), "operator approval is already bound")

    transaction_id = f"projection-shadow-approval-{uuid.uuid4().hex}"
    binding = approval_binding(transaction_id, inbox)
    with file_lock(ROOT / ".manager_session.lock"):
        atomic_commit(build_payloads(binding))

    require(load_json(APPROVAL_REL).get("transaction_id") == transaction_id,
            "approval commit marker mismatch")
    require(load_json("design/FAST_LOOP_POLICY.json")["active_repair_authorization"]
            ["implementation_authorized"] is True, "authorization commit did not stick")
    require(load_json("research/PIPELINE_STATE.json").get("current_stage") == "rtl",
            "pipeline stage changed unexpectedly")
    print(
        "ACE2_PROJECTION_SHADOW_OPERATOR_APPROVAL_BOUND "
        f"contract={CONTRACT} approved_at={APPROVED_AT} "
        f"transaction_id={transaction_id} stage=rtl"
    )


if __name__ == "__main__":
    main()
