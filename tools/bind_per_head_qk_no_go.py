#!/usr/bin/env python3
"""Seal the bounded per-head Q/K repair as a fail-closed RTL no-go."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "per_head_qk_repair" / "latest"
BASELINE = ROOT / "benchmark/raw/quality/operator-paired-smoke-20260731T081721Z/results.json"
CANDIDATE = ROOT / "benchmark/raw/quality/per-head-qk-candidate-smoke-128/results.json"
CANDIDATE_RUN_DIR = CANDIDATE.parent
CANDIDATE_RUN_CONTRACT = CANDIDATE_RUN_DIR / "run_contract.json"
CANDIDATE_DERIVED_SCALES = CANDIDATE_RUN_DIR / "derived_scales.json"
CANDIDATE_PACKET_LIVE = EVIDENCE / "candidate_evidence.json"
CANDIDATE_PACKET_ARCHIVE = (
    ROOT
    / "evidence/per_head_qk_repair/archive/"
    "candidate_evidence.1a9873129a41c702be6124748ffd29704904010f3084c5460d00d61c63124a5e.json"
)
RTL_MANIFEST = ROOT / "design/RTL_MANIFEST.json"
RTL_TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
PUBLIC_STATUS = ROOT / "research/PUBLIC_STATUS.json"
FAST_LOOP_POLICY = ROOT / "design/FAST_LOOP_POLICY.json"
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
RESULTS = EVIDENCE / "RESULTS.json"
ABORTED_FULL_SHELL = EVIDENCE / "aborted_pre_gate_full_shell_20260731.log"

NUMERICAL_RTL = [
    "rtl/ace2_pkg.sv",
    "rtl/ace2_rmsnorm_core.sv",
    "rtl/ace2_w4a8_proj_core.sv",
    "rtl/ace2_rope_core.sv",
    "rtl/ace2_attention_score_core.sv",
    "rtl/ace2_softmax_core.sv",
    "rtl/ace2_attention_compose_core.sv",
    "rtl/ace2_silu_gate_core.sv",
    "rtl/generated/ace2_silu_lut.svh",
    "rtl/ace2_shell.sv",
]
FOCUSED_LOGS = {
    "fixed_point_self_test": EVIDENCE / "fixed_point_self_test.log",
    "rtl_lint": ROOT / "evidence/frontier/latest/rtl_lint.log",
    "rtl_rope": ROOT / "evidence/frontier/latest/rtl_rope_sim.log",
    "rtl_attention_score": ROOT / "evidence/frontier/latest/rtl_attention_score_sim.log",
    "rtl_attention_score_shell": ROOT / "evidence/frontier/latest/rtl_attention_score_shell_sim.log",
}
PASS_MARKERS = {
    "fixed_point_self_test": "ACE2_FULL_MODEL_FIXED_POINT_SELF_TEST status=pass",
    "rtl_lint": "ACE2_RTL_LINT_PASS",
    "rtl_rope": "ACE2_ROPE_TB_PASS cases=5 beats_per_case=56",
    "rtl_attention_score": "ACE2_ATTN_SCORE_TB_PASS cases=4 context_max=8",
    "rtl_attention_score_shell": (
        "ACE2_SHELL_ATTN_SCORE_TB_PASS cases=4 unequal_scale_case=1 "
        "invalid_metadata_cases=6"
    ),
}
CURRENT_PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
CURRENT_CAPABILITIES = [
    "input_rmsnorm",
    "w4a8_projection_q_proj",
    "w4a8_projection_k_proj",
    "w4a8_projection_v_proj",
]
SMOKE_MATERIAL_SOURCES = {
    "benchmark/quality/PROMPT_MANIFEST.json",
    "benchmark/quality/QUALITY_CONFIG.json",
    "tools/ace2_full_model_fixed_point.py",
    "tools/ace2_quality_contracts.py",
    *NUMERICAL_RTL,
}
ALLOWED_HISTORICAL_DRIFT = {
    "evidence/per_head_qk_repair/latest/candidate_evidence.json": (
        "The mutable latest pointer was regenerated after the smoke. Its exact "
        "pre-smoke content is retained in the immutable archive bound below."
    ),
    "design/RTL_MANIFEST.json": (
        "The live manifest was intentionally rewritten by the no-go binder after "
        "the smoke. Its pre-smoke hash was a gate guard, not a numerical input; "
        "the exact historical manifest content was not retained."
    ),
    "tools/ace2_attention_score_reference.py": (
        "This reference is imported only by the runner's explicit self-test path, "
        "not by smoke execution; it changed after the retained smoke completed."
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(value))
    canonical.setdefault("integrity", {})["canonical_sha256"] = None
    payload = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_records() -> list[dict[str, Any]]:
    return [artifact(ROOT / relative) for relative in NUMERICAL_RTL]


def recorded_artifacts(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.append(value)
        for child in value.values():
            records.extend(recorded_artifacts(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(recorded_artifacts(child))
    return records


def verify_smoke_provenance(candidate: dict[str, Any]) -> dict[str, Any]:
    run_contract = load(CANDIDATE_RUN_CONTRACT)
    accepted_rtl = candidate["artifacts"]["accepted_rtl"]
    recorded_binding = accepted_rtl["binding"]
    require(
        sha256_file(CANDIDATE_PACKET_ARCHIVE) == recorded_binding["sha256"],
        "archived pre-smoke candidate packet hash differs",
    )
    require(
        CANDIDATE_PACKET_ARCHIVE.stat().st_size == recorded_binding["bytes"],
        "archived pre-smoke candidate packet size differs",
    )
    archived_packet = load(CANDIDATE_PACKET_ARCHIVE)
    require(
        archived_packet["candidate_id"] == run_contract["candidate"]["candidate_id"],
        "archived candidate ID differs from the run contract",
    )
    require(
        archived_packet["source_binding"]["ordered_source_hash_list_sha256"]
        == run_contract["candidate"]["ordered_source_hash_list_sha256"],
        "archived candidate RTL hash differs from the run contract",
    )
    require(
        sha256_file(CANDIDATE_RUN_CONTRACT)
        == candidate["artifacts"]["run_contract_sha256"],
        "candidate run contract hash differs",
    )
    require(
        sha256_file(CANDIDATE_DERIVED_SCALES)
        == candidate["artifacts"]["derived_scales_sha256"],
        "candidate derived-scale hash differs",
    )

    source_index = {
        item["path"]: item
        for item in candidate["artifacts"]["sources"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    material: list[dict[str, Any]] = []
    for relative in sorted(SMOKE_MATERIAL_SOURCES):
        require(relative in source_index, f"candidate omits material source {relative}")
        expected = source_index[relative]
        path = ROOT / relative
        require(path.is_file(), f"material smoke source is missing: {relative}")
        observed = artifact(path)
        require(
            observed["sha256"] == expected["sha256"]
            and observed["bytes"] == expected["bytes"],
            f"material smoke source drifted: {relative}",
        )
        material.append(observed)

    drift_by_path: dict[str, dict[str, Any]] = {}
    for expected in recorded_artifacts(candidate.get("artifacts", {})):
        relative = expected["path"]
        path = ROOT / relative
        if not path.is_file():
            drift_by_path[relative] = {
                "path": relative,
                "recorded_sha256": expected["sha256"],
                "recorded_bytes": expected.get("bytes"),
                "current_status": "missing",
            }
            continue
        observed_sha256 = sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_sha256 != expected["sha256"] or (
            "bytes" in expected and observed_bytes != expected["bytes"]
        ):
            drift_by_path[relative] = {
                "path": relative,
                "recorded_sha256": expected["sha256"],
                "recorded_bytes": expected.get("bytes"),
                "current_sha256": observed_sha256,
                "current_bytes": observed_bytes,
                "current_status": "drifted",
            }
    unexpected = sorted(set(drift_by_path) - set(ALLOWED_HISTORICAL_DRIFT))
    require(not unexpected, f"unexpected candidate evidence drift: {unexpected}")
    for relative, record in drift_by_path.items():
        record["classification"] = (
            "archived_exact_binding"
            if relative == "evidence/per_head_qk_repair/latest/candidate_evidence.json"
            else "historical_non_numerical_dependency"
        )
        record["reason"] = ALLOWED_HISTORICAL_DRIFT[relative]

    recorded_manifest = accepted_rtl["manifest"]
    return {
        "status": "material_smoke_sources_current_with_declared_historical_drift",
        "run_contract": artifact(CANDIDATE_RUN_CONTRACT),
        "derived_scales": artifact(CANDIDATE_DERIVED_SCALES),
        "archived_candidate_packet": artifact(CANDIDATE_PACKET_ARCHIVE),
        "material_source_count": len(material),
        "material_sources": material,
        "historical_drift": [drift_by_path[path] for path in sorted(drift_by_path)],
        "historical_manifest_guard": {
            "path": recorded_manifest["path"],
            "sha256": recorded_manifest["sha256"],
            "bytes": recorded_manifest["bytes"],
            "exact_snapshot_retained": False,
            "role": "premeasurement_gate_guard_not_numerical_input",
        },
        "reproducibility_limitations": [
            "The exact pre-smoke RTL_MANIFEST.json bytes were not retained; its "
            "recorded hash remains historical gate evidence only.",
            "The numerical result remains bound to the exact runner, quality "
            "contracts, numerical RTL, run contract, derived scales, model revision, "
            "input observations, seeds, and runtime package versions.",
        ],
    }


def render_traceability(rtl_hash: str, ratios: dict[str, Any]) -> str:
    return f"""# ACE-2 RTL traceability notes

This is the current `rtl`-stage record for the operator-authorized per-head Q/K
repair. The candidate is synthesizable and its focused checks pass, but it is
**rejected** because the frozen paired-smoke improvement gate failed. It is not
a stage-close packet and it does not authorize verification, PPA, prototype,
benchmark, signoff, tapeout, or silicon claims.

## Current contract frontier

- Ordered supported prefix: `{"`, `".join(CURRENT_PREFIX)}`.
- First unsupported operator: `layer_0.rope_q`.
- Candidate RTL hash: `{rtl_hash}`.
- Candidate decision: `bounded_no_go_per_head_qk_smoke_gate_failed`.
- Comparable smoke ratios: WikiText-2 baseline
  `{ratios['wikitext2']['baseline']:.12f}`, candidate
  `{ratios['wikitext2']['candidate']:.12f}`; C4-en baseline
  `{ratios['c4_en_512']['baseline']:.12f}`, candidate
  `{ratios['c4_en_512']['candidate']:.12f}`. Both required strict improvements;
  neither improved.

## Synthesizable source traceability

| Source/module | Frozen requirement | Current implementation/evidence |
| --- | --- | --- |
| `rtl/ace2_pkg.sv` | `design/SPEC.md` opcode, width, error, model, and memory-boundary constants | Shared first-party package; lint/elaboration input. |
| `rtl/ace2_rmsnorm_core.sv` | Accepted `input_rmsnorm` numerical contract | Unchanged first-party synthesizable core; retained current supported prefix. |
| `rtl/ace2_w4a8_proj_core.sv` | Accepted W4A8 Q/K/V projection arithmetic | Unchanged shared 4-lane first-party projection core; per-head Q/K output metadata remains external per-output requantization data. |
| `rtl/ace2_rope_core.sv` | `design/SPEC.md` signed-int8 RoPE with positive signed-Q6.9 conversion and ties-to-even saturation | Shared two-lane core receives one validated head conversion broadcast by the shell. Five generated oracle cases pass. |
| `rtl/ace2_attention_score_core.sv` | `design/SPEC.md` signed-int32 dot, per-query-head multiplier/u6 shift, signed-64 intermediate, ties-to-even Q6.9 result | Standalone first-party unit remains focused-testable; four generated oracle cases pass. |
| `rtl/ace2_softmax_core.sv` | Frozen fixed-point softmax contract | Retained first-party downstream RTL; not newly accepted under the failed candidate. |
| `rtl/ace2_attention_compose_core.sv` | Frozen cross-tile compose contract | Retained first-party downstream RTL; not newly accepted under the failed candidate. |
| `rtl/ace2_silu_gate_core.sv` and `rtl/generated/ace2_silu_lut.svh` | Frozen SiLU gate contract and generated LUT configuration | Retained first-party RTL; LUT provenance and regeneration remain recorded in `design/RTL_MANIFEST.json`. |
| `rtl/ace2_shell.sv` (`ace2_shell`) | Frozen CSR, direct command, DMA, reset/error, 16-record attention-head table, `cmd_aux_i[15:0]`, layer/address cache tag, and fail-closed metadata validation | Adds the exact 16-record table loader, query/KV ID checks, positive conversion/multiplier checks, reserved-bit checks, Q/K RoPE head selection, and ATTN_SCORE `aux[3:0]` selection. Focused shell test passes four selected heads and six malformed-record classes. |
| `rtl/ace2_shell.sv` (`ace2_attention_accumulator`) | Shared signed-int32 score/value accumulator lifetime | First-party synthesizable helper retained with explicit reset/fault/watchdog priority. |
| `rtl/ace2_shell.sv` (`ace2_state_shadow`) | Timing-safe committed-state handoff | First-party synthesizable helper retained; no CDC is introduced. |

All design RTL is first-party. No third-party RTL or generated IP is included.
Generated verification sources are regenerated by the commands recorded in
`design/RTL_MANIFEST.json`. The only generated RTL source is
`rtl/generated/ace2_silu_lut.svh`, with its existing pinned generator,
reference, configuration, and content hash.

## Fail-closed outcome

The candidate fixed-point model implements one static Q scale for each of 14
query heads, one static K scale for each of two KV heads, range-safe RoPE
conversion, 7:1 query-to-KV mapping, and per-query-head score metadata. The
128-token candidate run uses the exact baseline input observations, model
revision, and seeds. Because both ratios are worse, the operator-owned gate
forbids the full-shell regression and canonical SKY130 PPA. The earlier PPA
frontier remains historical pre-per-head evidence only. A mistakenly started
pre-gate full-shell attempt was aborted and retained as provenance; it produced
no completed regression result and is not acceptance evidence.

The exact pre-smoke candidate packet is retained under
`evidence/per_head_qk_repair/archive/`. The live manifest was intentionally
rewritten after the smoke and its exact pre-smoke bytes were not retained, so
that historical manifest hash is only a gate-guard record. The numerical no-go
remains bound to the exact runner, quality contracts, numerical RTL, run
contract, derived scales, model revision, inputs, seeds, and package versions.
"""


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    pipeline = load(PIPELINE_STATE)
    require(pipeline["current_stage"] == "rtl", "Manager-owned stage is not rtl")
    for name, path in FOCUSED_LOGS.items():
        require(path.is_file(), f"missing focused log: {path}")
        require(PASS_MARKERS[name] in path.read_text(encoding="utf-8"), f"focused check failed: {name}")
    require(ABORTED_FULL_SHELL.is_file(), "missing aborted pre-gate full-shell provenance")

    baseline = load(BASELINE)
    candidate = load(CANDIDATE)
    require(baseline["input_observations"] == candidate["input_observations"], "smoke inputs differ")
    require(baseline["model"] == candidate["model"], "model revisions differ")
    require(baseline["seeds"] == candidate["seeds"], "smoke seeds differ")
    prepared = load(CANDIDATE_PACKET_ARCHIVE)
    rtl_hash = prepared["source_binding"]["ordered_source_hash_list_sha256"]
    require(rtl_hash != "6388ec6ae7085bd7933419a6378f7ae03131f0fcac32babba5f93bbbba0b0be2", "RTL hash did not change")
    require(candidate["artifacts"]["accepted_rtl"]["candidate_rtl_hash"] == rtl_hash, "smoke is not bound to candidate RTL")
    smoke_provenance = verify_smoke_provenance(candidate)

    ratios: dict[str, Any] = {}
    for dataset in ("wikitext2", "c4_en_512"):
        baseline_ratio = float(baseline["metrics"][dataset]["ratio"])
        candidate_ratio = float(candidate["metrics"][dataset]["ratio"])
        ratios[dataset] = {
            "baseline": baseline_ratio,
            "candidate": candidate_ratio,
            "delta": candidate_ratio - baseline_ratio,
            "strictly_lower": candidate_ratio < baseline_ratio,
        }
    gate_passed = all(value["strictly_lower"] for value in ratios.values())
    require(not gate_passed, "no-go binder cannot seal a passing candidate")

    timestamp = utc_now()
    result = {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "status": "bounded_no_go_per_head_qk_smoke_gate_failed",
        "stage": "rtl",
        "stage_closing": False,
        "candidate_rtl_hash": rtl_hash,
        "accepted_prefix_unchanged": CURRENT_PREFIX,
        "first_unsupported_layer_operator": "layer_0.rope_q",
        "focused_checks": {name: artifact(path) for name, path in FOCUSED_LOGS.items()},
        "smoke_gate": {
            "pass_condition": "strictly_lower_ratio_on_both_datasets",
            "passed": False,
            "ratios": ratios,
            "same_input_observations": True,
            "same_model_revision": True,
            "same_seeds": True,
            "baseline": artifact(BASELINE),
            "candidate": artifact(CANDIDATE),
        },
        "smoke_provenance": smoke_provenance,
        "expensive_runs": {
            "full_shell_regression_run": False,
            "full_shell_attempt_started": True,
            "full_shell_attempt_completed": False,
            "aborted_full_shell_attempt": artifact(ABORTED_FULL_SHELL),
            "canonical_sky130_ppa_run": False,
            "reason": "forbidden_by_failed_operator_owned_smoke_gate",
        },
        "rtl_checklist_evidence": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "immutable_targets": {
            "quality_ratio_max": 1.05,
            "area_cap_non_sram_mm2": 2.0,
            "frequency_floor_mhz": 100.0,
            "targets_relaxed": False,
            "quality_target_met": False,
        },
        "required_next_action": "independent_review_of_the_bounded_no_go_then_manager_reroute_to_a_structurally_different_operator_authorized_numerical_mechanism",
        "claim_boundaries": [
            "This is a bounded negative result, not RTL stage closure or project completion.",
            "A pre-gate full-shell attempt was aborted; no completed regression result is claimed.",
            "No candidate full-shell, synthesis, STA, power, PPA, prototype, benchmark, signoff, tapeout, or silicon result is claimed.",
            "The exact pre-smoke RTL manifest bytes were not retained; the recorded manifest hash is historical gate evidence, not a reproduced numerical input.",
        ],
    }
    write(RESULTS, result)

    records = source_records()
    manifest = load(RTL_MANIFEST)
    candidate_evidence_hashes = {
        "candidate_packet_sha256": sha256_file(CANDIDATE_PACKET_ARCHIVE),
        "candidate_source_hash_list_sha256": sha256_file(
            EVIDENCE / "source_hashes.before"
        ),
        "per_head_qk_no_go_sha256": sha256_file(RESULTS),
        "baseline_smoke_sha256": sha256_file(BASELINE),
        "candidate_smoke_sha256": sha256_file(CANDIDATE),
        "aborted_pre_gate_full_shell_sha256": sha256_file(ABORTED_FULL_SHELL),
        **{
            f"{name}_sha256": sha256_file(path)
            for name, path in FOCUSED_LOGS.items()
        },
    }
    manifest.update(
        {
            "generated_at_utc": timestamp,
            "stage": "rtl",
            "stage_closing": False,
            "candidate_status": "rejected_bounded_no_go_smoke_gate_failed",
            "candidate_layer_operator": "per_head_qk_scale_contract_repair",
            "candidate_meets_numeric_acceptance": False,
            "candidate_requires_fresh_sky130_ppa": False,
            "candidate_rtl_hash": rtl_hash,
            "rtl_hash": rtl_hash,
            "candidate_source_hashes": records,
            "source_hashes": records,
            "accepted_frontier_item_count": len(CURRENT_PREFIX),
            "supported_layer_operator_prefix": CURRENT_PREFIX,
            "candidate_supported_layer_operator_prefix_after_review": CURRENT_PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "candidate_first_unsupported_layer_operator_after_review": "layer_0.rope_q",
            "supported_reusable_operator_capabilities": CURRENT_CAPABILITIES,
            "candidate_reusable_operator_capabilities_after_review": CURRENT_CAPABILITIES,
            "candidate_verification_complete": False,
            "candidate_verification_binding": RESULTS.relative_to(ROOT).as_posix(),
            "candidate_evidence_hashes": candidate_evidence_hashes,
            "candidate_review_binding": {
                "decision": "pending",
                "level": None,
                "review_status": "bounded_no_go_requires_independent_review",
                "reviewed_candidate_packet_sha256": None,
                "result_binding": RESULTS.relative_to(ROOT).as_posix(),
            },
            "independent_reviewer_acceptance": False,
            "independent_reviewer_verdict": "pending_no_go_review",
            "ppa_evidence_binding": None,
            "synthesized_modules": [],
            "interfaces": {
                **manifest.get("interfaces", {}),
                "command_dispatch": "direct descriptor ingress with cmd_aux_i[15:0]; ATTN_SCORE requires aux[3:0] query-head ID 0..13 and aux[15:4] zero",
                "attention_head_metadata": "sixteen 128-bit records at aligned scale_addr, cached by layer_id and full 64-bit address; records 0..13 query heads and 14..15 KV heads",
            },
            "parameters": {
                **manifest.get("parameters", {}),
                "ATTENTION_QUERY_HEADS": 14,
                "ATTENTION_KV_HEADS": 2,
                "ATTENTION_HEAD_METADATA_RECORDS": 16,
                "ATTENTION_HEAD_METADATA_BYTES": 256,
            },
            "traceability": [
                {
                    "requirement": "design/SPEC.md per-head Q/K metadata table and command aux semantics",
                    "implementation": "rtl/ace2_shell.sv validated 16-record cache plus head-indexed RoPE/score selection",
                    "verification": "focused RoPE, attention-score core, and shell tests; candidate rejected by comparable paired smoke",
                },
                {
                    "requirement": "design/SPEC.md positive Q6.9 conversion and positive signed-int32 score multiplier",
                    "implementation": "fail-closed conversion, multiplier, ID, mapping, shift, and reserved-bit checks before payload reads",
                    "verification": "six malformed metadata classes produce numeric error and zero output writes",
                },
                {
                    "requirement": "operator-owned paired-smoke improvement gate",
                    "implementation": "tools/ace2_full_model_fixed_point.py static 14-Q/2-K head calibration and 7:1 score metadata",
                    "verification": "identical 128-token inputs/model/seeds; both ratios worsened, so full shell and PPA were not run",
                },
            ],
            "latest_evidence": {
                **manifest.get("latest_evidence", {}),
                "per_head_qk_no_go": RESULTS.relative_to(ROOT).as_posix(),
                "candidate_smoke": CANDIDATE.relative_to(ROOT).as_posix(),
                "rtl_lint_log": FOCUSED_LOGS["rtl_lint"].relative_to(ROOT).as_posix(),
                "rtl_rope_sim_log": FOCUSED_LOGS["rtl_rope"].relative_to(ROOT).as_posix(),
                "rtl_attention_score_sim_log": FOCUSED_LOGS["rtl_attention_score"].relative_to(ROOT).as_posix(),
                "rtl_attention_score_shell_sim_log": FOCUSED_LOGS["rtl_attention_score_shell"].relative_to(ROOT).as_posix(),
                "aborted_pre_gate_full_shell_attempt": ABORTED_FULL_SHELL.relative_to(ROOT).as_posix(),
                "archived_pre_smoke_candidate_packet": CANDIDATE_PACKET_ARCHIVE.relative_to(ROOT).as_posix(),
            },
            "claim_boundaries": result["claim_boundaries"],
        }
    )
    write(RTL_MANIFEST, manifest)
    RTL_TRACEABILITY.write_text(render_traceability(rtl_hash, ratios), encoding="utf-8")

    policy = load(FAST_LOOP_POLICY)
    authorization = policy["active_repair_authorization"]
    authorization["execution_status"] = "completed_bounded_no_go"
    authorization["observed_candidate_rtl_hash"] = rtl_hash
    authorization["observed_smoke_ratios"] = ratios
    authorization["smoke_gate_passed"] = False
    authorization["full_shell_regression_run"] = False
    authorization["canonical_sky130_ppa_run"] = False
    authorization["result_binding"] = RESULTS.relative_to(ROOT).as_posix()
    write(FAST_LOOP_POLICY, policy)

    status = load(PUBLIC_STATUS)
    latest_ppa = status.get("implementation_frontier", {}).get("latest_ppa_frontier")
    if latest_ppa is not None:
        latest_ppa["contract_binding_status"] = "historical_pre_per_head_scale_contract"
        latest_ppa["current_architecture_first_unsupported"] = "layer_0.rope_q"
    frontier = status.setdefault("implementation_frontier", {})
    frontier.update(
        {
            "current_stage": "rtl",
            "current_mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": CURRENT_PREFIX,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "latest_decision": "stop_per_head_qk_direction_smoke_gate_failed_both_datasets",
            "latest_rtl_candidate": {
                "status": "bounded_no_go",
                "candidate_rtl_hash": rtl_hash,
                "smoke_ratios": ratios,
                "full_shell_regression_run": False,
                "canonical_sky130_ppa_run": False,
                "evidence": RESULTS.relative_to(ROOT).as_posix(),
            },
        }
    )
    status["stage"] = {
        "current_stage": "rtl",
        "current_stage_source": "research/PIPELINE_STATE.json",
        "completed_prior_stages": ["definition", "architecture", "environment"],
        "current_stage_status": "bounded_repair_no_go_requires_replan",
        "current_stage_checklist": {
            "rtl.contract-traceability": True,
            "rtl.hardware-discipline": True,
            "rtl.ip-provenance": True,
        },
        "current_stage_evidence": [
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            RESULTS.relative_to(ROOT).as_posix(),
        ],
        "stage_transition_owner": "Manager",
        "planner_may_advance_stage": False,
        "downstream_locked_until_manager_advance": [
            "verification", "ppa", "prototype", "benchmark", "signoff"
        ],
    }
    dashboard = status.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "current_stage": "rtl",
            "current_mode": "ADVANCE",
            "ordered_supported_layer_operator_prefix": CURRENT_PREFIX,
            "candidate_reusable_operator_capabilities_after_review": CURRENT_CAPABILITIES,
            "first_unsupported_layer_operator": "layer_0.rope_q",
            "latest_decision": "stop_per_head_qk_direction_smoke_gate_failed_both_datasets",
            "latest_rtl_candidate": frontier["latest_rtl_candidate"],
        }
    )
    if "latest_ppa_frontier" in dashboard:
        dashboard["latest_ppa_frontier"]["contract_binding_status"] = "historical_pre_per_head_scale_contract"
        dashboard["latest_ppa_frontier"]["current_architecture_first_unsupported"] = "layer_0.rope_q"
    status["blockers"] = [
        {
            "id": "per_head_qk_smoke_gate_failed",
            "stage": "rtl",
            "status": "active",
            "reason": "Both comparable 128-token W4A8/BF16 perplexity ratios worsened; the operator-owned gate forbids full-shell and PPA execution for this direction.",
            "evidence": RESULTS.relative_to(ROOT).as_posix(),
            "required_resolution": "Independent review of this bounded no-go, followed by Manager routing and operator authorization for any structurally different numerical contract; targets remain unchanged.",
        }
    ]
    status["generated_at_utc"] = timestamp
    status["last_updated_utc"] = timestamp
    public_artifacts = {
        item["path"]
        for item in status.get("artifact_hashes", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    public_artifacts.update(
        {
            "design/FAST_LOOP_POLICY.json",
            "design/RTL_MANIFEST.json",
            "design/RTL_TRACEABILITY.md",
            "research/PIPELINE_STATE.json",
            RESULTS.relative_to(ROOT).as_posix(),
            CANDIDATE.relative_to(ROOT).as_posix(),
            "tools/ace2_full_model_fixed_point.py",
            "tools/bind_per_head_qk_no_go.py",
            "tools/prepare_per_head_qk_candidate.py",
            CANDIDATE_PACKET_ARCHIVE.relative_to(ROOT).as_posix(),
            *(path.relative_to(ROOT).as_posix() for path in FOCUSED_LOGS.values()),
            ABORTED_FULL_SHELL.relative_to(ROOT).as_posix(),
        }
    )
    public_artifacts.discard(PUBLIC_STATUS.relative_to(ROOT).as_posix())
    public_artifacts.discard(CANDIDATE_PACKET_LIVE.relative_to(ROOT).as_posix())
    status["artifact_hashes"] = [
        artifact(ROOT / relative)
        for relative in sorted(public_artifacts)
        if (ROOT / relative).is_file()
    ]
    status["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": (
            "UTF-8, sorted keys, two-space indentation, trailing newline, with "
            "integrity.canonical_sha256 set to null"
        ),
        "canonical_sha256": None,
    }
    status["integrity"]["canonical_sha256"] = canonical_sha256(status)
    write(PUBLIC_STATUS, status)

    print(
        "ACE2_PER_HEAD_QK_NO_GO_BIND_PASS "
        f"rtl_hash={rtl_hash} wiki={ratios['wikitext2']['candidate']:.9f} "
        f"c4={ratios['c4_en_512']['candidate']:.9f}"
    )


if __name__ == "__main__":
    main()
