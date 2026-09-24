#!/usr/bin/env python3
"""Bind retained DPRF discriminator evidence without executing the workload again."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from run_down_projection_residual_fusion_verification import check_replays


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "shared_down_projection_residual_fusion_v1"
CANDIDATE_ID = "down_projection_residual_fusion_76ef0dda2e646558"
CANDIDATE_HASH = "76ef0dda2e646558ecb3e2047d3ec00d69f416cbe0a880b3402876614a4a7a28"
RTL_SHA256 = "b3f33cb503aa89f4f7c54b0fb4ffcc814061b28917bf911b35ad335fb2425bdf"
PREFIX = [
    "layer_0.input_rmsnorm",
    "layer_0.q_proj",
    "layer_0.k_proj",
    "layer_0.v_proj",
]
FIRST_UNSUPPORTED = "layer_0.rope_q"

FIRST_RAW = ROOT / "verification/raw/archive/pre-dprf-42fa63d925d813c8"
SECOND_RAW = ROOT / "verification/raw/latest"
FIRST_BASELINE = FIRST_RAW / "dprf_focused_baseline_replay/results.json"
FIRST_CANDIDATE = FIRST_RAW / "dprf_focused_candidate_replay/results.json"
SECOND_BASELINE = SECOND_RAW / "dprf_focused_baseline_replay/results.json"
SECOND_CANDIDATE = SECOND_RAW / "dprf_focused_candidate_replay/results.json"
EXPECTED_HASHES = {
    FIRST_BASELINE: "899925521c6734dafab1cc4c0fa175523e33c1fb84f95d2d020e565f3fc045b8",
    FIRST_CANDIDATE: "ecc6e52bab249d744c0afccdef0facd2485ff3ba7e19bfb2ec3ee41fe8f554be",
    SECOND_BASELINE: "33d45fc719b8067f0dd176a0fe4a3998c777085a5165332891038db4d4026a8c",
    SECOND_CANDIDATE: "c3942c6d51aa6c1382293fdb2f84f65bfa7a16962e3030fbe33b2ffa48789fcd",
}

LATEST = ROOT / "evidence" / CONTRACT / "latest"
DECISION = LATEST / "VERIFICATION_DECISION.json"
PRECHECK = LATEST / "PRECHECK.json"
RTL_REVIEW = ROOT / "evidence/review/rtl_checklist_shared_down_projection_residual_fusion_v1/decision.json"
OLD_L2 = ROOT / "evidence/review/focused_verification_shared_down_projection_residual_fusion_v1/decision.json"
RESULTS = ROOT / "verification/RESULTS.json"
ORACLE_MANIFEST = ROOT / "reference/ORACLE_MANIFEST.json"
PROPERTY_MANIFEST = ROOT / "formal/ACE2_VERIFICATION_PROPERTIES.json"
METADATA = ROOT / "reference/generated/down_projection_residual_fusion_metadata.json"
PUBLIC = ROOT / "research/PUBLIC_STATUS.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PLAN = ROOT / "verification/PLAN.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
ARCHIVE = ROOT / "evidence" / CONTRACT / "archive" / "duplicate_execution_integrity_rebind"

COMMAND_IDS = [
    "dprf_vector_generation",
    "dprf_vector_regeneration",
    "dprf_metadata_check",
    "dprf_hook_self_test",
    "dprf_reference_tests",
    "dprf_rtl_elaboration",
    "dprf_rtl_simulation",
    "dprf_stage_stress_elaboration",
    "dprf_stage_stress_simulation",
    "dprf_verilator_lint",
    "dprf_bounded_formal",
    "dprf_focused_baseline_replay",
    "dprf_focused_candidate_replay",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    clone = copy.deepcopy(value)
    clone.setdefault("integrity", {})["canonical_sha256"] = None
    encoded = (json.dumps(clone, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evidence_utc() -> str:
    value = os.environ.get("ACE2_EVIDENCE_UTC")
    require(value is not None, "ACE2_EVIDENCE_UTC must bind the evidence date")
    require(
        len(value) == 20
        and value[4] == "-"
        and value[7] == "-"
        and value[10] == "T"
        and value.endswith("Z"),
        "ACE2_EVIDENCE_UTC must use YYYY-MM-DDTHH:MM:SSZ",
    )
    require(value <= "2026-08-02T23:59:59Z", "evidence timestamp is after the current date")
    return value


def verify_sha256s(result_path: Path) -> None:
    manifest = result_path.parent / "SHA256SUMS"
    require(manifest.is_file(), f"missing SHA256SUMS beside {result_path}")
    seen = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        require(separator == "  ", f"malformed SHA256SUMS line: {line}")
        path = result_path.parent / relative
        require(path.is_file(), f"missing replay artifact: {path}")
        require(sha256(path) == digest, f"SHA256SUMS mismatch: {path}")
        seen = seen or path == result_path
    require(seen, f"results.json absent from {manifest}")


def strip_execution_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_execution_time(item)
            for key, item in value.items()
            if key != "generated_at_utc"
        }
    if isinstance(value, list):
        return [strip_execution_time(item) for item in value]
    return value


def archive_file(path: Path, category: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    digest = sha256(path)
    destination = ARCHIVE / f"{category}.{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
    return artifact(destination)


def trace_capture(result: dict[str, Any], metadata: dict[str, Any], dataset: str) -> dict[str, Any]:
    passed = result["down_projection_residual_fusion_contract"]["passes"][dataset]
    require(passed["ordered_layer_indices"] == list(range(24)), f"{dataset} order differs")
    require(passed["layers_executed"] == 24, f"{dataset} layer count differs")
    require(passed["all_lane_independent_oracle_match"] is True, f"{dataset} oracle mismatch")
    require(passed["sticky_numeric_overflow"] is False, f"{dataset} overflowed")
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(passed["layers"]):
        require(layer["layer_index"] == index, f"{dataset} layer row differs")
        trace = layer["independent_trace"]
        saturation = {
            "positive_saturations": trace["positive_saturations"],
            "negative_saturations": trace["negative_saturations"],
        }
        overflow = {
            "layer_execution_completed_without_overflow_exception": True,
            "pass_sticky_numeric_overflow": passed["sticky_numeric_overflow"],
        }
        row = {
            "layer_index": index,
            "input_descended_from_prior_fusion": layer["input_descended_from_prior_fusion"],
            "accumulator_s32_sha256": layer["accumulator_s32_sha256"],
            "residual_s8_sha256": layer["residual_s8_sha256"],
            "metadata_sha256": value_sha256(metadata["layers"][index]),
            "output_s8_sha256": layer["isolated_or_composed_fused_s8_sha256"],
            "saturation_observation": saturation,
            "saturation_observation_sha256": value_sha256(saturation),
            "overflow_observation": overflow,
            "overflow_observation_sha256": value_sha256(overflow),
        }
        row["input_tuple_sha256"] = value_sha256(
            {
                "accumulator_s32_sha256": row["accumulator_s32_sha256"],
                "residual_s8_sha256": row["residual_s8_sha256"],
                "metadata_sha256": row["metadata_sha256"],
            }
        )
        row["ordered_row_sha256"] = value_sha256(row)
        rows.append(row)
    boundaries = result["fixed_boundary_hashes"][dataset]
    require("model.norm" in boundaries and "lm_head" in boundaries, f"{dataset} final captures missing")
    return {
        "ordered_trace_sha256": passed["ordered_trace_sha256"],
        "ordered_rows": rows,
        "final_rmsnorm_sha256": boundaries["model.norm"],
        "final_lm_head_sha256": boundaries["lm_head"],
        "positive_saturations": passed["positive_saturations"],
        "negative_saturations": passed["negative_saturations"],
        "sticky_numeric_overflow": passed["sticky_numeric_overflow"],
    }


def command_records(raw_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for command_id in COMMAND_IDS:
        log = raw_root / f"{command_id}.log"
        require(log.is_file(), f"missing retained first-run log: {log}")
        lines = log.read_text(encoding="utf-8").splitlines()
        require(len(lines) >= 3 and lines[0].startswith("command="), f"malformed command log: {log}")
        require(lines[1] == "exit_status=0", f"retained command was not successful: {log}")
        require(lines[2].startswith("wall_seconds="), f"missing wall time: {log}")
        records.append(
            {
                "id": command_id,
                "command": shlex.split(lines[0].removeprefix("command=")),
                "exit_status": 0,
                "wall_seconds": float(lines[2].removeprefix("wall_seconds=")),
                "raw_log": artifact(log),
                "retention_note": "Executed while this directory was verification/raw/latest, then moved intact into the content-addressed archive.",
            }
        )
    return records


def update_public(now: str, decision_record: dict[str, Any]) -> None:
    public = load(PUBLIC)
    checklist = {
        "verification.independent-oracle": False,
        "verification.coverage-stress": True,
        "verification.reproducible-green": True,
    }
    status = "integrity_no_go_duplicate_discriminator_execution_review_pending"
    latest = "down_projection_residual_fusion_integrity_no_go_review_pending"
    public.update(
        {
            "generated_at_utc": now,
            "last_updated_utc": now,
            "latest_decision": latest,
            "routing_status": status,
            "required_operator_action": "none",
            "required_manager_action": "hold_verification_until_independent_l2_integrity_verdict",
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "current_mode": "ADVANCE",
            "verification_checklist": checklist,
            "stage_closing": False,
        }
    )
    stage = public.setdefault("stage", {})
    stage.update(
        {
            "current_stage": "verification",
            "current_stage_status": status,
            "current_stage_checklist": checklist,
            "current_stage_evidence": [
                "verification/PLAN.md",
                "verification/RESULTS.json",
                decision_record["path"],
            ],
            "stage_closing": False,
            "planner_may_advance_stage": False,
            "stage_transition_owner": "Manager",
        }
    )
    latest_verification = {
        "status": status,
        "contract_id": CONTRACT,
        "candidate_id": CANDIDATE_ID,
        "candidate_rtl_hash": CANDIDATE_HASH,
        "technical_disposition": "bounded_no_go",
        "execution_integrity": "failed_duplicate_successful_execution",
        "checklist": checklist,
        "decision": decision_record,
        "results": "verification/RESULTS.json",
        "independent_l2_review_requirement": "required_before_manager_routing",
        "paired_smoke_run": False,
        "ppa_run": False,
    }
    public["latest_verification_stage"] = latest_verification
    dashboard = public.setdefault("dashboard_fields", {})
    dashboard.update(
        {
            "current_stage": "verification",
            "current_mode": "ADVANCE",
            "latest_decision": latest,
            "candidate_meets_numeric_acceptance": False,
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "verification_checklist": checklist,
            "latest_verification_stage": latest_verification,
        }
    )
    dashboard.setdefault("candidate_mechanism", {}).update(
        {
            "contract_id": CONTRACT,
            "exact_scale32_dataset_discriminator_status": status,
            "implementation_authorized": False,
            "status": status,
        }
    )
    dashboard.setdefault("latest_rtl_candidate", {}).update(
        {
            "contract_id": CONTRACT,
            "candidate_id": CANDIDATE_ID,
            "candidate_rtl_hash": CANDIDATE_HASH,
            "status": status,
            "required_manager_action": "hold_verification_until_independent_l2_integrity_verdict",
            "stage_closing": False,
        }
    )
    public.setdefault("implementation_frontier", {}).update(
        {
            "current_stage": "verification",
            "current_mode": "ADVANCE",
            "latest_decision": latest,
            "supported_layer_operator_prefix": PREFIX,
            "ordered_supported_layer_operator_prefix": PREFIX,
            "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
            "verification_checklist": checklist,
            "latest_verification_stage": latest_verification,
        }
    )
    public.setdefault("integrity", {})["canonical_sha256"] = None
    public["integrity"]["canonical_sha256"] = canonical_sha256(public)
    dump(PUBLIC, public)


def main() -> int:
    now = evidence_utc()
    require(load(PIPELINE).get("current_stage") == "verification", "Manager-owned stage is not verification")
    require(sha256(ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv") == RTL_SHA256, "live RTL changed")
    precheck = load(PRECHECK)
    rtl_review = load(RTL_REVIEW)
    require(precheck.get("candidate_id") == CANDIDATE_ID, "precheck candidate differs")
    require(precheck.get("candidate_rtl_hash") == CANDIDATE_HASH, "precheck RTL hash differs")
    require(rtl_review.get("reviewer_status") == "done", "RTL review is not complete")
    require(rtl_review.get("stage_closing") is True, "RTL review is not stage-closing")

    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"missing retained replay: {path}")
        require(sha256(path) == expected, f"retained replay hash differs: {path}")
        verify_sha256s(path)

    first_baseline = load(FIRST_BASELINE)
    first_candidate = load(FIRST_CANDIDATE)
    second_baseline = load(SECOND_BASELINE)
    second_candidate = load(SECOND_CANDIDATE)
    require(first_baseline["generated_at_utc"] == "2026-08-02T00:14:50Z", "first baseline execution time differs")
    require(first_candidate["generated_at_utc"] == "2026-08-02T00:14:50Z", "first candidate execution time differs")
    require(second_baseline["generated_at_utc"] == "2026-08-02T00:17:24Z", "second baseline execution time differs")
    require(second_candidate["generated_at_utc"] == "2026-08-02T00:17:24Z", "second candidate execution time differs")
    require(strip_execution_time(first_baseline) == strip_execution_time(second_baseline), "baseline rerun payload differs")
    require(strip_execution_time(first_candidate) == strip_execution_time(second_candidate), "candidate rerun payload differs")

    quality_gate = check_replays(first_baseline, first_candidate)
    require(quality_gate["passed"] is False, "retained technical discriminator unexpectedly passed")
    metadata = load(METADATA)
    require(metadata.get("contract_id") == CONTRACT and len(metadata.get("layers", [])) == 24, "metadata binding differs")
    traces = {
        dataset: {
            "isolated_baseline_replay": trace_capture(first_baseline, metadata, dataset),
            "composed_recurrent_candidate": trace_capture(first_candidate, metadata, dataset),
        }
        for dataset in ("wikitext2", "c4_en_512")
    }

    superseded = {
        "verification_decision": archive_file(DECISION, "VERIFICATION_DECISION"),
        "verification_results": archive_file(RESULTS, "RESULTS"),
        "self_attested_l2_verdict": archive_file(OLD_L2, "SELF_ATTESTED_L2_DECISION"),
    }
    if OLD_L2.exists():
        OLD_L2.unlink()

    source_paths = [
        ROOT / "rtl/ace2_down_projection_residual_fusion_core.sv",
        ROOT / "tools/ace2_down_projection_residual_fusion_hook.py",
        ROOT / "tools/ace2_down_projection_residual_fusion_reference.py",
        ROOT / "tools/localize_score_to_lm_head.py",
        ROOT / "tools/run_down_projection_residual_fusion_verification.py",
        Path(__file__).resolve(),
        METADATA,
    ]
    decision: dict[str, Any] = {
        "schema_version": 2,
        "bound_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": CANDIDATE_ID,
        "candidate_rtl_hash": CANDIDATE_HASH,
        "status": "integrity_no_go",
        "decision": "integrity_no_go_duplicate_successful_discriminator_execution",
        "stage": "verification",
        "stage_closing": False,
        "candidate_verified": False,
        "candidate_capability_accepted": False,
        "quality_discriminator_complete": True,
        "technical_disposition": {
            "status": "bounded_no_go",
            "quality_gate": quality_gate,
            "authoritative_retained_execution": "first_successful_execution",
        },
        "execution_contract": {
            "required_successful_execution_count": 1,
            "observed_successful_execution_count": 2,
            "satisfied": False,
            "failure": "duplicate_successful_execution",
            "rerun_permitted": False,
            "first_successful_execution": {
                "generated_at_utc": "2026-08-02T00:14:50Z",
                "raw_root": FIRST_RAW.relative_to(ROOT).as_posix(),
                "baseline_replay": artifact(FIRST_BASELINE),
                "candidate_replay": artifact(FIRST_CANDIDATE),
            },
            "later_duplicate_execution": {
                "generated_at_utc": "2026-08-02T00:17:24Z",
                "raw_root": SECOND_RAW.relative_to(ROOT).as_posix(),
                "baseline_replay": artifact(SECOND_BASELINE),
                "candidate_replay": artifact(SECOND_CANDIDATE),
                "payload_equivalent_except_generated_at_utc": True,
            },
        },
        "ordered_trace_captures": traces,
        "trace_hash_encoding": "sha256(canonical-json-compact-sort-keys)",
        "ordered_supported_layer_operator_prefix": PREFIX,
        "first_unsupported_layer_operator": FIRST_UNSUPPORTED,
        "mode": "ADVANCE",
        "required_manager_action": "hold_verification_for_independent_l2_integrity_verdict",
        "required_operator_action": "none",
        "claim_boundary": [
            "The first retained execution is authoritative only for the technical bounded NO_GO evidence.",
            "The exact-one execution acceptance contract failed irreversibly because a second successful execution occurred.",
            "No third discriminator run and no PPA, prototype, benchmark, signoff, tapeout, or silicon run was performed by this rebinder.",
        ],
        "precheck": artifact(PRECHECK),
        "rtl_review": artifact(RTL_REVIEW),
        "source_hashes": [artifact(path) for path in source_paths],
        "superseded_artifacts": superseded,
        "independent_l2_review": "pending",
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    decision["integrity"]["canonical_sha256"] = canonical_sha256(decision)
    dump(DECISION, decision)
    decision_record = artifact(DECISION)

    oracle_manifest = {
        "schema_version": 4,
        "bound_at_utc": now,
        "contract_id": CONTRACT,
        "candidate_id": CANDIDATE_ID,
        "candidate_rtl_hash": CANDIDATE_HASH,
        "claim_boundary": "First retained technical execution only; duplicate-execution integrity NO_GO remains binding.",
        "independent_reference": artifact(ROOT / "tools/ace2_down_projection_residual_fusion_reference.py"),
        "exact_hook": artifact(ROOT / "tools/ace2_down_projection_residual_fusion_hook.py"),
        "model_image_metadata": {
            "manifest": artifact(METADATA),
            "binary": artifact(ROOT / "reference/generated/down_projection_residual_fusion_metadata.bin"),
            "quality_metrics_executed": True,
            "record_count": 21504,
        },
        "standalone_vectors": {
            "json": artifact(ROOT / "verification/generated/down_projection_residual_fusion_vectors.json"),
            "svh": artifact(ROOT / "verification/generated/down_projection_residual_fusion_vectors.svh"),
        },
        "focused_replays": {
            "baseline": artifact(FIRST_BASELINE),
            "candidate": artifact(FIRST_CANDIDATE),
            "all_lane_independent_oracle_match": True,
            "layers": 24,
            "datasets": ["wikitext2", "c4_en_512"],
            "capture_token_limit": 128,
        },
        "execution_integrity": decision["execution_contract"],
    }
    dump(ORACLE_MANIFEST, oracle_manifest)

    property_manifest = load(PROPERTY_MANIFEST)
    property_manifest.update(
        {
            "schema_version": 4,
            "bound_at_utc": now,
            "generated_at_utc": "2026-08-02T00:14:50Z",
            "formal_evidence": artifact(FIRST_RAW / "dprf_bounded_formal.log"),
            "execution_integrity": "failed_duplicate_successful_discriminator_execution",
        }
    )
    dump(PROPERTY_MANIFEST, property_manifest)

    prior_results = load(ARCHIVE / f"RESULTS.{superseded['verification_results']['sha256']}.json")
    results: dict[str, Any] = {
        "schema_version": 4,
        "bound_at_utc": now,
        "technical_execution_generated_at_utc": "2026-08-02T00:14:50Z",
        "stage": "verification",
        "pipeline_current_stage": "verification",
        "manager_stage_transition_owner": "Manager",
        "contract_id": CONTRACT,
        "candidate_id": CANDIDATE_ID,
        "candidate_rtl_hash": CANDIDATE_HASH,
        "status": "integrity_no_go_review_pending",
        "technical_status": "bounded_no_go",
        "stage_closing": False,
        "candidate_accepted": False,
        "checklist": {
            "verification.independent-oracle": False,
            "verification.coverage-stress": True,
            "verification.reproducible-green": True,
        },
        "checklist_explanation": {
            "verification.independent-oracle": "Standalone and all-lane oracle checks pass, but frozen quality constraints fail, so this checklist item remains false.",
            "verification.coverage-stress": "The first retained execution contains the completed reset, boundary, stall/backpressure, illegal Scale32, randomized, X/Z, saturation, overflow, formal, and representative-dataset evidence.",
            "verification.reproducible-green": "All first-execution simulator, formal, and replay commands exited successfully and their retained logs and hashes are present; exact-one process integrity is reported separately and failed.",
        },
        "quality_gate": quality_gate,
        "execution_integrity": decision["execution_contract"],
        "coverage": prior_results["coverage"],
        "commands": command_records(FIRST_RAW),
        "decision": decision_record,
        "oracle_manifest": artifact(ORACLE_MANIFEST),
        "property_manifest": artifact(PROPERTY_MANIFEST),
        "source_hashes": decision["source_hashes"],
        "required_manager_action": "hold_verification_until_independent_l2_integrity_verdict",
        "claim_boundary": decision["claim_boundary"],
        "integrity": {"algorithm": "sha256-canonical-json-v1", "canonical_sha256": None},
    }
    results["integrity"]["canonical_sha256"] = canonical_sha256(results)
    dump(RESULTS, results)

    PLAN.write_text(
        """# ACE-2 down-projection residual-fusion verification plan

This plan is limited to the active `verification` stage and candidate
`shared_down_projection_residual_fusion_v1` at RTL hash
`76ef0dda2e646558ecb3e2047d3ec00d69f416cbe0a880b3402876614a4a7a28`.
It does not run or claim PPA, prototype, benchmark, signoff, tapeout, or silicon evidence.

## Acceptance gates

- Independent oracle: bit-exact standalone arithmetic, all-lane isolated replay,
  composed ordered execution at all 24 layers, exact layer-0 pre-injection equality,
  and frozen quality constraints.
- Coverage stress: reset, clear, stalls/backpressure, illegal Scale32 records,
  randomized and boundary arithmetic, saturation, X/Z, workspace/latency bounds,
  single-clock CDC disposition, and both representative datasets.
- Reproducibility: retained raw commands, logs, result hashes, and execution history
  must be mutually consistent. The exact-one discriminator contract is binding.

## Current result

Technical status: `bounded_no_go`. Execution-integrity status:
`duplicate_successful_execution`; exact-one contract: `failed`. Stage closing: `false`.
The first successful execution is retained as technical evidence. A third execution is prohibited.
Independent L2 integrity review is pending.
""",
        encoding="utf-8",
    )
    CHECKPOINT.write_text(
        f"""# Goal

Close the bounded verification disposition for `shared_down_projection_residual_fusion_v1`
without entering downstream stages or concealing the duplicate discriminator execution.

# Current State

The Manager-owned stage remains `verification`. Candidate
`{CANDIDATE_ID}` remains bound to RTL hash `{CANDIDATE_HASH}`.
The first retained discriminator execution at `2026-08-02T00:14:50Z` is now the
authoritative technical evidence. No discriminator rerun was performed by this repair.

# Result

Status: `integrity_no_go_review_pending`. Stage closing: `false`.
The retained technical result is `bounded_no_go` with eight frozen quality failures.
The exact-one execution contract is unsatisfied because a second successful execution
occurred at `2026-08-02T00:17:24Z`; both executions are hash-bound and payload-equivalent
except for their generation timestamps. Independent L2 integrity review remains pending.

# Boundaries

The accepted prefix remains through `layer_0.v_proj`; first unsupported remains
`layer_0.rope_q`. The 2.0 mm2 non-SRAM cap and 100 MHz floor are unchanged.
No third discriminator, paired smoke, shell admission, PPA, prototype, benchmark,
signoff, tapeout, or silicon run was performed. Only the Manager may change `current_stage`.
""",
        encoding="utf-8",
    )
    update_public(now, decision_record)
    print(
        "ACE2_DPRF_AUTHORITATIVE_REBIND_PASS "
        f"decision_sha256={sha256(DECISION)} "
        "technical_status=bounded_no_go execution_integrity=duplicate_successful_execution"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
