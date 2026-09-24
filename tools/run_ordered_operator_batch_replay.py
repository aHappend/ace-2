#!/usr/bin/env python3
"""Bind the current-tree ordered replay to its first executable boundary.

This verifier is intentionally model-free and simulation-free.  It consumes
the retained model-bound token context and stops before RTL execution when the
first ordered operator lacks the exact input/reference tensors required for a
bit-exact comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence/verification/rtl-current-tree-ordered-operator-batch-replay-v1"
REPORT = OUTPUT / "ordered_prefix_report.json"
PROBE = OUTPUT / "boundary_probe.json"
LOG = OUTPUT / "batch_replay.log"
SUMS = OUTPUT / "SHA256SUMS"

MISSION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-current-tree-ordered-operator-batch-replay-v1/mission.json"
)
DEPENDENCY_ROUND = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "manager-record-rope-q-certified-frontier-v1/round-0001.json"
)
ROPE_Q_DECISION = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "rtl-rope-q-current-tree-frontier-certification-v1/decision.json"
)

WORKLOAD = ROOT / "design/WORKLOAD.md"
RTL_MANIFEST = ROOT / "design/RTL_MANIFEST.json"
TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
CAPTURE = ROOT / "evidence/diagnostics/c4-rope-q-token1-capture-replay-20260803-v2"
WITNESS = CAPTURE / "witness.json"
CAPTURE_SUMS = CAPTURE / "SHA256SUMS"
NONEXECUTABLE_SUMMARY = (
    ROOT / "evidence/diagnostics/rmsnorm-numerical-bisect-fastpath-20260802-r2/results.json"
)
PPA = ROOT / "evidence/canonical_sky130/rtl-rmsnorm-default-shift-canonical-sky130-ppa-v1"
PPA_SUMS = PPA / "SHA256SUMS"

EXPECTED_TREE = "e10d78dd4f257f882c87b29b01bc97527212f7782189de25afdb2b62e850f8be"
EXPECTED_PPA_SUMS = "03251f9dffde265a5ac916f43c271728c68e941b25c1f63efc4a5b3bf72ebfc4"
EXPECTED_CAPTURE_SUMS = "9d1640be7cc126c689b53f2a0009069af0fdd02050cf4114659d0e8a000893ac"
EXPECTED_WITNESS = "2ad63d87d0fd53664e606faed32e60d5f5e69c94903f07f93de6ae1822657404"
EXPECTED_ORDER = [
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope_q",
    "rope_k",
    "kv_write",
    "attention_score",
    "softmax",
    "attention_value",
    "o_proj",
    "attention_residual_add",
    "post_attention_rmsnorm",
    "mlp_gate_proj",
    "mlp_up_proj",
    "silu_gate",
    "mlp_down_proj",
    "mlp_residual_add",
]
PRIOR_PREFIX = [f"layer_0.{name}" for name in EXPECTED_ORDER[:5]]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(path: Path, public_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": public_path or relative(path),
        "sha256": sha256_file(path),
    }


def validate_sha256_manifest(manifest: Path, base: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, name = line.split("  ", 1)
        candidate = Path(name)
        path = candidate if candidate.is_absolute() else base / candidate
        require(path.is_file(), f"manifest member missing: {name}")
        require(sha256_file(path) == expected, f"manifest member changed: {name}")
        count += 1
    return count


def rtl_tree_binding() -> dict[str, Any]:
    paths = sorted(ROOT.glob("rtl/**/*.sv"))
    rows = [(sha256_file(path), relative(path)) for path in paths]
    manifest = "".join(f"{digest}  {path}\n" for digest, path in rows).encode()
    tree_sha256 = sha256_bytes(manifest)
    require(len(rows) == 23, f"RTL file count differs: {len(rows)}")
    require(tree_sha256 == EXPECTED_TREE, f"RTL tree differs: {tree_sha256}")
    return {
        "file_count": len(rows),
        "hash_method": "sha256 of UTF-8 path-sorted '<file_sha256>  <path>\\n' manifest for rtl/**/*.sv",
        "sha256": tree_sha256,
        "source_hashes": [{"path": path, "sha256": digest} for digest, path in rows],
    }


def workload_order() -> list[str]:
    text = WORKLOAD.read_text(encoding="utf-8")
    order = re.findall(r"^\d+\. `([^`]+)`$", text, flags=re.MULTILINE)[:18]
    require(order == EXPECTED_ORDER, f"frozen operator order differs: {order}")
    return order


def exact_rope_k_witnesses() -> list[str]:
    matches: list[str] = []
    pattern = re.compile(r'"operator"\s*:\s*"rope_k"')
    for path in sorted((ROOT / "evidence").rglob("*.json")):
        if OUTPUT in path.parents:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if pattern.search(text):
            matches.append(relative(path))
    return matches


def build() -> tuple[dict[str, Any], dict[str, Any], str]:
    require(sha256_file(PPA_SUMS) == EXPECTED_PPA_SUMS, "canonical PPA aggregate changed")
    ppa_member_count = validate_sha256_manifest(PPA_SUMS, ROOT)
    require(sha256_file(CAPTURE_SUMS) == EXPECTED_CAPTURE_SUMS, "RoPE-Q capture aggregate changed")
    capture_member_count = validate_sha256_manifest(CAPTURE_SUMS, CAPTURE)
    require(sha256_file(WITNESS) == EXPECTED_WITNESS, "RoPE-Q witness changed")

    order = workload_order()
    tree = rtl_tree_binding()
    witness = json.loads(WITNESS.read_text(encoding="utf-8"))
    summary = json.loads(NONEXECUTABLE_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(RTL_MANIFEST.read_text(encoding="utf-8"))

    coordinate = witness.get("coordinate", {})
    require(coordinate.get("dataset_record_index") == 64, "retained record differs")
    require(coordinate.get("token") == 1, "retained token differs")
    require(coordinate.get("head") == 0, "retained head differs")
    require(coordinate.get("operator") == "rope_q", "retained witness is not RoPE-Q")
    require(manifest.get("first_unsupported_layer_operator") == "layer_0.rope_k", "published boundary differs")
    require("exactly through `layer_0.rope_q`" in TRACEABILITY.read_text(encoding="utf-8"), "traceability boundary differs")

    c4_input = summary["input_observations"]["c4_en_512"]
    eval_input = witness["dataset_provenance"]["evaluation_witness"]
    require(c4_input["record_sha256"] == eval_input["record_sha256"], "retained record hash differs")
    require(
        c4_input["tokenized"]["token_sequence_sha256"] == eval_input["token_sequence_sha256"],
        "retained token-sequence hash differs",
    )

    k_projection = summary["comparisons"]["c4_en_512"]["model.layers.0.k_projection"]
    k_post_rope = summary["comparisons"]["c4_en_512"]["model.layers.0.k_post_rope"]
    k_metadata = summary["fixed_metadata"]["c4_en_512"]
    witness_matches = exact_rope_k_witnesses()
    require(not witness_matches, f"an exact RoPE-K witness now exists: {witness_matches}")

    retained_context = {
        "coordinate_without_operator": {key: value for key, value in coordinate.items() if key != "operator"},
        "evaluation_witness": eval_input,
        "model": witness["model"],
        "input_rmsnorm_s8": witness["projection"]["input_s8"],
        "input_rmsnorm_scale": witness["projection"]["input_scale"],
        "cos_q15_s16": witness["rope"]["cos_q15_s16"],
        "sin_q15_s16": witness["rope"]["sin_q15_s16"],
    }

    missing = [
        "layer_0.rope_k.input.k_projection_output_s8[batch0,head0,token1,64]",
        "layer_0.rope_k.input.exact_k_rope_conversion_q9_or_equivalent_input_output_scale_metadata",
        "layer_0.rope_k.reference.expected_output_preclamp[64]",
        "layer_0.rope_k.reference.expected_output_s8[64]",
    ]

    attempted = {
        "operator": "layer_0.rope_k",
        "status": "UNSUPPORTED_MISSING_EXECUTABLE_REFERENCE_ARTIFACT",
        "retained_context": {
            "dataset_record_index": 64,
            "token": 1,
            "head": 0,
            "token_id": coordinate["token_id"],
            "record_sha256": eval_input["record_sha256"],
            "token_sequence_sha256": eval_input["token_sequence_sha256"],
            "context_sha256": canonical_sha256(retained_context),
            "source": artifact(WITNESS),
        },
        "input_hashes": {
            "available_input_rmsnorm_s8_sha256": canonical_sha256(witness["projection"]["input_s8"]),
            "available_cos_q15_s16_sha256": canonical_sha256(witness["rope"]["cos_q15_s16"]),
            "available_sin_q15_s16_sha256": canonical_sha256(witness["rope"]["sin_q15_s16"]),
            "required_k_projection_output_s8_sha256": None,
            "required_k_rope_conversion_metadata_sha256": None,
        },
        "reference_hashes": {
            "independent_projection_reference": artifact(ROOT / "tools/ace2_projection_reference.py"),
            "independent_rope_reference": artifact(ROOT / "tools/ace2_rope_reference.py"),
            "retained_nonexecutable_summary": artifact(NONEXECUTABLE_SUMMARY),
            "required_expected_tensor_sha256": None,
        },
        "rtl_source_hashes": {
            "rope_core": artifact(ROOT / "rtl/ace2_rope_core.sv"),
            "shell": artifact(ROOT / "rtl/ace2_shell.sv"),
            "maintained_shell_harness": artifact(ROOT / "verification/tb/ace2_shell_tb.sv"),
            "existing_model_bound_harness_q_only": artifact(ROOT / "verification/tb/ace2_c4_rope_q_replay_tb.sv"),
        },
        "counts": {
            "contract_output_count": 64,
            "rtl_output_count": 0,
            "compared_output_count": 0,
            "mismatch_count": None,
        },
        "saturation": {
            "retained_summary_key_rope_saturated_elements": k_metadata["key_rope_saturated_elements"],
            "rtl_observed_saturated_elements": None,
            "result": "NOT_EVALUATED_REFERENCE_MISSING",
        },
        "cycle_handshake_latency": {
            "simulation_executed": False,
            "cycles": None,
            "handshake_result": "NOT_EVALUATED_REFERENCE_MISSING",
            "latency_result": "NOT_EVALUATED_REFERENCE_MISSING",
        },
        "existing_current_tree_ppa": {
            "covers_unchanged_hardware": True,
            "rtl_tree_sha256": tree["sha256"],
            "canonical_packet_aggregate_sha256": sha256_file(PPA_SUMS),
            "packet_member_count": ppa_member_count,
            "rationale": "The live 23-file RTL tree exactly matches the already accepted canonical packet; no RTL or constraints were changed or rerun.",
        },
        "retained_summary_only": {
            "classification": summary["classification"],
            "k_projection_candidate_sha256": k_projection["candidate_sha256"],
            "k_projection_reference_sha256": k_projection["reference_sha256"],
            "k_projection_shape": k_projection["shape"],
            "k_post_rope_candidate_sha256": k_post_rope["candidate_sha256"],
            "k_post_rope_reference_sha256": k_post_rope["reference_sha256"],
            "k_post_rope_shape": k_post_rope["shape"],
            "limitation": "Hash/statistical summaries are not executable tensors and cannot supply the 64 retained-coordinate input and expected lanes.",
        },
        "missing_artifacts": missing,
        "narrowest_reproducible_divergence": "At C4 validation record 64, layer 0, token 1, KV head 0, the retained package supplies the RMSNorm input and RoPE angle metadata but only Q projection/rope tensors. No exact K projection output, K conversion metadata, or K RoPE expected tensor is retained, so a bit-exact RTL comparison cannot begin without a new model execution or an unfrozen reconstruction.",
    }

    probe = {
        "schema_version": 1,
        "mission_id": "rtl-current-tree-ordered-operator-batch-replay-v1",
        "probe_operator": "layer_0.rope_k",
        "retained_coordinate": attempted["retained_context"],
        "exact_rope_k_operator_witness_search": {
            "match_count": len(witness_matches),
            "matches": witness_matches,
        },
        "available_model_bound_fields": [
            "layer_0.input_rmsnorm output_s8[896]",
            "layer_0.input_rmsnorm scale",
            "token-1/head-0 cos_q15_s16[64]",
            "token-1/head-0 sin_q15_s16[64]",
            "same-record K/K-RoPE whole-trace hashes and summary metadata",
        ],
        "missing_executable_fields": missing,
        "stop_reason": attempted["status"],
    }
    probe_sha256 = sha256_bytes(json_bytes(probe))

    report = {
        "schema_version": 1,
        "mission_id": "rtl-current-tree-ordered-operator-batch-replay-v1",
        "producer_role": "engineer",
        "review_status": "PENDING_FRESH_REVIEWER",
        "contract_bindings": {
            "mission": artifact(MISSION, "handoff:rtl-current-tree-ordered-operator-batch-replay-v1/mission.json"),
            "published_rope_q_dependency": artifact(DEPENDENCY_ROUND, "handoff:manager-record-rope-q-certified-frontier-v1/round-0001.json"),
            "rope_q_certification_decision": artifact(ROPE_Q_DECISION, "handoff:rtl-rope-q-current-tree-frontier-certification-v1/decision.json"),
            "workload": artifact(WORKLOAD),
            "rtl_manifest": artifact(RTL_MANIFEST),
            "traceability": artifact(TRACEABILITY),
            "tool": artifact(Path(__file__).resolve()),
        },
        "frozen_order": [f"layer_0.{name}" for name in order],
        "rtl_tree": tree,
        "retained_inputs": {
            "rope_q_capture_sha256_manifest": artifact(CAPTURE_SUMS),
            "capture_member_count": capture_member_count,
            "witness": artifact(WITNESS),
            "nonexecutable_same_record_summary": artifact(NONEXECUTABLE_SUMMARY),
        },
        "prior_frontier": {
            "accepted_prefix": PRIOR_PREFIX,
            "maximal_operator": "layer_0.rope_q",
            "first_unsupported": "layer_0.rope_k",
        },
        "attempted_operators": [attempted],
        "newly_passing_operators": [],
        "maximal_passing_prefix": PRIOR_PREFIX,
        "first_failing_or_unsupported_operator": {
            "operator": "layer_0.rope_k",
            "classification": "missing_executable_reference_artifact",
            "boundary_probe_path": relative(PROBE),
            "boundary_probe_sha256": probe_sha256,
        },
        "batch_result": "STOPPED_AT_FIRST_OPERATOR_MISSING_EXECUTABLE_REFERENCE",
        "later_operators_attempted": False,
        "forbidden_flows_run": [],
        "state_mutations": {
            "rtl": False,
            "frontier_ledger_manifest_traceability_pipeline_seals": False,
            "synthesis_opensta_ppa": False,
            "full_model_baseline_candidate_quality_benchmark_scale32": False,
        },
        "review_requirement": "Fresh Reviewer must independently accept or reject this ordered batch boundary; publication is separate.",
    }

    log = "\n".join(
        [
            "ACE2_ORDERED_OPERATOR_BATCH_REPLAY",
            f"rtl_tree_sha256={tree['sha256']}",
            "prior_frontier=layer_0.rope_q",
            "start_operator=layer_0.rope_k",
            "retained_context=c4_en_512:record64:layer0:token1:head0",
            "newly_passing_operators=0",
            "maximal_passing_operator=layer_0.rope_q",
            "first_unsupported_operator=layer_0.rope_k",
            "stop_reason=missing_executable_reference_artifact",
            "rtl_simulation_executed=false",
            "synthesis_opensta_ppa_executed=false",
            "result=BOUNDARY_CAPTURED_PENDING_FRESH_REVIEWER",
            "",
        ]
    )
    return report, probe, log


def write_outputs(report: dict[str, Any], probe: dict[str, Any], log: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(json_bytes(report))
    PROBE.write_bytes(json_bytes(probe))
    LOG.write_text(log, encoding="utf-8")
    rows = []
    for path in (PROBE, LOG, REPORT):
        rows.append(f"{sha256_file(path)}  {relative(path)}")
    SUMS.write_text("\n".join(rows) + "\n", encoding="utf-8")


def check_outputs(report: dict[str, Any], probe: dict[str, Any], log: str) -> None:
    require(REPORT.read_bytes() == json_bytes(report), "ordered-prefix report differs")
    require(PROBE.read_bytes() == json_bytes(probe), "boundary probe differs")
    require(LOG.read_text(encoding="utf-8") == log, "batch replay log differs")
    validate_sha256_manifest(SUMS, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report, probe, log = build()
    if args.check:
        check_outputs(report, probe, log)
        print(
            "ACE2_ORDERED_BATCH_REPLAY_CHECK_PASS "
            "maximal_prefix=layer_0.rope_q first_unsupported=layer_0.rope_k"
        )
    else:
        write_outputs(report, probe, log)
        print(
            "ACE2_ORDERED_BATCH_REPLAY_BOUNDARY_CAPTURED "
            "maximal_prefix=layer_0.rope_q first_unsupported=layer_0.rope_k"
        )


if __name__ == "__main__":
    main()
