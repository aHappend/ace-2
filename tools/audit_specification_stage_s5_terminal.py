#!/usr/bin/env python3
"""Audit the specification-stage closure after terminal S5 NO-GO."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
CHECKPOINT = ROOT / "CHECKPOINT.md"
INDEX = ROOT / "latest.json"
BUILD = ROOT / "build/bf16-full-finetune-successor-s5"
TERMINAL = BUILD / "backend-terminal-audit-20260808T150208Z.json"
PROVENANCE = BUILD / "submission-client-provenance-audit-20260808T151648Z.json"
FINAL_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s5-terminal-provenance-fresh-review-final-20260808.json"


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def companion_matches(path: Path) -> bool:
    companion = path.with_suffix(path.suffix + ".sha256")
    return companion.is_file() and companion.read_text(encoding="ascii").split() == [sha256(path), path.name]


def record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def table_count(text: str, start: str, end: str) -> int:
    body = text.split(start, 1)[1].split(end, 1)[0]
    return sum(1 for line in body.splitlines() if line.startswith("| `"))


def audit() -> dict[str, Any]:
    for path in (SPEC, BENCHMARK, PIPELINE, GROUND_TRUTH, CHECKPOINT, INDEX, TERMINAL, PROVENANCE, FINAL_REVIEW):
        require(path.is_file(), f"required specification input is absent: {path}")
    for path in (INDEX, TERMINAL, PROVENANCE, FINAL_REVIEW):
        require(companion_matches(path), f"checksum companion differs: {path}")

    spec = SPEC.read_text(encoding="utf-8")
    normalized_spec = " ".join(spec.split())
    ground = GROUND_TRUTH.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    benchmark = load(BENCHMARK)
    pipeline = load(PIPELINE)
    terminal = load(TERMINAL)
    provenance = load(PROVENANCE)
    final_review = load(FINAL_REVIEW).get("review", {})
    binding = benchmark.get("authoritative_s5_terminal_containment", {})

    parameter_count = table_count(spec, "| Parameter | Default | Legal value | Contract |", "Every public port below")
    port_count = table_count(spec, "| Port | Direction | Width (unsigned packed bits) | Contract |", "### Mission-specific acceptance matrix")
    behavior_ok = all(fragment in spec for fragment in (
        "### Public `ace2_shell` parameter and port contract",
        "Parameters are signed SystemVerilog `integer` elaboration constants",
        "there are no `signed` public port declarations",
        "implementation supports exactly the tuple below",
    )) and parameter_count == 12 and port_count == 64

    clock_reset_ok = all(fragment in normalized_spec for fragment in (
        "### Latency and completion semantics",
        "Under external backpressure there is no finite completion-latency guarantee",
        "### Reset, clock, and CDC rules",
        "A0 contains no internal clock-domain crossing",
        "Reset may assert asynchronously",
        "deassert synchronously",
        "cmd_done_valid_o",
    ))

    matrix_classes = ("Normal", "Semantic F", "Boundary", "Illegal", "Reset", "Stall", "Recovery")
    matrix_ok = "### Mission-specific acceptance matrix" in spec and all(f"| {name} |" in spec for name in matrix_classes)
    s5_matrix_ok = all(fragment.lower() in spec.lower() for fragment in (
        "terminal 27/56 versus 48",
        "60/60 terminal checks",
        "22/22 provenance checks",
        "transport equivalence unestablished",
        "any S5 retry",
    ))

    benchmark_ok = (
        benchmark.get("applies") is False
        and benchmark.get("external_contract") is None
        and benchmark.get("stage") == "specification"
        and "active_marker_free_s5_successor" not in benchmark
        and benchmark.get("contract_status") == "s5_terminal_dev_quality_no_go_client_provenance_deviation_selected_policy_null_downstream_closed"
        and binding.get("status") == "TERMINAL_S5_DEV_QUALITY_NO_GO_FRESH_REVIEW_DONE"
        and binding.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE"
        and binding.get("cardinality") == terminal.get("cardinality")
        and binding.get("dev") == {"executed_once": True, "response_count": 56, "hard_pass_count": 27, "minimum_hard_passes": 48, "status": "NO_GO"}
        and binding.get("retention_executed") is False
        and binding.get("holdout_executed") is False
        and binding.get("retry_resume_relaunch_resubmit_repair_rescore_or_attempt_0002_allowed") is False
        and binding.get("client_provenance_audit", {}).get("classification") == "BOUNDED_SUBMISSION_CLIENT_PROVENANCE_DEVIATION"
        and binding.get("client_provenance_audit", {}).get("canonical_executable_used_for_submission") is False
        and binding.get("client_provenance_audit", {}).get("transport_equivalence_claimed") is False
        and binding.get("final_fresh_review", {}).get("status") == "done"
    )

    terminal_ok = terminal.get("check_count") == 60 and terminal.get("failed_check_count") == 0 and all(terminal.get("checks", {}).values())
    provenance_ok = provenance.get("check_count") == 22 and provenance.get("failed_check_count") == 0 and all(provenance.get("checks", {}).values())
    review_text = str(final_review.get("ruling_text", final_review.get("reason", "")))
    required_rulings = (
        "s5.terminal-cardinality: supported",
        "s5.dev-quality-no-go: supported",
        "s5.client-provenance-deviation: supported",
        "s5.transport-equivalence-not-established: supported",
        "s5.no-retry-downstream: supported",
        "s5.stage-remains-specification: supported",
        "s5.checkpoint-current: supported",
    )
    review_ok = final_review.get("status") == "done" and all(ruling in review_text.lower() for ruling in required_rulings)

    no_stale_active_claim = not any(fragment in spec + ground + checkpoint + json.dumps(benchmark) for fragment in (
        "active unconsumed successor is the marker-free S5",
        "S5 remains marker-free and unconsumed",
        "ACCEPTED_MARKER_FREE_S5_PACKAGE_AWAITING_FRESH_OPERATOR_AUTHORITY",
        "fresh explicit operator authority for exactly one S5 lifecycle",
    ))

    checklist = {
        "spec.behavior-interface": {"status": "PASS" if behavior_ok else "FAIL", "parameter_count": parameter_count, "port_count": port_count},
        "spec.clock-reset-protocol": {"status": "PASS" if clock_reset_ok else "FAIL", "clock_domains": 1, "reset": "asynchronous assertion, synchronous deassertion", "backpressure": "unbounded unless watchdog is enabled"},
        "spec.acceptance-matrix": {"status": "PASS" if matrix_ok and s5_matrix_ok else "FAIL", "observed_classes": list(matrix_classes), "s5_terminal_no_go_bound": s5_matrix_ok},
        "spec.benchmark-interface-closure": {"status": "PASS" if benchmark_ok else "FAIL", "external_benchmark_applies": benchmark.get("applies"), "s5_terminal_status": binding.get("status")},
    }
    pipeline_ok = pipeline.get("current_stage") == "specification" and sha256(PIPELINE) == "1fdc818abc8458749eeccb2fa3dc6a09d2dc2ab7fd6ea2e446ee222f777999bc"
    status = "PASS" if all(item["status"] == "PASS" for item in checklist.values()) and pipeline_ok and terminal_ok and provenance_ok and review_ok and no_stale_active_claim else "FAIL"

    return {
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "read-only specification-stage S5 terminal closure audit; no submission, retry, evaluator access, hidden/golden inspection, W4A8, RTL simulation, formal, synthesis/PPA, FPGA/U280, or stage advance",
        "status": status,
        "checklist": checklist,
        "pipeline_integrity": {"status": "PASS" if pipeline_ok else "FAIL", "current_stage": pipeline.get("current_stage"), "manager_owned_state_mutated": False},
        "terminal_evidence": {"status": "PASS" if terminal_ok else "FAIL", "artifact": record(TERMINAL)},
        "client_provenance": {"status": "PASS" if provenance_ok else "FAIL", "artifact": record(PROVENANCE), "transport_equivalence_claimed": False},
        "fresh_reviewer": {"status": "PASS" if review_ok else "FAIL", "artifact": record(FINAL_REVIEW), "review_status": final_review.get("status")},
        "stale_active_s5_claim_absent": no_stale_active_claim,
        "bound_specification": {"spec": record(SPEC), "benchmark_interface": record(BENCHMARK), "ground_truth": record(GROUND_TRUTH), "checkpoint": record(CHECKPOINT), "context_index": record(INDEX)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(f"ACE2_S5_TERMINAL_SPECIFICATION_AUDIT_{result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
