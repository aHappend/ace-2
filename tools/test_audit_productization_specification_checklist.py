#!/usr/bin/env python3
"""Focused regressions for the specification-checklist auditor."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_productization_specification_checklist import (  # noqa: E402
    BENCHMARK,
    PIPELINE,
    PIPELINE_CHECKSUM,
    SPEC,
    checksum_companion_matches,
    evaluate,
    sha256,
)
from audit_rtl_stage_contract import audit as audit_shell_contract  # noqa: E402


class SpecificationChecklistAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec_text = SPEC.read_text(encoding="utf-8")
        cls.benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        cls.pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
        cls.shell_report = audit_shell_contract()
        inventory = ROOT / cls.benchmark["u280_environment_gate"]["latest_inventory"]
        cls.inventory_sha256 = sha256(inventory)

    def run_evaluation(
        self,
        *,
        spec_text: str | None = None,
        benchmark: dict | None = None,
        pipeline: dict | None = None,
        shell_report: dict | None = None,
    ) -> dict:
        return evaluate(
            spec_text=self.spec_text if spec_text is None else spec_text,
            benchmark=self.benchmark if benchmark is None else benchmark,
            pipeline=self.pipeline if pipeline is None else pipeline,
            shell_report=self.shell_report if shell_report is None else shell_report,
            pipeline_checksum_matches=checksum_companion_matches(
                PIPELINE, PIPELINE_CHECKSUM
            ),
            inventory_exists=True,
            inventory_sha256=self.inventory_sha256,
        )

    def test_live_contract_passes(self) -> None:
        expected_pipeline_integrity = checksum_companion_matches(
            PIPELINE, PIPELINE_CHECKSUM
        )
        result = self.run_evaluation()
        self.assertEqual(result["status"], "PASS_SPECIFICATION_CHECKLIST_COMPLETE")
        self.assertEqual(
            result["supporting_constraints"]["pipeline_state_integrity"]["status"],
            "PASS" if expected_pipeline_integrity else "FAIL",
        )
        self.assertEqual(
            result["conclusions"]["manager_pipeline_reconciliation_required"],
            not expected_pipeline_integrity,
        )
        self.assertEqual(
            result["supporting_constraints"]["stage_1_current_authority_gate"][
                "status"
            ],
            "PASS",
        )
        self.assertTrue(result["conclusions"]["stage_1_authority_blocked"])
        self.assertEqual(
            result["conclusions"]["stage_1_authority_blocker"],
            "NONOFFICIAL_RTL_REFERENCE_INTEGRATION_ACCEPTED_OFFICIAL_CHAT_DENIED",
        )
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])
        self.assertFalse(result["conclusions"]["product_completion_eligible"])

    def test_checksum_drift_marks_manager_reconciliation_required(self) -> None:
        result = evaluate(
            spec_text=self.spec_text,
            benchmark=self.benchmark,
            pipeline=self.pipeline,
            shell_report=self.shell_report,
            pipeline_checksum_matches=False,
            inventory_exists=True,
            inventory_sha256=self.inventory_sha256,
        )
        self.assertEqual(
            result["supporting_constraints"]["pipeline_state_integrity"]["status"],
            "FAIL",
        )
        self.assertTrue(
            result["conclusions"]["manager_pipeline_reconciliation_required"]
        )
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])

    def test_public_interface_count_drift_fails_behavior(self) -> None:
        shell_report = copy.deepcopy(self.shell_report)
        shell_report["checks"]["port_count_source"] = 63
        result = self.run_evaluation(shell_report=shell_report)
        self.assertEqual(result["checklist"]["spec.behavior-interface"]["status"], "FAIL")

    def test_fused_qkv_spec_base_drift_fails_behavior(self) -> None:
        spec_text = self.spec_text.replace(
            "`src1=0x0000000100000000 + L*0x000000000071c000`",
            "`src1=aligned image address`",
            1,
        )
        result = self.run_evaluation(spec_text=spec_text)
        self.assertEqual(result["checklist"]["spec.behavior-interface"]["status"], "FAIL")

    def test_fused_qkv_contract_stride_drift_fails_behavior(self) -> None:
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["fused_qkv_contract"]["command"]["memory_layout"][
            "weight_layer_stride"
        ] += 16
        result = self.run_evaluation(benchmark=benchmark)
        self.assertEqual(result["checklist"]["spec.behavior-interface"]["status"], "FAIL")

    def test_protocol_fragment_removal_fails_protocol(self) -> None:
        spec_text = self.spec_text.replace("No internal CDC is permitted.", "", 1)
        result = self.run_evaluation(spec_text=spec_text)
        self.assertEqual(result["checklist"]["spec.clock-reset-protocol"]["status"], "FAIL")

    def test_recovery_class_removal_fails_acceptance_matrix(self) -> None:
        spec_text = self.spec_text.replace("| Recovery |", "| Removed |")
        result = self.run_evaluation(spec_text=spec_text)
        self.assertEqual(result["checklist"]["spec.acceptance-matrix"]["status"], "FAIL")

    def test_external_benchmark_drift_fails_closure(self) -> None:
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["applies"] = True
        result = self.run_evaluation(benchmark=benchmark)
        self.assertEqual(
            result["checklist"]["spec.benchmark-interface-closure"]["status"],
            "FAIL",
        )

    def test_authority_spec_drift_fails_overall_status(self) -> None:
        spec_text = self.spec_text.replace(
            "Stage advancement remains a Manager action",
            "Stage advancement is locally self-authorized",
            1,
        )
        result = self.run_evaluation(spec_text=spec_text)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")
        self.assertEqual(
            result["supporting_constraints"]["stage_1_current_authority_gate"][
                "status"
            ],
            "FAIL",
        )

    def test_authority_contract_drift_fails_overall_status(self) -> None:
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["active_productization_contract"][
            "production_acceptance_authority_gate"
        ]["authorization"]["rtl_authorized"] = True
        result = self.run_evaluation(benchmark=benchmark)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")

    def test_forbidden_effect_removal_fails_closed(self) -> None:
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["active_productization_contract"][
            "production_acceptance_authority_gate"
        ]["forbidden_effects"].remove("Stage 2")
        result = self.run_evaluation(benchmark=benchmark)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])

    def test_exact_diagnosis_evidence_drift_fails_closed(self) -> None:
        benchmark = copy.deepcopy(self.benchmark)
        pipeline = copy.deepcopy(self.pipeline)
        replacement = "0" * 64
        pipeline["stage1_blocker"]["evidence"]["v_residual_structure_diagnosis"][
            "result"
        ]["sha256"] = replacement
        benchmark["active_productization_contract"][
            "production_acceptance_authority_gate"
        ]["canonical_stage1_blocker"]["evidence"][
            "v_residual_structure_diagnosis"
        ]["result"]["sha256"] = replacement
        result = self.run_evaluation(benchmark=benchmark, pipeline=pipeline)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])

    def test_pipeline_gate_release_without_manager_stage_advance_fails_closed(self) -> None:
        pipeline = copy.deepcopy(self.pipeline)
        pipeline["stage1_blocker"]["status"] = "RELEASED"
        result = self.run_evaluation(pipeline=pipeline)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])

    def test_pipeline_blocker_removal_fails_closed(self) -> None:
        pipeline = copy.deepcopy(self.pipeline)
        pipeline.pop("stage1_blocker")
        result = self.run_evaluation(pipeline=pipeline)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])

    def test_pipeline_stage_drift_fails_overall_status(self) -> None:
        pipeline = copy.deepcopy(self.pipeline)
        pipeline["current_stage"] = "rtl"
        result = self.run_evaluation(pipeline=pipeline)
        self.assertEqual(result["status"], "FAIL_SPECIFICATION_CHECKLIST")
        self.assertFalse(result["conclusions"]["stage_advancement_eligible"])


if __name__ == "__main__":
    unittest.main()
