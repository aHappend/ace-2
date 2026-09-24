#!/usr/bin/env python3
"""Build a separate Instruct-specific image using the accepted v2 geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_full_qwen_image_v2 as base
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    IMAGE_ID,
    REPOSITORY,
    REVISION,
    ROOT,
    SNAPSHOT,
    SOURCE_CONTRACT,
    file_record,
    load_json,
    require,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
)


SCALES = CALIBRATION_DIR / "derived_scales.json"
CALIBRATION_CONTRACT = CALIBRATION_DIR / "calibration_contract.json"
CALIBRATION_REPORT = CALIBRATION_DIR / "calibration_report.json"
COMMON = ROOT / "tools/qwen_instruct_option_b.py"
ORACLE = ROOT / "tools/qwen_instruct_w4a8_oracle.py"
RESPONSE_GATE = ROOT / "tools/qwen_instruct_response_gate.py"


def relative_label(path: Path) -> str:
    if path == SNAPSHOT / "model.safetensors":
        return f"hf-cache://{REPOSITORY}@{REVISION}/model.safetensors"
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def configure() -> None:
    verify_source_snapshot()
    source_contract = verify_source_contract()
    report = load_json(CALIBRATION_REPORT)
    require(report.get("status") == "PASS_INSTRUCT_SPECIFIC_W4A8_CALIBRATION", "Instruct calibration did not pass")
    require(report["derived_scales"]["sha256"] == sha256_file(SCALES), "Instruct scale artifact differs")
    preserved = {
        path: digest
        for path, digest in base.EXPECTED_INPUT_HASHES.items()
        if path
        not in {
            base.SCALES,
            base.SCALE_RUN_CONTRACT,
            base.SCALE_RESULTS,
            base.MODEL,
            base.SHELL,
            base.SHELL_TB,
        }
    }
    model = SNAPSHOT / "model.safetensors"
    extra = [SCALES, CALIBRATION_CONTRACT, CALIBRATION_REPORT, SOURCE_CONTRACT, COMMON, ORACLE, RESPONSE_GATE, Path(__file__)]
    base.MISSION_ID = IMAGE_ID
    base.REVISION = REVISION
    base.MODEL = model
    base.SCALES = SCALES
    base.SCALE_RUN_CONTRACT = CALIBRATION_CONTRACT
    base.SCALE_RESULTS = CALIBRATION_REPORT
    current_read_only_rtl = {
        base.SHELL: sha256_file(base.SHELL),
        base.SHELL_TB: sha256_file(base.SHELL_TB),
    }
    base.EXPECTED_INPUT_HASHES = preserved | current_read_only_rtl | {model: sha256_file(model)} | {path: sha256_file(path) for path in extra}
    base.relative_label = relative_label
    original_contract_body = base.contract_body

    def contract_body(
        generated_at_utc: str,
        inventory: dict[str, Any],
        input_records: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        contract = original_contract_body(generated_at_utc, inventory, input_records)
        contract["mission_id"] = IMAGE_ID
        contract["operator_authorization"] = {
            "authorization_id": "manager-option-b-genuine-instruct-chat-v1",
            "decision": "build a distinct authenticated Qwen2.5-0.5B-Instruct W4A8 image before demo retargeting",
            "base_model_substitution_authorized": False,
            "demo_retarget_authorized_in_this_step": False,
            "rtl_change_authorized": False,
        }
        contract["source_model"] = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "safetensors": input_records[relative_label(model)],
            "access": "safe_open raw tensors only; no model construction or call during image build",
        }
        contract["frozen_scale_provenance"] = {
            "derived_scales": input_records[relative_label(SCALES)],
            "calibration_contract": input_records[relative_label(CALIBRATION_CONTRACT)],
            "calibration_report": input_records[relative_label(CALIBRATION_REPORT)],
            "source_model_specific": True,
            "base_scale_artifact_reused": False,
            "recalibrated_during_image_build": False,
        }
        contract["option_b_identity_inputs"] = {
            "source_contract": input_records[relative_label(SOURCE_CONTRACT)],
            "shared_identity_source": input_records[relative_label(COMMON)],
            "oracle_source": input_records[relative_label(ORACLE)],
            "response_gate_source": input_records[relative_label(RESPONSE_GATE)],
            "chat_template_sha256": source_contract["source_model"]["chat_template_sha256"],
        }
        contract["builder_source"] = input_records[relative_label(Path(__file__))]
        contract["accepted_geometry_builder_source"] = file_record(ROOT / "tools/build_full_qwen_image_v2.py")
        contract["scope_guards"] = contract["scope_guards"] | {
            "base_image_mutation": False,
            "demo_retarget": False,
        }
        return contract

    base.contract_body = contract_body


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
