#!/usr/bin/env python3
"""Freeze image, oracle, and response-gate identities after image verification."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    REPOSITORY,
    REVISION,
    ROOT,
    SOURCE_CONTRACT,
    TERMINATION_TOKEN_IDS,
    canonical_bytes,
    file_record,
    load_json,
    require,
    sha256_bytes,
    sha256_file,
    verify_source_contract,
    verify_source_snapshot,
    write_json,
)


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
ORACLE = ROOT / "tools/qwen_instruct_w4a8_oracle.py"
GATE = ROOT / "tools/qwen_instruct_response_gate.py"
COMMON = ROOT / "tools/qwen_instruct_option_b.py"
BUILDER = ROOT / "tools/build_qwen_instruct_w4a8_image.py"
DEMO = ROOT / "tools/ace2_chat_demo.py"
RUNTIME = ROOT / "tools/run_full_qwen_command_schedule_runtime.py"


def wrap(contract: dict[str, Any]) -> dict[str, Any]:
    return {"contract": contract, "contract_sha256": sha256_bytes(canonical_bytes(contract))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-at-utc", required=True)
    args = parser.parse_args()
    require(TIMESTAMP.fullmatch(args.generated_at_utc) is not None, "timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    verify_source_contract()
    source = verify_source_snapshot()
    required = [
        "full_model_image.bin",
        "image_contract_v2.json",
        "manifest.json",
        "validation_report.json",
        "reproducibility.json",
        "independent_postbuild_audit.json",
    ]
    for name in required:
        require((IMAGE_DIR / name).is_file(), f"verified image artifact is missing: {name}")
    audit = load_json(IMAGE_DIR / "independent_postbuild_audit.json")
    require(audit.get("status") == "PASS_INDEPENDENT_POSTBUILD_AUDIT_PENDING_FRESH_REVIEW", "independent image audit did not pass")
    authenticated = {
        "calibration_contract": file_record(CALIBRATION_DIR / "calibration_contract.json"),
        "calibration_report": file_record(CALIBRATION_DIR / "calibration_report.json"),
        "derived_scales": file_record(CALIBRATION_DIR / "derived_scales.json"),
        "full_model_image": file_record(IMAGE_DIR / "full_model_image.bin"),
        "image_contract": file_record(IMAGE_DIR / "image_contract_v2.json"),
        "image_manifest": file_record(IMAGE_DIR / "manifest.json"),
        "image_validation": file_record(IMAGE_DIR / "validation_report.json"),
        "image_reproducibility": file_record(IMAGE_DIR / "reproducibility.json"),
        "independent_image_audit": file_record(IMAGE_DIR / "independent_postbuild_audit.json"),
        "source_contract": file_record(SOURCE_CONTRACT),
    }
    oracle_contract = {
        "schema_version": 1,
        "contract_id": "qwen-instruct-option-b-w4a8-oracle-v1",
        "generated_at_utc": args.generated_at_utc,
        "status": "FROZEN",
        "model": {"repository": REPOSITORY, "revision": REVISION, "source": source},
        "source": file_record(ORACLE),
        "shared_identity_source": file_record(COMMON),
        "fixed_point_semantics_source": file_record(ROOT / "tools/ace2_full_model_fixed_point.py"),
        "authenticated_artifacts": authenticated,
        "generation_policy": {
            "chat_template": True,
            "add_generation_prompt": True,
            "greedy_argmax": True,
            "sampling": False,
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
            "max_new_tokens_range": [1, 32],
            "cache_policy": "full-prefix recomputation for the software oracle; accelerator KV behavior is checked separately",
        },
        "scope": "quantized software oracle only; not accelerator completion",
    }
    write_json(IMAGE_DIR / "oracle_contract.json", wrap(oracle_contract))
    gate_contract = {
        "schema_version": 1,
        "contract_id": "qwen-instruct-option-b-response-gate-v1",
        "generated_at_utc": args.generated_at_utc,
        "status": "FROZEN",
        "source": file_record(GATE),
        "algorithm": {
            "normalization": "Unicode NFC followed by outer-whitespace stripping",
            "semantic_challenge": "Reviewer-selected non-frozen prompt requesting an exact nonce response",
            "required": [
                "normalized decoded text equals normalized expected nonce",
                "at least two generated tokens before the first termination token",
                "decoded text is nonempty",
                "no control, surrogate, or U+FFFD replacement character",
            ],
            "termination_token_ids": list(TERMINATION_TOKEN_IDS),
        },
        "prompt_storage": "record hashes and lengths, not prompt text, in runtime evidence",
    }
    write_json(IMAGE_DIR / "response_gate_contract.json", wrap(gate_contract))
    identity = {
        "schema_version": 1,
        "classification": "qwen_instruct_option_b_identity_freeze",
        "generated_at_utc": args.generated_at_utc,
        "status": "PASS_OPTION_B_IDENTITIES_FROZEN_DEMO_RETARGET_PENDING",
        "model": {"repository": REPOSITORY, "revision": REVISION},
        "image": authenticated,
        "oracle_contract": file_record(IMAGE_DIR / "oracle_contract.json"),
        "response_gate_contract": file_record(IMAGE_DIR / "response_gate_contract.json"),
        "sources": {
            "builder": file_record(BUILDER),
            "freeze_tool": file_record(Path(__file__)),
            "oracle": file_record(ORACLE),
            "response_gate": file_record(GATE),
            "shared_identity": file_record(COMMON),
        },
        "demo_target": {
            "retargeted": False,
            "base_bound_demo": file_record(DEMO),
            "base_bound_runtime": file_record(RUNTIME),
            "next_authorized_step": "separate post-freeze Instruct demo/runtime retarget and accelerator/reference verification",
        },
    }
    write_json(IMAGE_DIR / "option_b_identity_manifest.json", identity)
    names = [
        *required,
        "oracle_contract.json",
        "response_gate_contract.json",
        "option_b_identity_manifest.json",
    ]
    rows = [f"{sha256_file(IMAGE_DIR / name)}  {name}" for name in sorted(names)]
    (IMAGE_DIR / "OPTION_B_SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        "ACE2_QWEN_INSTRUCT_OPTION_B_IDENTITIES_FROZEN "
        f"image_sha256={authenticated['full_model_image']['sha256']} "
        f"manifest_sha256={authenticated['image_manifest']['sha256']}"
    )


if __name__ == "__main__":
    main()
