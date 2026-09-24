#!/usr/bin/env python3
"""Independent static verifier for the frozen Instruct-specific W4A8 image."""

from __future__ import annotations

import argparse
import json
import mmap
import re
from pathlib import Path

import verify_full_qwen_image_v2 as base
from qwen_instruct_option_b import (
    CALIBRATION_DIR,
    IMAGE_DIR,
    IMAGE_ID,
    REPOSITORY,
    REVISION,
    ROOT,
    SNAPSHOT,
    SOURCE_CONTRACT,
    canonical_bytes,
    file_record,
    require,
    sha256_file,
)


EXPECTED_INPUT_HASHES: dict[str, str] = {
    "builder": "9c073adb2d653ab9236b7ee137a9f2ad9bbfa5fee83905e25806cbb3969f6a5c",
    "calibration_contract": "084ab27de7e19b3f66cbcacbd7c219b67e7ef1cbc51c0d92e6da67af7e6df763",
    "calibration_report": "c86211c4853a7b8383c6fd7bd8af68996bce83594eb03dafd3cb67e8ffde6a3c",
    "inventory": "6472b1460c9908255655fcca45d0e6d5eadc065f6debf5b6605e9107568d1950",
    "model": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "oracle": "d9c5aaf0f4f87d85d7a29bdb94b69d82f50033acb5a90da0a39e9ffa2c13b591",
    "provenance": "659ea9edbe50bfe961086a4a652788b6be343e792b8470755ef41f7a20fa2841",
    "response_gate": "5906f85f461b379249aee23efb00c5bc5d677d53dafbe84af7e8a9fe54398226",
    "scales": "cb99f71eeca54dbbc1a703dbcde4a9f3e7868ea317b0734addbf1a45d19b07ca",
    "schedule": "838b2c019a6028a92ffef8b9cc087cdcb616f33f60a20c6b24cb33aed37bb002",
    "shared_identity": "ebe344dadeff417fe52a181718273506cab36a0f5692b1c82bcc98ab3670b44c",
    "shell": "1325b0fa8993f5d760d38ad15bff0065088eeb3e894fda431e5cdcad0a614bb9",
    "shell_tb": "0a086feb925d2301f3df83b7f5bc3a1ba49ba99705cb67d357ccc4244b198547",
    "source_contract": "0da7d95e92637e29c0369ce9aa9c8d0fe9d50802cf1144bc79095b129962576d",
    "v1_audit": "e5006f4c062693a65a02a732300468fddfe7bafb4ad3a4f5bc173fd5aed8122b",
}
EXPECTED_ARTIFACT_HASHES: dict[str, str] = {
    "full_model_image.bin": "ce1f94d930bf4d195b33c6aab4b18736419a34bb3671ea4d23ef2fe6be08ad42",
    "image_contract_v2.json": "c8aa96b6fe660ff78f659eadfca7bf1785d391fb8b3c00ee4a940813604f466e",
    "manifest.json": "b3c57252ca53044ea59e78e0afd739e88645d91db23570bb7de4f5306fdbe158",
    "reproducibility.json": "c4d19146d69cde18ede3cd901b98d6bb3112d88a14222d4622805178b93b4c5b",
    "validation_report.json": "d29779283c32a6093f833c6df3074daacfcf532b0ea0f4bf9d1d738f2d3d8a79",
}
EXPECTED_CONTRACT_SHA256 = "5afe2bad158001c7b493913a946e0ddbd3471fa52983b0627a283c41343aff76"
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def configure() -> None:
    require(EXPECTED_CONTRACT_SHA256 != "PENDING", "verifier identities are not frozen")
    paths = {
        "v1_audit": base.V1_AUDIT,
        "inventory": base.INVENTORY,
        "schedule": base.SCHEDULE,
        "provenance": base.PROVENANCE,
        "scales": CALIBRATION_DIR / "derived_scales.json",
        "calibration_contract": CALIBRATION_DIR / "calibration_contract.json",
        "calibration_report": CALIBRATION_DIR / "calibration_report.json",
        "shell": base.SHELL,
        "shell_tb": base.SHELL_TB,
        "builder": ROOT / "tools/build_qwen_instruct_w4a8_image.py",
        "model": SNAPSHOT / "model.safetensors",
        "source_contract": SOURCE_CONTRACT,
        "oracle": ROOT / "tools/qwen_instruct_w4a8_oracle.py",
        "response_gate": ROOT / "tools/qwen_instruct_response_gate.py",
        "shared_identity": ROOT / "tools/qwen_instruct_option_b.py",
    }
    require(set(paths) == set(EXPECTED_INPUT_HASHES), "verifier input identity labels differ")
    base.MISSION_ID = IMAGE_ID
    base.OUTPUT_DIR = IMAGE_DIR
    base.REVISION = REVISION
    base.MODEL = paths["model"]
    base.SCALES = paths["scales"]
    base.SCALE_RUN_CONTRACT = paths["calibration_contract"]
    base.SCALE_RESULTS = paths["calibration_report"]
    base.BUILDER = paths["builder"]
    base.EXPECTED_INPUT_HASHES = {paths[name]: digest for name, digest in EXPECTED_INPUT_HASHES.items()}
    base.EXPECTED_ARTIFACT_HASHES = EXPECTED_ARTIFACT_HASHES
    base.EXPECTED_CONTRACT_SHA256 = EXPECTED_CONTRACT_SHA256

    def relative_label(path: Path) -> str:
        if path == paths["model"]:
            return f"hf-cache://{REPOSITORY}@{REVISION}/model.safetensors"
        return path.resolve().relative_to(ROOT.resolve()).as_posix()

    base.relative_label = relative_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output", type=Path, default=IMAGE_DIR / "independent_postbuild_audit.json")
    args = parser.parse_args()
    require(TIMESTAMP.fullmatch(args.generated_at_utc) is not None, "timestamp must use YYYY-MM-DDTHH:MM:SSZ")
    configure()
    inputs = base.verify_inputs()
    artifact_identity = base.verify_artifact_identity()
    inventory = base.load_json(base.INVENTORY)
    schedule = base.load_json(base.SCHEDULE)
    scales = base.load_json(base.SCALES)
    contract_wrapper = base.load_json(IMAGE_DIR / "image_contract_v2.json")
    manifest = base.load_json(IMAGE_DIR / "manifest.json")
    validation = base.load_json(IMAGE_DIR / "validation_report.json")
    reproducibility = base.load_json(IMAGE_DIR / "reproducibility.json")
    require(manifest["status"] == "COMPLETE_IMAGE_BUILT_AND_SELF_VERIFIED_PENDING_FRESH_REVIEW", "manifest status differs")
    require(validation["status"] == "PASS", "self-validation status differs")
    contract_check = base.verify_contract(contract_wrapper)
    require(contract_wrapper["contract"]["source_model"]["repository"] == REPOSITORY, "Instruct repository identity differs")
    require(contract_wrapper["contract"]["frozen_scale_provenance"]["base_scale_artifact_reused"] is False, "Base scale substitution detected")
    image_path = IMAGE_DIR / "full_model_image.bin"
    image_handle = image_path.open("rb")
    image = mmap.mmap(image_handle.fileno(), 0, access=mmap.ACCESS_READ)
    tensors = base.RawSafetensors(base.MODEL)
    try:
        checks = {
            "immutable_inputs": inputs,
            "artifact_identity": artifact_identity,
            "contract": contract_check,
            "reproducibility": base.verify_reproducibility(reproducibility),
            "region_map": base.verify_region_map(inventory, contract_wrapper, manifest, image),
            "schedule_coverage": base.verify_schedule(inventory, schedule),
            "linear_tensors": base.verify_linears(inventory, scales, manifest, tensors, image),
            "rmsnorm_gains_first": base.verify_rmsnorm(inventory, schedule, scales, contract_wrapper, manifest, tensors, image),
            "operator_aux": base.verify_aux(inventory, scales, manifest, image),
        }
    finally:
        tensors.close()
        image.close()
        image_handle.close()
    report = {
        "schema_version": 1,
        "mission_id": IMAGE_ID,
        "generated_at_utc": args.generated_at_utc,
        "status": "PASS_INDEPENDENT_POSTBUILD_AUDIT_PENDING_FRESH_REVIEW",
        "classification": "independent_standard_library_raw_safetensors_instruct_image_verification",
        "checks": checks,
        "verifier": file_record(Path(__file__)),
        "accepted_geometry_verifier": file_record(ROOT / "tools/verify_full_qwen_image_v2.py"),
        "scope_guards": {
            "builder_imported": False,
            "torch_imported": False,
            "transformers_imported": False,
            "model_constructed": False,
            "model_called": False,
            "recalibration": False,
            "demo_retarget": False,
            "rtl_or_testbench_change": False,
            "synthesis_opensta_ppa": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(report))
    output.with_suffix(".SHA256SUMS").write_text(
        f"{sha256_file(output)}  {output.relative_to(ROOT).as_posix()}\n"
        f"{sha256_file(Path(__file__))}  {Path(__file__).relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )
    print(
        "ACE2_QWEN_INSTRUCT_W4A8_IMAGE_INDEPENDENT_AUDIT_PASS "
        f"image_sha256={EXPECTED_ARTIFACT_HASHES['full_model_image.bin']}"
    )


if __name__ == "__main__":
    main()
