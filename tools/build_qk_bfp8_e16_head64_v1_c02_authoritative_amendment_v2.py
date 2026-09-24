#!/usr/bin/env python3
"""Deterministically assemble the static V2 amendment and implementation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_AUTHORITATIVE_AMENDMENT_V2.json"
)
PACKAGE_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_SURFACE_IMPLEMENTATION_PACKAGE_V2.json"
)
EVALUATOR_REL = "tools/qk_bfp8_e16_head64_v1_c02_score_path_evaluator_v2.py"
SURFACE_REL = "tools/qk_bfp8_e16_head64_v1_c02_live_authority_surface_v2.py"
VERIFIER_REL = "tools/verify_qk_bfp8_e16_head64_v1_c02_authoritative_amendment_v2.py"
BUILDER_REL = "tools/build_qk_bfp8_e16_head64_v1_c02_authoritative_amendment_v2.py"
INTERPRETER = Path("/home/argustest/miniconda3/bin/python3.13")
ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0", "TZ": "UTC"}
LIVE_ROOT = "build/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2"
OUTPUT_ROOT = "build/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(compact_bytes(value)).hexdigest()


def write_static(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pretty_bytes(value)
    if path.exists() and path.read_bytes() == payload:
        return
    temporary = path.with_name("." + path.name + ".new")
    if temporary.exists():
        temporary.unlink()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def binding(path: str, expected: str | None = None) -> dict[str, str]:
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    actual = sha256_file(target)
    if expected is not None and actual != expected:
        raise RuntimeError(f"accepted binding mismatch for {path}: {actual}")
    return {"path": path, "sha256": actual}


def argv_for(lane: dict[str, str]) -> list[str]:
    return [
        str(INTERPRETER),
        EVALUATOR_REL,
        "--lane",
        lane["label"],
        "--amendment",
        AMENDMENT_REL,
        "--authority-ledger",
        f"{LIVE_ROOT}/lanes/{lane['namespace_label']}/authority-ledger.json",
        "--evaluation-package",
        "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json",
        "--input-metadata",
        lane["metadata_path"],
        "--tensor-bundle",
        lane["tensor_path"],
        "--output",
        lane["output_path"],
    ]


LANES = {
    "Base": {
        "accepted_invocation_sha256": "58b90f06f10d179cb12820ee5a376cf3807eb9f2af929990dacac138a8fa6731",
        "authoritative_tensors_sha256": "aa0d327f251e86cdbb5c00b4a2498bdfa4bbb5b167d55e2f370c1dd83a2285bd",
        "generation_id": "c02-score-path-eval-base-v1",
        "historical_logical_command_descriptor_sha256": "a89602cbcf8d4b5a3d7fd11737615e399ce2d91d869ea1ea46b7a0f1f1b6cb4f",
        "input_metadata_sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
        "input_tensor_bundle_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
        "label": "Base",
        "metadata_path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/result.json",
        "model_identity_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
        "namespace_label": "base",
        "output_path": OUTPUT_ROOT + "/base/result.json",
        "record_id": "c02-score-path-base-exactly-once-v2",
        "tensor_path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/base/attention-substage-tensors.bin",
        "tensor_records": {
            "bf16_oracle_scores": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765",
            "realized_key_source": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a",
            "realized_query_source": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429",
        },
    },
    "checkpoint-176": {
        "accepted_invocation_sha256": "78e83c2f2ad2a5ec15a96dc320bd3e86f5ed3fbedbf3de9871a03b40225c829b",
        "authoritative_tensors_sha256": "5af500371148338636f078e0ec69973666f54f70d3c0f19d350456cf74c5bdb2",
        "generation_id": "c02-score-path-eval-checkpoint-176-v1",
        "historical_logical_command_descriptor_sha256": "b9bb0df45304965056f5df21a44f194a22341995923eff0ce20ccd89022523ec",
        "input_metadata_sha256": "3f73b5359e021ddb6183ec468ae154b3cda0807365b248773d80466e477b4e86",
        "input_tensor_bundle_sha256": "07743a5df6bfa00d5e977dc622b70e90af9b6789feaeb5ae689448ac803ceb59",
        "label": "checkpoint-176",
        "metadata_path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/checkpoint-176/result.json",
        "model_identity_sha256": "94eb1fe142aa11925863386e796e1157341351cfb664283388291a4fa9d8bf98",
        "namespace_label": "checkpoint-176",
        "output_path": OUTPUT_ROOT + "/checkpoint-176/result.json",
        "record_id": "c02-score-path-checkpoint-176-exactly-once-v2",
        "tensor_path": "evidence/diagnostics/w4a8-c02-attention-substage-trace-v2/checkpoint-176/attention-substage-tensors.bin",
        "tensor_records": {
            "bf16_oracle_scores": "776be2e58246c8641fa483e331e0d63896f96f14b3756548e15af3870d89c456",
            "realized_key_source": "2c4b99f5bc97d0f95b886dfebf62242148940d7cc9021e197dfdee0aed9e9902",
            "realized_query_source": "18671805abe3835d8f2e23bffb041ced6a05f41b8eac4f698f41d302def73b18",
        },
    },
}


def accepted_bindings() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "evaluation": {
            "package": binding(
                "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json",
                "0bd51407f20950a93f47a520718a3bb4d5a5006d990d60aef626d16a5f9d976d",
            ),
            "proof": binding(
                "evidence/verification/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-package-v1/proof.json",
                "e4105ae4824cfd544470fd51e8fcc099813de6e9b303bb9d31633070a2a43543",
            ),
            "verifier": binding(
                "tools/verify_qk_bfp8_e16_head64_v1_c02_score_path_evaluation_package.py",
                "701bdf27b8a412956ed8b1fb2a34edde8a6a5dff1aedc68010f0dd69ba0beb49",
            ),
        },
        "external_authority": {
            "checksum_manifest": binding(
                "evidence/verification/qk-bfp8-e16-head64-v1-c02-score-path-external-exactly-once-authority-v1/SHA256SUMS",
                "5fe93639f8988f451be8caedaadda641fcc46ef47b6b4333ffaf6cfcde8aa43a",
            ),
            "package": binding(
                "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EXTERNAL_EXACTLY_ONCE_AUTHORITY_PACKAGE.json",
                "3f31da79730ce25040148365091c902fa8ce611b0499d9137a6a053fdbf7285c",
            ),
            "preflight_review": binding(
                "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/51a7d3bc902e/round-0001.json",
                "89ef3ebee0b07b559fa39c03a0dc84ea81a077fe8ba9d04eabe1fc4e387a2a84",
            ),
            "qualifying_review": binding(
                "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/8c1b4f7e2a96/round-0001.json",
                "88fd0fbc620e2978ba23db223276764e348eda96e37da7a1b39d74c116d2ffbb",
            ),
            "verifier": binding(
                "tools/verify_qk_bfp8_e16_head64_v1_c02_score_path_external_exactly_once_authority.py",
                "401852322e5b4d421c97e4579bb29a069c49995659eba78c9317df37a373f4aa",
            ),
        },
        "normalization": {
            "contract": binding(
                "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_CONTRACT.json",
                "539ff95d8f1ad253880c9999fa59e6ac41840761908b188cf2c4bce09f3c1b01",
            ),
            "pure_reference": binding(
                "reference/qk_bfp8_e16_head64_v1_score_normalization.py",
                "8b1db0b13ee3a4534848814e75d13cea9946e380cd09b8bec0e76a527df098a3",
            ),
            "readiness": binding(
                "reference/QK_BFP8_E16_HEAD64_V1_SCORE_NORMALIZATION_READINESS.json",
                "3fb8229b5b7940d9667b4220356b2f10a8532898f9e69a6839704e7f1001fea3",
            ),
            "verifier": binding(
                "tools/verify_qk_bfp8_e16_head64_v1_score_normalization.py",
                "1aaddcaa74045c3de83b833af016a25c492c878b1120ed9cf557ec548a89c5af",
            ),
        },
        "representation": {
            "contract": binding(
                "design/W4A8_C02_SATURATION_FREE_JOINT_QK_REPRESENTATION_PREREQUISITE_V1.json",
                "92840df94b6f9bc2b1bfc752fd4e24da58cde64f811b2fd84190462badd1a4ef",
            ),
            "pure_reference": binding(
                "reference/qk_bfp8_e16_head64_v1.py",
                "bfb8b0d95ef8a8b939694fe8fec710d4acb1abff8b38b4bbbb4a43bdea5bb704",
            ),
            "readiness": binding(
                "reference/QK_BFP8_E16_HEAD64_V1_READINESS.json",
                "400d4a4a8b627cd71e930f670852b89e5deead83f14be3f528e752f6d1705fa7",
            ),
            "verifier": binding(
                "tools/verify_qk_bfp8_e16_head64_v1_readiness.py",
                "ea40e12f39351fe6240d861a68488a4bc22517dab72434762f05a8bdf88fdd2e",
            ),
        },
        "v1_implementation": {
            "checksum_manifest": binding(
                "evidence/verification/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v1/SHA256SUMS",
                "4d6fcf2e32c4e2b40c260442055d3800fa8cc32118d2a3caa2a7fe9d14d9c4f8",
            ),
            "handoff_round_0002": binding(
                "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/c46e72ad9185/round-0002.json",
                "2ddd6326682bd225bb4b5c81ce07c62e7daf87fa0605f3bb8fe0a4cf4fc07f75",
            ),
            "implementation": binding(
                "tools/qk_bfp8_e16_head64_v1_c02_live_authority_surface.py",
                "4e6a07c6e352e59eea7ec4a38223a396210c6a72154dea5e97e35b3c3cd8c0c9",
            ),
            "package": binding(
                "reference/QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_SURFACE_IMPLEMENTATION_PACKAGE.json",
                "af1d8673d85308351e059625acb726a9dc6b41c442fba571382487d0da2599ff",
            ),
            "proof": binding(
                "evidence/verification/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v1/static_synthetic_proof.json",
                "7148719202d00b415dcb26e1d185d9bc28741f1de0600cd4bdee000dbcbe1d9a",
            ),
            "verifier": binding(
                "tools/verify_qk_bfp8_e16_head64_v1_c02_live_authority_surface.py",
                "bc07d5484da0059e07d18c253234044f99763f8b48c917a61b674c8b225472c5",
            ),
        },
    }


def build_amendment() -> dict[str, Any]:
    if platform.python_implementation() != "CPython" or platform.python_version() != "3.13.5":
        raise RuntimeError("package assembly requires CPython 3.13.5")
    if Path(sys.executable).resolve() != INTERPRETER:
        raise RuntimeError("package assembly interpreter identity mismatch")
    evaluator = binding(EVALUATOR_REL)
    lanes: dict[str, Any] = {}
    for label, source in LANES.items():
        argv = argv_for(source)
        lanes[label] = {
            "accepted_invocation_sha256": source["accepted_invocation_sha256"],
            "argv": argv,
            "authoritative_tensors_sha256": source["authoritative_tensors_sha256"],
            "environment": ENVIRONMENT,
            "generation_id": source["generation_id"],
            "historical_logical_command_descriptor_sha256": source[
                "historical_logical_command_descriptor_sha256"
            ],
            "input_metadata_sha256": source["input_metadata_sha256"],
            "input_tensor_bundle_sha256": source["input_tensor_bundle_sha256"],
            "materialized_argv_sha256": object_sha256(argv),
            "model_identity_sha256": source["model_identity_sha256"],
            "namespace_label": source["namespace_label"],
            "output_path": source["output_path"],
            "record_id": source["record_id"],
            "sealed_input_paths": {
                "metadata": source["metadata_path"],
                "tensor_bundle": source["tensor_path"],
            },
            "tensor_record_sha256": source["tensor_records"],
        }
    return {
        "accepted_bindings": accepted_bindings(),
        "amendment_id": "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_AUTHORITATIVE_AMENDMENT_V2",
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_score_path_authoritative_amendment_v2",
        "authority_state": "STATIC_FROZEN_NO_LIVE_CREDENTIAL",
        "claim_boundary": {
            "accepted_invocation_execution_count": 0,
            "authority_materialized": False,
            "candidate_created": False,
            "live_lane_namespace_created": False,
            "model_or_sealed_tensor_access_count": 0,
            "ready_record_created": False,
            "status": "NO_EXECUTION_PERFORMED",
        },
        "execution_package": {
            "evaluator": {
                "candidate_free": True,
                "evaluator_id": "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATOR_SPEC_V1",
                "evaluator_spec_sha256": "b719d18f571e664b25c926ecb972f9972f2a13c4b8013d89f8bf970cf8a99756",
                **evaluator,
            },
            "interpreter": {
                "implementation": "CPython",
                "path": str(INTERPRETER),
                "sha256": sha256_file(INTERPRETER),
                "version": "3.13.5",
            },
            "lanes": lanes,
            "minimal_environment": ENVIRONMENT,
        },
        "exactly_once": {
            "authority_cardinality": "exactly one immutable single-use record per lane",
            "crash_after_consumption": "CONSUMED without first-terminal is an immutable orphan; no retry, replay, resume, repair, or replacement",
            "durability": "O_EXCL create-only records; fsync file then parent; consumed ledger atomic replacement and fsync parent",
            "first_terminal": "the first terminal record is create-only, self-checksummed, and immutable",
            "ordering": ["Base", "checkpoint-176"],
            "transition": "READY_UNCONSUMED -> CONSUMED must durably commit before launch and therefore before tensor access",
        },
        "output_semantics": {
            "Base": "the shared output root may be absent or an empty plain directory; only Base's own immutable result namespace must be absent; checkpoint and unrelated entries fail closed",
            "checkpoint-176": "the shared output root and Base namespace must remain; Base ledger, self-checksummed SUCCEEDED terminal, result file SHA-256, and result self-hash must validate; only checkpoint-176's own namespace must be absent",
            "preservation": "Base records and results may never be deleted, relocated, overwritten, substituted, or treated as a checkpoint collision",
            "rejection": "symlinks, path escapes, aliases, unexpected entries, partial namespaces, or unrelated entries fail closed",
        },
        "prohibited_actions": [
            "materialize a live credential or READY record in this amendment package",
            "execute either accepted invocation during static verification",
            "load a model or accepted sealed c02 tensor during static verification",
            "create a candidate, calibrate, generate, replay, repair, retry, or resume",
            "edit or supersede any accepted V1 byte",
            "edit RTL, specifications, manifests, U280 state, constraints, or stage state",
        ],
        "schema_version": 2,
        "schemas": {
            "ledger": {
                "id": "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_LEDGER_V2",
                "states": ["CONSUMED", "INVALIDATED_TERMINAL"],
            },
            "ready": {
                "id": "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_READY_V2",
                "state": "READY_UNCONSUMED",
            },
            "terminal": {
                "id": "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_TERMINAL_V2",
                "states": ["FAILED_TERMINAL", "SUCCEEDED_TERMINAL"],
            },
        },
        "supersession": {
            "historical_records_remain_authoritative_for_their_time": True,
            "v1_package_sha256": "af1d8673d85308351e059625acb726a9dc6b41c442fba571382487d0da2599ff",
            "v1_status": "SUPERSEDED_HISTORICAL_RECORD_PRESERVED_BYTE_FOR_BYTE",
        },
    }


def build_package() -> dict[str, Any]:
    amendment = binding(AMENDMENT_REL)
    return {
        "amendment": amendment,
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_live_authority_surface_implementation_package_v2",
        "claim_boundary": {
            "accepted_invocation_execution_count": 0,
            "authority_materialized": False,
            "live_namespace_created": False,
            "model_or_sealed_tensor_access_count": 0,
            "status": "STATIC_ONLY_NO_EXECUTION_PERFORMED",
        },
        "implementation": binding(SURFACE_REL),
        "package_id": "QK_BFP8_E16_HEAD64_V1_C02_LIVE_AUTHORITY_SURFACE_IMPLEMENTATION_PACKAGE_V2",
        "schema_version": 2,
        "static_verification": {
            "builder": binding(BUILDER_REL),
            "checksum_manifest_path": "evidence/verification/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2/SHA256SUMS",
            "proof_path": "evidence/verification/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2/static_synthetic_proof.json",
            "review_request_path": "evidence/verification/qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2/fresh_l2_review_request.json",
            "verifier": binding(VERIFIER_REL),
        },
        "status": "FROZEN_IMPLEMENTATION_NO_LIVE_AUTHORITY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", required=True, choices=("amendment", "package"))
    args = parser.parse_args()
    if args.phase == "amendment":
        write_static(ROOT / AMENDMENT_REL, build_amendment())
        print(sha256_file(ROOT / AMENDMENT_REL))
    else:
        write_static(ROOT / PACKAGE_REL, build_package())
        print(sha256_file(ROOT / PACKAGE_REL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
