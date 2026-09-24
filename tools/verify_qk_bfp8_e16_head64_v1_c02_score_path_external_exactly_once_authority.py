#!/usr/bin/env python3
"""Fail-closed static verifier for the external exactly-once authority freeze."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_REL = (
    "reference/"
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EXTERNAL_EXACTLY_ONCE_AUTHORITY_PACKAGE.json"
)
EVIDENCE_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-score-path-external-exactly-once-authority-v1/"
    "fresh_l2_review_evidence.json"
)
CHECKSUMS_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-score-path-external-exactly-once-authority-v1/"
    "SHA256SUMS"
)
ACCEPTED_PACKAGE_REL = (
    "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json"
)
ACCEPTED_PROOF_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-score-path-evaluation-package-v1/proof.json"
)
ACCEPTED_VERIFIER_REL = (
    "tools/verify_qk_bfp8_e16_head64_v1_c02_score_path_evaluation_package.py"
)
REVIEW_SOURCE = (
    Path.home()
    / ".argus-skill-ace2/projects/s-c8ae985b/handoffs/ae13612d8866/round-0003.json"
)

PACKAGE = ROOT / PACKAGE_REL
EVIDENCE = ROOT / EVIDENCE_REL
CHECKSUMS = ROOT / CHECKSUMS_REL
VERIFIER = Path(__file__).resolve()

ACCEPTED_PACKAGE_SHA256 = "0bd51407f20950a93f47a520718a3bb4d5a5006d990d60aef626d16a5f9d976d"
ACCEPTED_VERIFIER_SHA256 = "701bdf27b8a412956ed8b1fb2a34edde8a6a5dff1aedc68010f0dd69ba0beb49"
ACCEPTED_PROOF_SHA256 = "e4105ae4824cfd544470fd51e8fcc099813de6e9b303bb9d31633070a2a43543"
QUALIFYING_REVIEW_SHA256 = "873441813217c70b41a9ba6554b3f7c862d12c6b81571f4bfee38df302943c70"
EVALUATOR_ID = "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATOR_SPEC_V1"
EVALUATOR_SPEC_SHA256 = "b719d18f571e664b25c926ecb972f9972f2a13c4b8013d89f8bf970cf8a99756"
OUTPUT_ROOT = "build/qk-bfp8-e16-head64-v1-c02-score-path-evaluation-v1"
AUTHORITY_PACKAGE_ID = (
    "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EXTERNAL_EXACTLY_ONCE_AUTHORITY_V1"
)

ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}

EXPECTED_LANES = [
    {
        "authoritative_tensors_sha256": "aa0d327f251e86cdbb5c00b4a2498bdfa4bbb5b167d55e2f370c1dd83a2285bd",
        "generation_id": "c02-score-path-eval-base-v1",
        "input_metadata_sha256": "4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a",
        "input_tensor_sha256": "7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175",
        "invocation_sha256": "58b90f06f10d179cb12820ee5a376cf3807eb9f2af929990dacac138a8fa6731",
        "label": "Base",
        "model_identity_sha256": "4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7",
        "namespace_label": "base",
        "output_path": OUTPUT_ROOT + "/base/result.json",
        "record_id": "c02-score-path-base-exactly-once-v1",
        "tensor_records": {
            "bf16_oracle_scores": "49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765",
            "realized_key_source": "401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a",
            "realized_query_source": "285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429",
        },
    },
    {
        "authoritative_tensors_sha256": "5af500371148338636f078e0ec69973666f54f70d3c0f19d350456cf74c5bdb2",
        "generation_id": "c02-score-path-eval-checkpoint-176-v1",
        "input_metadata_sha256": "3f73b5359e021ddb6183ec468ae154b3cda0807365b248773d80466e477b4e86",
        "input_tensor_sha256": "07743a5df6bfa00d5e977dc622b70e90af9b6789feaeb5ae689448ac803ceb59",
        "invocation_sha256": "78e83c2f2ad2a5ec15a96dc320bd3e86f5ed3fbedbf3de9871a03b40225c829b",
        "label": "checkpoint-176",
        "model_identity_sha256": "94eb1fe142aa11925863386e796e1157341351cfb664283388291a4fa9d8bf98",
        "namespace_label": "checkpoint-176",
        "output_path": OUTPUT_ROOT + "/checkpoint-176/result.json",
        "record_id": "c02-score-path-checkpoint-176-exactly-once-v1",
        "tensor_records": {
            "bf16_oracle_scores": "776be2e58246c8641fa483e331e0d63896f96f14b3756548e15af3870d89c456",
            "realized_key_source": "2c4b99f5bc97d0f95b886dfebf62242148940d7cc9021e197dfdee0aed9e9902",
            "realized_query_source": "18671805abe3835d8f2e23bffb041ced6a05f41b8eac4f698f41d302def73b18",
        },
    },
]


class VerificationError(RuntimeError):
    """Static authority-package verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
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


def object_sha256(value: Any) -> str:
    return hashlib.sha256(compact_bytes(value)).hexdigest()


def pretty_text(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def load_canonical_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON artifact: {path}")
    require(not path.is_symlink(), f"symlinked JSON artifact: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON artifact: {path}: {exc}") from exc
    require(text == pretty_text(value), f"noncanonical JSON artifact: {path}")
    return value


def verify_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(not path.is_symlink(), f"symlinked {label}: {path}")
    require(sha256_file(path) == expected, f"{label} SHA-256 mismatch")


def verify_no_execution_surface() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {
        "cocotb",
        "numpy",
        "onnxruntime",
        "subprocess",
        "tensorflow",
        "torch",
        "transformers",
    }
    forbidden_calls = {
        "Popen",
        "call",
        "eval",
        "exec",
        "from_pretrained",
        "generate",
        "popen",
        "run",
        "system",
    }
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    require(not imported.intersection(forbidden_modules), "execution/model module imported")
    require(not called.intersection(forbidden_calls), "prohibited execution call present")


def expected_command(lane: dict[str, Any]) -> dict[str, Any]:
    descriptor = {
        "arguments": {
            "accepted_evaluation_package_sha256": ACCEPTED_PACKAGE_SHA256,
            "invocation_sha256": lane["invocation_sha256"],
            "lane_label": lane["label"],
            "output_path": lane["output_path"],
        },
        "command_id": "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATE_ONCE_V1",
        "encoding": "canonical-json-external-command-descriptor-v1",
        "executor_operation": "evaluate_score_path_once",
        "shell_argv_materialized": False,
    }
    return {"command_sha256": object_sha256(descriptor), "descriptor": descriptor}


def verify_accepted_artifacts() -> dict[str, Any]:
    accepted_package_path = ROOT / ACCEPTED_PACKAGE_REL
    accepted_proof_path = ROOT / ACCEPTED_PROOF_REL
    accepted_verifier_path = ROOT / ACCEPTED_VERIFIER_REL
    verify_hash(accepted_package_path, ACCEPTED_PACKAGE_SHA256, "accepted package")
    verify_hash(accepted_verifier_path, ACCEPTED_VERIFIER_SHA256, "accepted verifier")
    verify_hash(accepted_proof_path, ACCEPTED_PROOF_SHA256, "accepted proof")
    verify_hash(REVIEW_SOURCE, QUALIFYING_REVIEW_SHA256, "qualifying Fresh-L2 review")

    accepted = load_canonical_json(accepted_package_path)
    proof = load_canonical_json(accepted_proof_path)
    review = load_canonical_json(REVIEW_SOURCE)
    require(accepted.get("authority") == "NO_EXECUTION_AUTHORITY", "accepted package authority")
    require(accepted.get("package_verifier", {}).get("sha256") == ACCEPTED_VERIFIER_SHA256, "accepted verifier binding")
    require(proof.get("package", {}).get("sha256") == ACCEPTED_PACKAGE_SHA256, "accepted proof package binding")
    require(proof.get("verifier", {}).get("sha256") == ACCEPTED_VERIFIER_SHA256, "accepted proof verifier binding")
    require(proof.get("static_observations", {}).get("invocation_count_performed") == 0, "accepted proof execution count")
    require(review.get("kind") == "round_reviewed_handoff", "Fresh-L2 review kind")
    require(review.get("mission_id") == "ae13612d8866", "Fresh-L2 review mission")
    require(review.get("producer_role") == "reviewer", "Fresh-L2 producer role")
    require(review.get("round") == 3, "Fresh-L2 round")
    require(review.get("review", {}).get("status") == "done", "Fresh-L2 status")

    invocations = accepted.get("invocations")
    require(isinstance(invocations, list) and len(invocations) == 2, "accepted invocation count")
    for lane, invocation in zip(EXPECTED_LANES, invocations, strict=True):
        descriptor = invocation.get("descriptor", {})
        require(invocation.get("invocation_sha256") == lane["invocation_sha256"], f"{lane['label']} invocation hash")
        require(object_sha256(descriptor) == lane["invocation_sha256"], f"{lane['label']} descriptor hash")
        require(descriptor.get("authoritative_tensors_sha256") == lane["authoritative_tensors_sha256"], f"{lane['label']} authoritative tensor map")
        require(descriptor.get("environment") == ENVIRONMENT, f"{lane['label']} accepted environment")
        require(descriptor.get("generation_id") == lane["generation_id"], f"{lane['label']} generation")
        require(descriptor.get("input_metadata_sha256") == lane["input_metadata_sha256"], f"{lane['label']} metadata")
        require(descriptor.get("input_tensor_sha256") == lane["input_tensor_sha256"], f"{lane['label']} tensor bundle")
        require(descriptor.get("invocation_cardinality") == 1, f"{lane['label']} invocation cardinality")
        require(descriptor.get("lane_label") == lane["label"], f"{lane['label']} label")
        require(descriptor.get("model_identity_sha256") == lane["model_identity_sha256"], f"{lane['label']} model")
        require(descriptor.get("namespace_label") == lane["namespace_label"], f"{lane['label']} namespace")
        require(descriptor.get("output_path") == lane["output_path"], f"{lane['label']} output")
        require(descriptor.get("terminal_no_retry_replay_resume_repair") is True, f"{lane['label']} terminal policy")
        require(descriptor.get("toolchain") == {"implementation": "CPython", "version": "3.13.5"}, f"{lane['label']} toolchain")

        accepted_lane = next(item for item in accepted["input_set"]["lanes"] if item["label"] == lane["label"])
        require(accepted_lane["model_identity_sha256"] == lane["model_identity_sha256"], f"{lane['label']} accepted lane model")
        require(accepted_lane["metadata"]["sha256"] == lane["input_metadata_sha256"], f"{lane['label']} accepted metadata")
        require(accepted_lane["tensor_bundle"]["sha256"] == lane["input_tensor_sha256"], f"{lane['label']} accepted tensor bundle")
        actual_records = {
            role: record["sha256"]
            for role, record in accepted_lane["authoritative_tensors"].items()
        }
        require(actual_records == lane["tensor_records"], f"{lane['label']} tensor records")
    return accepted


def verify_review_evidence() -> dict[str, Any]:
    evidence = load_canonical_json(EVIDENCE)
    require(evidence == {
        "accepted_artifacts": {
            "evaluation_package": {"path": ACCEPTED_PACKAGE_REL, "sha256": ACCEPTED_PACKAGE_SHA256},
            "proof": {"path": ACCEPTED_PROOF_REL, "sha256": ACCEPTED_PROOF_SHA256},
            "static_verifier": {"path": ACCEPTED_VERIFIER_REL, "sha256": ACCEPTED_VERIFIER_SHA256},
        },
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_score_path_qualifying_fresh_l2_review_evidence",
        "claim_boundary": {
            "authority_consumed": False,
            "authority_materialized": False,
            "execution_namespace_created": False,
            "fixed_input_payload_deserialized": False,
            "invocation_count_performed": 0,
            "model_loaded_or_executed": False,
            "repair_replay_resume_or_retry_performed": False,
            "score_computation_performed": False,
            "status": "NO_EXECUTION_PERFORMED",
        },
        "qualifying_review": {
            "kind": "round_reviewed_handoff",
            "mission_id": "ae13612d8866",
            "producer_role": "reviewer",
            "review_status": "done",
            "round": 3,
            "sha256": QUALIFYING_REVIEW_SHA256,
            "source_locator": "argus-handoff:ae13612d8866:round-0003",
        },
        "reviewed_observations": {
            "accepted_package_sha256": ACCEPTED_PACKAGE_SHA256,
            "accepted_verifier_result": "STATIC_VALID_NO_EXECUTION_AUTHORITY",
            "execution_namespace_absent_before_and_after": True,
            "negative_rejections": 19,
            "review_scope": "checksum-bound package, exact sealed lane/input bindings, deterministic schema, distinct invocation hashes, lane isolation, terminal no-retry semantics, and NO_EXECUTION_AUTHORITY",
        },
        "schema_version": 1,
    }, "Fresh-L2 review evidence mismatch")
    return evidence


def verify_lane(package_lane: dict[str, Any], expected: dict[str, Any]) -> None:
    require(set(package_lane) == {
        "accepted_invocation",
        "authority_record",
        "command",
        "environment",
        "generation_id",
        "lane_label",
        "model_identity_sha256",
        "namespace_label",
        "output_identity",
        "sealed_inputs",
    }, f"{expected['label']} lane keys")
    require(package_lane["lane_label"] == expected["label"], f"{expected['label']} lane order")
    require(package_lane["accepted_invocation"] == {
        "descriptor_sha256": expected["invocation_sha256"],
        "invocation_sha256": expected["invocation_sha256"],
    }, f"{expected['label']} invocation binding")
    require(package_lane["authority_record"] == {
        "authority_cardinality": 1,
        "live_credential": None,
        "materialization_state": "NOT_MATERIALIZED",
        "record_id": expected["record_id"],
        "single_use": True,
    }, f"{expected['label']} authority record")
    require(package_lane["command"] == expected_command(expected), f"{expected['label']} command")
    require(package_lane["environment"] == ENVIRONMENT, f"{expected['label']} environment")
    require(package_lane["generation_id"] == expected["generation_id"], f"{expected['label']} generation")
    require(package_lane["model_identity_sha256"] == expected["model_identity_sha256"], f"{expected['label']} model")
    require(package_lane["namespace_label"] == expected["namespace_label"], f"{expected['label']} namespace")
    require(package_lane["output_identity"] == {
        "output_path": expected["output_path"],
        "result_schema_id": "QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_RESULT_V1",
        "separate_from_other_lane": True,
    }, f"{expected['label']} output identity")
    require(package_lane["sealed_inputs"] == {
        "authoritative_tensor_records_sha256": expected["tensor_records"],
        "authoritative_tensors_sha256": expected["authoritative_tensors_sha256"],
        "input_metadata_sha256": expected["input_metadata_sha256"],
        "input_tensor_bundle_sha256": expected["input_tensor_sha256"],
    }, f"{expected['label']} sealed inputs")


def verify_package(package: dict[str, Any]) -> None:
    require(set(package) == {
        "accepted_evaluation",
        "artifact_kind",
        "authority_package_id",
        "authority_state",
        "claim_boundary",
        "exactly_once_semantics",
        "fail_closed_preflight",
        "implementation_identity",
        "lanes",
        "ordering",
        "prohibited_actions",
        "schema_version",
        "static_verifier",
        "terminal_record_contract",
    }, "package keys")
    require(package["artifact_kind"] == "qk_bfp8_e16_head64_v1_c02_score_path_external_exactly_once_execution_authority_package", "artifact kind")
    require(package["authority_package_id"] == AUTHORITY_PACKAGE_ID, "authority package id")
    require(package["authority_state"] == "STATIC_FROZEN_EXTERNAL_AUTHORITY_NO_LIVE_CREDENTIAL", "authority state")
    require(package["schema_version"] == 1, "schema version")
    require(package["claim_boundary"] == {
        "authority_consumed": False,
        "authority_materialized": False,
        "calibration_or_generation_performed": False,
        "execution_namespace_created": False,
        "fixed_input_payload_deserialized": False,
        "invocation_count_performed": 0,
        "model_loaded_or_executed": False,
        "repair_replay_resume_or_retry_performed": False,
        "rtl_spec_manifest_u280_or_stage_state_modified": False,
        "score_computation_performed": False,
        "status": "NO_EXECUTION_PERFORMED",
    }, "claim boundary")
    require(package["accepted_evaluation"] == {
        "evaluation_package": {"path": ACCEPTED_PACKAGE_REL, "sha256": ACCEPTED_PACKAGE_SHA256},
        "proof": {"path": ACCEPTED_PROOF_REL, "sha256": ACCEPTED_PROOF_SHA256},
        "qualifying_fresh_l2_review": {
            "evidence_path": EVIDENCE_REL,
            "evidence_sha256": sha256_file(EVIDENCE),
            "source_review_sha256": QUALIFYING_REVIEW_SHA256,
        },
        "static_verifier": {"path": ACCEPTED_VERIFIER_REL, "sha256": ACCEPTED_VERIFIER_SHA256},
    }, "accepted evaluation binding")
    require(package["implementation_identity"] == {
        "evaluator_id": EVALUATOR_ID,
        "evaluator_spec_sha256": EVALUATOR_SPEC_SHA256,
        "executable_implementation_included": False,
        "identity_kind": "CHECKSUM_BOUND_EVALUATION_CONTRACT",
        "preflight_requirement": "external implementation must attest this exact evaluator id and spec SHA-256 before authority consumption",
    }, "implementation identity")
    require(package["static_verifier"] == {
        "implementation": "CPython",
        "path": str(VERIFIER.relative_to(ROOT)),
        "python_version": "3.13.5",
        "sha256": sha256_file(VERIFIER),
    }, "static verifier identity")
    require(package["ordering"] == {
        "lane_order": ["Base", "checkpoint-176"],
        "policy": "Base first terminal record must be durably sealed before checkpoint-176 may consume authority",
        "publication_order_must_match_lane_order": True,
    }, "ordering")

    lanes = package["lanes"]
    require(isinstance(lanes, list) and len(lanes) == 2, "lane count")
    for package_lane, expected in zip(lanes, EXPECTED_LANES, strict=True):
        verify_lane(package_lane, expected)
    require(lanes[0]["output_identity"]["output_path"] != lanes[1]["output_identity"]["output_path"], "output alias")
    require(lanes[0]["authority_record"]["record_id"] != lanes[1]["authority_record"]["record_id"], "authority record alias")

    require(package["exactly_once_semantics"] == {
        "atomic_consume_before_execute": "after all static preflight checks pass, the external authority ledger must atomically transition the selected lane from READY_UNCONSUMED to CONSUMED and durably commit that transition before namespace creation, implementation or model loading, tensor deserialization, or score computation",
        "authority_cardinality": "exactly one independently isolated single-use record per lane; records and outcomes are not substitutable across lanes",
        "consume_failure": "if the atomic transition does not durably commit, execute nothing and atomically seal FAILED_TERMINAL for that lane",
        "first_terminal_record_immutability": "the first successfully inserted terminal record is final, create-only, and may never be amended, replaced, deleted, or superseded",
        "post_consume_failure": "any failure after durable consumption seals FAILED_TERMINAL; consumed authority is never restored",
        "preflight_failure": "any mismatch atomically invalidates the lane record and seals FAILED_TERMINAL without creating an execution namespace or performing execution",
        "prohibited_under_every_terminal_outcome": [
            "retry",
            "replay",
            "resume",
            "repair",
            "recalibration",
            "regeneration",
            "authority reuse",
            "partial-result substitution",
        ],
        "terminal_states": ["SUCCEEDED_TERMINAL", "FAILED_TERMINAL"],
    }, "exactly-once semantics")
    require(package["fail_closed_preflight"] == {
        "checks_in_order": [
            "authority package and checksum manifest are canonical and checksum-valid",
            "accepted package, accepted verifier, accepted proof, and qualifying Fresh-L2 review hashes match",
            "lane order, lane identity, accepted invocation descriptor hash, sealed inputs, model identity, command, environment, and output identity match",
            "the lane has exactly one matching READY_UNCONSUMED external record and no terminal record",
            "the prior lane ordering prerequisite is satisfied",
            "the declared output root and both lane output paths are absent",
            "the external implementation attests the exact evaluator id and evaluator spec SHA-256",
        ],
        "mismatch_result": "seal FAILED_TERMINAL and perform zero execution",
        "no_side_effects_before_consume": [
            "no execution namespace creation",
            "no model or implementation loading",
            "no tensor deserialization",
            "no score computation",
        ],
    }, "fail-closed preflight")
    require(package["terminal_record_contract"] == {
        "failure_record_required": True,
        "fields": [
            "authority_package_sha256",
            "record_id",
            "lane_label",
            "invocation_sha256",
            "command_sha256",
            "consumption_committed",
            "execution_started",
            "status",
            "reason_code",
            "first_record_immutable",
            "retry_replay_resume_repair_permitted",
        ],
        "first_record_immutable": True,
        "retry_replay_resume_repair_permitted": False,
        "write_mode": "atomic create-if-absent",
    }, "terminal record contract")
    require(package["prohibited_actions"] == [
        "materialize or consume live execution authority during package production or static verification",
        "create the declared execution namespace or either lane output path",
        "deserialize sealed tensor payloads",
        "load or execute a model or evaluator implementation",
        "compute scores, calibrate, or generate",
        "retry, replay, resume, repair, recalibrate, regenerate, or reuse authority",
        "modify RTL, specifications, manifests, constraints, U280 configuration, or stage state",
    ], "prohibited actions")


def verify_checksums() -> None:
    expected = "".join(
        f"{sha256_file(path)}  {relative}\n"
        for relative, path in [
            (PACKAGE_REL, PACKAGE),
            (str(VERIFIER.relative_to(ROOT)), VERIFIER),
            (EVIDENCE_REL, EVIDENCE),
        ]
    )
    require(CHECKSUMS.is_file(), "missing checksum manifest")
    require(not CHECKSUMS.is_symlink(), "symlinked checksum manifest")
    require(CHECKSUMS.read_text(encoding="ascii") == expected, "checksum manifest mismatch")


def verify_namespace_absence() -> None:
    output_root = ROOT / OUTPUT_ROOT
    require(not output_root.exists(), "declared execution namespace exists")
    for lane in EXPECTED_LANES:
        require(not (ROOT / lane["output_path"]).exists(), f"{lane['label']} output exists")


def main() -> int:
    verify_no_execution_surface()
    require(platform.python_implementation() == "CPython", "Python implementation mismatch")
    require(platform.python_version() == "3.13.5", "Python version mismatch")
    verify_namespace_absence()
    verify_accepted_artifacts()
    verify_review_evidence()
    package = load_canonical_json(PACKAGE)
    verify_package(package)
    verify_checksums()
    verify_namespace_absence()
    print(
        "STATIC_VALID_NO_EXECUTION_PERFORMED "
        f"authority_package_sha256={sha256_file(PACKAGE)} "
        "lanes=2 authority_cardinality_per_lane=1 execution_namespaces=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"STATIC_INVALID_NO_EXECUTION_PERFORMED: {exc}", file=sys.stderr)
        raise SystemExit(1)
