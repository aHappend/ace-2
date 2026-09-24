#!/usr/bin/env python3
"""Static synthetic verifier for the checksum-bound V2 QK authority surface."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import platform
import struct
import tempfile
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in os.sys.path:
    os.sys.path.insert(0, str(TOOLS))

import build_qk_bfp8_e16_head64_v1_c02_authoritative_amendment_v2 as builder
import qk_bfp8_e16_head64_v1_c02_live_authority_surface_v2 as core
import qk_bfp8_e16_head64_v1_c02_score_path_evaluator_v2 as evaluator


AMENDMENT_REL = builder.AMENDMENT_REL
PACKAGE_REL = builder.PACKAGE_REL
EVALUATOR_REL = builder.EVALUATOR_REL
SURFACE_REL = builder.SURFACE_REL
VERIFIER_REL = builder.VERIFIER_REL
BUILDER_REL = builder.BUILDER_REL
EVIDENCE_REL = (
    "evidence/verification/"
    "qk-bfp8-e16-head64-v1-c02-live-authority-surface-v2"
)
PROOF_REL = EVIDENCE_REL + "/static_synthetic_proof.json"
REVIEW_REL = EVIDENCE_REL + "/fresh_l2_review_request.json"
CHECKSUMS_REL = EVIDENCE_REL + "/SHA256SUMS"


class VerificationError(RuntimeError):
    """Static package verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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


def compact_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load_pretty(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked JSON: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    require(raw == pretty_bytes(value), f"noncanonical JSON: {path}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def expect_surface_error(callback: Any, label: str) -> None:
    try:
        callback()
    except core.SurfaceError:
        return
    raise VerificationError(f"mutation did not fail closed: {label}")


def expect_evaluation_error(callback: Any, label: str, reason_code: str) -> None:
    try:
        callback()
    except evaluator.EvaluationError as exc:
        require(exc.reason_code == reason_code, f"{label} reason code")
        return
    raise VerificationError(f"evaluator mutation did not fail closed: {label}")


def verify_exact_packages() -> tuple[dict[str, Any], dict[str, Any]]:
    amendment = load_pretty(ROOT / AMENDMENT_REL)
    package = load_pretty(ROOT / PACKAGE_REL)
    require(amendment == builder.build_amendment(), "amendment exact-byte model mismatch")
    require(package == builder.build_package(), "implementation package exact-byte model mismatch")
    require(sha256_file(ROOT / AMENDMENT_REL) == core.AMENDMENT_SHA256, "core amendment binding")
    core.verify_bound_artifacts(amendment)
    for lane_label in ("Base", "checkpoint-176"):
        lane = core.validate_lane_descriptor(amendment, lane_label)
        require(
            lane["historical_logical_command_descriptor_sha256"]
            != lane["materialized_argv_sha256"],
            f"{lane_label} descriptor/argv separation",
        )
    return amendment, package


def verify_source_boundaries() -> None:
    evaluator_source = (ROOT / EVALUATOR_REL).read_text(encoding="utf-8")
    verifier_source = (ROOT / VERIFIER_REL).read_text(encoding="utf-8")
    evaluator_tree = ast.parse(evaluator_source)
    verifier_tree = ast.parse(verifier_source)
    forbidden_evaluator_imports = {
        "cocotb",
        "numpy",
        "onnxruntime",
        "subprocess",
        "tensorflow",
        "torch",
        "transformers",
    }
    forbidden_verifier_imports = forbidden_evaluator_imports
    forbidden_calls = {
        "Popen",
        "from_pretrained",
        "generate",
        "popen",
        "run",
        "system",
    }

    def imports(tree: ast.AST) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        return found

    def calls(tree: ast.AST) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    found.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    found.add(node.func.attr)
        return found

    require(not imports(evaluator_tree).intersection(forbidden_evaluator_imports), "evaluator imports model/execution stack")
    require(not imports(verifier_tree).intersection(forbidden_verifier_imports), "verifier imports model/execution stack")
    require(not calls(verifier_tree).intersection(forbidden_calls), "verifier contains process/model launch")
    require("torch.load" not in evaluator_source, "evaluator unsafe tensor loader")
    require("candidate" not in {node.id for node in ast.walk(evaluator_tree) if isinstance(node, ast.Name)}, "evaluator candidate symbol")


def synthetic_bundle_test() -> None:
    payload = struct.pack("<HH", 0x3F80, 0xBF80)
    dtype = "torch.bfloat16"
    shape = (1, 2)
    with tempfile.TemporaryDirectory(prefix="qk-v2-evaluator-synthetic-") as temporary:
        path = Path(temporary) / "synthetic.bin"
        with path.open("wb") as handle:
            handle.write(evaluator.TENSOR_BUNDLE_MAGIC)
            handle.write(struct.pack(">I", 1))
            name = b"synthetic.bf16"
            dtype_raw = dtype.encode("ascii")
            handle.write(struct.pack(">H", len(name)))
            handle.write(name)
            handle.write(struct.pack(">B", len(dtype_raw)))
            handle.write(dtype_raw)
            handle.write(struct.pack(">B", len(shape)))
            for dimension in shape:
                handle.write(struct.pack(">Q", dimension))
            handle.write(struct.pack(">Q", len(payload)))
            handle.write(payload)
        records = evaluator.read_tensor_bundle(path)
        record = records["synthetic.bf16"]
        require(record["shape"] == shape, "synthetic bundle shape")
        require(record["sha256"] == evaluator.tensor_record_sha256(dtype, shape, payload), "synthetic tensor hash")
        require(evaluator.bf16_word(record, (0, 0)) == 0x3F80, "synthetic BF16 word 0")
        require(evaluator.bf16_word(record, (0, 1)) == 0xBF80, "synthetic BF16 word 1")
        oracle, top, ties = evaluator.exact_oracle_row((0x3F80, 0x3F00, 0xBF80))
        require(top == 0 and ties == 1 and oracle[0] == 0, "synthetic exact oracle")


def synthetic_non_finite_rejection_tests() -> list[str]:
    role_shapes = {
        "bf16_oracle_scores": (1, 14, 41, 41),
        "realized_query_source": (1, 14, 41, 64),
        "realized_key_source": (1, 2, 41, 64),
    }

    def selected_with_mutation(
        role: str, word: int, indices: tuple[int, ...]
    ) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for record_role, shape in role_shapes.items():
            element_count = 1
            for dimension in shape:
                element_count *= dimension
            payload = bytearray(2 * element_count)
            if record_role == role:
                require(len(indices) == len(shape), f"{role} synthetic mutation rank")
                flat = 0
                for index, dimension in zip(indices, shape):
                    require(0 <= index < dimension, f"{role} synthetic mutation index")
                    flat = flat * dimension + index
                struct.pack_into("<H", payload, flat * 2, word)
            payload_bytes = bytes(payload)
            selected[record_role] = {
                "dtype": "torch.bfloat16",
                "payload": payload_bytes,
                "sha256": evaluator.tensor_record_sha256(
                    "torch.bfloat16", shape, payload_bytes
                ),
                "shape": shape,
            }
        return selected

    cases = (
        (
            "masked_oracle_qnan_future_key",
            "bf16_oracle_scores",
            0x7FC0,
            (0, 0, 0, 1),
        ),
        ("bf16_oracle_scores_infinity", "bf16_oracle_scores", 0x7F80, (0, 0, 0, 0)),
        ("realized_query_source_qnan", "realized_query_source", 0x7FC0, (0, 0, 0, 0)),
        ("realized_query_source_infinity", "realized_query_source", 0x7F80, (0, 0, 0, 0)),
        ("realized_key_source_qnan", "realized_key_source", 0x7FC0, (0, 0, 0, 0)),
        ("realized_key_source_infinity", "realized_key_source", 0x7F80, (0, 0, 0, 0)),
    )
    original_import = evaluator.import_bound_reference

    def reject_metrics_stage(*args: Any, **kwargs: Any) -> Any:
        raise VerificationError("non-finite record reached reference import/metrics stage")

    labels: list[str] = []
    try:
        evaluator.import_bound_reference = reject_metrics_stage
        for label, role, word, indices in cases:
            selected = selected_with_mutation(role, word, indices)
            expect_evaluation_error(
                lambda selected=selected: evaluator.evaluate_selected_records(selected, {}, {}),
                label,
                "NON_FINITE_ORACLE_REJECTED",
            )
            labels.append(label)
    finally:
        evaluator.import_bound_reference = original_import
    return labels


def synthetic_result(lane_label: str) -> dict[str, Any]:
    package = load_pretty(ROOT / "reference/QK_BFP8_E16_HEAD64_V1_C02_SCORE_PATH_EVALUATION_PACKAGE.json")
    lane = evaluator.selected_lane(package, lane_label)
    metrics = {
        "invalid_accounting": {
            "cross_lane_record_count": 0,
            "invalid_or_non_finite_value_count": 0,
            "normalization_rejection_count": 0,
            "positive_centered_realized_score_count": 0,
            "saturation_event_count": 0,
        },
        "rank_margin": {
            "minimum_realized_margin_q12_20_lsb": 1,
            "oracle_tied_row_count": 0,
            "preserved_positive_margin_fraction": {"denominator": 1, "numerator": 1},
            "preserved_positive_margin_row_count": 560,
            "singleton_valid_key_row_count": 14,
            "unique_oracle_top_row_count": 560,
            "violation_count": 0,
        },
        "score_error": {
            "maximum_absolute_error_q12_20_lsb": 0,
            "sum_absolute_error_q12_20_lsb": 0,
            "sum_signed_error_q12_20_lsb": 0,
            "sum_squared_error_q40_40_lsb2": 0,
            "valid_value_count": 12054,
        },
        "top_key": {
            "matching_fraction": {"denominator": 1, "numerator": 1},
            "matching_row_count": 574,
            "mismatch_count": 0,
            "row_count": 574,
        },
    }
    thresholds = evaluator.evaluate_thresholds(metrics, package["evaluation_contract"]["thresholds"])
    require(thresholds["all_hard_thresholds_pass"] is True, "synthetic thresholds")
    return evaluator.result_object(
        lane,
        package,
        metrics,
        thresholds,
        "HARD_THRESHOLDS_PASSED",
    )


def rehash_result(result: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(result)
    value.pop("result_sha256", None)
    value["result_sha256"] = core.object_sha256(value)
    return value


def publish_synthetic_result(
    root: Path, lane_label: str, result: dict[str, Any] | None = None
) -> Path:
    namespace = "base" if lane_label == "Base" else "checkpoint-176"
    output_root = root / core.OUTPUT_ROOT_REL
    if not output_root.exists():
        output_root.mkdir(mode=0o700)
    lane_root = output_root / namespace
    lane_root.mkdir(mode=0o700)
    path = lane_root / "result.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, compact_bytes(synthetic_result(lane_label) if result is None else result))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def synthetic_two_lane_sequence(amendment: dict[str, Any]) -> dict[str, Any]:
    original_root = core.ROOT
    original_backend = core.LAUNCH_BACKEND
    launches: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="qk-v2-surface-synthetic-") as temporary:
            root = Path(temporary)
            (root / "build").mkdir(mode=0o700)
            core.ROOT = root

            def intercepted(argv: tuple[str, ...], environment: dict[str, str]) -> Any:
                lane_label = argv[argv.index("--lane") + 1]
                expected = amendment["execution_package"]["lanes"][lane_label]
                require(list(argv) == expected["argv"], f"{lane_label} exact argv interception")
                require(environment == expected["environment"], f"{lane_label} exact environment interception")
                ledger = load_pretty(core.lane_paths(lane_label)["ledger"])
                require(ledger["state"] == core.CONSUMED_STATE, f"{lane_label} consumed before launch")
                result_path = publish_synthetic_result(root, lane_label)
                launches.append(
                    {
                        "argv_sha256": core.object_sha256(list(argv)),
                        "lane_label": lane_label,
                        "result_file_sha256": sha256_file(result_path),
                    }
                )
                return types.SimpleNamespace(returncode=0)

            core.LAUNCH_BACKEND = intercepted
            credential = "1" * 64
            core.materialize_lane("Base", credential)
            base_terminal = core.consume_and_run("Base")
            require(base_terminal["status"] == "SUCCEEDED_TERMINAL", "synthetic Base terminal")
            base_result = root / core.OUTPUT_ROOT_REL / "base" / "result.json"
            base_result_hash = sha256_file(base_result)
            base_terminal_hash = sha256_file(core.lane_paths("Base")["terminal"])
            core.materialize_lane("checkpoint-176", "2" * 64)
            checkpoint_terminal = core.consume_and_run("checkpoint-176")
            require(checkpoint_terminal["status"] == "SUCCEEDED_TERMINAL", "synthetic checkpoint terminal")
            require(sha256_file(base_result) == base_result_hash, "Base result changed during checkpoint")
            require(sha256_file(core.lane_paths("Base")["terminal"]) == base_terminal_hash, "Base terminal changed during checkpoint")
            require(
                {entry.name for entry in (root / core.OUTPUT_ROOT_REL).iterdir()}
                == {"base", "checkpoint-176"},
                "shared parent two-lane contents",
            )
            expect_surface_error(lambda: core.consume_and_run("Base"), "Base replay")
            return {
                "base_result_present_before_checkpoint": True,
                "base_result_preserved": True,
                "base_terminal_preserved": True,
                "launches": launches,
                "ordered_lane_labels": [item["lane_label"] for item in launches],
            }
    finally:
        core.ROOT = original_root
        core.LAUNCH_BACKEND = original_backend


def synthetic_fail_closed_paths(amendment: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    original_root = core.ROOT
    original_backend = core.LAUNCH_BACKEND

    def build_root(temporary: str) -> Path:
        root = Path(temporary)
        (root / "build").mkdir(mode=0o700)
        core.ROOT = root
        return root

    def reject_mutated_result(label: str, result: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory(prefix="qk-v2-result-mutation-") as temporary:
            root = build_root(temporary)
            path = publish_synthetic_result(root, "Base", rehash_result(result))
            lane = core.validate_lane_descriptor(amendment, "Base")
            expect_surface_error(
                lambda: core.validate_result(path, amendment, "Base", lane),
                label,
            )
        labels.append(label)

    try:
        with tempfile.TemporaryDirectory(prefix="qk-v2-symlink-") as temporary:
            root = build_root(temporary)
            target = root / "target"
            target.mkdir()
            (root / core.OUTPUT_ROOT_REL).symlink_to(target, target_is_directory=True)
            expect_surface_error(lambda: core.ensure_output_state(amendment, "Base"), "symlink output root")
            labels.append("symlink_output_root")
        with tempfile.TemporaryDirectory(prefix="qk-v2-unrelated-") as temporary:
            root = build_root(temporary)
            output = root / core.OUTPUT_ROOT_REL
            output.mkdir(mode=0o700)
            (output / "unrelated").mkdir()
            expect_surface_error(lambda: core.ensure_output_state(amendment, "Base"), "unrelated output entry")
            labels.append("unrelated_output_entry")

        with tempfile.TemporaryDirectory(prefix="qk-v2-live-nested-regular-") as temporary:
            root = build_root(temporary)
            core.materialize_lane("Base", "3" * 64)
            (core.lane_paths("Base")["root"] / "unrelated").write_bytes(b"unrelated")
            core.LAUNCH_BACKEND = lambda argv, environment: (_ for _ in ()).throw(
                VerificationError("launch reached with unrelated live entry")
            )
            expect_surface_error(lambda: core.consume_and_run("Base"), "nested live regular entry")
            labels.append("nested_live_regular_entry")

        with tempfile.TemporaryDirectory(prefix="qk-v2-live-nested-symlink-") as temporary:
            root = build_root(temporary)
            core.materialize_lane("Base", "4" * 64)
            (core.lane_paths("Base")["root"] / "unrelated-link").symlink_to(root / "build")
            core.LAUNCH_BACKEND = lambda argv, environment: (_ for _ in ()).throw(
                VerificationError("launch reached with symlinked live entry")
            )
            expect_surface_error(lambda: core.consume_and_run("Base"), "nested live symlink entry")
            labels.append("nested_live_symlink_entry")

        with tempfile.TemporaryDirectory(prefix="qk-v2-output-nested-") as temporary:
            root = build_root(temporary)
            core.materialize_lane("Base", "5" * 64)

            def intercepted_with_unrelated_output(
                argv: tuple[str, ...], environment: dict[str, str]
            ) -> Any:
                publish_synthetic_result(root, "Base")
                (root / core.OUTPUT_ROOT_REL / "base" / "unrelated").write_bytes(b"unrelated")
                return types.SimpleNamespace(returncode=0)

            core.LAUNCH_BACKEND = intercepted_with_unrelated_output
            terminal = core.consume_and_run("Base")
            require(terminal["status"] == "FAILED_TERMINAL", "nested output entry failed terminal")
            labels.append("nested_current_result_namespace_entry")

        with tempfile.TemporaryDirectory(prefix="qk-v2-preserved-base-nested-") as temporary:
            root = build_root(temporary)
            core.materialize_lane("Base", "6" * 64)

            def intercepted_base(argv: tuple[str, ...], environment: dict[str, str]) -> Any:
                publish_synthetic_result(root, "Base")
                return types.SimpleNamespace(returncode=0)

            core.LAUNCH_BACKEND = intercepted_base
            terminal = core.consume_and_run("Base")
            require(terminal["status"] == "SUCCEEDED_TERMINAL", "negative fixture Base terminal")
            (root / core.OUTPUT_ROOT_REL / "base" / "unrelated").write_bytes(b"unrelated")
            expect_surface_error(
                lambda: core.ensure_output_state(amendment, "checkpoint-176"),
                "preserved Base namespace unrelated entry",
            )
            labels.append("preserved_base_namespace_unrelated_entry")

        mutated = copy.deepcopy(amendment)
        mutated["execution_package"]["lanes"]["Base"]["argv"][-1] += ".mutated"
        expect_surface_error(lambda: core.validate_lane_descriptor(mutated, "Base"), "argv mutation")
        labels.append("argv_mutation")
        mutated = copy.deepcopy(amendment)
        mutated["execution_package"]["lanes"]["Base"]["environment"]["TZ"] = "GMT"
        expect_surface_error(lambda: core.validate_lane_descriptor(mutated, "Base"), "environment mutation")
        labels.append("environment_mutation")
        mutated = copy.deepcopy(amendment)
        mutated["execution_package"]["lanes"]["Base"]["materialized_argv_sha256"] = mutated["execution_package"]["lanes"]["Base"]["historical_logical_command_descriptor_sha256"]
        expect_surface_error(lambda: core.validate_lane_descriptor(mutated, "Base"), "logical descriptor alias")
        labels.append("logical_descriptor_argv_alias")

        result = synthetic_result("Base")
        result["evaluator_spec_sha256"] = "0" * 64
        reject_mutated_result("forged_result_evaluator_spec", result)
        result = synthetic_result("Base")
        result["generation_id"] += "-forged"
        reject_mutated_result("forged_result_generation", result)
        result = synthetic_result("Base")
        result["input_bindings"]["sealed_set_id"] += "-forged"
        reject_mutated_result("forged_result_sealed_set", result)
        result = synthetic_result("Base")
        result["metrics"]["top_key"]["unexpected"] = 0
        reject_mutated_result("result_nested_schema_extra_field", result)
        result = synthetic_result("Base")
        result["metrics"]["top_key"]["matching_row_count"] -= 1
        reject_mutated_result("result_metric_population_invariant", result)
        result = synthetic_result("Base")
        result["threshold_evaluation"]["top_key_mismatch_count_maximum"]["pass"] = False
        reject_mutated_result("result_threshold_comparison_invariant", result)
        result = synthetic_result("Base")
        result["terminal"]["reason_code"] = "HARD_THRESHOLD_FAILED"
        reject_mutated_result("result_terminal_consistency", result)
        return labels
    finally:
        core.ROOT = original_root
        core.LAUNCH_BACKEND = original_backend


def verify_canonical_absence() -> None:
    paths = [
        ROOT / core.LIVE_ROOT_REL,
        ROOT / core.OUTPUT_ROOT_REL / "base",
        ROOT / core.OUTPUT_ROOT_REL / "checkpoint-176",
    ]
    require(all(not os.path.lexists(path) for path in paths), "canonical live/result lane namespace exists")


def emit_evidence(
    amendment: dict[str, Any], package: dict[str, Any], sequence: dict[str, Any], negatives: list[str]
) -> None:
    artifacts = {
        AMENDMENT_REL: sha256_file(ROOT / AMENDMENT_REL),
        BUILDER_REL: sha256_file(ROOT / BUILDER_REL),
        EVALUATOR_REL: sha256_file(ROOT / EVALUATOR_REL),
        PACKAGE_REL: sha256_file(ROOT / PACKAGE_REL),
        SURFACE_REL: sha256_file(ROOT / SURFACE_REL),
        VERIFIER_REL: sha256_file(ROOT / VERIFIER_REL),
    }
    proof = {
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_live_authority_surface_v2_static_synthetic_proof",
        "claim_boundary": {
            "accepted_invocation_execution_count": 0,
            "canonical_live_authority_materialization_count": 0,
            "model_access_count": 0,
            "sealed_c02_tensor_access_count": 0,
            "synthetic_launch_interception_count": 2,
        },
        "exact_artifact_sha256": artifacts,
        "mutation_closure": {"count": len(negatives), "labels": negatives},
        "schema_version": 2,
        "status": "STATIC_VALID_SYNTHETIC_ONLY_NO_EXECUTION_PERFORMED",
        "synthetic_evaluator": {
            "deterministic_bundle_parser_exercised": True,
            "exact_bf16_oracle_arithmetic_exercised": True,
            "masked_oracle_non_finite_rejection_exercised": True,
            "per_role_nan_inf_negative_count": 6,
            "pickle_or_torch_loader_used": False,
            "record_wide_non_finite_validation_exercised": True,
        },
        "synthetic_two_lane_sequence": sequence,
        "v1_preservation": {
            name: binding["sha256"]
            for name, binding in amendment["accepted_bindings"]["v1_implementation"].items()
        },
    }
    atomic_write(ROOT / PROOF_REL, pretty_bytes(proof))
    artifacts[PROOF_REL] = sha256_file(ROOT / PROOF_REL)
    review = {
        "artifact_kind": "qk_bfp8_e16_head64_v1_c02_live_authority_surface_v2_fresh_l2_review_request",
        "checksum_manifest_path": CHECKSUMS_REL,
        "exact_artifact_sha256": artifacts,
        "requested_verdict": "accept only if exact final bytes independently reproduce STATIC_VALID_SYNTHETIC_ONLY_NO_EXECUTION_PERFORMED",
        "review_state": "PENDING_INDEPENDENT_FRESH_L2_REVIEW",
        "schema_version": 2,
    }
    atomic_write(ROOT / REVIEW_REL, pretty_bytes(review))
    artifacts[REVIEW_REL] = sha256_file(ROOT / REVIEW_REL)
    manifest = "".join(f"{digest}  {path}\n" for path, digest in sorted(artifacts.items()))
    atomic_write(ROOT / CHECKSUMS_REL, manifest.encode("ascii"))


def main() -> int:
    require(platform.python_implementation() == "CPython", "interpreter implementation")
    require(platform.python_version() == "3.13.5", "interpreter version")
    verify_canonical_absence()
    amendment, package = verify_exact_packages()
    verify_source_boundaries()
    synthetic_bundle_test()
    evaluator_negatives = synthetic_non_finite_rejection_tests()
    sequence = synthetic_two_lane_sequence(amendment)
    negatives = evaluator_negatives + synthetic_fail_closed_paths(amendment)
    verify_canonical_absence()
    emit_evidence(amendment, package, sequence, negatives)
    print("STATIC_VALID_SYNTHETIC_ONLY_NO_EXECUTION_PERFORMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
