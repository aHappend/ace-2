#!/usr/bin/env python3
"""Exactly-once, read-only localization of sealed repair-0003 failed gates."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACTION_ID = "localize-repair-0003-failed-gates-from-sealed-artifacts-v1"
GENERATION_ID = "repair-0003-failed-gate-localization-v1"
SOURCE_PATH = Path("tools/localize_w4a8_c02_repair_0003_failed_gates.py")
CONTRACT_PATH = Path(
    "design/W4A8_C02_REPAIR_0003_FAILED_GATE_LOCALIZATION_PREREQUISITE_V1.json"
)
FRESH_L2_PATH = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/"
    "handoffs/fb3fe605c196/round-0001.json"
)
AUTHORITY_PATH = Path(
    "research/raw/specification/"
    "w4a8-c02-repair-0003-failed-gate-localization-v1-operator-authority.json"
)
PIPELINE_PATH = Path("research/PIPELINE_STATE.json")
REPAIR_ROOT = Path(
    "evidence/diagnostics/w4a8-c02-qk-score-rank-margin-repair-design-v1"
)
REPAIR_0002 = REPAIR_ROOT / "repair-0002"
REPAIR_0003 = REPAIR_ROOT / "repair-0003"
OUTPUT_ROOT = REPAIR_ROOT / GENERATION_ID
MODEL_ORDER = ["base", "checkpoint-176"]
ALTERNATIVE_ORDER = [
    "qk_s4_residual_cross_term_scale32_v1",
    "qk_group8_diagonal_scale32_rebalance_v1",
]
STAGE_ORDER = ["fixed_diagnostic", "validation"]

EXPECTED_HASHES = {
    CONTRACT_PATH: "42b3708806a73a8563100fc4cc97ae2784ef74a74cd0bb8725036701e015b0b8",
    FRESH_L2_PATH: "19eacd81bce1b80cd941249fdf58ff5d50a61ee5f2f4e7afc5ee3b24dc33800c",
    REPAIR_0002 / "SHA256SUMS": "1271deb99a33fdcf42960f1dfbdd7301117be62dcbe467bf3cf61d946a43cad4",
    REPAIR_0003 / "SHA256SUMS": "efa5d931c40aa81eaeb22c36e15c9ea85855c868e68472c67d9c1c84fbfb81e9",
    REPAIR_0003 / "selection.json": "cd83ab48b2a52204bf3dcfcc5b754792fccb42cf511fda23765a4f684af81a71",
    REPAIR_0003 / "base/alternative-results.json": "71c6fd129b9c2a76f29ee6df8f48c04ba2a59356cb1824208fa3d62b27ea02ce",
    REPAIR_0003
    / "checkpoint-176/alternative-results.json": "8a0a687d73a864d907fd921f2a300640d731c591ee0454b461ca0eb81af1666f",
    PIPELINE_PATH: "e1638668904f54b3eb46c209cc91390c27b0c9676cb5d917f28f06f26be43358",
}


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with absolute(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(absolute(path).read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_manifest(root: Path, expected_sha256: str) -> dict[str, str]:
    root_abs = absolute(root)
    manifest = root_abs / "SHA256SUMS"
    require(sha256_file(manifest) == expected_sha256, f"manifest hash drift: {root}")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        relative_path = Path(relative)
        require(relative not in entries, f"duplicate manifest entry: {root}/{relative}")
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe manifest entry: {root}/{relative}",
        )
        require(
            sha256_file(root / relative_path) == digest,
            f"manifest member hash drift: {root}/{relative}",
        )
        entries[relative] = digest

    actual = {
        path.relative_to(root_abs).as_posix()
        for path in root_abs.rglob("*")
        if path.is_file()
    }
    omitted_manifests = {
        "SHA256SUMS",
        "run-0001/SHA256SUMS",
        "run-0002/SHA256SUMS",
    }
    require(
        actual - omitted_manifests == set(entries),
        f"root manifest closure mismatch: {root}",
    )

    determinism = load_json(root / "determinism-check.json")
    require(determinism["runs_byte_identical"] is True, "repair repeats differ")
    require(determinism["independent_repeat_count"] == 2, "repeat count drift")
    nested_binding = next(
        item for item in determinism["compared_members"] if item["member"] == "SHA256SUMS"
    )
    for run_name, digest_key in (
        ("run-0001", "run_0001_sha256"),
        ("run-0002", "run_0002_sha256"),
    ):
        run_root = root / run_name
        run_root_abs = absolute(run_root)
        run_manifest = run_root / "SHA256SUMS"
        require(
            sha256_file(run_manifest) == nested_binding[digest_key],
            f"nested manifest binding drift: {run_name}",
        )
        nested_entries: dict[str, str] = {}
        for line in absolute(run_manifest).read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            relative_path = Path(relative)
            require(
                relative not in nested_entries
                and not relative_path.is_absolute()
                and ".." not in relative_path.parts,
                f"invalid nested manifest entry: {run_name}/{relative}",
            )
            require(
                sha256_file(run_root / relative_path) == digest,
                f"nested member hash drift: {run_name}/{relative}",
            )
            nested_entries[relative] = digest
        nested_actual = {
            path.relative_to(run_root_abs).as_posix()
            for path in run_root_abs.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        require(
            nested_actual == set(nested_entries),
            f"nested manifest closure mismatch: {run_name}",
        )
    return entries


def validate_authority_and_inputs() -> dict[str, Any]:
    for path, expected in EXPECTED_HASHES.items():
        require(sha256_file(path) == expected, f"bound input hash drift: {path}")

    contract = load_json(CONTRACT_PATH)
    fresh_l2 = load_json(FRESH_L2_PATH)
    pipeline = load_json(PIPELINE_PATH)
    authority = load_json(AUTHORITY_PATH)
    authority_hash = sha256_file(AUTHORITY_PATH)
    sidecar = absolute(Path(str(AUTHORITY_PATH) + ".sha256")).read_text(
        encoding="utf-8"
    )
    require(sidecar == f"{authority_hash}  {AUTHORITY_PATH.as_posix()}\n", "authority sidecar drift")

    require(contract["generation_id"] == GENERATION_ID, "contract generation drift")
    action = contract["consuming_action"]
    require(action["action_id"] == ACTION_ID, "contract action drift")
    require(action["cardinality"] == action["execution_count"] == 1, "cardinality drift")
    require(action["allowed_source_path"] == SOURCE_PATH.as_posix(), "source path drift")
    require(action["output_root"] == OUTPUT_ROOT.as_posix(), "output path drift")
    require(action["input_mode"] == "read_only_sealed_artifacts_only", "input mode drift")
    require(
        contract["acceptance_contract"]["required_verdicts"]
        == [
            "LOCALIZATION_COMPLETE_NO_EXECUTION_AUTHORITY",
            "LOCALIZATION_INCOMPLETE_NO_EXECUTION_AUTHORITY",
        ],
        "allowed verdict drift",
    )

    require(fresh_l2["producer_role"] == "reviewer", "Fresh-L2 role drift")
    require(fresh_l2["review"]["status"] == "done", "Fresh-L2 status drift")
    require(
        "exact frozen prerequisite contract" in fresh_l2["review"]["reason"],
        "Fresh-L2 scope mismatch",
    )
    require(pipeline["current_stage"] == "rtl", "pipeline stage drift")

    require(authority["status"] == "VALID_FOR_ONE_TERMINAL_CONSUMPTION", "authority status drift")
    require(authority["action"]["action_id"] == ACTION_ID, "authority action drift")
    require(authority["action"]["generation_id"] == GENERATION_ID, "authority generation drift")
    require(authority["action"]["cardinality"] == 1, "authority cardinality drift")
    require(authority["action"]["execution_count"] == 1, "authority count drift")
    require(authority["action"]["no_retry_replay_resume"] is True, "retry prohibition drift")
    require(authority["bindings"]["current_stage"] == "rtl", "authority stage drift")
    binding_pairs = {
        "contract": (CONTRACT_PATH, EXPECTED_HASHES[CONTRACT_PATH]),
        "fresh_l2_acceptance": (FRESH_L2_PATH, EXPECTED_HASHES[FRESH_L2_PATH]),
        "pipeline_state": (PIPELINE_PATH, EXPECTED_HASHES[PIPELINE_PATH]),
        "repair_0002_manifest": (
            REPAIR_0002 / "SHA256SUMS",
            EXPECTED_HASHES[REPAIR_0002 / "SHA256SUMS"],
        ),
        "repair_0003_manifest": (
            REPAIR_0003 / "SHA256SUMS",
            EXPECTED_HASHES[REPAIR_0003 / "SHA256SUMS"],
        ),
    }
    for name, (path, digest) in binding_pairs.items():
        binding = authority["bindings"][name]
        require(binding["path"] == path.as_posix(), f"authority path drift: {name}")
        require(binding["sha256"] == digest, f"authority hash drift: {name}")

    raw_event = authority["raw_operator_event"]
    require(raw_event["channel"] == "user", "operator provenance channel drift")
    require(
        raw_event["authority_label"] == "LIVE MANAGER / OPERATOR DIRECTIVES — HIGHEST PRIORITY",
        "operator authority label drift",
    )
    raw_text = raw_event["exact_text"]
    require(
        raw_text.startswith(f"AUTHORIZE EXACTLY ONE {GENERATION_ID} consuming action."),
        "operator event is not affirmative",
    )
    for required_text in (
        ACTION_ID,
        EXPECTED_HASHES[CONTRACT_PATH],
        EXPECTED_HASHES[FRESH_L2_PATH],
        EXPECTED_HASHES[REPAIR_0002 / "SHA256SUMS"],
        EXPECTED_HASHES[REPAIR_0003 / "SHA256SUMS"],
        EXPECTED_HASHES[PIPELINE_PATH],
        "current_stage=rtl",
        "No retry, replay, resume, expansion",
        "terminally consume this authorization whether execution succeeds or fails",
    ):
        require(required_text in raw_text, f"operator raw-text binding missing: {required_text}")
    event_time = datetime.fromisoformat(raw_event["observed_at_utc"].replace("Z", "+00:00"))
    fresh_time = datetime.fromtimestamp(fresh_l2["created_at"], timezone.utc)
    require(event_time > fresh_time, "operator event does not follow Fresh-L2")

    repair_0002_entries = verify_manifest(
        REPAIR_0002, EXPECTED_HASHES[REPAIR_0002 / "SHA256SUMS"]
    )
    repair_0003_entries = verify_manifest(
        REPAIR_0003, EXPECTED_HASHES[REPAIR_0003 / "SHA256SUMS"]
    )
    for relative in (
        "run-contract.json",
        "selection.json",
        "determinism-check.json",
        "base/alternative-results.json",
        "checkpoint-176/alternative-results.json",
    ):
        require(
            repair_0003_entries[relative] != repair_0002_entries[relative],
            f"repair-0003 reused predecessor generation field: {relative}",
        )

    selection = load_json(REPAIR_0003 / "selection.json")
    require(selection["verdict"] == "PRE_CANDIDATE_NO_GO", "selection verdict drift")
    require(selection["selected_alternative_id"] is None, "unexpected selection")
    require(selection["candidate_package_created"] is False, "candidate package exists")
    require(
        [item["alternative_id"] for item in selection["alternatives"]]
        == ALTERNATIVE_ORDER,
        "alternative inventory/order drift",
    )
    require(
        all(
            item["pass"] is False
            and list(item["model_results"]) == MODEL_ORDER
            and not any(item["model_results"].values())
            for item in selection["alternatives"]
        ),
        "selection model result drift",
    )
    return {
        "authority_sha256": authority_hash,
        "contract": contract,
        "repair_0002_manifest_entry_count": len(repair_0002_entries),
        "repair_0003_manifest_entry_count": len(repair_0003_entries),
        "selection": selection,
    }


def available_clamp_coordinate_fields(stage: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if "clamp" in key.lower() and key != "clamp_events":
                    fields[child_path] = child
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(stage, "")
    return fields


def build_localization(context: dict[str, Any]) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    saturation_required = False
    joint_qk_change_required = False

    for model in MODEL_ORDER:
        result = load_json(REPAIR_0003 / model / "alternative-results.json")
        require(result["model_alias"] == model, f"model alias drift: {model}")
        require(
            result["claim_boundary"]["current_stage_remains"] == "rtl"
            and result["claim_boundary"]["candidate_package_created"] is False,
            f"claim-boundary drift: {model}",
        )
        alternatives = {item["alternative_id"]: item for item in result["alternatives"]}
        require(list(alternatives) == ALTERNATIVE_ORDER, f"alternative order drift: {model}")
        for alternative_id in ALTERNATIVE_ORDER:
            alternative = alternatives[alternative_id]
            require(alternative["pass"] is False, f"unexpected passing alternative: {model}")
            stage_counts: dict[str, Any] = {}
            total_clamp_events = 0
            for stage_name in STAGE_ORDER:
                stage = alternative[stage_name]
                count = stage["integer_safety"]["clamp_events"]
                require(isinstance(count, int) and count >= 0, "invalid clamp count")
                total_clamp_events += count
                coordinate_fields = available_clamp_coordinate_fields(stage)
                if count:
                    saturation_required = True
                    if not coordinate_fields:
                        missing.append(
                            {
                                "alternative_id": alternative_id,
                                "model": model,
                                "required_field": "clamp_event_coordinates",
                                "stage": stage_name,
                            }
                        )
                stage_counts[stage_name] = {
                    "available_coordinate_fields": coordinate_fields,
                    "clamp_event_count": count,
                    "coordinate_completeness": (
                        "COMPLETE_ZERO_EVENTS"
                        if count == 0
                        else (
                            "AVAILABLE_IN_SEALED_ARTIFACT"
                            if coordinate_fields
                            else "NOT_PRESENT_IN_SEALED_ARTIFACT"
                        )
                    ),
                }

            agreement = alternative["validation"]["top_key_agreement"]
            baseline = agreement["baseline_top_key_index_equal_fraction"]
            corrected = agreement["corrected_top_key_index_equal_fraction"]
            improvement = agreement["absolute_improvement"]
            require(
                abs((corrected - baseline) - improvement) <= 1e-15,
                f"top-key improvement arithmetic drift: {model}/{alternative_id}",
            )
            regressions = sorted(agreement["head_regressions"])
            derived_regressions = sorted(
                item["head"]
                for item in agreement["per_layer_head"]
                if item["improvement"] < 0.0
            )
            require(regressions == derived_regressions, "head-regression list is incomplete")
            require(len(regressions) == len(set(regressions)), "duplicate regressed head")
            if corrected < 0.9 or regressions:
                joint_qk_change_required = True
            records.append(
                {
                    "alternative_id": alternative_id,
                    "integer_clamp_events": {
                        "stages": stage_counts,
                        "total": total_clamp_events,
                    },
                    "model": model,
                    "validation_top_key_agreement": {
                        "absolute_improvement_from_unchanged_c01_baseline": improvement,
                        "corrected_top_key_index_equal_fraction": corrected,
                        "regressed_heads": regressions,
                        "unchanged_c01_baseline_top_key_index_equal_fraction": baseline,
                    },
                }
            )

    if saturation_required and joint_qk_change_required:
        disposition = "ELIMINATE_SATURATION_AND_CHANGE_QK_REPRESENTATION_JOINTLY"
    elif saturation_required:
        disposition = "ELIMINATE_SATURATION"
    elif joint_qk_change_required:
        disposition = "CHANGE_QK_REPRESENTATION_JOINTLY"
    else:
        raise ValueError("sealed failures support none of the allowed structural dispositions")

    verdict = (
        "LOCALIZATION_INCOMPLETE_NO_EXECUTION_AUTHORITY"
        if missing
        else "LOCALIZATION_COMPLETE_NO_EXECUTION_AUTHORITY"
    )
    require(
        verdict in context["contract"]["acceptance_contract"]["required_verdicts"],
        "verdict outside contract",
    )
    return {
        "artifact_kind": "w4a8_c02_repair_0003_failed_gate_localization",
        "candidate_or_package_selected": False,
        "completeness": {
            "missing_required_sealed_fields": missing,
            "statement": (
                "The sealed artifacts provide exact clamp-event counts but no event coordinates "
                "for stages with nonzero counts. Those coordinates were not reconstructed, and "
                "no model inference, evaluator scoring, or repair campaign was rerun."
                if missing
                else "Every required metric and available coordinate was present in the sealed artifacts."
            ),
        },
        "generation_id": GENERATION_ID,
        "localizations": records,
        "schema_version": 1,
        "selection_preserved": {
            "repair_0003_selected_alternative_id": None,
            "repair_0003_verdict": context["selection"]["verdict"],
        },
        "structural_disposition": {
            "decision": disposition,
            "reason": (
                "Every frozen alternative has nonzero validation clamp events, and every "
                "model/alternative pair remains below the 0.9 validation top-key gate with "
                "at least one regressed head. The next separately authorized design must "
                "therefore address both saturation and the joint Q/K representation."
            ),
        },
        "verdict": verdict,
    }


def write_required_outputs(run_contract: dict[str, Any], localization: dict[str, Any]) -> None:
    run_path = absolute(OUTPUT_ROOT / "run-contract.json")
    localization_path = absolute(OUTPUT_ROOT / "localization.json")
    run_path.write_bytes(canonical_bytes(run_contract))
    localization_path.write_bytes(canonical_bytes(localization))
    manifest = (
        f"{sha256_file(OUTPUT_ROOT / 'run-contract.json')}  run-contract.json\n"
        f"{sha256_file(OUTPUT_ROOT / 'localization.json')}  localization.json\n"
    )
    absolute(OUTPUT_ROOT / "SHA256SUMS").write_text(manifest, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("this analyzer accepts no arguments and has no retry/check mode")
    output = absolute(OUTPUT_ROOT)
    if output.exists():
        raise SystemExit("authorization already consumed: frozen output root exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started_at = utc_now()
    source_sha256 = sha256_file(SOURCE_PATH)
    initial = {
        "action_id": ACTION_ID,
        "artifact_kind": "w4a8_c02_repair_0003_failed_gate_localization_run_contract",
        "execution": {
            "cardinality": 1,
            "execution_count": 1,
            "no_retry_replay_resume": True,
            "started_at_utc": started_at,
            "status": "EXECUTION_STARTED_TERMINAL_NO_RETRY",
            "terminal_authorization_consumption": True,
        },
        "generation_id": GENERATION_ID,
        "schema_version": 1,
    }
    absolute(OUTPUT_ROOT / "run-contract.json").write_bytes(canonical_bytes(initial))

    try:
        context = validate_authority_and_inputs()
        localization = build_localization(context)
        run_contract = {
            "action_id": ACTION_ID,
            "analyzer": {
                "invocation": (
                    "PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python "
                    "tools/localize_w4a8_c02_repair_0003_failed_gates.py"
                ),
                "source_path": SOURCE_PATH.as_posix(),
                "source_sha256": source_sha256,
            },
            "artifact_kind": "w4a8_c02_repair_0003_failed_gate_localization_run_contract",
            "authority": {
                "path": AUTHORITY_PATH.as_posix(),
                "sha256": context["authority_sha256"],
                "terminally_consumed": True,
            },
            "bound_inputs": {
                path.as_posix(): digest for path, digest in EXPECTED_HASHES.items()
            },
            "claim_boundary": {
                "candidate_or_package_created": False,
                "current_stage_remains": "rtl",
                "evaluator_scoring_performed": False,
                "model_inference_performed": False,
                "repair_campaign_rerun": False,
                "rtl_spec_manifest_u280_mutated": False,
                "stage_transition_performed": False,
            },
            "execution": {
                "cardinality": 1,
                "execution_count": 1,
                "finished_at_utc": utc_now(),
                "no_retry_replay_resume": True,
                "started_at_utc": started_at,
                "status": "SUCCEEDED_TERMINAL_AUTHORIZATION_CONSUMED",
                "terminal_authorization_consumption": True,
            },
            "generation_id": GENERATION_ID,
            "manifest_closure": {
                "repair_0002_entry_count": context["repair_0002_manifest_entry_count"],
                "repair_0003_entry_count": context["repair_0003_manifest_entry_count"],
                "status": "PASS_TRANSITIVE_CLOSURE_AND_FRESH_GENERATION_SEMANTICS",
            },
            "output_root": OUTPUT_ROOT.as_posix(),
            "schema_version": 1,
        }
        write_required_outputs(run_contract, localization)
        print(
            f"WROTE {OUTPUT_ROOT.as_posix()} "
            f"{localization['verdict']} TERMINAL_AUTHORIZATION_CONSUMED"
        )
        return 0
    except Exception as exc:
        localization = {
            "artifact_kind": "w4a8_c02_repair_0003_failed_gate_localization",
            "candidate_or_package_selected": False,
            "completeness": {
                "missing_required_sealed_fields": [
                    {"required_field": "terminal_execution_error"}
                ],
                "statement": "The one authorized execution failed closed and was not retried.",
            },
            "generation_id": GENERATION_ID,
            "localizations": [],
            "schema_version": 1,
            "structural_disposition": None,
            "terminal_error": f"{type(exc).__name__}: {exc}",
            "verdict": "LOCALIZATION_INCOMPLETE_NO_EXECUTION_AUTHORITY",
        }
        run_contract = {
            "action_id": ACTION_ID,
            "analyzer": {
                "source_path": SOURCE_PATH.as_posix(),
                "source_sha256": source_sha256,
            },
            "artifact_kind": "w4a8_c02_repair_0003_failed_gate_localization_run_contract",
            "execution": {
                "cardinality": 1,
                "execution_count": 1,
                "finished_at_utc": utc_now(),
                "no_retry_replay_resume": True,
                "started_at_utc": started_at,
                "status": "FAILED_TERMINAL_AUTHORIZATION_CONSUMED",
                "terminal_authorization_consumption": True,
            },
            "generation_id": GENERATION_ID,
            "output_root": OUTPUT_ROOT.as_posix(),
            "schema_version": 1,
            "terminal_error": f"{type(exc).__name__}: {exc}",
        }
        write_required_outputs(run_contract, localization)
        print(
            f"FAILED {OUTPUT_ROOT.as_posix()} TERMINAL_AUTHORIZATION_CONSUMED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
