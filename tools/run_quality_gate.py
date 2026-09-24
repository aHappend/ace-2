#!/usr/bin/env python3
"""Audit readiness and bind immutable official ACE-2 quality evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ace2_quality_contracts import validate_oracle_manifest, validate_rtl_binding


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmark" / "raw"
RESULTS = RAW / "latest" / "quality_results.json"
PREFLIGHT = RAW / "latest" / "quality_gate_preflight.json"
OFFICIAL_POINTER = RAW / "latest" / "quality_official_pointer.json"
PROMPT_MANIFEST = ROOT / "benchmark" / "quality" / "PROMPT_MANIFEST.json"
QUALITY_CONFIG = ROOT / "benchmark" / "quality" / "QUALITY_CONFIG.json"
QUALITY_REQUIREMENTS = ROOT / "benchmark" / "quality" / "requirements.txt"
QUALITY_BLOCKER = ROOT / "benchmark" / "quality" / "QUALITY_BLOCKER.json"
LM_EVAL_TASKS = ROOT / "benchmark" / "quality" / "lm_eval_tasks"
FIXED_POINT_MODEL = ROOT / "tools" / "ace2_full_model_fixed_point.py"
OFFICIAL_RUNNER = ROOT / "tools" / "run_official_quality.py"
ORACLE_MANIFEST = ROOT / "reference" / "ORACLE_MANIFEST.json"
RTL_BINDING = ROOT / "benchmark" / "quality" / "RTL_BINDING.json"
PROJECT_VENV = ROOT / ".venv"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_SOFTWARE = {
    "datasets": "4.8.5",
    "lm_eval": "0.4.9.2",
    "torch": "2.11.0",
    "transformers": "4.57.6",
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_fixed_point_self_test() -> dict[str, Any]:
    command = [sys.executable, str(FIXED_POINT_MODEL), "--mode", "self-test"]
    status: dict[str, Any] = {
        "command": command,
        "source": artifact(FIXED_POINT_MODEL),
        "passed": False,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        status["error"] = str(exc)
        return status
    status.update(
        {
            "exit_code": completed.returncode,
            "stderr": completed.stderr.strip(),
            "stdout": completed.stdout.strip(),
            "passed": completed.returncode == 0,
        }
    )
    return status


def verify_hash_manifest(manifest_path: Path) -> dict[str, Any]:
    run_dir = manifest_path.parent.resolve()
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64:
            raise ValueError(f"malformed hash-manifest line: {line!r}")
        path = (run_dir / relative).resolve()
        if run_dir not in path.parents:
            raise ValueError(f"hash-manifest path escapes run directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"hash-manifest artifact is missing: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"hash mismatch for {relative}: {observed} != {expected}")
        entries.append({"path": relative, "sha256": observed})
    if not entries:
        raise ValueError("official hash manifest has no entries")
    return {"entry_count": len(entries), "entries": entries}


def load_official_evidence() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    status: dict[str, Any] = {
        "pointer": artifact(OFFICIAL_POINTER),
        "valid": False,
    }
    if not OFFICIAL_POINTER.exists():
        return None, status
    try:
        pointer = json.loads(OFFICIAL_POINTER.read_text(encoding="utf-8"))
        run_dir = (ROOT / pointer["run_path"]).resolve()
        quality_root = (RAW / "quality").resolve()
        if quality_root not in run_dir.parents:
            raise ValueError("official run path is outside benchmark/raw/quality")
        manifest_path = run_dir / "SHA256SUMS"
        result_path = run_dir / "artifacts" / "results.json"
        if sha256_file(manifest_path) != pointer["sha256s_sha256"]:
            raise ValueError("official SHA256SUMS hash differs from pointer")
        verification = verify_hash_manifest(manifest_path)
        if sha256_file(result_path) != pointer["results_sha256"]:
            raise ValueError("official results hash differs from pointer")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("mode") != "official":
            raise ValueError("pointed result is not an official measurement")
        status.update(
            {
                "manifest": artifact(manifest_path),
                "manifest_verification": verification,
                "result": artifact(result_path),
                "run_path": run_dir.relative_to(ROOT).as_posix(),
                "valid": True,
            }
        )
        return result, status
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        status["error"] = str(exc)
        return None, status


def enforce_quality_blocker_precedence(
    official_result: dict[str, Any] | None,
    official_status: dict[str, Any],
    quality_blocker: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if official_result is None or quality_blocker is None:
        return official_result, official_status
    rejected_status = {
        **official_status,
        "error": "official evidence cannot override an active quality blocker",
        "rejected_by_active_blocker": quality_blocker["classification"],
        "valid": False,
    }
    return None, rejected_status


def enforce_preflight_precedence(
    official_result: dict[str, Any] | None,
    official_status: dict[str, Any],
    missing_prerequisites: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if official_result is None or not missing_prerequisites:
        return official_result, official_status
    rejected_status = {
        **official_status,
        "error": "official evidence cannot override failed preflight prerequisites",
        "rejected_by_precondition_failures": list(missing_prerequisites),
        "valid": False,
    }
    return None, rejected_status


def load_quality_blocker(
    config: dict[str, Any],
    accepted_rtl: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    status: dict[str, Any] = {
        "binding": artifact(QUALITY_BLOCKER),
        "active": False,
        "valid": False,
    }
    if not QUALITY_BLOCKER.exists():
        status["valid"] = True
        return None, status
    try:
        binding = json.loads(QUALITY_BLOCKER.read_text(encoding="utf-8"))
        if binding.get("schema_version") != 1:
            raise ValueError("quality blocker binding schema differs from version 1")
        blocker_status = binding.get("status")
        supported_statuses = {
            "independently_reproduced_numerical_contract_failure",
            "independently_reproduced_rope_scale_range_failure",
        }
        if blocker_status not in supported_statuses:
            raise ValueError("quality blocker binding has an unsupported status")
        if not accepted_rtl.get("valid"):
            raise ValueError("accepted RTL must validate before quality blocker evidence")
        if binding["accepted_rtl_hash"] != accepted_rtl["candidate_rtl_hash"]:
            raise ValueError("quality blocker evidence is bound to different accepted RTL")

        frozen_sources = {}
        for name, expected in binding["frozen_sources"].items():
            path = (ROOT / expected["path"]).resolve()
            if ROOT.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"quality blocker source is not a repository file: {name}")
            observed = artifact(path)
            if observed["sha256"] != expected["sha256"]:
                raise ValueError(
                    f"quality blocker source hash differs for {name}: "
                    f"{observed['sha256']} != {expected['sha256']}"
                )
            frozen_sources[name] = observed

        def load_run_evidence(name: str) -> dict[str, Any]:
            expected = binding["evidence"][name]
            run_dir = (ROOT / expected["run_path"]).resolve()
            quality_root = (RAW / "quality").resolve()
            if quality_root not in run_dir.parents:
                raise ValueError(f"{name} evidence is outside benchmark/raw/quality")
            manifest_path = run_dir / "SHA256SUMS"
            result_path = run_dir / "results.json"
            if sha256_file(manifest_path) != expected["sha256s_sha256"]:
                raise ValueError(f"{name} SHA256SUMS hash differs from blocker binding")
            manifest_verification = verify_hash_manifest(manifest_path)
            if sha256_file(result_path) != expected["results_sha256"]:
                raise ValueError(f"{name} results hash differs from blocker binding")
            return {
                "manifest": artifact(manifest_path),
                "manifest_verification": manifest_verification,
                "result": artifact(result_path),
                "run_path": run_dir.relative_to(ROOT).as_posix(),
                "value": json.loads(result_path.read_text(encoding="utf-8")),
            }

        evidence: dict[str, Any] = {"smoke": load_run_evidence("smoke")}
        if blocker_status == "independently_reproduced_numerical_contract_failure":
            evidence["localization"] = load_run_evidence("localization")

        smoke = evidence["smoke"]["value"]
        expected = binding["expected_observations"]
        if smoke.get("mode") != "smoke" or smoke.get("gate_passed") is not False:
            raise ValueError("bound smoke evidence is not a failed smoke result")
        if smoke.get("model", {}).get("revision") != MODEL_REVISION:
            raise ValueError("bound smoke evidence uses a different model revision")
        if smoke.get("thresholds") != config["acceptance_thresholds"]:
            raise ValueError("bound smoke thresholds differ from the frozen quality contract")
        if any(smoke.get("checks", {}).values()) or not smoke.get("checks"):
            raise ValueError("bound smoke evidence does not fail every reported check")
        smoke_ratios = {
            name: smoke["metrics"][name]["ratio"]
            for name in ("c4_en_512", "wikitext2")
        }
        if smoke_ratios != expected["smoke_ratios"]:
            raise ValueError("bound smoke ratios differ from expected observations")
        smoke_rtl = smoke["artifacts"]["accepted_rtl"]
        if smoke_rtl["candidate_rtl_hash"] != binding["accepted_rtl_hash"]:
            raise ValueError("bound smoke evidence uses different accepted RTL")

        if blocker_status == "independently_reproduced_rope_scale_range_failure":
            derived_scales_path = (
                ROOT / evidence["smoke"]["run_path"] / "derived_scales.json"
            )
            derived_scales_hash = sha256_file(derived_scales_path)
            if derived_scales_hash != expected["derived_scales_sha256"]:
                raise ValueError("bound derived-scale hash differs from blocker binding")
            if smoke["artifacts"]["derived_scales_sha256"] != derived_scales_hash:
                raise ValueError("smoke result points to a different derived-scale table")
            derived_scales = json.loads(
                derived_scales_path.read_text(encoding="utf-8")
            )
            attention = derived_scales["attention"]

            def layer_index(name: str) -> int:
                return int(name.split(".")[2])

            query_fractions = [
                layer["query_rope_output_saturation"]["fraction"]
                for layer in attention.values()
            ]
            key_fractions = [
                layer["key_rope_output_saturation"]["fraction"]
                for layer in attention.values()
            ]
            range_unsafe_layers = sorted(
                layer_index(name)
                for name, layer in attention.items()
                if not layer["rope_range_safe"]
            )
            score_requantization_error_layers = sorted(
                layer_index(name)
                for name, layer in attention.items()
                if abs(
                    layer["score_realized_multiplier"]
                    - layer["score_real_multiplier"]
                )
                / layer["score_real_multiplier"]
                > 1e-6
            )
            rope_observations = {
                "derived_scales_sha256": derived_scales_hash,
                "query_rope_output_saturation_fraction_range": {
                    "min": min(query_fractions),
                    "max": max(query_fractions),
                },
                "key_rope_output_saturation_fraction_range": {
                    "min": min(key_fractions),
                    "max": max(key_fractions),
                },
                "rope_range_unsafe_layers": range_unsafe_layers,
                "score_requantization_error_gt_1ppm_layers": (
                    score_requantization_error_layers
                ),
            }
            for name, observed in rope_observations.items():
                if observed != expected[name]:
                    raise ValueError(f"bound RoPE observation differs for {name}")
            evidence["smoke"]["derived_scales"] = artifact(derived_scales_path)
            for item in evidence.values():
                del item["value"]
            blocker = {
                "accepted_rtl_hash": binding["accepted_rtl_hash"],
                "classification": binding["status"],
                "evidence": evidence,
                "rope_scale_range": rope_observations,
                "smoke_ratios": smoke_ratios,
            }
            status.update(blocker)
            status["active"] = True
            status["frozen_sources"] = frozen_sources
            status["valid"] = True
            return blocker, status

        localization = evidence["localization"]["value"]
        if localization.get("classification") != "diagnostic_localization_not_acceptance_evidence":
            raise ValueError("bound localization has an unsupported classification")
        if localization.get("model", {}).get("revision") != MODEL_REVISION:
            raise ValueError("bound localization uses a different model revision")
        first_changed = localization["method"]["first_changed_boundary"]
        if first_changed != expected["first_changed_boundary"]:
            raise ValueError("bound localization has a different first changed boundary")
        rmsnorm = localization["comparisons"][first_changed]
        if (
            rmsnorm["relative_l2_error"] != expected["rmsnorm_relative_l2_error"]
            or rmsnorm["candidate_zero_fraction"]
            != expected["rmsnorm_w4a8_zero_fraction"]
        ):
            raise ValueError("bound RMSNorm observations differ from expected values")
        localization_rtl = localization["contract"]["accepted_rtl"]
        if localization_rtl["candidate_rtl_hash"] != binding["accepted_rtl_hash"]:
            raise ValueError("bound localization evidence uses different accepted RTL")

        for item in evidence.values():
            del item["value"]
        blocker = {
            "accepted_rtl_hash": binding["accepted_rtl_hash"],
            "classification": binding["status"],
            "evidence": evidence,
            "first_changed_boundary": first_changed,
            "rmsnorm": {
                "relative_l2_error": rmsnorm["relative_l2_error"],
                "w4a8_zero_fraction": rmsnorm["candidate_zero_fraction"],
            },
            "smoke_ratios": smoke_ratios,
        }
        status.update(blocker)
        status["active"] = True
        status["frozen_sources"] = frozen_sources
        status["valid"] = True
        return blocker, status
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        status["error"] = str(exc)
        return None, status


def quality_blocker_reason(quality_blocker: dict[str, Any]) -> str:
    if (
        quality_blocker["classification"]
        == "independently_reproduced_rope_scale_range_failure"
    ):
        return (
            "Independent paired smoke evidence remains orders of magnitude outside both "
            "perplexity limits after the repaired RMSNorm contract, and bound scale "
            "diagnostics show pervasive RoPE Q2.13 saturation."
        )
    return (
        "Independent paired smoke evidence is orders of magnitude outside both "
        "perplexity limits, and paired localization identifies the accepted unit-scale "
        "integer RMSNorm output as the first changed boundary."
    )


def quality_blocker_required_resolution(quality_blocker: dict[str, Any]) -> str:
    if (
        quality_blocker["classification"]
        == "independently_reproduced_rope_scale_range_failure"
    ):
        return (
            "Open a separately scoped RoPE/attention scale-range repair or no-go mission; "
            "do not run official quality measurement against this known failing contract."
        )
    return (
        "Open a separately scoped RMSNorm numerical-format/RTL mission and obtain "
        "operator approval before changing the frozen contract; do not run official "
        "quality measurement against this known failing contract."
    )


def run_gate() -> dict[str, Any]:
    config = json.loads(QUALITY_CONFIG.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    if config["software"] != EXPECTED_SOFTWARE:
        raise ValueError("quality software pins differ from preflight contract")
    package_checks = {
        name: {
            "expected": expected,
            "observed": package_version("lm-eval" if name == "lm_eval" else name),
        }
        for name, expected in EXPECTED_SOFTWARE.items()
    }
    for check in package_checks.values():
        check["matches"] = check["observed"] == check["expected"]
    required_files = {
        "prompt_manifest": artifact(PROMPT_MANIFEST),
        "quality_config_with_scale_derivation": artifact(QUALITY_CONFIG),
        "pinned_requirements": artifact(QUALITY_REQUIREMENTS),
        "full_model_fixed_point_reference": artifact(FIXED_POINT_MODEL),
        "official_measurement_runner": artifact(OFFICIAL_RUNNER),
        "operator_oracle_manifest": artifact(ORACLE_MANIFEST),
        "accepted_rtl_binding": artifact(RTL_BINDING),
    }
    task_files = sorted(path for path in LM_EVAL_TASKS.iterdir() if path.is_file())
    task_artifacts = [artifact(path) for path in task_files]
    environment_checks = {
        "project_venv_exists": PROJECT_VENV.is_dir(),
        "project_venv_interpreter_active": Path(sys.prefix).resolve()
        == PROJECT_VENV.resolve(),
        "transformers_module_available": importlib.util.find_spec("transformers") is not None,
        "datasets_module_available": importlib.util.find_spec("datasets") is not None,
        "torch_module_available": importlib.util.find_spec("torch") is not None,
        "lm_eval_module_available": importlib.util.find_spec("lm_eval") is not None,
    }
    missing = [
        name for name, item in required_files.items() if not item["exists"]
    ]
    missing.extend(
        f"lm_eval_task:{item['path']}" for item in task_artifacts if not item["exists"]
    )
    missing.extend(name for name, passed in environment_checks.items() if not passed)
    missing.extend(
        f"package:{name}"
        for name, check in package_checks.items()
        if not check["matches"]
    )
    fixed_point_self_test = run_fixed_point_self_test()
    if not fixed_point_self_test["passed"]:
        missing.append("fixed_point_self_test")
    try:
        oracle_contract = validate_oracle_manifest(ROOT)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        oracle_contract = {
            "manifest": artifact(ORACLE_MANIFEST),
            "error": str(exc),
            "valid": False,
        }
        missing.append("operator_oracle_manifest_validation")
    try:
        accepted_rtl = validate_rtl_binding(ROOT)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        accepted_rtl = {
            "binding": artifact(RTL_BINDING),
            "error": str(exc),
            "valid": False,
        }
        missing.append("accepted_rtl_binding_validation")
    try:
        piqa = prompt_manifest["lm_eval"]["tasks"]["piqa"]
        piqa_hash_pins = {
            "archive": piqa["raw_sources"]["train_dev"],
            "records": piqa["records"],
            "valid": True,
        }
        if len(piqa_hash_pins["archive"]["sha256"]) != 64:
            raise ValueError("PIQA archive SHA-256 pin is malformed")
        for split in ("train", "validation"):
            expected = piqa_hash_pins["records"][split]
            if expected["count"] <= 0 or len(expected["sha256"]) != 64:
                raise ValueError(f"PIQA {split} record pin is malformed")
    except (KeyError, TypeError, ValueError) as exc:
        piqa_hash_pins = {"error": str(exc), "valid": False}
        missing.append("piqa_archive_record_hash_pins")
    quality_blocker, quality_blocker_status = load_quality_blocker(config, accepted_rtl)
    if not quality_blocker_status["valid"]:
        missing.append("quality_blocker_validation")
    measurement_ready = not missing and quality_blocker is None

    official_result, official_status = enforce_preflight_precedence(
        *load_official_evidence(),
        missing,
    )
    official_result, official_status = enforce_quality_blocker_precedence(
        official_result,
        official_status,
        quality_blocker,
    )
    generated_at = utc_now()
    if official_result is None:
        result = {
            "schema_version": 2,
            "generated_at_utc": generated_at,
            "command": "./.venv/bin/python tools/run_quality_gate.py",
            "status": (
                f"blocked_{quality_blocker['classification']}"
                if quality_blocker is not None
                else (
                    "ready_for_explicit_official_measurement"
                    if measurement_ready
                    else "blocked_precondition_failure"
                )
            ),
            "gate_passed": False,
            "measurement_started": False,
            "model": {
                "repository": "Qwen/Qwen2.5-0.5B",
                "revision": MODEL_REVISION,
            },
            "prerequisites": {
                **required_files,
                "accepted_rtl_validation": accepted_rtl,
                "environment": environment_checks,
                "fixed_point_self_test": fixed_point_self_test,
                "lm_eval_tasks": task_artifacts,
                "operator_oracles": oracle_contract,
                "packages": package_checks,
                "piqa_hash_pins": piqa_hash_pins,
                "quality_blocker_validation": quality_blocker_status,
            },
            "missing_prerequisites": missing,
            "blocking_conditions": (
                [quality_blocker["classification"]]
                if quality_blocker is not None
                else list(missing)
            ),
            "metrics": {
                "wikitext2": {
                    "bf16_perplexity": None,
                    "w4a8_perplexity": None,
                    "ratio": None,
                },
                "c4_en_512": {
                    "bf16_perplexity": None,
                    "w4a8_perplexity": None,
                    "ratio": None,
                },
                "lm_eval": {
                    "bf16_average_normalized_accuracy": None,
                    "w4a8_average_normalized_accuracy": None,
                    "drop_percentage_points": None,
                },
            },
            "reason": (
                quality_blocker_reason(quality_blocker)
                if quality_blocker is not None
                else (
                    "The contract-faithful paired runner, frozen revisions, and exact project "
                    "runtime are ready; an explicit official run is still required."
                    if measurement_ready
                    else "The official quality measurement has unresolved preconditions."
                )
            ),
            "required_resolution": (
                quality_blocker_required_resolution(quality_blocker)
                if quality_blocker is not None
                else (
                    "Run ./.venv/bin/python tools/run_official_quality.py and retain the "
                    "immutable run directory and SHA256SUMS."
                    if measurement_ready
                    else "Resolve every missing prerequisite before starting official measurement."
                )
            ),
        }
        if quality_blocker is not None:
            result["known_blocker"] = quality_blocker
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        result = official_result
        shutil.copyfile(
            ROOT / official_status["result"]["path"],
            RESULTS,
        )

    preflight = {
        "schema_version": 2,
        "generated_at_utc": generated_at,
        "status": result["classification"] if official_result is not None else result["status"],
        "gate_passed": bool(result["gate_passed"]) if official_result is not None else False,
        "measurement_ready": measurement_ready,
        "measurement_started": official_result is not None,
        "required": {
            "model": "Qwen/Qwen2.5-0.5B",
            "bf16_revision": MODEL_REVISION,
            "wikitext2_perplexity_ratio_max": 1.05,
            "c4_en_512_prompt_perplexity_ratio_max": 1.05,
            "lm_eval_average_normalized_accuracy_drop_percentage_points_max": 2.0,
            "lm_eval_tasks": [
                "piqa",
                "hellaswag",
                "winogrande",
                "arc_easy",
                "arc_challenge",
            ],
        },
        "preflight": {
            "required_files": required_files,
            "environment": environment_checks,
            "fixed_point_self_test": fixed_point_self_test,
            "packages": package_checks,
            "lm_eval_tasks": task_artifacts,
            "missing_artifacts": missing,
            "operator_oracles": oracle_contract,
            "accepted_rtl": accepted_rtl,
            "piqa_hash_pins": piqa_hash_pins,
            "quality_blocker": quality_blocker_status,
            "gate_runner": artifact(Path(__file__)),
            "gate_result": artifact(RESULTS),
        },
        "official_evidence": official_status,
        "operator_oracle_manifest": oracle_contract,
        "required_resolution": (
            "No measurement action remains; independently rehash SHA256SUMS and review metrics."
            if official_result is not None
            else result["required_resolution"]
        ),
    }
    PREFLIGHT.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    result = run_gate()
    print(
        "ACE2_QUALITY_GATE "
        f"status={result.get('classification', result['status'])} "
        f"measurement_started={str(result.get('mode') == 'official').lower()} "
        f"gate_passed={str(result['gate_passed']).lower()}"
    )


if __name__ == "__main__":
    main()
