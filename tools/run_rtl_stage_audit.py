#!/usr/bin/env python3
"""Run the current-stage ACE-2 RTL contract, lint, and elaboration audit.

This runner intentionally performs no simulation, formal proof, synthesis,
timing, PPA, benchmark, runtime, or FPGA flow. Each invocation creates a new
immutable output directory unless the caller supplies a new explicit path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from audit_rtl_stage_contract import audit, make_rtl_sources


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STATE = ROOT / "research/PIPELINE_STATE.json"
RTL_STAGE_AUDIT = ROOT / "research/RTL_STAGE_AUDIT.md"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
RTL_TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
BENCHMARK_INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
DYNAMIC_SCALE32_TOPS = (
    "ace2_dynamic_scale32_group_core",
    "ace2_dynamic_scale32_sidecar_builder_core",
    "ace2_dynamic_scale32_sidecar_validator_core",
    "ace2_scale32_tagged_accumulator_core",
)
DISALLOWED_ACTIVE_LINT_WARNINGS = {
    "ALWCOMBORDER",
    "CASEINCOMPLETE",
    "COMBDLY",
    "IMPLICIT",
    "LATCH",
    "MULTIDRIVEN",
    "PINCONNECTEMPTY",
    "PINMISSING",
    "SELRANGE",
    "SYNCASYNCNET",
    "UNDRIVEN",
    "UNOPTFLAT",
    "WIDTH",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def command_text(argv: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in argv)


def run_command(
    argv: list[str],
    *,
    log_path: Path,
    records: list[dict[str, object]],
    heading: str | None = None,
    append: bool = False,
) -> int:
    started = utc_now()
    completed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
    mode = "a" if append else "w"
    with log_path.open(mode) as handle:
        if heading is not None:
            handle.write(f"{heading}\n")
        handle.write(completed.stdout)
        handle.write(completed.stderr)
    records.append(
        {
            "argv": argv,
            "command": command_text(argv),
            "cwd": ROOT.as_posix(),
            "exit_status": completed.returncode,
            "finished_at_utc": utc_now(),
            "log": log_path.relative_to(ROOT).as_posix(),
            "started_at_utc": started,
        }
    )
    return completed.returncode


def tool_version(executable: str, args: list[str]) -> dict[str, object]:
    resolved = shutil.which(executable)
    if resolved is None:
        return {"executable": executable, "status": "NOT_FOUND"}
    completed = subprocess.run(
        [executable, *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return {
        "argv": [executable, *args],
        "executable": resolved,
        "exit_status": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "status": "RECORDED" if completed.returncode == 0 else "VERSION_COMMAND_FAILED",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def current_tree_binding_audit(interface: dict[str, object]) -> dict[str, object]:
    manifest = load_json(ROOT / "design/RTL_MANIFEST.json")
    benchmark = load_json(BENCHMARK_INTERFACE)
    traceability = RTL_TRACEABILITY.read_text()
    active = interface["active_rtl_input_closure"]
    manifest_binding = manifest["current_rtl_stage_binding"]
    manifest_closure = manifest_binding["active_rtl_input_closure"]
    benchmark_closure = benchmark["active_rtl_input_closure"]
    public_interface = interface["checks"]
    source_hashes = {
        record["path"]: record["sha256"] for record in active["source_hashes"]
    }
    checks = {
        "active_closure_is_exact_18_file_audited_closure": (
            active["file_count"] == 18
            and active["manifest_sha256"]
            == "54a9b34e502928eaf7515a64f167ce00e2206624a1de1dcf6ad6434fe5419fbc"
        ),
        "manifest_closure_matches_fresh_enumeration": manifest_closure == active,
        "benchmark_closure_matches_fresh_enumeration": (
            benchmark_closure["file_count"] == active["file_count"]
            and benchmark_closure["manifest_sha256"] == active["manifest_sha256"]
        ),
        "public_interface_is_exact_14_parameter_64_port_contract": (
            public_interface["parameter_count_source"] == 14
            and public_interface["port_count_source"] == 64
        ),
        "shell_hash_matches_proven_sidecar_integration": (
            source_hashes.get("rtl/ace2_shell.sv")
            == "77eadf2546f36672a3b76c67ab536e7962de5fbed1069804befc7f96627281bc"
            and benchmark["module"]["source"]["sha256"]
            == source_hashes.get("rtl/ace2_shell.sv")
            and manifest_binding["repaired_source"]["sha256"]
            == source_hashes.get("rtl/ace2_shell.sv")
        ),
        "package_hash_matches_current_rtl": (
            benchmark["module"]["package_source"]["sha256"]
            == sha256(ROOT / "rtl/ace2_pkg.sv")
        ),
        "layer23_v_sidecar_hash_matches_proven_source": (
            source_hashes.get(
                "rtl/ace2_layer23_v_rank1_integer_correction_sidecar.sv"
            )
            == "5851d5fe3d9473ae38837dc27b668d7cd2b403dd7f52256bdab2eeca381be5f8"
        ),
        "layer23_v_payload_hash_matches_proven_source": (
            source_hashes.get(
                "rtl/generated/ace2_layer23_v_rank1_integer_correction_payload.svh"
            )
            == "145a8d57b8a92e55390f912dac14e7b72ce2208300fed979d8376b60744ebd85"
        ),
        "traceability_preserves_proven_layer23_v_semantics": all(
            phrase in traceability
            for phrase in (
                "896-element signed-int8 V input",
                "128-element signed-int8 correction output",
                "eight ordered 16-lane output beats",
                "V-only operation",
                "hard/soft-reset isolation",
                "ENABLE_LAYER23_V_RANK1_SIDECAR=0",
                "LAYER23_V_RANK1_CONFIG_VALID=0",
            )
        ),
    }
    return {
        "checks": checks,
        "proven_semantics_binding": {
            "activation_elements": 896,
            "activation_width_bits": 8,
            "output_elements": 128,
            "output_width_bits": 8,
            "beat_count": 8,
            "lanes_per_beat": 16,
            "operation": "V_ONLY",
            "reset_isolation": ["hard_reset", "soft_reset"],
        },
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def bind_package(package_dir: Path | None) -> dict[str, object]:
    if package_dir is None:
        return {"status": "NOT_BOUND"}
    resolved = package_dir.resolve()
    if (
        not resolved.is_relative_to(ROOT)
        or resolved.is_symlink()
        or not resolved.is_dir()
    ):
        return {"status": "FAIL"}
    sums_path = resolved / "SHA256SUMS"
    root_path = resolved / "TREE_ROOT.sha256"
    if not sums_path.is_file() or not root_path.is_file():
        return {"status": "FAIL"}
    manifest_sha256 = sha256(sums_path)
    expected_root = f"{manifest_sha256}  SHA256SUMS"
    return {
        "package_path": resolved.relative_to(ROOT).as_posix(),
        "package_tree_root": manifest_sha256,
        "source_bindings_sha256": sha256(resolved / "source-bindings.sha256"),
        "source_closure_sha256": sha256(resolved / "source-closure.json"),
        "status": (
            "PASS"
            if root_path.read_text(encoding="ascii").strip() == expected_root
            else "FAIL"
        ),
    }


def make_stage_gate(
    pipeline: dict[str, object],
    *,
    manager_approved_read_only_current_tree_audit: bool,
) -> dict[str, object]:
    stages = pipeline.get("stages", {})
    specification = stages.get("specification", {}) if isinstance(stages, dict) else {}
    rtl_stage = stages.get("rtl", {}) if isinstance(stages, dict) else {}
    current_stage = pipeline.get("current_stage")
    nested_status_consistent = (
        isinstance(specification, dict)
        and specification.get("status") == "done"
        and isinstance(rtl_stage, dict)
        and rtl_stage.get("status") in {"in_progress", "authorized"}
    )
    approved_diagnosis_audit = (
        manager_approved_read_only_current_tree_audit
        and current_stage == "stage1_software_diagnosis"
    )
    return {
        "current_stage": current_stage,
        "default_expected_stage": "rtl",
        "manager_approved_read_only_current_tree_audit": (
            manager_approved_read_only_current_tree_audit
        ),
        "manager_approved_stage_exception_applied": approved_diagnosis_audit,
        "nested_specification_status": (
            specification.get("status") if isinstance(specification, dict) else None
        ),
        "nested_rtl_status": (
            rtl_stage.get("status") if isinstance(rtl_stage, dict) else None
        ),
        "manager_owned_nested_status_consistent": nested_status_consistent,
        "manager_reconciliation_required": not nested_status_consistent,
        "project_stage_mutated": False,
        "selection_authority": (
            "top-level current_stage; explicit Manager-approved read-only audit mode "
            "permits stage1_software_diagnosis without changing project stage"
        ),
        "status": "PASS"
        if current_stage == "rtl" or approved_diagnosis_audit
        else "FAIL",
    }


def warning_records(path: Path) -> list[dict[str, object]]:
    records = []
    pattern = re.compile(
        r"^%Warning-([A-Z0-9_]+):\s+([^:\n]+):(\d+):(\d+):", re.MULTILINE
    )
    for warning_class, source, line, column in pattern.findall(path.read_text()):
        records.append(
            {
                "class": warning_class,
                "column": int(column),
                "line": int(line),
                "source": source,
            }
        )
    return records


def warning_counts(path: Path) -> dict[str, int]:
    counts = collections.Counter(record["class"] for record in warning_records(path))
    return dict(sorted(counts.items()))


def rejected_dynamic_rope_disposition(
    repository_warning_records: list[dict[str, object]],
    elaborated_modules: list[str],
) -> dict[str, object]:
    manifest_path = ROOT / "design/RTL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = "rtl/ace2_dynamic_rope_head_core.sv"
    diagnostic = next(
        item
        for item in manifest["diagnostic_rtl_sources"]
        if item.get("path") == source_path
    )
    review_path = ROOT / diagnostic["review"]
    review = json.loads(review_path.read_text())
    disallowed_records = [
        record
        for record in repository_warning_records
        if record["class"] in DISALLOWED_ACTIVE_LINT_WARNINGS
    ]
    disallowed_sources = sorted({str(record["source"]) for record in disallowed_records})
    checks = {
        "current_source_hash_matches_manifest": (
            sha256(ROOT / source_path) == diagnostic["sha256"]
        ),
        "disallowed_repository_warnings_confined_to_rejected_source": (
            disallowed_sources == [source_path]
        ),
        "rejected_module_not_in_elaborated_shell_hierarchy": (
            diagnostic["module"] not in elaborated_modules
        ),
        "manifest_status": diagnostic["status"],
        "review_rtl_modification_authorized": review["next_task_constraints"][
            "rtl_modification_authorized"
        ],
        "review_status": review["decision"]["status"],
    }
    passed = (
        checks["current_source_hash_matches_manifest"]
        and checks["disallowed_repository_warnings_confined_to_rejected_source"]
        and checks["rejected_module_not_in_elaborated_shell_hierarchy"]
        and checks["manifest_status"] == "rejected_diagnostic_not_in_accepted_shell_frontier"
        and checks["review_status"] == "done"
        and checks["review_rtl_modification_authorized"] is False
    )
    return {
        "checks": checks,
        "disallowed_warning_records": disallowed_records,
        "review": review_path.relative_to(ROOT).as_posix(),
        "review_sha256": sha256(review_path),
        "source": source_path,
        "source_sha256": sha256(ROOT / source_path),
        "status": "PASS" if passed else "FAIL",
    }


def write_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "SHA256SUMS"
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        records.append(f"{sha256(path)}  {path.name}\n")
    manifest_path.write_text("".join(records))
    (output_dir / "SHA256SUMS.sha256").write_text(
        f"{sha256(manifest_path)}  SHA256SUMS\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument(
        "--manager-approved-read-only-current-tree-audit",
        action="store_true",
        help=(
            "permit this read-only audit at stage1_software_diagnosis without "
            "changing the Manager-owned project stage"
        ),
    )
    args = parser.parse_args()

    interface = audit()
    current_binding = current_tree_binding_audit(interface)
    package_binding = bind_package(args.package_dir)
    pipeline = load_json(PIPELINE_STATE)
    stage_gate = make_stage_gate(
        pipeline,
        manager_approved_read_only_current_tree_audit=(
            args.manager_approved_read_only_current_tree_audit
        ),
    )
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        closure = str(interface["active_rtl_input_closure"]["manifest_sha256"])
        output_dir = ROOT / "build" / f"rtl-stage-audit-{timestamp}-{closure[:12]}"
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    write_json(output_dir / "interface_audit.json", interface)
    versions = {
        "generated_at_utc": utc_now(),
        "iverilog": tool_version(args.iverilog, ["-V"]),
        "verilator": tool_version(args.verilator, ["--version"]),
    }
    write_json(output_dir / "tool_versions.json", versions)

    records: list[dict[str, object]] = []
    relative_output = output_dir.relative_to(ROOT)
    rtl_sources = [path.relative_to(ROOT).as_posix() for path in make_rtl_sources()]
    iverilog_status = run_command(
        [
            args.iverilog,
            "-g2012",
            "-Wall",
            "-Irtl",
            "-s",
            "ace2_shell",
            "-o",
            (relative_output / "ace2_shell_elaboration.vvp").as_posix(),
            *rtl_sources,
        ],
        log_path=output_dir / "iverilog_elaboration.log",
        records=records,
    )
    shell_lint_status = run_command(
        [
            args.verilator,
            "--lint-only",
            "--language",
            "1800-2017",
            "-Wall",
            "-Wno-fatal",
            "-Irtl",
            "--top-module",
            "ace2_shell",
            *rtl_sources,
        ],
        log_path=output_dir / "verilator_shell_lint.log",
        records=records,
    )
    hierarchy_status = run_command(
        [
            args.verilator,
            "--xml-only",
            "--language",
            "1800-2017",
            "-Irtl",
            "--top-module",
            "ace2_shell",
            "--xml-output",
            (relative_output / "ace2_shell_hierarchy.xml").as_posix(),
            *rtl_sources,
        ],
        log_path=output_dir / "verilator_shell_hierarchy.log",
        records=records,
    )

    dynamic_statuses = []
    dynamic_log = output_dir / "verilator_dynamic_scale32_lint.log"
    for index, top in enumerate(DYNAMIC_SCALE32_TOPS):
        status = run_command(
            [
                args.verilator,
                "--lint-only",
                "--language",
                "1800-2017",
                "-Wall",
                "-Wno-fatal",
                "--top-module",
                top,
                "rtl/ace2_dynamic_scale32_core.sv",
            ],
            log_path=dynamic_log,
            records=records,
            heading=f"TOP={top}",
            append=index != 0,
        )
        dynamic_statuses.append({"top": top, "exit_status": status})

    repository_lint_status = run_command(
        [
            args.verilator,
            "--lint-only",
            "--language",
            "1800-2017",
            "-Wall",
            "-Wno-fatal",
            "-Wno-MULTITOP",
            "-Irtl",
            "-Irtl/generated",
            *[
                path.relative_to(ROOT).as_posix()
                for path in sorted((ROOT / "rtl").glob("*.sv"))
            ],
        ],
        log_path=output_dir / "verilator_repository_rtl_lint.log",
        records=records,
    )

    with (output_dir / "commands.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    shell_warnings = warning_counts(output_dir / "verilator_shell_lint.log")
    dynamic_warnings = warning_counts(dynamic_log)
    repository_lint_log = output_dir / "verilator_repository_rtl_lint.log"
    repository_warning_records = warning_records(repository_lint_log)
    repository_warnings = warning_counts(repository_lint_log)
    hierarchy_xml = output_dir / "ace2_shell_hierarchy.xml"
    elaborated_modules: list[str] = []
    if hierarchy_status == 0 and hierarchy_xml.is_file():
        for module in ET.parse(hierarchy_xml).getroot().findall(".//module"):
            name = module.get("origName") or module.get("name")
            if name and name not in elaborated_modules:
                elaborated_modules.append(name)
    repository_warning_disposition = rejected_dynamic_rope_disposition(
        repository_warning_records,
        elaborated_modules,
    )
    active_disallowed = sorted(
        (set(shell_warnings) | set(dynamic_warnings)) & DISALLOWED_ACTIVE_LINT_WARNINGS
    )
    checks = {
        "active_disallowed_lint_warning_classes": active_disallowed,
        "interface_contract": interface["status"],
        "current_tree_binding": current_binding,
        "package_binding": package_binding,
        "iverilog_elaboration_exit_status": iverilog_status,
        "stage_gate": stage_gate,
        "verilator_elaborated_shell_modules": elaborated_modules,
        "verilator_shell_hierarchy_exit_status": hierarchy_status,
        "verilator_active_dynamic_scale32_warning_counts": dynamic_warnings,
        "verilator_active_shell_warning_counts": shell_warnings,
        "verilator_dynamic_scale32_lint": dynamic_statuses,
        "verilator_repository_lint_exit_status": repository_lint_status,
        "verilator_repository_warning_disposition": repository_warning_disposition,
        "verilator_repository_warning_counts": repository_warnings,
        "verilator_shell_lint_exit_status": shell_lint_status,
    }
    passed = (
        interface["status"] == "PASS"
        and current_binding["status"] == "PASS"
        and (
            not args.manager_approved_read_only_current_tree_audit
            or package_binding["status"] == "PASS"
        )
        and stage_gate["status"] == "PASS"
        and iverilog_status == 0
        and shell_lint_status == 0
        and hierarchy_status == 0
        and "ace2_shell" in elaborated_modules
        and not active_disallowed
        and all(item["exit_status"] == 0 for item in dynamic_statuses)
        and repository_lint_status == 0
        and repository_warning_disposition["status"] == "PASS"
    )
    results = {
        "audit_input_bindings": {
            "Makefile": sha256(ROOT / "Makefile"),
            "design/BENCHMARK_INTERFACE.json": sha256(BENCHMARK_INTERFACE),
            "design/RTL_MANIFEST.json": sha256(ROOT / "design/RTL_MANIFEST.json"),
            "design/RTL_TRACEABILITY.md": sha256(RTL_TRACEABILITY),
            "design/SPEC.md": sha256(ROOT / "design/SPEC.md"),
            "evidence/review/dynamic_rope_head_scale_v1_no_go_l2/decision.json": sha256(
                ROOT / "evidence/review/dynamic_rope_head_scale_v1_no_go_l2/decision.json"
            ),
            "research/GROUND_TRUTH.md": sha256(GROUND_TRUTH),
            "research/PIPELINE_STATE.json": sha256(PIPELINE_STATE),
            "research/RTL_STAGE_AUDIT.md": sha256(RTL_STAGE_AUDIT),
            "tools/audit_rtl_stage_contract.py": sha256(
                ROOT / "tools/audit_rtl_stage_contract.py"
            ),
            "tools/run_rtl_stage_audit.py": sha256(ROOT / "tools/run_rtl_stage_audit.py"),
        },
        "package_binding": package_binding,
        "checks": checks,
        "completed_at_utc": utc_now(),
        "rtl_compilation_input_closure_file_count": interface["active_rtl_input_closure"]["file_count"],
        "rtl_compilation_input_closure_manifest_sha256": interface["active_rtl_input_closure"]["manifest_sha256"],
        "active_rtl_input_closure_file_count": interface["active_rtl_input_closure"]["file_count"],
        "active_rtl_input_closure_manifest_sha256": interface["active_rtl_input_closure"]["manifest_sha256"],
        "rtl_repository_inventory_file_count": interface["rtl_repository_inventory"]["file_count"],
        "rtl_repository_inventory_manifest_sha256": interface["rtl_repository_inventory"]["manifest_sha256"],
        "rtl_sv_compatibility_file_count": interface["rtl_tree"]["file_count"],
        "rtl_sv_compatibility_manifest_sha256": interface["rtl_tree"]["manifest_sha256"],
        "schema_version": 2,
        "scope": (
            "current RTL-stage interface, lint, and elaboration only; no simulation, "
            "formal, synthesis, timing, PPA, benchmark, runtime, FPGA, or downstream-stage claim"
        ),
        "status": "PASS" if passed else "FAIL",
    }
    write_json(output_dir / "results.json", results)
    audit_verdict = {
        "checks": {
            "bounded_lint_and_compile_only_audit": passed,
            "current_tree_binding": current_binding["status"],
            "interface_contract": interface["status"],
            "package_binding": package_binding["status"],
            "stage_gate": stage_gate["status"],
        },
        "completed_at_utc": results["completed_at_utc"],
        "execution_authority_created": False,
        "model_or_rtl_workload_executed": False,
        "package_binding": package_binding,
        "review_created": False,
        "scope": results["scope"],
        "verdict": results["status"],
    }
    write_json(output_dir / "AUDIT_VERDICT.json", audit_verdict)
    write_manifest(output_dir)
    print(json.dumps({"output_dir": relative_output.as_posix(), **results}, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
