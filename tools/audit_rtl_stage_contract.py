#!/usr/bin/env python3
"""Audit the frozen public ace2_shell RTL contract without running simulation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "design/SPEC.md"
SHELL = ROOT / "rtl/ace2_shell.sv"
PACKAGE = ROOT / "rtl/ace2_pkg.sv"
MAKEFILE = ROOT / "Makefile"
BENCHMARK_INTERFACE = ROOT / "design/BENCHMARK_INTERFACE.json"
RTL_TRACEABILITY = ROOT / "design/RTL_TRACEABILITY.md"
DYNAMIC_SCALE32_RTL = ROOT / "rtl/ace2_dynamic_scale32_core.sv"
RTL_INCLUDE_ROOTS = (ROOT / "rtl", ROOT / "rtl" / "generated")
RTL_INCLUDE_PATTERN = re.compile(r'^\s*`include\s+"([^"]+)"', re.MULTILINE)

SUPERSEDED_HISTORICAL_CSR_SYMBOLS = (
    "ACE2_CSR_DESC_SIZE",
    "ACE2_CSR_LAST_RESULT",
    "ACE2_CSR_LAST_ERROR_INFO",
)
SUPERSEDED_HISTORICAL_CSR_NAMES = (
    "DESC_SIZE",
    "LAST_RESULT",
    "LAST_ERROR_INFO",
)

IMPLEMENTED_CSRS = {
    "ID": ("000", "ID"),
    "VERSION": ("008", "VERSION"),
    "CAPABILITIES": ("010", "CAPABILITIES"),
    "CONTROL": ("018", "CONTROL"),
    "STATUS": ("020", "STATUS"),
    "ERROR_STATUS": ("028", "ERROR_STATUS"),
    "INTERRUPT_ENABLE": ("030", "INTERRUPT_EN"),
    "INTERRUPT_STATUS": ("038", "INTERRUPT_ST"),
    "DESC_BASE": ("040", "DESC_BASE"),
    "DESC_HEAD": ("048", "DESC_HEAD"),
    "DESC_TAIL": ("050", "DESC_TAIL"),
    "DOORBELL": ("058", "DOORBELL"),
    "WATCHDOG_LIMIT": ("060", "WATCHDOG_LIMIT"),
    "PERF_CYCLE": ("068", "PERF_CYCLE"),
    "PERF_BYTE": ("070", "PERF_BYTE"),
    "PERF_TOKEN": ("078", "PERF_TOKEN"),
    "PERF_STALL": ("080", "PERF_STALL"),
}

DYNAMIC_SCALE32_SPEC_FRAGMENTS = (
    "### Active layer-0 Dynamic Scale32 contract",
    "Flag value `0x49`",
    "`src0=0x0000001000000000`",
    "`src1=0x0000000100000000`",
    "`src1=0x0000000100062000`",
    "`src1=0x0000000100070000`",
    "Model identity `0xadcc20785542c188`",
    "seven immutable base Scale32 records `0x00008000`",
    "accepts only deltas `[-24,+4]`",
    "outside `[-128,127]`",
    "symmetric mantissa range `[-127,127]`",
    "writes four ordered 16-byte",
)

DYNAMIC_SCALE32_TRACEABILITY_FRAGMENTS = (
    "Four exact flag-`0x49` tuples",
    "Sidecar location and four-beat fetch",
    "Fixed live shape and base scales",
    "Producer identity",
    "Byte-exact `BFP1` validation",
    "Delta/effective exponent legality",
    "Projection dynamic-to-base application",
    "RMSNorm base-to-dynamic output",
    "Generated RMSNorm output sidecar",
)

PACKAGE_INTERFACE_CONSTANTS = (
    "ACE2_HIDDEN_SIZE",
    "ACE2_VECTOR_LANES",
    "ACE2_SRAM_BANKS",
    "ACE2_SRAM_ADDR_WIDTH",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def eval_expr(expression: str, values: dict[str, int]) -> int:
    tree = ast.parse(expression.replace(" ", ""), mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        raise ValueError(f"unsupported width/default expression: {expression}")

    return visit(tree)


def source_width(range_expr: str | None, values: dict[str, int]) -> int:
    if range_expr is None:
        return 1
    high, low = (part.strip() for part in range_expr.split(":", 1))
    return eval_expr(high, values) - eval_expr(low, values) + 1


def parse_contract() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    text = SPEC.read_text()
    section = text.split("### Public `ace2_shell` parameter and port contract", 1)[1]
    parameter_text, remainder = section.split("Every public port below", 1)
    port_text = remainder.split("### Mission-specific acceptance matrix", 1)[0]

    parameters = []
    for cells in table_rows(parameter_text):
        if len(cells) < 3 or cells[0] == "Parameter":
            continue
        single_legal = re.fullmatch(r"(\d+) only", cells[2])
        binary_legal = re.fullmatch(r"(\d+) or (\d+) only", cells[2])
        if single_legal is not None:
            legal_values = [int(single_legal.group(1))]
        elif binary_legal is not None:
            legal_values = [int(binary_legal.group(1)), int(binary_legal.group(2))]
        else:
            raise ValueError(f"unsupported legal parameter cell: {cells[2]}")
        parameters.append(
            {"name": cells[0], "default": cells[1], "legal_values": legal_values}
        )

    ports = []
    for cells in table_rows(port_text):
        if len(cells) < 3 or cells[0] == "Port":
            continue
        ports.append({"name": cells[0], "direction": cells[1], "width": cells[2]})
    return parameters, ports


def parse_source() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    text = SHELL.read_text()
    header = text.split("module ace2_shell #(", 1)[1].split(");", 1)[0]
    parameter_text, port_text = header.split(") (", 1)

    parameters = [
        {"name": match.group(1), "default": match.group(2).strip()}
        for match in re.finditer(
            r"parameter\s+integer\s+(\w+)\s*=\s*([^,\n]+)", parameter_text
        )
    ]
    ports = []
    pattern = re.compile(
        r"\s*(input|output)\s+(?:wire|reg|logic)\s+"
        r"(signed\s+)?(?:\[([^\]]+)\]\s+)?(\w+)\s*,?\s*$"
    )
    for line in port_text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        direction, signed, range_expr, name = match.groups()
        ports.append(
            {
                "name": name,
                "direction": direction,
                "range": range_expr,
                "signed": bool(signed),
            }
        )
    return parameters, ports


def resolve_rtl_include(source: Path, include: str) -> Path:
    candidates = (source.parent / include, *(root / include for root in RTL_INCLUDE_ROOTS))
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise ValueError(
            f"unresolved RTL include {include!r} from {source.relative_to(ROOT)}"
        )
    if not resolved.is_relative_to(ROOT):
        raise ValueError(
            f"RTL include escapes repository: {include!r} from {source.relative_to(ROOT)}"
        )
    return resolved


def rtl_include_closure(sources: list[Path]) -> list[Path]:
    pending = [source.resolve() for source in sources]
    closure: list[Path] = []
    seen: set[Path] = set()
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        if not path.is_file():
            raise ValueError(f"RTL input missing: {path.relative_to(ROOT)}")
        seen.add(path)
        closure.append(path)
        pending.extend(
            resolve_rtl_include(path, include)
            for include in RTL_INCLUDE_PATTERN.findall(path.read_text())
        )
    return closure


def parse_package_interface_constants() -> dict[str, int]:
    sources = rtl_include_closure([PACKAGE])
    values: dict[str, int] = {}
    for name in PACKAGE_INTERFACE_CONSTANTS:
        matches = [
            match
            for source in sources
            if (
                match := re.search(
                    rf"^\s*localparam\s+integer\s+{re.escape(name)}\s*=\s*(\d+)\s*;",
                    source.read_text(),
                    re.MULTILINE,
                )
            )
        ]
        if not matches:
            raise ValueError(f"package interface constant not found: {name}")
        if len(matches) != 1:
            raise ValueError(f"package interface constant is duplicated: {name}")
        values[name] = int(matches[0].group(1))
    return values


def source_manifest(patterns: tuple[str, ...]) -> tuple[list[dict[str, str]], str]:
    records = []
    paths = {
        path
        for pattern in patterns
        for path in (ROOT / "rtl").rglob(pattern)
    }
    for path in sorted(paths):
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    payload = "".join(f"{record['sha256']}  {record['path']}\n" for record in records).encode()
    return records, hashlib.sha256(payload).hexdigest()


def manifest_for_paths(paths: list[Path]) -> tuple[list[dict[str, str]], str]:
    records = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path in sorted(set(paths))
    ]
    payload = "".join(f"{record['sha256']}  {record['path']}\n" for record in records).encode()
    return records, hashlib.sha256(payload).hexdigest()


def make_rtl_sources() -> list[Path]:
    match = re.search(r"^RTL_SOURCES\s*:=\s*(.+)$", MAKEFILE.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError("Makefile RTL_SOURCES assignment not found")
    paths = [ROOT / item for item in match.group(1).split()]
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Makefile RTL_SOURCES paths missing: {missing}")
    return paths


def active_rtl_input_closure() -> tuple[list[dict[str, str]], str]:
    return manifest_for_paths(rtl_include_closure(make_rtl_sources()))


def audit() -> dict[str, object]:
    shell_text = SHELL.read_text()
    package_text = PACKAGE.read_text()
    spec_text = SPEC.read_text()
    traceability_text = RTL_TRACEABILITY.read_text()
    dynamic_scale32_text = DYNAMIC_SCALE32_RTL.read_text()
    spec_parameters, spec_ports = parse_contract()
    source_parameters, source_ports = parse_source()
    benchmark_interface = json.loads(BENCHMARK_INTERFACE.read_text())
    benchmark_text = BENCHMARK_INTERFACE.read_text()
    local_contract = benchmark_interface.get("local_contract", {})
    legal_values = {
        str(item["name"]): int(item["legal_values"][0]) for item in spec_parameters
    }
    aliases = parse_package_interface_constants()
    expression_values = legal_values | aliases

    parameter_mismatches = []
    for expected, actual in zip(spec_parameters, source_parameters):
        actual_value = eval_expr(str(actual["default"]), expression_values)
        if (
            expected["name"] != actual["name"]
            or actual_value not in expected["legal_values"]
        ):
            parameter_mismatches.append({"spec": expected, "source": actual, "source_value": actual_value})

    port_mismatches = []
    for expected, actual in zip(spec_ports, source_ports):
        expected_width = eval_expr(str(expected["width"]), expression_values)
        actual_width = source_width(actual["range"], expression_values)
        if (
            expected["name"] != actual["name"]
            or expected["direction"] != actual["direction"]
            or expected_width != actual_width
            or actual["signed"]
        ):
            port_mismatches.append(
                {
                    "spec": expected,
                    "source": actual,
                    "spec_width_value": expected_width,
                    "source_width_value": actual_width,
                }
            )

    sv_records, sv_manifest_sha256 = source_manifest(("*.sv",))
    repository_records, repository_manifest_sha256 = source_manifest(("*.sv", "*.svh"))
    active_records, active_manifest_sha256 = active_rtl_input_closure()
    csr_section = spec_text.split(
        "### Current implemented CSR protocol and register map", 1
    )[1].split("### Public `ace2_shell` parameter and port contract", 1)[0]
    csr_rows_present = all(
        f"| `0x{offset}` | `{name}` |" in csr_section
        for name, (offset, _) in IMPLEMENTED_CSRS.items()
    )
    package_csr_offsets_match = all(
        re.search(
            rf"localparam\s+\[31:0\]\s+ACE2_CSR_{re.escape(package_name)}\s*=\s*32'h{offset}\s*;",
            package_text,
        )
        is not None
        for offset, package_name in IMPLEMENTED_CSRS.values()
    )
    traceability_maps_every_csr = all(
        f"`{name}` `0x{offset}`" in traceability_text
        for name, (offset, _) in IMPLEMENTED_CSRS.items()
    )
    dynamic_scale32_contract_present = all(
        fragment in spec_text for fragment in DYNAMIC_SCALE32_SPEC_FRAGMENTS
    )
    dynamic_scale32_traceability_present = all(
        fragment in traceability_text
        for fragment in DYNAMIC_SCALE32_TRACEABILITY_FRAGMENTS
    )
    dynamic_scale32_source_matches_contract = all(
        fragment in shell_text
        for fragment in (
            "localparam [7:0] DYNAMIC_SCALE32_FROZEN_FLAGS = 8'h49;",
            "wire rms_dynamic_initial_buf_w",
            "wire proj_dynamic_q_buf_w",
            "wire proj_dynamic_k_buf_w",
            "wire proj_dynamic_v_buf_w",
            "wire [38*32-1:0] dynamic_initial_base_scales_w =",
            "{38{32'h0000_8000}}",
            ".expected_group_lanes_u8_i(8'd128)",
            ".expected_group_count_u6_i(6'd7)",
            ".expected_tensor_elements_u32_i(32'd896)",
            ".expected_model_identity_u64_i(64'hadcc_2078_5542_c188)",
            "function automatic [8:0] dynamic_apply_delta_s8;",
            "function automatic [8:0] dynamic_quantize_delta_s8;",
            "dynamic_output_sidecar_w[191:128] = 64'hadcc_2078_5542_c188;",
            "dynamic_output_sidecar_w[247:192] = dynamic_sidecar_q[247:192];",
        )
    ) and all(
        fragment in dynamic_scale32_text
        for fragment in (
            "sidecar_i[31:0] != 32'h3150_4642",
            "sidecar_i[39:32] != 8'd1",
            "sidecar_i[511:496] != 16'h0000",
            "delta < -8'sd24",
            "effective_exponent > 9'sd4",
        )
    )
    implementation_contract_checks = {
        "completion_sumsq_uses_latched_field": bool(
            re.search(
                r"assign\s+cmd_done_sumsq_o\s*=.*done_valid_q.*"
                r"OP_KIND_RMSNORM.*done_sumsq_q",
                shell_text,
                re.DOTALL,
            )
        ),
        "completion_inv_rms_uses_latched_field": bool(
            re.search(
                r"assign\s+cmd_done_inv_rms_q30_o\s*=.*done_valid_q.*"
                r"OP_KIND_RMSNORM.*done_inv_rms_q30_q",
                shell_text,
                re.DOTALL,
            )
        ),
        "rms_completion_fields_captured": (
            "done_sumsq_q <= core_sumsq_w;" in shell_text
            and "done_inv_rms_q30_q <= core_inv_w;" in shell_text
        ),
        "descriptor_head_tail_are_32_bit": (
            "reg [31:0] desc_head_q;" in shell_text
            and "reg [31:0] desc_tail_q;" in shell_text
        ),
        "descriptor_tail_increment_state_declared": (
            "reg desc_tail_inc_q;" in shell_text
        ),
        "no_orphan_provisional_csr_or_completion_state": not any(
            token in shell_text
            for token in (
                "reg [12:0] desc_size_q;",
                "reg [63:0] last_result_q;",
                "reg [63:0] last_error_info_q;",
                "reg done_recorded_q;",
                "reg [7:0] active_opcode_q;",
                "reg [7:0] result_code_q;",
            )
        ),
        "canonical_contract_omits_superseded_historical_csrs": not any(
            token in spec_text or token in benchmark_text
            for token in SUPERSEDED_HISTORICAL_CSR_NAMES
        ),
        "superseded_historical_csr_symbols_absent_from_rtl": not any(
            token in shell_text or token in package_text
            for token in SUPERSEDED_HISTORICAL_CSR_SYMBOLS
        ),
        "canonical_contract_maps_all_implemented_csrs": csr_rows_present,
        "package_implemented_csr_offsets_match_contract": package_csr_offsets_match,
        "traceability_maps_all_implemented_csrs": traceability_maps_every_csr,
        "canonical_contract_disclaims_autonomous_descriptor_ring": (
            "There is no CSR-driven" in csr_section
            and "descriptor fetch engine" in csr_section
            and "bit 3 is exactly `!cmd_valid_i`" in csr_section
        ),
        "csr_response_error_holds_under_backpressure": bool(
            re.search(
                r"if\s*\(csr_rvalid_o\s*&&\s*csr_rready_i\)\s*begin\s*"
                r"csr_rvalid_o\s*<=\s*1'b0;\s*"
                r"csr_error_o\s*<=\s*1'b0;\s*end",
                shell_text,
                re.DOTALL,
            )
        ),
        "csr_unknown_address_sets_response_error": (
            "csr_error_o <= !csr_known_addr(csr_req_addr_low_q, csr_req_high_zero_q);"
            in shell_text
        ),
        "canonical_dynamic_scale32_contract_present": dynamic_scale32_contract_present,
        "dynamic_scale32_traceability_present": dynamic_scale32_traceability_present,
        "dynamic_scale32_source_matches_contract": dynamic_scale32_source_matches_contract,
    }
    passed = (
        len(spec_parameters) == len(source_parameters) == 14
        and len(spec_ports) == len(source_ports) == 64
        and not parameter_mismatches
        and not port_mismatches
        and benchmark_interface.get("applies") is False
        and benchmark_interface.get("external_contract") is None
        and local_contract.get("top_module") == "ace2_shell"
        and local_contract.get("public_interface_source")
        == "design/SPEC.md#public-ace2_shell-parameter-and-port-contract"
        and all(implementation_contract_checks.values())
    )
    return {
        "schema_version": 1,
        "scope": "frozen ace2_shell parameter/port contract and current RTL source binding only",
        "specification": "design/SPEC.md#public-ace2_shell-parameter-and-port-contract",
        "source": "rtl/ace2_shell.sv",
        "input_bindings": {
            "design/BENCHMARK_INTERFACE.json": sha256(BENCHMARK_INTERFACE),
            "design/RTL_TRACEABILITY.md": sha256(RTL_TRACEABILITY),
            "design/SPEC.md": sha256(SPEC),
            "Makefile": sha256(MAKEFILE),
            "rtl/ace2_dynamic_scale32_core.sv": sha256(DYNAMIC_SCALE32_RTL),
            "rtl/ace2_shell.sv": sha256(SHELL),
            **{
                path.relative_to(ROOT).as_posix(): sha256(path)
                for path in rtl_include_closure([PACKAGE])
            },
        },
        "checks": {
            "benchmark_applies": benchmark_interface.get("applies"),
            "benchmark_external_contract": benchmark_interface.get("external_contract"),
            "benchmark_local_top_module": local_contract.get("top_module"),
            "benchmark_public_interface_source": local_contract.get(
                "public_interface_source"
            ),
            "package_interface_constants": aliases,
            "parameter_count_spec": len(spec_parameters),
            "parameter_count_source": len(source_parameters),
            "port_count_spec": len(spec_ports),
            "port_count_source": len(source_ports),
            "parameter_mismatches": parameter_mismatches,
            "port_mismatches": port_mismatches,
            "implementation_contract_checks": implementation_contract_checks,
        },
        "rtl_tree": {
            "definition": "compatibility manifest over rtl/**/*.sv",
            "file_count": len(sv_records),
            "manifest_sha256": sv_manifest_sha256,
            "source_hashes": sv_records,
        },
        "active_rtl_input_closure": {
            "definition": (
                "Makefile RTL_SOURCES compilation inputs plus recursively resolved local "
                "`include files; this set may contain modules outside the elaborated ace2_shell hierarchy"
            ),
            "file_count": len(active_records),
            "manifest_sha256": active_manifest_sha256,
            "source_hashes": active_records,
        },
        "rtl_repository_inventory": {
            "definition": "repository inventory over rtl/**/*.sv and rtl/**/*.svh, including inactive diagnostic candidates",
            "file_count": len(repository_records),
            "manifest_sha256": repository_manifest_sha256,
            "source_hashes": repository_records,
        },
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, sort_keys=True, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
