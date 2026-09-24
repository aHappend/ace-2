#!/usr/bin/env python3
"""Generate focused vectors for the layer-23 rank-1 integer correction RTL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ace2_layer23_v_rank1_integer_correction_reference import (
    CONTRACT_ID,
    INPUT_WIDTH,
    OUTPUT_WIDTH,
    POSITION_COUNT,
    apply_rank1_correction,
    load_frozen_config,
    load_frozen_position_vectors,
    load_generalization_position_vectors,
    pack_s8,
    sha256_bytes,
    sha256_file,
    synthetic_cases,
    verify_generalization_vectors,
    verify_frozen_vectors,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "verification/generated/layer23_v_rank1_integer_correction"
OUT_JSON = ROOT / "verification/generated/layer23_v_rank1_integer_correction_vectors.json"
OUT_SVH = ROOT / "verification/generated/layer23_v_rank1_integer_correction_vectors.svh"
OUT_SYNTHETIC_SVH = (
    ROOT / "verification/generated/layer23_v_rank1_integer_correction_synthetic_cases.svh"
)
OUT_PAYLOAD_SVH = ROOT / "rtl/generated/ace2_layer23_v_rank1_integer_correction_payload.svh"


def sv_signed(width: int, value: int) -> str:
    return f"-{width}'sd{abs(value)}" if value < 0 else f"{width}'sd{value}"


def hex_lines(values: Iterable[int], bits: int) -> str:
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    return "".join(f"{int(value) & mask:0{digits}x}\n" for value in values)


def sv_unsigned(width: int, value: int) -> str:
    return f"{width}'d{int(value)}"


def sv_packed_hex(values: Iterable[int], bits: int) -> str:
    masked = [(int(value) & ((1 << bits) - 1)) for value in values]
    digits = (bits + 3) // 4
    return "_".join(f"{value:0{digits}x}" for value in reversed(masked))


def write_text(path: Path, text: str) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def svh_case_function(
    *,
    name: str,
    return_type: str,
    index_type: str,
    index_width: int,
    values: Iterable[int],
    formatter,
) -> list[str]:
    lines = [f"function automatic {return_type} {name}(input {index_type} index);", "    begin", "        case (index)"]
    for index, value in enumerate(values):
        lines.append(f"            {index_width}'d{index}: {name} = {formatter(value)};")
    lines.extend([f"            default: {name} = {formatter(0)};", "        endcase", "    end", "endfunction", ""])
    return lines


def main() -> int:
    summary = verify_frozen_vectors()
    generalization_summary = verify_generalization_vectors()
    config = load_frozen_config()
    vectors = load_frozen_position_vectors()
    generalization_vectors = load_generalization_position_vectors()

    flattened_input = [value for vector in vectors for value in vector.input_s8]
    flattened_baseline = [value for vector in vectors for value in vector.baseline_v_s8]
    flattened_rank_acc = [vector.expected_rank_accumulator_s32 for vector in vectors]
    flattened_rank_rounded = [vector.expected_rank_rounded_s32 for vector in vectors]
    flattened_rank_s8 = [vector.expected_rank_intermediate_s8 for vector in vectors]
    flattened_corr_acc = [
        value for vector in vectors for value in vector.expected_correction_accumulator_s32
    ]
    flattened_corr_rounded = [
        value for vector in vectors for value in vector.expected_correction_rounded_s32
    ]
    flattened_corr_s8 = [value for vector in vectors for value in vector.expected_correction_s8]
    flattened_corrected_s8 = [
        value for vector in vectors for value in vector.expected_corrected_v_s8
    ]
    flattened_generalization_input = [
        value for named in generalization_vectors for value in named.vector.input_s8
    ]
    flattened_generalization_baseline = [
        value for named in generalization_vectors for value in named.vector.baseline_v_s8
    ]
    flattened_generalization_rank_acc = [
        named.vector.expected_rank_accumulator_s32 for named in generalization_vectors
    ]
    flattened_generalization_rank_rounded = [
        named.vector.expected_rank_rounded_s32 for named in generalization_vectors
    ]
    flattened_generalization_rank_s8 = [
        named.vector.expected_rank_intermediate_s8 for named in generalization_vectors
    ]
    flattened_generalization_corr_acc = [
        value
        for named in generalization_vectors
        for value in named.vector.expected_correction_accumulator_s32
    ]
    flattened_generalization_corr_rounded = [
        value
        for named in generalization_vectors
        for value in named.vector.expected_correction_rounded_s32
    ]
    flattened_generalization_corr_s8 = [
        value for named in generalization_vectors for value in named.vector.expected_correction_s8
    ]
    flattened_generalization_corrected_s8 = [
        value for named in generalization_vectors for value in named.vector.expected_corrected_v_s8
    ]

    generated = {
        "input_to_rank_s8_hex": write_text(
            OUT_DIR / "input_to_rank_s8.hex",
            hex_lines(config.input_to_rank_s8, 8),
        ),
        "rank_to_channel_s8_hex": write_text(
            OUT_DIR / "rank_to_channel_s8.hex",
            hex_lines(config.rank_to_channel_s8, 8),
        ),
        "second_stage_multiplier_s32_hex": write_text(
            OUT_DIR / "second_stage_multiplier_s32.hex",
            hex_lines(config.second_stage_multiplier_s32, 32),
        ),
        "second_stage_right_shift_u6_hex": write_text(
            OUT_DIR / "second_stage_right_shift_u6.hex",
            hex_lines(config.second_stage_right_shift_u6, 8),
        ),
        "input_payload_s8_hex": write_text(
            OUT_DIR / "input_payload_s8.hex",
            hex_lines(flattened_input, 8),
        ),
        "baseline_v_output_s8_hex": write_text(
            OUT_DIR / "baseline_v_output_s8.hex",
            hex_lines(flattened_baseline, 8),
        ),
        "expected_rank_accumulator_s32_hex": write_text(
            OUT_DIR / "expected_rank_accumulator_s32.hex",
            hex_lines(flattened_rank_acc, 32),
        ),
        "expected_rank_rounded_s32_hex": write_text(
            OUT_DIR / "expected_rank_rounded_s32.hex",
            hex_lines(flattened_rank_rounded, 32),
        ),
        "expected_rank_intermediate_s8_hex": write_text(
            OUT_DIR / "expected_rank_intermediate_s8.hex",
            hex_lines(flattened_rank_s8, 8),
        ),
        "expected_correction_accumulator_s32_hex": write_text(
            OUT_DIR / "expected_correction_accumulator_s32.hex",
            hex_lines(flattened_corr_acc, 32),
        ),
        "expected_correction_rounded_s32_hex": write_text(
            OUT_DIR / "expected_correction_rounded_s32.hex",
            hex_lines(flattened_corr_rounded, 32),
        ),
        "expected_correction_s8_hex": write_text(
            OUT_DIR / "expected_correction_s8.hex",
            hex_lines(flattened_corr_s8, 8),
        ),
        "expected_corrected_v_output_s8_hex": write_text(
            OUT_DIR / "expected_corrected_v_output_s8.hex",
            hex_lines(flattened_corrected_s8, 8),
        ),
        "generalization_input_payload_s8_hex": write_text(
            OUT_DIR / "generalization_input_payload_s8.hex",
            hex_lines(flattened_generalization_input, 8),
        ),
        "generalization_baseline_v_output_s8_hex": write_text(
            OUT_DIR / "generalization_baseline_v_output_s8.hex",
            hex_lines(flattened_generalization_baseline, 8),
        ),
        "generalization_expected_rank_accumulator_s32_hex": write_text(
            OUT_DIR / "generalization_expected_rank_accumulator_s32.hex",
            hex_lines(flattened_generalization_rank_acc, 32),
        ),
        "generalization_expected_rank_rounded_s32_hex": write_text(
            OUT_DIR / "generalization_expected_rank_rounded_s32.hex",
            hex_lines(flattened_generalization_rank_rounded, 32),
        ),
        "generalization_expected_rank_intermediate_s8_hex": write_text(
            OUT_DIR / "generalization_expected_rank_intermediate_s8.hex",
            hex_lines(flattened_generalization_rank_s8, 8),
        ),
        "generalization_expected_correction_accumulator_s32_hex": write_text(
            OUT_DIR / "generalization_expected_correction_accumulator_s32.hex",
            hex_lines(flattened_generalization_corr_acc, 32),
        ),
        "generalization_expected_correction_rounded_s32_hex": write_text(
            OUT_DIR / "generalization_expected_correction_rounded_s32.hex",
            hex_lines(flattened_generalization_corr_rounded, 32),
        ),
        "generalization_expected_correction_s8_hex": write_text(
            OUT_DIR / "generalization_expected_correction_s8.hex",
            hex_lines(flattened_generalization_corr_s8, 8),
        ),
        "generalization_expected_corrected_v_output_s8_hex": write_text(
            OUT_DIR / "generalization_expected_corrected_v_output_s8.hex",
            hex_lines(flattened_generalization_corrected_s8, 8),
        ),
    }

    synthetic = list(synthetic_cases())
    payload = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "generator": "tools/gen_layer23_v_rank1_integer_correction_vectors.py",
        "geometry": {
            "positions": POSITION_COUNT,
            "input_width": INPUT_WIDTH,
            "output_width": OUTPUT_WIDTH,
            "rank": 1,
        },
        "frozen_candidate_summary": summary,
        "frozen_generalization_summary": generalization_summary,
        "generated_hex": generated,
        "synthetic_cases": synthetic,
        "byte_equality": {
            "rank_intermediate_s8": sha256_bytes(pack_s8(flattened_rank_s8)),
            "correction_s8": sha256_bytes(pack_s8(flattened_corr_s8)),
            "corrected_v_output_s8": sha256_bytes(pack_s8(flattened_corrected_s8)),
            "generalization_input_payload_s8": sha256_bytes(pack_s8(flattened_generalization_input)),
            "generalization_correction_s8": sha256_bytes(pack_s8(flattened_generalization_corr_s8)),
            "generalization_corrected_v_output_s8": sha256_bytes(
                pack_s8(flattened_generalization_corrected_s8)
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    svh = [
        "// Generated by tools/gen_layer23_v_rank1_integer_correction_vectors.py",
        f"localparam integer ACE2_R1_INPUT_WIDTH = {INPUT_WIDTH};",
        f"localparam integer ACE2_R1_OUTPUT_WIDTH = {OUTPUT_WIDTH};",
        f"localparam integer ACE2_R1_POSITION_COUNT = {POSITION_COUNT};",
        f"localparam integer ACE2_R1_GENERALIZATION_POSITION_COUNT = {len(generalization_vectors)};",
        "localparam logic [2:0] ACE2_R1_CFG_INPUT_TO_RANK = 3'd0;",
        "localparam logic [2:0] ACE2_R1_CFG_RANK_TO_CHANNEL = 3'd1;",
        "localparam logic [2:0] ACE2_R1_CFG_SECOND_MULTIPLIER = 3'd2;",
        "localparam logic [2:0] ACE2_R1_CFG_SECOND_SHIFT = 3'd3;",
        "localparam logic [2:0] ACE2_R1_CFG_FIRST_MULTIPLIER = 3'd4;",
        "localparam logic [2:0] ACE2_R1_CFG_FIRST_SHIFT = 3'd5;",
        f"localparam logic signed [31:0] ACE2_R1_FIRST_MULTIPLIER_S32 = {sv_signed(32, config.first_stage_multiplier_s32)};",
        f"localparam logic signed [31:0] ACE2_R1_FIRST_SHIFT_S32 = {sv_signed(32, config.first_stage_right_shift_u6)};",
        f"`define ACE2_R1_INPUT_TO_RANK_HEX \"{generated['input_to_rank_s8_hex']['path']}\"",
        f"`define ACE2_R1_RANK_TO_CHANNEL_HEX \"{generated['rank_to_channel_s8_hex']['path']}\"",
        f"`define ACE2_R1_SECOND_MULTIPLIER_HEX \"{generated['second_stage_multiplier_s32_hex']['path']}\"",
        f"`define ACE2_R1_SECOND_SHIFT_HEX \"{generated['second_stage_right_shift_u6_hex']['path']}\"",
        f"`define ACE2_R1_INPUT_PAYLOAD_HEX \"{generated['input_payload_s8_hex']['path']}\"",
        f"`define ACE2_R1_BASELINE_V_HEX \"{generated['baseline_v_output_s8_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_RANK_ACC_HEX \"{generated['expected_rank_accumulator_s32_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_RANK_ROUNDED_HEX \"{generated['expected_rank_rounded_s32_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_RANK_S8_HEX \"{generated['expected_rank_intermediate_s8_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_CORR_ACC_HEX \"{generated['expected_correction_accumulator_s32_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_CORR_ROUNDED_HEX \"{generated['expected_correction_rounded_s32_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_CORR_S8_HEX \"{generated['expected_correction_s8_hex']['path']}\"",
        f"`define ACE2_R1_EXPECTED_CORRECTED_S8_HEX \"{generated['expected_corrected_v_output_s8_hex']['path']}\"",
        f"`define ACE2_R1_GEN_INPUT_PAYLOAD_HEX \"{generated['generalization_input_payload_s8_hex']['path']}\"",
        f"`define ACE2_R1_GEN_BASELINE_V_HEX \"{generated['generalization_baseline_v_output_s8_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_RANK_ACC_HEX \"{generated['generalization_expected_rank_accumulator_s32_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_RANK_ROUNDED_HEX \"{generated['generalization_expected_rank_rounded_s32_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_RANK_S8_HEX \"{generated['generalization_expected_rank_intermediate_s8_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_CORR_ACC_HEX \"{generated['generalization_expected_correction_accumulator_s32_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_CORR_ROUNDED_HEX \"{generated['generalization_expected_correction_rounded_s32_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_CORR_S8_HEX \"{generated['generalization_expected_correction_s8_hex']['path']}\"",
        f"`define ACE2_R1_GEN_EXPECTED_CORRECTED_S8_HEX \"{generated['generalization_expected_corrected_v_output_s8_hex']['path']}\"",
        "",
    ]
    OUT_SVH.write_text("\n".join(svh), encoding="ascii")

    input_factor_hex = sv_packed_hex(config.input_to_rank_s8, 8)
    rank_factor_hex = sv_packed_hex(config.rank_to_channel_s8, 8)
    second_multiplier_hex = sv_packed_hex(config.second_stage_multiplier_s32, 32)
    second_shift_hex = sv_packed_hex(config.second_stage_right_shift_u6, 8)

    payload_svh = [
        "// Generated by tools/gen_layer23_v_rank1_integer_correction_vectors.py",
        f"localparam integer ACE2_L23_V_R1_INPUT_WIDTH = {INPUT_WIDTH};",
        f"localparam integer ACE2_L23_V_R1_OUTPUT_WIDTH = {OUTPUT_WIDTH};",
        "localparam logic [9:0] ACE2_L23_V_R1_INPUT_LAST_INDEX = 10'd895;",
        f"localparam logic signed [31:0] ACE2_L23_V_R1_FIRST_MULTIPLIER_S32 = {sv_signed(32, config.first_stage_multiplier_s32)};",
        f"localparam logic [5:0] ACE2_L23_V_R1_FIRST_SHIFT_U6 = 6'd{config.first_stage_right_shift_u6};",
        f"localparam logic [ACE2_L23_V_R1_INPUT_WIDTH*8-1:0] ACE2_L23_V_R1_INPUT_TO_RANK_PACKED = {INPUT_WIDTH*8}'h{input_factor_hex};",
        f"localparam logic [ACE2_L23_V_R1_OUTPUT_WIDTH*8-1:0] ACE2_L23_V_R1_RANK_TO_CHANNEL_PACKED = {OUTPUT_WIDTH*8}'h{rank_factor_hex};",
        f"localparam logic [ACE2_L23_V_R1_OUTPUT_WIDTH*32-1:0] ACE2_L23_V_R1_SECOND_MULTIPLIER_PACKED = {OUTPUT_WIDTH*32}'h{second_multiplier_hex};",
        f"localparam logic [ACE2_L23_V_R1_OUTPUT_WIDTH*8-1:0] ACE2_L23_V_R1_SECOND_SHIFT_PACKED = {OUTPUT_WIDTH*8}'h{second_shift_hex};",
        "",
        "function automatic logic signed [7:0] ace2_l23_v_r1_input_to_rank_s8(input logic [9:0] index);",
        "    begin",
        "        ace2_l23_v_r1_input_to_rank_s8 = (index <= ACE2_L23_V_R1_INPUT_LAST_INDEX) ?",
        "            $signed(ACE2_L23_V_R1_INPUT_TO_RANK_PACKED[index*8 +: 8]) :",
        "            8'sd0;",
        "    end",
        "endfunction",
        "",
        "function automatic logic signed [7:0] ace2_l23_v_r1_rank_to_channel_s8(input logic [6:0] index);",
        "    begin",
        "        ace2_l23_v_r1_rank_to_channel_s8 = $signed(ACE2_L23_V_R1_RANK_TO_CHANNEL_PACKED[index*8 +: 8]);",
        "    end",
        "endfunction",
        "",
        "function automatic logic signed [31:0] ace2_l23_v_r1_second_multiplier_s32(input logic [6:0] index);",
        "    begin",
        "        ace2_l23_v_r1_second_multiplier_s32 = $signed(ACE2_L23_V_R1_SECOND_MULTIPLIER_PACKED[index*32 +: 32]);",
        "    end",
        "endfunction",
        "",
        "function automatic logic [5:0] ace2_l23_v_r1_second_shift_u6(input logic [6:0] index);",
        "    begin",
        "        ace2_l23_v_r1_second_shift_u6 = ACE2_L23_V_R1_SECOND_SHIFT_PACKED[index*8 +: 6];",
        "    end",
        "endfunction",
        "",
    ]
    write_text(OUT_PAYLOAD_SVH, "\n".join(payload_svh))

    case_lines = ["// Generated by tools/gen_layer23_v_rank1_integer_correction_vectors.py"]
    for case in synthetic:
        expected = case["expected"]
        case_lines.append(
            "run_synthetic_case("
            f"\"{case['name']}\", "
            f"{sv_signed(8, case['input0_s8'])}, "
            f"{sv_signed(8, case['input_to_rank0_s8'])}, "
            f"{sv_signed(32, case['first_multiplier_s32'])}, "
            f"6'd{case['first_shift_u6']}, "
            f"{sv_signed(8, case['rank_to_channel0_s8'])}, "
            f"{sv_signed(32, case['second_multiplier0_s32'])}, "
            f"6'd{case['second_shift0_u6']}, "
            f"{sv_signed(8, case['baseline0_s8'])}, "
            f"{sv_signed(32, expected['rank_accumulator_s32'])}, "
            f"{sv_signed(32, expected['rank_rounded_s32'])}, "
            f"{sv_signed(8, expected['rank_intermediate_s8'])}, "
            f"{sv_signed(32, expected['correction_accumulator0_s32'])}, "
            f"{sv_signed(32, expected['correction_rounded0_s32'])}, "
            f"{sv_signed(8, expected['correction0_s8'])}, "
            f"{sv_signed(8, expected['corrected0_s8'])}, "
            f"1'b{int(expected['rank_saturation'])}, "
            f"1'b{int(expected['correction_saturation0'])}, "
            f"1'b{int(expected['add_saturation0'])});"
        )
    case_lines.append("")
    OUT_SYNTHETIC_SVH.write_text("\n".join(case_lines), encoding="ascii")

    print(
        "generated "
        f"{OUT_JSON.relative_to(ROOT)} "
        f"{OUT_SVH.relative_to(ROOT)} "
        f"{OUT_SYNTHETIC_SVH.relative_to(ROOT)} "
        f"{OUT_PAYLOAD_SVH.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
