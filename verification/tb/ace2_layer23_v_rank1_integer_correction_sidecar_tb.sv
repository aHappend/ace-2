`timescale 1ns/1ps
`default_nettype none

module ace2_layer23_v_rank1_integer_correction_sidecar_tb;
    `include "layer23_v_rank1_integer_correction_vectors.svh"

    localparam integer LANES = 16;
    localparam integer ACT_WIDTH = 8;
    localparam logic [7:0] OPCODE_FUSED_QKV = 8'h0b;
    localparam logic [7:0] TARGET_LAYER = 8'd23;
    localparam logic [1:0] QKV_PHASE_V = 2'd2;
    localparam logic [63:0] TARGET_SRC0_ADDR = 64'h0000_0010_0000_0700;
    localparam logic [63:0] TARGET_SRC1_ADDR = 64'h0000_0001_0a38_4000;
    localparam logic [63:0] TARGET_DST_ADDR = 64'h0000_0010_0000_0a80;
    localparam logic [63:0] TARGET_SCALE_ADDR = 64'h0000_0002_0047_2800;
    localparam logic [63:0] TARGET_SCRATCH_ADDR = 64'h0000_0000_0000_0000;

    logic [7:0] descriptor_opcode_i = OPCODE_FUSED_QKV;
    logic [7:0] descriptor_flags_i = 8'd0;
    logic [7:0] descriptor_layer_id_i = TARGET_LAYER;
    logic [15:0] descriptor_m_i = 16'd1;
    logic [15:0] descriptor_n_i = 16'd896;
    logic [15:0] descriptor_k_i = 16'd896;
    logic [63:0] descriptor_src0_addr_i = TARGET_SRC0_ADDR;
    logic [63:0] descriptor_src1_addr_i = TARGET_SRC1_ADDR;
    logic [63:0] descriptor_dst_addr_i = TARGET_DST_ADDR;
    logic [63:0] descriptor_scale_addr_i = TARGET_SCALE_ADDR;
    logic [63:0] descriptor_scratch_addr_i = TARGET_SCRATCH_ADDR;
    logic qkv_fused_i = 1'b1;
    logic [1:0] qkv_phase_i = QKV_PHASE_V;
    logic [2:0] output_beat_i = 3'd0;
    logic [ACE2_R1_INPUT_WIDTH*ACT_WIDTH-1:0] activation_flat_i = '0;
    logic [LANES*ACT_WIDTH-1:0] baseline_beat_i = '0;
    logic [ACE2_R1_INPUT_WIDTH*ACT_WIDTH-1:0] activation_flat_next;
    logic [LANES*ACT_WIDTH-1:0] baseline_beat_next;

    logic enabled_command_match_o;
    logic enabled_descriptor_match_o;
    logic enabled_config_complete_o;
    logic enabled_fail_closed_o;
    logic [LANES*ACT_WIDTH-1:0] enabled_corrected_beat_o;
    logic signed [31:0] enabled_rank_accumulator_s32_o;
    logic signed [31:0] enabled_rank_rounded_s32_o;
    logic signed [7:0] enabled_rank_intermediate_s8_o;
    logic enabled_rank_saturation_o;
    logic enabled_correction_saturation_o;
    logic enabled_add_saturation_o;
    logic enabled_numeric_overflow_o;
    logic disabled_command_match_o;
    logic [LANES*ACT_WIDTH-1:0] disabled_corrected_beat_o;
    logic invalid_command_match_o;
    logic invalid_descriptor_match_o;
    logic invalid_config_complete_o;
    logic invalid_fail_closed_o;
    logic [LANES*ACT_WIDTH-1:0] invalid_corrected_beat_o;

    logic signed [7:0] frozen_input_payload [0:ACE2_R1_POSITION_COUNT*ACE2_R1_INPUT_WIDTH-1];
    logic signed [7:0] frozen_baseline_v [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [31:0] expected_rank_accumulator [0:ACE2_R1_POSITION_COUNT-1];
    logic signed [31:0] expected_rank_rounded [0:ACE2_R1_POSITION_COUNT-1];
    logic signed [7:0] expected_rank_s8 [0:ACE2_R1_POSITION_COUNT-1];
    logic signed [7:0] expected_corrected_s8 [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [7:0] gen_input_payload [0:ACE2_R1_GENERALIZATION_POSITION_COUNT*ACE2_R1_INPUT_WIDTH-1];
    logic signed [7:0] gen_baseline_v [0:ACE2_R1_GENERALIZATION_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [31:0] gen_expected_rank_accumulator [0:ACE2_R1_GENERALIZATION_POSITION_COUNT-1];
    logic signed [31:0] gen_expected_rank_rounded [0:ACE2_R1_GENERALIZATION_POSITION_COUNT-1];
    logic signed [7:0] gen_expected_rank_s8 [0:ACE2_R1_GENERALIZATION_POSITION_COUNT-1];
    logic signed [7:0] gen_expected_corrected_s8 [0:ACE2_R1_GENERALIZATION_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];

    logic [LANES*ACT_WIDTH-1:0] held_corrected_beat;
    integer pos_i;
    integer beat_i;
    integer lane_i;
    integer input_i;
    integer frozen_checked;
    integer generalization_checked;

    ace2_layer23_v_rank1_integer_correction_sidecar #(
        .ENABLE_SIDECAR(1),
        .CONFIG_VALID(1),
        .HIDDEN_SIZE(ACE2_R1_INPUT_WIDTH),
        .OUTPUT_WIDTH(ACE2_R1_OUTPUT_WIDTH),
        .LANES(LANES),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut_enabled (
        .descriptor_opcode_i,
        .descriptor_flags_i,
        .descriptor_layer_id_i,
        .descriptor_m_i,
        .descriptor_n_i,
        .descriptor_k_i,
        .descriptor_src0_addr_i,
        .descriptor_src1_addr_i,
        .descriptor_dst_addr_i,
        .descriptor_scale_addr_i,
        .descriptor_scratch_addr_i,
        .qkv_fused_i,
        .qkv_phase_i,
        .output_beat_i,
        .activation_flat_i,
        .baseline_beat_i,
        .command_match_o(enabled_command_match_o),
        .descriptor_match_o(enabled_descriptor_match_o),
        .config_complete_o(enabled_config_complete_o),
        .fail_closed_o(enabled_fail_closed_o),
        .corrected_beat_o(enabled_corrected_beat_o),
        .rank_accumulator_s32_o(enabled_rank_accumulator_s32_o),
        .rank_rounded_s32_o(enabled_rank_rounded_s32_o),
        .rank_intermediate_s8_o(enabled_rank_intermediate_s8_o),
        .rank_saturation_o(enabled_rank_saturation_o),
        .correction_saturation_o(enabled_correction_saturation_o),
        .add_saturation_o(enabled_add_saturation_o),
        .numeric_overflow_o(enabled_numeric_overflow_o)
    );

    ace2_layer23_v_rank1_integer_correction_sidecar #(
        .ENABLE_SIDECAR(0),
        .CONFIG_VALID(1),
        .HIDDEN_SIZE(ACE2_R1_INPUT_WIDTH),
        .OUTPUT_WIDTH(ACE2_R1_OUTPUT_WIDTH),
        .LANES(LANES),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut_disabled (
        .descriptor_opcode_i,
        .descriptor_flags_i,
        .descriptor_layer_id_i,
        .descriptor_m_i,
        .descriptor_n_i,
        .descriptor_k_i,
        .descriptor_src0_addr_i,
        .descriptor_src1_addr_i,
        .descriptor_dst_addr_i,
        .descriptor_scale_addr_i,
        .descriptor_scratch_addr_i,
        .qkv_fused_i,
        .qkv_phase_i,
        .output_beat_i,
        .activation_flat_i,
        .baseline_beat_i,
        .command_match_o(disabled_command_match_o),
        .descriptor_match_o(),
        .config_complete_o(),
        .fail_closed_o(),
        .corrected_beat_o(disabled_corrected_beat_o),
        .rank_accumulator_s32_o(),
        .rank_rounded_s32_o(),
        .rank_intermediate_s8_o(),
        .rank_saturation_o(),
        .correction_saturation_o(),
        .add_saturation_o(),
        .numeric_overflow_o()
    );

    ace2_layer23_v_rank1_integer_correction_sidecar #(
        .ENABLE_SIDECAR(1),
        .CONFIG_VALID(0),
        .HIDDEN_SIZE(ACE2_R1_INPUT_WIDTH),
        .OUTPUT_WIDTH(ACE2_R1_OUTPUT_WIDTH),
        .LANES(LANES),
        .ACT_WIDTH(ACT_WIDTH)
    ) dut_invalid_config (
        .descriptor_opcode_i,
        .descriptor_flags_i,
        .descriptor_layer_id_i,
        .descriptor_m_i,
        .descriptor_n_i,
        .descriptor_k_i,
        .descriptor_src0_addr_i,
        .descriptor_src1_addr_i,
        .descriptor_dst_addr_i,
        .descriptor_scale_addr_i,
        .descriptor_scratch_addr_i,
        .qkv_fused_i,
        .qkv_phase_i,
        .output_beat_i,
        .activation_flat_i,
        .baseline_beat_i,
        .command_match_o(invalid_command_match_o),
        .descriptor_match_o(invalid_descriptor_match_o),
        .config_complete_o(invalid_config_complete_o),
        .fail_closed_o(invalid_fail_closed_o),
        .corrected_beat_o(invalid_corrected_beat_o),
        .rank_accumulator_s32_o(),
        .rank_rounded_s32_o(),
        .rank_intermediate_s8_o(),
        .rank_saturation_o(),
        .correction_saturation_o(),
        .add_saturation_o(),
        .numeric_overflow_o()
    );

    initial begin
        $readmemh(`ACE2_R1_INPUT_PAYLOAD_HEX, frozen_input_payload);
        $readmemh(`ACE2_R1_BASELINE_V_HEX, frozen_baseline_v);
        $readmemh(`ACE2_R1_EXPECTED_RANK_ACC_HEX, expected_rank_accumulator);
        $readmemh(`ACE2_R1_EXPECTED_RANK_ROUNDED_HEX, expected_rank_rounded);
        $readmemh(`ACE2_R1_EXPECTED_RANK_S8_HEX, expected_rank_s8);
        $readmemh(`ACE2_R1_EXPECTED_CORRECTED_S8_HEX, expected_corrected_s8);
        $readmemh(`ACE2_R1_GEN_INPUT_PAYLOAD_HEX, gen_input_payload);
        $readmemh(`ACE2_R1_GEN_BASELINE_V_HEX, gen_baseline_v);
        $readmemh(`ACE2_R1_GEN_EXPECTED_RANK_ACC_HEX, gen_expected_rank_accumulator);
        $readmemh(`ACE2_R1_GEN_EXPECTED_RANK_ROUNDED_HEX, gen_expected_rank_rounded);
        $readmemh(`ACE2_R1_GEN_EXPECTED_RANK_S8_HEX, gen_expected_rank_s8);
        $readmemh(`ACE2_R1_GEN_EXPECTED_CORRECTED_S8_HEX, gen_expected_corrected_s8);
    end

    function automatic logic signed [7:0] beat_byte(
        input logic [LANES*ACT_WIDTH-1:0] beat,
        input integer lane
    );
        begin
            beat_byte = $signed(beat[lane*ACT_WIDTH +: ACT_WIDTH]);
        end
    endfunction

    task automatic set_matching_descriptor;
        begin
            descriptor_opcode_i = OPCODE_FUSED_QKV;
            descriptor_flags_i = 8'd0;
            descriptor_layer_id_i = TARGET_LAYER;
            descriptor_m_i = 16'd1;
            descriptor_n_i = 16'd896;
            descriptor_k_i = 16'd896;
            descriptor_src0_addr_i = TARGET_SRC0_ADDR;
            descriptor_src1_addr_i = TARGET_SRC1_ADDR;
            descriptor_dst_addr_i = TARGET_DST_ADDR;
            descriptor_scale_addr_i = TARGET_SCALE_ADDR;
            descriptor_scratch_addr_i = TARGET_SCRATCH_ADDR;
            qkv_fused_i = 1'b1;
            qkv_phase_i = QKV_PHASE_V;
        end
    endtask

    task automatic load_frozen_position(input integer position);
        begin
            activation_flat_next = '0;
            for (input_i = 0; input_i < ACE2_R1_INPUT_WIDTH; input_i = input_i + 1) begin
                activation_flat_next[input_i*ACT_WIDTH +: ACT_WIDTH] =
                    frozen_input_payload[position*ACE2_R1_INPUT_WIDTH + input_i];
            end
            activation_flat_i = activation_flat_next;
        end
    endtask

    task automatic load_generalization_position(input integer position);
        begin
            activation_flat_next = '0;
            for (input_i = 0; input_i < ACE2_R1_INPUT_WIDTH; input_i = input_i + 1) begin
                activation_flat_next[input_i*ACT_WIDTH +: ACT_WIDTH] =
                    gen_input_payload[position*ACE2_R1_INPUT_WIDTH + input_i];
            end
            activation_flat_i = activation_flat_next;
        end
    endtask

    task automatic load_frozen_beat(input integer position, input integer beat);
        begin
            output_beat_i = beat[2:0];
            baseline_beat_next = '0;
            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                baseline_beat_next[lane_i*ACT_WIDTH +: ACT_WIDTH] =
                    frozen_baseline_v[position*ACE2_R1_OUTPUT_WIDTH + beat*LANES + lane_i];
            end
            baseline_beat_i = baseline_beat_next;
        end
    endtask

    task automatic load_generalization_beat(input integer position, input integer beat);
        begin
            output_beat_i = beat[2:0];
            baseline_beat_next = '0;
            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                baseline_beat_next[lane_i*ACT_WIDTH +: ACT_WIDTH] =
                    gen_baseline_v[position*ACE2_R1_OUTPUT_WIDTH + beat*LANES + lane_i];
            end
            baseline_beat_i = baseline_beat_next;
        end
    endtask

    task automatic check_common_bypass(input [255:0] label);
        begin
            if (disabled_command_match_o)
                $fatal(1, "%0s: disabled sidecar matched descriptor", label);
            if (disabled_corrected_beat_o !== baseline_beat_i)
                $fatal(1, "%0s: disabled sidecar changed baseline bytes", label);
            if (!invalid_command_match_o || !invalid_descriptor_match_o ||
                invalid_config_complete_o || !invalid_fail_closed_o)
                $fatal(1, "%0s: invalid config did not fail closed", label);
            if (invalid_corrected_beat_o !== baseline_beat_i)
                $fatal(1, "%0s: invalid config changed baseline bytes", label);
        end
    endtask

    task automatic check_frozen_position(input integer position);
        begin
            load_frozen_position(position);
            #1;
            if (enabled_rank_accumulator_s32_o !== expected_rank_accumulator[position])
                $fatal(1, "frozen position %0d rank accumulator mismatch", position);
            if (enabled_rank_rounded_s32_o !== expected_rank_rounded[position])
                $fatal(1, "frozen position %0d rank rounded mismatch", position);
            if (enabled_rank_intermediate_s8_o !== expected_rank_s8[position])
                $fatal(1, "frozen position %0d rank s8 mismatch", position);
            for (beat_i = 0; beat_i < ACE2_R1_OUTPUT_WIDTH/LANES; beat_i = beat_i + 1) begin
                load_frozen_beat(position, beat_i);
                #1;
                if (!enabled_command_match_o || !enabled_descriptor_match_o ||
                    !enabled_config_complete_o || enabled_fail_closed_o)
                    $fatal(1, "frozen position %0d beat %0d descriptor guard failed", position, beat_i);
                if (enabled_numeric_overflow_o)
                    $fatal(1, "frozen position %0d beat %0d numeric overflow", position, beat_i);
                check_common_bypass("frozen");
                for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                    if (beat_byte(enabled_corrected_beat_o, lane_i) !==
                        expected_corrected_s8[position*ACE2_R1_OUTPUT_WIDTH + beat_i*LANES + lane_i])
                        $fatal(1, "frozen position %0d beat %0d lane %0d corrected mismatch",
                               position, beat_i, lane_i);
                end
            end
        end
    endtask

    task automatic check_generalization_position(input integer position);
        begin
            load_generalization_position(position);
            #1;
            if (enabled_rank_accumulator_s32_o !== gen_expected_rank_accumulator[position])
                $fatal(1, "generalization position %0d rank accumulator mismatch", position);
            if (enabled_rank_rounded_s32_o !== gen_expected_rank_rounded[position])
                $fatal(1, "generalization position %0d rank rounded mismatch", position);
            if (enabled_rank_intermediate_s8_o !== gen_expected_rank_s8[position])
                $fatal(1, "generalization position %0d rank s8 mismatch", position);
            for (beat_i = 0; beat_i < ACE2_R1_OUTPUT_WIDTH/LANES; beat_i = beat_i + 1) begin
                load_generalization_beat(position, beat_i);
                #1;
                check_common_bypass("generalization");
                for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
                    if (beat_byte(enabled_corrected_beat_o, lane_i) !==
                        gen_expected_corrected_s8[position*ACE2_R1_OUTPUT_WIDTH + beat_i*LANES + lane_i])
                        $fatal(1, "generalization position %0d beat %0d lane %0d corrected mismatch",
                               position, beat_i, lane_i);
                end
            end
        end
    endtask

    initial begin
        frozen_checked = 0;
        generalization_checked = 0;
        set_matching_descriptor();
        #1;

        load_frozen_position(0);
        load_frozen_beat(0, 0);
        #1;
        held_corrected_beat = enabled_corrected_beat_o;
        #5;
        if (enabled_corrected_beat_o !== held_corrected_beat)
            $fatal(1, "sidecar output was not stable while inputs were held");

        descriptor_layer_id_i = 8'd22;
        #1;
        if (enabled_descriptor_match_o)
            $fatal(1, "non-layer23 descriptor matched sidecar guard");
        if (enabled_corrected_beat_o !== baseline_beat_i)
            $fatal(1, "non-layer23 descriptor changed baseline bytes");
        set_matching_descriptor();

        for (pos_i = 0; pos_i < ACE2_R1_POSITION_COUNT; pos_i = pos_i + 1) begin
            check_frozen_position(pos_i);
            frozen_checked = frozen_checked + 1;
        end
        for (pos_i = 0; pos_i < ACE2_R1_GENERALIZATION_POSITION_COUNT; pos_i = pos_i + 1) begin
            check_generalization_position(pos_i);
            generalization_checked = generalization_checked + 1;
        end

        $display("ACE2_LAYER23_V_RANK1_SIDECAR_RTL_PASS frozen_positions=%0d generalization_positions=%0d beats_per_position=%0d",
                 frozen_checked, generalization_checked, ACE2_R1_OUTPUT_WIDTH/LANES);
        $finish;
    end
endmodule

`default_nettype wire
