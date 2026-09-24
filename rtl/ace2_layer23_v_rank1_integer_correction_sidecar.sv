`default_nettype none

module ace2_layer23_v_rank1_integer_correction_sidecar #(
    parameter integer ENABLE_SIDECAR = 0,
    parameter integer CONFIG_VALID = 1,
    parameter integer HIDDEN_SIZE = 896,
    parameter integer OUTPUT_WIDTH = 128,
    parameter integer LANES = 16,
    parameter integer ACT_WIDTH = 8
) (
    input  logic [7:0]                        descriptor_opcode_i,
    input  logic [7:0]                        descriptor_flags_i,
    input  logic [7:0]                        descriptor_layer_id_i,
    input  logic [15:0]                       descriptor_m_i,
    input  logic [15:0]                       descriptor_n_i,
    input  logic [15:0]                       descriptor_k_i,
    input  logic [63:0]                       descriptor_src0_addr_i,
    input  logic [63:0]                       descriptor_src1_addr_i,
    input  logic [63:0]                       descriptor_dst_addr_i,
    input  logic [63:0]                       descriptor_scale_addr_i,
    input  logic [63:0]                       descriptor_scratch_addr_i,
    input  logic                              qkv_fused_i,
    input  logic [1:0]                        qkv_phase_i,
    input  logic [2:0]                        output_beat_i,
    input  logic [HIDDEN_SIZE*ACT_WIDTH-1:0]  activation_flat_i,
    input  logic [LANES*ACT_WIDTH-1:0]        baseline_beat_i,
    output logic                              command_match_o,
    output logic                              descriptor_match_o,
    output logic                              config_complete_o,
    output logic                              fail_closed_o,
    output logic [LANES*ACT_WIDTH-1:0]        corrected_beat_o,
    output logic signed [31:0]                rank_accumulator_s32_o,
    output logic signed [31:0]                rank_rounded_s32_o,
    output logic signed [7:0]                 rank_intermediate_s8_o,
    output logic                              rank_saturation_o,
    output logic                              correction_saturation_o,
    output logic                              add_saturation_o,
    output logic                              numeric_overflow_o
);
    `include "generated/ace2_layer23_v_rank1_integer_correction_payload.svh"

    localparam logic [7:0] OPCODE_FUSED_QKV = 8'h0b;
    localparam logic [7:0] TARGET_LAYER = 8'd23;
    localparam logic [15:0] TARGET_M = 16'd1;
    localparam logic [15:0] TARGET_N = 16'd896;
    localparam logic [15:0] TARGET_K = 16'd896;
    localparam logic [63:0] TARGET_SRC0_ADDR = 64'h0000_0010_0000_0700;
    localparam logic [63:0] TARGET_SRC1_ADDR = 64'h0000_0001_0a38_4000;
    localparam logic [63:0] TARGET_DST_ADDR = 64'h0000_0010_0000_0a80;
    localparam logic [63:0] TARGET_SCALE_ADDR = 64'h0000_0002_0047_2800;
    localparam logic [63:0] TARGET_SCRATCH_ADDR = 64'h0000_0000_0000_0000;
    localparam logic [1:0] QKV_PHASE_V = 2'd2;
    localparam bit PROFILE_SUPPORTED =
        (HIDDEN_SIZE == ACE2_L23_V_R1_INPUT_WIDTH) &&
        (OUTPUT_WIDTH == ACE2_L23_V_R1_OUTPUT_WIDTH) &&
        (LANES == 16) &&
        (ACT_WIDTH == 8);

    logic signed [32:0] rank_accumulator_s33_r;
    logic signed [15:0] input_product_s16_r;
    logic signed [63:0] rank_product_s64_r;
    logic signed [63:0] rank_rounded_s64_r;
    logic signed [7:0] rank_saturated_s8_r;
    logic rank_accumulator_overflow_r;
    logic rank_rounded_overflow_r;
    logic rank_saturation_r;
    logic signed [15:0] correction_accumulator_s16_r;
    logic signed [63:0] correction_product_s64_r;
    logic signed [63:0] correction_rounded_s64_r;
    logic signed [7:0] correction_saturated_s8_r;
    logic signed [8:0] add_sum_s9_r;
    logic signed [7:0] corrected_s8_r;
    logic correction_rounded_overflow_r;
    logic correction_saturation_r;
    logic add_saturation_r;
    logic signed [7:0] activation_s8_r;
    logic signed [7:0] input_factor_s8_r;
    logic signed [7:0] rank_factor_s8_r;
    logic signed [7:0] baseline_s8_r;
    logic [6:0] channel_index_r;
    logic [3:0] lane_index_r;
    wire command_match_w;
    wire descriptor_match_w;
    wire active_w;
    wire rank_compute_enabled_w;

    integer input_index;
    integer lane_index;

    function automatic logic signed [63:0] round_shift_even_s64(
        input logic signed [63:0] value,
        input logic [5:0] shift
    );
        logic sign;
        logic [63:0] magnitude, quotient, remainder, mask, half;
        begin
            if (shift == 6'd0) begin
                round_shift_even_s64 = value;
            end else begin
                sign = value[63];
                magnitude = sign ? $unsigned(-value) : $unsigned(value);
                quotient = magnitude >> shift;
                mask = {64{1'b1}} >> (64 - shift);
                remainder = magnitude & mask;
                half = 64'd1 << (shift - 1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 64'd1;
                round_shift_even_s64 = sign ? -$signed(quotient) : $signed(quotient);
            end
        end
    endfunction

    function automatic logic signed [7:0] saturate_s8_s64(input logic signed [63:0] value);
        begin
            if (value > 64'sd127)
                saturate_s8_s64 = 8'sd127;
            else if (value < -64'sd128)
                saturate_s8_s64 = -8'sd128;
            else
                saturate_s8_s64 = value[7:0];
        end
    endfunction

    function automatic logic signed [7:0] saturate_s8_s9(input logic signed [8:0] value);
        begin
            if (value > 9'sd127)
                saturate_s8_s9 = 8'sd127;
            else if (value < -9'sd128)
                saturate_s8_s9 = -8'sd128;
            else
                saturate_s8_s9 = value[7:0];
        end
    endfunction

    assign command_match_w =
        (ENABLE_SIDECAR != 0) &&
        PROFILE_SUPPORTED &&
        (descriptor_opcode_i == OPCODE_FUSED_QKV) &&
        (descriptor_flags_i == 8'd0) &&
        (descriptor_layer_id_i == TARGET_LAYER) &&
        (descriptor_m_i == TARGET_M) &&
        (descriptor_n_i == TARGET_N) &&
        (descriptor_k_i == TARGET_K) &&
        (descriptor_src0_addr_i == TARGET_SRC0_ADDR) &&
        (descriptor_src1_addr_i == TARGET_SRC1_ADDR) &&
        (descriptor_dst_addr_i == TARGET_DST_ADDR) &&
        (descriptor_scale_addr_i == TARGET_SCALE_ADDR) &&
        (descriptor_scratch_addr_i == TARGET_SCRATCH_ADDR);
    assign descriptor_match_w = command_match_w && qkv_fused_i && (qkv_phase_i == QKV_PHASE_V);
    assign active_w = descriptor_match_w && (CONFIG_VALID != 0);
    assign rank_compute_enabled_w = (ENABLE_SIDECAR != 0) && (CONFIG_VALID != 0) && PROFILE_SUPPORTED;
    assign command_match_o = command_match_w;
    assign descriptor_match_o = descriptor_match_w;
    assign config_complete_o = command_match_w && (CONFIG_VALID != 0);
    assign fail_closed_o = command_match_w && (CONFIG_VALID == 0);
    assign rank_accumulator_s32_o = rank_accumulator_s33_r[31:0];
    assign rank_rounded_s32_o = rank_rounded_s64_r[31:0];
    assign rank_intermediate_s8_o = rank_saturated_s8_r;
    assign rank_saturation_o = active_w && rank_saturation_r;
    assign correction_saturation_o = active_w && correction_saturation_r;
    assign add_saturation_o = active_w && add_saturation_r;
    assign numeric_overflow_o =
        active_w &&
        (rank_accumulator_overflow_r ||
         rank_rounded_overflow_r ||
         correction_rounded_overflow_r);

    always @* begin
        rank_accumulator_s33_r = 33'sd0;
        activation_s8_r = 8'sd0;
        input_factor_s8_r = 8'sd0;
        input_product_s16_r = 16'sd0;
        if (rank_compute_enabled_w) begin
            for (input_index = 0; input_index < ACE2_L23_V_R1_INPUT_WIDTH; input_index = input_index + 1) begin
                activation_s8_r = $signed(activation_flat_i[input_index*ACT_WIDTH +: ACT_WIDTH]);
                input_factor_s8_r = ace2_l23_v_r1_input_to_rank_s8(input_index[9:0]);
                input_product_s16_r = activation_s8_r * input_factor_s8_r;
                rank_accumulator_s33_r =
                    rank_accumulator_s33_r +
                    $signed({{17{input_product_s16_r[15]}}, input_product_s16_r});
            end
        end
        rank_accumulator_overflow_r =
            rank_accumulator_s33_r[32] != rank_accumulator_s33_r[31];
        rank_product_s64_r =
            $signed(rank_accumulator_s33_r[31:0]) * ACE2_L23_V_R1_FIRST_MULTIPLIER_S32;
        rank_rounded_s64_r =
            round_shift_even_s64(rank_product_s64_r, ACE2_L23_V_R1_FIRST_SHIFT_U6);
        rank_rounded_overflow_r =
            (rank_rounded_s64_r > 64'sd2147483647) ||
            (rank_rounded_s64_r < -64'sd2147483648);
        rank_saturated_s8_r = saturate_s8_s64(rank_rounded_s64_r);
        rank_saturation_r =
            (rank_rounded_s64_r > 64'sd127) ||
            (rank_rounded_s64_r < -64'sd128);
    end

    always @* begin
        corrected_beat_o = baseline_beat_i;
        lane_index_r = 4'd0;
        channel_index_r = 7'd0;
        rank_factor_s8_r = 8'sd0;
        correction_accumulator_s16_r = 16'sd0;
        correction_product_s64_r = 64'sd0;
        correction_rounded_s64_r = 64'sd0;
        correction_saturated_s8_r = 8'sd0;
        baseline_s8_r = 8'sd0;
        add_sum_s9_r = 9'sd0;
        corrected_s8_r = 8'sd0;
        correction_saturation_r = 1'b0;
        add_saturation_r = 1'b0;
        correction_rounded_overflow_r = 1'b0;
        if (active_w) begin
            for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
                lane_index_r = lane_index[3:0];
                channel_index_r = {output_beat_i, lane_index_r};
                rank_factor_s8_r = ace2_l23_v_r1_rank_to_channel_s8(channel_index_r);
                correction_accumulator_s16_r = rank_saturated_s8_r * rank_factor_s8_r;
                correction_product_s64_r =
                    $signed({{16{correction_accumulator_s16_r[15]}}, correction_accumulator_s16_r}) *
                    ace2_l23_v_r1_second_multiplier_s32(channel_index_r);
                correction_rounded_s64_r = round_shift_even_s64(
                    correction_product_s64_r,
                    ace2_l23_v_r1_second_shift_u6(channel_index_r)
                );
                correction_saturated_s8_r = saturate_s8_s64(correction_rounded_s64_r);
                baseline_s8_r = $signed(baseline_beat_i[lane_index*ACT_WIDTH +: ACT_WIDTH]);
                add_sum_s9_r =
                    $signed({baseline_s8_r[7], baseline_s8_r}) +
                    $signed({correction_saturated_s8_r[7], correction_saturated_s8_r});
                corrected_s8_r = saturate_s8_s9(add_sum_s9_r);
                corrected_beat_o[lane_index*ACT_WIDTH +: ACT_WIDTH] = corrected_s8_r;
                correction_saturation_r =
                    correction_saturation_r ||
                    (correction_rounded_s64_r > 64'sd127) ||
                    (correction_rounded_s64_r < -64'sd128);
                correction_rounded_overflow_r =
                    correction_rounded_overflow_r ||
                    (correction_rounded_s64_r > 64'sd2147483647) ||
                    (correction_rounded_s64_r < -64'sd2147483648);
                add_saturation_r =
                    add_saturation_r ||
                    (add_sum_s9_r > 9'sd127) ||
                    (add_sum_s9_r < -9'sd128);
            end
        end
    end
endmodule

`default_nettype wire
