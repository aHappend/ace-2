`default_nettype none
/* verilator lint_off DECLFILENAME */

// Exact four-lane Scale32 conversion and aligned sum for the selected
// checkpoint-176 layer-16 source repair. Both carried source payloads remain
// signed A8. Each source is converted to its model-derived group Scale32,
// then both group values are aligned and rounded once into the common sum
// Scale32 consumed by layer-17 RMSNorm.
module ace2_group4_scale32_quant_sum_core (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       clear_i,

    input  logic                       in_valid_i,
    output logic                       in_ready_o,
    input  logic signed [4*8-1:0]      attention_source_s8_i,
    input  logic signed [4*8-1:0]      down_source_s8_i,
    input  logic        [31:0]         attention_base_scale32_i,
    input  logic        [31:0]         down_base_scale32_i,
    input  logic        [31:0]         attention_group_scale32_i,
    input  logic        [31:0]         down_group_scale32_i,
    input  logic        [31:0]         sum_scale32_i,

    output logic                       out_valid_o,
    input  logic                       out_ready_i,
    output logic signed [4*8-1:0]      attention_group_s8_o,
    output logic signed [4*8-1:0]      down_group_s8_o,
    output logic signed [4*8-1:0]      sum_s8_o,
    output logic        [3:0]          attention_saturation_o,
    output logic        [3:0]          down_saturation_o,
    output logic        [3:0]          sum_saturation_o,
    output logic                       descriptor_error_o,
    output logic                       numeric_overflow_o
);
    logic out_valid_q;
    logic signed [31:0] attention_group_q;
    logic signed [31:0] down_group_q;
    logic signed [31:0] sum_q;
    logic [3:0] attention_saturation_q;
    logic [3:0] down_saturation_q;
    logic [3:0] sum_saturation_q;
    logic descriptor_error_q;
    logic numeric_overflow_q;

    logic signed [31:0] attention_group_w;
    logic signed [31:0] down_group_w;
    logic signed [31:0] sum_w;
    logic [3:0] attention_saturation_w;
    logic [3:0] down_saturation_w;
    logic [3:0] sum_saturation_w;
    logic descriptor_error_w;
    logic numeric_overflow_w;

    integer lane;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'h00) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    function automatic logic signed [63:0] round_divide_even_signed(
        input logic signed [63:0] numerator,
        input logic        [63:0] denominator
    );
        logic [63:0] magnitude;
        logic [63:0] quotient;
        logic [63:0] remainder;
        logic [63:0] doubled_remainder;
        logic [63:0] rounded;
        begin
            magnitude = numerator[63] ? $unsigned(-numerator) : $unsigned(numerator);
            quotient = magnitude / denominator;
            remainder = magnitude % denominator;
            doubled_remainder = remainder << 1;
            rounded = quotient;
            if ((doubled_remainder > denominator) ||
                ((doubled_remainder == denominator) && quotient[0]))
                rounded = quotient + 64'd1;
            round_divide_even_signed = numerator[63] ? -$signed(rounded) : $signed(rounded);
        end
    endfunction

    function automatic logic signed [63:0] scale32_term(
        input logic signed [7:0] sample,
        input logic        [15:0] significand,
        input logic signed [7:0] exponent,
        input logic signed [7:0] common_exponent
    );
        logic signed [24:0] product;
        logic [5:0] shift;
        begin
            product = sample * $signed({1'b0, significand});
            shift = 6'($unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            ));
            scale32_term = $signed({{39{product[24]}}, product}) <<< shift;
        end
    endfunction

    function automatic logic [63:0] scale32_denominator(
        input logic        [15:0] significand,
        input logic signed [7:0] exponent,
        input logic signed [7:0] common_exponent
    );
        logic [5:0] shift;
        begin
            shift = 6'($unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            ));
            scale32_denominator = {{48{1'b0}}, significand} << shift;
        end
    endfunction

    function automatic logic signed [7:0] saturate_s8(
        input logic signed [63:0] value
    );
        begin
            if (value > 64'sd127)
                saturate_s8 = 8'sd127;
            else if (value < -64'sd128)
                saturate_s8 = -8'sd128;
            else
                saturate_s8 = value[7:0];
        end
    endfunction

    assign in_ready_o = rst_ni && !clear_i && (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign attention_group_s8_o = attention_group_q;
    assign down_group_s8_o = down_group_q;
    assign sum_s8_o = sum_q;
    assign attention_saturation_o = attention_saturation_q;
    assign down_saturation_o = down_saturation_q;
    assign sum_saturation_o = sum_saturation_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always @* begin : quantize_and_sum_comb
        logic signed [7:0] attention_base_exp;
        logic signed [7:0] down_base_exp;
        logic signed [7:0] attention_group_exp;
        logic signed [7:0] down_group_exp;
        logic signed [7:0] sum_exp;
        logic signed [7:0] attention_common_exp;
        logic signed [7:0] down_common_exp;
        logic signed [7:0] sum_common_exp;
        logic [63:0] attention_denominator;
        logic [63:0] down_denominator;
        logic [63:0] sum_denominator;
        logic signed [63:0] attention_numerator;
        logic signed [63:0] down_numerator;
        logic signed [63:0] sum_numerator;
        logic signed [7:0] attention_source;
        logic signed [7:0] down_source;
        logic signed [7:0] attention_group;
        logic signed [7:0] down_group;
        logic signed [63:0] attention_rounded;
        logic signed [63:0] down_rounded;
        logic signed [63:0] sum_rounded;

        attention_group_w = '0;
        down_group_w = '0;
        sum_w = '0;
        attention_saturation_w = '0;
        down_saturation_w = '0;
        sum_saturation_w = '0;
        descriptor_error_w =
            !scale32_valid(attention_base_scale32_i) ||
            !scale32_valid(down_base_scale32_i) ||
            !scale32_valid(attention_group_scale32_i) ||
            !scale32_valid(down_group_scale32_i) ||
            !scale32_valid(sum_scale32_i);
        numeric_overflow_w = 1'b0;

        attention_base_exp = $signed(attention_base_scale32_i[23:16]);
        down_base_exp = $signed(down_base_scale32_i[23:16]);
        attention_group_exp = $signed(attention_group_scale32_i[23:16]);
        down_group_exp = $signed(down_group_scale32_i[23:16]);
        sum_exp = $signed(sum_scale32_i[23:16]);
        attention_common_exp = (attention_base_exp < attention_group_exp) ?
                               attention_base_exp : attention_group_exp;
        down_common_exp = (down_base_exp < down_group_exp) ?
                          down_base_exp : down_group_exp;
        sum_common_exp = attention_group_exp;
        if (down_group_exp < sum_common_exp)
            sum_common_exp = down_group_exp;
        if (sum_exp < sum_common_exp)
            sum_common_exp = sum_exp;

        attention_denominator = scale32_denominator(
            attention_group_scale32_i[15:0],
            attention_group_exp,
            attention_common_exp
        );
        down_denominator = scale32_denominator(
            down_group_scale32_i[15:0],
            down_group_exp,
            down_common_exp
        );
        sum_denominator = scale32_denominator(
            sum_scale32_i[15:0],
            sum_exp,
            sum_common_exp
        );

        for (lane = 0; lane < 4; lane = lane + 1) begin
            attention_source = attention_source_s8_i[lane*8 +: 8];
            down_source = down_source_s8_i[lane*8 +: 8];
            attention_numerator = scale32_term(
                attention_source,
                attention_base_scale32_i[15:0],
                attention_base_exp,
                attention_common_exp
            );
            down_numerator = scale32_term(
                down_source,
                down_base_scale32_i[15:0],
                down_base_exp,
                down_common_exp
            );
            attention_rounded = descriptor_error_w ? 64'sd0 :
                round_divide_even_signed(attention_numerator, attention_denominator);
            down_rounded = descriptor_error_w ? 64'sd0 :
                round_divide_even_signed(down_numerator, down_denominator);
            attention_group = saturate_s8(attention_rounded);
            down_group = saturate_s8(down_rounded);
            attention_group_w[lane*8 +: 8] = attention_group;
            down_group_w[lane*8 +: 8] = down_group;
            attention_saturation_w[lane] =
                (attention_rounded > 64'sd127) ||
                (attention_rounded < -64'sd128);
            down_saturation_w[lane] =
                (down_rounded > 64'sd127) ||
                (down_rounded < -64'sd128);

            sum_numerator = scale32_term(
                attention_group,
                attention_group_scale32_i[15:0],
                attention_group_exp,
                sum_common_exp
            ) + scale32_term(
                down_group,
                down_group_scale32_i[15:0],
                down_group_exp,
                sum_common_exp
            );
            sum_rounded = descriptor_error_w ? 64'sd0 :
                round_divide_even_signed(sum_numerator, sum_denominator);
            sum_w[lane*8 +: 8] = saturate_s8(sum_rounded);
            sum_saturation_w[lane] =
                (sum_rounded > 64'sd127) ||
                (sum_rounded < -64'sd128);
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_valid_q <= 1'b0;
            attention_group_q <= '0;
            down_group_q <= '0;
            sum_q <= '0;
            attention_saturation_q <= '0;
            down_saturation_q <= '0;
            sum_saturation_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else if (clear_i) begin
            out_valid_q <= 1'b0;
            attention_group_q <= '0;
            down_group_q <= '0;
            sum_q <= '0;
            attention_saturation_q <= '0;
            down_saturation_q <= '0;
            sum_saturation_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;
            if (in_valid_i && in_ready_o) begin
                out_valid_q <= 1'b1;
                attention_group_q <= attention_group_w;
                down_group_q <= down_group_w;
                sum_q <= sum_w;
                attention_saturation_q <= attention_saturation_w;
                down_saturation_q <= down_saturation_w;
                sum_saturation_q <= sum_saturation_w;
                descriptor_error_q <= descriptor_error_w;
                numeric_overflow_q <= numeric_overflow_w;
            end
        end
    end
endmodule


// Focused transaction wrapper. It captures all 224 group-4 sums, executes the
// accepted integer RMSNorm core, stores its signed-A8 output, and serves one
// W4 Q-projection row at a time through the accepted projection core.
module ace2_layer16_group4_rmsnorm_q_core #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer RMS_LANES = 16,
    parameter integer MAC_LANES = 4,
    parameter integer TAG_WIDTH = 16,
    parameter integer ROW_TAG_WIDTH = 10,
    parameter integer GROUP_COUNT = HIDDEN_SIZE / MAC_LANES,
    parameter integer RMS_BEATS = HIDDEN_SIZE / RMS_LANES,
    parameter integer GROUP_INDEX_WIDTH = (GROUP_COUNT <= 1) ? 1 : $clog2(GROUP_COUNT),
    parameter integer RMS_INDEX_WIDTH = (RMS_BEATS <= 1) ? 1 : $clog2(RMS_BEATS + 1)
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic                              clear_i,

    input  logic                              start_valid_i,
    output logic                              start_ready_o,
    input  logic [TAG_WIDTH-1:0]              transaction_tag_i,
    input  logic [31:0]                       attention_base_scale32_i,
    input  logic [31:0]                       down_base_scale32_i,
    input  logic [31:0]                       sum_scale32_i,

    input  logic                              group_valid_i,
    output logic                              group_ready_o,
    input  logic signed [MAC_LANES*8-1:0]     attention_source_s8_i,
    input  logic signed [MAC_LANES*8-1:0]     down_source_s8_i,
    input  logic [31:0]                       attention_group_scale32_i,
    input  logic [31:0]                       down_group_scale32_i,

    input  logic                              gain_valid_i,
    output logic                              gain_ready_o,
    input  logic signed [RMS_LANES*16-1:0]    gain_s16_q8_i,

    input  logic                              q_row_start_valid_i,
    output logic                              q_row_start_ready_o,
    input  logic [ROW_TAG_WIDTH-1:0]          q_row_tag_i,
    input  logic                              q_last_row_i,
    input  logic                              q_weight_valid_i,
    output logic                              q_weight_ready_o,
    input  logic signed [MAC_LANES*4-1:0]     q_weight_s4_i,
    input  logic                              q_meta_valid_i,
    output logic                              q_meta_ready_o,
    input  logic signed [31:0]                q_multiplier_s32_i,
    input  logic [5:0]                        q_right_shift_u6_i,

    output logic                              q_out_valid_o,
    input  logic                              q_out_ready_i,
    output logic signed [7:0]                 q_out_s8_o,
    output logic signed [31:0]                q_accumulator_s32_o,
    output logic [ROW_TAG_WIDTH-1:0]          q_row_tag_o,
    output logic                              q_saturation_o,
    output logic                              q_accumulator_overflow_o,

    output logic                              group_debug_valid_o,
    output logic [GROUP_INDEX_WIDTH-1:0]      group_debug_index_o,
    output logic signed [MAC_LANES*8-1:0]     group_debug_attention_s8_o,
    output logic signed [MAC_LANES*8-1:0]     group_debug_down_s8_o,
    output logic signed [MAC_LANES*8-1:0]     group_debug_sum_s8_o,
    output logic [MAC_LANES-1:0]              group_debug_attention_saturation_o,
    output logic [MAC_LANES-1:0]              group_debug_down_saturation_o,
    output logic [MAC_LANES-1:0]              group_debug_sum_saturation_o,
    output logic                              norm_debug_valid_o,
    output logic [RMS_INDEX_WIDTH-1:0]        norm_debug_index_o,
    output logic signed [RMS_LANES*8-1:0]     norm_debug_s8_o,

    output logic                              done_valid_o,
    input  logic                              done_ready_i,
    output logic [TAG_WIDTH-1:0]              done_tag_o,
    output logic [11:0]                       attention_saturation_count_o,
    output logic [11:0]                       down_saturation_count_o,
    output logic [11:0]                       sum_saturation_count_o,
    output logic                              rms_saturation_o,
    output logic [10:0]                       q_saturation_count_o,
    output logic                              descriptor_error_o,
    output logic                              numeric_overflow_o
);
    typedef enum logic [2:0] {
        WRAP_IDLE,
        WRAP_CAPTURE,
        WRAP_RMS_START,
        WRAP_RMS_ACTIVE,
        WRAP_Q_READY,
        WRAP_DONE
    } wrapper_state_t;

    localparam logic [GROUP_INDEX_WIDTH:0] GROUP_COUNT_VALUE =
        (GROUP_INDEX_WIDTH+1)'(GROUP_COUNT);
    localparam logic [GROUP_INDEX_WIDTH-1:0] GROUP_COUNT_INDEX_VALUE =
        GROUP_INDEX_WIDTH'(GROUP_COUNT);
    localparam logic [GROUP_INDEX_WIDTH-1:0] LAST_GROUP_VALUE =
        GROUP_INDEX_WIDTH'(GROUP_COUNT-1);
    localparam logic [RMS_INDEX_WIDTH-1:0] RMS_BEATS_VALUE =
        RMS_INDEX_WIDTH'(RMS_BEATS);

    wrapper_state_t state_q;
    logic [TAG_WIDTH-1:0] transaction_tag_q;
    logic [31:0] attention_base_scale32_q;
    logic [31:0] down_base_scale32_q;
    logic [31:0] sum_scale32_q;
    logic [GROUP_INDEX_WIDTH:0] group_accept_count_q;
    logic [GROUP_INDEX_WIDTH-1:0] group_store_index_q;
    logic [RMS_INDEX_WIDTH-1:0] rms_collect_index_q;
    logic [RMS_INDEX_WIDTH-1:0] rms_scale_index_q;
    logic [RMS_INDEX_WIDTH-1:0] rms_output_index_q;
    logic [GROUP_INDEX_WIDTH-1:0] q_feed_index_q;
    logic [ROW_TAG_WIDTH-1:0] q_row_tag_q;
    logic q_last_row_q;
    logic done_valid_q;
    logic [11:0] attention_saturation_count_q;
    logic [11:0] down_saturation_count_q;
    logic [11:0] sum_saturation_count_q;
    logic rms_saturation_q;
    logic [10:0] q_saturation_count_q;
    logic descriptor_error_q;
    logic numeric_overflow_q;
    logic signed [7:0] sum_mem_q [0:HIDDEN_SIZE-1];
    logic signed [7:0] norm_mem_q [0:HIDDEN_SIZE-1];
    wire signed [HIDDEN_SIZE*8-1:0] sum_mem_flat_w;
    wire signed [HIDDEN_SIZE*8-1:0] norm_mem_flat_w;

    logic quant_in_valid;
    logic quant_in_ready;
    logic quant_out_valid;
    logic quant_out_ready;
    logic signed [MAC_LANES*8-1:0] quant_attention_group;
    logic signed [MAC_LANES*8-1:0] quant_down_group;
    logic signed [MAC_LANES*8-1:0] quant_sum;
    logic [MAC_LANES-1:0] quant_attention_saturation;
    logic [MAC_LANES-1:0] quant_down_saturation;
    logic [MAC_LANES-1:0] quant_sum_saturation;
    logic quant_descriptor_error;
    logic quant_numeric_overflow;

    logic rms_start_valid;
    logic rms_start_ready;
    logic rms_in_valid;
    logic rms_in_ready;
    logic [RMS_LANES*8-1:0] rms_in_data;
    logic rms_gain_ready;
    logic rms_scale_act_valid;
    logic rms_scale_act_ready;
    logic [RMS_LANES*8-1:0] rms_scale_act_data;
    logic [RMS_LANES*8-1:0] rms_scale_act_hold_q;
    logic [RMS_LANES*16-1:0] rms_gain_hold_q;
    logic rms_out_valid;
    logic [RMS_LANES*8-1:0] rms_out_data;
    logic rms_done_valid;
    logic [47:0] rms_sumsq_unused;
    logic [31:0] rms_inv_unused;
    logic rms_saturation_seen;

    logic q_core_start_ready;
    logic q_core_pair_ready;
    logic q_core_meta_ready;
    logic q_core_out_valid;
    logic [7:0] q_core_out_data;
    logic signed [31:0] q_core_accumulator;
    logic q_core_accumulator_overflow;
    logic q_core_saturation;
    logic [MAC_LANES*8-1:0] q_core_act_data;

    integer lane_index;
    genvar memory_lane;

    function automatic logic [2:0] popcount4(input logic [3:0] value);
        begin
            popcount4 = {2'd0, value[0]} + {2'd0, value[1]} +
                        {2'd0, value[2]} + {2'd0, value[3]};
        end
    endfunction

    assign start_ready_o = rst_ni && !clear_i && (state_q == WRAP_IDLE) && !done_valid_q;
    assign quant_in_valid = (state_q == WRAP_CAPTURE) && group_valid_i &&
                            (group_accept_count_q < GROUP_COUNT_VALUE);
    assign group_ready_o = (state_q == WRAP_CAPTURE) &&
                           (group_accept_count_q < GROUP_COUNT_VALUE) && quant_in_ready;
    assign quant_out_ready = (state_q == WRAP_CAPTURE);

    ace2_group4_scale32_quant_sum_core quant_sum_unit (
        .clk_i,
        .rst_ni,
        .clear_i,
        .in_valid_i(quant_in_valid),
        .in_ready_o(quant_in_ready),
        .attention_source_s8_i,
        .down_source_s8_i,
        .attention_base_scale32_i(attention_base_scale32_q),
        .down_base_scale32_i(down_base_scale32_q),
        .attention_group_scale32_i,
        .down_group_scale32_i,
        .sum_scale32_i(sum_scale32_q),
        .out_valid_o(quant_out_valid),
        .out_ready_i(quant_out_ready),
        .attention_group_s8_o(quant_attention_group),
        .down_group_s8_o(quant_down_group),
        .sum_s8_o(quant_sum),
        .attention_saturation_o(quant_attention_saturation),
        .down_saturation_o(quant_down_saturation),
        .sum_saturation_o(quant_sum_saturation),
        .descriptor_error_o(quant_descriptor_error),
        .numeric_overflow_o(quant_numeric_overflow)
    );

    generate
        for (memory_lane = 0; memory_lane < HIDDEN_SIZE; memory_lane = memory_lane + 1) begin : gen_memory_views
            assign sum_mem_flat_w[memory_lane*8 +: 8] = sum_mem_q[memory_lane];
            assign norm_mem_flat_w[memory_lane*8 +: 8] = norm_mem_q[memory_lane];
        end
    endgenerate

    assign rms_in_data = (rms_collect_index_q < RMS_BEATS_VALUE) ?
        sum_mem_flat_w[rms_collect_index_q*RMS_LANES*8 +: RMS_LANES*8] : '0;
    assign rms_scale_act_data = (rms_scale_index_q < RMS_BEATS_VALUE) ?
        sum_mem_flat_w[rms_scale_index_q*RMS_LANES*8 +: RMS_LANES*8] : '0;
    assign q_core_act_data = (q_feed_index_q < GROUP_COUNT_INDEX_VALUE) ?
        norm_mem_flat_w[q_feed_index_q*MAC_LANES*8 +: MAC_LANES*8] : '0;

    assign rms_start_valid = (state_q == WRAP_RMS_START);
    assign rms_in_valid = (state_q == WRAP_RMS_ACTIVE) &&
                          (rms_collect_index_q < RMS_BEATS_VALUE);
    assign rms_scale_act_valid = (state_q == WRAP_RMS_ACTIVE) &&
                                 (rms_scale_index_q < RMS_BEATS_VALUE);
    assign gain_ready_o = (state_q == WRAP_RMS_ACTIVE) && rms_gain_ready;

    ace2_rmsnorm_core #(
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LANES(RMS_LANES)
    ) rmsnorm (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(rms_start_valid),
        .start_ready_o(rms_start_ready),
        .in_valid_i(rms_in_valid),
        .in_ready_o(rms_in_ready),
        .in_data_i(rms_in_data),
        .gain_valid_i(gain_valid_i && (state_q == WRAP_RMS_ACTIVE)),
        .gain_ready_o(rms_gain_ready),
        .gain_data_i(rms_gain_hold_q),
        .scale_act_valid_i(rms_scale_act_valid),
        .scale_act_ready_o(rms_scale_act_ready),
        .scale_act_data_i(rms_scale_act_hold_q),
        .out_valid_o(rms_out_valid),
        .out_ready_i(state_q == WRAP_RMS_ACTIVE),
        .out_data_o(rms_out_data),
        .done_valid_o(rms_done_valid),
        .done_ready_i(state_q == WRAP_RMS_ACTIVE),
        .sumsq_o(rms_sumsq_unused),
        .inv_rms_q30_o(rms_inv_unused),
        .saturation_seen_o(rms_saturation_seen)
    );

    assign q_row_start_ready_o = (state_q == WRAP_Q_READY) && q_core_start_ready;
    assign q_weight_ready_o = (state_q == WRAP_Q_READY) && q_core_pair_ready;
    assign q_meta_ready_o = (state_q == WRAP_Q_READY) && q_core_meta_ready;
    assign q_out_valid_o = (state_q == WRAP_Q_READY) && q_core_out_valid;
    assign q_out_s8_o = q_core_out_data;
    assign q_accumulator_s32_o = q_core_accumulator;
    assign q_row_tag_o = q_row_tag_q;
    assign q_saturation_o = q_core_saturation;
    assign q_accumulator_overflow_o = q_core_accumulator_overflow;

    ace2_w4a8_proj_core #(
        .K_SIZE(HIDDEN_SIZE),
        .MAC_LANES(MAC_LANES)
    ) q_projection (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i(q_row_start_valid_i && (state_q == WRAP_Q_READY)),
        .start_ready_o(q_core_start_ready),
        .last_group_i(LAST_GROUP_VALUE),
        .pair_valid_i(q_weight_valid_i && (state_q == WRAP_Q_READY)),
        .pair_ready_o(q_core_pair_ready),
        .act_data_i(q_core_act_data),
        .weight_data_i(q_weight_s4_i),
        .meta_valid_i(q_meta_valid_i && (state_q == WRAP_Q_READY)),
        .meta_ready_o(q_core_meta_ready),
        .multiplier_i(q_multiplier_s32_i),
        .right_shift_i(q_right_shift_u6_i),
        .output_zero_point_i(8'sd0),
        .bias_accumulator_i(32'sd0),
        .out_valid_o(q_core_out_valid),
        .out_ready_i(q_out_ready_i && (state_q == WRAP_Q_READY)),
        .out_data_o(q_core_out_data),
        .acc_o(q_core_accumulator),
        .accumulator_overflow_o(q_core_accumulator_overflow),
        .saturation_seen_o(q_core_saturation)
    );

    assign done_valid_o = done_valid_q;
    assign done_tag_o = transaction_tag_q;
    assign attention_saturation_count_o = attention_saturation_count_q;
    assign down_saturation_count_o = down_saturation_count_q;
    assign sum_saturation_count_o = sum_saturation_count_q;
    assign rms_saturation_o = rms_saturation_q;
    assign q_saturation_count_o = q_saturation_count_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= WRAP_IDLE;
            transaction_tag_q <= '0;
            attention_base_scale32_q <= '0;
            down_base_scale32_q <= '0;
            sum_scale32_q <= '0;
            group_accept_count_q <= '0;
            group_store_index_q <= '0;
            rms_collect_index_q <= '0;
            rms_scale_index_q <= '0;
            rms_output_index_q <= '0;
            rms_scale_act_hold_q <= '0;
            rms_gain_hold_q <= '0;
            q_feed_index_q <= '0;
            q_row_tag_q <= '0;
            q_last_row_q <= 1'b0;
            done_valid_q <= 1'b0;
            attention_saturation_count_q <= '0;
            down_saturation_count_q <= '0;
            sum_saturation_count_q <= '0;
            rms_saturation_q <= 1'b0;
            q_saturation_count_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            group_debug_valid_o <= 1'b0;
            group_debug_index_o <= '0;
            group_debug_attention_s8_o <= '0;
            group_debug_down_s8_o <= '0;
            group_debug_sum_s8_o <= '0;
            group_debug_attention_saturation_o <= '0;
            group_debug_down_saturation_o <= '0;
            group_debug_sum_saturation_o <= '0;
            norm_debug_valid_o <= 1'b0;
            norm_debug_index_o <= '0;
            norm_debug_s8_o <= '0;
        end else if (clear_i) begin
            state_q <= WRAP_IDLE;
            group_accept_count_q <= '0;
            group_store_index_q <= '0;
            rms_collect_index_q <= '0;
            rms_scale_index_q <= '0;
            rms_output_index_q <= '0;
            rms_scale_act_hold_q <= '0;
            rms_gain_hold_q <= '0;
            q_feed_index_q <= '0;
            q_row_tag_q <= '0;
            q_last_row_q <= 1'b0;
            done_valid_q <= 1'b0;
            attention_saturation_count_q <= '0;
            down_saturation_count_q <= '0;
            sum_saturation_count_q <= '0;
            rms_saturation_q <= 1'b0;
            q_saturation_count_q <= '0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            group_debug_valid_o <= 1'b0;
            norm_debug_valid_o <= 1'b0;
        end else begin
            group_debug_valid_o <= 1'b0;
            norm_debug_valid_o <= 1'b0;

            if (done_valid_q && done_ready_i) begin
                done_valid_q <= 1'b0;
                state_q <= WRAP_IDLE;
            end

            if (group_valid_i && group_ready_o)
                group_accept_count_q <= group_accept_count_q + 1'b1;

            if (quant_out_valid && quant_out_ready) begin
                for (lane_index = 0; lane_index < MAC_LANES; lane_index = lane_index + 1)
                    sum_mem_q[group_store_index_q*MAC_LANES + lane_index] <=
                        quant_sum[lane_index*8 +: 8];
                attention_saturation_count_q <= attention_saturation_count_q +
                    {9'd0, popcount4(quant_attention_saturation)};
                down_saturation_count_q <= down_saturation_count_q +
                    {9'd0, popcount4(quant_down_saturation)};
                sum_saturation_count_q <= sum_saturation_count_q +
                    {9'd0, popcount4(quant_sum_saturation)};
                descriptor_error_q <= descriptor_error_q | quant_descriptor_error;
                numeric_overflow_q <= numeric_overflow_q | quant_numeric_overflow;
                group_debug_valid_o <= 1'b1;
                group_debug_index_o <= group_store_index_q;
                group_debug_attention_s8_o <= quant_attention_group;
                group_debug_down_s8_o <= quant_down_group;
                group_debug_sum_s8_o <= quant_sum;
                group_debug_attention_saturation_o <= quant_attention_saturation;
                group_debug_down_saturation_o <= quant_down_saturation;
                group_debug_sum_saturation_o <= quant_sum_saturation;
                if (group_store_index_q == LAST_GROUP_VALUE) begin
                    group_store_index_q <= '0;
                    state_q <= WRAP_RMS_START;
                end else begin
                    group_store_index_q <= group_store_index_q + 1'b1;
                end
            end

            if (rms_in_valid && rms_in_ready)
                rms_collect_index_q <= rms_collect_index_q + 1'b1;
            if (gain_valid_i && gain_ready_o && rms_scale_act_valid && rms_scale_act_ready) begin
                rms_scale_act_hold_q <= rms_scale_act_data;
                rms_gain_hold_q <= gain_s16_q8_i;
                rms_scale_index_q <= rms_scale_index_q + 1'b1;
            end
            if (rms_out_valid && (state_q == WRAP_RMS_ACTIVE)) begin
                for (lane_index = 0; lane_index < RMS_LANES; lane_index = lane_index + 1)
                    norm_mem_q[rms_output_index_q*RMS_LANES + lane_index] <=
                        rms_out_data[lane_index*8 +: 8];
                norm_debug_valid_o <= 1'b1;
                norm_debug_index_o <= rms_output_index_q;
                norm_debug_s8_o <= rms_out_data;
                rms_output_index_q <= rms_output_index_q + 1'b1;
            end
            if (rms_done_valid && (state_q == WRAP_RMS_ACTIVE)) begin
                rms_saturation_q <= rms_saturation_seen;
                state_q <= WRAP_Q_READY;
            end

            if (q_row_start_valid_i && q_row_start_ready_o) begin
                q_feed_index_q <= '0;
                q_row_tag_q <= q_row_tag_i;
                q_last_row_q <= q_last_row_i;
            end
            if (q_weight_valid_i && q_weight_ready_o)
                q_feed_index_q <= q_feed_index_q + 1'b1;
            if (q_out_valid_o && q_out_ready_i) begin
                if (q_core_saturation)
                    q_saturation_count_q <= q_saturation_count_q + 1'b1;
                numeric_overflow_q <= numeric_overflow_q | q_core_accumulator_overflow;
                if (q_last_row_q) begin
                    done_valid_q <= 1'b1;
                    state_q <= WRAP_DONE;
                end
            end

            case (state_q)
                WRAP_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        transaction_tag_q <= transaction_tag_i;
                        attention_base_scale32_q <= attention_base_scale32_i;
                        down_base_scale32_q <= down_base_scale32_i;
                        sum_scale32_q <= sum_scale32_i;
                        group_accept_count_q <= '0;
                        group_store_index_q <= '0;
                        rms_collect_index_q <= '0;
                        rms_scale_index_q <= '0;
                        rms_output_index_q <= '0;
                        q_feed_index_q <= '0;
                        done_valid_q <= 1'b0;
                        attention_saturation_count_q <= '0;
                        down_saturation_count_q <= '0;
                        sum_saturation_count_q <= '0;
                        rms_saturation_q <= 1'b0;
                        q_saturation_count_q <= '0;
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        state_q <= WRAP_CAPTURE;
                    end
                end

                WRAP_RMS_START: begin
                    if (rms_start_valid && rms_start_ready) begin
                        rms_collect_index_q <= '0;
                        rms_scale_index_q <= '0;
                        rms_output_index_q <= '0;
                        state_q <= WRAP_RMS_ACTIVE;
                    end
                end

                default: begin
                end
            endcase
        end
    end
endmodule

`default_nettype wire
