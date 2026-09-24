`timescale 1ns/1ps
`default_nettype none

module ace2_layer23_v_rank1_integer_correction_core #(
    parameter integer INPUT_WIDTH = 896,
    parameter integer OUTPUT_WIDTH = 128
) (
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  logic                      clear_i,
    input  logic                      enable_i,
    input  logic                      cfg_valid_i,
    output logic                      cfg_ready_o,
    input  logic        [2:0]         cfg_kind_i,
    input  logic        [9:0]         cfg_index_i,
    input  logic signed [31:0]        cfg_data_i,
    output logic                      cfg_error_o,
    output logic                      config_complete_o,
    input  logic                      start_valid_i,
    output logic                      start_ready_o,
    input  logic                      input_valid_i,
    output logic                      input_ready_o,
    input  logic signed [7:0]         input_s8_i,
    input  logic                      baseline_valid_i,
    output logic                      baseline_ready_o,
    input  logic signed [7:0]         baseline_v_s8_i,
    output logic                      output_valid_o,
    input  logic                      output_ready_i,
    output logic        [6:0]         output_channel_o,
    output logic                      output_last_o,
    output logic signed [31:0]        rank_accumulator_s32_o,
    output logic signed [31:0]        rank_rounded_s32_o,
    output logic signed [7:0]         rank_intermediate_s8_o,
    output logic                      rank_valid_o,
    output logic signed [31:0]        correction_accumulator_s32_o,
    output logic signed [31:0]        correction_rounded_s32_o,
    output logic signed [7:0]         correction_s8_o,
    output logic signed [7:0]         corrected_v_s8_o,
    output logic                      input_saturation_o,
    output logic                      rank_saturation_o,
    output logic                      correction_saturation_o,
    output logic                      add_saturation_o,
    output logic                      descriptor_error_o,
    output logic                      numeric_overflow_o
);
    localparam logic [2:0] CFG_INPUT_TO_RANK = 3'd0;
    localparam logic [2:0] CFG_RANK_TO_CHANNEL = 3'd1;
    localparam logic [2:0] CFG_SECOND_MULTIPLIER = 3'd2;
    localparam logic [2:0] CFG_SECOND_SHIFT = 3'd3;
    localparam logic [2:0] CFG_FIRST_MULTIPLIER = 3'd4;
    localparam logic [2:0] CFG_FIRST_SHIFT = 3'd5;
    localparam logic [9:0] INPUT_LAST_INDEX = 10'(INPUT_WIDTH - 1);
    localparam logic [6:0] OUTPUT_LAST_INDEX = 7'(OUTPUT_WIDTH - 1);

    typedef enum logic [1:0] {
        R1_IDLE,
        R1_INPUT,
        R1_OUTPUT
    } state_t;

    state_t state_q;
    logic signed [7:0] input_to_rank_q [0:INPUT_WIDTH-1];
    logic signed [7:0] rank_to_channel_q [0:OUTPUT_WIDTH-1];
    logic signed [31:0] second_multiplier_q [0:OUTPUT_WIDTH-1];
    logic [5:0] second_shift_q [0:OUTPUT_WIDTH-1];
    logic signed [31:0] first_multiplier_q;
    logic [5:0] first_shift_q;
    logic [INPUT_WIDTH-1:0] input_to_rank_loaded_q;
    logic [OUTPUT_WIDTH-1:0] rank_to_channel_loaded_q;
    logic [OUTPUT_WIDTH-1:0] second_multiplier_loaded_q;
    logic [OUTPUT_WIDTH-1:0] second_shift_loaded_q;
    logic first_multiplier_loaded_q, first_shift_loaded_q;
    logic cfg_error_q;

    logic [9:0] input_index_q;
    logic [6:0] output_index_q;
    logic signed [31:0] rank_accumulator_q;
    logic signed [31:0] rank_rounded_q;
    logic signed [7:0] rank_intermediate_q;
    logic rank_valid_q;
    logic input_saturation_q, rank_saturation_q;
    logic descriptor_error_q, numeric_overflow_q;

    logic output_valid_q, output_last_q;
    logic [6:0] output_channel_q;
    logic signed [31:0] correction_accumulator_q;
    logic signed [31:0] correction_rounded_q;
    logic signed [7:0] correction_s8_q, corrected_v_s8_q;
    logic correction_saturation_q, add_saturation_q;

    logic signed [7:0] selected_input_factor_w;
    logic signed [7:0] selected_rank_factor_w;
    logic signed [31:0] selected_second_multiplier_w;
    logic [5:0] selected_second_shift_w;
    logic signed [15:0] input_product_s16_w;
    logic signed [32:0] rank_accumulator_next_s33_w;
    logic rank_accumulator_overflow_w;
    logic signed [63:0] first_product_s64_w;
    logic signed [63:0] rank_rounded_s64_w;
    logic rank_rounded_overflow_w;
    logic signed [7:0] rank_saturated_s8_w;
    logic rank_saturation_w;
    logic input_last_w;
    logic signed [15:0] correction_accumulator_s16_w;
    logic signed [63:0] second_product_s64_w;
    logic signed [63:0] correction_rounded_s64_w;
    logic correction_rounded_overflow_w;
    logic signed [7:0] correction_saturated_s8_w;
    logic correction_saturation_w;
    logic signed [8:0] add_sum_s9_w;
    logic signed [7:0] corrected_s8_w;
    logic add_saturation_w;
    logic config_complete_w;
    logic cfg_input_index_valid_w;
    logic cfg_output_index_valid_w;
    logic [6:0] cfg_output_index_w;

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

    assign selected_input_factor_w = input_to_rank_q[input_index_q];
    assign selected_rank_factor_w = rank_to_channel_q[output_index_q];
    assign selected_second_multiplier_w = second_multiplier_q[output_index_q];
    assign selected_second_shift_w = second_shift_q[output_index_q];
    assign input_product_s16_w = input_s8_i * selected_input_factor_w;
    assign rank_accumulator_next_s33_w =
        $signed({rank_accumulator_q[31], rank_accumulator_q}) +
        $signed({{17{input_product_s16_w[15]}}, input_product_s16_w});
    assign rank_accumulator_overflow_w =
        rank_accumulator_next_s33_w[32] != rank_accumulator_next_s33_w[31];
    assign first_product_s64_w =
        $signed(rank_accumulator_next_s33_w[31:0]) * first_multiplier_q;
    assign rank_rounded_s64_w = round_shift_even_s64(first_product_s64_w, first_shift_q);
    assign rank_rounded_overflow_w =
        (rank_rounded_s64_w > 64'sd2147483647) ||
        (rank_rounded_s64_w < -64'sd2147483648);
    assign rank_saturated_s8_w = saturate_s8_s64(rank_rounded_s64_w);
    assign rank_saturation_w =
        (rank_rounded_s64_w > 64'sd127) ||
        (rank_rounded_s64_w < -64'sd128);
    assign input_last_w = (input_index_q == INPUT_LAST_INDEX);
    assign cfg_input_index_valid_w = (cfg_index_i <= INPUT_LAST_INDEX);
    assign cfg_output_index_w = cfg_index_i[6:0];
    assign cfg_output_index_valid_w = (cfg_index_i[9:7] == 3'd0);

    assign correction_accumulator_s16_w = rank_intermediate_q * selected_rank_factor_w;
    assign second_product_s64_w =
        $signed({{16{correction_accumulator_s16_w[15]}}, correction_accumulator_s16_w}) *
        selected_second_multiplier_w;
    assign correction_rounded_s64_w =
        round_shift_even_s64(second_product_s64_w, selected_second_shift_w);
    assign correction_rounded_overflow_w =
        (correction_rounded_s64_w > 64'sd2147483647) ||
        (correction_rounded_s64_w < -64'sd2147483648);
    assign correction_saturated_s8_w = saturate_s8_s64(correction_rounded_s64_w);
    assign correction_saturation_w =
        (correction_rounded_s64_w > 64'sd127) ||
        (correction_rounded_s64_w < -64'sd128);
    assign add_sum_s9_w =
        $signed({baseline_v_s8_i[7], baseline_v_s8_i}) +
        $signed({correction_saturated_s8_w[7], correction_saturated_s8_w});
    assign corrected_s8_w = saturate_s8_s9(add_sum_s9_w);
    assign add_saturation_w = (add_sum_s9_w > 9'sd127) || (add_sum_s9_w < -9'sd128);

    assign config_complete_w =
        first_multiplier_loaded_q &&
        first_shift_loaded_q &&
        (&input_to_rank_loaded_q) &&
        (&rank_to_channel_loaded_q) &&
        (&second_multiplier_loaded_q) &&
        (&second_shift_loaded_q) &&
        !cfg_error_q;

    assign cfg_ready_o = rst_ni && !clear_i && !enable_i &&
                         (state_q == R1_IDLE) && !output_valid_q;
    assign cfg_error_o = cfg_error_q;
    assign config_complete_o = config_complete_w;
    assign start_ready_o = rst_ni && !clear_i && enable_i && config_complete_w &&
                           (state_q == R1_IDLE) && !output_valid_q;
    assign input_ready_o = rst_ni && !clear_i && (state_q == R1_INPUT);
    assign baseline_ready_o = rst_ni && !clear_i && (state_q == R1_OUTPUT) &&
                              (!output_valid_q || (output_ready_i && !output_last_q));
    assign output_valid_o = output_valid_q;
    assign output_channel_o = output_channel_q;
    assign output_last_o = output_last_q;
    assign rank_accumulator_s32_o = rank_accumulator_q;
    assign rank_rounded_s32_o = rank_rounded_q;
    assign rank_intermediate_s8_o = rank_intermediate_q;
    assign rank_valid_o = rank_valid_q;
    assign correction_accumulator_s32_o = correction_accumulator_q;
    assign correction_rounded_s32_o = correction_rounded_q;
    assign correction_s8_o = correction_s8_q;
    assign corrected_v_s8_o = corrected_v_s8_q;
    assign input_saturation_o = input_saturation_q;
    assign rank_saturation_o = rank_saturation_q;
    assign correction_saturation_o = correction_saturation_q;
    assign add_saturation_o = add_saturation_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= R1_IDLE;
            first_multiplier_q <= '0;
            first_shift_q <= '0;
            first_multiplier_loaded_q <= 1'b0;
            first_shift_loaded_q <= 1'b0;
            input_to_rank_loaded_q <= '0;
            rank_to_channel_loaded_q <= '0;
            second_multiplier_loaded_q <= '0;
            second_shift_loaded_q <= '0;
            cfg_error_q <= 1'b0;
            input_index_q <= '0;
            output_index_q <= '0;
            rank_accumulator_q <= '0;
            rank_rounded_q <= '0;
            rank_intermediate_q <= '0;
            rank_valid_q <= 1'b0;
            input_saturation_q <= 1'b0;
            rank_saturation_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            output_valid_q <= 1'b0;
            output_last_q <= 1'b0;
            output_channel_q <= '0;
            correction_accumulator_q <= '0;
            correction_rounded_q <= '0;
            correction_s8_q <= '0;
            corrected_v_s8_q <= '0;
            correction_saturation_q <= 1'b0;
            add_saturation_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= R1_IDLE;
            input_index_q <= '0;
            output_index_q <= '0;
            rank_valid_q <= 1'b0;
            output_valid_q <= 1'b0;
            output_last_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            cfg_error_q <= 1'b0;
            correction_saturation_q <= 1'b0;
            add_saturation_q <= 1'b0;
        end else begin
            if (cfg_valid_i && cfg_ready_o) begin
                case (cfg_kind_i)
                    CFG_INPUT_TO_RANK: begin
                        if (cfg_input_index_valid_w) begin
                            input_to_rank_q[cfg_index_i] <= cfg_data_i[7:0];
                            input_to_rank_loaded_q[cfg_index_i] <= 1'b1;
                        end else begin
                            cfg_error_q <= 1'b1;
                        end
                    end
                    CFG_RANK_TO_CHANNEL: begin
                        if (cfg_output_index_valid_w) begin
                            rank_to_channel_q[cfg_output_index_w] <= cfg_data_i[7:0];
                            rank_to_channel_loaded_q[cfg_output_index_w] <= 1'b1;
                        end else begin
                            cfg_error_q <= 1'b1;
                        end
                    end
                    CFG_SECOND_MULTIPLIER: begin
                        if (cfg_output_index_valid_w) begin
                            second_multiplier_q[cfg_output_index_w] <= cfg_data_i;
                            second_multiplier_loaded_q[cfg_output_index_w] <= 1'b1;
                        end else begin
                            cfg_error_q <= 1'b1;
                        end
                    end
                    CFG_SECOND_SHIFT: begin
                        if (cfg_output_index_valid_w && (cfg_data_i[31:6] == 26'd0)) begin
                            second_shift_q[cfg_output_index_w] <= cfg_data_i[5:0];
                            second_shift_loaded_q[cfg_output_index_w] <= 1'b1;
                        end else begin
                            cfg_error_q <= 1'b1;
                        end
                    end
                    CFG_FIRST_MULTIPLIER: begin
                        if ((cfg_index_i == 10'd0) && (cfg_data_i > 32'sd0)) begin
                            first_multiplier_q <= cfg_data_i;
                            first_multiplier_loaded_q <= 1'b1;
                        end else begin
                            cfg_error_q <= 1'b1;
                        end
                    end
                    CFG_FIRST_SHIFT: begin
                        if ((cfg_index_i == 10'd0) && (cfg_data_i[31:6] == 26'd0)) begin
                            first_shift_q <= cfg_data_i[5:0];
                            first_shift_loaded_q <= 1'b1;
                        end else begin
                            cfg_error_q <= 1'b1;
                        end
                    end
                    default: cfg_error_q <= 1'b1;
                endcase
            end

            if (output_valid_q && output_ready_i) begin
                output_valid_q <= 1'b0;
                if (output_last_q) begin
                    state_q <= R1_IDLE;
                    rank_valid_q <= 1'b0;
                    output_last_q <= 1'b0;
                end
            end

            case (state_q)
                R1_IDLE: begin
                    if (start_valid_i && start_ready_o) begin
                        state_q <= R1_INPUT;
                        input_index_q <= '0;
                        output_index_q <= '0;
                        rank_accumulator_q <= '0;
                        rank_rounded_q <= '0;
                        rank_intermediate_q <= '0;
                        rank_valid_q <= 1'b0;
                        input_saturation_q <= 1'b0;
                        rank_saturation_q <= 1'b0;
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        output_valid_q <= 1'b0;
                        output_last_q <= 1'b0;
                        output_channel_q <= '0;
                        correction_accumulator_q <= '0;
                        correction_rounded_q <= '0;
                        correction_s8_q <= '0;
                        corrected_v_s8_q <= '0;
                        correction_saturation_q <= 1'b0;
                        add_saturation_q <= 1'b0;
                    end
                end
                R1_INPUT: begin
                    if (input_valid_i && input_ready_o) begin
                        rank_accumulator_q <= rank_accumulator_next_s33_w[31:0];
                        if (rank_accumulator_overflow_w || rank_rounded_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            descriptor_error_q <= 1'b1;
                            state_q <= R1_IDLE;
                        end else if (input_last_w) begin
                            rank_rounded_q <= rank_rounded_s64_w[31:0];
                            rank_intermediate_q <= rank_saturated_s8_w;
                            rank_saturation_q <= rank_saturation_w;
                            rank_valid_q <= 1'b1;
                            output_index_q <= '0;
                            state_q <= R1_OUTPUT;
                        end else begin
                            input_index_q <= input_index_q + 10'd1;
                        end
                    end
                end
                R1_OUTPUT: begin
                    if (baseline_valid_i && baseline_ready_o) begin
                        correction_accumulator_q <=
                            {{16{correction_accumulator_s16_w[15]}}, correction_accumulator_s16_w};
                        correction_rounded_q <= correction_rounded_s64_w[31:0];
                        correction_s8_q <= correction_saturated_s8_w;
                        corrected_v_s8_q <= corrected_s8_w;
                        correction_saturation_q <= correction_saturation_w;
                        add_saturation_q <= add_saturation_w;
                        output_channel_q <= output_index_q;
                        output_last_q <= (output_index_q == OUTPUT_LAST_INDEX);
                        output_valid_q <= 1'b1;
                        if (correction_rounded_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            descriptor_error_q <= 1'b1;
                        end
                        if (output_index_q != OUTPUT_LAST_INDEX)
                            output_index_q <= output_index_q + 7'd1;
                    end
                end
                default: state_q <= R1_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
