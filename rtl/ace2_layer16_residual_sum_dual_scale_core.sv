`timescale 1ns/1ps
`default_nettype none

// Layer-16 down_proj output-to-residual-sum boundary.
// Both carried operands remain signed A8 with independent Scale32 metadata.
// The operands are aligned into one bounded signed integer domain, added once,
// then rounded once (ties-to-even) and saturated once to signed A8.
module ace2_layer16_residual_sum_dual_scale_core (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       clear_i,

    input  logic                       in_valid_i,
    output logic                       in_ready_o,
    input  logic signed [7:0]          residual_s8_i,
    input  logic        [31:0]         residual_scale32_i,
    input  logic signed [7:0]          down_s8_i,
    input  logic        [31:0]         down_scale32_i,
    input  logic        [31:0]         output_scale32_i,
    input  logic        [9:0]          channel_i,
    input  logic                       last_i,

    output logic                       out_valid_o,
    input  logic                       out_ready_i,
    output logic signed [7:0]          sum_s8_o,
    output logic        [31:0]         output_scale32_o,
    output logic        [9:0]          channel_o,
    output logic                       last_o,
    output logic                       saturation_o,
    output logic                       descriptor_error_o,
    output logic                       numeric_overflow_o,
    output logic signed [7:0]          common_exponent_s8_o,
    output logic        [4:0]          latency_cycles_u5_o
);
    typedef enum logic [1:0] {
        RS_IDLE,
        RS_PREPARE,
        RS_DIVIDE,
        RS_FINALIZE
    } state_t;

    state_t state_q;
    logic out_valid_q;
    logic signed [7:0] sum_q;
    logic [31:0] output_scale_q;
    logic [9:0] channel_q;
    logic last_q;
    logic saturation_q;
    logic descriptor_error_q;
    logic numeric_overflow_q;
    logic signed [7:0] common_exponent_q;
    logic [4:0] latency_cycles_q;

    logic numerator_negative_q;
    logic [63:0] numerator_magnitude_q;
    logic [63:0] denominator_q;
    logic [63:0] remainder_q;
    logic [7:0] quotient_q;
    logic [2:0] divide_bit_q;
    logic positive_preclamp_q;
    logic negative_preclamp_q;

    logic residual_scale_valid_w;
    logic down_scale_valid_w;
    logic output_scale_valid_w;
    logic all_scales_valid_w;
    logic [15:0] residual_sig_w;
    logic [15:0] down_sig_w;
    logic [15:0] output_sig_w;
    logic signed [7:0] residual_exp_w;
    logic signed [7:0] down_exp_w;
    logic signed [7:0] output_exp_w;
    logic signed [7:0] common_exponent_w;
    logic [5:0] residual_shift_w;
    logic [5:0] down_shift_w;
    logic [5:0] output_shift_w;
    logic signed [23:0] residual_product_w;
    logic signed [23:0] down_product_w;
    logic signed [63:0] residual_term_w;
    logic signed [63:0] down_term_w;
    logic signed [64:0] widened_sum_w;
    logic numerator_overflow_w;
    logic signed [63:0] numerator_w;
    logic [63:0] numerator_magnitude_w;
    logic [63:0] denominator_w;
    logic denominator_overflow_w;

    logic [64:0] doubled_magnitude_w;
    logic [64:0] positive_threshold_w;
    logic [64:0] negative_threshold_w;
    logic positive_preclamp_w;
    logic negative_preclamp_w;
    logic [63:0] shifted_denominator_w;
    logic [63:0] remainder_next_w;
    logic [7:0] quotient_next_w;
    logic [64:0] doubled_remainder_w;
    logic [64:0] extended_denominator_w;
    logic round_increment_w;
    logic [8:0] rounded_magnitude_w;
    logic positive_saturation_w;
    logic negative_saturation_w;
    logic signed [7:0] rounded_s8_w;

    function automatic logic scale32_valid(input logic [31:0] record);
        logic signed [7:0] exponent;
        begin
            exponent = $signed(record[23:16]);
            scale32_valid =
                (record[31:24] == 8'd0) &&
                (record[15:0] >= 16'h8000) &&
                (exponent >= -8'sd24) &&
                (exponent <= 8'sd4);
        end
    endfunction

    function automatic logic [5:0] exponent_delta(
        input logic signed [7:0] exponent,
        input logic signed [7:0] common_exponent
    );
        begin
            exponent_delta = 6'($unsigned(
                $signed({exponent[7], exponent}) -
                $signed({common_exponent[7], common_exponent})
            ));
        end
    endfunction

    function automatic logic [64:0] multiply_u64_by_255(input logic [63:0] value);
        logic [64:0] extended;
        begin
            extended = {1'b0, value};
            multiply_u64_by_255 = (extended << 8) - extended;
        end
    endfunction

    function automatic logic [64:0] multiply_u64_by_257(input logic [63:0] value);
        logic [64:0] extended;
        begin
            extended = {1'b0, value};
            multiply_u64_by_257 = (extended << 8) + extended;
        end
    endfunction

    assign in_ready_o = rst_ni && !clear_i && (state_q == RS_IDLE) &&
                        (!out_valid_q || out_ready_i);
    assign out_valid_o = out_valid_q;
    assign sum_s8_o = sum_q;
    assign output_scale32_o = output_scale_q;
    assign channel_o = channel_q;
    assign last_o = last_q;
    assign saturation_o = saturation_q;
    assign descriptor_error_o = descriptor_error_q;
    assign numeric_overflow_o = numeric_overflow_q;
    assign common_exponent_s8_o = common_exponent_q;
    assign latency_cycles_u5_o = latency_cycles_q;

    always @* begin
        residual_scale_valid_w = scale32_valid(residual_scale32_i);
        down_scale_valid_w = scale32_valid(down_scale32_i);
        output_scale_valid_w = scale32_valid(output_scale32_i);
        all_scales_valid_w = residual_scale_valid_w &&
                             down_scale_valid_w &&
                             output_scale_valid_w;

        residual_sig_w = residual_scale32_i[15:0];
        down_sig_w = down_scale32_i[15:0];
        output_sig_w = output_scale32_i[15:0];
        residual_exp_w = $signed(residual_scale32_i[23:16]);
        down_exp_w = $signed(down_scale32_i[23:16]);
        output_exp_w = $signed(output_scale32_i[23:16]);

        common_exponent_w = residual_exp_w;
        if (down_exp_w < common_exponent_w)
            common_exponent_w = down_exp_w;
        if (output_exp_w < common_exponent_w)
            common_exponent_w = output_exp_w;

        residual_shift_w = 6'd0;
        down_shift_w = 6'd0;
        output_shift_w = 6'd0;
        residual_product_w = 24'sd0;
        down_product_w = 24'sd0;
        residual_term_w = 64'sd0;
        down_term_w = 64'sd0;
        widened_sum_w = 65'sd0;
        numerator_overflow_w = 1'b0;
        numerator_w = 64'sd0;
        numerator_magnitude_w = 64'd0;
        denominator_w = 64'd0;
        denominator_overflow_w = 1'b0;

        if (all_scales_valid_w) begin
            residual_shift_w = exponent_delta(residual_exp_w, common_exponent_w);
            down_shift_w = exponent_delta(down_exp_w, common_exponent_w);
            output_shift_w = exponent_delta(output_exp_w, common_exponent_w);
            residual_product_w = residual_s8_i * $signed({1'b0, residual_sig_w});
            down_product_w = down_s8_i * $signed({1'b0, down_sig_w});
            residual_term_w =
                $signed({{40{residual_product_w[23]}}, residual_product_w})
                <<< residual_shift_w;
            down_term_w =
                $signed({{40{down_product_w[23]}}, down_product_w})
                <<< down_shift_w;
            widened_sum_w =
                $signed({residual_term_w[63], residual_term_w}) +
                $signed({down_term_w[63], down_term_w});
            numerator_overflow_w = widened_sum_w[64] != widened_sum_w[63];
            numerator_w = widened_sum_w[63:0];
            numerator_magnitude_w = numerator_w[63] ?
                $unsigned(-numerator_w) : $unsigned(numerator_w);
            denominator_w = {{48{1'b0}}, output_sig_w} << output_shift_w;
            denominator_overflow_w = denominator_w == 64'd0;
        end

        doubled_magnitude_w = {numerator_magnitude_q, 1'b0};
        positive_threshold_w = multiply_u64_by_255(denominator_q);
        negative_threshold_w = multiply_u64_by_257(denominator_q);
        positive_preclamp_w = !numerator_negative_q &&
                              (doubled_magnitude_w >= positive_threshold_w);
        negative_preclamp_w = numerator_negative_q &&
                              (doubled_magnitude_w > negative_threshold_w);

        shifted_denominator_w = denominator_q << divide_bit_q;
        remainder_next_w = remainder_q;
        quotient_next_w = quotient_q;
        if (remainder_q >= shifted_denominator_w) begin
            remainder_next_w = remainder_q - shifted_denominator_w;
            quotient_next_w[divide_bit_q] = 1'b1;
        end

        doubled_remainder_w = {remainder_q, 1'b0};
        extended_denominator_w = {1'b0, denominator_q};
        round_increment_w =
            (doubled_remainder_w > extended_denominator_w) ||
            ((doubled_remainder_w == extended_denominator_w) && quotient_q[0]);
        rounded_magnitude_w = {1'b0, quotient_q} +
                              {{8{1'b0}}, round_increment_w};
        positive_saturation_w = positive_preclamp_q ||
            (!numerator_negative_q && (rounded_magnitude_w > 9'd127));
        negative_saturation_w = negative_preclamp_q ||
            (numerator_negative_q && (rounded_magnitude_w > 9'd128));
        rounded_s8_w = numerator_negative_q ?
            -$signed(rounded_magnitude_w[7:0]) :
             $signed(rounded_magnitude_w[7:0]);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= RS_IDLE;
            out_valid_q <= 1'b0;
            sum_q <= 8'sd0;
            output_scale_q <= 32'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            common_exponent_q <= 8'sd0;
            latency_cycles_q <= 5'd0;
            numerator_negative_q <= 1'b0;
            numerator_magnitude_q <= 64'd0;
            denominator_q <= 64'd0;
            remainder_q <= 64'd0;
            quotient_q <= 8'd0;
            divide_bit_q <= 3'd0;
            positive_preclamp_q <= 1'b0;
            negative_preclamp_q <= 1'b0;
        end else if (clear_i) begin
            state_q <= RS_IDLE;
            out_valid_q <= 1'b0;
            sum_q <= 8'sd0;
            output_scale_q <= 32'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
            descriptor_error_q <= 1'b0;
            numeric_overflow_q <= 1'b0;
            common_exponent_q <= 8'sd0;
            latency_cycles_q <= 5'd0;
            numerator_negative_q <= 1'b0;
            numerator_magnitude_q <= 64'd0;
            denominator_q <= 64'd0;
            remainder_q <= 64'd0;
            quotient_q <= 8'd0;
            divide_bit_q <= 3'd0;
            positive_preclamp_q <= 1'b0;
            negative_preclamp_q <= 1'b0;
        end else begin
            if (out_valid_q && out_ready_i)
                out_valid_q <= 1'b0;

            case (state_q)
                RS_IDLE: begin
                    if (in_valid_i && in_ready_o) begin
                        sum_q <= 8'sd0;
                        output_scale_q <= output_scale32_i;
                        channel_q <= channel_i;
                        last_q <= last_i;
                        saturation_q <= 1'b0;
                        descriptor_error_q <= 1'b0;
                        numeric_overflow_q <= 1'b0;
                        common_exponent_q <= common_exponent_w;
                        latency_cycles_q <= 5'd0;
                        if (!all_scales_valid_w) begin
                            descriptor_error_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else if (numerator_overflow_w || denominator_overflow_w) begin
                            numeric_overflow_q <= 1'b1;
                            out_valid_q <= 1'b1;
                        end else begin
                            numerator_negative_q <= numerator_w[63];
                            numerator_magnitude_q <= numerator_magnitude_w;
                            denominator_q <= denominator_w;
                            state_q <= RS_PREPARE;
                        end
                    end
                end

                RS_PREPARE: begin
                    positive_preclamp_q <= positive_preclamp_w;
                    negative_preclamp_q <= negative_preclamp_w;
                    remainder_q <= (positive_preclamp_w || negative_preclamp_w) ?
                                   64'd0 : numerator_magnitude_q;
                    quotient_q <= 8'd0;
                    divide_bit_q <= 3'd7;
                    latency_cycles_q <= latency_cycles_q + 5'd1;
                    state_q <= RS_DIVIDE;
                end

                RS_DIVIDE: begin
                    remainder_q <= remainder_next_w;
                    quotient_q <= quotient_next_w;
                    latency_cycles_q <= latency_cycles_q + 5'd1;
                    if (divide_bit_q == 3'd0)
                        state_q <= RS_FINALIZE;
                    else
                        divide_bit_q <= divide_bit_q - 3'd1;
                end

                RS_FINALIZE: begin
                    saturation_q <= positive_saturation_w || negative_saturation_w;
                    if (positive_saturation_w)
                        sum_q <= 8'sd127;
                    else if (negative_saturation_w)
                        sum_q <= -8'sd128;
                    else
                        sum_q <= rounded_s8_w;
                    latency_cycles_q <= latency_cycles_q + 5'd1;
                    out_valid_q <= 1'b1;
                    state_q <= RS_IDLE;
                end

                default: begin
                    state_q <= RS_IDLE;
                    numeric_overflow_q <= 1'b1;
                    out_valid_q <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
