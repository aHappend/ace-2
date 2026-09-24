`default_nettype none

module ace2_down_projection_residual_fusion_formal;
    (* anyseq *) logic clk_i;
    (* anyseq *) logic rst_ni;
    (* anyseq *) logic clear_i;
    (* anyseq *) logic start_valid_i;
    (* anyseq *) logic signed [31:0] accumulator_s32_i;
    (* anyseq *) logic signed [7:0] residual_s8_i;
    (* anyseq *) logic [31:0] accumulator_scale32_i;
    (* anyseq *) logic [31:0] residual_scale32_i;
    (* anyseq *) logic [31:0] destination_scale32_i;
    (* anyseq *) logic out_ready_i;

    logic start_ready_o;
    logic out_valid_o;
    logic signed [7:0] fused_s8_o;
    logic positive_saturation_o;
    logic negative_saturation_o;
    logic descriptor_error_o;
    logic numeric_overflow_o;
    logic signed [95:0] numerator_s96_o;
    logic [63:0] denominator_u64_o;
    logic signed [7:0] common_exponent_s8_o;
    logic [4:0] latency_cycles_u5_o;
    logic past_valid_q;

    ace2_down_projection_residual_fusion_core dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(start_valid_i),
        .start_ready_o(start_ready_o),
        .accumulator_s32_i(accumulator_s32_i),
        .residual_s8_i(residual_s8_i),
        .accumulator_scale32_i(accumulator_scale32_i),
        .residual_scale32_i(residual_scale32_i),
        .destination_scale32_i(destination_scale32_i),
        .out_valid_o(out_valid_o),
        .out_ready_i(out_ready_i),
        .fused_s8_o(fused_s8_o),
        .positive_saturation_o(positive_saturation_o),
        .negative_saturation_o(negative_saturation_o),
        .descriptor_error_o(descriptor_error_o),
        .numeric_overflow_o(numeric_overflow_o),
        .numerator_s96_o(numerator_s96_o),
        .denominator_u64_o(denominator_u64_o),
        .common_exponent_s8_o(common_exponent_s8_o),
        .latency_cycles_u5_o(latency_cycles_u5_o)
    );

    initial past_valid_q = 1'b0;

    always @* begin
        assert (!(positive_saturation_o && negative_saturation_o));
        assert (latency_cycles_u5_o <= 5'd10);
        if (!rst_ni || clear_i)
            assert (!start_ready_o);
        if (positive_saturation_o)
            assert (fused_s8_o == 8'sd127);
        if (negative_saturation_o)
            assert (fused_s8_o == -8'sd128);
        if (out_valid_o)
            assert (!(descriptor_error_o && numeric_overflow_o));
    end

    always @(posedge clk_i) begin
        past_valid_q <= 1'b1;
        if (past_valid_q && rst_ni && !clear_i &&
            $past(rst_ni && !clear_i && out_valid_o && !out_ready_i)) begin
            assert (out_valid_o);
            assert (fused_s8_o == $past(fused_s8_o));
            assert (positive_saturation_o == $past(positive_saturation_o));
            assert (negative_saturation_o == $past(negative_saturation_o));
            assert (descriptor_error_o == $past(descriptor_error_o));
            assert (numeric_overflow_o == $past(numeric_overflow_o));
            assert (numerator_s96_o == $past(numerator_s96_o));
            assert (denominator_u64_o == $past(denominator_u64_o));
            assert (common_exponent_s8_o == $past(common_exponent_s8_o));
            assert (latency_cycles_u5_o == $past(latency_cycles_u5_o));
        end
    end
endmodule

`default_nettype wire
