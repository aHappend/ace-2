`default_nettype none

module ace2_dynamic_scale32_formal;
    (* anyseq *) logic clk_i;
    (* anyseq *) logic rst_ni;
    (* anyseq *) logic clear_i;
    (* anyseq *) logic start_valid_i;
    (* anyseq *) logic [15:0] accumulator_tag_u16_i;
    (* anyseq *) logic event_valid_i;
    (* anyseq *) logic signed [31:0] partial_s32_i;
    (* anyseq *) logic [31:0] scale_a32_i;
    (* anyseq *) logic [31:0] scale_b32_i;
    (* anyseq *) logic event_last_i;
    (* anyseq *) logic result_ready_i;

    logic start_ready_o;
    logic event_ready_o;
    logic result_valid_o;
    logic signed [159:0] accumulator_s160_o;
    logic signed [7:0] canonical_exponent_s8_o;
    logic [15:0] accumulator_tag_u16_o;
    logic descriptor_error_o;
    logic numeric_overflow_o;
    logic past_valid_q;

    ace2_scale32_tagged_accumulator_core dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i,
        .start_ready_o,
        .accumulator_tag_u16_i,
        .event_valid_i,
        .event_ready_o,
        .partial_s32_i,
        .scale_a32_i,
        .scale_b32_i,
        .event_last_i,
        .result_valid_o,
        .result_ready_i,
        .accumulator_s160_o,
        .canonical_exponent_s8_o,
        .accumulator_tag_u16_o,
        .descriptor_error_o,
        .numeric_overflow_o
    );

    initial past_valid_q = 1'b0;

    always @* begin
        assert (!(descriptor_error_o && numeric_overflow_o));
        assert (canonical_exponent_s8_o == -8'sd78);
        if (!rst_ni || clear_i) begin
            assert (!start_ready_o);
            assert (!event_ready_o);
        end
    end

    always @(posedge clk_i) begin
        past_valid_q <= 1'b1;
        if (past_valid_q && rst_ni && !clear_i &&
            $past(rst_ni && !clear_i && result_valid_o && !result_ready_i)) begin
            assert (result_valid_o);
            assert (accumulator_s160_o == $past(accumulator_s160_o));
            assert (accumulator_tag_u16_o == $past(accumulator_tag_u16_o));
            assert (descriptor_error_o == $past(descriptor_error_o));
            assert (numeric_overflow_o == $past(numeric_overflow_o));
        end
    end
endmodule

`default_nettype wire

