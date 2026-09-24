`default_nettype none

module ace2_attn_max_predicate_retime_formal (
    input wire        clk_i,
    input wire        rst_ni,
    input wire        watchdog_fire_i,
    input wire        response_fault_i,
    input wire        idle_clear_i,
    input wire        round_i,
    input wire        wait_out_i,
    input wire [2:0]  token_idx_i,
    input wire        score_negative_i,
    input wire [64:0] rounded_abs_w_i,
    input wire [64:0] current_max_magnitude_i,
    input wire        current_max_negative_i
);
    wire captured_predicate_q =
        (token_idx_i == 3'd0) ||
        (!score_negative_i && current_max_negative_i) ||
        ((score_negative_i == current_max_negative_i) &&
         (score_negative_i ?
          (rounded_abs_w_i < current_max_magnitude_i) :
          (rounded_abs_w_i > current_max_magnitude_i)));

    wire legacy_wait_predicate =
        (token_idx_i == 3'd0) ||
        (!score_negative_i && current_max_negative_i) ||
        ((score_negative_i == current_max_negative_i) &&
         (score_negative_i ?
          (rounded_abs_w_i < current_max_magnitude_i) :
          (rounded_abs_w_i > current_max_magnitude_i)));

    reg [64:0] legacy_next_magnitude;
    reg        legacy_next_negative;
    reg [64:0] retimed_next_magnitude;
    reg        retimed_next_negative;
    reg [64:0] captured_rounded_q;
    reg [64:0] captured_max_magnitude_q;
    reg        captured_score_negative_q;
    reg        captured_max_negative_q;
    reg [2:0]  captured_token_idx_q;
    reg        sequential_predicate_q;
    reg        sequential_valid_q;

    wire captured_legacy_predicate =
        (captured_token_idx_q == 3'd0) ||
        (!captured_score_negative_q && captured_max_negative_q) ||
        ((captured_score_negative_q == captured_max_negative_q) &&
         (captured_score_negative_q ?
          (captured_rounded_q < captured_max_magnitude_q) :
          (captured_rounded_q > captured_max_magnitude_q)));

    initial begin
        captured_rounded_q = 65'd0;
        captured_max_magnitude_q = 65'd0;
        captured_score_negative_q = 1'b0;
        captured_max_negative_q = 1'b0;
        captured_token_idx_q = 3'd0;
        sequential_predicate_q = 1'b0;
        sequential_valid_q = 1'b0;
    end

    always @(posedge clk_i) begin
        if (!rst_ni || watchdog_fire_i || idle_clear_i) begin
            captured_rounded_q <= 65'd0;
            sequential_predicate_q <= 1'b0;
            sequential_valid_q <= 1'b0;
        end else if (!response_fault_i) begin
            if (round_i) begin
                captured_rounded_q <= rounded_abs_w_i;
                captured_max_magnitude_q <= current_max_magnitude_i;
                captured_score_negative_q <= score_negative_i;
                captured_max_negative_q <= current_max_negative_i;
                captured_token_idx_q <= token_idx_i;
                sequential_predicate_q <= captured_predicate_q;
                sequential_valid_q <= 1'b1;
            end else if (wait_out_i && sequential_valid_q) begin
                assert(sequential_predicate_q == captured_legacy_predicate);
                assert(
                    (sequential_predicate_q ? captured_rounded_q :
                     captured_max_magnitude_q) ==
                    (captured_legacy_predicate ? captured_rounded_q :
                     captured_max_magnitude_q)
                );
                assert(
                    (sequential_predicate_q ? captured_score_negative_q :
                     captured_max_negative_q) ==
                    (captured_legacy_predicate ? captured_score_negative_q :
                     captured_max_negative_q)
                );
                sequential_valid_q <= 1'b0;
            end
        end
    end

    always @* begin
        legacy_next_magnitude = current_max_magnitude_i;
        legacy_next_negative = current_max_negative_i;
        retimed_next_magnitude = current_max_magnitude_i;
        retimed_next_negative = current_max_negative_i;

        if (!rst_ni || watchdog_fire_i || idle_clear_i) begin
            legacy_next_magnitude = 65'd0;
            legacy_next_negative = 1'b0;
            retimed_next_magnitude = 65'd0;
            retimed_next_negative = 1'b0;
        end else if (!response_fault_i && wait_out_i) begin
            if (legacy_wait_predicate) begin
                legacy_next_magnitude = rounded_abs_w_i;
                legacy_next_negative = score_negative_i;
            end
            if (captured_predicate_q) begin
                retimed_next_magnitude = rounded_abs_w_i;
                retimed_next_negative = score_negative_i;
            end
        end

        assert(captured_predicate_q == legacy_wait_predicate);
        assert(retimed_next_magnitude == legacy_next_magnitude);
        assert(retimed_next_negative == legacy_next_negative);
    end
endmodule

`default_nettype wire
