`default_nettype none

module ace2_rmsnorm_dividend_equiv_formal;
    localparam [3:0] ST_COLLECT     = 4'd1;
    localparam [3:0] ST_MEAN_DIV    = 4'd2;
    localparam [3:0] ST_SQRT_DECIDE = 4'd4;
    localparam [3:0] ST_INV_DIV     = 4'd5;

    (* anyseq *) logic clk_i;
    (* anyseq *) logic rst_ni;
    (* anyseq *) logic clear_i;
    (* anyseq *) logic start_valid_i;
    (* anyseq *) logic in_valid_i;
    (* anyseq *) logic [7:0] in_data_i;
    (* anyseq *) logic gain_valid_i;
    (* anyseq *) logic [15:0] gain_data_i;
    (* anyseq *) logic scale_act_valid_i;
    (* anyseq *) logic [7:0] scale_act_data_i;
    (* anyseq *) logic out_ready_i;
    (* anyseq *) logic done_ready_i;

    wire pre_start_ready;
    wire pre_in_ready;
    wire pre_gain_ready;
    wire pre_scale_act_ready;
    wire pre_out_valid;
    wire [7:0] pre_out_data;
    wire pre_done_valid;
    wire [47:0] pre_sumsq;
    wire [31:0] pre_inv_rms;
    wire pre_saturation;

    wire post_start_ready;
    wire post_in_ready;
    wire post_gain_ready;
    wire post_scale_act_ready;
    wire post_out_valid;
    wire [7:0] post_out_data;
    wire post_done_valid;
    wire [47:0] post_sumsq;
    wire [31:0] post_inv_rms;
    wire post_saturation;

    wire [3:0] pre_state;
    wire [1:0] pre_collect_idx;
    wire [1:0] pre_scale_idx;
    wire pre_lane_idx;
    wire [7:0] pre_collect_beat;
    wire [7:0] pre_out_work;
    wire [47:0] pre_sumsq_q;
    wire [47:0] pre_mean_square;
    wire [15:0] pre_collect_square;
    wire [15:0] pre_rms_candidate;
    wire [31:0] pre_rms_square;
    wire [31:0] pre_inv_rms_q;
    wire [47:0] pre_dividend;
    wire [10:0] pre_divisor;
    wire [29:0] pre_quotient;
    wire [10:0] pre_remainder;
    wire [5:0] pre_count;
    wire pre_out_valid_q;
    wire pre_done_valid_q;
    wire pre_saturation_q;
    wire pre_sqrt_done;
    wire pre_collect_active;
    wire pre_collect_square_valid;
    wire pre_lane_last;
    wire pre_scale_active;
    wire pre_scale_product_sign;
    wire pre_scale_round_active;
    wire [7:0] pre_scale_act;
    wire [15:0] pre_scale_gain;
    wire [63:0] pre_mul_acc;
    wire [63:0] pre_mul_multiplicand;
    wire [31:0] pre_mul_multiplier;
    wire [31:0] pre_mul_addend_hi;
    wire pre_mul_carry;
    wire [5:0] pre_mul_count;

    wire [3:0] post_state;
    wire [1:0] post_collect_idx;
    wire [1:0] post_scale_idx;
    wire post_lane_idx;
    wire [7:0] post_collect_beat;
    wire [7:0] post_out_work;
    wire [47:0] post_sumsq_q;
    wire [47:0] post_mean_square;
    wire [15:0] post_collect_square;
    wire [15:0] post_rms_candidate;
    wire [31:0] post_rms_square;
    wire [31:0] post_inv_rms_q;
    wire [47:0] post_dividend;
    wire [10:0] post_divisor;
    wire [29:0] post_quotient;
    wire [10:0] post_remainder;
    wire [5:0] post_count;
    wire post_out_valid_q;
    wire post_done_valid_q;
    wire post_saturation_q;
    wire post_sqrt_done;
    wire post_collect_active;
    wire post_collect_square_valid;
    wire post_lane_last;
    wire post_scale_active;
    wire post_scale_product_sign;
    wire post_scale_round_active;
    wire [7:0] post_scale_act;
    wire [15:0] post_scale_gain;
    wire [63:0] post_mul_acc;
    wire [63:0] post_mul_multiplicand;
    wire [31:0] post_mul_multiplier;
    wire [31:0] post_mul_addend_hi;
    wire post_mul_carry;
    wire [5:0] post_mul_count;

    ace2_rmsnorm_core_prechange_formal #(
        .HIDDEN_SIZE(2),
        .LANES(1)
    ) pre_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i,
        .start_ready_o(pre_start_ready),
        .in_valid_i,
        .in_ready_o(pre_in_ready),
        .in_data_i,
        .gain_valid_i,
        .gain_ready_o(pre_gain_ready),
        .gain_data_i,
        .scale_act_valid_i,
        .scale_act_ready_o(pre_scale_act_ready),
        .scale_act_data_i,
        .out_valid_o(pre_out_valid),
        .out_ready_i,
        .out_data_o(pre_out_data),
        .done_valid_o(pre_done_valid),
        .done_ready_i,
        .sumsq_o(pre_sumsq),
        .inv_rms_q30_o(pre_inv_rms),
        .saturation_seen_o(pre_saturation),
        .formal_state_o(pre_state),
        .formal_collect_idx_o(pre_collect_idx),
        .formal_scale_idx_o(pre_scale_idx),
        .formal_lane_idx_o(pre_lane_idx),
        .formal_collect_beat_o(pre_collect_beat),
        .formal_out_work_o(pre_out_work),
        .formal_sumsq_q_o(pre_sumsq_q),
        .formal_mean_square_o(pre_mean_square),
        .formal_collect_square_o(pre_collect_square),
        .formal_rms_candidate_o(pre_rms_candidate),
        .formal_rms_square_o(pre_rms_square),
        .formal_inv_rms_q_o(pre_inv_rms_q),
        .formal_dividend_o(pre_dividend),
        .formal_divisor_o(pre_divisor),
        .formal_quotient_o(pre_quotient),
        .formal_remainder_o(pre_remainder),
        .formal_count_o(pre_count),
        .formal_out_valid_q_o(pre_out_valid_q),
        .formal_done_valid_q_o(pre_done_valid_q),
        .formal_saturation_q_o(pre_saturation_q),
        .formal_sqrt_done_o(pre_sqrt_done),
        .formal_collect_active_o(pre_collect_active),
        .formal_collect_square_valid_o(pre_collect_square_valid),
        .formal_lane_last_o(pre_lane_last),
        .formal_scale_active_o(pre_scale_active),
        .formal_scale_product_sign_o(pre_scale_product_sign),
        .formal_scale_round_active_o(pre_scale_round_active),
        .formal_scale_act_o(pre_scale_act),
        .formal_scale_gain_o(pre_scale_gain),
        .formal_mul_acc_o(pre_mul_acc),
        .formal_mul_multiplicand_o(pre_mul_multiplicand),
        .formal_mul_multiplier_o(pre_mul_multiplier),
        .formal_mul_addend_hi_o(pre_mul_addend_hi),
        .formal_mul_carry_o(pre_mul_carry),
        .formal_mul_count_o(pre_mul_count)
    );

    ace2_rmsnorm_core_formal #(
        .HIDDEN_SIZE(2),
        .LANES(1)
    ) post_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i,
        .start_ready_o(post_start_ready),
        .in_valid_i,
        .in_ready_o(post_in_ready),
        .in_data_i,
        .gain_valid_i,
        .gain_ready_o(post_gain_ready),
        .gain_data_i,
        .scale_act_valid_i,
        .scale_act_ready_o(post_scale_act_ready),
        .scale_act_data_i,
        .out_valid_o(post_out_valid),
        .out_ready_i,
        .out_data_o(post_out_data),
        .done_valid_o(post_done_valid),
        .done_ready_i,
        .sumsq_o(post_sumsq),
        .inv_rms_q30_o(post_inv_rms),
        .saturation_seen_o(post_saturation),
        .formal_state_o(post_state),
        .formal_collect_idx_o(post_collect_idx),
        .formal_scale_idx_o(post_scale_idx),
        .formal_lane_idx_o(post_lane_idx),
        .formal_collect_beat_o(post_collect_beat),
        .formal_out_work_o(post_out_work),
        .formal_sumsq_q_o(post_sumsq_q),
        .formal_mean_square_o(post_mean_square),
        .formal_collect_square_o(post_collect_square),
        .formal_rms_candidate_o(post_rms_candidate),
        .formal_rms_square_o(post_rms_square),
        .formal_inv_rms_q_o(post_inv_rms_q),
        .formal_dividend_o(post_dividend),
        .formal_divisor_o(post_divisor),
        .formal_quotient_o(post_quotient),
        .formal_remainder_o(post_remainder),
        .formal_count_o(post_count),
        .formal_out_valid_q_o(post_out_valid_q),
        .formal_done_valid_q_o(post_done_valid_q),
        .formal_saturation_q_o(post_saturation_q),
        .formal_sqrt_done_o(post_sqrt_done),
        .formal_collect_active_o(post_collect_active),
        .formal_collect_square_valid_o(post_collect_square_valid),
        .formal_lane_last_o(post_lane_last),
        .formal_scale_active_o(post_scale_active),
        .formal_scale_product_sign_o(post_scale_product_sign),
        .formal_scale_round_active_o(post_scale_round_active),
        .formal_scale_act_o(post_scale_act),
        .formal_scale_gain_o(post_scale_gain),
        .formal_mul_acc_o(post_mul_acc),
        .formal_mul_multiplicand_o(post_mul_multiplicand),
        .formal_mul_multiplier_o(post_mul_multiplier),
        .formal_mul_addend_hi_o(post_mul_addend_hi),
        .formal_mul_carry_o(post_mul_carry),
        .formal_mul_count_o(post_mul_count)
    );

    wire post_final_collect = (post_state == ST_COLLECT) &&
                              post_collect_active && post_collect_square_valid &&
                              post_lane_last && (post_collect_idx == 2'd1);
    wire post_sqrt_load = (post_state == ST_SQRT_DECIDE) && post_sqrt_done;
    wire [47:0] post_collect_load = post_sumsq_q +
                                    {{32{1'b0}}, post_collect_square} + 48'd1;
    wire [47:0] post_dividend_next = {post_dividend[46:0], 1'b0};
    wire [10:0] post_remainder_shift = {post_remainder[9:0], post_dividend[47]};
    wire post_subtract = post_remainder_shift >= post_divisor;
    wire [10:0] post_remainder_next = post_subtract ?
                                      (post_remainder_shift - post_divisor) :
                                      post_remainder_shift;
    wire [29:0] post_quotient_next = {post_quotient[28:0], post_subtract};
    wire [5:0] post_count_next = (post_count == 6'd1) ?
                                 6'd0 : (post_count - 6'd1);

    wire external_equal =
        (pre_start_ready === post_start_ready) &&
        (pre_in_ready === post_in_ready) &&
        (pre_gain_ready === post_gain_ready) &&
        (pre_scale_act_ready === post_scale_act_ready) &&
        (pre_out_valid === post_out_valid) &&
        (pre_out_data === post_out_data) &&
        (pre_done_valid === post_done_valid) &&
        (pre_sumsq === post_sumsq) &&
        (pre_inv_rms === post_inv_rms) &&
        (pre_saturation === post_saturation);

    wire retained_state_equal =
        ({pre_state, pre_collect_idx, pre_scale_idx, pre_lane_idx,
          pre_out_work, pre_sumsq_q, pre_mean_square, pre_collect_square,
          pre_rms_candidate, pre_rms_square, pre_inv_rms_q, pre_divisor,
          pre_quotient, pre_remainder, pre_count, pre_out_valid_q,
          pre_done_valid_q, pre_saturation_q, pre_sqrt_done,
          pre_collect_active, pre_collect_square_valid, pre_lane_last,
          pre_scale_active, pre_scale_product_sign, pre_scale_round_active,
          pre_scale_act, pre_scale_gain, pre_mul_acc, pre_mul_multiplicand,
          pre_mul_multiplier, pre_mul_addend_hi, pre_mul_carry, pre_mul_count} ===
         {post_state, post_collect_idx, post_scale_idx, post_lane_idx,
          post_out_work, post_sumsq_q, post_mean_square, post_collect_square,
          post_rms_candidate, post_rms_square, post_inv_rms_q, post_divisor,
          post_quotient, post_remainder, post_count, post_out_valid_q,
          post_done_valid_q, post_saturation_q, post_sqrt_done,
          post_collect_active, post_collect_square_valid, post_lane_last,
          post_scale_active, post_scale_product_sign, post_scale_round_active,
          post_scale_act, post_scale_gain, post_mul_acc, post_mul_multiplicand,
          post_mul_multiplier, post_mul_addend_hi, post_mul_carry, post_mul_count});

    logic history_valid_q;
    logic previous_rst_ni_q;
    logic [3:0] previous_state_q;
    logic previous_final_collect_q;
    logic previous_sqrt_load_q;
    logic [47:0] previous_collect_load_q;
    logic [47:0] previous_dividend_next_q;
    logic [10:0] previous_remainder_next_q;
    logic [29:0] previous_quotient_next_q;
    logic [5:0] previous_count_next_q;

    initial history_valid_q = 1'b0;
    initial previous_rst_ni_q = 1'b0;
    initial previous_state_q = 4'd0;
    initial previous_final_collect_q = 1'b0;
    initial previous_sqrt_load_q = 1'b0;
    initial previous_collect_load_q = 48'd0;
    initial previous_dividend_next_q = 48'd0;
    initial previous_remainder_next_q = 11'd0;
    initial previous_quotient_next_q = 30'd0;
    initial previous_count_next_q = 6'd0;

    wire stable_reset_history = history_valid_q && rst_ni && previous_rst_ni_q;
    wire property_external_equal = external_equal;
    wire property_retained_state_equal = retained_state_equal;
    wire property_collect_beat_equal = !post_collect_active ||
                                       (pre_collect_beat === post_collect_beat);
    wire property_divide_dividend_equal = ((post_state != ST_MEAN_DIV) &&
                                           (post_state != ST_INV_DIV)) ||
                                          (pre_dividend === post_dividend);
    wire property_candidate_update = !stable_reset_history ||
        (previous_final_collect_q ?
            (post_dividend == previous_collect_load_q) :
         previous_sqrt_load_q ?
            (post_dividend == (48'd1 << 30)) :
            (post_dividend == previous_dividend_next_q));
    wire property_mean_entry = !stable_reset_history ||
        !((post_state == ST_MEAN_DIV) && (previous_state_q != ST_MEAN_DIV)) ||
        (previous_final_collect_q && (post_dividend == previous_collect_load_q));
    wire property_inv_entry = !stable_reset_history ||
        !((post_state == ST_INV_DIV) && (previous_state_q != ST_INV_DIV)) ||
        (previous_sqrt_load_q && (post_dividend == (48'd1 << 30)));
    wire property_divider_transition = !stable_reset_history ||
        ((previous_state_q != ST_MEAN_DIV) && (previous_state_q != ST_INV_DIV)) ||
        ((post_dividend == previous_dividend_next_q) &&
         (post_remainder == previous_remainder_next_q) &&
         (post_quotient == previous_quotient_next_q) &&
         (post_count == previous_count_next_q));
    (* keep *) wire invariant_relation = property_external_equal &&
                                         property_retained_state_equal &&
                                         property_collect_beat_equal &&
                                         property_divide_dividend_equal;
    (* keep *) wire local_transition_properties = property_candidate_update &&
                                                  property_mean_entry &&
                                                  property_inv_entry &&
                                                  property_divider_transition;

    always @* begin
        assert (property_external_equal);
        assert (property_retained_state_equal);
        assert (property_collect_beat_equal);
        assert (property_divide_dividend_equal);
        assert (property_candidate_update);
        assert (property_mean_entry);
        assert (property_inv_entry);
        assert (property_divider_transition);
    end

    always @(posedge clk_i) begin
        history_valid_q <= 1'b1;
        previous_rst_ni_q <= rst_ni;
        previous_state_q <= post_state;
        previous_final_collect_q <= post_final_collect;
        previous_sqrt_load_q <= post_sqrt_load;
        previous_collect_load_q <= post_collect_load;
        previous_dividend_next_q <= post_dividend_next;
        previous_remainder_next_q <= post_remainder_next;
        previous_quotient_next_q <= post_quotient_next;
        previous_count_next_q <= post_count_next;
    end
endmodule

`default_nettype wire
