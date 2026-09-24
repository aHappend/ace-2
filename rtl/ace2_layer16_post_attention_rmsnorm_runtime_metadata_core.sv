`default_nettype none

module ace2_layer16_post_attention_rmsnorm_runtime_metadata_core #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer OUTPUT_GROUP_SIZE = 128
) (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 in_valid_i,
    output wire                 in_ready_o,
    input  wire signed [7:0]    activation_s8_i,
    input  wire [31:0]          input_scale32_i,
    input  wire signed [15:0]   gain_s16_i,
    input  wire [31:0]          gain_scale32_i,
    input  wire [4:0]           position_i,
    input  wire [9:0]           channel_i,
    input  wire                 last_i,

    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire signed [7:0]    q_o,
    output wire [31:0]          group_scale32_o,
    output wire [4:0]           position_o,
    output wire [9:0]           channel_o,
    output wire                 last_o,
    output wire                 saturation_o,
    output wire                 warning_o,
    output wire signed [11:0]   aligned_activation_o,
    output wire signed [15:0]   gain_s16_q8_o,

    output wire signed [7:0]    common_exponent_o,
    output wire [5:0]           rebase_shift_o,
    output wire [47:0]          sumsq_o,
    output wire [47:0]          mean_square_o,
    output wire [11:0]          rms_ceil_o,
    output wire [30:0]          inv_rms_q30_o
);
    localparam integer GROUP_COUNT = HIDDEN_SIZE / OUTPUT_GROUP_SIZE;
    localparam integer INDEX_WIDTH = (HIDDEN_SIZE <= 1) ? 1 : $clog2(HIDDEN_SIZE);
    localparam integer GROUP_WIDTH = (GROUP_COUNT <= 1) ? 1 : $clog2(GROUP_COUNT);
    localparam integer GROUP_OFFSET_WIDTH = (OUTPUT_GROUP_SIZE <= 1) ? 1 : $clog2(OUTPUT_GROUP_SIZE);
    localparam [INDEX_WIDTH-1:0] LAST_INDEX = INDEX_WIDTH'(HIDDEN_SIZE - 1);
    localparam [47:0] HIDDEN_VALUE = 48'(HIDDEN_SIZE);

    localparam [3:0] ST_COLLECT  = 4'd0;
    localparam [3:0] ST_FIND_MAX = 4'd1;
    localparam [3:0] ST_REBASE   = 4'd2;
    localparam [3:0] ST_ALIGN    = 4'd3;
    localparam [3:0] ST_SQRT     = 4'd4;
    localparam [3:0] ST_DIVIDE   = 4'd5;
    localparam [3:0] ST_SCALE    = 4'd6;
    localparam [3:0] ST_OUTPUT   = 4'd7;
    localparam [3:0] ST_DRAIN    = 4'd8;

    reg [3:0] state_q;
    reg [INDEX_WIDTH-1:0] collect_index_q;
    reg [INDEX_WIDTH-1:0] scan_index_q;
    reg [INDEX_WIDTH-1:0] output_index_q;
    reg [GROUP_WIDTH-1:0] scale_group_q;
    reg [GROUP_WIDTH-1:0] output_group_q;

    reg signed [7:0] activation_mem [0:HIDDEN_SIZE-1];
    reg [15:0] input_significand_mem [0:HIDDEN_SIZE-1];
    reg signed [7:0] input_exponent_mem [0:HIDDEN_SIZE-1];
    reg signed [15:0] gain_integer_mem [0:HIDDEN_SIZE-1];
    reg [15:0] gain_significand_mem [0:HIDDEN_SIZE-1];
    reg signed [7:0] gain_exponent_mem [0:HIDDEN_SIZE-1];
    reg signed [11:0] aligned_mem [0:HIDDEN_SIZE-1];
    reg signed [7:0] group_exponent_mem [0:GROUP_COUNT-1];

    reg [4:0] frame_position_q;
    reg frame_warning_q;
    reg signed [7:0] common_exponent_q;
    reg [55:0] maximum_exact_magnitude_q;
    reg [5:0] rebase_shift_q;
    reg [47:0] sumsq_q;
    reg [47:0] mean_square_q;
    reg [11:0] rms_candidate_q;
    reg [11:0] rms_ceil_q;
    reg [30:0] inv_rms_q30_q;
    reg signed [7:0] scale_candidate_exponent_q;
    reg signed [11:0] output_aligned_q;
    reg signed [15:0] output_gain_q8_q;

    reg [30:0] divide_dividend_q;
    reg [10:0] divide_remainder_q;
    reg [29:0] divide_quotient_q;
    reg [5:0] divide_count_q;

    wire input_scale_valid_w;
    wire gain_scale_valid_w;
    wire signed [7:0] input_exponent_w;
    wire signed [7:0] gain_exponent_w;
    wire [15:0] sanitized_input_significand_w;
    wire signed [7:0] sanitized_input_exponent_w;
    wire [15:0] sanitized_gain_significand_w;
    wire signed [7:0] sanitized_gain_exponent_w;

    wire signed [7:0] scan_activation_w;
    wire [15:0] scan_input_significand_w;
    wire signed [7:0] scan_input_exponent_w;
    wire signed [15:0] scan_gain_integer_w;
    wire [15:0] scan_gain_significand_w;
    wire signed [7:0] scan_gain_exponent_w;
    wire signed [24:0] scan_activation_product_w;
    wire [24:0] scan_activation_product_abs_w;
    wire [5:0] scan_exponent_delta_w;
    wire [55:0] scan_exact_magnitude_w;
    wire [11:0] scan_rounded_magnitude_w;
    wire signed [11:0] scan_aligned_w;
    wire [11:0] scan_aligned_abs_w;
    wire [23:0] scan_aligned_square_w;
    wire [47:0] sumsq_next_w;
    wire [47:0] mean_square_next_w;
    wire [23:0] rms_square_w;

    wire [15:0] scan_gain_abs_w;
    wire [31:0] scan_gain_magnitude_w;
    wire signed [11:0] scan_stored_aligned_w;
    wire [11:0] scan_stored_aligned_abs_w;
    wire [43:0] scan_aligned_gain_magnitude_w;
    wire [74:0] scan_output_magnitude_w;
    reg [95:0] dynamic_limit_w;
    reg [95:0] gain_floor_left_w;
    reg [95:0] gain_floor_right_w;
    reg scale_lane_fits_w;
    integer dynamic_shift_w;
    integer gain_floor_shift_w;

    wire [11:0] divide_remainder_shift_w;
    wire divide_subtract_w;
    wire [10:0] divide_remainder_subtracted_w;
    wire [10:0] divide_remainder_next_w;
    wire [30:0] divide_quotient_next_w;
    wire [30:0] divide_dividend_next_w;

    wire signed [15:0] output_gain_integer_w;
    wire [15:0] output_gain_significand_w;
    wire signed [7:0] output_gain_exponent_w;
    wire signed [7:0] output_group_exponent_w;
    wire signed [32:0] output_gain_product_w;
    reg signed [63:0] output_gain_q8_wide_w;
    reg signed [15:0] output_gain_q8_w;
    reg output_gain_overflow_w;
    integer output_gain_shift_w;
    wire [31:0] output_group_scale32_w;

    wire core_in_valid_w;
    wire core_in_ready_w;
    wire core_out_valid_w;
    wire signed [7:0] core_q_w;
    wire [31:0] core_scale_w;
    wire [4:0] core_position_w;
    wire [9:0] core_channel_w;
    wire core_last_w;
    wire core_saturation_w;
    wire scan_group_last_w;
    wire output_group_last_w;

    function automatic scale32_valid;
        input [31:0] record;
        reg signed [8:0] exponent;
        begin
            exponent = $signed({record[23], record[23:16]});
            scale32_valid = (record[31:24] == 8'h00) &&
                            (record[15:0] >= 16'h8000) &&
                            (exponent >= -9'sd24) &&
                            (exponent <= 9'sd4);
        end
    endfunction

    function automatic [55:0] round_shift_even_u56;
        input [55:0] value;
        input [5:0] shift;
        reg [55:0] base;
        reg [55:0] mask;
        reg [55:0] remainder;
        reg [55:0] half;
        begin
            if (shift == 0) begin
                round_shift_even_u56 = value;
            end else begin
                base = value >> shift;
                mask = (56'd1 << shift) - 56'd1;
                remainder = value & mask;
                half = 56'd1 << (shift - 1'b1);
                if ((remainder > half) || ((remainder == half) && base[0]))
                    base = base + 56'd1;
                round_shift_even_u56 = base;
            end
        end
    endfunction

    function automatic [47:0] round_divide_even_hidden;
        input [47:0] value;
        reg [47:0] quotient;
        reg [47:0] remainder;
        begin
            quotient = value / HIDDEN_VALUE;
            remainder = value % HIDDEN_VALUE;
            if (((remainder << 1) > HIDDEN_VALUE) ||
                (((remainder << 1) == HIDDEN_VALUE) && quotient[0]))
                quotient = quotient + 48'd1;
            round_divide_even_hidden = quotient;
        end
    endfunction

    function automatic [11:0] round_shift_even_u56_to_u12;
        input [55:0] value;
        input [5:0] shift;
        begin
            round_shift_even_u56_to_u12 = 12'(round_shift_even_u56(value, shift));
        end
    endfunction

    function automatic signed [63:0] round_shift_even_s64;
        input signed [63:0] value;
        input integer shift;
        reg negative;
        reg [63:0] magnitude;
        reg [63:0] base;
        reg [63:0] mask;
        reg [63:0] remainder;
        reg [63:0] half;
        begin
            if (shift <= 0) begin
                round_shift_even_s64 = value <<< (-shift);
            end else begin
                negative = value[63];
                magnitude = negative ? (~value + 64'd1) : value;
                base = magnitude >> shift;
                mask = (64'd1 << shift) - 64'd1;
                remainder = magnitude & mask;
                half = 64'd1 << (shift - 1);
                if ((remainder > half) || ((remainder == half) && base[0]))
                    base = base + 64'd1;
                round_shift_even_s64 = negative ? -$signed(base) : $signed(base);
            end
        end
    endfunction

    assign input_scale_valid_w = scale32_valid(input_scale32_i);
    assign gain_scale_valid_w = scale32_valid(gain_scale32_i);
    assign input_exponent_w = $signed(input_scale32_i[23:16]);
    assign gain_exponent_w = $signed(gain_scale32_i[23:16]);
    assign sanitized_input_significand_w = input_scale_valid_w ? input_scale32_i[15:0] : 16'h8000;
    assign sanitized_input_exponent_w = input_scale_valid_w ? input_exponent_w : -8'sd24;
    assign sanitized_gain_significand_w = gain_scale_valid_w ? gain_scale32_i[15:0] : 16'h8000;
    assign sanitized_gain_exponent_w = gain_scale_valid_w ? gain_exponent_w : -8'sd24;

    assign scan_activation_w = activation_mem[scan_index_q];
    assign scan_input_significand_w = input_significand_mem[scan_index_q];
    assign scan_input_exponent_w = input_exponent_mem[scan_index_q];
    assign scan_gain_integer_w = gain_integer_mem[scan_index_q];
    assign scan_gain_significand_w = gain_significand_mem[scan_index_q];
    assign scan_gain_exponent_w = gain_exponent_mem[scan_index_q];
    assign scan_activation_product_w = scan_activation_w * $signed({1'b0, scan_input_significand_w});
    assign scan_activation_product_abs_w = scan_activation_product_w[24] ?
                                           (~scan_activation_product_w + 25'd1) :
                                           scan_activation_product_w;
    assign scan_exponent_delta_w = 6'(scan_input_exponent_w - common_exponent_q);
    assign scan_exact_magnitude_w = {{31{1'b0}}, scan_activation_product_abs_w} << scan_exponent_delta_w;
    assign scan_rounded_magnitude_w = round_shift_even_u56_to_u12(scan_exact_magnitude_w, rebase_shift_q);
    assign scan_aligned_w = scan_activation_w[7] ? -$signed(scan_rounded_magnitude_w) :
                                                  $signed(scan_rounded_magnitude_w);
    assign scan_aligned_abs_w = scan_rounded_magnitude_w;
    assign scan_aligned_square_w = scan_aligned_abs_w * scan_aligned_abs_w;
    assign sumsq_next_w = sumsq_q + {{24{1'b0}}, scan_aligned_square_w};
    assign mean_square_next_w = round_divide_even_hidden(sumsq_next_w);
    assign rms_square_w = rms_candidate_q * rms_candidate_q;

    assign scan_gain_abs_w = scan_gain_integer_w[15] ?
                             (~scan_gain_integer_w + 16'd1) : scan_gain_integer_w;
    assign scan_gain_magnitude_w = scan_gain_abs_w * scan_gain_significand_w;
    assign scan_stored_aligned_w = aligned_mem[scan_index_q];
    assign scan_stored_aligned_abs_w = scan_stored_aligned_w[11] ?
                                       (~scan_stored_aligned_w + 12'd1) :
                                       scan_stored_aligned_w;
    assign scan_aligned_gain_magnitude_w = scan_stored_aligned_abs_w *
                                           scan_gain_magnitude_w;
    assign scan_output_magnitude_w = scan_aligned_gain_magnitude_w * inv_rms_q30_q;

    always @* begin
        dynamic_shift_w = $signed({{24{scale_candidate_exponent_q[7]}}, scale_candidate_exponent_q}) -
                          $signed({{24{scan_gain_exponent_w[7]}}, scan_gain_exponent_w}) + 45;
        gain_floor_shift_w = $signed({{24{scale_candidate_exponent_q[7]}}, scale_candidate_exponent_q}) -
                             $signed({{24{scan_gain_exponent_w[7]}}, scan_gain_exponent_w}) + 7;
        dynamic_limit_w = 96'd127 << dynamic_shift_w;
        if (gain_floor_shift_w >= 0) begin
            gain_floor_left_w = {{64{1'b0}}, scan_gain_magnitude_w};
            gain_floor_right_w = 96'd32767 << gain_floor_shift_w;
        end else begin
            gain_floor_left_w = {{64{1'b0}}, scan_gain_magnitude_w} << (-gain_floor_shift_w);
            gain_floor_right_w = 96'd32767;
        end
        scale_lane_fits_w = ({{21{1'b0}}, scan_output_magnitude_w} <= dynamic_limit_w) &&
                            (gain_floor_left_w <= gain_floor_right_w);
    end

    assign divide_remainder_shift_w = {divide_remainder_q, divide_dividend_q[30]};
    assign divide_subtract_w = divide_remainder_shift_w >= rms_ceil_q;
    assign divide_remainder_subtracted_w = 11'(divide_remainder_shift_w - rms_ceil_q);
    assign divide_remainder_next_w = divide_subtract_w ?
                                     divide_remainder_subtracted_w :
                                     divide_remainder_shift_w[10:0];
    assign divide_quotient_next_w = {divide_quotient_q, divide_subtract_w};
    assign divide_dividend_next_w = {divide_dividend_q[29:0], 1'b0};

    assign output_gain_integer_w = gain_integer_mem[output_index_q];
    assign output_gain_significand_w = gain_significand_mem[output_index_q];
    assign output_gain_exponent_w = gain_exponent_mem[output_index_q];
    assign output_group_exponent_w = group_exponent_mem[output_group_q];
    assign output_gain_product_w = output_gain_integer_w * $signed({1'b0, output_gain_significand_w});
    assign output_group_scale32_w = {8'h00, output_group_exponent_w[7:0], 16'h8000};

    always @* begin
        output_gain_shift_w = $signed({{24{output_group_exponent_w[7]}}, output_group_exponent_w}) + 7 -
                              $signed({{24{output_gain_exponent_w[7]}}, output_gain_exponent_w});
        output_gain_q8_wide_w = round_shift_even_s64(
            {{31{output_gain_product_w[32]}}, output_gain_product_w},
            output_gain_shift_w
        );
        output_gain_overflow_w = 1'b0;
        if (output_gain_q8_wide_w > 64'sd32767) begin
            output_gain_q8_w = 16'sh7fff;
            output_gain_overflow_w = 1'b1;
        end else if (output_gain_q8_wide_w < -64'sd32768) begin
            output_gain_q8_w = -16'sd32768;
            output_gain_overflow_w = 1'b1;
        end else begin
            output_gain_q8_w = output_gain_q8_wide_w[15:0];
        end
    end

    assign in_ready_o = state_q == ST_COLLECT;
    assign core_in_valid_w = state_q == ST_OUTPUT;
    assign out_valid_o = core_out_valid_w;
    assign q_o = core_q_w;
    assign group_scale32_o = core_scale_w;
    assign position_o = core_position_w;
    assign channel_o = core_channel_w;
    assign last_o = core_last_w;
    assign saturation_o = core_saturation_w;
    assign warning_o = frame_warning_q;
    assign aligned_activation_o = output_aligned_q;
    assign gain_s16_q8_o = output_gain_q8_q;
    assign common_exponent_o = common_exponent_q;
    assign rebase_shift_o = rebase_shift_q;
    assign sumsq_o = sumsq_q;
    assign mean_square_o = mean_square_q;
    assign rms_ceil_o = rms_ceil_q;
    assign inv_rms_q30_o = inv_rms_q30_q;
    assign scan_group_last_w = &scan_index_q[GROUP_OFFSET_WIDTH-1:0];
    assign output_group_last_w = &output_index_q[GROUP_OFFSET_WIDTH-1:0];

    ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core output_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .in_valid_i(core_in_valid_w),
        .in_ready_o(core_in_ready_w),
        .aligned_activation_i(aligned_mem[output_index_q]),
        .gain_s16_q8_i(output_gain_q8_w),
        .inv_rms_q30_i({1'b0, inv_rms_q30_q}),
        .group_scale32_i(output_group_scale32_w),
        .position_i(frame_position_q),
        .channel_i(10'(output_index_q)),
        .last_i(output_index_q == LAST_INDEX),
        .out_valid_o(core_out_valid_w),
        .out_ready_i(out_ready_i),
        .q_o(core_q_w),
        .group_scale32_o(core_scale_w),
        .position_o(core_position_w),
        .channel_o(core_channel_w),
        .last_o(core_last_w),
        .saturation_o(core_saturation_w)
    );

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_COLLECT;
            collect_index_q <= {INDEX_WIDTH{1'b0}};
            scan_index_q <= {INDEX_WIDTH{1'b0}};
            output_index_q <= {INDEX_WIDTH{1'b0}};
            scale_group_q <= {GROUP_WIDTH{1'b0}};
            output_group_q <= {GROUP_WIDTH{1'b0}};
            frame_position_q <= 5'd0;
            frame_warning_q <= 1'b0;
            common_exponent_q <= -8'sd24;
            maximum_exact_magnitude_q <= 56'd0;
            rebase_shift_q <= 6'd0;
            sumsq_q <= 48'd0;
            mean_square_q <= 48'd0;
            rms_candidate_q <= 12'd1;
            rms_ceil_q <= 12'd1;
            inv_rms_q30_q <= 31'd0;
            scale_candidate_exponent_q <= -8'sd24;
            output_aligned_q <= 12'sd0;
            output_gain_q8_q <= 16'sd0;
            divide_dividend_q <= 31'd0;
            divide_remainder_q <= 11'd0;
            divide_quotient_q <= 30'd0;
            divide_count_q <= 6'd0;
        end else if (clear_i) begin
            state_q <= ST_COLLECT;
            collect_index_q <= {INDEX_WIDTH{1'b0}};
            scan_index_q <= {INDEX_WIDTH{1'b0}};
            output_index_q <= {INDEX_WIDTH{1'b0}};
            scale_group_q <= {GROUP_WIDTH{1'b0}};
            output_group_q <= {GROUP_WIDTH{1'b0}};
            frame_position_q <= 5'd0;
            frame_warning_q <= 1'b0;
            common_exponent_q <= -8'sd24;
            maximum_exact_magnitude_q <= 56'd0;
            rebase_shift_q <= 6'd0;
            sumsq_q <= 48'd0;
            mean_square_q <= 48'd0;
            rms_candidate_q <= 12'd1;
            rms_ceil_q <= 12'd1;
            inv_rms_q30_q <= 31'd0;
            scale_candidate_exponent_q <= -8'sd24;
            output_aligned_q <= 12'sd0;
            output_gain_q8_q <= 16'sd0;
            divide_dividend_q <= 31'd0;
            divide_remainder_q <= 11'd0;
            divide_quotient_q <= 30'd0;
            divide_count_q <= 6'd0;
        end else begin
            case (state_q)
                ST_COLLECT: begin
                    if (in_valid_i) begin
                        activation_mem[collect_index_q] <= activation_s8_i;
                        input_significand_mem[collect_index_q] <= sanitized_input_significand_w;
                        input_exponent_mem[collect_index_q] <= sanitized_input_exponent_w;
                        gain_integer_mem[collect_index_q] <= gain_s16_i;
                        gain_significand_mem[collect_index_q] <= sanitized_gain_significand_w;
                        gain_exponent_mem[collect_index_q] <= sanitized_gain_exponent_w;
                        if (collect_index_q == {INDEX_WIDTH{1'b0}}) begin
                            frame_position_q <= position_i;
                            common_exponent_q <= sanitized_input_exponent_w;
                            frame_warning_q <= !input_scale_valid_w ||
                                               !gain_scale_valid_w ||
                                               (channel_i != 10'd0) ||
                                               last_i;
                        end else begin
                            if (sanitized_input_exponent_w < common_exponent_q)
                                common_exponent_q <= sanitized_input_exponent_w;
                            frame_warning_q <= frame_warning_q ||
                                               !input_scale_valid_w ||
                                               !gain_scale_valid_w ||
                                               (position_i != frame_position_q) ||
                                               (channel_i != 10'(collect_index_q)) ||
                                               (last_i != (collect_index_q == LAST_INDEX));
                        end
                        if (collect_index_q == LAST_INDEX) begin
                            collect_index_q <= {INDEX_WIDTH{1'b0}};
                            scan_index_q <= {INDEX_WIDTH{1'b0}};
                            maximum_exact_magnitude_q <= 56'd0;
                            state_q <= ST_FIND_MAX;
                        end else begin
                            collect_index_q <= collect_index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                end

                ST_FIND_MAX: begin
                    if (scan_exact_magnitude_w > maximum_exact_magnitude_q)
                        maximum_exact_magnitude_q <= scan_exact_magnitude_w;
                    if (scan_index_q == LAST_INDEX) begin
                        if (scan_exact_magnitude_w > maximum_exact_magnitude_q)
                            maximum_exact_magnitude_q <= scan_exact_magnitude_w;
                        scan_index_q <= {INDEX_WIDTH{1'b0}};
                        rebase_shift_q <= 6'd0;
                        state_q <= ST_REBASE;
                    end else begin
                        scan_index_q <= scan_index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                    end
                end

                ST_REBASE: begin
                    if (round_shift_even_u56(maximum_exact_magnitude_q, rebase_shift_q) <= 56'd2047) begin
                        scan_index_q <= {INDEX_WIDTH{1'b0}};
                        sumsq_q <= 48'd0;
                        state_q <= ST_ALIGN;
                    end else if (rebase_shift_q == 6'd55) begin
                        frame_warning_q <= 1'b1;
                        scan_index_q <= {INDEX_WIDTH{1'b0}};
                        sumsq_q <= 48'd0;
                        state_q <= ST_ALIGN;
                    end else begin
                        rebase_shift_q <= rebase_shift_q + 6'd1;
                    end
                end

                ST_ALIGN: begin
                    aligned_mem[scan_index_q] <= scan_aligned_w;
                    sumsq_q <= sumsq_next_w;
                    if (scan_index_q == LAST_INDEX) begin
                        mean_square_q <= mean_square_next_w;
                        rms_candidate_q <= 12'd1;
                        scan_index_q <= {INDEX_WIDTH{1'b0}};
                        state_q <= ST_SQRT;
                    end else begin
                        scan_index_q <= scan_index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                    end
                end

                ST_SQRT: begin
                    if (({{24{1'b0}}, rms_square_w} >= mean_square_q) ||
                        (rms_candidate_q == 12'd2047)) begin
                        rms_ceil_q <= rms_candidate_q;
                        divide_dividend_q <= 31'h40000000;
                        divide_remainder_q <= 11'd0;
                        divide_quotient_q <= 30'd0;
                        divide_count_q <= 6'd31;
                        state_q <= ST_DIVIDE;
                    end else begin
                        rms_candidate_q <= rms_candidate_q + 12'd1;
                    end
                end

                ST_DIVIDE: begin
                    divide_dividend_q <= divide_dividend_next_w;
                    divide_remainder_q <= divide_remainder_next_w;
                    divide_quotient_q <= divide_quotient_next_w[29:0];
                    if (divide_count_q == 6'd1) begin
                        inv_rms_q30_q <= divide_quotient_next_w;
                        scan_index_q <= {INDEX_WIDTH{1'b0}};
                        scale_group_q <= {GROUP_WIDTH{1'b0}};
                        scale_candidate_exponent_q <= -8'sd24;
                        state_q <= ST_SCALE;
                    end else begin
                        divide_count_q <= divide_count_q - 6'd1;
                    end
                end

                ST_SCALE: begin
                    if (scale_lane_fits_w || (scale_candidate_exponent_q == 8'sd4)) begin
                        if (!scale_lane_fits_w)
                            frame_warning_q <= 1'b1;
                        if (scan_group_last_w || (scan_index_q == LAST_INDEX)) begin
                            group_exponent_mem[scale_group_q] <= scale_candidate_exponent_q;
                            if (scan_index_q == LAST_INDEX) begin
                                output_index_q <= {INDEX_WIDTH{1'b0}};
                                output_group_q <= {GROUP_WIDTH{1'b0}};
                                state_q <= ST_OUTPUT;
                            end else begin
                                scan_index_q <= scan_index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                                scale_group_q <= scale_group_q + {{(GROUP_WIDTH-1){1'b0}}, 1'b1};
                                scale_candidate_exponent_q <= -8'sd24;
                            end
                        end else begin
                            scan_index_q <= scan_index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end else begin
                        scale_candidate_exponent_q <= scale_candidate_exponent_q + 8'sd1;
                    end
                end

                ST_OUTPUT: begin
                    if (core_in_ready_w) begin
                        output_aligned_q <= aligned_mem[output_index_q];
                        output_gain_q8_q <= output_gain_q8_w;
                        if (output_gain_overflow_w)
                            frame_warning_q <= 1'b1;
                        if (output_index_q == LAST_INDEX) begin
                            state_q <= ST_DRAIN;
                        end else begin
                            if (output_group_last_w)
                                output_group_q <= output_group_q + {{(GROUP_WIDTH-1){1'b0}}, 1'b1};
                            output_index_q <= output_index_q + {{(INDEX_WIDTH-1){1'b0}}, 1'b1};
                        end
                    end
                end

                ST_DRAIN: begin
                    if (core_out_valid_w && out_ready_i) begin
                        collect_index_q <= {INDEX_WIDTH{1'b0}};
                        frame_warning_q <= 1'b0;
                        state_q <= ST_COLLECT;
                    end
                end

                default: begin
                    state_q <= ST_COLLECT;
                    collect_index_q <= {INDEX_WIDTH{1'b0}};
                    frame_warning_q <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
