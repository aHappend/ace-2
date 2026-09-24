`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_group1_scale32_tb;
    `include "ace2_layer16_group1_scale32/ace2_layer16_group1_scale32_constants.svh"
    `define ASSERT_KNOWN(signal) \
        if ($isunknown(signal)) $fatal(1, "X/Z detected on %s at cycle %0d", `"signal`", cycle_count)

    localparam integer Q_GROUPS = MODEL_HIDDEN_SIZE;
    localparam integer RMS_BEATS = MODEL_HIDDEN_SIZE / 16;
    localparam integer Q_WEIGHT_WORDS = MODEL_Q_OUTPUTS * Q_GROUPS;

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;
    integer cycle_count = 0;
    integer failures = 0;
    integer directed_cases = 0;
    integer source_stall_cycles = 0;
    integer gain_stall_cycles = 0;
    integer weight_stall_cycles = 0;
    integer output_stall_cycles = 0;
    integer group_debug_count = 0;
    integer norm_debug_count = 0;
    logic model_check_active = 1'b0;

    always #5 clk_i = ~clk_i;
    always @(posedge clk_i)
        cycle_count <= cycle_count + 1;

    logic d_in_valid_i = 1'b0;
    logic d_in_ready_o;
    logic signed [7:0] d_attention_source_s8_i = '0;
    logic signed [7:0] d_down_source_s8_i = '0;
    logic [31:0] d_attention_base_scale32_i = '0;
    logic [31:0] d_down_base_scale32_i = '0;
    logic [31:0] d_attention_group_scale32_i = '0;
    logic [31:0] d_down_group_scale32_i = '0;
    logic [31:0] d_sum_scale32_i = '0;
    logic d_out_valid_o;
    logic d_out_ready_i = 1'b0;
    logic signed [7:0] d_attention_group_s8_o;
    logic signed [7:0] d_down_group_s8_o;
    logic signed [7:0] d_sum_s8_o;
    logic [0:0] d_attention_saturation_o;
    logic [0:0] d_down_saturation_o;
    logic [0:0] d_sum_saturation_o;
    logic d_descriptor_error_o;
    logic d_numeric_overflow_o;

    ace2_group1_scale32_quant_sum_core directed_dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .in_valid_i(d_in_valid_i),
        .in_ready_o(d_in_ready_o),
        .attention_source_s8_i(d_attention_source_s8_i),
        .down_source_s8_i(d_down_source_s8_i),
        .attention_base_scale32_i(d_attention_base_scale32_i),
        .down_base_scale32_i(d_down_base_scale32_i),
        .attention_group_scale32_i(d_attention_group_scale32_i),
        .down_group_scale32_i(d_down_group_scale32_i),
        .sum_scale32_i(d_sum_scale32_i),
        .out_valid_o(d_out_valid_o),
        .out_ready_i(d_out_ready_i),
        .attention_group_s8_o(d_attention_group_s8_o),
        .down_group_s8_o(d_down_group_s8_o),
        .sum_s8_o(d_sum_s8_o),
        .attention_saturation_o(d_attention_saturation_o),
        .down_saturation_o(d_down_saturation_o),
        .sum_saturation_o(d_sum_saturation_o),
        .descriptor_error_o(d_descriptor_error_o),
        .numeric_overflow_o(d_numeric_overflow_o)
    );

    logic start_valid_i = 1'b0;
    logic start_ready_o;
    logic [15:0] transaction_tag_i = '0;
    logic [31:0] attention_base_scale32_i = '0;
    logic [31:0] down_base_scale32_i = '0;
    logic [31:0] sum_scale32_i = '0;
    logic group_valid_i = 1'b0;
    logic group_ready_o;
    logic signed [7:0] attention_source_s8_i = '0;
    logic signed [7:0] down_source_s8_i = '0;
    logic [31:0] attention_group_scale32_i = '0;
    logic [31:0] down_group_scale32_i = '0;
    logic gain_valid_i = 1'b0;
    logic gain_ready_o;
    logic signed [255:0] gain_s16_q8_i = '0;
    logic q_row_start_valid_i = 1'b0;
    logic q_row_start_ready_o;
    logic [9:0] q_row_tag_i = '0;
    logic q_last_row_i = 1'b0;
    logic q_weight_valid_i = 1'b0;
    logic q_weight_ready_o;
    logic signed [3:0] q_weight_s4_i = '0;
    logic q_meta_valid_i = 1'b0;
    logic q_meta_ready_o;
    logic signed [31:0] q_multiplier_s32_i = '0;
    logic [5:0] q_right_shift_u6_i = '0;
    logic q_out_valid_o;
    logic q_out_ready_i = 1'b0;
    logic signed [7:0] q_out_s8_o;
    logic signed [31:0] q_accumulator_s32_o;
    logic [9:0] q_row_tag_o;
    logic q_saturation_o;
    logic q_accumulator_overflow_o;
    logic group_debug_valid_o;
    logic [9:0] group_debug_index_o;
    logic signed [7:0] group_debug_attention_s8_o;
    logic signed [7:0] group_debug_down_s8_o;
    logic signed [7:0] group_debug_sum_s8_o;
    logic [0:0] group_debug_attention_saturation_o;
    logic [0:0] group_debug_down_saturation_o;
    logic [0:0] group_debug_sum_saturation_o;
    logic norm_debug_valid_o;
    logic [5:0] norm_debug_index_o;
    logic signed [127:0] norm_debug_s8_o;
    logic done_valid_o;
    logic done_ready_i = 1'b0;
    logic [15:0] done_tag_o;
    logic [11:0] attention_saturation_count_o;
    logic [11:0] down_saturation_count_o;
    logic [11:0] sum_saturation_count_o;
    logic rms_saturation_o;
    logic [10:0] q_saturation_count_o;
    logic descriptor_error_o;
    logic numeric_overflow_o;

    ace2_layer16_group1_rmsnorm_q_core dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .start_valid_i,
        .start_ready_o,
        .transaction_tag_i,
        .attention_base_scale32_i,
        .down_base_scale32_i,
        .sum_scale32_i,
        .group_valid_i,
        .group_ready_o,
        .attention_source_s8_i,
        .down_source_s8_i,
        .attention_group_scale32_i,
        .down_group_scale32_i,
        .gain_valid_i,
        .gain_ready_o,
        .gain_s16_q8_i,
        .q_row_start_valid_i,
        .q_row_start_ready_o,
        .q_row_tag_i,
        .q_last_row_i,
        .q_weight_valid_i,
        .q_weight_ready_o,
        .q_weight_s4_i,
        .q_meta_valid_i,
        .q_meta_ready_o,
        .q_multiplier_s32_i,
        .q_right_shift_u6_i,
        .q_out_valid_o,
        .q_out_ready_i,
        .q_out_s8_o,
        .q_accumulator_s32_o,
        .q_row_tag_o,
        .q_saturation_o,
        .q_accumulator_overflow_o,
        .group_debug_valid_o,
        .group_debug_index_o,
        .group_debug_attention_s8_o,
        .group_debug_down_s8_o,
        .group_debug_sum_s8_o,
        .group_debug_attention_saturation_o,
        .group_debug_down_saturation_o,
        .group_debug_sum_saturation_o,
        .norm_debug_valid_o,
        .norm_debug_index_o,
        .norm_debug_s8_o,
        .done_valid_o,
        .done_ready_i,
        .done_tag_o,
        .attention_saturation_count_o,
        .down_saturation_count_o,
        .sum_saturation_count_o,
        .rms_saturation_o,
        .q_saturation_count_o,
        .descriptor_error_o,
        .numeric_overflow_o
    );

    logic [7:0] attention_source_mem [0:MODEL_GROUP_COUNT-1];
    logic [7:0] down_source_mem [0:MODEL_GROUP_COUNT-1];
    logic [31:0] attention_scale_mem [0:MODEL_GROUP_COUNT-1];
    logic [31:0] down_scale_mem [0:MODEL_GROUP_COUNT-1];
    logic [7:0] expected_attention_mem [0:MODEL_GROUP_COUNT-1];
    logic [7:0] expected_down_mem [0:MODEL_GROUP_COUNT-1];
    logic [7:0] expected_sum_mem [0:MODEL_GROUP_COUNT-1];
    logic [255:0] gain_mem [0:RMS_BEATS-1];
    logic [127:0] expected_rms_mem [0:RMS_BEATS-1];
    logic [3:0] q_weight_mem [0:Q_WEIGHT_WORDS-1];
    logic signed [31:0] q_multiplier_mem [0:MODEL_Q_OUTPUTS-1];
    logic [7:0] q_shift_mem [0:MODEL_Q_OUTPUTS-1];
    logic signed [31:0] q_expected_acc_mem [0:MODEL_Q_OUTPUTS-1];
    logic signed [7:0] q_expected_mem [0:MODEL_Q_OUTPUTS-1];
    logic [7:0] q_expected_saturation_mem [0:MODEL_Q_OUTPUTS-1];

    task automatic directed_case(
        input string name,
        input logic signed [7:0] attention_source,
        input logic signed [7:0] down_source,
        input logic [31:0] attention_base_scale,
        input logic [31:0] down_base_scale,
        input logic [31:0] attention_group_scale,
        input logic [31:0] down_group_scale,
        input logic [31:0] sum_scale,
        input logic signed [7:0] expected_attention,
        input logic signed [7:0] expected_down,
        input logic signed [7:0] expected_sum,
        input logic [0:0] expected_attention_saturation,
        input logic [0:0] expected_down_saturation,
        input logic [0:0] expected_sum_saturation,
        input logic expected_descriptor_error
    );
        logic signed [7:0] held_attention;
        logic signed [7:0] held_down;
        logic signed [7:0] held_sum;
        integer timeout;
        begin
            timeout = 0;
            while (!d_in_ready_o) begin
                @(posedge clk_i);
                timeout = timeout + 1;
                if (timeout > 20)
                    $fatal(1, "%s: directed input timeout", name);
            end
            @(negedge clk_i);
            d_attention_source_s8_i = attention_source;
            d_down_source_s8_i = down_source;
            d_attention_base_scale32_i = attention_base_scale;
            d_down_base_scale32_i = down_base_scale;
            d_attention_group_scale32_i = attention_group_scale;
            d_down_group_scale32_i = down_group_scale;
            d_sum_scale32_i = sum_scale;
            d_in_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            d_in_valid_i = 1'b0;
            timeout = 0;
            while (!d_out_valid_o) begin
                @(posedge clk_i);
                #1;
                timeout = timeout + 1;
                if (timeout > 3)
                    $fatal(1, "%s: directed output timeout", name);
            end
            if (d_attention_group_s8_o !== expected_attention ||
                d_down_group_s8_o !== expected_down ||
                d_sum_s8_o !== expected_sum ||
                d_attention_saturation_o !== expected_attention_saturation ||
                d_down_saturation_o !== expected_down_saturation ||
                d_sum_saturation_o !== expected_sum_saturation ||
                d_descriptor_error_o !== expected_descriptor_error ||
                d_numeric_overflow_o !== 1'b0)
                $fatal(1, "%s: directed result mismatch", name);
            held_attention = d_attention_group_s8_o;
            held_down = d_down_group_s8_o;
            held_sum = d_sum_s8_o;
            repeat (2) begin
                @(posedge clk_i);
                #1;
                if (!d_out_valid_o ||
                    d_attention_group_s8_o !== held_attention ||
                    d_down_group_s8_o !== held_down ||
                    d_sum_s8_o !== held_sum)
                    $fatal(1, "%s: directed output changed under stall", name);
            end
            @(negedge clk_i);
            d_out_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            d_out_ready_i = 1'b0;
            directed_cases = directed_cases + 1;
        end
    endtask

    task automatic start_transaction(input logic [15:0] tag);
        integer timeout;
        begin
            timeout = 0;
            while (!start_ready_o) begin
                @(posedge clk_i);
                timeout = timeout + 1;
                if (timeout > 20)
                    $fatal(1, "transaction start timeout");
            end
            @(negedge clk_i);
            transaction_tag_i = tag;
            attention_base_scale32_i = MODEL_ATTENTION_BASE_SCALE32;
            down_base_scale32_i = MODEL_DOWN_BASE_SCALE32;
            sum_scale32_i = MODEL_SUM_SCALE32;
            start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            start_valid_i = 1'b0;
        end
    endtask

    task automatic send_groups(input integer count);
        integer group_index;
        integer timeout;
        begin
            group_index = 0;
            timeout = 0;
            while (group_index < count) begin
                @(negedge clk_i);
                if (((cycle_count + group_index) % 5) == 0) begin
                    group_valid_i = 1'b0;
                    source_stall_cycles = source_stall_cycles + 1;
                end else begin
                    attention_source_s8_i = attention_source_mem[group_index];
                    down_source_s8_i = down_source_mem[group_index];
                    attention_group_scale32_i = attention_scale_mem[group_index];
                    down_group_scale32_i = down_scale_mem[group_index];
                    group_valid_i = 1'b1;
                end
                @(posedge clk_i);
                if (group_valid_i && group_ready_o) begin
                    group_index = group_index + 1;
                    timeout = 0;
                end else begin
                    timeout = timeout + 1;
                    if (timeout > 100)
                        $fatal(1, "group stream timeout at %0d", group_index);
                end
            end
            @(negedge clk_i);
            group_valid_i = 1'b0;
        end
    endtask

    task automatic send_gains;
        integer beat;
        integer timeout;
        begin
            beat = 0;
            timeout = 0;
            while (beat < RMS_BEATS) begin
                @(negedge clk_i);
                gain_s16_q8_i = gain_mem[beat];
                if (((cycle_count + beat) % 7) == 0) begin
                    gain_valid_i = 1'b0;
                    gain_stall_cycles = gain_stall_cycles + 1;
                end else begin
                    gain_valid_i = 1'b1;
                end
                @(posedge clk_i);
                if (gain_valid_i && gain_ready_o) begin
                    beat = beat + 1;
                    timeout = 0;
                end else begin
                    timeout = timeout + 1;
                    if (timeout > 20000)
                        $fatal(1, "gain stream timeout at %0d", beat);
                end
            end
            @(negedge clk_i);
            gain_valid_i = 1'b0;
        end
    endtask

    task automatic run_q_row(input integer row);
        integer group_index;
        integer timeout;
        integer stall_count;
        logic metadata_accepted;
        logic signed [7:0] held_output;
        logic signed [31:0] held_accumulator;
        logic held_saturation;
        begin
            timeout = 0;
            while (!q_row_start_ready_o) begin
                @(posedge clk_i);
                timeout = timeout + 1;
                if (timeout > 20000)
                    $fatal(1, "Q row start timeout at %0d", row);
            end
            @(negedge clk_i);
            q_row_tag_i = row[9:0];
            q_last_row_i = (row == MODEL_Q_OUTPUTS-1);
            q_row_start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            q_row_start_valid_i = 1'b0;

            group_index = 0;
            timeout = 0;
            while (group_index < Q_GROUPS) begin
                @(negedge clk_i);
                q_weight_s4_i = q_weight_mem[row*Q_GROUPS + group_index];
                if (((cycle_count + group_index + row) % 11) == 0) begin
                    q_weight_valid_i = 1'b0;
                    weight_stall_cycles = weight_stall_cycles + 1;
                end else begin
                    q_weight_valid_i = 1'b1;
                end
                @(posedge clk_i);
                if (q_weight_valid_i && q_weight_ready_o) begin
                    group_index = group_index + 1;
                    timeout = 0;
                end else begin
                    timeout = timeout + 1;
                    if (timeout > 100)
                        $fatal(1, "Q weight timeout row=%0d group=%0d", row, group_index);
                end
            end
            @(negedge clk_i);
            q_weight_valid_i = 1'b0;

            q_multiplier_s32_i = q_multiplier_mem[row];
            q_right_shift_u6_i = q_shift_mem[row][5:0];
            q_meta_valid_i = 1'b1;
            timeout = 0;
            metadata_accepted = 1'b0;
            while (!metadata_accepted) begin
                @(posedge clk_i);
                metadata_accepted = q_meta_valid_i && q_meta_ready_o;
                if (!metadata_accepted) begin
                    timeout = timeout + 1;
                    if (timeout > 20)
                        $fatal(1, "Q metadata timeout at %0d", row);
                end
            end
            @(negedge clk_i);
            q_meta_valid_i = 1'b0;

            timeout = 0;
            while (!q_out_valid_o) begin
                @(posedge clk_i);
                #1;
                timeout = timeout + 1;
                if (timeout > 300)
                    $fatal(1, "Q output timeout at %0d", row);
            end
            if (q_out_s8_o !== q_expected_mem[row] ||
                q_accumulator_s32_o !== q_expected_acc_mem[row] ||
                q_row_tag_o !== row[9:0] ||
                q_saturation_o !== q_expected_saturation_mem[row][0] ||
                q_accumulator_overflow_o !== 1'b0)
                $fatal(1, "Q output mismatch row=%0d out=%0d expected=%0d acc=%0d expected_acc=%0d sat=%0d expected_sat=%0d",
                    row, q_out_s8_o, q_expected_mem[row], q_accumulator_s32_o,
                    q_expected_acc_mem[row], q_saturation_o,
                    q_expected_saturation_mem[row][0]);
            held_output = q_out_s8_o;
            held_accumulator = q_accumulator_s32_o;
            held_saturation = q_saturation_o;
            stall_count = 1 + (row % 3);
            repeat (stall_count) begin
                @(posedge clk_i);
                #1;
                output_stall_cycles = output_stall_cycles + 1;
                if (!q_out_valid_o || q_out_s8_o !== held_output ||
                    q_accumulator_s32_o !== held_accumulator ||
                    q_saturation_o !== held_saturation)
                    $fatal(1, "Q output changed under stall row=%0d", row);
            end
            @(negedge clk_i);
            q_out_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            q_out_ready_i = 1'b0;
        end
    endtask

    always @(posedge clk_i) begin
        #1;
        if (rst_ni && !clear_i) begin
            `ASSERT_KNOWN(d_in_ready_o);
            `ASSERT_KNOWN(d_out_valid_o);
            `ASSERT_KNOWN(d_attention_group_s8_o);
            `ASSERT_KNOWN(d_down_group_s8_o);
            `ASSERT_KNOWN(d_sum_s8_o);
            `ASSERT_KNOWN(d_attention_saturation_o);
            `ASSERT_KNOWN(d_down_saturation_o);
            `ASSERT_KNOWN(d_sum_saturation_o);
            `ASSERT_KNOWN(d_descriptor_error_o);
            `ASSERT_KNOWN(d_numeric_overflow_o);
            `ASSERT_KNOWN(start_ready_o);
            `ASSERT_KNOWN(group_ready_o);
            `ASSERT_KNOWN(gain_ready_o);
            `ASSERT_KNOWN(q_row_start_ready_o);
            `ASSERT_KNOWN(q_weight_ready_o);
            `ASSERT_KNOWN(q_meta_ready_o);
            `ASSERT_KNOWN(q_out_valid_o);
            `ASSERT_KNOWN(q_out_s8_o);
            `ASSERT_KNOWN(q_accumulator_s32_o);
            `ASSERT_KNOWN(q_row_tag_o);
            `ASSERT_KNOWN(q_saturation_o);
            `ASSERT_KNOWN(q_accumulator_overflow_o);
            `ASSERT_KNOWN(group_debug_valid_o);
            `ASSERT_KNOWN(group_debug_index_o);
            `ASSERT_KNOWN(group_debug_attention_s8_o);
            `ASSERT_KNOWN(group_debug_down_s8_o);
            `ASSERT_KNOWN(group_debug_sum_s8_o);
            `ASSERT_KNOWN(group_debug_attention_saturation_o);
            `ASSERT_KNOWN(group_debug_down_saturation_o);
            `ASSERT_KNOWN(group_debug_sum_saturation_o);
            `ASSERT_KNOWN(norm_debug_valid_o);
            `ASSERT_KNOWN(norm_debug_index_o);
            `ASSERT_KNOWN(norm_debug_s8_o);
            `ASSERT_KNOWN(done_valid_o);
            `ASSERT_KNOWN(done_tag_o);
            `ASSERT_KNOWN(attention_saturation_count_o);
            `ASSERT_KNOWN(down_saturation_count_o);
            `ASSERT_KNOWN(sum_saturation_count_o);
            `ASSERT_KNOWN(rms_saturation_o);
            `ASSERT_KNOWN(q_saturation_count_o);
            `ASSERT_KNOWN(descriptor_error_o);
            `ASSERT_KNOWN(numeric_overflow_o);
        end
        if (model_check_active && group_debug_valid_o) begin
            if (group_debug_index_o !== group_debug_count[9:0] ||
                group_debug_attention_s8_o !== expected_attention_mem[group_debug_count] ||
                group_debug_down_s8_o !== expected_down_mem[group_debug_count] ||
                group_debug_sum_s8_o !== expected_sum_mem[group_debug_count] ||
                group_debug_attention_saturation_o !== 1'b0 ||
                group_debug_down_saturation_o !== 1'b0 ||
                group_debug_sum_saturation_o !== 1'b0)
                $fatal(1, "group debug mismatch at %0d", group_debug_count);
            group_debug_count = group_debug_count + 1;
        end
        if (model_check_active && norm_debug_valid_o) begin
            if (norm_debug_index_o !== norm_debug_count[5:0] ||
                norm_debug_s8_o !== expected_rms_mem[norm_debug_count])
                $fatal(1, "RMSNorm debug mismatch at %0d actual=%032h expected=%032h",
                    norm_debug_count, norm_debug_s8_o, expected_rms_mem[norm_debug_count]);
            norm_debug_count = norm_debug_count + 1;
        end
    end

    initial begin
        $readmemh("verification/generated/ace2_layer16_group1_scale32/attention_source_s8.hex", attention_source_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/down_source_s8.hex", down_source_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/attention_group_scale32.hex", attention_scale_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/down_group_scale32.hex", down_scale_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/expected_attention_group_s8.hex", expected_attention_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/expected_down_group_s8.hex", expected_down_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/expected_sum_s8.hex", expected_sum_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/rms_gain_s16_q8.hex", gain_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/expected_rms_s8.hex", expected_rms_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/q_weight_s4.hex", q_weight_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/q_multiplier_s32.hex", q_multiplier_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/q_shift_u6.hex", q_shift_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/q_expected_accumulator_s32.hex", q_expected_acc_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/q_expected_s8.hex", q_expected_mem);
        $readmemh("verification/generated/ace2_layer16_group1_scale32/q_expected_saturation.hex", q_expected_saturation_mem);

        repeat (4) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        if (!d_in_ready_o || !start_ready_o)
            $fatal(1, "cores did not become ready after reset release");

        directed_case(
            "signed_ties_even",
            8'sd5,
            -8'sd5,
            32'h00008000,
            32'h00008000,
            32'h00018000,
            32'h00018000,
            32'h00008000,
            8'sd2,
            -8'sd2,
            8'sd0,
            1'b0,
            1'b0,
            1'b0,
            1'b0
        );
        directed_case(
            "positive_saturation",
            8'sd127,
            8'sd127,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            8'sd127,
            8'sd127,
            8'sd127,
            1'b0,
            1'b0,
            1'b1,
            1'b0
        );
        directed_case(
            "negative_saturation",
            -8'sd128,
            -8'sd128,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            -8'sd128,
            -8'sd128,
            -8'sd128,
            1'b0,
            1'b0,
            1'b1,
            1'b0
        );
        directed_case(
            "descriptor_reject",
            8'sd4,
            8'sd8,
            32'h01008000,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            32'h00008000,
            8'sd0,
            8'sd0,
            8'sd0,
            1'b0,
            1'b0,
            1'b0,
            1'b1
        );

        @(negedge clk_i);
        d_attention_source_s8_i = 8'sd4;
        d_down_source_s8_i = 8'sd8;
        d_attention_base_scale32_i = 32'h00008000;
        d_down_base_scale32_i = 32'h00008000;
        d_attention_group_scale32_i = 32'h00008000;
        d_down_group_scale32_i = 32'h00008000;
        d_sum_scale32_i = 32'h00008000;
        d_in_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        d_in_valid_i = 1'b0;
        if (!d_out_valid_o)
            $fatal(1, "directed recovery setup did not produce output");
        clear_i = 1'b1;
        #1;
        if (d_in_ready_o || start_ready_o)
            $fatal(1, "clear did not suppress ready immediately");
        @(posedge clk_i);
        #1;
        if (d_out_valid_o)
            $fatal(1, "synchronous clear did not remove directed output");
        @(negedge clk_i);
        clear_i = 1'b0;
        @(posedge clk_i);
        #1;
        if (!d_in_ready_o || !start_ready_o)
            $fatal(1, "clear recovery did not restore ready");

        @(negedge clk_i);
        d_in_valid_i = 1'b1;
        @(posedge clk_i);
        #1;
        rst_ni = 1'b0;
        #1;
        if (d_out_valid_o || d_in_ready_o || start_ready_o)
            $fatal(1, "asynchronous reset did not clear active interfaces");
        @(negedge clk_i);
        d_in_valid_i = 1'b0;
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        if (!d_in_ready_o || !start_ready_o)
            $fatal(1, "asynchronous reset recovery failed");

        start_transaction(16'h1601);
        send_groups(5);
        @(negedge clk_i);
        clear_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        clear_i = 1'b0;
        @(posedge clk_i);
        #1;
        if (!start_ready_o || q_out_valid_o || done_valid_o)
            $fatal(1, "wrapper clear/recovery failed");

        group_debug_count = 0;
        norm_debug_count = 0;
        model_check_active = 1'b1;
        start_transaction(16'h1701);
        send_groups(MODEL_GROUP_COUNT);
        send_gains();
        while (!q_row_start_ready_o)
            @(posedge clk_i);
        #1;
        if (group_debug_count != MODEL_GROUP_COUNT)
            $fatal(1, "group debug count=%0d expected=%0d", group_debug_count, MODEL_GROUP_COUNT);
        if (norm_debug_count != RMS_BEATS)
            $fatal(1, "RMS debug count=%0d expected=%0d", norm_debug_count, RMS_BEATS);

        for (integer row = 0; row < MODEL_Q_OUTPUTS; row = row + 1)
            run_q_row(row);

        if (!done_valid_o)
            $fatal(1, "completion did not assert after last Q row");
        if (done_tag_o !== 16'h1701 ||
            attention_saturation_count_o !== 12'd0 ||
            down_saturation_count_o !== 12'd0 ||
            sum_saturation_count_o !== 12'd0 ||
            rms_saturation_o !== 1'b0 ||
            q_saturation_count_o !== MODEL_Q_SATURATION_COUNT ||
            descriptor_error_o !== 1'b0 || numeric_overflow_o !== 1'b0)
            $fatal(1, "completion status mismatch tag=%h att_sat=%0d down_sat=%0d sum_sat=%0d rms_sat=%0d q_sat=%0d desc=%0d overflow=%0d",
                done_tag_o, attention_saturation_count_o, down_saturation_count_o,
                sum_saturation_count_o, rms_saturation_o, q_saturation_count_o,
                descriptor_error_o, numeric_overflow_o);
        repeat (2) begin
            @(posedge clk_i);
            #1;
            if (!done_valid_o || done_tag_o !== 16'h1701)
                $fatal(1, "completion changed under backpressure");
        end
        @(negedge clk_i);
        done_ready_i = 1'b1;
        @(posedge clk_i);
        #1;
        @(negedge clk_i);
        done_ready_i = 1'b0;
        model_check_active = 1'b0;
        @(posedge clk_i);
        #1;
        if (!start_ready_o || done_valid_o)
            $fatal(1, "wrapper did not return to idle after completion");

        $display(
            "ACE2_LAYER16_GROUP1_SCALE32_RTL_PASS directed=%0d groups=%0d rms_beats=%0d q_outputs=%0d q_saturations=%0d source_stalls=%0d gain_stalls=%0d weight_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1",
            directed_cases, group_debug_count, norm_debug_count, MODEL_Q_OUTPUTS,
            MODEL_Q_SATURATION_COUNT, source_stall_cycles, gain_stall_cycles,
            weight_stall_cycles, output_stall_cycles, cycle_count
        );
        $finish;
    end

    initial begin
        repeat (4000000) @(posedge clk_i);
        $fatal(1, "focused RTL global timeout");
    end
endmodule

`undef ASSERT_KNOWN

`default_nettype wire
