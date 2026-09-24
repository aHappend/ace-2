`timescale 1ns/1ps
`default_nettype none

module ace2_candidate_layer17_dynamic_q_tb;
    localparam integer STEPS = 4;
    localparam integer CHANNELS = 896;
    localparam integer K_SIZE = 896;
    localparam integer MAC_LANES = 4;
    localparam integer GROUPS = K_SIZE / MAC_LANES;
    localparam integer GROUP_INDEX_WIDTH = $clog2(GROUPS + 1);
    localparam integer ACT_WORDS = STEPS * GROUPS;
    localparam integer WEIGHT_WORDS = CHANNELS * GROUPS;
    localparam integer OUTPUT_WORDS = STEPS * CHANNELS;

    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg [GROUP_INDEX_WIDTH-1:0] last_group;
    reg pair_valid;
    wire pair_ready;
    reg [31:0] act_data;
    reg [15:0] weight_data;
    reg meta_valid;
    wire meta_ready;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    reg signed [7:0] output_zero_point;
    reg signed [31:0] bias_accumulator;
    wire out_valid;
    reg out_ready;
    wire [7:0] out_data;
    wire signed [31:0] acc;
    wire accumulator_overflow;
    wire saturation_seen;

    reg [31:0] activation_mem [0:ACT_WORDS-1];
    reg [15:0] weight_mem [0:WEIGHT_WORDS-1];
    reg [31:0] multiplier_mem [0:OUTPUT_WORDS-1];
    reg [7:0] shift_mem [0:OUTPUT_WORDS-1];
    reg [7:0] expected_mem [0:OUTPUT_WORDS-1];
    reg [31:0] accumulator_mem [0:OUTPUT_WORDS-1];
    reg [7:0] saturation_mem [0:OUTPUT_WORDS-1];

    reg [2047:0] vector_dir;
    reg [4095:0] vector_path;
    integer failures;
    integer xz_count;
    integer step_index;
    integer channel_index;
    integer group_index;
    integer flat_output;
    integer guard;
    integer stall_count;
    reg [7:0] held_out;
    reg signed [31:0] held_acc;
    reg held_sat;

    ace2_w4a8_proj_core #(
        .K_SIZE(K_SIZE),
        .MAC_LANES(MAC_LANES)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .last_group_i(last_group),
        .pair_valid_i(pair_valid),
        .pair_ready_o(pair_ready),
        .act_data_i(act_data),
        .weight_data_i(weight_data),
        .meta_valid_i(meta_valid),
        .meta_ready_o(meta_ready),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .output_zero_point_i(output_zero_point),
        .bias_accumulator_i(bias_accumulator),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .acc_o(acc),
        .accumulator_overflow_o(accumulator_overflow),
        .saturation_seen_o(saturation_seen)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            if ((start_ready !== 1'b0) && (start_ready !== 1'b1)) xz_count = xz_count + 1;
            if ((pair_ready !== 1'b0) && (pair_ready !== 1'b1)) xz_count = xz_count + 1;
            if ((meta_ready !== 1'b0) && (meta_ready !== 1'b1)) xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1)) xz_count = xz_count + 1;
            if (out_valid && ((^out_data === 1'bx) || (^acc === 1'bx) ||
                              (saturation_seen === 1'bx) ||
                              (accumulator_overflow === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            start_valid = 1'b0;
            pair_valid = 1'b0;
            meta_valid = 1'b0;
            act_data = 32'd0;
            weight_data = 16'd0;
            multiplier = 32'sd0;
            right_shift = 6'd0;
            output_zero_point = 8'sd0;
            bias_accumulator = 32'sd0;
            out_ready = 1'b0;
            last_group = GROUP_INDEX_WIDTH'(GROUPS - 1);
        end
    endtask

    task apply_reset;
        begin
            rst_n = 1'b0;
            clear = 1'b0;
            idle_inputs();
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
            if (!start_ready || out_valid) begin
                $display("DYNAMIC_Q_RESET_RECOVERY_FAIL start_ready=%b out_valid=%b", start_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task start_transaction;
        input integer groups_minus_one;
        begin
            while (!start_ready) @(posedge clk);
            last_group = GROUP_INDEX_WIDTH'(groups_minus_one);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task send_pair;
        input [31:0] selected_act;
        input [15:0] selected_weight;
        input integer insert_stall;
        begin
            if (insert_stall) @(posedge clk);
            while (!pair_ready) @(posedge clk);
            @(negedge clk);
            act_data = selected_act;
            weight_data = selected_weight;
            pair_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            pair_valid = 1'b0;
        end
    endtask

    task send_meta;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input signed [31:0] selected_bias;
        input integer insert_stall;
        begin
            if (insert_stall) repeat (2) @(posedge clk);
            while (!meta_ready) @(posedge clk);
            @(negedge clk);
            multiplier = selected_multiplier;
            right_shift = selected_shift;
            bias_accumulator = selected_bias;
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;
        end
    endtask

    task consume_and_check;
        input [7:0] expected_output;
        input signed [31:0] expected_accumulator;
        input expected_saturation;
        input integer backpressure_cycles;
        begin
            guard = 0;
            while (!out_valid && guard < 8192) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("DYNAMIC_Q_TIMEOUT");
                failures = failures + 1;
            end else begin
                held_out = out_data;
                held_acc = acc;
                held_sat = saturation_seen;
                for (stall_count = 0; stall_count < backpressure_cycles; stall_count = stall_count + 1) begin
                    @(posedge clk);
                    if (!out_valid || out_data !== held_out || acc !== held_acc ||
                        saturation_seen !== held_sat) begin
                        $display("DYNAMIC_Q_BACKPRESSURE_STABILITY_FAIL");
                        failures = failures + 1;
                    end
                end
                if (out_data !== expected_output ||
                    acc !== expected_accumulator ||
                    saturation_seen !== expected_saturation ||
                    accumulator_overflow) begin
                    $display("DYNAMIC_Q_MISMATCH out=%02x expected=%02x acc=%0d expected_acc=%0d sat=%0d expected_sat=%0d overflow=%0d",
                             out_data, expected_output, acc, expected_accumulator,
                             saturation_seen, expected_saturation, accumulator_overflow);
                    failures = failures + 1;
                end
                @(negedge clk);
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end
    endtask

    task clear_recovery_probe;
        begin
            start_transaction(GROUPS - 1);
            send_pair(activation_mem[0], weight_mem[0], 0);
            send_pair(activation_mem[1], weight_mem[1], 0);
            @(negedge clk);
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            repeat (3) @(posedge clk);
            if (!start_ready || out_valid) begin
                $display("DYNAMIC_Q_CLEAR_RECOVERY_FAIL start_ready=%b out_valid=%b", start_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task async_reset_recovery_probe;
        begin
            start_transaction(GROUPS - 1);
            send_pair(activation_mem[0], weight_mem[0], 0);
            @(negedge clk);
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
            if (!start_ready || out_valid) begin
                $display("DYNAMIC_Q_ASYNC_RESET_RECOVERY_FAIL start_ready=%b out_valid=%b", start_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task synthetic_case;
        input signed [31:0] selected_bias;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input [7:0] expected_output;
        input expected_saturation;
        begin
            start_transaction(0);
            send_pair(32'd0, 16'd0, 1);
            send_meta(selected_multiplier, selected_shift, selected_bias, 1);
            consume_and_check(expected_output, selected_bias, expected_saturation, 2);
        end
    endtask

    task run_model_channel;
        input integer selected_step;
        input integer selected_channel;
        begin
            flat_output = selected_step * CHANNELS + selected_channel;
            start_transaction(GROUPS - 1);
            for (group_index = 0; group_index < GROUPS; group_index = group_index + 1)
                send_pair(
                    activation_mem[selected_step * GROUPS + group_index],
                    weight_mem[selected_channel * GROUPS + group_index],
                    ((group_index + selected_channel + selected_step) % 11) == 0
                );
            send_meta(
                $signed(multiplier_mem[flat_output]),
                shift_mem[flat_output][5:0],
                32'sd0,
                ((selected_channel + selected_step) % 5) == 0
            );
            consume_and_check(
                expected_mem[flat_output],
                $signed(accumulator_mem[flat_output]),
                saturation_mem[flat_output][0],
                1 + ((selected_channel + selected_step) % 3)
            );
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir)) begin
            $display("DYNAMIC_Q_FAIL missing VECTOR_DIR plusarg");
            $finish_and_return(2);
        end
        $sformat(vector_path, "%0s/q_activation.hex", vector_dir);
        $readmemh(vector_path, activation_mem);
        $sformat(vector_path, "%0s/q_weight.hex", vector_dir);
        $readmemh(vector_path, weight_mem);
        $sformat(vector_path, "%0s/q_multiplier.hex", vector_dir);
        $readmemh(vector_path, multiplier_mem);
        $sformat(vector_path, "%0s/q_shift.hex", vector_dir);
        $readmemh(vector_path, shift_mem);
        $sformat(vector_path, "%0s/q_expected.hex", vector_dir);
        $readmemh(vector_path, expected_mem);
        $sformat(vector_path, "%0s/q_accumulator.hex", vector_dir);
        $readmemh(vector_path, accumulator_mem);
        $sformat(vector_path, "%0s/q_saturation.hex", vector_dir);
        $readmemh(vector_path, saturation_mem);

        apply_reset();
        clear_recovery_probe();
        async_reset_recovery_probe();
        synthetic_case(32'sd5, 32'sd1, 6'd1, 8'h02, 1'b0);
        synthetic_case(32'sd3, 32'sd1, 6'd1, 8'h02, 1'b0);
        synthetic_case(32'sd1000, 32'sd1, 6'd0, 8'h7f, 1'b1);
        synthetic_case(-32'sd1000, 32'sd1, 6'd0, 8'h80, 1'b1);

        for (step_index = 0; step_index < STEPS; step_index = step_index + 1) begin
            for (channel_index = 0; channel_index < CHANNELS; channel_index = channel_index + 1)
                run_model_channel(step_index, channel_index);
            $display("DYNAMIC_Q_STEP_PASS step=%0d channels=%0d", step_index, CHANNELS);
        end

        if (xz_count != 0) begin
            $display("DYNAMIC_Q_XZ_FAIL count=%0d", xz_count);
            failures = failures + xz_count;
        end
        if (failures == 0) begin
            $display("ACE2_CANDIDATE_LAYER17_DYNAMIC_Q_PASS steps=%0d channels=%0d reset=pass clear=pass stalls=pass rne=pass saturation=pass xz=0",
                     STEPS, CHANNELS);
            $finish;
        end
        $display("ACE2_CANDIDATE_LAYER17_DYNAMIC_Q_FAIL failures=%0d xz=%0d", failures, xz_count);
        $finish_and_return(1);
    end
endmodule

`default_nettype wire
