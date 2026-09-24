`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_post_attention_residual_sum_scale_alignment_tb;
    `include "ace2_layer16_post_attention_residual_sum_scale_alignment/ace2_layer16_post_attention_residual_sum_scale_alignment_constants.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg signed [7:0] residual_s8;
    reg [31:0] residual_scale32;
    reg signed [7:0] down_s8;
    reg [31:0] down_scale32;
    reg [31:0] output_scale32;
    reg [9:0] channel;
    reg last;
    wire out_valid;
    reg out_ready;
    wire signed [7:0] sum_s8;
    wire [31:0] output_scale32_out;
    wire [9:0] channel_out;
    wire last_out;
    wire saturation;
    wire descriptor_error;
    wire numeric_overflow;
    wire signed [7:0] common_exponent_s8;
    wire [4:0] latency_cycles_u5;

    reg [7:0] residual_mem [0:MODEL_HIDDEN-1];
    reg [31:0] residual_scale_mem [0:MODEL_HIDDEN-1];
    reg [7:0] down_mem [0:MODEL_HIDDEN-1];
    reg [31:0] down_scale_mem [0:MODEL_HIDDEN-1];
    reg [31:0] output_scale_mem [0:MODEL_HIDDEN-1];
    reg [7:0] expected_mem [0:MODEL_HIDDEN-1];
    reg [7:0] saturation_mem [0:MODEL_HIDDEN-1];
    reg [7:0] common_exponent_mem [0:MODEL_HIDDEN-1];

    integer failures;
    integer xz_count;
    integer directed_count;
    integer output_count;
    integer saturation_count;
    integer input_backpressure_cycles;
    integer output_stalls;
    integer cycle_count;
    integer index;
    integer guard;
    integer stall_index;
    reg signed [7:0] held_sum;
    reg [31:0] held_scale;
    reg [9:0] held_channel;
    reg held_last;
    reg held_saturation;
    reg held_descriptor_error;
    reg held_numeric_overflow;
    reg signed [7:0] held_common_exponent;
    reg [4:0] held_latency;

    ace2_layer16_post_attention_residual_sum_scale_alignment_core dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .in_valid_i(in_valid), .in_ready_o(in_ready),
        .residual_s8_i(residual_s8), .residual_scale32_i(residual_scale32),
        .attention_s8_i(down_s8), .attention_scale32_i(down_scale32),
        .output_scale32_i(output_scale32), .channel_i(channel), .last_i(last),
        .out_valid_o(out_valid), .out_ready_i(out_ready), .sum_s8_o(sum_s8),
        .output_scale32_o(output_scale32_out), .channel_o(channel_out),
        .last_o(last_out), .saturation_o(saturation),
        .descriptor_error_o(descriptor_error), .numeric_overflow_o(numeric_overflow),
        .common_exponent_s8_o(common_exponent_s8),
        .latency_cycles_u5_o(latency_cycles_u5)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (rst_n) begin
            if ((in_ready !== 1'b0) && (in_ready !== 1'b1))
                xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1))
                xz_count = xz_count + 1;
            if (out_valid && ((^sum_s8 === 1'bx) || (^output_scale32_out === 1'bx) ||
                              (^channel_out === 1'bx) || (last_out === 1'bx) ||
                              (saturation === 1'bx) || (descriptor_error === 1'bx) ||
                              (numeric_overflow === 1'bx) ||
                              (^common_exponent_s8 === 1'bx) ||
                              (^latency_cycles_u5 === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            in_valid = 1'b0;
            residual_s8 = 8'sd0;
            residual_scale32 = 32'h00008000;
            down_s8 = 8'sd0;
            down_scale32 = 32'h00008000;
            output_scale32 = 32'h00008000;
            channel = 10'd0;
            last = 1'b0;
            out_ready = 1'b0;
        end
    endtask

    task apply_reset;
        begin
            rst_n = 1'b0;
            clear = 1'b0;
            idle_inputs();
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) failures = failures + 1;
        end
    endtask

    task send_and_check;
        input signed [7:0] selected_residual;
        input [31:0] selected_residual_scale;
        input signed [7:0] selected_down;
        input [31:0] selected_down_scale;
        input [31:0] selected_output_scale;
        input [9:0] selected_channel;
        input selected_last;
        input [7:0] expected_sum;
        input expected_saturation;
        input expected_descriptor_error;
        input expected_numeric_overflow;
        input [7:0] expected_common_exponent;
        input [4:0] expected_latency;
        input integer output_stall_cycles;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            residual_s8 = selected_residual;
            residual_scale32 = selected_residual_scale;
            down_s8 = selected_down;
            down_scale32 = selected_down_scale;
            output_scale32 = selected_output_scale;
            channel = selected_channel;
            last = selected_last;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            guard = 0;
            while (!out_valid && guard < 32) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) failures = failures + 1;
            held_sum = sum_s8;
            held_scale = output_scale32_out;
            held_channel = channel_out;
            held_last = last_out;
            held_saturation = saturation;
            held_descriptor_error = descriptor_error;
            held_numeric_overflow = numeric_overflow;
            held_common_exponent = common_exponent_s8;
            held_latency = latency_cycles_u5;
            for (stall_index = 0; stall_index < output_stall_cycles; stall_index = stall_index + 1) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || sum_s8 !== held_sum || output_scale32_out !== held_scale ||
                    channel_out !== held_channel || last_out !== held_last ||
                    saturation !== held_saturation ||
                    descriptor_error !== held_descriptor_error ||
                    numeric_overflow !== held_numeric_overflow ||
                    common_exponent_s8 !== held_common_exponent ||
                    latency_cycles_u5 !== held_latency)
                    failures = failures + 1;
            end
            if (sum_s8 !== expected_sum || saturation !== expected_saturation ||
                descriptor_error !== expected_descriptor_error ||
                numeric_overflow !== expected_numeric_overflow ||
                output_scale32_out !== selected_output_scale ||
                channel_out !== selected_channel || last_out !== selected_last ||
                common_exponent_s8 !== expected_common_exponent ||
                latency_cycles_u5 !== expected_latency) begin
                $display("RESIDUAL_SUM_MISMATCH channel=%0d sum=%02x expected=%02x sat=%0d expected_sat=%0d desc=%0d overflow=%0d common_exp=%0d latency=%0d", selected_channel, sum_s8, expected_sum, saturation, expected_saturation, descriptor_error, numeric_overflow, common_exponent_s8, latency_cycles_u5);
                failures = failures + 1;
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    task held_valid_backpressure_probe;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            residual_s8 = 8'sd3;
            residual_scale32 = 32'h00ff8000;
            down_s8 = 8'sd0;
            down_scale32 = 32'h00008000;
            output_scale32 = 32'h00008000;
            channel = 10'd900;
            last = 1'b0;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            residual_s8 = 8'sd1;
            residual_scale32 = 32'h00ff8000;
            down_s8 = 8'sd1;
            down_scale32 = 32'h00ff8000;
            output_scale32 = 32'h00008000;
            channel = 10'd901;
            last = 1'b1;
            repeat (3) begin
                @(posedge clk);
                input_backpressure_cycles = input_backpressure_cycles + 1;
                if (in_ready) failures = failures + 1;
            end
            while (!out_valid) @(posedge clk);
            if (sum_s8 !== 8'sd2 || channel_out !== 10'd900 || latency_cycles_u5 !== 5'd10)
                failures = failures + 1;
            repeat (2) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || in_ready) failures = failures + 1;
            end
            @(negedge clk);
            out_ready = 1'b1;
            #0;
            if (!in_ready) failures = failures + 1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            out_ready = 1'b0;
            guard = 0;
            while (!out_valid && guard < 32) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid || sum_s8 !== 8'sd1 || channel_out !== 10'd901 ||
                !last_out || latency_cycles_u5 !== 5'd10)
                failures = failures + 1;
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    task clear_recovery_probe;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            residual_s8 = 8'sd7;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            repeat (2) @(posedge clk);
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) failures = failures + 1;
        end
    endtask

    task async_reset_recovery_probe;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            residual_s8 = 8'sd5;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            rst_n = 1'b0;
            #1;
            if (out_valid || in_ready) failures = failures + 1;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) failures = failures + 1;
        end
    endtask

    task idle_xz_probe;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            residual_s8 = 8'sbx;
            residual_scale32 = 32'bz;
            down_s8 = 8'sbz;
            down_scale32 = 32'bx;
            output_scale32 = 32'bz;
            channel = 10'bx;
            last = 1'bx;
            repeat (3) begin
                @(posedge clk);
                if (!in_ready || out_valid) failures = failures + 1;
            end
            @(negedge clk);
            idle_inputs();
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        directed_count = 0;
        output_count = 0;
        saturation_count = 0;
        input_backpressure_cycles = 0;
        output_stalls = 0;
        cycle_count = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();

        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/residual_s8.hex", residual_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/residual_scale32.hex", residual_scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/down_s8.hex", down_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/down_scale32.hex", down_scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/output_scale32.hex", output_scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/expected_sum_s8.hex", expected_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/expected_saturation.hex", saturation_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_residual_sum_scale_alignment/expected_common_exponent_s8.hex", common_exponent_mem);

        apply_reset();
        send_and_check(8'sd1, 32'h00ff8000, 8'sd0, 32'h00008000, 32'h00008000, 10'd0, 1'b0, 8'h00, 1'b0, 1'b0, 1'b0, 8'hff, 5'd10, 1);
        directed_count = directed_count + 1;
        send_and_check(8'sd3, 32'h00ff8000, 8'sd0, 32'h00008000, 32'h00008000, 10'd1, 1'b0, 8'h02, 1'b0, 1'b0, 1'b0, 8'hff, 5'd10, 0);
        directed_count = directed_count + 1;
        send_and_check(-8'sd1, 32'h00ff8000, 8'sd0, 32'h00008000, 32'h00008000, 10'd2, 1'b0, 8'h00, 1'b0, 1'b0, 1'b0, 8'hff, 5'd10, 2);
        directed_count = directed_count + 1;
        send_and_check(-8'sd3, 32'h00ff8000, 8'sd0, 32'h00008000, 32'h00008000, 10'd3, 1'b0, 8'hfe, 1'b0, 1'b0, 1'b0, 8'hff, 5'd10, 0);
        directed_count = directed_count + 1;
        send_and_check(8'sd1, 32'h00ff8000, 8'sd1, 32'h00ff8000, 32'h00008000, 10'd4, 1'b0, 8'h01, 1'b0, 1'b0, 1'b0, 8'hff, 5'd10, 1);
        directed_count = directed_count + 1;
        send_and_check(8'sd127, 32'h00018000, 8'sd0, 32'h00008000, 32'h00008000, 10'd5, 1'b0, 8'h7f, 1'b1, 1'b0, 1'b0, 8'h00, 5'd10, 0);
        directed_count = directed_count + 1;
        send_and_check(-8'sd128, 32'h00018000, 8'sd0, 32'h00008000, 32'h00008000, 10'd6, 1'b0, 8'h80, 1'b1, 1'b0, 1'b0, 8'h00, 5'd10, 0);
        directed_count = directed_count + 1;
        send_and_check(8'sd1, 32'h00000000, 8'sd0, 32'h00008000, 32'h00008000, 10'd7, 1'b0, 8'h00, 1'b0, 1'b1, 1'b0, 8'h00, 5'd0, 0);
        directed_count = directed_count + 1;
        send_and_check(8'sd1, 32'h00018000, 8'sd1, 32'h00008000, 32'h00008000, 10'd8, 1'b0, 8'h03, 1'b0, 1'b0, 1'b0, 8'h00, 5'd10, 0);
        directed_count = directed_count + 1;

        held_valid_backpressure_probe();
        clear_recovery_probe();
        async_reset_recovery_probe();
        idle_xz_probe();

        for (index = 0; index < MODEL_HIDDEN; index = index + 1) begin
            send_and_check(
                residual_mem[index], residual_scale_mem[index],
                down_mem[index], down_scale_mem[index], output_scale_mem[index],
                index[9:0], index == MODEL_HIDDEN - 1, expected_mem[index],
                saturation_mem[index][0], 1'b0, 1'b0,
                common_exponent_mem[index], 5'd10, index % 4
            );
            output_count = output_count + 1;
            saturation_count = saturation_count + saturation_mem[index][0];
        end

        if (output_count != MODEL_HIDDEN ||
            saturation_count != MODEL_SATURATION_COUNT || xz_count != 0)
            failures = failures + 1;
        if (failures == 0) begin
            $display("ACE2_LAYER16_POST_ATTENTION_RESIDUAL_SUM_SCALE_ALIGNMENT_RTL_PASS directed=%0d outputs=%0d groups=%0d saturations=%0d input_backpressure=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1 stable_backpressure=1", directed_count, output_count, MODEL_GROUP_COUNT, saturation_count, input_backpressure_cycles, output_stalls, cycle_count);
        end else begin
            $display("ACE2_LAYER16_POST_ATTENTION_RESIDUAL_SUM_SCALE_ALIGNMENT_RTL_FAIL failures=%0d xz=%0d", failures, xz_count);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
