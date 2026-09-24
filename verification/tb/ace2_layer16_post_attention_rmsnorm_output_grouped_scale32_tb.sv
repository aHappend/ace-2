`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_tb;
    `include "ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_constants.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg signed [11:0] aligned_activation;
    reg signed [15:0] gain_s16_q8;
    reg [31:0] inv_rms_q30;
    reg [31:0] group_scale32;
    reg [4:0] position;
    reg [9:0] channel;
    reg last;
    wire out_valid;
    reg out_ready;
    wire signed [7:0] q;
    wire [31:0] output_scale32;
    wire [4:0] output_position;
    wire [9:0] output_channel;
    wire output_last;
    wire saturation;

    reg [11:0] aligned_mem [0:MODEL_SAMPLES-1];
    reg [15:0] gain_mem [0:MODEL_SAMPLES-1];
    reg [31:0] inv_mem [0:MODEL_POSITIONS-1];
    reg [31:0] scale_mem [0:MODEL_SCALE_RECORDS-1];
    reg [7:0] expected_mem [0:MODEL_SAMPLES-1];
    reg [7:0] saturation_mem [0:MODEL_SAMPLES-1];

    integer failures;
    integer xz_count;
    integer directed_count;
    integer output_count;
    integer input_stalls;
    integer output_stalls;
    integer cycle_count;
    integer index;
    integer position_index;
    integer channel_index;
    integer scale_index;
    integer stall_index;
    integer guard;
    reg signed [7:0] held_q;
    reg [31:0] held_scale;
    reg [4:0] held_position;
    reg [9:0] held_channel;
    reg held_last;
    reg held_saturation;

    ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .aligned_activation_i(aligned_activation),
        .gain_s16_q8_i(gain_s16_q8),
        .inv_rms_q30_i(inv_rms_q30),
        .group_scale32_i(group_scale32),
        .position_i(position),
        .channel_i(channel),
        .last_i(last),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .q_o(q),
        .group_scale32_o(output_scale32),
        .position_o(output_position),
        .channel_o(output_channel),
        .last_o(output_last),
        .saturation_o(saturation)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (rst_n) begin
            if ((in_ready !== 1'b0) && (in_ready !== 1'b1)) xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1)) xz_count = xz_count + 1;
            if (out_valid && ((^q === 1'bx) || (^output_scale32 === 1'bx) ||
                              (^output_position === 1'bx) ||
                              (^output_channel === 1'bx) ||
                              (output_last === 1'bx) ||
                              (saturation === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            in_valid = 1'b0;
            aligned_activation = 12'sd0;
            gain_s16_q8 = 16'sd0;
            inv_rms_q30 = 32'd0;
            group_scale32 = 32'h00e88000;
            position = 5'd0;
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
            if (!in_ready || out_valid) begin
                $display("LAYER16_POST_RMSNORM_OUTPUT_RESET_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task produce_only;
        input signed [11:0] selected_activation;
        input signed [15:0] selected_gain;
        input [31:0] selected_inv;
        input [31:0] selected_scale;
        input [4:0] selected_position;
        input [9:0] selected_channel;
        input selected_last;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            aligned_activation = selected_activation;
            gain_s16_q8 = selected_gain;
            inv_rms_q30 = selected_inv;
            group_scale32 = selected_scale;
            position = selected_position;
            channel = selected_channel;
            last = selected_last;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            guard = 0;
            while (!out_valid && guard < 16) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("LAYER16_POST_RMSNORM_OUTPUT_PRODUCE_TIMEOUT");
                failures = failures + 1;
            end
        end
    endtask

    task send_and_check;
        input signed [11:0] selected_activation;
        input signed [15:0] selected_gain;
        input [31:0] selected_inv;
        input [31:0] selected_scale;
        input [4:0] selected_position;
        input [9:0] selected_channel;
        input selected_last;
        input [7:0] expected_q;
        input expected_saturation;
        input integer stall_cycles;
        begin
            produce_only(selected_activation, selected_gain, selected_inv,
                         selected_scale, selected_position, selected_channel,
                         selected_last);
            held_q = q;
            held_scale = output_scale32;
            held_position = output_position;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            for (stall_index = 0; stall_index < stall_cycles; stall_index = stall_index + 1) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || q !== held_q || output_scale32 !== held_scale ||
                    output_position !== held_position || output_channel !== held_channel ||
                    output_last !== held_last || saturation !== held_saturation) begin
                    $display("LAYER16_POST_RMSNORM_OUTPUT_BACKPRESSURE_STABILITY_FAIL position=%0d channel=%0d", selected_position, selected_channel);
                    failures = failures + 1;
                end
            end
            if (q !== expected_q || saturation !== expected_saturation ||
                output_scale32 !== selected_scale || output_position !== selected_position ||
                output_channel !== selected_channel || output_last !== selected_last) begin
                $display("LAYER16_POST_RMSNORM_OUTPUT_MISMATCH position=%0d channel=%0d q=%02x expected=%02x sat=%0d expected_sat=%0d scale=%08x expected_scale=%08x",
                         selected_position, selected_channel, q, expected_q,
                         saturation, expected_saturation, output_scale32, selected_scale);
                failures = failures + 1;
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
            output_count = output_count + 1;
        end
    endtask

    task held_valid_backpressure_probe;
        begin
            produce_only(12'sd1, 16'sd256, 32'h20000000,
                         32'h00f88000, 5'd1, 10'd9, 1'b0);
            held_q = q;
            held_scale = output_scale32;
            held_position = output_position;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            @(negedge clk);
            aligned_activation = -12'sd3;
            gain_s16_q8 = 16'sd256;
            inv_rms_q30 = 32'h20000000;
            group_scale32 = 32'h00f88001;
            position = 5'd2;
            channel = 10'd10;
            last = 1'b0;
            in_valid = 1'b1;
            repeat (3) begin
                @(posedge clk);
                input_stalls = input_stalls + 1;
                if (in_ready || !out_valid || q !== held_q ||
                    output_scale32 !== held_scale || output_position !== held_position ||
                    output_channel !== held_channel || output_last !== held_last ||
                    saturation !== held_saturation) begin
                    $display("LAYER16_POST_RMSNORM_OUTPUT_HELD_VALID_BACKPRESSURE_FAIL");
                    failures = failures + 1;
                end
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
            in_valid = 1'b0;
            if (!out_valid || q !== 8'hfe || output_scale32 !== 32'h00f88001 ||
                output_position !== 5'd2 || output_channel !== 10'd10 ||
                output_last !== 1'b0 || saturation !== 1'b0) begin
                $display("LAYER16_POST_RMSNORM_OUTPUT_HELD_VALID_CAPTURE_FAIL");
                failures = failures + 1;
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    task clear_recovery_probe;
        begin
            produce_only(12'sd7, 16'sd256, 32'h40000000,
                         32'h00f88000, 5'd3, 10'd11, 1'b0);
            @(negedge clk);
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("LAYER16_POST_RMSNORM_OUTPUT_CLEAR_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        directed_count = 0;
        output_count = 0;
        input_stalls = 0;
        output_stalls = 0;
        cycle_count = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();

        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/aligned_activation_s12.hex", aligned_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/gain_s16_q8.hex", gain_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/inv_rms_q30.hex", inv_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/output_group_scale32.hex", scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/expected_s8.hex", expected_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_output_grouped_scale32/saturation.hex", saturation_mem);

        apply_reset();
        send_and_check(12'sd1, 16'sd256, 32'h20000000, 32'h00f88000,
                       5'd0, 10'd0, 1'b0, 8'h00, 1'b0, 1);
        directed_count = directed_count + 1;
        send_and_check(12'sd3, 16'sd256, 32'h20000000, 32'h00f88000,
                       5'd0, 10'd1, 1'b0, 8'h02, 1'b0, 1);
        directed_count = directed_count + 1;
        send_and_check(-12'sd1, 16'sd256, 32'h20000000, 32'h00f88000,
                       5'd0, 10'd2, 1'b0, 8'h00, 1'b0, 1);
        directed_count = directed_count + 1;
        send_and_check(-12'sd3, 16'sd256, 32'h20000000, 32'h00f88000,
                       5'd0, 10'd3, 1'b0, 8'hfe, 1'b0, 1);
        directed_count = directed_count + 1;
        send_and_check(12'sd2047, 16'sd32767, 32'h40000000, 32'h00f88000,
                       5'd0, 10'd4, 1'b0, 8'h7f, 1'b1, 1);
        directed_count = directed_count + 1;
        send_and_check(12'sd1, 16'sd256, 32'h20000000, 32'hff000001,
                       5'd0, 10'd5, 1'b0, 8'h00, 1'b1, 1);
        directed_count = directed_count + 1;
        held_valid_backpressure_probe();
        clear_recovery_probe();

        for (index = 0; index < MODEL_SAMPLES; index = index + 1) begin
            position_index = index / MODEL_HIDDEN;
            channel_index = index % MODEL_HIDDEN;
            scale_index = position_index * MODEL_GROUPS_PER_POSITION +
                          channel_index / MODEL_OUTPUT_GROUP_SIZE;
            send_and_check(
                aligned_mem[index],
                gain_mem[index],
                inv_mem[position_index],
                scale_mem[scale_index],
                position_index[4:0],
                channel_index[9:0],
                channel_index == MODEL_HIDDEN - 1,
                expected_mem[index],
                saturation_mem[index][0],
                (index % 257 == 0) ? 2 : 0
            );
        end

        if (output_count != MODEL_SAMPLES + directed_count) begin
            $display("LAYER16_POST_RMSNORM_OUTPUT_COUNT_FAIL got=%0d expected=%0d", output_count, MODEL_SAMPLES + directed_count);
            failures = failures + 1;
        end
        if (xz_count != 0) begin
            $display("LAYER16_POST_RMSNORM_OUTPUT_XZ_FAIL count=%0d", xz_count);
            failures = failures + 1;
        end
        if (failures == 0) begin
            $display("ACE2_LAYER16_POST_ATTENTION_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_PASS samples=%0d positions=%0d groups=%0d nonzero=%0d saturations=%0d input_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1 backpressure_stable=1",
                     MODEL_SAMPLES, MODEL_POSITIONS, MODEL_SCALE_RECORDS,
                     MODEL_NONZERO_OUTPUTS, MODEL_SATURATIONS, input_stalls,
                     output_stalls, cycle_count);
        end else begin
            $fatal(1, "LAYER16_POST_RMSNORM_OUTPUT_FAIL failures=%0d", failures);
        end
        $finish;
    end
endmodule

`default_nettype wire
