`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_down_proj_output_grouped_scale32_tb;
    `include "ace2_layer16_down_proj_output_grouped_scale32/ace2_layer16_down_proj_output_grouped_scale32_constants.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg signed [7:0] down;
    reg [31:0] down_base_scale32;
    reg [31:0] group_scale32;
    reg [9:0] channel;
    reg last;
    wire out_valid;
    reg out_ready;
    wire signed [7:0] grouped_down;
    wire [31:0] output_scale32;
    wire [9:0] output_channel;
    wire output_last;
    wire saturation;

    reg [7:0] down_mem [0:MODEL_HIDDEN-1];
    reg [31:0] base_scale_mem [0:0];
    reg [31:0] group_scale_mem [0:MODEL_GROUP_COUNT-1];
    reg [7:0] expected_mem [0:MODEL_HIDDEN-1];
    reg [7:0] saturation_mem [0:MODEL_HIDDEN-1];

    integer failures;
    integer xz_count;
    integer directed_count;
    integer output_count;
    integer saturation_count;
    integer input_stalls;
    integer output_stalls;
    integer cycle_count;
    integer index;
    integer guard;
    integer stall_index;
    reg signed [7:0] held_down;
    reg [31:0] held_scale;
    reg [9:0] held_channel;
    reg held_last;
    reg held_saturation;

    ace2_layer16_down_proj_output_grouped_scale32_core dut (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(clear),
        .in_valid_i(in_valid), .in_ready_o(in_ready),
        .down_i(down), .down_base_scale32_i(down_base_scale32),
        .group_scale32_i(group_scale32), .channel_i(channel), .last_i(last),
        .out_valid_o(out_valid), .out_ready_i(out_ready), .down_o(grouped_down),
        .group_scale32_o(output_scale32), .channel_o(output_channel),
        .last_o(output_last), .saturation_o(saturation)
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
            if (out_valid && ((^grouped_down === 1'bx) || (^output_scale32 === 1'bx) ||
                              (^output_channel === 1'bx) || (output_last === 1'bx) ||
                              (saturation === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            in_valid = 1'b0;
            down = 8'sd0;
            down_base_scale32 = 32'h00008000;
            group_scale32 = 32'h00008000;
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
        input signed [7:0] selected_down;
        input [31:0] selected_base_scale;
        input [31:0] selected_group_scale;
        input [9:0] selected_channel;
        input selected_last;
        input [7:0] expected_down;
        input expected_saturation;
        input integer input_stall_cycles;
        input integer output_stall_cycles;
        begin
            if (input_stall_cycles > 0) begin
                repeat (input_stall_cycles) @(posedge clk);
                input_stalls = input_stalls + input_stall_cycles;
            end
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            down = selected_down;
            down_base_scale32 = selected_base_scale;
            group_scale32 = selected_group_scale;
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
            held_down = grouped_down;
            held_scale = output_scale32;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            for (stall_index = 0; stall_index < output_stall_cycles; stall_index = stall_index + 1) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || grouped_down !== held_down || output_scale32 !== held_scale ||
                    output_channel !== held_channel || output_last !== held_last ||
                    saturation !== held_saturation)
                    failures = failures + 1;
            end
            if (grouped_down !== expected_down || saturation !== expected_saturation ||
                output_scale32 !== selected_group_scale ||
                output_channel !== selected_channel || output_last !== selected_last) begin
                $display("DOWN_OUTPUT_MISMATCH channel=%0d down=%02x expected=%02x sat=%0d expected_sat=%0d", selected_channel, grouped_down, expected_down, saturation, expected_saturation);
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
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            down = 8'sd1;
            down_base_scale32 = 32'h00008000;
            group_scale32 = 32'h00008000;
            channel = 10'd7;
            last = 1'b0;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            if (!out_valid) failures = failures + 1;
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
            down = 8'sd1;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            rst_n = 1'b0;
            #1;
            if (out_valid) failures = failures + 1;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) failures = failures + 1;
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        directed_count = 0;
        output_count = 0;
        saturation_count = 0;
        input_stalls = 0;
        output_stalls = 0;
        cycle_count = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();

        $readmemh("verification/generated/ace2_layer16_down_proj_output_grouped_scale32/down_source_s8.hex", down_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_output_grouped_scale32/down_base_scale32.hex", base_scale_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_output_grouped_scale32/down_group_scale32.hex", group_scale_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_output_grouped_scale32/expected_down_s8.hex", expected_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_output_grouped_scale32/expected_saturation.hex", saturation_mem);

        apply_reset();
        send_and_check(8'sd1, 32'h00ff8000, 32'h00008000, 10'd0, 1'b0, 8'h00, 1'b0, 0, 1);
        directed_count = directed_count + 1;
        send_and_check(8'sd3, 32'h00ff8000, 32'h00008000, 10'd1, 1'b0, 8'h02, 1'b0, 1, 0);
        directed_count = directed_count + 1;
        send_and_check(-8'sd1, 32'h00ff8000, 32'h00008000, 10'd2, 1'b0, 8'h00, 1'b0, 0, 2);
        directed_count = directed_count + 1;
        send_and_check(-8'sd3, 32'h00ff8000, 32'h00008000, 10'd3, 1'b0, 8'hfe, 1'b0, 0, 0);
        directed_count = directed_count + 1;
        send_and_check(8'sd127, 32'h00018000, 32'h00008000, 10'd4, 1'b0, 8'h7f, 1'b1, 0, 1);
        directed_count = directed_count + 1;
        send_and_check(-8'sd128, 32'h00018000, 32'h00008000, 10'd5, 1'b0, 8'h80, 1'b1, 0, 0);
        directed_count = directed_count + 1;
        send_and_check(8'sd7, 32'h01008000, 32'h00008000, 10'd6, 1'b0, 8'h00, 1'b1, 0, 0);
        directed_count = directed_count + 1;
        clear_recovery_probe();
        async_reset_recovery_probe();

        for (index = 0; index < MODEL_HIDDEN; index = index + 1) begin
            send_and_check(
                down_mem[index], base_scale_mem[0],
                group_scale_mem[index / MODEL_GROUP_SIZE], index[9:0],
                index == MODEL_HIDDEN - 1, expected_mem[index],
                saturation_mem[index][0], (index % 17) == 0, index % 4
            );
            output_count = output_count + 1;
            saturation_count = saturation_count + saturation_mem[index][0];
        end

        if (output_count != MODEL_HIDDEN || saturation_count != MODEL_SATURATION_COUNT || xz_count != 0)
            failures = failures + 1;
        if (failures == 0) begin
            $display("ACE2_LAYER16_DOWN_PROJ_OUTPUT_GROUPED_SCALE32_RTL_PASS directed=%0d outputs=%0d groups=%0d saturations=%0d input_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1", directed_count, output_count, MODEL_GROUP_COUNT, saturation_count, input_stalls, output_stalls, cycle_count);
        end else begin
            $display("ACE2_LAYER16_DOWN_PROJ_OUTPUT_GROUPED_SCALE32_RTL_FAIL failures=%0d xz=%0d", failures, xz_count);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
