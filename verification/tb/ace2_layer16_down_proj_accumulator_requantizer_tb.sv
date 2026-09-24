`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_down_proj_accumulator_requantizer_tb;
    `include "ace2_layer16_down_proj_accumulator_requantizer/ace2_layer16_down_proj_accumulator_requantizer_constants.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg signed [31:0] accumulator;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    reg [31:0] group_scale32;
    reg [9:0] channel;
    reg last;
    wire out_valid;
    reg out_ready;
    wire signed [7:0] down;
    wire [31:0] output_scale32;
    wire [9:0] output_channel;
    wire output_last;
    wire saturation;

    reg [31:0] accumulator_mem [0:MODEL_DOWN_OUTPUTS-1];
    reg [31:0] multiplier_mem [0:MODEL_DOWN_OUTPUTS-1];
    reg [7:0] shift_mem [0:MODEL_DOWN_OUTPUTS-1];
    reg [31:0] scale_mem [0:MODEL_DOWN_OUTPUT_GROUP_COUNT-1];
    reg [7:0] expected_mem [0:MODEL_DOWN_OUTPUTS-1];
    reg [7:0] saturation_mem [0:MODEL_DOWN_OUTPUTS-1];

    integer failures;
    integer xz_count;
    integer directed_count;
    integer output_count;
    integer group_count;
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

    ace2_layer16_down_proj_accumulator_requantizer_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .accumulator_i(accumulator),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .group_scale32_i(group_scale32),
        .channel_i(channel),
        .last_i(last),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .down_o(down),
        .group_scale32_o(output_scale32),
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
            if ((in_ready !== 1'b0) && (in_ready !== 1'b1))
                xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1))
                xz_count = xz_count + 1;
            if (out_valid && ((^down === 1'bx) || (^output_scale32 === 1'bx) ||
                              (^output_channel === 1'bx) ||
                              (output_last === 1'bx) ||
                              (saturation === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            in_valid = 1'b0;
            accumulator = 32'sd0;
            multiplier = 32'sd0;
            right_shift = 6'd0;
            group_scale32 = 32'd0;
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
                $display("LAYER16_DOWN_ACC_RESET_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task produce_only;
        input signed [31:0] selected_accumulator;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input [31:0] selected_scale;
        input [9:0] selected_channel;
        input selected_last;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            accumulator = selected_accumulator;
            multiplier = selected_multiplier;
            right_shift = selected_shift;
            group_scale32 = selected_scale;
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
            if (!out_valid) begin
                $display("LAYER16_DOWN_ACC_PRODUCE_TIMEOUT");
                failures = failures + 1;
            end
        end
    endtask

    task send_and_check;
        input signed [31:0] selected_accumulator;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input [31:0] selected_scale;
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
            produce_only(
                selected_accumulator,
                selected_multiplier,
                selected_shift,
                selected_scale,
                selected_channel,
                selected_last
            );
            held_down = down;
            held_scale = output_scale32;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            for (stall_index = 0; stall_index < output_stall_cycles; stall_index = stall_index + 1) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || down !== held_down || output_scale32 !== held_scale ||
                    output_channel !== held_channel || output_last !== held_last ||
                    saturation !== held_saturation) begin
                    $display("LAYER16_DOWN_ACC_BACKPRESSURE_STABILITY_FAIL channel=%0d", selected_channel);
                    failures = failures + 1;
                end
            end
            if (down !== expected_down || saturation !== expected_saturation ||
                output_scale32 !== selected_scale || output_channel !== selected_channel ||
                output_last !== selected_last) begin
                $display("LAYER16_DOWN_ACC_MISMATCH channel=%0d down=%02x expected=%02x sat=%0d expected_sat=%0d scale=%08x expected_scale=%08x last=%0d expected_last=%0d",
                         selected_channel, down, expected_down, saturation, expected_saturation,
                         output_scale32, selected_scale, output_last, selected_last);
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
            produce_only(32'sd3, 32'sd1, 6'd1, 32'h00fb0001, 10'd7, 1'b0);
            @(negedge clk);
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("LAYER16_DOWN_ACC_CLEAR_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task async_reset_recovery_probe;
        begin
            produce_only(-32'sd3, 32'sd1, 6'd1, 32'h00fb0001, 10'd8, 1'b0);
            @(negedge clk);
            rst_n = 1'b0;
            #1;
            if (out_valid) begin
                $display("LAYER16_DOWN_ACC_ASYNC_RESET_DID_NOT_CLEAR_VALID");
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("LAYER16_DOWN_ACC_ASYNC_RESET_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task idle_xz_injection_probe;
        begin
            @(negedge clk);
            in_valid = 1'b0;
            accumulator = 32'sbx;
            multiplier = 32'sbz;
            right_shift = 6'bx;
            group_scale32 = 32'bz;
            channel = 10'bx;
            last = 1'bx;
            repeat (2) @(posedge clk);
            if (out_valid) begin
                $display("LAYER16_DOWN_ACC_IDLE_XZ_CREATED_OUTPUT");
                failures = failures + 1;
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
        group_count = 0;
        saturation_count = 0;
        input_stalls = 0;
        output_stalls = 0;
        cycle_count = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();

        $readmemh("verification/generated/ace2_layer16_down_proj_accumulator_requantizer/down_accumulator_s32.hex", accumulator_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_accumulator_requantizer/down_multiplier_s32.hex", multiplier_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_accumulator_requantizer/down_shift_u6.hex", shift_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_accumulator_requantizer/down_output_group_scale32.hex", scale_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_accumulator_requantizer/down_expected_s8.hex", expected_mem);
        $readmemh("verification/generated/ace2_layer16_down_proj_accumulator_requantizer/down_expected_saturation.hex", saturation_mem);

        apply_reset();
        send_and_check(32'sd0, 32'sd1, 6'd0, 32'h00fb0001, 10'd0, 1'b0, 8'h00, 1'b0, 0, 1);
        directed_count = directed_count + 1;
        send_and_check(32'sd1, 32'sd1, 6'd1, 32'h00fb0002, 10'd1, 1'b0, 8'h00, 1'b0, 1, 0);
        directed_count = directed_count + 1;
        send_and_check(32'sd3, 32'sd1, 6'd1, 32'h00fb0003, 10'd2, 1'b0, 8'h02, 1'b0, 0, 2);
        directed_count = directed_count + 1;
        send_and_check(-32'sd3, 32'sd1, 6'd1, 32'h00fb0004, 10'd3, 1'b0, 8'hfe, 1'b0, 0, 0);
        directed_count = directed_count + 1;
        send_and_check(32'sd200, 32'sd1, 6'd0, 32'h00fb0005, 10'd4, 1'b0, 8'h7f, 1'b1, 0, 1);
        directed_count = directed_count + 1;
        send_and_check(-32'sd200, 32'sd1, 6'd0, 32'h00fb0006, 10'd5, 1'b0, 8'h80, 1'b1, 0, 0);
        directed_count = directed_count + 1;
        send_and_check(32'sd7, 32'sd1, 6'd0, 32'h01000001, 10'd6, 1'b0, 8'h00, 1'b1, 0, 1);
        directed_count = directed_count + 1;

        clear_recovery_probe();
        async_reset_recovery_probe();
        idle_xz_injection_probe();

        for (index = 0; index < MODEL_DOWN_OUTPUTS; index = index + 1) begin
            send_and_check(
                accumulator_mem[index],
                multiplier_mem[index],
                shift_mem[index][5:0],
                scale_mem[index / MODEL_DOWN_OUTPUT_GROUP_SIZE],
                index[9:0],
                index == MODEL_DOWN_OUTPUTS - 1,
                expected_mem[index],
                saturation_mem[index][0],
                (index % 17) == 0,
                index % 4
            );
            output_count = output_count + 1;
            if ((index % MODEL_DOWN_OUTPUT_GROUP_SIZE) == 0)
                group_count = group_count + 1;
            saturation_count = saturation_count + saturation_mem[index][0];
        end

        if (output_count != MODEL_DOWN_OUTPUTS ||
            group_count != MODEL_DOWN_OUTPUT_GROUP_COUNT ||
            directed_count != 7 || xz_count != 0)
            failures = failures + 1;

        if (failures == 0) begin
            $display("ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_RTL_PASS directed=%0d outputs=%0d groups=%0d saturations=%0d input_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1 idle_xz_injection=1",
                     directed_count, output_count, group_count, saturation_count,
                     input_stalls, output_stalls, cycle_count);
        end else begin
            $display("ACE2_LAYER16_DOWN_PROJ_ACCUMULATOR_REQUANT_RTL_FAIL failures=%0d xz=%0d", failures, xz_count);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
