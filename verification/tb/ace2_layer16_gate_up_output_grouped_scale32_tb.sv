`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_gate_up_output_grouped_scale32_tb;
    `include "ace2_layer16_gate_up_output_grouped_scale32/ace2_layer16_gate_up_output_grouped_scale32_constants.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg stream;
    reg signed [31:0] accumulator;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    reg [31:0] group_scale32;
    reg [12:0] channel;
    reg last;
    wire out_valid;
    reg out_ready;
    wire output_stream;
    wire signed [7:0] q;
    wire [31:0] output_scale32;
    wire [12:0] output_channel;
    wire output_last;
    wire saturation;

    reg [31:0] accumulator_mem [0:MODEL_OUTPUTS-1];
    reg [31:0] multiplier_mem [0:MODEL_OUTPUTS-1];
    reg [7:0] shift_mem [0:MODEL_OUTPUTS-1];
    reg [7:0] expected_mem [0:MODEL_OUTPUTS-1];
    reg [7:0] saturation_mem [0:MODEL_OUTPUTS-1];
    reg [31:0] scale_mem [0:MODEL_SCALE_RECORDS-1];

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
    integer stream_index;
    integer channel_index;
    integer scale_index;
    reg signed [7:0] held_q;
    reg held_stream;
    reg [31:0] held_scale;
    reg [12:0] held_channel;
    reg held_last;
    reg held_saturation;

    ace2_layer16_gate_up_output_grouped_scale32_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .stream_i(stream),
        .accumulator_i(accumulator),
        .multiplier_i(multiplier),
        .right_shift_i(right_shift),
        .group_scale32_i(group_scale32),
        .channel_i(channel),
        .last_i(last),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .stream_o(output_stream),
        .q_o(q),
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
            if ((in_ready !== 1'b0) && (in_ready !== 1'b1)) xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1)) xz_count = xz_count + 1;
            if (out_valid && ((output_stream === 1'bx) || (^q === 1'bx) ||
                              (^output_scale32 === 1'bx) ||
                              (^output_channel === 1'bx) ||
                              (output_last === 1'bx) ||
                              (saturation === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            in_valid = 1'b0;
            stream = 1'b0;
            accumulator = 32'sd0;
            multiplier = 32'sd0;
            right_shift = 6'd0;
            group_scale32 = 32'd0;
            channel = 13'd0;
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
                $display("LAYER16_GATE_UP_OUTPUT_RESET_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task produce_only;
        input selected_stream;
        input signed [31:0] selected_accumulator;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input [31:0] selected_scale;
        input [12:0] selected_channel;
        input selected_last;
        begin
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            stream = selected_stream;
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
                $display("LAYER16_GATE_UP_OUTPUT_PRODUCE_TIMEOUT");
                failures = failures + 1;
            end
        end
    endtask

    task send_and_check;
        input selected_stream;
        input signed [31:0] selected_accumulator;
        input signed [31:0] selected_multiplier;
        input [5:0] selected_shift;
        input [31:0] selected_scale;
        input [12:0] selected_channel;
        input selected_last;
        input [7:0] expected_q;
        input expected_saturation;
        input integer output_stall_cycles;
        begin
            produce_only(
                selected_stream,
                selected_accumulator,
                selected_multiplier,
                selected_shift,
                selected_scale,
                selected_channel,
                selected_last
            );
            held_q = q;
            held_stream = output_stream;
            held_scale = output_scale32;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            for (stall_index = 0; stall_index < output_stall_cycles; stall_index = stall_index + 1) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || q !== held_q || output_stream !== held_stream ||
                    output_scale32 !== held_scale || output_channel !== held_channel ||
                    output_last !== held_last || saturation !== held_saturation) begin
                    $display("LAYER16_GATE_UP_OUTPUT_BACKPRESSURE_STABILITY_FAIL stream=%0d channel=%0d", selected_stream, selected_channel);
                    failures = failures + 1;
                end
            end
            if (q !== expected_q || saturation !== expected_saturation ||
                output_stream !== selected_stream || output_scale32 !== selected_scale ||
                output_channel !== selected_channel || output_last !== selected_last) begin
                $display("LAYER16_GATE_UP_OUTPUT_MISMATCH stream=%0d channel=%0d q=%02x expected=%02x sat=%0d expected_sat=%0d scale=%08x expected_scale=%08x last=%0d expected_last=%0d",
                         selected_stream, selected_channel, q, expected_q,
                         saturation, expected_saturation, output_scale32,
                         selected_scale, output_last, selected_last);
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
            produce_only(1'b0, 32'sd3, 32'sd1, 6'd1, 32'h00fb0001, 13'd9, 1'b0);
            held_q = q;
            held_stream = output_stream;
            held_scale = output_scale32;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            @(negedge clk);
            stream = 1'b1;
            accumulator = -32'sd3;
            multiplier = 32'sd1;
            right_shift = 6'd1;
            group_scale32 = 32'h00fb0002;
            channel = 13'd10;
            last = 1'b0;
            in_valid = 1'b1;
            repeat (3) begin
                @(posedge clk);
                input_stalls = input_stalls + 1;
                if (in_ready || !out_valid || q !== held_q ||
                    output_stream !== held_stream || output_scale32 !== held_scale ||
                    output_channel !== held_channel || output_last !== held_last ||
                    saturation !== held_saturation) begin
                    $display("LAYER16_GATE_UP_OUTPUT_HELD_VALID_BACKPRESSURE_FAIL");
                    failures = failures + 1;
                end
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
            in_valid = 1'b0;
            if (!out_valid || q !== 8'hfe || output_stream !== 1'b1 ||
                output_scale32 !== 32'h00fb0002 || output_channel !== 13'd10 ||
                output_last !== 1'b0 || saturation !== 1'b0) begin
                $display("LAYER16_GATE_UP_OUTPUT_HELD_VALID_CAPTURE_FAIL");
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
            produce_only(1'b1, 32'sd3, 32'sd1, 6'd1, 32'h00fb0003, 13'd11, 1'b0);
            @(negedge clk);
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("LAYER16_GATE_UP_OUTPUT_CLEAR_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task async_reset_recovery_probe;
        begin
            produce_only(1'b0, -32'sd3, 32'sd1, 6'd1, 32'h00fb0004, 13'd12, 1'b0);
            @(negedge clk);
            rst_n = 1'b0;
            #1;
            if (out_valid) begin
                $display("LAYER16_GATE_UP_OUTPUT_ASYNC_RESET_DID_NOT_CLEAR_VALID");
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("LAYER16_GATE_UP_OUTPUT_ASYNC_RESET_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
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

        $readmemh("verification/generated/ace2_layer16_gate_up_output_grouped_scale32/accumulator.hex", accumulator_mem);
        $readmemh("verification/generated/ace2_layer16_gate_up_output_grouped_scale32/multiplier.hex", multiplier_mem);
        $readmemh("verification/generated/ace2_layer16_gate_up_output_grouped_scale32/shift.hex", shift_mem);
        $readmemh("verification/generated/ace2_layer16_gate_up_output_grouped_scale32/expected.hex", expected_mem);
        $readmemh("verification/generated/ace2_layer16_gate_up_output_grouped_scale32/saturation.hex", saturation_mem);
        $readmemh("verification/generated/ace2_layer16_gate_up_output_grouped_scale32/scale.hex", scale_mem);

        apply_reset();
        send_and_check(1'b0, 32'sd0, 32'sd1, 6'd0, 32'h00fb0010, 13'd0, 1'b0, 8'h00, 1'b0, 1);
        directed_count = directed_count + 1;
        send_and_check(1'b0, 32'sd1, 32'sd1, 6'd1, 32'h00fb0011, 13'd1, 1'b0, 8'h00, 1'b0, 0);
        directed_count = directed_count + 1;
        send_and_check(1'b1, 32'sd3, 32'sd1, 6'd1, 32'h00fb0012, 13'd2, 1'b0, 8'h02, 1'b0, 2);
        directed_count = directed_count + 1;
        send_and_check(1'b1, -32'sd3, 32'sd1, 6'd1, 32'h00fb0013, 13'd3, 1'b0, 8'hfe, 1'b0, 0);
        directed_count = directed_count + 1;
        send_and_check(1'b0, 32'sd200, 32'sd1, 6'd0, 32'h00fb0014, 13'd4, 1'b0, 8'h7f, 1'b1, 1);
        directed_count = directed_count + 1;
        send_and_check(1'b1, -32'sd200, 32'sd1, 6'd0, 32'h00fb0015, 13'd5, 1'b0, 8'h80, 1'b1, 0);
        directed_count = directed_count + 1;

        held_valid_backpressure_probe();
        clear_recovery_probe();
        async_reset_recovery_probe();

        for (index = 0; index < MODEL_OUTPUTS; index = index + 1) begin
            stream_index = index / MODEL_STREAM_CHANNELS;
            channel_index = index % MODEL_STREAM_CHANNELS;
            scale_index = stream_index * MODEL_GROUPS_PER_STREAM +
                          channel_index / MODEL_OUTPUT_GROUP_SIZE;
            send_and_check(
                stream_index[0],
                accumulator_mem[index],
                multiplier_mem[index],
                shift_mem[index][5:0],
                scale_mem[scale_index],
                channel_index[12:0],
                index == MODEL_OUTPUTS - 1,
                expected_mem[index],
                saturation_mem[index][0],
                index % 4
            );
            output_count = output_count + 1;
            if ((channel_index % MODEL_OUTPUT_GROUP_SIZE) == 0)
                group_count = group_count + 1;
            saturation_count = saturation_count + saturation_mem[index][0];
        end

        if (output_count != MODEL_OUTPUTS ||
            group_count != MODEL_SCALE_RECORDS ||
            saturation_count != MODEL_SATURATION_COUNT || xz_count != 0)
            failures = failures + 1;

        if (failures == 0) begin
            $display("ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RTL_PASS directed=%0d outputs=%0d groups=%0d saturations=%0d input_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1 held_valid=1",
                     directed_count, output_count, group_count, saturation_count,
                     input_stalls, output_stalls, cycle_count);
        end else begin
            $display("ACE2_LAYER16_GATE_UP_OUTPUT_GROUPED_SCALE32_RTL_FAIL failures=%0d xz=%0d", failures, xz_count);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
