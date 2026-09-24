`timescale 1ns/1ps
`default_nettype none

module ace2_lora_v4_layer0_qproj_tb;
    localparam integer HIDDEN = 896;
    localparam integer MAC_LANES = 4;
    localparam integer GROUPS = HIDDEN / MAC_LANES;
    localparam integer GROUP_INDEX_WIDTH = $clog2(GROUPS + 1);

    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg pair_valid;
    wire pair_ready;
    reg [31:0] act_data;
    reg [15:0] weight_data;
    reg [GROUP_INDEX_WIDTH-1:0] last_group;
    reg meta_valid;
    wire meta_ready;
    reg signed [31:0] multiplier;
    reg [5:0] right_shift;
    wire out_valid;
    reg out_ready;
    wire [7:0] out_data;
    wire signed [31:0] acc;
    wire accumulator_overflow;
    wire saturation_seen;

    reg [31:0] activation_mem [0:GROUPS-1];
    reg [15:0] weight_mem [0:HIDDEN*GROUPS-1];
    reg [31:0] multiplier_mem [0:HIDDEN-1];
    reg [7:0] shift_mem [0:HIDDEN-1];
    reg [7:0] expected_mem [0:HIDDEN-1];
    reg [31:0] expected_acc_mem [0:HIDDEN-1];
    reg [7:0] expected_saturation_mem [0:HIDDEN-1];

    integer channel;
    integer group_index;
    integer guard;
    integer failures;

    ace2_w4a8_proj_core #(
        .K_SIZE(HIDDEN),
        .MAC_LANES(MAC_LANES)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(1'b0),
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
        .output_zero_point_i(8'sd0),
        .bias_accumulator_i(32'sd0),
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

    task apply_reset;
        begin
            rst_n = 1'b0;
            start_valid = 1'b0;
            pair_valid = 1'b0;
            act_data = 32'd0;
            weight_data = 16'd0;
            last_group = GROUP_INDEX_WIDTH'(GROUPS - 1);
            meta_valid = 1'b0;
            multiplier = 32'sd0;
            right_shift = 6'd0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_channel;
        input integer selected_channel;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
                while (!pair_ready) @(posedge clk);
                act_data = activation_mem[group_index];
                weight_data = weight_mem[selected_channel*GROUPS + group_index];
                @(negedge clk);
                pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                pair_valid = 1'b0;
            end

            while (!meta_ready) @(posedge clk);
            multiplier = multiplier_mem[selected_channel];
            right_shift = shift_mem[selected_channel][5:0];
            @(negedge clk);
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;

            guard = 0;
            while (!out_valid && guard < 4096) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("RTL_TIMEOUT channel=%0d", selected_channel);
                failures = failures + 1;
            end else begin
                $display("RTL_RESULT channel=%0d out_u8=%0d acc=%0d saturation=%0d overflow=%0d",
                         selected_channel, out_data, acc, saturation_seen, accumulator_overflow);
                if ((out_data !== expected_mem[selected_channel]) ||
                    (acc !== $signed(expected_acc_mem[selected_channel])) ||
                    (saturation_seen !== expected_saturation_mem[selected_channel][0]) ||
                    accumulator_overflow) begin
                    $display("RTL_MISMATCH channel=%0d got_out=%02x expected_out=%02x got_acc=%0d expected_acc=%0d got_sat=%0d expected_sat=%0d overflow=%0d",
                             selected_channel, out_data, expected_mem[selected_channel], acc,
                             $signed(expected_acc_mem[selected_channel]), saturation_seen,
                             expected_saturation_mem[selected_channel][0], accumulator_overflow);
                    failures = failures + 1;
                end
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end
    endtask

    initial begin
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/activation.hex", activation_mem);
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/qweight.hex", weight_mem);
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/multiplier.hex", multiplier_mem);
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/right_shift.hex", shift_mem);
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/expected_output.hex", expected_mem);
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/expected_accumulator.hex", expected_acc_mem);
        $readmemh("evidence/verification/lora-v4-layer0-qproj-w4a8-rtl-v1/attempt-0002/vectors/expected_saturation.hex", expected_saturation_mem);

        failures = 0;
        apply_reset();
        for (channel = 0; channel < HIDDEN; channel = channel + 1) begin
            run_channel(channel);
        end
        if (failures != 0) begin
            $display("ACE2_LORA_V4_LAYER0_QPROJ_RTL_FAIL failures=%0d", failures);
            $fatal(1, "ACE2_LORA_V4_LAYER0_QPROJ_RTL_FAIL");
        end
        $display("ACE2_LORA_V4_LAYER0_QPROJ_RTL_PASS channels=%0d groups_per_channel=%0d", HIDDEN, GROUPS);
        $finish;
    end
endmodule

`default_nettype wire
