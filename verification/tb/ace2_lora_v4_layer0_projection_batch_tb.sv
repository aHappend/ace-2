`timescale 1ns/1ps
`default_nettype none

module ace2_lora_v4_layer0_projection_batch_tb;
    localparam integer CASES = 7;
    localparam integer MAC_LANES = 4;
    localparam integer MAX_K = 4864;
    localparam integer MAX_GROUPS = MAX_K / MAC_LANES;
    localparam integer GROUP_INDEX_WIDTH = $clog2(MAX_GROUPS + 1);
    localparam integer TOTAL_INPUT_GROUPS = 2560;
    localparam integer TOTAL_WEIGHT_GROUPS = 3727360;
    localparam integer TOTAL_OUTPUTS = 12672;

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

    reg [31:0] activation_mem [0:TOTAL_INPUT_GROUPS-1];
    reg [15:0] weight_mem [0:TOTAL_WEIGHT_GROUPS-1];
    reg [31:0] multiplier_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] shift_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_mem [0:TOTAL_OUTPUTS-1];
    reg [31:0] expected_acc_mem [0:TOTAL_OUTPUTS-1];
    reg [7:0] expected_saturation_mem [0:TOTAL_OUTPUTS-1];

    integer case_index;
    integer channel;
    integer group_index;
    integer guard;
    integer failures;
    integer groups;
    integer outputs;
    integer input_offset;
    integer weight_offset;
    integer output_offset;

    ace2_w4a8_proj_core #(
        .K_SIZE(MAX_K),
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

    task select_case;
        input integer selected_case;
        begin
            case (selected_case)
                0: begin groups = 224; outputs = 896; input_offset = 0;    weight_offset = 0;       output_offset = 0;     end
                1: begin groups = 224; outputs = 128; input_offset = 224;  weight_offset = 200704;  output_offset = 896;   end
                2: begin groups = 224; outputs = 128; input_offset = 448;  weight_offset = 229376;  output_offset = 1024;  end
                3: begin groups = 224; outputs = 896; input_offset = 672;  weight_offset = 258048;  output_offset = 1152;  end
                4: begin groups = 224; outputs = 4864; input_offset = 896; weight_offset = 458752;  output_offset = 2048;  end
                5: begin groups = 224; outputs = 4864; input_offset = 1120; weight_offset = 1548288; output_offset = 6912; end
                default: begin groups = 1216; outputs = 896; input_offset = 1344; weight_offset = 2637824; output_offset = 11776; end
            endcase
        end
    endtask

    task apply_reset;
        begin
            rst_n = 1'b0;
            start_valid = 1'b0;
            pair_valid = 1'b0;
            act_data = 32'd0;
            weight_data = 16'd0;
            last_group = {GROUP_INDEX_WIDTH{1'b0}};
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
        input integer selected_case;
        input integer selected_channel;
        integer flat_output;
        begin
            select_case(selected_case);
            flat_output = output_offset + selected_channel;
            while (!start_ready) @(posedge clk);
            last_group = GROUP_INDEX_WIDTH'(groups - 1);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            for (group_index = 0; group_index < groups; group_index = group_index + 1) begin
                while (!pair_ready) @(posedge clk);
                act_data = activation_mem[input_offset + group_index];
                weight_data = weight_mem[weight_offset + selected_channel*groups + group_index];
                @(negedge clk);
                pair_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                pair_valid = 1'b0;
            end

            while (!meta_ready) @(posedge clk);
            multiplier = multiplier_mem[flat_output];
            right_shift = shift_mem[flat_output][5:0];
            @(negedge clk);
            meta_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            meta_valid = 1'b0;

            guard = 0;
            while (!out_valid && guard < 8192) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("RTL_PROJ_TIMEOUT operator=%0d channel=%0d", selected_case, selected_channel);
                failures = failures + 1;
            end else begin
                $display("RTL_PROJ_RESULT operator=%0d channel=%0d out_u8=%0d acc=%0d saturation=%0d overflow=%0d",
                         selected_case, selected_channel, out_data, acc,
                         saturation_seen, accumulator_overflow);
                if ((out_data !== expected_mem[flat_output]) ||
                    (acc !== $signed(expected_acc_mem[flat_output])) ||
                    (saturation_seen !== expected_saturation_mem[flat_output][0]) ||
                    accumulator_overflow) begin
                    $display("RTL_PROJ_MISMATCH operator=%0d channel=%0d got=%02x expected=%02x got_acc=%0d expected_acc=%0d got_sat=%0d expected_sat=%0d overflow=%0d",
                             selected_case, selected_channel, out_data,
                             expected_mem[flat_output], acc,
                             $signed(expected_acc_mem[flat_output]), saturation_seen,
                             expected_saturation_mem[flat_output][0], accumulator_overflow);
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
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_activation.hex", activation_mem);
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_weight.hex", weight_mem);
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_multiplier.hex", multiplier_mem);
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_shift.hex", shift_mem);
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_expected.hex", expected_mem);
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_accumulator.hex", expected_acc_mem);
        $readmemh("evidence/verification/lora-v4-layer0-full-w4a8-rtl-v1/attempt-0001/vectors/projection_saturation.hex", expected_saturation_mem);
        failures = 0;
        apply_reset();
        for (case_index = 0; case_index < CASES; case_index = case_index + 1) begin
            select_case(case_index);
            for (channel = 0; channel < outputs; channel = channel + 1)
                run_channel(case_index, channel);
            $display("RTL_PROJ_OPERATOR_PASS operator=%0d channels=%0d groups=%0d", case_index, outputs, groups);
        end
        if (failures != 0)
            $fatal(1, "ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_FAIL failures=%0d", failures);
        $display("ACE2_LORA_V4_LAYER0_PROJECTION_BATCH_PASS operators=%0d channels=%0d", CASES, TOTAL_OUTPUTS);
        $finish;
    end
endmodule

`default_nettype wire
