`timescale 1ns/1ps
`default_nettype none

module ace2_layer23_v_rank1_integer_correction_tb;
    `include "layer23_v_rank1_integer_correction_vectors.svh"

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;
    logic enable_i = 1'b0;
    logic cfg_valid_i = 1'b0;
    logic cfg_ready_o;
    logic [2:0] cfg_kind_i = '0;
    logic [9:0] cfg_index_i = '0;
    logic signed [31:0] cfg_data_i = '0;
    logic cfg_error_o;
    logic config_complete_o;
    logic start_valid_i = 1'b0;
    logic start_ready_o;
    logic input_valid_i = 1'b0;
    logic input_ready_o;
    logic signed [7:0] input_s8_i = '0;
    logic baseline_valid_i = 1'b0;
    logic baseline_ready_o;
    logic signed [7:0] baseline_v_s8_i = '0;
    logic output_valid_o;
    logic output_ready_i = 1'b0;
    logic [6:0] output_channel_o;
    logic output_last_o;
    logic signed [31:0] rank_accumulator_s32_o;
    logic signed [31:0] rank_rounded_s32_o;
    logic signed [7:0] rank_intermediate_s8_o;
    logic rank_valid_o;
    logic signed [31:0] correction_accumulator_s32_o;
    logic signed [31:0] correction_rounded_s32_o;
    logic signed [7:0] correction_s8_o;
    logic signed [7:0] corrected_v_s8_o;
    logic input_saturation_o;
    logic rank_saturation_o;
    logic correction_saturation_o;
    logic add_saturation_o;
    logic descriptor_error_o;
    logic numeric_overflow_o;

    logic signed [7:0] frozen_input_to_rank [0:ACE2_R1_INPUT_WIDTH-1];
    logic signed [7:0] frozen_rank_to_channel [0:ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [31:0] frozen_second_multiplier [0:ACE2_R1_OUTPUT_WIDTH-1];
    logic [5:0] frozen_second_shift [0:ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [7:0] frozen_input_payload [0:ACE2_R1_POSITION_COUNT*ACE2_R1_INPUT_WIDTH-1];
    logic signed [7:0] frozen_baseline_v [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [31:0] expected_rank_accumulator [0:ACE2_R1_POSITION_COUNT-1];
    logic signed [31:0] expected_rank_rounded [0:ACE2_R1_POSITION_COUNT-1];
    logic signed [7:0] expected_rank_s8 [0:ACE2_R1_POSITION_COUNT-1];
    logic signed [31:0] expected_correction_accumulator [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [31:0] expected_correction_rounded [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [7:0] expected_correction_s8 [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];
    logic signed [7:0] expected_corrected_s8 [0:ACE2_R1_POSITION_COUNT*ACE2_R1_OUTPUT_WIDTH-1];

    integer cfg_i;
    integer pos_i;
    integer lane_i;
    integer channel_i;
    integer flat_i;
    integer frozen_positions_checked = 0;

    always #5 clk_i = ~clk_i;

    ace2_layer23_v_rank1_integer_correction_core dut (
        .clk_i,
        .rst_ni,
        .clear_i,
        .enable_i,
        .cfg_valid_i,
        .cfg_ready_o,
        .cfg_kind_i,
        .cfg_index_i,
        .cfg_data_i,
        .cfg_error_o,
        .config_complete_o,
        .start_valid_i,
        .start_ready_o,
        .input_valid_i,
        .input_ready_o,
        .input_s8_i,
        .baseline_valid_i,
        .baseline_ready_o,
        .baseline_v_s8_i,
        .output_valid_o,
        .output_ready_i,
        .output_channel_o,
        .output_last_o,
        .rank_accumulator_s32_o,
        .rank_rounded_s32_o,
        .rank_intermediate_s8_o,
        .rank_valid_o,
        .correction_accumulator_s32_o,
        .correction_rounded_s32_o,
        .correction_s8_o,
        .corrected_v_s8_o,
        .input_saturation_o,
        .rank_saturation_o,
        .correction_saturation_o,
        .add_saturation_o,
        .descriptor_error_o,
        .numeric_overflow_o
    );

    initial begin
        $readmemh(`ACE2_R1_INPUT_TO_RANK_HEX, frozen_input_to_rank);
        $readmemh(`ACE2_R1_RANK_TO_CHANNEL_HEX, frozen_rank_to_channel);
        $readmemh(`ACE2_R1_SECOND_MULTIPLIER_HEX, frozen_second_multiplier);
        $readmemh(`ACE2_R1_SECOND_SHIFT_HEX, frozen_second_shift);
        $readmemh(`ACE2_R1_INPUT_PAYLOAD_HEX, frozen_input_payload);
        $readmemh(`ACE2_R1_BASELINE_V_HEX, frozen_baseline_v);
        $readmemh(`ACE2_R1_EXPECTED_RANK_ACC_HEX, expected_rank_accumulator);
        $readmemh(`ACE2_R1_EXPECTED_RANK_ROUNDED_HEX, expected_rank_rounded);
        $readmemh(`ACE2_R1_EXPECTED_RANK_S8_HEX, expected_rank_s8);
        $readmemh(`ACE2_R1_EXPECTED_CORR_ACC_HEX, expected_correction_accumulator);
        $readmemh(`ACE2_R1_EXPECTED_CORR_ROUNDED_HEX, expected_correction_rounded);
        $readmemh(`ACE2_R1_EXPECTED_CORR_S8_HEX, expected_correction_s8);
        $readmemh(`ACE2_R1_EXPECTED_CORRECTED_S8_HEX, expected_corrected_s8);
    end

    function automatic logic signed [31:0] sx8(input logic signed [7:0] value);
        begin
            sx8 = {{24{value[7]}}, value};
        end
    endfunction

    function automatic logic signed [31:0] ux6(input logic [5:0] value);
        begin
            ux6 = {26'd0, value};
        end
    endfunction

    task automatic reset_core;
        begin
            rst_ni = 1'b0;
            clear_i = 1'b0;
            enable_i = 1'b0;
            cfg_valid_i = 1'b0;
            start_valid_i = 1'b0;
            input_valid_i = 1'b0;
            baseline_valid_i = 1'b0;
            output_ready_i = 1'b0;
            input_s8_i = '0;
            baseline_v_s8_i = '0;
            repeat (3) @(posedge clk_i);
            rst_ni = 1'b1;
            @(posedge clk_i);
            #1;
            if (!cfg_ready_o || config_complete_o || cfg_error_o)
                $fatal(1, "reset did not expose a clean configuration boundary");
        end
    endtask

    task automatic cfg_write(
        input logic [2:0] kind,
        input integer index,
        input logic signed [31:0] data
    );
        begin
            @(negedge clk_i);
            cfg_kind_i = kind;
            cfg_index_i = index[9:0];
            cfg_data_i = data;
            cfg_valid_i = 1'b1;
            if (!cfg_ready_o)
                $fatal(1, "configuration interface was not ready");
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            cfg_valid_i = 1'b0;
            cfg_data_i = '0;
        end
    endtask

    task automatic configure_synthetic(
        input logic signed [7:0] input_to_rank0,
        input logic signed [31:0] first_multiplier,
        input logic [5:0] first_shift,
        input logic signed [7:0] rank_to_channel0,
        input logic signed [31:0] second_multiplier0,
        input logic [5:0] second_shift0
    );
        begin
            for (cfg_i = 0; cfg_i < ACE2_R1_INPUT_WIDTH; cfg_i = cfg_i + 1)
                cfg_write(
                    ACE2_R1_CFG_INPUT_TO_RANK,
                    cfg_i,
                    cfg_i == 0 ? sx8(input_to_rank0) : 32'sd0
                );
            for (cfg_i = 0; cfg_i < ACE2_R1_OUTPUT_WIDTH; cfg_i = cfg_i + 1) begin
                cfg_write(
                    ACE2_R1_CFG_RANK_TO_CHANNEL,
                    cfg_i,
                    cfg_i == 0 ? sx8(rank_to_channel0) : 32'sd0
                );
                cfg_write(
                    ACE2_R1_CFG_SECOND_MULTIPLIER,
                    cfg_i,
                    cfg_i == 0 ? second_multiplier0 : 32'sd0
                );
                cfg_write(
                    ACE2_R1_CFG_SECOND_SHIFT,
                    cfg_i,
                    cfg_i == 0 ? ux6(second_shift0) : 32'sd0
                );
            end
            cfg_write(ACE2_R1_CFG_FIRST_MULTIPLIER, 0, first_multiplier);
            cfg_write(ACE2_R1_CFG_FIRST_SHIFT, 0, ux6(first_shift));
            if (!config_complete_o || cfg_error_o)
                $fatal(1, "synthetic configuration did not complete");
        end
    endtask

    task automatic configure_frozen;
        begin
            for (cfg_i = 0; cfg_i < ACE2_R1_INPUT_WIDTH; cfg_i = cfg_i + 1)
                cfg_write(ACE2_R1_CFG_INPUT_TO_RANK, cfg_i, sx8(frozen_input_to_rank[cfg_i]));
            for (cfg_i = 0; cfg_i < ACE2_R1_OUTPUT_WIDTH; cfg_i = cfg_i + 1) begin
                cfg_write(ACE2_R1_CFG_RANK_TO_CHANNEL, cfg_i, sx8(frozen_rank_to_channel[cfg_i]));
                cfg_write(ACE2_R1_CFG_SECOND_MULTIPLIER, cfg_i, frozen_second_multiplier[cfg_i]);
                cfg_write(ACE2_R1_CFG_SECOND_SHIFT, cfg_i, ux6(frozen_second_shift[cfg_i]));
            end
            cfg_write(ACE2_R1_CFG_FIRST_MULTIPLIER, 0, ACE2_R1_FIRST_MULTIPLIER_S32);
            cfg_write(ACE2_R1_CFG_FIRST_SHIFT, 0, ACE2_R1_FIRST_SHIFT_S32);
            if (!config_complete_o || cfg_error_o)
                $fatal(1, "frozen configuration did not complete");
        end
    endtask

    task automatic start_transaction(input [255:0] name);
        begin
            enable_i = 1'b1;
            @(negedge clk_i);
            if (!start_ready_o)
                $fatal(1, "%0s: start interface was not ready", name);
            start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            start_valid_i = 1'b0;
        end
    endtask

    task automatic send_input_lane(input logic signed [7:0] value);
        begin
            if (!input_ready_o)
                $fatal(1, "input stream unexpectedly stalled");
            input_s8_i = value;
            input_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
        end
    endtask

    task automatic send_output_baseline(
        input logic signed [7:0] baseline,
        input bit stall_once
    );
        begin
            if (!baseline_ready_o)
                $fatal(1, "baseline stream unexpectedly stalled");
            baseline_v_s8_i = baseline;
            baseline_valid_i = 1'b1;
            output_ready_i = 1'b0;
            @(posedge clk_i);
            #1;
            baseline_valid_i = 1'b0;
            if (!output_valid_o)
                $fatal(1, "output did not become valid");
            if (stall_once) begin
                @(posedge clk_i);
                #1;
                if (!output_valid_o)
                    $fatal(1, "output did not hold under backpressure");
            end
        end
    endtask

    task automatic accept_output;
        begin
            @(negedge clk_i);
            output_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            output_ready_i = 1'b0;
        end
    endtask

    task automatic run_synthetic_case(
        input [255:0] name,
        input logic signed [7:0] input0,
        input logic signed [7:0] input_to_rank0,
        input logic signed [31:0] first_multiplier,
        input logic [5:0] first_shift,
        input logic signed [7:0] rank_to_channel0,
        input logic signed [31:0] second_multiplier0,
        input logic [5:0] second_shift0,
        input logic signed [7:0] baseline0,
        input logic signed [31:0] expected_rank_accumulator,
        input logic signed [31:0] expected_rank_rounded,
        input logic signed [7:0] expected_rank,
        input logic signed [31:0] expected_correction_accumulator,
        input logic signed [31:0] expected_correction_rounded,
        input logic signed [7:0] expected_correction,
        input logic signed [7:0] expected_corrected,
        input logic expected_rank_saturation,
        input logic expected_correction_saturation,
        input logic expected_add_saturation
    );
        begin
            reset_core();
            enable_i = 1'b1;
            #1;
            if (start_ready_o)
                $fatal(1, "%0s: unconfigured core accepted start", name);
            enable_i = 1'b0;
            cfg_write(ACE2_R1_CFG_INPUT_TO_RANK, ACE2_R1_INPUT_WIDTH, 32'sd1);
            if (!cfg_error_o)
                $fatal(1, "%0s: invalid config index did not raise cfg_error", name);
            clear_i = 1'b1;
            @(posedge clk_i);
            #1;
            clear_i = 1'b0;
            configure_synthetic(
                input_to_rank0,
                first_multiplier,
                first_shift,
                rank_to_channel0,
                second_multiplier0,
                second_shift0
            );
            start_transaction(name);
            send_input_lane(input0);
            for (lane_i = 1; lane_i < ACE2_R1_INPUT_WIDTH; lane_i = lane_i + 1)
                send_input_lane(8'sd0);
            input_valid_i = 1'b0;
            if (!rank_valid_o ||
                rank_accumulator_s32_o !== expected_rank_accumulator ||
                rank_rounded_s32_o !== expected_rank_rounded ||
                rank_intermediate_s8_o !== expected_rank ||
                rank_saturation_o !== expected_rank_saturation ||
                descriptor_error_o ||
                numeric_overflow_o)
                $fatal(1, "%0s: rank output mismatch", name);

            send_output_baseline(baseline0, 1'b1);
            if (output_channel_o !== 7'd0 ||
                correction_accumulator_s32_o !== expected_correction_accumulator ||
                correction_rounded_s32_o !== expected_correction_rounded ||
                correction_s8_o !== expected_correction ||
                corrected_v_s8_o !== expected_corrected ||
                correction_saturation_o !== expected_correction_saturation ||
                add_saturation_o !== expected_add_saturation ||
                output_last_o ||
                descriptor_error_o ||
                numeric_overflow_o)
                $fatal(1, "%0s: channel-0 output mismatch", name);
            accept_output();
            for (channel_i = 1; channel_i < ACE2_R1_OUTPUT_WIDTH; channel_i = channel_i + 1) begin
                send_output_baseline(8'sd0, 1'b0);
                if (output_channel_o !== channel_i[6:0] ||
                    correction_accumulator_s32_o !== 32'sd0 ||
                    correction_rounded_s32_o !== 32'sd0 ||
                    correction_s8_o !== 8'sd0 ||
                    corrected_v_s8_o !== 8'sd0 ||
                    correction_saturation_o ||
                    add_saturation_o ||
                    output_last_o !== (channel_i == ACE2_R1_OUTPUT_WIDTH - 1))
                    $fatal(1, "%0s: zero-tail output mismatch", name);
                accept_output();
            end
            if (!start_ready_o || rank_valid_o)
                $fatal(1, "%0s: core did not return to idle", name);
        end
    endtask

    task automatic run_frozen_position(input integer position);
        begin
            start_transaction("frozen_position");
            for (lane_i = 0; lane_i < ACE2_R1_INPUT_WIDTH; lane_i = lane_i + 1) begin
                flat_i = position * ACE2_R1_INPUT_WIDTH + lane_i;
                send_input_lane(frozen_input_payload[flat_i]);
            end
            input_valid_i = 1'b0;
            if (!rank_valid_o ||
                rank_accumulator_s32_o !== expected_rank_accumulator[position] ||
                rank_rounded_s32_o !== expected_rank_rounded[position] ||
                rank_intermediate_s8_o !== expected_rank_s8[position] ||
                rank_saturation_o ||
                descriptor_error_o ||
                numeric_overflow_o)
                $fatal(1, "frozen position %0d rank mismatch", position);

            for (channel_i = 0; channel_i < ACE2_R1_OUTPUT_WIDTH; channel_i = channel_i + 1) begin
                flat_i = position * ACE2_R1_OUTPUT_WIDTH + channel_i;
                send_output_baseline(frozen_baseline_v[flat_i], (channel_i[2:0] == 3'd3));
                if (output_channel_o !== channel_i[6:0] ||
                    correction_accumulator_s32_o !== expected_correction_accumulator[flat_i] ||
                    correction_rounded_s32_o !== expected_correction_rounded[flat_i] ||
                    correction_s8_o !== expected_correction_s8[flat_i] ||
                    corrected_v_s8_o !== expected_corrected_s8[flat_i] ||
                    correction_saturation_o ||
                    add_saturation_o ||
                    output_last_o !== (channel_i == ACE2_R1_OUTPUT_WIDTH - 1) ||
                    descriptor_error_o ||
                    numeric_overflow_o)
                    $fatal(1, "frozen position %0d channel %0d mismatch", position, channel_i);
                accept_output();
            end
            if (!start_ready_o || rank_valid_o)
                $fatal(1, "frozen position %0d did not return to idle", position);
            frozen_positions_checked = frozen_positions_checked + 1;
        end
    endtask

    initial begin
        `include "layer23_v_rank1_integer_correction_synthetic_cases.svh"

        reset_core();
        configure_frozen();
        enable_i = 1'b1;
        for (pos_i = 0; pos_i < ACE2_R1_POSITION_COUNT; pos_i = pos_i + 1)
            run_frozen_position(pos_i);

        $display(
            "ACE2_LAYER23_V_RANK1_INTEGER_RTL_PASS synthetic_cases=4 frozen_positions=%0d outputs_per_position=%0d",
            frozen_positions_checked,
            ACE2_R1_OUTPUT_WIDTH
        );
        $finish;
    end
endmodule

`default_nettype wire
