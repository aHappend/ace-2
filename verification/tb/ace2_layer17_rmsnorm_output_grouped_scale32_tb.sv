`timescale 1ns/1ps
`default_nettype none

module ace2_layer17_rmsnorm_output_grouped_scale32_tb;
    `include "ace2_layer17_rmsnorm_output_grouped_scale32/ace2_layer17_rmsnorm_output_grouped_scale32_constants.svh"

    localparam integer ACT_WIDTH = 8;
    localparam integer GAIN_WIDTH = 16;

    reg clk;
    reg rst_n;
    reg clear;
    reg start_valid;
    wire start_ready;
    reg in_valid;
    wire in_ready;
    reg [MODEL_LANES*ACT_WIDTH-1:0] in_data;
    reg scale_valid;
    wire scale_ready;
    reg [MODEL_LANES*ACT_WIDTH-1:0] scale_act_data;
    reg [MODEL_LANES*GAIN_WIDTH-1:0] gain_data;
    reg [31:0] group_scale32;
    reg [3:0] group_id;
    reg last;
    wire out_valid;
    reg out_ready;
    wire [MODEL_LANES*ACT_WIDTH-1:0] out_data;
    wire [31:0] output_group_scale32;
    wire [3:0] output_group_id;
    wire output_last;
    wire done_valid;
    reg done_ready;
    wire [47:0] sumsq;
    wire [31:0] inv_rms_q30;
    wire saturation_seen;

    reg [7:0] source_mem [0:MODEL_HIDDEN-1];
    reg [15:0] gain_mem [0:MODEL_HIDDEN-1];
    reg [31:0] scale_mem [0:MODEL_RMS_GROUP_COUNT-1];
    reg [7:0] expected_mem [0:MODEL_HIDDEN-1];

    integer failures;
    integer xz_count;
    integer cycle_count;
    integer input_stalls;
    integer output_stalls;
    integer beat_index;
    integer lane_index;
    integer guard;
    integer stall_index;
    reg [MODEL_LANES*ACT_WIDTH-1:0] held_data;
    reg [31:0] held_scale;
    reg [3:0] held_group;
    reg held_last;

    function automatic [MODEL_LANES*ACT_WIDTH-1:0] packed_source;
        input integer selected_beat;
        integer lane;
        begin
            packed_source = {MODEL_LANES*ACT_WIDTH{1'b0}};
            for (lane = 0; lane < MODEL_LANES; lane = lane + 1)
                packed_source[lane*ACT_WIDTH +: ACT_WIDTH] =
                    source_mem[selected_beat*MODEL_LANES + lane];
        end
    endfunction

    function automatic [MODEL_LANES*GAIN_WIDTH-1:0] packed_gain;
        input integer selected_beat;
        integer lane;
        begin
            packed_gain = {MODEL_LANES*GAIN_WIDTH{1'b0}};
            for (lane = 0; lane < MODEL_LANES; lane = lane + 1)
                packed_gain[lane*GAIN_WIDTH +: GAIN_WIDTH] =
                    gain_mem[selected_beat*MODEL_LANES + lane];
        end
    endfunction

    function automatic [MODEL_LANES*ACT_WIDTH-1:0] packed_expected;
        input integer selected_beat;
        integer lane;
        begin
            packed_expected = {MODEL_LANES*ACT_WIDTH{1'b0}};
            for (lane = 0; lane < MODEL_LANES; lane = lane + 1)
                packed_expected[lane*ACT_WIDTH +: ACT_WIDTH] =
                    expected_mem[selected_beat*MODEL_LANES + lane];
        end
    endfunction

    ace2_layer17_rmsnorm_output_grouped_scale32_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .start_valid_i(start_valid),
        .start_ready_o(start_ready),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .in_data_i(in_data),
        .scale_valid_i(scale_valid),
        .scale_ready_o(scale_ready),
        .scale_act_data_i(scale_act_data),
        .gain_data_i(gain_data),
        .group_scale32_i(group_scale32),
        .group_id_i(group_id),
        .last_i(last),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .group_scale32_o(output_group_scale32),
        .group_id_o(output_group_id),
        .last_o(output_last),
        .done_valid_o(done_valid),
        .done_ready_i(done_ready),
        .sumsq_o(sumsq),
        .inv_rms_q30_o(inv_rms_q30),
        .saturation_seen_o(saturation_seen)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (rst_n) begin
            if ((start_ready !== 1'b0) && (start_ready !== 1'b1))
                xz_count = xz_count + 1;
            if ((in_ready !== 1'b0) && (in_ready !== 1'b1))
                xz_count = xz_count + 1;
            if ((scale_ready !== 1'b0) && (scale_ready !== 1'b1))
                xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1))
                xz_count = xz_count + 1;
            if ((done_valid !== 1'b0) && (done_valid !== 1'b1))
                xz_count = xz_count + 1;
            if (out_valid && ((^out_data === 1'bx) ||
                              (^output_group_scale32 === 1'bx) ||
                              (^output_group_id === 1'bx) ||
                              (output_last === 1'bx)))
                xz_count = xz_count + 1;
            if (done_valid && ((^sumsq === 1'bx) ||
                               (^inv_rms_q30 === 1'bx) ||
                               (saturation_seen === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            start_valid = 1'b0;
            in_valid = 1'b0;
            in_data = {MODEL_LANES*ACT_WIDTH{1'b0}};
            scale_valid = 1'b0;
            scale_act_data = {MODEL_LANES*ACT_WIDTH{1'b0}};
            gain_data = {MODEL_LANES*GAIN_WIDTH{1'b0}};
            group_scale32 = 32'd0;
            group_id = 4'd0;
            last = 1'b0;
            out_ready = 1'b0;
            done_ready = 1'b0;
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
            if (!start_ready || out_valid || done_valid) begin
                $display("LAYER17_RMSNORM_OUTPUT_RESET_RECOVERY_FAIL start=%b out=%b done=%b",
                         start_ready, out_valid, done_valid);
                failures = failures + 1;
            end
        end
    endtask

    task start_transaction;
        begin
            while (!start_ready) @(posedge clk);
            @(negedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task send_collect_beat;
        input integer selected_beat;
        begin
            while (!in_ready) begin
                input_stalls = input_stalls + 1;
                @(posedge clk);
            end
            @(negedge clk);
            in_data = packed_source(selected_beat);
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    task reset_mid_collect_probe;
        begin
            start_transaction();
            send_collect_beat(0);
            @(negedge clk);
            rst_n = 1'b0;
            #1;
            if (out_valid || done_valid) begin
                $display("LAYER17_RMSNORM_OUTPUT_ASYNC_RESET_DID_NOT_CLEAR");
                failures = failures + 1;
            end
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!start_ready || out_valid || done_valid) begin
                $display("LAYER17_RMSNORM_OUTPUT_ASYNC_RESET_RECOVERY_FAIL");
                failures = failures + 1;
            end
        end
    endtask

    task clear_mid_collect_probe;
        begin
            start_transaction();
            send_collect_beat(0);
            @(negedge clk);
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            repeat (2) @(posedge clk);
            if (!start_ready || out_valid || done_valid) begin
                $display("LAYER17_RMSNORM_OUTPUT_CLEAR_RECOVERY_FAIL");
                failures = failures + 1;
            end
        end
    endtask

    task send_scale_and_check;
        input integer selected_beat;
        integer expected_group;
        integer selected_stalls;
        begin
            expected_group = selected_beat / MODEL_BEATS_PER_GROUP;
            while (!scale_ready) @(posedge clk);
            @(negedge clk);
            scale_act_data = packed_source(selected_beat);
            gain_data = packed_gain(selected_beat);
            group_scale32 = scale_mem[expected_group];
            group_id = expected_group[3:0];
            last = selected_beat == MODEL_BEATS - 1;
            scale_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            scale_valid = 1'b0;

            guard = 0;
            while (!out_valid && guard < 100000) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("LAYER17_RMSNORM_OUTPUT_TIMEOUT beat=%0d", selected_beat);
                failures = failures + 1;
            end
            if (out_data !== packed_expected(selected_beat) ||
                output_group_scale32 !== scale_mem[expected_group] ||
                output_group_id !== expected_group[3:0] ||
                output_last !== (selected_beat == MODEL_BEATS - 1)) begin
                $display("LAYER17_RMSNORM_OUTPUT_MISMATCH beat=%0d group=%0d got_scale=%08x expected_scale=%08x got_id=%0d",
                         selected_beat, expected_group, output_group_scale32,
                         scale_mem[expected_group], output_group_id);
                failures = failures + 1;
            end

            held_data = out_data;
            held_scale = output_group_scale32;
            held_group = output_group_id;
            held_last = output_last;
            selected_stalls = selected_beat % 4;
            for (stall_index = 0; stall_index < selected_stalls; stall_index = stall_index + 1) begin
                @(posedge clk);
                output_stalls = output_stalls + 1;
                if (!out_valid || out_data !== held_data ||
                    output_group_scale32 !== held_scale ||
                    output_group_id !== held_group || output_last !== held_last) begin
                    $display("LAYER17_RMSNORM_OUTPUT_BACKPRESSURE_STABILITY_FAIL beat=%0d",
                             selected_beat);
                    failures = failures + 1;
                end
            end
            @(negedge clk);
            out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            out_ready = 1'b0;
        end
    endtask

    task run_model_case;
        begin
            start_transaction();
            for (beat_index = 0; beat_index < MODEL_BEATS; beat_index = beat_index + 1)
                send_collect_beat(beat_index);

            for (beat_index = 0; beat_index < MODEL_BEATS; beat_index = beat_index + 1)
                send_scale_and_check(beat_index);

            guard = 0;
            while (!done_valid && guard < 100000) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!done_valid) begin
                $display("LAYER17_RMSNORM_OUTPUT_DONE_TIMEOUT");
                failures = failures + 1;
            end
            if (sumsq !== MODEL_EXPECTED_SUMSQ ||
                inv_rms_q30 !== MODEL_EXPECTED_INV_RMS_Q30 ||
                saturation_seen !== MODEL_EXPECTED_SATURATION) begin
                $display("LAYER17_RMSNORM_OUTPUT_DONE_MISMATCH sumsq=%0d expected=%0d inv=%0d expected_inv=%0d sat=%0d expected_sat=%0d",
                         sumsq, MODEL_EXPECTED_SUMSQ, inv_rms_q30,
                         MODEL_EXPECTED_INV_RMS_Q30, saturation_seen,
                         MODEL_EXPECTED_SATURATION);
                failures = failures + 1;
            end
            @(negedge clk);
            done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        cycle_count = 0;
        input_stalls = 0;
        output_stalls = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();

        $readmemh("verification/generated/ace2_layer17_rmsnorm_output_grouped_scale32/sum_s8.hex", source_mem);
        $readmemh("verification/generated/ace2_layer17_rmsnorm_output_grouped_scale32/rms_gain_s16_q8.hex", gain_mem);
        $readmemh("verification/generated/ace2_layer17_rmsnorm_output_grouped_scale32/rms_output_group_scale32.hex", scale_mem);
        $readmemh("verification/generated/ace2_layer17_rmsnorm_output_grouped_scale32/expected_rms_s8.hex", expected_mem);

        apply_reset();
        reset_mid_collect_probe();
        clear_mid_collect_probe();
        run_model_case();

        if (xz_count != 0)
            failures = failures + 1;
        if (failures == 0) begin
            $display("ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_PASS samples=%0d beats=%0d groups=%0d input_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1 backpressure_stable=1",
                     MODEL_HIDDEN, MODEL_BEATS, MODEL_RMS_GROUP_COUNT,
                     input_stalls, output_stalls, cycle_count);
        end else begin
            $display("ACE2_LAYER17_RMSNORM_OUTPUT_GROUPED_SCALE32_RTL_FAIL failures=%0d xz=%0d",
                     failures, xz_count);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
