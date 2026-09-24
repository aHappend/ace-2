`timescale 1ns/1ps
`default_nettype none

module ace2_layer16_post_attention_rmsnorm_runtime_metadata_tb;
    `include "ace2_layer16_post_attention_rmsnorm_runtime_metadata/ace2_layer16_post_attention_rmsnorm_runtime_metadata_constants.svh"

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg signed [7:0] activation_s8;
    reg [31:0] input_scale32;
    reg signed [15:0] gain_s16;
    reg [31:0] gain_scale32;
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
    wire warning;
    wire signed [11:0] aligned_activation;
    wire signed [15:0] gain_s16_q8;
    wire signed [7:0] common_exponent;
    wire [5:0] rebase_shift;
    wire [47:0] sumsq;
    wire [47:0] mean_square;
    wire [11:0] rms_ceil;
    wire [30:0] inv_rms_q30;

    reg [7:0] input_activation_mem [0:MODEL_SAMPLES-1];
    reg [31:0] input_scale_mem [0:MODEL_SAMPLES-1];
    reg [15:0] gain_mem [0:MODEL_HIDDEN-1];
    reg [31:0] gain_scale_mem [0:MODEL_HIDDEN-1];
    reg [11:0] expected_aligned_mem [0:MODEL_SAMPLES-1];
    reg [15:0] expected_qgain_mem [0:MODEL_SAMPLES-1];
    reg [31:0] expected_scale_mem [0:MODEL_SCALE_RECORDS-1];
    reg [7:0] expected_mem [0:MODEL_SAMPLES-1];
    reg [7:0] expected_saturation_mem [0:MODEL_SAMPLES-1];
    reg [7:0] expected_common_exponent_mem [0:MODEL_POSITIONS-1];
    reg [7:0] expected_rebase_shift_mem [0:MODEL_POSITIONS-1];
    reg [47:0] expected_sumsq_mem [0:MODEL_POSITIONS-1];
    reg [47:0] expected_mean_square_mem [0:MODEL_POSITIONS-1];
    reg [11:0] expected_rms_ceil_mem [0:MODEL_POSITIONS-1];
    reg [30:0] expected_inv_rms_mem [0:MODEL_POSITIONS-1];

    integer failures;
    integer xz_count;
    integer cycle_count;
    integer input_stalls;
    integer output_stalls;
    integer output_count;
    integer frame_index;
    integer channel_index;
    integer sample_index;
    integer scale_index;
    integer guard;
    integer stall_index;

    reg signed [7:0] held_q;
    reg [31:0] held_scale;
    reg [4:0] held_position;
    reg [9:0] held_channel;
    reg held_last;
    reg held_saturation;
    reg held_warning;
    reg signed [11:0] held_aligned;
    reg signed [15:0] held_qgain;
    reg signed [7:0] held_common_exponent;
    reg [5:0] held_rebase_shift;
    reg [47:0] held_sumsq;
    reg [47:0] held_mean_square;
    reg [11:0] held_rms_ceil;
    reg [30:0] held_inv_rms;

    ace2_layer16_post_attention_rmsnorm_runtime_metadata_core dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .activation_s8_i(activation_s8),
        .input_scale32_i(input_scale32),
        .gain_s16_i(gain_s16),
        .gain_scale32_i(gain_scale32),
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
        .saturation_o(saturation),
        .warning_o(warning),
        .aligned_activation_o(aligned_activation),
        .gain_s16_q8_o(gain_s16_q8),
        .common_exponent_o(common_exponent),
        .rebase_shift_o(rebase_shift),
        .sumsq_o(sumsq),
        .mean_square_o(mean_square),
        .rms_ceil_o(rms_ceil),
        .inv_rms_q30_o(inv_rms_q30)
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
            if (out_valid && ((^q === 1'bx) || (^output_scale32 === 1'bx) ||
                              (^output_position === 1'bx) ||
                              (^output_channel === 1'bx) ||
                              (output_last === 1'bx) ||
                              (saturation === 1'bx) ||
                              (warning === 1'bx) ||
                              (^aligned_activation === 1'bx) ||
                              (^gain_s16_q8 === 1'bx) ||
                              (^common_exponent === 1'bx) ||
                              (^rebase_shift === 1'bx) ||
                              (^sumsq === 1'bx) ||
                              (^mean_square === 1'bx) ||
                              (^rms_ceil === 1'bx) ||
                              (^inv_rms_q30 === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task idle_inputs;
        begin
            in_valid = 1'b0;
            activation_s8 = 8'sd0;
            input_scale32 = 32'h00e88000;
            gain_s16 = 16'sd0;
            gain_scale32 = 32'h00e88000;
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
                $display("RUNTIME_RMSNORM_RESET_RECOVERY_FAIL ready=%b valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    task warning_and_clear_probe;
        begin
            @(negedge clk);
            activation_s8 = 8'sd1;
            input_scale32 = 32'hff000001;
            gain_s16 = 16'sd1;
            gain_scale32 = 32'h00e88000;
            position = 5'd3;
            channel = 10'd0;
            last = 1'b0;
            in_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            if (warning !== 1'b1) begin
                $display("RUNTIME_RMSNORM_INVALID_SCALE_WARNING_FAIL warning=%b", warning);
                failures = failures + 1;
            end
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            @(posedge clk);
            if (!in_ready || out_valid || warning) begin
                $display("RUNTIME_RMSNORM_CLEAR_RECOVERY_FAIL ready=%b valid=%b warning=%b", in_ready, out_valid, warning);
                failures = failures + 1;
            end
        end
    endtask

    task send_model_frame;
        input integer selected_frame;
        begin
            for (channel_index = 0; channel_index < MODEL_HIDDEN; channel_index = channel_index + 1) begin
                sample_index = selected_frame * MODEL_HIDDEN + channel_index;
                if ((channel_index != 0) && ((channel_index % 257) == 0)) begin
                    @(negedge clk);
                    in_valid = 1'b0;
                    @(posedge clk);
                end
                while (!in_ready) @(posedge clk);
                @(negedge clk);
                activation_s8 = input_activation_mem[sample_index];
                input_scale32 = input_scale_mem[sample_index];
                gain_s16 = gain_mem[channel_index];
                gain_scale32 = gain_scale_mem[channel_index];
                position = selected_frame[4:0];
                channel = channel_index[9:0];
                last = channel_index == MODEL_HIDDEN - 1;
                in_valid = 1'b1;
                @(posedge clk);
            end
            @(negedge clk);
            activation_s8 = input_activation_mem[selected_frame * MODEL_HIDDEN];
            input_scale32 = input_scale_mem[selected_frame * MODEL_HIDDEN];
            gain_s16 = gain_mem[0];
            gain_scale32 = gain_scale_mem[0];
            position = selected_frame[4:0];
            channel = 10'd0;
            last = 1'b0;
            in_valid = 1'b1;
            repeat (3) begin
                @(posedge clk);
                input_stalls = input_stalls + 1;
                if (in_ready) begin
                    $display("RUNTIME_RMSNORM_INPUT_BACKPRESSURE_FAIL frame=%0d", selected_frame);
                    failures = failures + 1;
                end
            end
            @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    task capture_held_output;
        begin
            held_q = q;
            held_scale = output_scale32;
            held_position = output_position;
            held_channel = output_channel;
            held_last = output_last;
            held_saturation = saturation;
            held_warning = warning;
            held_aligned = aligned_activation;
            held_qgain = gain_s16_q8;
            held_common_exponent = common_exponent;
            held_rebase_shift = rebase_shift;
            held_sumsq = sumsq;
            held_mean_square = mean_square;
            held_rms_ceil = rms_ceil;
            held_inv_rms = inv_rms_q30;
        end
    endtask

    task check_held_output;
        input integer selected_frame;
        input integer selected_channel;
        begin
            if (!out_valid || q !== held_q || output_scale32 !== held_scale ||
                output_position !== held_position || output_channel !== held_channel ||
                output_last !== held_last || saturation !== held_saturation ||
                warning !== held_warning || aligned_activation !== held_aligned ||
                gain_s16_q8 !== held_qgain || common_exponent !== held_common_exponent ||
                rebase_shift !== held_rebase_shift || sumsq !== held_sumsq ||
                mean_square !== held_mean_square || rms_ceil !== held_rms_ceil ||
                inv_rms_q30 !== held_inv_rms) begin
                $display("RUNTIME_RMSNORM_HELD_OUTPUT_STABILITY_FAIL frame=%0d channel=%0d", selected_frame, selected_channel);
                failures = failures + 1;
            end
        end
    endtask

    task receive_model_frame;
        input integer selected_frame;
        begin
            for (channel_index = 0; channel_index < MODEL_HIDDEN; channel_index = channel_index + 1) begin
                sample_index = selected_frame * MODEL_HIDDEN + channel_index;
                scale_index = selected_frame * MODEL_GROUPS_PER_POSITION +
                              channel_index / MODEL_OUTPUT_GROUP_SIZE;
                guard = 0;
                while (!out_valid && guard < 20000) begin
                    guard = guard + 1;
                    @(posedge clk);
                end
                if (!out_valid) begin
                    $display("RUNTIME_RMSNORM_OUTPUT_TIMEOUT frame=%0d channel=%0d", selected_frame, channel_index);
                    failures = failures + 1;
                end
                @(negedge clk);
                if (q !== expected_mem[sample_index] ||
                    output_scale32 !== expected_scale_mem[scale_index] ||
                    output_position !== selected_frame[4:0] ||
                    output_channel !== channel_index[9:0] ||
                    output_last !== (channel_index == MODEL_HIDDEN - 1) ||
                    saturation !== expected_saturation_mem[sample_index][0] ||
                    warning !== 1'b0 ||
                    aligned_activation !== expected_aligned_mem[sample_index] ||
                    gain_s16_q8 !== expected_qgain_mem[sample_index] ||
                    common_exponent !== expected_common_exponent_mem[selected_frame] ||
                    rebase_shift !== expected_rebase_shift_mem[selected_frame][5:0] ||
                    sumsq !== expected_sumsq_mem[selected_frame] ||
                    mean_square !== expected_mean_square_mem[selected_frame] ||
                    rms_ceil !== expected_rms_ceil_mem[selected_frame] ||
                    inv_rms_q30 !== expected_inv_rms_mem[selected_frame]) begin
                    $display("RUNTIME_RMSNORM_MISMATCH frame=%0d channel=%0d q=%02x expected=%02x scale=%08x expected_scale=%08x aligned=%03x expected_aligned=%03x qgain=%04x expected_qgain=%04x common=%02x expected_common=%02x rebase=%0d expected_rebase=%0d sumsq=%012x expected_sumsq=%012x mean=%012x expected_mean=%012x rms=%03x expected_rms=%03x inv=%08x expected_inv=%08x warning=%b sat=%b expected_sat=%b",
                             selected_frame, channel_index, q, expected_mem[sample_index],
                             output_scale32, expected_scale_mem[scale_index],
                             aligned_activation, expected_aligned_mem[sample_index],
                             gain_s16_q8, expected_qgain_mem[sample_index],
                             common_exponent, expected_common_exponent_mem[selected_frame],
                             rebase_shift, expected_rebase_shift_mem[selected_frame][5:0],
                             sumsq, expected_sumsq_mem[selected_frame],
                             mean_square, expected_mean_square_mem[selected_frame],
                             rms_ceil, expected_rms_ceil_mem[selected_frame],
                             inv_rms_q30, expected_inv_rms_mem[selected_frame],
                             warning, saturation, expected_saturation_mem[sample_index][0]);
                    failures = failures + 1;
                end
                capture_held_output();
                if ((sample_index % 251) == 0) begin
                    for (stall_index = 0; stall_index < 2; stall_index = stall_index + 1) begin
                        @(posedge clk);
                        output_stalls = output_stalls + 1;
                        check_held_output(selected_frame, channel_index);
                    end
                end
                @(negedge clk);
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
                output_count = output_count + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        cycle_count = 0;
        input_stalls = 0;
        output_stalls = 0;
        output_count = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        idle_inputs();

        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/input_activation_s8.hex", input_activation_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/input_scale32.hex", input_scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/gain_s16.hex", gain_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/gain_scale32.hex", gain_scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_aligned_s12.hex", expected_aligned_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_qgain_s16_q8.hex", expected_qgain_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_output_scale32.hex", expected_scale_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_s8.hex", expected_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_saturation.hex", expected_saturation_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_common_exponent_s8.hex", expected_common_exponent_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_rebase_shift_u6.hex", expected_rebase_shift_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_sumsq_u48.hex", expected_sumsq_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_mean_square_u48.hex", expected_mean_square_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_rms_ceil_u12.hex", expected_rms_ceil_mem);
        $readmemh("verification/generated/ace2_layer16_post_attention_rmsnorm_runtime_metadata/expected_inv_rms_q30.hex", expected_inv_rms_mem);

        apply_reset();
        warning_and_clear_probe();

        for (frame_index = 0; frame_index < MODEL_POSITIONS; frame_index = frame_index + 1) begin
            send_model_frame(frame_index);
            receive_model_frame(frame_index);
        end

        if (output_count != MODEL_SAMPLES) begin
            $display("RUNTIME_RMSNORM_OUTPUT_COUNT_FAIL got=%0d expected=%0d", output_count, MODEL_SAMPLES);
            failures = failures + 1;
        end
        if (xz_count != 0) begin
            $display("RUNTIME_RMSNORM_XZ_FAIL count=%0d", xz_count);
            failures = failures + 1;
        end
        if (failures == 0) begin
            $display("ACE2_LAYER16_POST_ATTENTION_RMSNORM_RUNTIME_METADATA_RTL_PASS samples=%0d positions=%0d groups=%0d nonzero=%0d saturations=%0d input_stalls=%0d output_stalls=%0d cycles=%0d xz_clean=1 reset_recovery=1 clear_recovery=1 invalid_scale_warning=1 backpressure_stable=1 metadata_live_derived=1",
                     MODEL_SAMPLES, MODEL_POSITIONS, MODEL_SCALE_RECORDS,
                     MODEL_NONZERO_OUTPUTS, MODEL_SATURATIONS, input_stalls,
                     output_stalls, cycle_count);
        end else begin
            $fatal(1, "RUNTIME_RMSNORM_FAIL failures=%0d", failures);
        end
        $finish;
    end
endmodule

`default_nettype wire
