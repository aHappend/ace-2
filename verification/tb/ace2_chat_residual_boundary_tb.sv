`timescale 1ns/1ps
`default_nettype none

module ace2_chat_residual_boundary_tb;
    localparam integer CASE_COUNT = 896;
    localparam integer VECTOR_WIDTH = 290;

    logic clk_i = 1'b0;
    logic rst_ni = 1'b0;
    logic clear_i = 1'b0;
    logic start_valid_i = 1'b0;
    logic start_ready_o;
    logic signed [31:0] accumulator_s32_i = '0;
    logic signed [7:0] residual_s8_i = '0;
    logic [31:0] accumulator_scale32_i = '0;
    logic [31:0] residual_scale32_i = '0;
    logic [31:0] destination_scale32_i = '0;
    logic out_valid_o;
    logic out_ready_i = 1'b0;
    logic signed [7:0] fused_s8_o;
    logic positive_saturation_o, negative_saturation_o;
    logic descriptor_error_o, numeric_overflow_o;
    logic signed [95:0] numerator_s96_o;
    logic [63:0] denominator_u64_o;
    logic signed [7:0] common_exponent_s8_o;
    logic [4:0] latency_cycles_u5_o;

    logic [VECTOR_WIDTH-1:0] vectors [0:CASE_COUNT-1];
    string vector_path;
    integer case_index;
    integer cycles;
    integer positive_saturations = 0;
    integer negative_saturations = 0;

    always #5 clk_i = ~clk_i;

    ace2_down_projection_residual_fusion_core dut (.*);

    task automatic run_lane(input logic [VECTOR_WIDTH-1:0] vector);
        logic signed [7:0] lhs;
        logic signed [7:0] rhs;
        logic [31:0] lhs_scale;
        logic [31:0] rhs_scale;
        logic [31:0] destination_scale;
        logic signed [7:0] expected_output;
        logic signed [95:0] expected_numerator;
        logic [63:0] expected_denominator;
        logic signed [7:0] expected_common_exponent;
        logic expected_positive_saturation;
        logic expected_negative_saturation;
        begin
            lhs = vector[7:0];
            rhs = vector[15:8];
            lhs_scale = vector[47:16];
            rhs_scale = vector[79:48];
            destination_scale = vector[111:80];
            expected_output = vector[119:112];
            expected_numerator = vector[215:120];
            expected_denominator = vector[279:216];
            expected_common_exponent = vector[287:280];
            expected_positive_saturation = vector[288];
            expected_negative_saturation = vector[289];

            @(negedge clk_i);
            if (!start_ready_o)
                $fatal(1, "lane %0d: start interface was not ready", case_index);
            accumulator_s32_i = {{24{lhs[7]}}, lhs};
            residual_s8_i = rhs;
            accumulator_scale32_i = lhs_scale;
            residual_scale32_i = rhs_scale;
            destination_scale32_i = destination_scale;
            start_valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            cycles = 0;
            @(negedge clk_i);
            start_valid_i = 1'b0;
            while (!out_valid_o) begin
                @(posedge clk_i);
                #1;
                cycles = cycles + 1;
                if (cycles > 12)
                    $fatal(1, "lane %0d: timed out", case_index);
            end
            if (cycles !== 10 || latency_cycles_u5_o !== 5'd10)
                $fatal(1, "lane %0d: latency=%0d reported=%0d", case_index, cycles, latency_cycles_u5_o);
            if (fused_s8_o !== expected_output ||
                numerator_s96_o !== expected_numerator ||
                denominator_u64_o !== expected_denominator ||
                common_exponent_s8_o !== expected_common_exponent ||
                positive_saturation_o !== expected_positive_saturation ||
                negative_saturation_o !== expected_negative_saturation ||
                descriptor_error_o !== 1'b0 ||
                numeric_overflow_o !== 1'b0)
                $fatal(1, "lane %0d: RTL/reference mismatch", case_index);

            positive_saturations = positive_saturations + expected_positive_saturation;
            negative_saturations = negative_saturations + expected_negative_saturation;
            @(negedge clk_i);
            out_ready_i = 1'b1;
            @(posedge clk_i);
            #1;
            @(negedge clk_i);
            out_ready_i = 1'b0;
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS=<path>");
        $readmemh(vector_path, vectors);

        repeat (3) @(posedge clk_i);
        rst_ni = 1'b1;
        @(posedge clk_i);
        #1;
        if (!start_ready_o)
            $fatal(1, "core was not ready after reset");

        for (case_index = 0; case_index < CASE_COUNT; case_index = case_index + 1)
            run_lane(vectors[case_index]);

        $display(
            "ACE2_CHAT_RESIDUAL_BOUNDARY_RTL_PASS cases=%0d positive_saturations=%0d negative_saturations=%0d valid_latency=10",
            CASE_COUNT,
            positive_saturations,
            negative_saturations
        );
        $finish;
    end
endmodule

`default_nettype wire
