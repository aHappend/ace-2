module down_projection_residual_fusion_env_probe_tb;
    logic signed [31:0] accumulator_i;
    logic signed [7:0] residual_i;
    logic [15:0] accumulator_sig_i;
    logic [15:0] residual_sig_i;
    logic [15:0] output_sig_i;
    logic [5:0] accumulator_shift_i;
    logic [5:0] residual_shift_i;
    logic [5:0] output_shift_i;
    logic signed [7:0] result_o;

    down_projection_residual_fusion_env_probe dut (.*);

    task automatic check(input integer a, input integer r, input integer out_shift,
                         input integer expected);
        begin
            accumulator_i = a;
            residual_i = r;
            accumulator_sig_i = 16'd32768;
            residual_sig_i = 16'd32768;
            output_sig_i = 16'd32768;
            accumulator_shift_i = 0;
            residual_shift_i = 0;
            output_shift_i = out_shift;
            #1;
            if (result_o !== expected)
                $fatal(1, "mismatch a=%0d r=%0d shift=%0d got=%0d expected=%0d",
                       a, r, out_shift, result_o, expected);
        end
    endtask

    initial begin
        check(3, 1, 0, 4);
        check(1, 0, 1, 0);
        check(3, 0, 1, 2);
        check(-3, 0, 1, -2);
        check(500, 0, 0, 127);
        check(-500, 0, 0, -128);
        $display("ACE2_DOWN_PROJECTION_ENV_ARITHMETIC_SIM_PASS");
        $finish;
    end
endmodule
