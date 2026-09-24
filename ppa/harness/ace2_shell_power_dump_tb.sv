`timescale 1ns/1ps
`default_nettype none

module ace2_shell_power_dump_tb;
    ace2_shell_tb tb();

    initial begin
        $dumpfile("ppa/raw/latest/ace2_shell_lm_head_power.vcd");
        $dumpvars(1, tb.dut);
    end
endmodule

`default_nettype wire
