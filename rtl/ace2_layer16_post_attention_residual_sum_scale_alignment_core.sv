`timescale 1ns/1ps
`default_nettype none

// Layer-16 self-attention-output to post-attention-residual boundary.
// The carried residual and self-attention output remain signed A8 with
// independent Scale32 metadata. Arithmetic is delegated to the frozen
// one-add/one-round bounded implementation used by the predecessor.
module ace2_layer16_post_attention_residual_sum_scale_alignment_core (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       clear_i,

    input  logic                       in_valid_i,
    output logic                       in_ready_o,
    input  logic signed [7:0]          residual_s8_i,
    input  logic        [31:0]         residual_scale32_i,
    input  logic signed [7:0]          attention_s8_i,
    input  logic        [31:0]         attention_scale32_i,
    input  logic        [31:0]         output_scale32_i,
    input  logic        [9:0]          channel_i,
    input  logic                       last_i,

    output logic                       out_valid_o,
    input  logic                       out_ready_i,
    output logic signed [7:0]          sum_s8_o,
    output logic        [31:0]         output_scale32_o,
    output logic        [9:0]          channel_o,
    output logic                       last_o,
    output logic                       saturation_o,
    output logic                       descriptor_error_o,
    output logic                       numeric_overflow_o,
    output logic signed [7:0]          common_exponent_s8_o,
    output logic        [4:0]          latency_cycles_u5_o
);
    ace2_layer16_residual_sum_dual_scale_core arithmetic_i (
        .clk_i,
        .rst_ni,
        .clear_i,
        .in_valid_i,
        .in_ready_o,
        .residual_s8_i,
        .residual_scale32_i,
        .down_s8_i(attention_s8_i),
        .down_scale32_i(attention_scale32_i),
        .output_scale32_i,
        .channel_i,
        .last_i,
        .out_valid_o,
        .out_ready_i,
        .sum_s8_o,
        .output_scale32_o,
        .channel_o,
        .last_o,
        .saturation_o,
        .descriptor_error_o,
        .numeric_overflow_o,
        .common_exponent_s8_o,
        .latency_cycles_u5_o
    );
endmodule

`default_nettype wire
