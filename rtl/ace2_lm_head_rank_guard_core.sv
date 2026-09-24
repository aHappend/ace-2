`timescale 1ns/1ps
`default_nettype none

module ace2_lm_head_rank_guard_core #(
    parameter integer SCORE_WIDTH = 16,
    parameter integer TOKEN_WIDTH = 18
) (
    input  wire                          clk_i,
    input  wire                          rst_ni,
    input  wire                          clear_i,

    input  wire                          in_valid_i,
    output wire                          in_ready_o,
    input  wire signed [SCORE_WIDTH-1:0] in_score_i,
    input  wire [TOKEN_WIDTH-1:0]        in_token_i,
    input  wire                          in_last_i,

    output wire                          out_valid_o,
    input  wire                          out_ready_i,
    output wire signed [SCORE_WIDTH-1:0] out_score_o,
    output wire [TOKEN_WIDTH-1:0]        out_token_o
);
    reg best_valid_q;
    reg signed [SCORE_WIDTH-1:0] best_score_q;
    reg [TOKEN_WIDTH-1:0] best_token_q;
    reg out_valid_q;
    reg signed [SCORE_WIDTH-1:0] out_score_q;
    reg [TOKEN_WIDTH-1:0] out_token_q;

    wire candidate_wins_w;

    assign in_ready_o = !out_valid_q;
    assign out_valid_o = out_valid_q;
    assign out_score_o = out_score_q;
    assign out_token_o = out_token_q;
    assign candidate_wins_w =
        !best_valid_q ||
        ($signed(in_score_i) > $signed(best_score_q)) ||
        (($signed(in_score_i) == $signed(best_score_q)) &&
         (in_token_i < best_token_q));

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            best_valid_q <= 1'b0;
            best_score_q <= {SCORE_WIDTH{1'b0}};
            best_token_q <= {TOKEN_WIDTH{1'b0}};
            out_valid_q <= 1'b0;
            out_score_q <= {SCORE_WIDTH{1'b0}};
            out_token_q <= {TOKEN_WIDTH{1'b0}};
        end else if (clear_i) begin
            best_valid_q <= 1'b0;
            best_score_q <= {SCORE_WIDTH{1'b0}};
            best_token_q <= {TOKEN_WIDTH{1'b0}};
            out_valid_q <= 1'b0;
            out_score_q <= {SCORE_WIDTH{1'b0}};
            out_token_q <= {TOKEN_WIDTH{1'b0}};
        end else begin
            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
            end

            if (in_valid_i && in_ready_o) begin
                if (in_last_i) begin
                    out_valid_q <= 1'b1;
                    out_score_q <= candidate_wins_w ? in_score_i : best_score_q;
                    out_token_q <= candidate_wins_w ? in_token_i : best_token_q;
                    best_valid_q <= 1'b0;
                end else if (candidate_wins_w) begin
                    best_valid_q <= 1'b1;
                    best_score_q <= in_score_i;
                    best_token_q <= in_token_i;
                end
            end
        end
    end
endmodule

`default_nettype wire
