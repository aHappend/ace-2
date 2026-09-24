`timescale 1ns/1ps
`default_nettype none

module ace2_candidate_dynamic_q_rank_guard_tb;
    localparam integer SCORE_WIDTH = 16;
    localparam integer TOKEN_WIDTH = 18;
    localparam integer TOKEN_COUNT = 151936;
    localparam integer STEP_COUNT = 4;

    reg clk;
    reg rst_n;
    reg clear;
    reg in_valid;
    wire in_ready;
    reg signed [SCORE_WIDTH-1:0] in_score;
    reg [TOKEN_WIDTH-1:0] in_token;
    reg in_last;
    wire out_valid;
    reg out_ready;
    wire signed [SCORE_WIDTH-1:0] out_score;
    wire [TOKEN_WIDTH-1:0] out_token;
    reg [2047:0] vector_dir;
    reg [4095:0] vector_path;
    reg [SCORE_WIDTH-1:0] score_mem [0:STEP_COUNT*TOKEN_COUNT-1];
    reg [TOKEN_WIDTH-1:0] expected_mem [0:STEP_COUNT-1];
    integer failures;
    integer xz_count;
    integer step_index;
    integer token_index;
    integer guard;
    integer stall;
    reg signed [SCORE_WIDTH-1:0] held_score;
    reg [TOKEN_WIDTH-1:0] held_token;

    ace2_lm_head_rank_guard_core #(
        .SCORE_WIDTH(SCORE_WIDTH),
        .TOKEN_WIDTH(TOKEN_WIDTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .in_score_i(in_score),
        .in_token_i(in_token),
        .in_last_i(in_last),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_score_o(out_score),
        .out_token_o(out_token)
    );

    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    always @(posedge clk) begin
        if (rst_n) begin
            if ((in_ready !== 1'b0) && (in_ready !== 1'b1)) xz_count = xz_count + 1;
            if ((out_valid !== 1'b0) && (out_valid !== 1'b1)) xz_count = xz_count + 1;
            if (out_valid && ((^out_score === 1'bx) || (^out_token === 1'bx)))
                xz_count = xz_count + 1;
        end
    end

    task reset_and_clear_probes;
        begin
            rst_n = 1'b0;
            clear = 1'b0;
            in_valid = 1'b0;
            in_last = 1'b0;
            in_score = 0;
            in_token = 0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            @(negedge clk);
            in_valid = 1'b1;
            in_score = 16'sd7;
            in_token = 18'd9;
            in_last = 1'b0;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            clear = 1'b1;
            @(posedge clk);
            @(negedge clk);
            clear = 1'b0;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("RANK_CLEAR_RECOVERY_FAIL in_ready=%b out_valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
            @(negedge clk);
            in_valid = 1'b1;
            in_score = 16'sd5;
            in_token = 18'd4;
            @(posedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (!in_ready || out_valid) begin
                $display("RANK_RESET_RECOVERY_FAIL in_ready=%b out_valid=%b", in_ready, out_valid);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        xz_count = 0;
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir)) begin
            $display("RANK_FAIL missing VECTOR_DIR plusarg");
            $finish_and_return(2);
        end
        $sformat(vector_path, "%0s/rank_scores_s16.hex", vector_dir);
        $readmemh(vector_path, score_mem);
        $sformat(vector_path, "%0s/rank_expected_u18.hex", vector_dir);
        $readmemh(vector_path, expected_mem);
        reset_and_clear_probes();

        for (step_index = 0; step_index < STEP_COUNT; step_index = step_index + 1) begin
            for (token_index = 0; token_index < TOKEN_COUNT; token_index = token_index + 1) begin
                if (((token_index + step_index) % 13) == 0) @(posedge clk);
                while (!in_ready) @(posedge clk);
                @(negedge clk);
                in_valid = 1'b1;
                in_score = score_mem[step_index*TOKEN_COUNT + token_index];
                in_token = TOKEN_WIDTH'(token_index);
                in_last = (token_index == TOKEN_COUNT - 1);
                @(posedge clk);
                @(negedge clk);
                in_valid = 1'b0;
                in_last = 1'b0;
            end
            guard = 0;
            while (!out_valid && guard < 32) begin guard = guard + 1; @(posedge clk); end
            if (!out_valid) begin
                $display("RANK_TIMEOUT step=%0d", step_index);
                failures = failures + 1;
            end else begin
                held_score = out_score;
                held_token = out_token;
                for (stall = 0; stall < 3; stall = stall + 1) begin
                    @(posedge clk);
                    if (!out_valid || out_score !== held_score || out_token !== held_token) begin
                        $display("RANK_BACKPRESSURE_STABILITY_FAIL step=%0d", step_index);
                        failures = failures + 1;
                    end
                end
                if (out_token !== expected_mem[step_index]) begin
                    $display("RANK_MISMATCH step=%0d got=%0d expected=%0d", step_index, out_token, expected_mem[step_index]);
                    failures = failures + 1;
                end
                $display("DYNAMIC_Q_RANK_STEP step=%0d token=%0d score=%0d", step_index, out_token, out_score);
                @(negedge clk);
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end
        if (xz_count != 0) failures = failures + xz_count;
        if (failures == 0) begin
            $display("ACE2_CANDIDATE_DYNAMIC_Q_RANK_PASS steps=4 reset=pass clear=pass stalls=pass xz=0");
            $finish;
        end
        $display("ACE2_CANDIDATE_DYNAMIC_Q_RANK_FAIL failures=%0d xz=%0d", failures, xz_count);
        $finish_and_return(1);
    end
endmodule

`default_nettype wire
