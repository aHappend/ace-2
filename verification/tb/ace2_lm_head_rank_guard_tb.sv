`timescale 1ns/1ps
`default_nettype none

module ace2_lm_head_rank_guard_tb;
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

    integer score_fd;
    integer expected_fd;
    integer scan_status;
    integer failures;
    integer step_index;
    integer token_index;
    integer guard;
    reg [SCORE_WIDTH-1:0] score_word;
    reg [TOKEN_WIDTH-1:0] expected_token;
    reg [1023:0] score_path;
    reg [1023:0] expected_path;

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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        failures = 0;
        rst_n = 1'b0;
        clear = 1'b0;
        in_valid = 1'b0;
        in_score = {SCORE_WIDTH{1'b0}};
        in_token = {TOKEN_WIDTH{1'b0}};
        in_last = 1'b0;
        out_ready = 1'b0;

        if (!$value$plusargs("SCORES=%s", score_path)) begin
            $display("ACE2_LM_HEAD_RANK_GUARD_FAIL missing SCORES plusarg");
            $finish_and_return(2);
        end
        if (!$value$plusargs("EXPECTED=%s", expected_path)) begin
            $display("ACE2_LM_HEAD_RANK_GUARD_FAIL missing EXPECTED plusarg");
            $finish_and_return(2);
        end
        score_fd = $fopen(score_path, "r");
        expected_fd = $fopen(expected_path, "r");
        if ((score_fd == 0) || (expected_fd == 0)) begin
            $display("ACE2_LM_HEAD_RANK_GUARD_FAIL vector open failed");
            $finish_and_return(2);
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (step_index = 0; step_index < STEP_COUNT; step_index = step_index + 1) begin
            scan_status = $fscanf(expected_fd, "%h\n", expected_token);
            if (scan_status != 1) begin
                $display("ACE2_LM_HEAD_RANK_GUARD_FAIL expected vector truncated step=%0d", step_index);
                $finish_and_return(2);
            end

            for (token_index = 0; token_index < TOKEN_COUNT; token_index = token_index + 1) begin
                scan_status = $fscanf(score_fd, "%h\n", score_word);
                if (scan_status != 1) begin
                    $display("ACE2_LM_HEAD_RANK_GUARD_FAIL score vector truncated step=%0d token=%0d", step_index, token_index);
                    $finish_and_return(2);
                end
                while (!in_ready) @(posedge clk);
                @(negedge clk);
                in_valid = 1'b1;
                in_score = score_word;
                in_token = TOKEN_WIDTH'(token_index);
                in_last = (token_index == TOKEN_COUNT - 1);
                if ((^in_score === 1'bx) || (^in_token === 1'bx) ||
                    (in_last === 1'bx) || (in_ready === 1'bx)) begin
                    $display("ACE2_LM_HEAD_RANK_GUARD_XZ_INPUT step=%0d token=%0d", step_index, token_index);
                    failures = failures + 1;
                end
                @(posedge clk);
            end
            @(negedge clk);
            in_valid = 1'b0;
            in_last = 1'b0;

            guard = 0;
            while (!out_valid && (guard < 32)) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!out_valid) begin
                $display("ACE2_LM_HEAD_RANK_GUARD_TIMEOUT step=%0d", step_index);
                failures = failures + 1;
            end else begin
                if ((^out_score === 1'bx) || (^out_token === 1'bx)) begin
                    $display("ACE2_LM_HEAD_RANK_GUARD_XZ_OUTPUT step=%0d", step_index);
                    failures = failures + 1;
                end
                if (out_token !== expected_token) begin
                    $display("ACE2_LM_HEAD_RANK_GUARD_MISMATCH step=%0d got=%0d expected=%0d score=%0d",
                             step_index, out_token, expected_token, out_score);
                    failures = failures + 1;
                end
                $display("ACE2_LM_HEAD_RANK_GUARD_STEP step=%0d token=%0d score=%0d",
                         step_index, out_token, out_score);
                out_ready = 1'b1;
                @(posedge clk);
                @(negedge clk);
                out_ready = 1'b0;
            end
        end

        scan_status = $fscanf(score_fd, "%h\n", score_word);
        if (scan_status == 1) begin
            $display("ACE2_LM_HEAD_RANK_GUARD_FAIL score vector has trailing data");
            failures = failures + 1;
        end
        scan_status = $fscanf(expected_fd, "%h\n", expected_token);
        if (scan_status == 1) begin
            $display("ACE2_LM_HEAD_RANK_GUARD_FAIL expected vector has trailing data");
            failures = failures + 1;
        end

        $fclose(score_fd);
        $fclose(expected_fd);
        if (failures == 0) begin
            $display("ACE2_LM_HEAD_RANK_GUARD_PASS steps=%0d tokens_per_step=%0d xz=0", STEP_COUNT, TOKEN_COUNT);
            $finish;
        end
        $display("ACE2_LM_HEAD_RANK_GUARD_FAIL failures=%0d", failures);
        $finish_and_return(1);
    end
endmodule

`default_nettype wire
