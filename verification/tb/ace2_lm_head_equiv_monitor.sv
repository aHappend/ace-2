`timescale 1ns/1ps

module ace2_lm_head_equiv_monitor;
    integer accepted_commands;
    integer completed_commands;
    integer protocol_violations;
    reg in_flight;
    reg [15:0] in_flight_tag;
    reg held_done;
    reg [15:0] held_done_tag;
    reg held_done_error;
    reg held_done_saturation;

    initial begin
        accepted_commands = 0;
        completed_commands = 0;
        protocol_violations = 0;
        in_flight = 1'b0;
        in_flight_tag = 16'd0;
        held_done = 1'b0;
        held_done_tag = 16'd0;
        held_done_error = 1'b0;
        held_done_saturation = 1'b0;
    end

    always @(posedge ace2_shell_tb.clk) begin
        if (!ace2_shell_tb.rst_n) begin
            in_flight = 1'b0;
            held_done = 1'b0;
        end else begin
            if (ace2_shell_tb.busy && ace2_shell_tb.cmd_ready) begin
                protocol_violations = protocol_violations + 1;
                $display("EQUIV_VIOLATION cycle=%0d kind=ready_while_busy",
                         ace2_shell_tb.cycle_count);
            end
            if (ace2_shell_tb.cmd_done_valid && !ace2_shell_tb.busy) begin
                protocol_violations = protocol_violations + 1;
                $display("EQUIV_VIOLATION cycle=%0d kind=done_while_not_busy",
                         ace2_shell_tb.cycle_count);
            end
            if (ace2_shell_tb.cmd_valid && ace2_shell_tb.cmd_ready) begin
                if (in_flight) begin
                    protocol_violations = protocol_violations + 1;
                    $display("EQUIV_VIOLATION cycle=%0d kind=overlapping_command",
                             ace2_shell_tb.cycle_count);
                end
                accepted_commands = accepted_commands + 1;
                in_flight = 1'b1;
                in_flight_tag = ace2_shell_tb.cmd_completion_tag;
                $display("EQUIV_CMD cycle=%0d opcode=%02x flags=%02x layer=%0d m=%0d n=%0d k=%0d src0=%016x src1=%016x dst=%016x scale=%016x scratch=%016x tag=%04x busy=%0d ready=%0d x=%0d",
                         ace2_shell_tb.cycle_count,
                         ace2_shell_tb.cmd_opcode,
                         ace2_shell_tb.cmd_flags,
                         ace2_shell_tb.cmd_layer_id,
                         ace2_shell_tb.cmd_m,
                         ace2_shell_tb.cmd_n,
                         ace2_shell_tb.cmd_k,
                         ace2_shell_tb.cmd_src0_addr,
                         ace2_shell_tb.cmd_src1_addr,
                         ace2_shell_tb.cmd_dst_addr,
                         ace2_shell_tb.cmd_scale_addr,
                         ace2_shell_tb.cmd_scratch_addr,
                         ace2_shell_tb.cmd_completion_tag,
                         ace2_shell_tb.busy,
                         ace2_shell_tb.cmd_ready,
                         $isunknown({
                             ace2_shell_tb.cmd_opcode,
                             ace2_shell_tb.cmd_flags,
                             ace2_shell_tb.cmd_layer_id,
                             ace2_shell_tb.cmd_m,
                             ace2_shell_tb.cmd_n,
                             ace2_shell_tb.cmd_k,
                             ace2_shell_tb.cmd_src0_addr,
                             ace2_shell_tb.cmd_src1_addr,
                             ace2_shell_tb.cmd_dst_addr,
                             ace2_shell_tb.cmd_scale_addr,
                             ace2_shell_tb.cmd_scratch_addr,
                             ace2_shell_tb.cmd_completion_tag
                         }));
            end

            if (ace2_shell_tb.mem_req_valid &&
                ace2_shell_tb.mem_req_ready &&
                !ace2_shell_tb.mem_req_write &&
                ((ace2_shell_tb.mem_req_tag == 8'h40) ||
                 (ace2_shell_tb.mem_req_tag == 8'h41) ||
                 (ace2_shell_tb.mem_req_tag == 8'h42) ||
                 (ace2_shell_tb.mem_req_tag == 8'h43))) begin
                $display("EQUIV_READ cycle=%0d case=%0d tag=%02x addr=%016x x=%0d",
                         ace2_shell_tb.cycle_count,
                         ace2_shell_tb.current_proj_case,
                         ace2_shell_tb.mem_req_tag,
                         ace2_shell_tb.mem_req_addr,
                         $isunknown({
                             ace2_shell_tb.mem_req_tag,
                             ace2_shell_tb.mem_req_addr
                         }));
            end

            if (ace2_shell_tb.dut.proj_pair_valid_w &&
                ace2_shell_tb.dut.proj_pair_ready_w) begin
                $display("EQUIV_PAIR cycle=%0d case=%0d row=%0d out=%0d group=%0d act=%h weight=%h x=%0d",
                         ace2_shell_tb.cycle_count,
                         ace2_shell_tb.current_proj_case,
                         ace2_shell_tb.dut.proj_row_idx_q,
                         ace2_shell_tb.dut.proj_out_idx_q,
                         ace2_shell_tb.dut.proj_group_idx_q,
                         ace2_shell_tb.dut.proj_act_pair_w,
                         ace2_shell_tb.dut.proj_weight_q,
                         $isunknown({
                             ace2_shell_tb.dut.proj_row_idx_q,
                             ace2_shell_tb.dut.proj_out_idx_q,
                             ace2_shell_tb.dut.proj_group_idx_q,
                             ace2_shell_tb.dut.proj_act_pair_w,
                             ace2_shell_tb.dut.proj_weight_q
                         }));
            end

            if (ace2_shell_tb.dut.proj_out_valid_w &&
                ace2_shell_tb.dut.proj_out_ready_w) begin
                $display("EQUIV_LOGIT cycle=%0d case=%0d row=%0d out=%0d data=%02x acc=%h saturation=%0d x=%0d",
                         ace2_shell_tb.cycle_count,
                         ace2_shell_tb.current_proj_case,
                         ace2_shell_tb.dut.proj_row_idx_q,
                         ace2_shell_tb.dut.proj_out_idx_q,
                         ace2_shell_tb.dut.proj_out_data_w,
                         ace2_shell_tb.dut.proj_acc_w,
                         ace2_shell_tb.dut.proj_saturation_w,
                         $isunknown({
                             ace2_shell_tb.dut.proj_row_idx_q,
                             ace2_shell_tb.dut.proj_out_idx_q,
                             ace2_shell_tb.dut.proj_out_data_w,
                             ace2_shell_tb.dut.proj_acc_w,
                             ace2_shell_tb.dut.proj_saturation_w
                         }));
            end

            if (ace2_shell_tb.cmd_done_valid &&
                !ace2_shell_tb.cmd_done_ready) begin
                if (held_done &&
                    ((held_done_tag != ace2_shell_tb.cmd_done_tag) ||
                     (held_done_error != ace2_shell_tb.cmd_done_error) ||
                     (held_done_saturation !=
                      ace2_shell_tb.cmd_done_saturation))) begin
                    protocol_violations = protocol_violations + 1;
                    $display("EQUIV_VIOLATION cycle=%0d kind=unstable_completion",
                             ace2_shell_tb.cycle_count);
                end
                held_done = 1'b1;
                held_done_tag = ace2_shell_tb.cmd_done_tag;
                held_done_error = ace2_shell_tb.cmd_done_error;
                held_done_saturation =
                    ace2_shell_tb.cmd_done_saturation;
            end else begin
                held_done = 1'b0;
            end

            if (ace2_shell_tb.cmd_done_valid &&
                ace2_shell_tb.cmd_done_ready) begin
                if (!in_flight ||
                    (in_flight_tag != ace2_shell_tb.cmd_done_tag)) begin
                    protocol_violations = protocol_violations + 1;
                    $display("EQUIV_VIOLATION cycle=%0d kind=completion_tag_mismatch expected=%04x got=%04x",
                             ace2_shell_tb.cycle_count,
                             in_flight_tag,
                             ace2_shell_tb.cmd_done_tag);
                end
                completed_commands = completed_commands + 1;
                in_flight = 1'b0;
                $display("EQUIV_DONE cycle=%0d tag=%04x error=%0d saturation=%0d sumsq=%h inv_rms=%h busy=%0d ready=%0d x=%0d",
                         ace2_shell_tb.cycle_count,
                         ace2_shell_tb.cmd_done_tag,
                         ace2_shell_tb.cmd_done_error,
                         ace2_shell_tb.cmd_done_saturation,
                         ace2_shell_tb.cmd_done_sumsq,
                         ace2_shell_tb.cmd_done_inv,
                         ace2_shell_tb.busy,
                         ace2_shell_tb.cmd_ready,
                         $isunknown({
                             ace2_shell_tb.cmd_done_tag,
                             ace2_shell_tb.cmd_done_error,
                             ace2_shell_tb.cmd_done_saturation,
                             ace2_shell_tb.cmd_done_sumsq,
                             ace2_shell_tb.cmd_done_inv
                         }));
            end
        end
    end

    final begin
        $display("EQUIV_PROTOCOL accepted=%0d completed=%0d violations=%0d in_flight=%0d",
                 accepted_commands,
                 completed_commands,
                 protocol_violations,
                 in_flight);
    end
endmodule
