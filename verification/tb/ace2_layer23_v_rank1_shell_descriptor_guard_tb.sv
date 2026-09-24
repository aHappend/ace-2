`timescale 1ns/1ps
`default_nettype none

import ace2_pkg::ACE2_CSR_CONTROL;
import ace2_pkg::ACE2_OPCODE_FUSED_QKV;

module ace2_layer23_v_rank1_shell_descriptor_guard_tb;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic csr_valid = 1'b0;
    logic csr_ready;
    logic csr_write = 1'b0;
    logic [31:0] csr_addr = 32'd0;
    logic [63:0] csr_wdata = 64'd0;
    logic [7:0] csr_wstrb = 8'd0;
    logic csr_rvalid;
    logic csr_rready = 1'b0;
    logic [63:0] csr_rdata;
    logic csr_error;
    logic irq;
    logic cmd_valid = 1'b0;
    logic cmd_ready;
    logic [7:0] cmd_opcode = 8'd0;
    logic [7:0] cmd_flags = 8'd0;
    logic [7:0] cmd_layer_id = 8'd0;
    logic [15:0] cmd_m = 16'd0;
    logic [15:0] cmd_n = 16'd0;
    logic [15:0] cmd_k = 16'd0;
    logic [15:0] cmd_sequence_position = 16'd0;
    logic [15:0] cmd_completion_tag = 16'd0;
    logic [63:0] cmd_src0_addr = 64'd0;
    logic [63:0] cmd_src1_addr = 64'd0;
    logic [63:0] cmd_dst_addr = 64'd0;
    logic [63:0] cmd_scale_addr = 64'd0;
    logic [63:0] cmd_scratch_addr = 64'd0;
    logic mem_req_valid;
    logic mem_req_ready = 1'b1;
    logic mem_req_write;
    logic [63:0] mem_req_addr;
    logic [15:0] mem_req_len;
    logic [7:0] mem_req_tag;
    logic mem_wvalid;
    logic mem_wready = 1'b1;
    logic [127:0] mem_wdata;
    logic [15:0] mem_wstrb;
    logic [7:0] mem_wtag;
    logic mem_rvalid = 1'b0;
    logic mem_rready;
    logic [127:0] mem_rdata = 128'd0;
    logic [7:0] mem_rtag = 8'd0;
    logic mem_rerror = 1'b0;
    logic mem_bvalid = 1'b0;
    logic mem_bready;
    logic [7:0] mem_btag = 8'd0;
    logic mem_berror = 1'b0;
    logic [7:0] sram_req_valid;
    logic [7:0] sram_req_ready = 8'hff;
    logic [7:0] sram_write;
    logic [95:0] sram_addr;
    logic [1023:0] sram_wdata;
    logic [127:0] sram_wstrb;
    logic [1023:0] sram_rdata = 1024'd0;
    logic [7:0] sram_rvalid = 8'd0;
    logic busy;
    logic cmd_done_valid;
    logic cmd_done_ready = 1'b0;
    logic [15:0] cmd_done_tag;
    logic cmd_done_error;
    logic [47:0] cmd_done_sumsq;
    logic [31:0] cmd_done_inv_rms_q30;
    logic cmd_done_saturation_seen;

    integer guard;
    integer unexpected_memory_events;

    always #5 clk = ~clk;

    ace2_shell #(
        .ENABLE_LAYER23_V_RANK1_SIDECAR(1),
        .LAYER23_V_RANK1_CONFIG_VALID(0)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .csr_valid_i(csr_valid),
        .csr_ready_o(csr_ready),
        .csr_write_i(csr_write),
        .csr_addr_i(csr_addr),
        .csr_wdata_i(csr_wdata),
        .csr_wstrb_i(csr_wstrb),
        .csr_rvalid_o(csr_rvalid),
        .csr_rready_i(csr_rready),
        .csr_rdata_o(csr_rdata),
        .csr_error_o(csr_error),
        .irq_o(irq),
        .cmd_valid_i(cmd_valid),
        .cmd_ready_o(cmd_ready),
        .cmd_opcode_i(cmd_opcode),
        .cmd_flags_i(cmd_flags),
        .cmd_layer_id_i(cmd_layer_id),
        .cmd_m_i(cmd_m),
        .cmd_n_i(cmd_n),
        .cmd_k_i(cmd_k),
        .cmd_sequence_position_i(cmd_sequence_position),
        .cmd_completion_tag_i(cmd_completion_tag),
        .cmd_src0_addr_i(cmd_src0_addr),
        .cmd_src1_addr_i(cmd_src1_addr),
        .cmd_dst_addr_i(cmd_dst_addr),
        .cmd_scale_addr_i(cmd_scale_addr),
        .cmd_scratch_addr_i(cmd_scratch_addr),
        .mem_req_valid_o(mem_req_valid),
        .mem_req_ready_i(mem_req_ready),
        .mem_req_write_o(mem_req_write),
        .mem_req_addr_o(mem_req_addr),
        .mem_req_len_o(mem_req_len),
        .mem_req_tag_o(mem_req_tag),
        .mem_wvalid_o(mem_wvalid),
        .mem_wready_i(mem_wready),
        .mem_wdata_o(mem_wdata),
        .mem_wstrb_o(mem_wstrb),
        .mem_wtag_o(mem_wtag),
        .mem_rvalid_i(mem_rvalid),
        .mem_rready_o(mem_rready),
        .mem_rdata_i(mem_rdata),
        .mem_rtag_i(mem_rtag),
        .mem_rerror_i(mem_rerror),
        .mem_bvalid_i(mem_bvalid),
        .mem_bready_o(mem_bready),
        .mem_btag_i(mem_btag),
        .mem_berror_i(mem_berror),
        .sram_req_valid_o(sram_req_valid),
        .sram_req_ready_i(sram_req_ready),
        .sram_write_o(sram_write),
        .sram_addr_o(sram_addr),
        .sram_wdata_o(sram_wdata),
        .sram_wstrb_o(sram_wstrb),
        .sram_rdata_i(sram_rdata),
        .sram_rvalid_i(sram_rvalid),
        .busy_o(busy),
        .cmd_done_valid_o(cmd_done_valid),
        .cmd_done_ready_i(cmd_done_ready),
        .cmd_done_tag_o(cmd_done_tag),
        .cmd_done_error_o(cmd_done_error),
        .cmd_done_sumsq_o(cmd_done_sumsq),
        .cmd_done_inv_rms_q30_o(cmd_done_inv_rms_q30),
        .cmd_done_saturation_seen_o(cmd_done_saturation_seen)
    );

    always @(posedge clk) begin
        if (rst_n && (mem_req_valid || mem_wvalid || (|sram_req_valid))) begin
            unexpected_memory_events = unexpected_memory_events + 1;
        end
    end

    task automatic write_control_enable;
        begin
            @(negedge clk);
            csr_addr = ACE2_CSR_CONTROL;
            csr_wdata = 64'd1;
            csr_wstrb = 8'h01;
            csr_write = 1'b1;
            csr_valid = 1'b1;
            if (!csr_ready)
                $fatal(1, "CSR control write not ready");
            @(posedge clk);
            @(negedge clk);
            csr_valid = 1'b0;
            csr_write = 1'b0;
            csr_wstrb = 8'd0;
            csr_rready = 1'b1;
            guard = 0;
            while (!csr_rvalid && guard < 16) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!csr_rvalid || csr_error)
                $fatal(1, "CSR control write failed");
            @(negedge clk);
            csr_rready = 1'b0;
        end
    endtask

    task automatic send_guarded_descriptor;
        begin
            @(negedge clk);
            cmd_opcode = ACE2_OPCODE_FUSED_QKV;
            cmd_flags = 8'h00;
            cmd_layer_id = 8'd23;
            cmd_m = 16'd1;
            cmd_n = 16'd896;
            cmd_k = 16'd896;
            cmd_sequence_position = 16'd0;
            cmd_completion_tag = 16'h23c0;
            cmd_src0_addr = 64'h0000001000000700;
            cmd_src1_addr = 64'h000000010a384000;
            cmd_dst_addr = 64'h0000001000000a80;
            cmd_scale_addr = 64'h0000000200472800;
            cmd_scratch_addr = 64'h0000000000000000;
            cmd_valid = 1'b1;
            guard = 0;
            while (!cmd_ready && guard < 64) begin
                guard = guard + 1;
                @(posedge clk);
            end
            if (!cmd_ready)
                $fatal(1, "guarded descriptor was not accepted for fail-closed completion");
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    initial begin
        unexpected_memory_events = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);
        write_control_enable();
        send_guarded_descriptor();
        guard = 0;
        while (!cmd_done_valid && guard < 64) begin
            guard = guard + 1;
            @(posedge clk);
        end
        if (!cmd_done_valid)
            $fatal(1, "invalid-config guarded descriptor did not complete");
        if (!cmd_done_error || (cmd_done_tag !== 16'h23c0))
            $fatal(1, "invalid-config completion did not report descriptor error");
        if (unexpected_memory_events != 0)
            $fatal(1, "invalid-config fail-closed descriptor issued memory traffic");
        @(negedge clk);
        cmd_done_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cmd_done_ready = 1'b0;
        $display("ACE2_LAYER23_V_RANK1_SHELL_DESCRIPTOR_GUARD_PASS invalid_config_fail_closed=1 memory_events=%0d", unexpected_memory_events);
        $finish;
    end
endmodule

`default_nettype wire
