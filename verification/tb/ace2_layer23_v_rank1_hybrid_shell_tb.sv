`timescale 1ns/1ps
`default_nettype none

module ace2_layer23_v_rank1_hybrid_shell_wrap #(
    parameter integer ENABLE_SIDECAR = 0
) (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         csr_valid_i,
    output wire         csr_ready_o,
    input  wire         csr_write_i,
    input  wire [31:0]  csr_addr_i,
    input  wire [63:0]  csr_wdata_i,
    input  wire [7:0]   csr_wstrb_i,
    input  wire         cmd_valid_i,
    output wire         cmd_ready_o,
    input  wire [7:0]   cmd_opcode_i,
    input  wire [7:0]   cmd_flags_i,
    input  wire [7:0]   cmd_layer_id_i,
    input  wire [15:0]  cmd_m_i,
    input  wire [15:0]  cmd_n_i,
    input  wire [15:0]  cmd_k_i,
    input  wire [15:0]  cmd_sequence_position_i,
    input  wire [15:0]  cmd_completion_tag_i,
    input  wire [63:0]  cmd_src0_addr_i,
    input  wire [63:0]  cmd_src1_addr_i,
    input  wire [63:0]  cmd_dst_addr_i,
    input  wire [63:0]  cmd_scale_addr_i,
    input  wire [63:0]  cmd_scratch_addr_i,
    output wire         mem_req_valid_o,
    input  wire         mem_req_ready_i,
    output wire         mem_req_write_o,
    output wire [63:0]  mem_req_addr_o,
    output wire [15:0]  mem_req_len_o,
    output wire [7:0]   mem_req_tag_o,
    output wire         mem_wvalid_o,
    input  wire         mem_wready_i,
    output wire [127:0] mem_wdata_o,
    output wire [15:0]  mem_wstrb_o,
    output wire [7:0]   mem_wtag_o,
    input  wire         mem_rvalid_i,
    output wire         mem_rready_o,
    input  wire [127:0] mem_rdata_i,
    input  wire [7:0]   mem_rtag_i,
    input  wire         mem_rerror_i,
    input  wire         mem_bvalid_i,
    output wire         mem_bready_o,
    input  wire [7:0]   mem_btag_i,
    input  wire         mem_berror_i,
    output wire         busy_o,
    output wire         cmd_done_valid_o,
    input  wire         cmd_done_ready_i,
    output wire [15:0]  cmd_done_tag_o,
    output wire         cmd_done_error_o,
    output wire         cmd_done_saturation_o,
    output wire         sidecar_command_match_o,
    output wire         sidecar_descriptor_match_o,
    output wire signed [31:0] sidecar_rank_accumulator_o,
    output wire signed [31:0] sidecar_rank_rounded_o,
    output wire signed [7:0]  sidecar_rank_s8_o,
    output wire         sidecar_rank_saturation_o,
    output wire         sidecar_correction_saturation_o,
    output wire         sidecar_add_saturation_o,
    output wire         sidecar_numeric_overflow_o,
    output wire [127:0] sidecar_baseline_beat_o
);
    wire csr_rvalid_unused;
    wire [63:0] csr_rdata_unused;
    wire csr_error_unused;
    wire irq_unused;
    wire [7:0] sram_req_valid_unused;
    wire [7:0] sram_write_unused;
    wire [8*12-1:0] sram_addr_unused;
    wire [8*128-1:0] sram_wdata_unused;
    wire [8*16-1:0] sram_wstrb_unused;
    wire [47:0] cmd_done_sumsq_unused;
    wire [31:0] cmd_done_inv_rms_unused;

    ace2_shell #(
        .ENABLE_LAYER23_V_RANK1_SIDECAR(ENABLE_SIDECAR),
        .LAYER23_V_RANK1_CONFIG_VALID(1)
    ) dut (
        .clk_i,
        .rst_ni,
        .csr_valid_i,
        .csr_ready_o,
        .csr_write_i,
        .csr_addr_i,
        .csr_wdata_i,
        .csr_wstrb_i,
        .csr_rvalid_o(csr_rvalid_unused),
        .csr_rready_i(1'b1),
        .csr_rdata_o(csr_rdata_unused),
        .csr_error_o(csr_error_unused),
        .irq_o(irq_unused),
        .cmd_valid_i,
        .cmd_ready_o,
        .cmd_opcode_i,
        .cmd_flags_i,
        .cmd_layer_id_i,
        .cmd_m_i,
        .cmd_n_i,
        .cmd_k_i,
        .cmd_sequence_position_i,
        .cmd_completion_tag_i,
        .cmd_src0_addr_i,
        .cmd_src1_addr_i,
        .cmd_dst_addr_i,
        .cmd_scale_addr_i,
        .cmd_scratch_addr_i,
        .mem_req_valid_o,
        .mem_req_ready_i,
        .mem_req_write_o,
        .mem_req_addr_o,
        .mem_req_len_o,
        .mem_req_tag_o,
        .mem_wvalid_o,
        .mem_wready_i,
        .mem_wdata_o,
        .mem_wstrb_o,
        .mem_wtag_o,
        .mem_rvalid_i,
        .mem_rready_o,
        .mem_rdata_i,
        .mem_rtag_i,
        .mem_rerror_i,
        .mem_bvalid_i,
        .mem_bready_o,
        .mem_btag_i,
        .mem_berror_i,
        .sram_req_valid_o(sram_req_valid_unused),
        .sram_req_ready_i(8'hff),
        .sram_write_o(sram_write_unused),
        .sram_addr_o(sram_addr_unused),
        .sram_wdata_o(sram_wdata_unused),
        .sram_wstrb_o(sram_wstrb_unused),
        .sram_rdata_i({8*128{1'b0}}),
        .sram_rvalid_i(8'h00),
        .busy_o,
        .cmd_done_valid_o,
        .cmd_done_ready_i,
        .cmd_done_tag_o,
        .cmd_done_error_o,
        .cmd_done_sumsq_o(cmd_done_sumsq_unused),
        .cmd_done_inv_rms_q30_o(cmd_done_inv_rms_unused),
        .cmd_done_saturation_seen_o(cmd_done_saturation_o)
    );

    assign sidecar_command_match_o = dut.layer23_v_rank1_command_match_w;
    assign sidecar_descriptor_match_o = dut.layer23_v_rank1_descriptor_match_w;
    assign sidecar_rank_accumulator_o = dut.layer23_v_rank1_rank_accumulator_w;
    assign sidecar_rank_rounded_o = dut.layer23_v_rank1_rank_rounded_w;
    assign sidecar_rank_s8_o = dut.layer23_v_rank1_rank_s8_w;
    assign sidecar_rank_saturation_o = dut.layer23_v_rank1_rank_saturation_w;
    assign sidecar_correction_saturation_o =
        dut.layer23_v_rank1_correction_saturation_w;
    assign sidecar_add_saturation_o = dut.layer23_v_rank1_add_saturation_w;
    assign sidecar_numeric_overflow_o =
        dut.layer23_v_rank1_numeric_overflow_sidecar_w;
    assign sidecar_baseline_beat_o = dut.shared_payload_q;
endmodule

module ace2_layer23_v_rank1_hybrid_shell_tb;
    localparam [7:0] OPCODE_FUSED_QKV = 8'h0b;
    localparam [63:0] ACT_BASE = 64'h0000_0010_0000_0700;
    localparam [63:0] WGT_BASE = 64'h0000_0001_0a38_4000;
    localparam [63:0] DST_BASE = 64'h0000_0010_0000_0a80;
    localparam [63:0] META_BASE = 64'h0000_0002_0047_2800;
    localparam integer K_SIZE = 896;
    localparam integer Q_OUTPUTS = 896;
    localparam integer KV_OUTPUTS = 128;
    localparam integer TOTAL_OUTPUTS = Q_OUTPUTS + 2*KV_OUTPUTS;
    localparam integer WEIGHT_BYTES_PER_OUTPUT = K_SIZE / 2;
    localparam integer Q_WEIGHT_BYTES = Q_OUTPUTS * WEIGHT_BYTES_PER_OUTPUT;
    localparam integer KV_WEIGHT_BYTES = KV_OUTPUTS * WEIGHT_BYTES_PER_OUTPUT;
    localparam integer Q_META_BYTES = Q_OUTPUTS * 16;
    localparam integer KV_META_BYTES = KV_OUTPUTS * 16;
    localparam integer V_OUTPUT_OFFSET = Q_OUTPUTS + KV_OUTPUTS;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg csr_valid = 1'b0;
    reg csr_write = 1'b0;
    reg [31:0] csr_addr = 32'd0;
    reg [63:0] csr_wdata = 64'd0;
    reg [7:0] csr_wstrb = 8'd0;
    reg cmd_valid = 1'b0;
    reg [7:0] cmd_opcode = OPCODE_FUSED_QKV;
    reg [7:0] cmd_flags = 8'd0;
    reg [7:0] cmd_layer_id = 8'd23;
    reg [15:0] cmd_m = 16'd1;
    reg [15:0] cmd_n = 16'd896;
    reg [15:0] cmd_k = 16'd896;
    reg [15:0] cmd_sequence_position = 16'd0;
    reg [15:0] cmd_completion_tag = 16'h2301;
    reg [63:0] cmd_src0_addr = ACT_BASE;
    reg [63:0] cmd_src1_addr = WGT_BASE;
    reg [63:0] cmd_dst_addr = DST_BASE;
    reg [63:0] cmd_scale_addr = META_BASE;
    reg [63:0] cmd_scratch_addr = 64'd0;
    reg mem_rvalid = 1'b0;
    reg [127:0] mem_rdata = 128'd0;
    reg [7:0] mem_rtag = 8'd0;
    reg mem_bvalid = 1'b0;
    reg [7:0] mem_btag = 8'd0;
    reg cmd_done_ready = 1'b0;

    wire csr_ready_e, csr_ready_d;
    wire cmd_ready_e, cmd_ready_d;
    wire mem_req_valid_e, mem_req_valid_d;
    wire mem_req_write_e, mem_req_write_d;
    wire [63:0] mem_req_addr_e, mem_req_addr_d;
    wire [15:0] mem_req_len_e, mem_req_len_d;
    wire [7:0] mem_req_tag_e, mem_req_tag_d;
    wire mem_wvalid_e, mem_wvalid_d;
    wire [127:0] mem_wdata_e, mem_wdata_d;
    wire [15:0] mem_wstrb_e, mem_wstrb_d;
    wire [7:0] mem_wtag_e, mem_wtag_d;
    wire mem_rready_e, mem_rready_d;
    wire mem_bready_e, mem_bready_d;
    wire busy_e, busy_d;
    wire cmd_done_valid_e, cmd_done_valid_d;
    wire [15:0] cmd_done_tag_e, cmd_done_tag_d;
    wire cmd_done_error_e, cmd_done_error_d;
    wire cmd_done_saturation_e, cmd_done_saturation_d;
    wire sidecar_command_match_e, sidecar_descriptor_match_e;
    wire signed [31:0] sidecar_rank_accumulator_e;
    wire signed [31:0] sidecar_rank_rounded_e;
    wire signed [7:0] sidecar_rank_s8_e;
    wire sidecar_rank_saturation_e;
    wire sidecar_correction_saturation_e;
    wire sidecar_add_saturation_e;
    wire sidecar_numeric_overflow_e;
    wire [127:0] sidecar_baseline_beat_e;
    wire sidecar_command_match_d, sidecar_descriptor_match_d;
    wire signed [31:0] sidecar_rank_accumulator_d;
    wire signed [31:0] sidecar_rank_rounded_d;
    wire signed [7:0] sidecar_rank_s8_d;
    wire sidecar_rank_saturation_d;
    wire sidecar_correction_saturation_d;
    wire sidecar_add_saturation_d;
    wire sidecar_numeric_overflow_d;
    wire [127:0] sidecar_baseline_beat_d;

    reg [7:0] activation_mem [0:K_SIZE-1];
    reg [7:0] v_weight_mem [0:KV_OUTPUTS*WEIGHT_BYTES_PER_OUTPUT-1];
    reg [7:0] v_meta_mem [0:KV_META_BYTES-1];
    reg [7:0] baseline_mem [0:KV_OUTPUTS-1];
    reg [7:0] corrected_mem [0:KV_OUTPUTS-1];
    reg [31:0] rank_expected_mem [0:1];
    reg [7:0] rank_s8_expected_mem [0:0];
    reg [7:0] saturation_expected_mem [0:0];
    reg [2047:0] activation_path;
    reg [2047:0] weight_path;
    reg [2047:0] meta_path;
    reg [2047:0] baseline_path;
    reg [2047:0] corrected_path;
    reg [2047:0] rank_path;
    reg [2047:0] rank_s8_path;
    reg [2047:0] saturation_path;

    reg write_request_pending = 1'b0;
    reg [63:0] write_request_addr = 64'd0;
    reg [7:0] write_request_tag = 8'd0;
    integer cycles = 0;
    integer read_requests = 0;
    integer write_requests = 0;
    integer v_write_beats = 0;
    integer descriptor_accept_beats = 0;
    integer lane;
    integer offset;
    integer local_offset;
    reg observed_rank_saturation = 1'b0;
    reg observed_correction_saturation = 1'b0;
    reg observed_add_saturation = 1'b0;
    reg observed_numeric_overflow = 1'b0;

    wire mem_req_ready = rst_n && !mem_rvalid && !write_request_pending;
    wire mem_wready = rst_n && write_request_pending && !mem_bvalid;

    function automatic [127:0] memory_beat(input [63:0] address);
        reg [127:0] beat;
        integer index;
        integer byte_index;
        begin
            beat = 128'd0;
            if ((address >= ACT_BASE) && (address < ACT_BASE + K_SIZE)) begin
                index = address - ACT_BASE;
                for (byte_index = 0; byte_index < 16; byte_index = byte_index + 1)
                    beat[byte_index*8 +: 8] = activation_mem[index + byte_index];
            end else if ((address >= WGT_BASE + Q_WEIGHT_BYTES + KV_WEIGHT_BYTES) &&
                         (address < WGT_BASE + Q_WEIGHT_BYTES +
                                          2*KV_WEIGHT_BYTES)) begin
                index = address - (WGT_BASE + Q_WEIGHT_BYTES + KV_WEIGHT_BYTES);
                for (byte_index = 0; byte_index < 16; byte_index = byte_index + 1)
                    beat[byte_index*8 +: 8] = v_weight_mem[index + byte_index];
            end else if ((address >= WGT_BASE) &&
                         (address < WGT_BASE + Q_WEIGHT_BYTES +
                                          2*KV_WEIGHT_BYTES)) begin
                beat = 128'd0;
            end else if ((address >= META_BASE + Q_META_BYTES + KV_META_BYTES) &&
                         (address < META_BASE + Q_META_BYTES + 2*KV_META_BYTES)) begin
                index = address - (META_BASE + Q_META_BYTES + KV_META_BYTES);
                for (byte_index = 0; byte_index < 16; byte_index = byte_index + 1)
                    beat[byte_index*8 +: 8] = v_meta_mem[index + byte_index];
            end else if ((address >= META_BASE) &&
                         (address < META_BASE + Q_META_BYTES + 2*KV_META_BYTES)) begin
                beat = 128'd0;
            end else begin
                $fatal(1, "read outside frozen fused-QKV ranges address=%016x", address);
            end
            memory_beat = beat;
        end
    endfunction

`define CONNECT_WRAP(PREFIX) \
        .clk_i(clk), \
        .rst_ni(rst_n), \
        .csr_valid_i(csr_valid), \
        .csr_ready_o(csr_ready_``PREFIX), \
        .csr_write_i(csr_write), \
        .csr_addr_i(csr_addr), \
        .csr_wdata_i(csr_wdata), \
        .csr_wstrb_i(csr_wstrb), \
        .cmd_valid_i(cmd_valid), \
        .cmd_ready_o(cmd_ready_``PREFIX), \
        .cmd_opcode_i(cmd_opcode), \
        .cmd_flags_i(cmd_flags), \
        .cmd_layer_id_i(cmd_layer_id), \
        .cmd_m_i(cmd_m), \
        .cmd_n_i(cmd_n), \
        .cmd_k_i(cmd_k), \
        .cmd_sequence_position_i(cmd_sequence_position), \
        .cmd_completion_tag_i(cmd_completion_tag), \
        .cmd_src0_addr_i(cmd_src0_addr), \
        .cmd_src1_addr_i(cmd_src1_addr), \
        .cmd_dst_addr_i(cmd_dst_addr), \
        .cmd_scale_addr_i(cmd_scale_addr), \
        .cmd_scratch_addr_i(cmd_scratch_addr), \
        .mem_req_valid_o(mem_req_valid_``PREFIX), \
        .mem_req_ready_i(mem_req_ready), \
        .mem_req_write_o(mem_req_write_``PREFIX), \
        .mem_req_addr_o(mem_req_addr_``PREFIX), \
        .mem_req_len_o(mem_req_len_``PREFIX), \
        .mem_req_tag_o(mem_req_tag_``PREFIX), \
        .mem_wvalid_o(mem_wvalid_``PREFIX), \
        .mem_wready_i(mem_wready), \
        .mem_wdata_o(mem_wdata_``PREFIX), \
        .mem_wstrb_o(mem_wstrb_``PREFIX), \
        .mem_wtag_o(mem_wtag_``PREFIX), \
        .mem_rvalid_i(mem_rvalid), \
        .mem_rready_o(mem_rready_``PREFIX), \
        .mem_rdata_i(mem_rdata), \
        .mem_rtag_i(mem_rtag), \
        .mem_rerror_i(1'b0), \
        .mem_bvalid_i(mem_bvalid), \
        .mem_bready_o(mem_bready_``PREFIX), \
        .mem_btag_i(mem_btag), \
        .mem_berror_i(1'b0), \
        .busy_o(busy_``PREFIX), \
        .cmd_done_valid_o(cmd_done_valid_``PREFIX), \
        .cmd_done_ready_i(cmd_done_ready), \
        .cmd_done_tag_o(cmd_done_tag_``PREFIX), \
        .cmd_done_error_o(cmd_done_error_``PREFIX), \
        .cmd_done_saturation_o(cmd_done_saturation_``PREFIX), \
        .sidecar_command_match_o(sidecar_command_match_``PREFIX), \
        .sidecar_descriptor_match_o(sidecar_descriptor_match_``PREFIX), \
        .sidecar_rank_accumulator_o(sidecar_rank_accumulator_``PREFIX), \
        .sidecar_rank_rounded_o(sidecar_rank_rounded_``PREFIX), \
        .sidecar_rank_s8_o(sidecar_rank_s8_``PREFIX), \
        .sidecar_rank_saturation_o(sidecar_rank_saturation_``PREFIX), \
        .sidecar_correction_saturation_o(sidecar_correction_saturation_``PREFIX), \
        .sidecar_add_saturation_o(sidecar_add_saturation_``PREFIX), \
        .sidecar_numeric_overflow_o(sidecar_numeric_overflow_``PREFIX), \
        .sidecar_baseline_beat_o(sidecar_baseline_beat_``PREFIX)

    ace2_layer23_v_rank1_hybrid_shell_wrap #(.ENABLE_SIDECAR(1)) enabled (
        `CONNECT_WRAP(e)
    );
    ace2_layer23_v_rank1_hybrid_shell_wrap #(.ENABLE_SIDECAR(0)) disabled (
        `CONNECT_WRAP(d)
    );
`undef CONNECT_WRAP

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n) begin
            cycles <= 0;
            mem_rvalid <= 1'b0;
            mem_bvalid <= 1'b0;
            write_request_pending <= 1'b0;
        end else begin
            cycles <= cycles + 1;
            if (cycles > 5000000)
                $fatal(1, "frozen fused-QKV transaction timed out");

            if ({csr_ready_e, cmd_ready_e, mem_req_valid_e, mem_req_write_e,
                 mem_req_addr_e, mem_req_len_e, mem_req_tag_e, mem_wvalid_e,
                 mem_wstrb_e, mem_wtag_e, mem_rready_e, mem_bready_e, busy_e} !==
                {csr_ready_d, cmd_ready_d, mem_req_valid_d, mem_req_write_d,
                 mem_req_addr_d, mem_req_len_d, mem_req_tag_d, mem_wvalid_d,
                 mem_wstrb_d, mem_wtag_d, mem_rready_d, mem_bready_d, busy_d})
                $fatal(1, "enabled/disabled shell control paths lost lockstep");

            if (mem_rvalid && mem_rready_e)
                mem_rvalid <= 1'b0;
            if (mem_bvalid && mem_bready_e)
                mem_bvalid <= 1'b0;

            if (mem_req_valid_e && mem_req_ready) begin
                if (mem_req_len_e != 16'd1)
                    $fatal(1, "unexpected memory request length");
                if (mem_req_write_e) begin
                    if (write_request_pending)
                        $fatal(1, "overlapping write requests");
                    write_request_pending <= 1'b1;
                    write_request_addr <= mem_req_addr_e;
                    write_request_tag <= mem_req_tag_e;
                    write_requests <= write_requests + 1;
                end else begin
                    if (mem_rvalid)
                        $fatal(1, "overlapping read responses");
                    mem_rdata <= memory_beat(mem_req_addr_e);
                    mem_rtag <= mem_req_tag_e;
                    mem_rvalid <= 1'b1;
                    read_requests <= read_requests + 1;
                end
            end

            if (mem_wvalid_e && mem_wready) begin
                if (!write_request_pending ||
                    (mem_wtag_e != write_request_tag) ||
                    (mem_wtag_d != write_request_tag))
                    $fatal(1, "write data without matching request");
                if ((mem_wstrb_e != 16'hffff) || (mem_wstrb_d != 16'hffff))
                    $fatal(1, "partial fused-QKV output write");
                if ((write_request_addr < DST_BASE) ||
                    (write_request_addr >= DST_BASE + TOTAL_OUTPUTS))
                    $fatal(1, "write outside frozen destination");
                offset = write_request_addr - DST_BASE;
                if (offset >= V_OUTPUT_OFFSET) begin
                    local_offset = offset - V_OUTPUT_OFFSET;
                    if (!sidecar_command_match_e || !sidecar_descriptor_match_e)
                        $fatal(1, "sidecar did not accept frozen V descriptor");
                    if (sidecar_command_match_d || sidecar_descriptor_match_d)
                        $fatal(1, "disabled control accepted sidecar descriptor");
                    if ($signed(sidecar_rank_accumulator_e) !==
                        $signed(rank_expected_mem[0]))
                        $fatal(1, "rank accumulator mismatch got=%0d expected=%0d",
                               $signed(sidecar_rank_accumulator_e),
                               $signed(rank_expected_mem[0]));
                    if ($signed(sidecar_rank_rounded_e) !==
                        $signed(rank_expected_mem[1]))
                        $fatal(1, "rank rounded mismatch got=%0d expected=%0d",
                               $signed(sidecar_rank_rounded_e),
                               $signed(rank_expected_mem[1]));
                    if (sidecar_rank_s8_e !== $signed(rank_s8_expected_mem[0]))
                        $fatal(1, "rank s8 mismatch got=%0d expected=%0d",
                               sidecar_rank_s8_e,
                               $signed(rank_s8_expected_mem[0]));
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        if (sidecar_baseline_beat_e[lane*8 +: 8] !==
                            baseline_mem[local_offset + lane])
                            $fatal(1, "enabled baseline mismatch channel=%0d",
                                   local_offset + lane);
                        if (sidecar_baseline_beat_d[lane*8 +: 8] !==
                            baseline_mem[local_offset + lane])
                            $fatal(1, "disabled baseline mismatch channel=%0d",
                                   local_offset + lane);
                        if (mem_wdata_d[lane*8 +: 8] !==
                            baseline_mem[local_offset + lane])
                            $fatal(1, "disabled control mismatch channel=%0d",
                                   local_offset + lane);
                        if (mem_wdata_e[lane*8 +: 8] !==
                            corrected_mem[local_offset + lane])
                            $fatal(1, "corrected output mismatch channel=%0d got=%0d expected=%0d",
                                   local_offset + lane,
                                   $signed(mem_wdata_e[lane*8 +: 8]),
                                   $signed(corrected_mem[local_offset + lane]));
                    end
                    observed_rank_saturation <=
                        observed_rank_saturation | sidecar_rank_saturation_e;
                    observed_correction_saturation <=
                        observed_correction_saturation |
                        sidecar_correction_saturation_e;
                    observed_add_saturation <=
                        observed_add_saturation | sidecar_add_saturation_e;
                    observed_numeric_overflow <=
                        observed_numeric_overflow | sidecar_numeric_overflow_e;
                    v_write_beats <= v_write_beats + 1;
                    descriptor_accept_beats <= descriptor_accept_beats + 1;
                    $display(
                        "ACE2_HYBRID_V_BEAT beat=%0d rank_acc=%0d rank_rounded=%0d rank_s8=%0d baseline=%032x corrected=%032x",
                        local_offset / 16,
                        $signed(sidecar_rank_accumulator_e),
                        $signed(sidecar_rank_rounded_e),
                        sidecar_rank_s8_e,
                        mem_wdata_d,
                        mem_wdata_e
                    );
                end else if (mem_wdata_e !== mem_wdata_d) begin
                    $fatal(1, "sidecar changed non-V fused-QKV output");
                end
                write_request_pending <= 1'b0;
                mem_bvalid <= 1'b1;
                mem_btag <= mem_wtag_e;
            end
        end
    end

    task automatic enable_shells;
        begin
            @(negedge clk);
            csr_valid = 1'b1;
            csr_write = 1'b1;
            csr_addr = 32'h18;
            csr_wdata = 64'd1;
            csr_wstrb = 8'hff;
            while (!csr_ready_e || !csr_ready_d)
                @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            csr_valid = 1'b0;
            csr_write = 1'b0;
            csr_wstrb = 8'd0;
            repeat (2) @(posedge clk);
        end
    endtask

    initial begin
        if (!$value$plusargs("ACTIVATION=%s", activation_path) ||
            !$value$plusargs("V_WEIGHT=%s", weight_path) ||
            !$value$plusargs("V_META=%s", meta_path) ||
            !$value$plusargs("BASELINE=%s", baseline_path) ||
            !$value$plusargs("CORRECTED=%s", corrected_path) ||
            !$value$plusargs("RANK=%s", rank_path) ||
            !$value$plusargs("RANK_S8=%s", rank_s8_path) ||
            !$value$plusargs("SATURATION=%s", saturation_path))
            $fatal(1, "missing frozen vector plusargs");
        $readmemh(activation_path, activation_mem);
        $readmemh(weight_path, v_weight_mem);
        $readmemh(meta_path, v_meta_mem);
        $readmemh(baseline_path, baseline_mem);
        $readmemh(corrected_path, corrected_mem);
        $readmemh(rank_path, rank_expected_mem);
        $readmemh(rank_s8_path, rank_s8_expected_mem);
        $readmemh(saturation_path, saturation_expected_mem);

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        enable_shells();

        while (!cmd_ready_e || !cmd_ready_d)
            @(negedge clk);
        cmd_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        cmd_valid = 1'b0;

        while (!cmd_done_valid_e || !cmd_done_valid_d)
            @(negedge clk);
        if ((cmd_done_tag_e != cmd_completion_tag) ||
            (cmd_done_tag_d != cmd_completion_tag))
            $fatal(1, "completion tag mismatch");
        if ((v_write_beats != 8) || (descriptor_accept_beats != 8))
            $fatal(1, "V coverage mismatch writes=%0d descriptor=%0d",
                   v_write_beats, descriptor_accept_beats);
        if (observed_rank_saturation !== saturation_expected_mem[0][0])
            $fatal(1, "rank saturation mismatch");
        if (observed_correction_saturation !== saturation_expected_mem[0][1])
            $fatal(1, "correction saturation mismatch");
        if (observed_add_saturation !== saturation_expected_mem[0][2])
            $fatal(1, "add saturation mismatch");
        if (observed_numeric_overflow !== saturation_expected_mem[0][3])
            $fatal(1, "numeric overflow mismatch");
        if (cmd_done_saturation_d !== saturation_expected_mem[0][4])
            $fatal(1, "disabled completion saturation mismatch");
        if (cmd_done_error_d !== saturation_expected_mem[0][4])
            $fatal(1, "disabled completion error mismatch");
        if (cmd_done_saturation_e !==
            (saturation_expected_mem[0][0] |
             saturation_expected_mem[0][1] |
             saturation_expected_mem[0][2] |
             saturation_expected_mem[0][4]))
            $fatal(1, "enabled completion saturation mismatch");
        if (cmd_done_error_e !== cmd_done_saturation_e)
            $fatal(1, "enabled completion error/saturation contract mismatch");

        cmd_done_ready = 1'b1;
        @(posedge clk);
        $display(
            "ACE2_LAYER23_V_RANK1_HYBRID_SHELL_PASS outputs=128 descriptor_accept=1 disabled_control=1 rank_exact=1 saturation=%0d overflow=%0d cycles=%0d reads=%0d writes=%0d",
            cmd_done_saturation_e,
            observed_numeric_overflow,
            cycles,
            read_requests,
            write_requests
        );
        $finish;
    end
endmodule

`default_nettype wire
