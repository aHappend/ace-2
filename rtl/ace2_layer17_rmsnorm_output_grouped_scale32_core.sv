`default_nettype none

module ace2_layer17_rmsnorm_output_grouped_scale32_core #(
    parameter integer HIDDEN_SIZE = 896,
    parameter integer LANES = 16,
    parameter integer ACT_WIDTH = 8,
    parameter integer GAIN_WIDTH = 16,
    parameter integer ACC_WIDTH = 48,
    parameter integer INV_RMS_FRAC = 30,
    parameter integer GAIN_FRAC = 8
) (
    input  wire                              clk_i,
    input  wire                              rst_ni,
    input  wire                              clear_i,

    input  wire                              start_valid_i,
    output wire                              start_ready_o,

    input  wire                              in_valid_i,
    output wire                              in_ready_o,
    input  wire [LANES*ACT_WIDTH-1:0]       in_data_i,

    input  wire                              scale_valid_i,
    output wire                              scale_ready_o,
    input  wire [LANES*ACT_WIDTH-1:0]       scale_act_data_i,
    input  wire [LANES*GAIN_WIDTH-1:0]      gain_data_i,
    input  wire [31:0]                       group_scale32_i,
    input  wire [3:0]                        group_id_i,
    input  wire                              last_i,

    output wire                              out_valid_o,
    input  wire                              out_ready_i,
    output wire [LANES*ACT_WIDTH-1:0]       out_data_o,
    output wire [31:0]                       group_scale32_o,
    output wire [3:0]                        group_id_o,
    output wire                              last_o,

    output wire                              done_valid_o,
    input  wire                              done_ready_i,
    output wire [ACC_WIDTH-1:0]             sumsq_o,
    output wire [INV_RMS_FRAC+1:0]          inv_rms_q30_o,
    output wire                              saturation_seen_o
);
    wire core_gain_ready_w;
    wire core_scale_ready_w;
    wire core_out_valid_w;
    wire core_out_ready_w;
    wire [LANES*ACT_WIDTH-1:0] core_out_data_w;
    reg metadata_pending_q;
    reg [31:0] group_scale32_q;
    reg [3:0] group_id_q;
    reg last_q;

    wire scale_handshake_w;
    wire output_handshake_w;

    assign scale_ready_o = core_gain_ready_w && core_scale_ready_w &&
                           !metadata_pending_q;
    assign scale_handshake_w = scale_valid_i && scale_ready_o;
    assign core_out_ready_w = out_ready_i && metadata_pending_q;
    assign out_valid_o = core_out_valid_w && metadata_pending_q;
    assign out_data_o = core_out_data_w;
    assign group_scale32_o = group_scale32_q;
    assign group_id_o = group_id_q;
    assign last_o = last_q;
    assign output_handshake_w = out_valid_o && out_ready_i;

    ace2_rmsnorm_core #(
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LANES(LANES),
        .ACT_WIDTH(ACT_WIDTH),
        .GAIN_WIDTH(GAIN_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .INV_RMS_FRAC(INV_RMS_FRAC),
        .GAIN_FRAC(GAIN_FRAC)
    ) rmsnorm_core (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_i),
        .start_valid_i(start_valid_i),
        .start_ready_o(start_ready_o),
        .in_valid_i(in_valid_i),
        .in_ready_o(in_ready_o),
        .in_data_i(in_data_i),
        .gain_valid_i(scale_valid_i && !metadata_pending_q),
        .gain_ready_o(core_gain_ready_w),
        .gain_data_i(gain_data_i),
        .scale_act_valid_i(scale_valid_i && !metadata_pending_q),
        .scale_act_ready_o(core_scale_ready_w),
        .scale_act_data_i(scale_act_data_i),
        .out_valid_o(core_out_valid_w),
        .out_ready_i(core_out_ready_w),
        .out_data_o(core_out_data_w),
        .done_valid_o(done_valid_o),
        .done_ready_i(done_ready_i),
        .sumsq_o(sumsq_o),
        .inv_rms_q30_o(inv_rms_q30_o),
        .saturation_seen_o(saturation_seen_o)
    );

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            metadata_pending_q <= 1'b0;
            group_scale32_q <= 32'd0;
            group_id_q <= 4'd0;
            last_q <= 1'b0;
        end else if (clear_i) begin
            metadata_pending_q <= 1'b0;
            group_scale32_q <= 32'd0;
            group_id_q <= 4'd0;
            last_q <= 1'b0;
        end else begin
            if (output_handshake_w) begin
                metadata_pending_q <= 1'b0;
            end
            if (scale_handshake_w) begin
                metadata_pending_q <= 1'b1;
                group_scale32_q <= group_scale32_i;
                group_id_q <= group_id_i;
                last_q <= last_i;
            end
        end
    end
endmodule

`default_nettype wire
