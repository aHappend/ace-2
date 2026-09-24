`default_nettype none

module ace2_layer16_post_attention_rmsnorm_output_grouped_scale32_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 in_valid_i,
    output wire                 in_ready_o,
    input  wire signed [11:0]   aligned_activation_i,
    input  wire signed [15:0]   gain_s16_q8_i,
    input  wire [31:0]          inv_rms_q30_i,
    input  wire [31:0]          group_scale32_i,
    input  wire [4:0]           position_i,
    input  wire [9:0]           channel_i,
    input  wire                 last_i,

    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire signed [7:0]    q_o,
    output wire [31:0]          group_scale32_o,
    output wire [4:0]           position_o,
    output wire [9:0]           channel_o,
    output wire                 last_o,
    output wire                 saturation_o
);
    reg                         valid_q;
    reg signed [7:0]            data_q;
    reg [31:0]                  scale_q;
    reg [4:0]                   position_q;
    reg [9:0]                   channel_q;
    reg                         last_q;
    reg                         saturation_q;

    wire signed [27:0] activation_gain_w;
    wire signed [32:0] inv_rms_signed_w;
    wire signed [60:0] product_narrow_w;
    wire signed [63:0] product_w;
    wire scale_record_valid_w;
    reg signed [63:0] rounded_w;
    reg signed [7:0] quantized_w;
    reg saturation_w;

    function automatic signed [63:0] round_shift_even_38;
        input signed [63:0] value;
        reg negative;
        reg [63:0] magnitude;
        reg [63:0] base;
        reg [37:0] remainder;
        begin
            negative = value[63];
            magnitude = negative ? (~value + 64'd1) : value;
            base = magnitude >> 38;
            remainder = magnitude[37:0];
            if ((remainder > 38'h2000000000) ||
                ((remainder == 38'h2000000000) && base[0]))
                base = base + 64'd1;
            round_shift_even_38 = negative ? -$signed(base) : $signed(base);
        end
    endfunction

    assign activation_gain_w = aligned_activation_i * gain_s16_q8_i;
    assign inv_rms_signed_w = $signed({1'b0, inv_rms_q30_i});
    assign product_narrow_w = activation_gain_w * inv_rms_signed_w;
    assign product_w = {{3{product_narrow_w[60]}}, product_narrow_w};
    assign scale_record_valid_w = !(|group_scale32_i[31:24]) &&
                                  (group_scale32_i[15:0] >= 16'h8000);
    assign in_ready_o = !valid_q || out_ready_i;
    assign out_valid_o = valid_q;
    assign q_o = data_q;
    assign group_scale32_o = scale_q;
    assign position_o = position_q;
    assign channel_o = channel_q;
    assign last_o = last_q;
    assign saturation_o = saturation_q;

    always @* begin
        rounded_w = round_shift_even_38(product_w);
        saturation_w = 1'b0;
        if (!scale_record_valid_w) begin
            quantized_w = 8'sd0;
            saturation_w = 1'b1;
        end else if (rounded_w > 64'sd127) begin
            quantized_w = 8'sd127;
            saturation_w = 1'b1;
        end else if (rounded_w < -64'sd128) begin
            quantized_w = -8'sd128;
            saturation_w = 1'b1;
        end else begin
            quantized_w = rounded_w[7:0];
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_q <= 1'b0;
            data_q <= 8'sd0;
            scale_q <= 32'd0;
            position_q <= 5'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (clear_i) begin
            valid_q <= 1'b0;
            data_q <= 8'sd0;
            scale_q <= 32'd0;
            position_q <= 5'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (in_ready_o) begin
            if (in_valid_i) begin
                valid_q <= 1'b1;
                data_q <= quantized_w;
                scale_q <= group_scale32_i;
                position_q <= position_i;
                channel_q <= channel_i;
                last_q <= last_i;
                saturation_q <= saturation_w;
            end else begin
                valid_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
