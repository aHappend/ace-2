`default_nettype none

module ace2_layer17_rmsnorm_input_grouped_scale32_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 in_valid_i,
    output wire                 in_ready_o,
    input  wire signed [7:0]    attention_i,
    input  wire [31:0]          attention_scale32_i,
    input  wire signed [7:0]    down_i,
    input  wire [31:0]          down_scale32_i,
    input  wire [31:0]          group_scale32_i,
    input  wire [9:0]           channel_i,
    input  wire                 last_i,

    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire signed [7:0]    sum_o,
    output wire [31:0]          group_scale32_o,
    output wire [9:0]           channel_o,
    output wire                 last_o,
    output wire                 saturation_o
);
    reg                         valid_q;
    reg signed [7:0]            sum_q;
    reg [31:0]                  scale_q;
    reg [9:0]                   channel_q;
    reg                         last_q;
    reg                         saturation_q;

    reg signed [63:0] numerator_w;
    reg signed [63:0] attention_term_w;
    reg signed [63:0] down_term_w;
    reg [63:0] denominator_w;
    reg [63:0] magnitude_w;
    reg [63:0] quotient_w;
    reg [63:0] remainder_w;
    reg [63:0] rounded_magnitude_w;
    reg signed [64:0] rounded_w;
    reg signed [7:0] quantized_w;
    reg saturation_w;
    reg signed [7:0] attention_exp_w;
    reg signed [7:0] down_exp_w;
    reg signed [7:0] group_exp_w;
    reg signed [7:0] common_exp_w;
    reg [7:0] attention_shift_w;
    reg [7:0] down_shift_w;
    reg [7:0] group_shift_w;

    wire [15:0] attention_sig_w = attention_scale32_i[15:0];
    wire [15:0] down_sig_w = down_scale32_i[15:0];
    wire [15:0] group_sig_w = group_scale32_i[15:0];
    wire scale_records_valid_w =
        !(|attention_scale32_i[31:24]) &&
        !(|down_scale32_i[31:24]) &&
        !(|group_scale32_i[31:24]);

    assign in_ready_o = !valid_q || out_ready_i;
    assign out_valid_o = valid_q;
    assign sum_o = sum_q;
    assign group_scale32_o = scale_q;
    assign channel_o = channel_q;
    assign last_o = last_q;
    assign saturation_o = saturation_q;

    always @* begin
        attention_exp_w = attention_scale32_i[23:16];
        down_exp_w = down_scale32_i[23:16];
        group_exp_w = group_scale32_i[23:16];
        common_exp_w = attention_exp_w;
        if (down_exp_w < common_exp_w)
            common_exp_w = down_exp_w;
        if (group_exp_w < common_exp_w)
            common_exp_w = group_exp_w;

        attention_shift_w = attention_exp_w - common_exp_w;
        down_shift_w = down_exp_w - common_exp_w;
        group_shift_w = group_exp_w - common_exp_w;
        attention_term_w = $signed(attention_i) * $signed({1'b0, attention_sig_w});
        down_term_w = $signed(down_i) * $signed({1'b0, down_sig_w});
        numerator_w = (attention_term_w <<< attention_shift_w) +
                      (down_term_w <<< down_shift_w);
        denominator_w = {48'd0, group_sig_w} << group_shift_w;

        magnitude_w = numerator_w[63] ? (~numerator_w + 64'd1) : numerator_w;
        quotient_w = magnitude_w / denominator_w;
        remainder_w = magnitude_w % denominator_w;
        rounded_magnitude_w = quotient_w;
        if ((remainder_w > (denominator_w >> 1)) ||
            (((remainder_w << 1) == denominator_w) && quotient_w[0]))
            rounded_magnitude_w = quotient_w + 64'd1;
        rounded_w = numerator_w[63] ? -$signed({1'b0, rounded_magnitude_w}) :
                                      $signed({1'b0, rounded_magnitude_w});

        saturation_w = 1'b0;
        if (rounded_w > 65'sd127) begin
            quantized_w = 8'sd127;
            saturation_w = 1'b1;
        end else if (rounded_w < -65'sd128) begin
            quantized_w = -8'sd128;
            saturation_w = 1'b1;
        end else begin
            quantized_w = rounded_w[7:0];
        end
        if (!scale_records_valid_w) begin
            quantized_w = 8'sd0;
            saturation_w = 1'b1;
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_q <= 1'b0;
            sum_q <= 8'sd0;
            scale_q <= 32'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (clear_i) begin
            valid_q <= 1'b0;
            sum_q <= 8'sd0;
            scale_q <= 32'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (in_ready_o) begin
            if (in_valid_i) begin
                valid_q <= 1'b1;
                sum_q <= quantized_w;
                scale_q <= group_scale32_i;
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
