`default_nettype none

module ace2_layer16_down_proj_accumulator_requantizer_core (
    input  wire                 clk_i,
    input  wire                 rst_ni,
    input  wire                 clear_i,

    input  wire                 in_valid_i,
    output wire                 in_ready_o,
    input  wire signed [31:0]   accumulator_i,
    input  wire signed [31:0]   multiplier_i,
    input  wire [5:0]           right_shift_i,
    input  wire [31:0]          group_scale32_i,
    input  wire [9:0]           channel_i,
    input  wire                 last_i,

    output wire                 out_valid_o,
    input  wire                 out_ready_i,
    output wire signed [7:0]    down_o,
    output wire [31:0]          group_scale32_o,
    output wire [9:0]           channel_o,
    output wire                 last_o,
    output wire                 saturation_o
);
    reg                         valid_q;
    reg signed [7:0]            data_q;
    reg [31:0]                  scale_q;
    reg [9:0]                   channel_q;
    reg                         last_q;
    reg                         saturation_q;

    wire signed [63:0] product_w;
    wire scale_record_valid_w;
    reg signed [63:0] rounded_w;
    reg signed [7:0] quantized_w;
    reg saturation_w;

    function automatic signed [63:0] round_shift_even;
        input signed [63:0] value;
        input [5:0] shift;
        reg negative;
        reg [63:0] magnitude;
        reg [63:0] base;
        reg [63:0] remainder;
        reg [63:0] mask;
        reg [63:0] half;
        begin
            if (shift == 6'd0) begin
                round_shift_even = value;
            end else begin
                negative = value[63];
                magnitude = negative ? (~value + 64'd1) : value;
                base = magnitude >> shift;
                mask = (64'h0000000000000001 << shift) - 64'd1;
                remainder = magnitude & mask;
                half = 64'h0000000000000001 << (shift - 6'd1);
                if ((remainder > half) || ((remainder == half) && base[0]))
                    base = base + 64'd1;
                round_shift_even = negative ? -$signed(base) : $signed(base);
            end
        end
    endfunction

    assign product_w = accumulator_i * multiplier_i;
    assign scale_record_valid_w =
        !(|group_scale32_i[31:24]) && (group_scale32_i[15:0] != 16'd0);
    assign in_ready_o = !valid_q || out_ready_i;
    assign out_valid_o = valid_q;
    assign down_o = data_q;
    assign group_scale32_o = scale_q;
    assign channel_o = channel_q;
    assign last_o = last_q;
    assign saturation_o = saturation_q;

    always @* begin
        rounded_w = round_shift_even(product_w, right_shift_i);
        quantized_w = 8'sd0;
        saturation_w = 1'b0;
        if (!scale_record_valid_w || multiplier_i < 0) begin
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
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (clear_i) begin
            valid_q <= 1'b0;
            data_q <= 8'sd0;
            scale_q <= 32'd0;
            channel_q <= 10'd0;
            last_q <= 1'b0;
            saturation_q <= 1'b0;
        end else if (in_ready_o) begin
            if (in_valid_i) begin
                valid_q <= 1'b1;
                data_q <= quantized_w;
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
