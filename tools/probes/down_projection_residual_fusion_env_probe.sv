module down_projection_residual_fusion_env_probe (
    input  logic signed [31:0] accumulator_i,
    input  logic signed [7:0]  residual_i,
    input  logic        [15:0] accumulator_sig_i,
    input  logic        [15:0] residual_sig_i,
    input  logic        [15:0] output_sig_i,
    input  logic        [5:0]  accumulator_shift_i,
    input  logic        [5:0]  residual_shift_i,
    input  logic        [5:0]  output_shift_i,
    output logic signed [7:0]  result_o
);
    logic signed [95:0] accumulator_term;
    logic signed [95:0] residual_term;
    logic signed [95:0] numerator;
    logic        [63:0] denominator;
    logic        [95:0] denominator_extended;
    logic        [95:0] magnitude;
    logic        [95:0] quotient;
    logic        [95:0] remainder_wide;
    logic        [63:0] remainder;
    logic        [64:0] twice_remainder;
    logic               increment;
    logic signed [96:0] rounded;

    always_comb begin
        accumulator_term = $signed(accumulator_i) * $signed({1'b0, accumulator_sig_i});
        accumulator_term = accumulator_term <<< accumulator_shift_i;
        residual_term = $signed(residual_i) * $signed({1'b0, residual_sig_i});
        residual_term = residual_term <<< residual_shift_i;
        numerator = accumulator_term + residual_term;
        denominator = {48'd0, output_sig_i} << output_shift_i;
        denominator_extended = {32'd0, denominator};
        magnitude = numerator[95] ? -numerator : numerator;
        quotient = magnitude / denominator_extended;
        remainder_wide = magnitude % denominator_extended;
        remainder = remainder_wide[63:0];
        twice_remainder = {1'b0, remainder} << 1;
        increment = (twice_remainder > {1'b0, denominator}) ||
                    ((twice_remainder == {1'b0, denominator}) && quotient[0]);
        rounded = $signed({1'b0, quotient}) + (increment ? 97'sd1 : 97'sd0);
        if (numerator[95])
            rounded = -rounded;
        if (rounded > 97'sd127)
            result_o = 8'sd127;
        else if (rounded < -97'sd128)
            result_o = -8'sd128;
        else
            result_o = rounded[7:0];
    end
endmodule
