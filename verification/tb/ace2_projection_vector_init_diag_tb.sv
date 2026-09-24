`timescale 1ns/1ps
`default_nettype none

module ace2_projection_vector_init_diag_tb;
    integer failures;
    integer case_index;
    integer row_index;
    integer out_index;
    integer group_index;
    integer input_flat;
    integer weight_flat;
    integer meta_flat;
    integer expected_flat;
    reg [127:0] input_word;
    reg [127:0] weight_word;
    reg [127:0] meta_word;
    reg [127:0] expected_word;
    reg [31:0] input_slice;
    reg [15:0] weight_slice;

    `include "../generated/projection_vectors.svh"

    task check_output;
        input integer selected_case;
        input integer selected_row;
        input integer selected_out;
        begin
            case_index = selected_case;
            row_index = selected_row;
            out_index = selected_out;
            for (group_index = 0; group_index < proj_case_groups[case_index]; group_index = group_index + 1) begin
                input_flat = case_index*PROJ_MAX_ROWS*PROJ_MAX_INPUT_BEATS +
                             row_index*PROJ_MAX_INPUT_BEATS +
                             (group_index / PROJ_GROUPS_PER_INPUT_BEAT);
                weight_flat = proj_case_weight_offset[case_index] +
                              out_index*proj_case_weight_beats_per_output[case_index] +
                              (group_index / PROJ_GROUPS_PER_WEIGHT_BEAT);
                input_word = proj_input_beats[input_flat];
                weight_word = proj_weight_beats[weight_flat];
                input_slice = input_word[((group_index % PROJ_GROUPS_PER_INPUT_BEAT)*PROJ_MAC_LANES*8) +: PROJ_MAC_LANES*8];
                weight_slice = weight_word[((group_index % PROJ_GROUPS_PER_WEIGHT_BEAT)*PROJ_MAC_LANES*4) +: PROJ_MAC_LANES*4];
                if ((^input_word === 1'bx) || (^weight_word === 1'bx) ||
                    (^input_slice === 1'bx) || (^weight_slice === 1'bx)) begin
                    $display("PROJ_VECTOR_INIT_X case=%0d row=%0d out=%0d group=%0d input_flat=%0d weight_flat=%0d",
                             case_index, row_index, out_index, group_index,
                             input_flat, weight_flat);
                    failures = failures + 1;
                    group_index = proj_case_groups[case_index];
                end
            end
            meta_flat = proj_case_meta_offset[case_index] + out_index;
            expected_flat = case_index*PROJ_MAX_ROWS*PROJ_MAX_OUTPUT_BEATS +
                            row_index*PROJ_MAX_OUTPUT_BEATS + (out_index >> 4);
            meta_word = proj_meta_beats[meta_flat];
            expected_word = proj_expected_beats[expected_flat];
            if ((^meta_word === 1'bx) || (^expected_word === 1'bx)) begin
                $display("PROJ_VECTOR_META_EXPECTED_X case=%0d row=%0d out=%0d meta_flat=%0d expected_flat=%0d",
                         case_index, row_index, out_index, meta_flat, expected_flat);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        #1;
        check_output(0, 0, 0);
        check_output(0, 1, 17);
        check_output(0, 1, 895);
        check_output(1, 0, 0);
        check_output(1, 0, 63);
        check_output(1, 0, 895);
        check_output(PROJ_CASE_MLP_GATE, 0, 0);
        check_output(PROJ_CASE_MLP_GATE, 0, 1023);
        check_output(PROJ_CASE_MLP_GATE, 0, 4863);
        check_output(PROJ_CASE_MLP_DOWN, 0, 0);
        check_output(PROJ_CASE_MLP_DOWN, 0, 127);
        check_output(PROJ_CASE_MLP_DOWN, 0, 895);
        check_output(PROJ_CASE_RMSNORM_CONSUMER, 0, 0);
        check_output(PROJ_CASE_RMSNORM_CONSUMER, 0, 17);
        check_output(PROJ_CASE_RMSNORM_CONSUMER, 0, 31);
        check_output(PROJ_CASE_C4_V_BIAS, 0, PROJ_C4_V_OUTPUT_CHANNEL);
        if (failures != 0) begin
            $fatal(1, "ACE2_PROJECTION_VECTOR_INIT_DIAG_FAIL failures=%0d", failures);
        end
        $display("ACE2_PROJECTION_VECTOR_INIT_DIAG_PASS checked_outputs=16");
        $finish;
    end
endmodule

`default_nettype wire
