// ============================================================================
// systolic_array.sv
//
// Parameterized TPU-style systolic array.
//
// Dataflow:
//
//                         WEIGHTS
//
//                    W0    W1    W2          WN
//                     |     |     |           |
//                     v     v     v           v
//
//              A0 -> [PE]--[PE]--[PE] ... --[PE]
//                     |     |     |           |
//              A1 -> [PE]--[PE]--[PE] ... --[PE]
//                     |     |     |           |
//              A2 -> [PE]--[PE]--[PE] ... --[PE]
//                     |     |     |           |
//                    ...   ...   ...         ...
//                     |     |     |           |
//              AN -> [PE]--[PE]--[PE] ... --[PE]
//
//                     |
//                     v
//
// Activations move LEFT -> RIGHT.
// Weights move TOP -> BOTTOM.
//
// Each PE computes:
//
//     accumulator += activation * weight
//
// Default:
//     ARRAY_SIZE = 8
//     DATA_W     = 8
//     ACC_W      = 32
//
//     64 MAC processing elements
// ============================================================================

module systolic_array #(
    parameter int ARRAY_SIZE = 8,
    parameter int DATA_W     = 8,
    parameter int ACC_W      = 32
)(
    input logic clk,
    input logic rst_n,

    // ------------------------------------------------------------------------
    // Global array control
    // ------------------------------------------------------------------------

    input logic enable,
    input logic clear_acc,

    // ------------------------------------------------------------------------
    // Activation inputs
    //
    // activation_in[row]
    //
    // Each activation enters from the LEFT side of its corresponding row.
    // ------------------------------------------------------------------------

    input logic signed [DATA_W-1:0]
        activation_in [0:ARRAY_SIZE-1],

    // ------------------------------------------------------------------------
    // Weight inputs
    //
    // weight_in[column]
    //
    // Each weight enters from the TOP of its corresponding column.
    // ------------------------------------------------------------------------

    input logic signed [DATA_W-1:0]
        weight_in [0:ARRAY_SIZE-1],

    // ------------------------------------------------------------------------
    // Accumulator outputs
    //
    // result[row][column]
    // ------------------------------------------------------------------------

    output logic signed [ACC_W-1:0]
        result [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    // ========================================================================
    // Internal activation interconnect
    //
    // One additional column is used for the external left-side input.
    //
    // activation_bus[row][0]
    //      = activation entering the array
    //
    // activation_bus[row][1]
    //      = output of PE[row][0]
    //
    // ...
    //
    // activation_bus[row][ARRAY_SIZE]
    //      = output of the final PE in the row
    // ========================================================================

    logic signed [DATA_W-1:0]
        activation_bus [0:ARRAY_SIZE-1][0:ARRAY_SIZE];

    // ========================================================================
    // Internal weight interconnect
    //
    // One additional row is used for the external top-side input.
    //
    // weight_bus[0][column]
    //      = weight entering the array
    //
    // weight_bus[1][column]
    //      = output of PE[0][column]
    //
    // ...
    //
    // weight_bus[ARRAY_SIZE][column]
    //      = output of final PE in the column
    // ========================================================================

    logic signed [DATA_W-1:0]
        weight_bus [0:ARRAY_SIZE][0:ARRAY_SIZE-1];

    // ========================================================================
    // Connect external activations to the LEFT edge.
    // ========================================================================

    genvar row;
    genvar col;

    generate

        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
            assign activation_bus[row][0] = activation_in[row];
        end

    endgenerate

    // ========================================================================
    // Connect external weights to the TOP edge.
    // ========================================================================

    generate

        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
            assign weight_bus[0][col] = weight_in[col];
        end

    endgenerate

    // ========================================================================
    // Processing Element Array
    //
    // PE[row][col]
    //
    // Activation:
    //
    //     activation_bus[row][col]
    //                  |
    //                  v
    //                [ PE ]
    //                  |
    //                  +----> activation_bus[row][col+1]
    //
    //
    // Weight:
    //
    //     weight_bus[row][col]
    //              |
    //              v
    //            [ PE ]
    //              |
    //              v
    //     weight_bus[row+1][col]
    //
    // ========================================================================

    generate

        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : gen_rows

            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : gen_cols

                pe #(
                    .DATA_W (DATA_W),
                    .ACC_W  (ACC_W)
                ) u_pe (
                    .clk            (clk),
                    .rst_n          (rst_n),

                    .enable         (enable),
                    .clear_acc      (clear_acc),

                    // Activation moves horizontally.
                    .activation_in  (activation_bus[row][col]),
                    .activation_out (activation_bus[row][col+1]),

                    // Weight moves vertically.
                    .weight_in      (weight_bus[row][col]),
                    .weight_out     (weight_bus[row+1][col]),

                    // Local accumulated dot-product result.
                    .accumulator    (result[row][col])
                );

            end

        end

    endgenerate

endmodule