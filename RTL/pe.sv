// ============================================================================
// pe.sv
//
// Processing Element (PE) for TPU-style systolic array.
//
// Dataflow:
//
//                     weight_in
//                         |
//                         v
//                    +---------+
// activation_in ---> |   PE    | ---> activation_out
//                    |         |
//                    | A x W   |
//                    |   +     |
//                    |  ACC    |
//                    +---------+
//                         |
//                         v
//                     weight_out
//
// Operation:
//
//     accumulator <= accumulator + (activation_in * weight_in)
//
// Default datapath:
//     Activation : signed INT8
//     Weight     : signed INT8
//     Product    : signed INT16
//     Accumulator: signed INT32
//
// Activations propagate horizontally.
// Weights propagate vertically.
// ============================================================================

module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
)(
    input logic clk,
    input logic rst_n,

    // ------------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------------

    input logic enable,
    input logic clear_acc,

    // ------------------------------------------------------------------------
    // Input from PE to the left
    // ------------------------------------------------------------------------

    input logic signed [DATA_W-1:0] activation_in,

    // ------------------------------------------------------------------------
    // Input from PE above
    // ------------------------------------------------------------------------

    input logic signed [DATA_W-1:0] weight_in,

    // ------------------------------------------------------------------------
    // Output to PE on the right
    // ------------------------------------------------------------------------

    output logic signed [DATA_W-1:0] activation_out,

    // ------------------------------------------------------------------------
    // Output to PE below
    // ------------------------------------------------------------------------

    output logic signed [DATA_W-1:0] weight_out,

    // ------------------------------------------------------------------------
    // Local accumulated result
    // ------------------------------------------------------------------------

    output logic signed [ACC_W-1:0] accumulator
);

    // ========================================================================
    // Product
    //
    // INT8 x INT8 produces an INT16 result by default.
    //
    // Keep the full multiplication width before extending it to ACC_W.
    // ========================================================================

    localparam int PRODUCT_W = 2 * DATA_W;

    logic signed [PRODUCT_W-1:0] product;

    // ========================================================================
    // Sign-extended product
    //
    // Example:
    //
    //     DATA_W    = 8
    //     PRODUCT_W = 16
    //     ACC_W     = 32
    //
    //     INT16 product -> INT32 accumulator
    // ========================================================================

    logic signed [ACC_W-1:0] product_extended;

    // ========================================================================
    // Combinational multiplier
    // ========================================================================

    always_comb begin

        product = activation_in * weight_in;

    end

    // ========================================================================
    // Sign extension
    // ========================================================================

    generate

        if (ACC_W > PRODUCT_W) begin : gen_sign_extend

            always_comb begin

                product_extended = {
                    {(ACC_W - PRODUCT_W){product[PRODUCT_W-1]}},
                    product
                };

            end

        end

        else begin : gen_no_sign_extend

            always_comb begin

                product_extended = product[ACC_W-1:0];

            end

        end

    endgenerate

    // ========================================================================
    // Sequential PE logic
    //
    // Every enabled cycle:
    //
    //     1. Forward activation to the right
    //     2. Forward weight downward
    //     3. Multiply current activation and weight
    //     4. Add product into accumulator
    //
    // Nonblocking assignments ensure that the multiplication corresponds to
    // the values entering this PE during the current cycle.
    // ========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            activation_out <= '0;
            weight_out     <= '0;
            accumulator    <= '0;

        end

        else begin

            // ---------------------------------------------------------------
            // Accumulator clear has priority over computation.
            // ---------------------------------------------------------------

            if (clear_acc) begin

                accumulator <= '0;

            end

            else if (enable) begin

                accumulator <= accumulator + product_extended;

            end

            // ---------------------------------------------------------------
            // Systolic data movement
            // ---------------------------------------------------------------

            if (enable) begin

                activation_out <= activation_in;
                weight_out     <= weight_in;

            end

        end

    end

endmodule