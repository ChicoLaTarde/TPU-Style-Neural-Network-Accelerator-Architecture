// ============================================================================
// output_buffer.sv
//
// Output buffer for TPU-style systolic-array accelerator.
//
// Stores the final accumulated matrix results produced by the systolic array.
//
// Default configuration:
//
//     ARRAY_SIZE = 8
//     DATA_W     = 32
//
// Therefore:
//
//     8 x 8 = 64 output elements
//
// ---------------------------------------------------------------------------
// ADDRESS MAPPING
// ---------------------------------------------------------------------------
//
// Matrix:
//
//     C[0][0] C[0][1] ... C[0][7]
//     C[1][0] C[1][1] ... C[1][7]
//        ...     ...          ...
//     C[7][0] C[7][1] ... C[7][7]
//
// Linear output-buffer mapping:
//
//     address = row * ARRAY_SIZE + column
//
// Example:
//
//     Address 0  -> C[0][0]
//     Address 1  -> C[0][1]
//     Address 7  -> C[0][7]
//
//     Address 8  -> C[1][0]
//     Address 9  -> C[1][1]
//
//     ...
//
//     Address 63 -> C[7][7]
//
// ============================================================================

module output_buffer #(
    parameter int ARRAY_SIZE = 8,
    parameter int DATA_W     = 32,
    parameter int ADDR_W     = 10
)(
    input logic clk,

    // ========================================================================
    // Accelerator write interface
    //
    // The controller writes completed systolic-array accumulator values here.
    // ========================================================================

    input logic wr_en,

    input logic [ADDR_W-1:0] wr_addr,

    input logic signed [DATA_W-1:0] wr_data,

    // ========================================================================
    // Host read interface
    // ========================================================================

    input logic [ADDR_W-1:0] rd_addr,

    output logic signed [DATA_W-1:0] rd_data
);

    // ========================================================================
    // Number of output elements
    //
    // For an 8 x 8 array:
    //
    //     NUM_OUTPUTS = 64
    // ========================================================================

    localparam int NUM_OUTPUTS = ARRAY_SIZE * ARRAY_SIZE;

    // ========================================================================
    // Output memory
    // ========================================================================

    logic signed [DATA_W-1:0]
        memory [0:NUM_OUTPUTS-1];

    // ========================================================================
    // Write logic
    //
    // Writes occur on the rising clock edge.
    // ========================================================================

    always_ff @(posedge clk) begin

        if (wr_en) begin

            if (wr_addr < NUM_OUTPUTS) begin

                memory[wr_addr] <= wr_data;

            end

        end

    end

    // ========================================================================
    // Read logic
    //
    // Asynchronous read for simple host access.
    //
    // Invalid addresses return zero.
    // ========================================================================

    always_comb begin

        rd_data = '0;

        if (rd_addr < NUM_OUTPUTS) begin

            rd_data = memory[rd_addr];

        end

    end

endmodule