// ============================================================================
// activation_buffer.sv
//
// Banked activation buffer for TPU-style systolic array.
//
// Each systolic-array row has its own activation memory bank.
//
// For an ARRAY_SIZE = 8 design:
//
//     Bank 0 -> activation_in[0]
//     Bank 1 -> activation_in[1]
//     ...
//     Bank 7 -> activation_in[7]
//
// Host writes:
//     one activation value at a time.
//
// Accelerator reads:
//     ARRAY_SIZE activation values in parallel.
//
// ---------------------------------------------------------------------------
// SYSTOLIC SKEWING
// ---------------------------------------------------------------------------
//
// Matrix multiplication:
//
//     C[i][j] = SUM_k A[i][k] * B[k][j]
//
// Activation A[i][k] must enter row i at:
//
//     t = k + i
//
// Therefore at global read step t:
//
//     row 0 reads address t
//     row 1 reads address t - 1
//     row 2 reads address t - 2
//     ...
//
// Before a row becomes active, zero is injected.
//
// Example:
//
// Cycle       Row0      Row1      Row2      Row3
// ------------------------------------------------
//   0         A00        0         0         0
//   1         A01       A10        0         0
//   2         A02       A11       A20        0
//   3         A03       A12       A21       A30
//   4          0        A13       A22       A31
//
// ============================================================================

module activation_buffer #(
    parameter int ARRAY_SIZE = 8,
    parameter int DATA_W     = 8,
    parameter int ADDR_W     = 10
)(
    input logic clk,

    // ========================================================================
    // Host write interface
    // ========================================================================

    input logic                         wr_en,

    input logic [$clog2(ARRAY_SIZE)-1:0] wr_lane,

    input logic [ADDR_W-1:0]            wr_addr,

    input logic signed [DATA_W-1:0]     wr_data,

    // ========================================================================
    // Accelerator read interface
    //
    // rd_addr represents the GLOBAL systolic-array time step.
    //
    // Individual banks automatically apply row skewing.
    // ========================================================================

    input logic [ADDR_W-1:0] rd_addr,

    // ========================================================================
    // Parallel activation outputs
    //
    // rd_data[row] feeds the left edge of systolic-array row "row".
    // ========================================================================

    output logic signed [DATA_W-1:0]
        rd_data [0:ARRAY_SIZE-1]
);

    // ========================================================================
    // Memory depth
    // ========================================================================

    localparam int DEPTH = (1 << ADDR_W);

    // ========================================================================
    // Banked activation SRAM
    //
    // memory[row][address]
    //
    // Each row is an independent logical memory bank.
    // ========================================================================

    logic signed [DATA_W-1:0]
        memory [0:ARRAY_SIZE-1][0:DEPTH-1];

    // ========================================================================
    // Host Write Logic
    // ========================================================================

    always_ff @(posedge clk) begin

        if (wr_en) begin

            memory[wr_lane][wr_addr] <= wr_data;

        end

    end

    // ========================================================================
    // Parallel Read + Systolic Skew Logic
    //
    // For row i:
    //
    //     local_address = global_time - i
    //
    // Therefore:
    //
    //     A[i][k] appears when:
    //
    //         global_time = k + i
    //
    // which is exactly the required activation injection schedule.
    //
    // Reads are currently asynchronous to keep the first implementation
    // simple. These can later be converted to synchronous SRAM reads.
    // ========================================================================

    integer row;

    integer local_addr;

    always_comb begin

        // Default all lanes to zero.
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin

            rd_data[row] = '0;

        end

        // --------------------------------------------------------------------
        // Generate skewed reads.
        // --------------------------------------------------------------------

        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin

            // Row becomes valid once global time reaches its row index.
            if (rd_addr >= row) begin

                local_addr = rd_addr - row;

                // Prevent accesses outside the physical memory.
                if (local_addr < DEPTH) begin

                    rd_data[row] =
                        memory[row][local_addr];

                end

            end

        end

    end

endmodule