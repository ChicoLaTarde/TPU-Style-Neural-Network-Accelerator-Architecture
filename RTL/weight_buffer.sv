// ============================================================================
// weight_buffer.sv
//
// Banked weight buffer for TPU-style systolic array.
//
// Each systolic-array column has its own weight memory bank:
//
//     Bank 0 -> weight_in[0]
//     Bank 1 -> weight_in[1]
//     ...
//     Bank 7 -> weight_in[7]
//
// Matrix B is stored by column:
//
//     memory[column][k] = B[k][column]
//
// ---------------------------------------------------------------------------
// SYSTOLIC SKEWING
// ---------------------------------------------------------------------------
//
// Matrix multiplication:
//
//     C[i][j] = SUM_k A[i][k] * B[k][j]
//
// Weight B[k][j] must enter column j at:
//
//     t = k + j
//
// Therefore, at global read step t:
//
//     column 0 reads address t
//     column 1 reads address t - 1
//     column 2 reads address t - 2
//     ...
//
// Before a column becomes active, zero is injected.
//
// Example:
//
// Cycle       Col0      Col1      Col2      Col3
// ------------------------------------------------
//   0         B00        0         0         0
//   1         B10       B01        0         0
//   2         B20       B11       B02        0
//   3         B30       B21       B12       B03
//
// ============================================================================

module weight_buffer #(
    parameter int ARRAY_SIZE = 8,
    parameter int DATA_W     = 8,
    parameter int ADDR_W     = 10
)(
    input logic clk,

    // ========================================================================
    // Host write interface
    // ========================================================================

    input logic wr_en,

    input logic [$clog2(ARRAY_SIZE)-1:0] wr_lane,

    input logic [ADDR_W-1:0] wr_addr,

    input logic signed [DATA_W-1:0] wr_data,

    // ========================================================================
    // Accelerator read interface
    //
    // rd_addr represents the global systolic-array injection step.
    // ========================================================================

    input logic [ADDR_W-1:0] rd_addr,

    // ========================================================================
    // Parallel weight outputs
    //
    // rd_data[column] feeds the top of systolic-array column "column".
    // ========================================================================

    output logic signed [DATA_W-1:0]
        rd_data [0:ARRAY_SIZE-1]
);

    // ========================================================================
    // Memory depth
    // ========================================================================

    localparam int DEPTH = (1 << ADDR_W);

    // ========================================================================
    // Banked weight memory
    //
    // memory[column][k]
    //
    // For matrix B:
    //
    //     memory[0][0] = B[0][0]
    //     memory[0][1] = B[1][0]
    //     memory[0][2] = B[2][0]
    //
    //     memory[1][0] = B[0][1]
    //     memory[1][1] = B[1][1]
    //
    // etc.
    // ========================================================================

    logic signed [DATA_W-1:0]
        memory [0:ARRAY_SIZE-1][0:DEPTH-1];

    // ========================================================================
    // Host write
    // ========================================================================

    always_ff @(posedge clk) begin

        if (wr_en) begin
            memory[wr_lane][wr_addr] <= wr_data;
        end

    end

    // ========================================================================
    // Parallel skewed read
    //
    // For column j:
    //
    //     local_address = global_time - j
    //
    // Therefore B[k][j] appears when:
    //
    //     global_time = k + j
    //
    // matching the required systolic injection schedule.
    // ========================================================================

    integer col;
    integer local_addr;

    always_comb begin

        // Default all columns to zero.
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
            rd_data[col] = '0;
        end

        // Generate skewed column reads.
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin

            if (rd_addr >= col) begin

                local_addr = rd_addr - col;

                if (local_addr < DEPTH) begin
                    rd_data[col] = memory[col][local_addr];
                end

            end

        end

    end

endmodule