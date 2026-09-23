// ============================================================================
// controller.sv
//
// Controller for TPU-style systolic-array accelerator.
//
// Sequence:
//   IDLE -> LOAD -> COMPUTE -> DRAIN -> STORE -> DONE -> IDLE
//
// Assumptions:
//   - ARRAY_SIZE x ARRAY_SIZE systolic array
//   - Input buffers have been populated before start
//   - COMPUTE streams ARRAY_SIZE input steps
//   - DRAIN allows data to propagate through the systolic array
//   - STORE writes all ARRAY_SIZE^2 results into the output buffer
//
// NOTE:
//   This controller is intended as a clean first RTL implementation.
//   Matrix dimensions / tiling can be added later.
// ============================================================================

module controller #(
    parameter int ARRAY_SIZE = 8,
    parameter int ADDR_W     = 10
)(
    input  logic clk,
    input  logic rst_n,

    // ------------------------------------------------------------------------
    // Host interface
    // ------------------------------------------------------------------------
    input  logic start,

    output logic busy,
    output logic done,

    // ------------------------------------------------------------------------
    // Systolic array control
    // ------------------------------------------------------------------------
    output logic compute_en,
    output logic clear_acc,

    // ------------------------------------------------------------------------
    // Input-buffer addressing
    // ------------------------------------------------------------------------
    output logic [ADDR_W-1:0] act_rd_addr,
    output logic [ADDR_W-1:0] weight_rd_addr,

    // ------------------------------------------------------------------------
    // Output-buffer interface
    // ------------------------------------------------------------------------
    output logic              output_wr_en,
    output logic [ADDR_W-1:0] output_wr_addr
);

    // ========================================================================
    // Local parameters
    // ========================================================================

    // Number of cycles used to feed an ARRAY_SIZE-wide matrix.
    localparam int COMPUTE_CYCLES = ARRAY_SIZE;

    // Allow values injected into the array to propagate to the final PE.
    //
    // For an N x N array, 2*(N-1) cycles covers propagation from the
    // top/left edges to the bottom-right PE.
    localparam int DRAIN_CYCLES = (2 * ARRAY_SIZE) - 2;

    // Number of results in one ARRAY_SIZE x ARRAY_SIZE output tile.
    localparam int NUM_OUTPUTS = ARRAY_SIZE * ARRAY_SIZE;

    // Counter width.
    localparam int COUNT_W =
        (NUM_OUTPUTS <= 1) ? 1 : $clog2(NUM_OUTPUTS + 1);

    // ========================================================================
    // FSM definition
    // ========================================================================

    typedef enum logic [2:0] {
        IDLE,
        LOAD,
        COMPUTE,
        DRAIN,
        STORE,
        DONE
    } state_t;

    state_t state;
    state_t next_state;

    // ========================================================================
    // Counters
    // ========================================================================

    logic [COUNT_W-1:0] compute_count;
    logic [COUNT_W-1:0] drain_count;
    logic [COUNT_W-1:0] store_count;

    // ========================================================================
    // State register
    // ========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // ========================================================================
    // Next-state logic
    // ========================================================================

    always_comb begin

        next_state = state;

        case (state)

            // ---------------------------------------------------------------
            // Wait for host to request an operation.
            // ---------------------------------------------------------------
            IDLE: begin
                if (start)
                    next_state = LOAD;
            end

            // ---------------------------------------------------------------
            // Setup cycle.
            //
            // Input SRAMs are assumed to already contain the matrix tile.
            // ---------------------------------------------------------------
            LOAD: begin
                next_state = COMPUTE;
            end

            // ---------------------------------------------------------------
            // Feed operands into the systolic array.
            // ---------------------------------------------------------------
            COMPUTE: begin
                if (compute_count == COMPUTE_CYCLES - 1)
                    next_state = DRAIN;
            end

            // ---------------------------------------------------------------
            // Allow remaining pipeline data to propagate.
            // ---------------------------------------------------------------
            DRAIN: begin
                if (drain_count == DRAIN_CYCLES - 1)
                    next_state = STORE;
            end

            // ---------------------------------------------------------------
            // Copy PE accumulators into output SRAM.
            // ---------------------------------------------------------------
            STORE: begin
                if (store_count == NUM_OUTPUTS - 1)
                    next_state = DONE;
            end

            // ---------------------------------------------------------------
            // Completion pulse.
            // ---------------------------------------------------------------
            DONE: begin
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end

        endcase
    end

    // ========================================================================
    // Compute counter
    // ========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin
            compute_count <= '0;
        end

        else begin

            if (state != COMPUTE) begin
                compute_count <= '0;
            end

            else begin
                compute_count <= compute_count + 1'b1;
            end

        end

    end

    // ========================================================================
    // Drain counter
    // ========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin
            drain_count <= '0;
        end

        else begin

            if (state != DRAIN) begin
                drain_count <= '0;
            end

            else begin
                drain_count <= drain_count + 1'b1;
            end

        end

    end

    // ========================================================================
    // Store counter
    // ========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin
            store_count <= '0;
        end

        else begin

            if (state != STORE) begin
                store_count <= '0;
            end

            else begin
                store_count <= store_count + 1'b1;
            end

        end

    end

    // ========================================================================
    // Output/control logic
    // ========================================================================

    always_comb begin

        // Defaults
        busy           = 1'b0;
        done           = 1'b0;

        compute_en     = 1'b0;
        clear_acc      = 1'b0;

        act_rd_addr    = '0;
        weight_rd_addr = '0;

        output_wr_en   = 1'b0;
        output_wr_addr = '0;

        case (state)

            // ---------------------------------------------------------------
            IDLE: begin

                busy      = 1'b0;

                // Clear accumulators while waiting for a new operation.
                clear_acc = 1'b1;

            end

            // ---------------------------------------------------------------
            LOAD: begin

                busy      = 1'b1;
                clear_acc = 1'b1;

                act_rd_addr    = '0;
                weight_rd_addr = '0;

            end

            // ---------------------------------------------------------------
            COMPUTE: begin

                busy       = 1'b1;
                compute_en = 1'b1;

                // Current simplified memory model:
                // one activation/weight address per compute cycle.
                act_rd_addr =
                    {{(ADDR_W-COUNT_W){1'b0}}, compute_count};

                weight_rd_addr =
                    {{(ADDR_W-COUNT_W){1'b0}}, compute_count};

            end

            // ---------------------------------------------------------------
            DRAIN: begin

                busy = 1'b1;

                // Keep the systolic pipeline clock-enabled while existing
                // operands propagate through the remaining PEs.
                compute_en = 1'b1;

            end

            // ---------------------------------------------------------------
            STORE: begin

                busy = 1'b1;

                output_wr_en   = 1'b1;
                output_wr_addr =
                    {{(ADDR_W-COUNT_W){1'b0}}, store_count};

            end

            // ---------------------------------------------------------------
            DONE: begin

                busy = 1'b0;
                done = 1'b1;

            end

            // ---------------------------------------------------------------
            default: begin

                busy = 1'b0;

            end

        endcase

    end

endmodule