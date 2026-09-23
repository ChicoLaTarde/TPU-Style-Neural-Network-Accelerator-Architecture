// ============================================================================
// tpu_top.sv
//
// Top-level TPU-style neural-network accelerator.
//
// Architecture:
//                         +----------------+
//                         |   Controller   |
//                         +-------+--------+
//                                 |
//                  +--------------+--------------+
//                  |                             |
//                  v                             v
//          Activation Buffer               Weight Buffer
//          8 parallel lanes                8 parallel lanes
//                  |                             |
//                  v                             v
//             +-------------------------------------+
//             |         8 x 8 Systolic Array        |
//             |                                     |
//             | PE PE PE PE PE PE PE PE             |
//             | PE PE PE PE PE PE PE PE             |
//             | ...                                 |
//             | PE PE PE PE PE PE PE PE             |
//             +------------------+------------------+
//                                |
//                                v
//                         Output Buffer
//
// Default:
//   8 x 8 systolic array
//   INT8 activations
//   INT8 weights
//   INT32 accumulators
// ============================================================================

module tpu_top #(
    parameter int ARRAY_SIZE = 8,
    parameter int DATA_W     = 8,
    parameter int ACC_W      = 32,
    parameter int ADDR_W     = 10
)(
    input logic clk,
    input logic rst_n,

    // ========================================================================
    // Host control interface
    // ========================================================================

    input  logic start,

    output logic busy,
    output logic done,

    // ========================================================================
    // Activation buffer host write interface
    //
    // lane selects which systolic-array row receives the data.
    // ========================================================================

    input logic                     act_wr_en,
    input logic [$clog2(ARRAY_SIZE)-1:0] act_wr_lane,
    input logic [ADDR_W-1:0]        act_wr_addr,
    input logic signed [DATA_W-1:0] act_wr_data,

    // ========================================================================
    // Weight buffer host write interface
    //
    // lane selects which systolic-array column receives the data.
    // ========================================================================

    input logic                     weight_wr_en,
    input logic [$clog2(ARRAY_SIZE)-1:0] weight_wr_lane,
    input logic [ADDR_W-1:0]        weight_wr_addr,
    input logic signed [DATA_W-1:0] weight_wr_data,

    // ========================================================================
    // Output buffer host read interface
    // ========================================================================

    input  logic [ADDR_W-1:0]        out_rd_addr,
    output logic signed [ACC_W-1:0]  out_rd_data
);

    // ========================================================================
    // Controller signals
    // ========================================================================

    logic compute_en;
    logic clear_acc;

    logic [ADDR_W-1:0] act_rd_addr;
    logic [ADDR_W-1:0] weight_rd_addr;

    logic output_wr_en;
    logic [ADDR_W-1:0] output_wr_addr;

    // ========================================================================
    // Parallel systolic-array input lanes
    //
    // activation_data[i]
    //     feeds row i from the LEFT.
    //
    // weight_data[i]
    //     feeds column i from the TOP.
    // ========================================================================

    logic signed [DATA_W-1:0]
        activation_data [0:ARRAY_SIZE-1];

    logic signed [DATA_W-1:0]
        weight_data [0:ARRAY_SIZE-1];

    // ========================================================================
    // Systolic-array output
    //
    // result[row][column]
    // ========================================================================

    logic signed [ACC_W-1:0]
        array_result [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // ========================================================================
    // Output-buffer write data
    // ========================================================================

    logic signed [ACC_W-1:0] output_wr_data;

    // ========================================================================
    // Controller
    // ========================================================================

    controller #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .ADDR_W     (ADDR_W)
    ) u_controller (
        .clk            (clk),
        .rst_n          (rst_n),

        .start          (start),

        .busy           (busy),
        .done           (done),

        .compute_en     (compute_en),
        .clear_acc      (clear_acc),

        .act_rd_addr    (act_rd_addr),
        .weight_rd_addr (weight_rd_addr),

        .output_wr_en   (output_wr_en),
        .output_wr_addr (output_wr_addr)
    );

    // ========================================================================
    // Activation Buffer
    //
    // Banked memory:
    //
    //      bank 0 -> systolic row 0
    //      bank 1 -> systolic row 1
    //      ...
    //      bank 7 -> systolic row 7
    //
    // All banks are read in parallel.
    // ========================================================================

    activation_buffer #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_W     (DATA_W),
        .ADDR_W     (ADDR_W)
    ) u_activation_buffer (
        .clk        (clk),

        .wr_en      (act_wr_en),
        .wr_lane    (act_wr_lane),
        .wr_addr    (act_wr_addr),
        .wr_data    (act_wr_data),

        .rd_addr    (act_rd_addr),

        .rd_data    (activation_data)
    );

    // ========================================================================
    // Weight Buffer
    //
    // Banked memory:
    //
    //      bank 0 -> systolic column 0
    //      bank 1 -> systolic column 1
    //      ...
    //      bank 7 -> systolic column 7
    //
    // All banks are read in parallel.
    // ========================================================================

    weight_buffer #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_W     (DATA_W),
        .ADDR_W     (ADDR_W)
    ) u_weight_buffer (
        .clk        (clk),

        .wr_en      (weight_wr_en),
        .wr_lane    (weight_wr_lane),
        .wr_addr    (weight_wr_addr),
        .wr_data    (weight_wr_data),

        .rd_addr    (weight_rd_addr),

        .rd_data    (weight_data)
    );

    // ========================================================================
    // Systolic Array
    //
    // activation_data[0:7]
    //       enter from the LEFT.
    //
    // weight_data[0:7]
    //       enter from the TOP.
    //
    // Each PE performs:
    //
    // accumulator += activation * weight
    //
    // Activations propagate horizontally.
    // Weights propagate vertically.
    // ========================================================================

    systolic_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_W     (DATA_W),
        .ACC_W      (ACC_W)
    ) u_systolic_array (
        .clk            (clk),
        .rst_n          (rst_n),

        .enable         (compute_en),
        .clear_acc      (clear_acc),

        .activation_in  (activation_data),
        .weight_in      (weight_data),

        .result         (array_result)
    );

    // ========================================================================
    // Output-address decoder
    //
    // Controller presents a linear address:
    //
    //     0  1  2  3 ... 7
    //     8  9 10 11 ...15
    //    16 17 18 19 ...23
    //          ...
    //
    // Convert:
    //
    //     linear address
    //
    // into:
    //
    //     result[row][column]
    //
    // For ARRAY_SIZE = 8:
    //
    //     row    = address / 8
    //     column = address % 8
    // ========================================================================

    integer output_row;
    integer output_col;

    always_comb begin

        output_row = output_wr_addr / ARRAY_SIZE;
        output_col = output_wr_addr % ARRAY_SIZE;

        output_wr_data = '0;

        if ((output_row >= 0) &&
            (output_row < ARRAY_SIZE) &&
            (output_col >= 0) &&
            (output_col < ARRAY_SIZE)) begin

            output_wr_data =
                array_result[output_row][output_col];

        end

    end

    // ========================================================================
    // Output Buffer
    //
    // Stores ARRAY_SIZE x ARRAY_SIZE accumulated results.
    //
    // Controller writes results sequentially.
    // Host can read results using a linear address.
    // ========================================================================

    output_buffer #(
        .DATA_W (ACC_W),
        .ADDR_W (ADDR_W)
    ) u_output_buffer (
        .clk     (clk),

        .wr_en   (output_wr_en),
        .wr_addr (output_wr_addr),
        .wr_data (output_wr_data),

        .rd_addr (out_rd_addr),
        .rd_data (out_rd_data)
    );

endmodule