`timescale 1ns/1ps

// ============================================================================
// tpu_tb.sv
//
// End-to-end self-checking testbench for TPU accelerator.
//
// Verifies:
//
//   Host
//     |
//     +--> Activation Buffer
//     |
//     +--> Weight Buffer
//              |
//              v
//          Controller
//              |
//              v
//        Systolic Array
//              |
//              v
//         Output Buffer
//              |
//              v
//            Host
//
// Tests:
//   1. Reset / idle state
//   2. Identity multiplication
//   3. Basic positive GEMM
//   4. Signed GEMM
//   5. Zero matrix
//   6. Negative identity
//   7. Sparse matrices
//   8. INT8 maximum
//   9. INT8 minimum
//  10. INT8 min x max
//  11. Back-to-back operations
//  12. done pulse behavior
//  13. busy behavior
//  14. Output-buffer address mapping
//  15. Randomized GEMMs
//  16. Reset after operation
//
// ============================================================================

module tpu_tb;

    // ========================================================================
    // Parameters
    // ========================================================================

    localparam int ARRAY_SIZE = 3;
    localparam int DATA_W     = 8;
    localparam int ACC_W      = 32;
    localparam int ADDR_W     = 6;

    localparam int LANE_W =
        (ARRAY_SIZE <= 1) ? 1 : $clog2(ARRAY_SIZE);

    localparam time CLK_PERIOD = 10ns;

    // Safety timeout. The DUT should finish far before this.
    localparam int TIMEOUT_CYCLES = 200;

    // ========================================================================
    // DUT signals
    // ========================================================================

    logic clk;
    logic rst_n;

    logic start;

    logic busy;
    logic done;

    // Activation write interface

    logic                     act_wr_en;
    logic [LANE_W-1:0]        act_wr_lane;
    logic [ADDR_W-1:0]        act_wr_addr;
    logic signed [DATA_W-1:0] act_wr_data;

    // Weight write interface

    logic                     weight_wr_en;
    logic [LANE_W-1:0]        weight_wr_lane;
    logic [ADDR_W-1:0]        weight_wr_addr;
    logic signed [DATA_W-1:0] weight_wr_data;

    // Output read interface

    logic [ADDR_W-1:0]        out_rd_addr;
    logic signed [ACC_W-1:0]  out_rd_data;

    // ========================================================================
    // Test matrices
    // ========================================================================

    logic signed [DATA_W-1:0]
        matrix_a [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    logic signed [DATA_W-1:0]
        matrix_b [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    logic signed [ACC_W-1:0]
        expected [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // ========================================================================
    // Statistics
    // ========================================================================

    integer tests_run;
    integer tests_failed;

    integer r;
    integer c;
    integer trial;

    // ========================================================================
    // DUT
    // ========================================================================

    tpu_top #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_W     (DATA_W),
        .ACC_W      (ACC_W),
        .ADDR_W     (ADDR_W)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),

        .start          (start),

        .busy           (busy),
        .done           (done),

        .act_wr_en      (act_wr_en),
        .act_wr_lane    (act_wr_lane),
        .act_wr_addr    (act_wr_addr),
        .act_wr_data    (act_wr_data),

        .weight_wr_en   (weight_wr_en),
        .weight_wr_lane (weight_wr_lane),
        .weight_wr_addr (weight_wr_addr),
        .weight_wr_data (weight_wr_data),

        .out_rd_addr    (out_rd_addr),
        .out_rd_data    (out_rd_data)
    );

    // ========================================================================
    // Clock
    // ========================================================================

    initial begin

        clk = 1'b0;

        forever begin
            #(CLK_PERIOD / 2);
            clk = ~clk;
        end

    end

    // ========================================================================
    // Initialize matrices
    // ========================================================================

    task automatic zero_matrices();

        integer i;
        integer j;

        begin

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    matrix_a[i][j] = '0;
                    matrix_b[i][j] = '0;
                    expected[i][j] = '0;

                end

            end

        end

    endtask

    // ========================================================================
    // Golden model
    //
    // C = A x B
    // ========================================================================

    task automatic calculate_expected();

        integer i;
        integer j;
        integer k;

        logic signed [(2*DATA_W)-1:0] product;

        begin

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    expected[i][j] = '0;

                    for (k = 0; k < ARRAY_SIZE; k = k + 1) begin

                        product =
                            matrix_a[i][k] *
                            matrix_b[k][j];

                        expected[i][j] =
                            expected[i][j] + product;

                    end

                end

            end

        end

    endtask

    // ========================================================================
    // Write activation
    //
    // Activation buffer organization:
    //
    //     bank/row i
    //     address k
    //
    // stores:
    //
    //     A[i][k]
    // ========================================================================

    task automatic write_activation(
        input integer row,
        input integer index,
        input logic signed [DATA_W-1:0] value
    );

        begin

            @(negedge clk);

            act_wr_en   = 1'b1;
            act_wr_lane = row[LANE_W-1:0];
            act_wr_addr = index[ADDR_W-1:0];
            act_wr_data = value;

            @(posedge clk);
            #1;

            @(negedge clk);

            act_wr_en = 1'b0;

        end

    endtask

    // ========================================================================
    // Write weight
    //
    // Weight buffer organization:
    //
    //     bank/column j
    //     address k
    //
    // stores:
    //
    //     B[k][j]
    // ========================================================================

    task automatic write_weight(
        input integer column,
        input integer index,
        input logic signed [DATA_W-1:0] value
    );

        begin

            @(negedge clk);

            weight_wr_en   = 1'b1;
            weight_wr_lane = column[LANE_W-1:0];
            weight_wr_addr = index[ADDR_W-1:0];
            weight_wr_data = value;

            @(posedge clk);
            #1;

            @(negedge clk);

            weight_wr_en = 1'b0;

        end

    endtask

    // ========================================================================
    // Load both matrices into DUT
    // ========================================================================

    task automatic load_matrices();

        integer i;
        integer j;

        begin

            // ---------------------------------------------------------------
            // Load matrix A by rows.
            // ---------------------------------------------------------------

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    write_activation(
                        i,
                        j,
                        matrix_a[i][j]
                    );

                end

            end

            // ---------------------------------------------------------------
            // Load matrix B by columns.
            //
            // memory[column][k] = B[k][column]
            // ---------------------------------------------------------------

            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                    write_weight(
                        j,
                        i,
                        matrix_b[i][j]
                    );

                end

            end

        end

    endtask

    // ========================================================================
    // Start accelerator
    // ========================================================================

    task automatic start_accelerator();

        begin

            @(negedge clk);

            start = 1'b1;

            @(posedge clk);
            #1;

            @(negedge clk);

            start = 1'b0;

        end

    endtask

    // ========================================================================
    // Wait for completion
    //
    // Also checks that busy asserted during operation.
    // ========================================================================

    task automatic wait_for_done();

        integer cycles;
        logic saw_busy;

        begin

            cycles   = 0;
            saw_busy = 1'b0;

            while ((done !== 1'b1) &&
                   (cycles < TIMEOUT_CYCLES)) begin

                @(posedge clk);
                #1;

                if (busy)
                    saw_busy = 1'b1;

                cycles++;

            end

            tests_run++;

            if (cycles >= TIMEOUT_CYCLES) begin

                $error(
                    "Accelerator TIMEOUT after %0d cycles",
                    TIMEOUT_CYCLES
                );

                tests_failed++;

            end

            else begin

                $display(
                    "[PASS] Accelerator completed in %0d cycles",
                    cycles
                );

            end

            tests_run++;

            if (!saw_busy) begin

                $error(
                    "busy never asserted during operation"
                );

                tests_failed++;

            end

            else begin

                $display(
                    "[PASS] busy asserted during operation"
                );

            end

        end

    endtask

    // ========================================================================
    // Read one output
    // ========================================================================

    task automatic read_output(
        input integer row,
        input integer column,
        output logic signed [ACC_W-1:0] value
    );

        integer address;

        begin

            address =
                (row * ARRAY_SIZE) + column;

            out_rd_addr =
                address[ADDR_W-1:0];

            #1;

            value = out_rd_data;

        end

    endtask

    // ========================================================================
    // Check complete output matrix
    // ========================================================================

    task automatic check_outputs(
        input string test_name
    );

        integer i;
        integer j;

        logic signed [ACC_W-1:0] actual;

        begin

            $display("");
            $display("----------------------------------------");
            $display("Checking %s", test_name);
            $display("----------------------------------------");

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    read_output(
                        i,
                        j,
                        actual
                    );

                    tests_run++;

                    if (actual !== expected[i][j]) begin

                        $error(
                            "%s C[%0d][%0d] FAILED expected=%0d actual=%0d",
                            test_name,
                            i,
                            j,
                            expected[i][j],
                            actual
                        );

                        tests_failed++;

                    end

                    else begin

                        $display(
                            "[PASS] C[%0d][%0d] = %0d",
                            i,
                            j,
                            actual
                        );

                    end

                end

            end

        end

    endtask

    // ========================================================================
    // Run complete GEMM
    //
    // Host:
    //      write A
    //      write B
    //      start
    //      wait
    //      read C
    // ========================================================================

    task automatic run_gemm(
        input string test_name
    );

        begin

            calculate_expected();

            load_matrices();

            start_accelerator();

            wait_for_done();

            check_outputs(
                test_name
            );

        end

    endtask

    // ========================================================================
    // Main test sequence
    // ========================================================================

    initial begin

        tests_run    = 0;
        tests_failed = 0;

        rst_n = 1'b0;
        start = 1'b0;

        act_wr_en   = 1'b0;
        act_wr_lane = '0;
        act_wr_addr = '0;
        act_wr_data = '0;

        weight_wr_en   = 1'b0;
        weight_wr_lane = '0;
        weight_wr_addr = '0;
        weight_wr_data = '0;

        out_rd_addr = '0;

        zero_matrices();

        // ====================================================================
        // TEST 1: RESET / IDLE
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 1: RESET / IDLE");
        $display("========================================");

        repeat (3) @(posedge clk);

        #1;

        tests_run++;

        if (busy !== 1'b0) begin

            $error("busy should be 0 during reset");
            tests_failed++;

        end
        else begin

            $display("[PASS] busy=0 during reset");

        end

        tests_run++;

        if (done !== 1'b0) begin

            $error("done should be 0 during reset");
            tests_failed++;

        end
        else begin

            $display("[PASS] done=0 during reset");

        end

        @(negedge clk);
        rst_n = 1'b1;

        // ====================================================================
        // TEST 2: IDENTITY
        //
        // A x I = A
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 2: IDENTITY");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd1;
        matrix_a[0][1] = 8'sd2;
        matrix_a[0][2] = 8'sd3;

        matrix_a[1][0] = 8'sd4;
        matrix_a[1][1] = 8'sd5;
        matrix_a[1][2] = 8'sd6;

        matrix_a[2][0] = 8'sd7;
        matrix_a[2][1] = 8'sd8;
        matrix_a[2][2] = 8'sd9;

        matrix_b[0][0] = 8'sd1;
        matrix_b[1][1] = 8'sd1;
        matrix_b[2][2] = 8'sd1;

        run_gemm(
            "Identity"
        );

        // ====================================================================
        // TEST 3: BASIC POSITIVE GEMM
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 3: BASIC POSITIVE GEMM");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd1;
        matrix_a[0][1] = 8'sd2;
        matrix_a[0][2] = 8'sd3;

        matrix_a[1][0] = 8'sd4;
        matrix_a[1][1] = 8'sd5;
        matrix_a[1][2] = 8'sd6;

        matrix_a[2][0] = 8'sd7;
        matrix_a[2][1] = 8'sd8;
        matrix_a[2][2] = 8'sd9;

        matrix_b[0][0] = 8'sd9;
        matrix_b[0][1] = 8'sd8;
        matrix_b[0][2] = 8'sd7;

        matrix_b[1][0] = 8'sd6;
        matrix_b[1][1] = 8'sd5;
        matrix_b[1][2] = 8'sd4;

        matrix_b[2][0] = 8'sd3;
        matrix_b[2][1] = 8'sd2;
        matrix_b[2][2] = 8'sd1;

        run_gemm(
            "Basic positive GEMM"
        );

        // ====================================================================
        // TEST 4: SIGNED GEMM
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 4: SIGNED GEMM");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] =  8'sd1;
        matrix_a[0][1] = -8'sd2;
        matrix_a[0][2] =  8'sd3;

        matrix_a[1][0] = -8'sd4;
        matrix_a[1][1] =  8'sd5;
        matrix_a[1][2] = -8'sd6;

        matrix_a[2][0] =  8'sd7;
        matrix_a[2][1] = -8'sd8;
        matrix_a[2][2] =  8'sd9;

        matrix_b[0][0] = -8'sd2;
        matrix_b[0][1] =  8'sd3;
        matrix_b[0][2] = -8'sd4;

        matrix_b[1][0] =  8'sd5;
        matrix_b[1][1] = -8'sd6;
        matrix_b[1][2] =  8'sd7;

        matrix_b[2][0] = -8'sd8;
        matrix_b[2][1] =  8'sd9;
        matrix_b[2][2] = -8'sd10;

        run_gemm(
            "Signed GEMM"
        );

        // ====================================================================
        // TEST 5: ZERO MATRIX
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 5: ZERO MATRIX");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd127;
        matrix_a[0][1] = -8'sd128;
        matrix_a[1][0] = 8'sd42;
        matrix_a[2][2] = -8'sd91;

        // B remains zero.

        run_gemm(
            "Zero matrix"
        );

        // ====================================================================
        // TEST 6: NEGATIVE IDENTITY
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 6: NEGATIVE IDENTITY");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd1;
        matrix_a[0][1] = 8'sd2;
        matrix_a[0][2] = 8'sd3;

        matrix_a[1][0] = 8'sd4;
        matrix_a[1][1] = 8'sd5;
        matrix_a[1][2] = 8'sd6;

        matrix_a[2][0] = 8'sd7;
        matrix_a[2][1] = 8'sd8;
        matrix_a[2][2] = 8'sd9;

        matrix_b[0][0] = -8'sd1;
        matrix_b[1][1] = -8'sd1;
        matrix_b[2][2] = -8'sd1;

        run_gemm(
            "Negative identity"
        );

        // ====================================================================
        // TEST 7: SPARSE MATRICES
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 7: SPARSE MATRICES");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd5;
        matrix_a[1][2] = -8'sd7;
        matrix_a[2][1] = 8'sd11;

        matrix_b[0][2] = 8'sd3;
        matrix_b[1][0] = -8'sd4;
        matrix_b[2][1] = 8'sd6;

        run_gemm(
            "Sparse GEMM"
        );

        // ====================================================================
        // TEST 8: INT8 MAXIMUM
        //
        // Each result:
        //
        //     3 * 127 * 127 = 48387
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 8: INT8 MAXIMUM");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                matrix_a[r][c] = 8'sd127;
                matrix_b[r][c] = 8'sd127;

            end

        end

        run_gemm(
            "INT8 maximum"
        );

        // ====================================================================
        // TEST 9: INT8 MINIMUM
        //
        // Each result:
        //
        //     3 * (-128) * (-128) = 49152
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 9: INT8 MINIMUM");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                matrix_a[r][c] = -8'sd128;
                matrix_b[r][c] = -8'sd128;

            end

        end

        run_gemm(
            "INT8 minimum"
        );

        // ====================================================================
        // TEST 10: INT8 MIN x MAX
        //
        // Each result:
        //
        //     3 * (-128) * 127 = -48768
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 10: INT8 MIN x MAX");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                matrix_a[r][c] = -8'sd128;
                matrix_b[r][c] =  8'sd127;

            end

        end

        run_gemm(
            "INT8 min x max"
        );

        // ====================================================================
        // TEST 11: BACK-TO-BACK OPERATIONS
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 11: BACK-TO-BACK");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            matrix_a[r][r] = 8'sd2;
            matrix_b[r][r] = 8'sd3;

        end

        run_gemm(
            "Back-to-back #1"
        );

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            matrix_a[r][r] = -8'sd4;
            matrix_b[r][r] =  8'sd5;

        end

        run_gemm(
            "Back-to-back #2"
        );

        // ====================================================================
        // TEST 12: DONE PULSE
        //
        // DONE should return low after the completion state.
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 12: DONE PULSE");
        $display("========================================");

        @(posedge clk);
        #1;

        tests_run++;

        if (done !== 1'b0) begin

            $error(
                "done remained asserted longer than expected"
            );

            tests_failed++;

        end

        else begin

            $display(
                "[PASS] done returned low"
            );

        end

        // ====================================================================
        // TEST 13: RANDOMIZED END-TO-END GEMM
        //
        // 50 complete host -> accelerator -> host operations.
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 13: RANDOMIZED END-TO-END GEMM");
        $display("========================================");

        for (trial = 0; trial < 50; trial = trial + 1) begin

            zero_matrices();

            for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

                for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                    matrix_a[r][c] =
                        $signed($urandom_range(0, 255));

                    matrix_b[r][c] =
                        $signed($urandom_range(0, 255));

                end

            end

            run_gemm(
                $sformatf(
                    "Random GEMM %0d",
                    trial
                )
            );

        end

        // ====================================================================
        // TEST 14: RESET AFTER ACTIVITY
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 14: RESET AFTER ACTIVITY");
        $display("========================================");

        @(negedge clk);

        rst_n = 1'b0;

        #1;

        tests_run++;

        if (busy !== 1'b0) begin

            $error(
                "busy not cleared by reset"
            );

            tests_failed++;

        end

        else begin

            $display(
                "[PASS] busy cleared by reset"
            );

        end

        tests_run++;

        if (done !== 1'b0) begin

            $error(
                "done not cleared by reset"
            );

            tests_failed++;

        end

        else begin

            $display(
                "[PASS] done cleared by reset"
            );

        end

        // ====================================================================
        // FINAL SUMMARY
        // ====================================================================

        $display("");
        $display("========================================");
        $display("          TPU TEST SUMMARY");
        $display("========================================");

        $display(
            "Tests run    : %0d",
            tests_run
        );

        $display(
            "Tests passed : %0d",
            tests_run - tests_failed
        );

        $display(
            "Tests failed : %0d",
            tests_failed
        );

        $display("========================================");

        if (tests_failed == 0) begin

            $display("");
            $display("      ALL TPU TESTS PASSED");
            $display("");

        end

        else begin

            $fatal(
                1,
                "%0d TPU tests failed",
                tests_failed
            );

        end

        $finish;

    end

endmodule