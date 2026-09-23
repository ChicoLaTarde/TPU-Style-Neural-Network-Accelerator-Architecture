// ============================================================================
// systolic_array_tb.sv
//
// Self-checking testbench for systolic_array.sv
//
// Tests:
//   1. Asynchronous reset
//   2. clear_acc
//   3. Identity matrix
//   4. Basic positive matrix multiplication
//   5. Signed matrix multiplication
//   6. Zero matrix
//   7. Negative identity matrix
//   8. INT8 maximum values
//   9. INT8 minimum / maximum combinations
//  10. Sparse matrices
//  11. enable = 0 / pipeline hold
//  12. clear_acc priority
//  13. Back-to-back matrix operations
//  14. Randomized matrix multiplication
//  15. Multiple randomized trials
//
// Architecture under test:
//
//                   B weights
//                     |
//                     v
//
// A activations --> [PE] --> [PE] --> [PE]
//                   |        |        |
//                   v        v        v
//                  [PE] --> [PE] --> [PE]
//                   |        |        |
//                   v        v        v
//                  [PE] --> [PE] --> [PE]
//
// Activations move LEFT -> RIGHT.
// Weights move TOP -> BOTTOM.
//
// IMPORTANT:
//
// Inputs are skewed by the testbench:
//
//     activation_in[i] = A[i][t-i]
//     weight_in[j]     = B[t-j][j]
//
// This ensures matching k values meet at PE[i][j].
//
// ============================================================================

`timescale 1ns/1ps

module systolic_array_tb;

    // ========================================================================
    // Parameters
    // ========================================================================

    localparam int ARRAY_SIZE = 3;
    localparam int DATA_W     = 8;
    localparam int ACC_W      = 32;

    localparam time CLK_PERIOD = 10ns;

    // Number of cycles required to inject an NxN tile:
    //
    //     last injection = (N-1) + (N-1)
    //
    // therefore:
    //
    //     2N - 1 cycles
    //
    localparam int INJECT_CYCLES =
        (2 * ARRAY_SIZE) - 1;

    // After the final injection, allow the final values to propagate
    // through the remainder of the array.
    localparam int DRAIN_CYCLES =
        (2 * ARRAY_SIZE) - 2;

    // ========================================================================
    // DUT signals
    // ========================================================================

    logic clk;
    logic rst_n;

    logic enable;
    logic clear_acc;

    logic signed [DATA_W-1:0]
        activation_in [0:ARRAY_SIZE-1];

    logic signed [DATA_W-1:0]
        weight_in [0:ARRAY_SIZE-1];

    logic signed [ACC_W-1:0]
        result [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // ========================================================================
    // Matrices
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
    integer k;
    integer t;
    integer trial;

    // ========================================================================
    // DUT
    // ========================================================================

    systolic_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_W     (DATA_W),
        .ACC_W      (ACC_W)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),

        .enable         (enable),
        .clear_acc      (clear_acc),

        .activation_in  (activation_in),
        .weight_in      (weight_in),

        .result         (result)
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
    // Clear input signals
    // ========================================================================

    task automatic zero_inputs();

        integer i;

        begin

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                activation_in[i] = '0;
                weight_in[i]     = '0;

            end

        end

    endtask

    // ========================================================================
    // Clear matrix storage
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
    // Golden matrix multiplication
    //
    // expected = matrix_a x matrix_b
    // ========================================================================

    task automatic calculate_expected();

        integer i;
        integer j;
        integer x;

        logic signed [(2*DATA_W)-1:0] product;

        begin

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    expected[i][j] = '0;

                    for (x = 0; x < ARRAY_SIZE; x = x + 1) begin

                        product =
                            matrix_a[i][x] *
                            matrix_b[x][j];

                        expected[i][j] =
                            expected[i][j] + product;

                    end

                end

            end

        end

    endtask

    // ========================================================================
    // Clear DUT accumulators
    // ========================================================================

    task automatic clear_array();

        begin

            @(negedge clk);

            enable    = 1'b0;
            clear_acc = 1'b1;

            zero_inputs();

            @(posedge clk);
            #1;

            @(negedge clk);

            clear_acc = 1'b0;

        end

    endtask

    // ========================================================================
    // Drive one skewed systolic cycle
    //
    // At global time t:
    //
    // activation row i:
    //
    //     k = t - i
    //
    // weight column j:
    //
    //     k = t - j
    //
    // ========================================================================

    task automatic drive_systolic_cycle(
        input integer cycle
    );

        integer i;
        integer index;

        begin

            zero_inputs();

            // ---------------------------------------------------------------
            // Activation injection
            // ---------------------------------------------------------------

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                index = cycle - i;

                if ((index >= 0) &&
                    (index < ARRAY_SIZE)) begin

                    activation_in[i] =
                        matrix_a[i][index];

                end

            end

            // ---------------------------------------------------------------
            // Weight injection
            // ---------------------------------------------------------------

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                index = cycle - i;

                if ((index >= 0) &&
                    (index < ARRAY_SIZE)) begin

                    weight_in[i] =
                        matrix_b[index][i];

                end

            end

        end

    endtask

    // ========================================================================
    // Run matrix multiplication
    // ========================================================================

    task automatic run_matrix();

        integer cycle;

        begin

            calculate_expected();

            clear_array();

            // ---------------------------------------------------------------
            // Inject skewed matrix data.
            // ---------------------------------------------------------------

            for (cycle = 0;
                 cycle < INJECT_CYCLES;
                 cycle = cycle + 1) begin

                @(negedge clk);

                enable = 1'b1;

                drive_systolic_cycle(cycle);

                @(posedge clk);
                #1;

            end

            // ---------------------------------------------------------------
            // Drain the systolic pipeline.
            //
            // Inject zeros while existing values continue moving.
            // ---------------------------------------------------------------

            for (cycle = 0;
                 cycle < DRAIN_CYCLES;
                 cycle = cycle + 1) begin

                @(negedge clk);

                enable = 1'b1;

                zero_inputs();

                @(posedge clk);
                #1;

            end

            @(negedge clk);

            enable = 1'b0;

            zero_inputs();

        end

    endtask

    // ========================================================================
    // Check entire result matrix
    // ========================================================================

    task automatic check_matrix(
        input string test_name
    );

        integer i;
        integer j;

        begin

            $display("");
            $display("----------------------------------------");
            $display("Checking: %s", test_name);
            $display("----------------------------------------");

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    tests_run++;

                    if (result[i][j] !== expected[i][j]) begin

                        $error(
                            "%s PE[%0d][%0d] FAILED: expected=%0d actual=%0d",
                            test_name,
                            i,
                            j,
                            expected[i][j],
                            result[i][j]
                        );

                        tests_failed++;

                    end

                    else begin

                        $display(
                            "[PASS] PE[%0d][%0d] expected=%0d actual=%0d",
                            i,
                            j,
                            expected[i][j],
                            result[i][j]
                        );

                    end

                end

            end

        end

    endtask

    // ========================================================================
    // Test reset state
    // ========================================================================

    task automatic check_all_zero(
        input string test_name
    );

        integer i;
        integer j;

        begin

            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin

                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin

                    tests_run++;

                    if (result[i][j] !== '0) begin

                        $error(
                            "%s PE[%0d][%0d] expected zero, got %0d",
                            test_name,
                            i,
                            j,
                            result[i][j]
                        );

                        tests_failed++;

                    end

                end

            end

        end

    endtask

    // ========================================================================
    // Main test sequence
    // ========================================================================

    initial begin

        tests_run    = 0;
        tests_failed = 0;

        rst_n     = 1'b0;
        enable    = 1'b0;
        clear_acc = 1'b0;

        zero_inputs();
        zero_matrices();

        // ====================================================================
        // TEST 1
        // ASYNCHRONOUS RESET
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 1: RESET");
        $display("========================================");

        #2;

        check_all_zero(
            "Asynchronous reset"
        );

        repeat (2) @(posedge clk);

        @(negedge clk);

        rst_n = 1'b1;

        // ====================================================================
        // TEST 2
        // IDENTITY MATRIX
        //
        // A x I = A
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 2: IDENTITY MATRIX");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] =  8'sd1;
        matrix_a[0][1] =  8'sd2;
        matrix_a[0][2] =  8'sd3;

        matrix_a[1][0] =  8'sd4;
        matrix_a[1][1] =  8'sd5;
        matrix_a[1][2] =  8'sd6;

        matrix_a[2][0] =  8'sd7;
        matrix_a[2][1] =  8'sd8;
        matrix_a[2][2] =  8'sd9;

        matrix_b[0][0] = 8'sd1;
        matrix_b[1][1] = 8'sd1;
        matrix_b[2][2] = 8'sd1;

        run_matrix();

        check_matrix(
            "Identity matrix"
        );

        // ====================================================================
        // TEST 3
        // BASIC POSITIVE MATRIX
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 3: BASIC POSITIVE MATRIX");
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

        run_matrix();

        check_matrix(
            "Basic positive matrix"
        );

        // ====================================================================
        // TEST 4
        // SIGNED MATRIX
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 4: SIGNED MATRIX");
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

        run_matrix();

        check_matrix(
            "Signed matrix"
        );

        // ====================================================================
        // TEST 5
        // ZERO MATRIX
        //
        // A x 0 = 0
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 5: ZERO MATRIX");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd20;
        matrix_a[0][1] = -8'sd40;
        matrix_a[0][2] = 8'sd60;

        matrix_a[1][0] = -8'sd10;
        matrix_a[1][1] = 8'sd30;
        matrix_a[1][2] = -8'sd50;

        matrix_a[2][0] = 8'sd100;
        matrix_a[2][1] = -8'sd100;
        matrix_a[2][2] = 8'sd127;

        // B remains zero.

        run_matrix();

        check_matrix(
            "Zero matrix"
        );

        // ====================================================================
        // TEST 6
        // NEGATIVE IDENTITY
        //
        // A x (-I) = -A
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

        run_matrix();

        check_matrix(
            "Negative identity"
        );

        // ====================================================================
        // TEST 7
        // INT8 MAXIMUM
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 7: INT8 MAXIMUM");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                matrix_a[r][c] = 8'sd127;
                matrix_b[r][c] = 8'sd127;

            end

        end

        run_matrix();

        check_matrix(
            "INT8 maximum"
        );

        // Each result should be:
        //
        // 3 * 127 * 127
        //
        // = 48387

        // ====================================================================
        // TEST 8
        // INT8 MINIMUM
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 8: INT8 MINIMUM");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                matrix_a[r][c] = -8'sd128;
                matrix_b[r][c] = -8'sd128;

            end

        end

        run_matrix();

        check_matrix(
            "INT8 minimum"
        );

        // ====================================================================
        // TEST 9
        // MINIMUM x MAXIMUM
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 9: INT8 MIN x MAX");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                matrix_a[r][c] = -8'sd128;
                matrix_b[r][c] =  8'sd127;

            end

        end

        run_matrix();

        check_matrix(
            "INT8 min x max"
        );

        // ====================================================================
        // TEST 10
        // SPARSE MATRIX
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 10: SPARSE MATRIX");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd5;
        matrix_a[1][2] = -8'sd7;
        matrix_a[2][1] = 8'sd11;

        matrix_b[0][2] = 8'sd3;
        matrix_b[1][0] = -8'sd4;
        matrix_b[2][1] = 8'sd6;

        run_matrix();

        check_matrix(
            "Sparse matrix"
        );

        // ====================================================================
        // TEST 11
        // ENABLE HOLD
        //
        // Verify that disabling the array freezes every PE.
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 11: ENABLE HOLD");
        $display("========================================");

        zero_matrices();

        matrix_a[0][0] = 8'sd2;
        matrix_b[0][0] = 8'sd3;

        run_matrix();

        // Result is now established.

        @(negedge clk);

        enable = 1'b0;

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            activation_in[r] = 8'sd100;
            weight_in[r]     = 8'sd100;

        end

        repeat (3) @(posedge clk);

        #1;

        check_matrix(
            "Enable hold"
        );

        zero_inputs();

        // ====================================================================
        // TEST 12
        // CLEAR ACCUMULATOR
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 12: CLEAR ACCUMULATOR");
        $display("========================================");

        clear_array();

        check_all_zero(
            "Clear accumulator"
        );

        // ====================================================================
        // TEST 13
        // CLEAR PRIORITY OVER COMPUTE
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 13: CLEAR PRIORITY");
        $display("========================================");

        @(negedge clk);

        enable    = 1'b1;
        clear_acc = 1'b1;

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

            activation_in[r] = 8'sd127;
            weight_in[r]     = 8'sd127;

        end

        @(posedge clk);
        #1;

        check_all_zero(
            "Clear priority"
        );

        @(negedge clk);

        enable    = 1'b0;
        clear_acc = 1'b0;

        zero_inputs();

        // ====================================================================
        // TEST 14
        // BACK-TO-BACK MATRIX OPERATIONS
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 14: BACK-TO-BACK OPERATIONS");
        $display("========================================");

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            matrix_a[r][r] = 8'sd2;
            matrix_b[r][r] = 8'sd3;
        end

        run_matrix();

        check_matrix(
            "Back-to-back operation 1"
        );

        zero_matrices();

        for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
            matrix_a[r][r] = -8'sd4;
            matrix_b[r][r] =  8'sd5;
        end

        run_matrix();

        check_matrix(
            "Back-to-back operation 2"
        );

        // ====================================================================
        // TEST 15
        // RANDOMIZED MATRIX MULTIPLICATION
        //
        // 50 complete randomized GEMMs.
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 15: RANDOMIZED GEMM");
        $display("========================================");

        for (trial = 0; trial < 50; trial = trial + 1) begin

            zero_matrices();

            for (r = 0; r < ARRAY_SIZE; r = r + 1) begin

                for (c = 0; c < ARRAY_SIZE; c = c + 1) begin

                    // Generate full signed INT8 range.
                    matrix_a[r][c] =
                        $signed($urandom_range(0, 255));

                    matrix_b[r][c] =
                        $signed($urandom_range(0, 255));

                end

            end

            run_matrix();

            check_matrix(
                $sformatf(
                    "Random GEMM trial %0d",
                    trial
                )
            );

        end

        // ====================================================================
        // TEST 16
        // RESET AFTER ACTIVITY
        // ====================================================================

        $display("");
        $display("========================================");
        $display("TEST 16: RESET AFTER ACTIVITY");
        $display("========================================");

        @(negedge clk);

        rst_n = 1'b0;

        #1;

        check_all_zero(
            "Reset after activity"
        );

        // ====================================================================
        // FINAL SUMMARY
        // ====================================================================

        $display("");
        $display("========================================");
        $display("      SYSTOLIC ARRAY TEST SUMMARY");
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
            $display("   ALL SYSTOLIC ARRAY TESTS PASSED");
            $display("");

        end

        else begin

            $fatal(
                1,
                "%0d systolic-array tests failed",
                tests_failed
            );

        end

        $finish;

    end

endmodule