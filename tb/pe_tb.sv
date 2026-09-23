`timescale 1ns/1ps

// ============================================================================
// pe_tb.sv
//
// Self-checking testbench for pe.sv
//
// Tests:
//   1. Reset
//   2. Basic positive MAC
//   3. Positive x negative
//   4. Negative x positive
//   5. Negative x negative
//   6. Zero activation
//   7. Zero weight
//   8. INT8 maximum values
//   9. INT8 minimum values
//  10. INT8 min x max
//  11. Multi-cycle accumulation
//  12. enable = 0 behavior
//  13. clear_acc behavior
//  14. clear_acc priority over enable
//  15. Activation forwarding
//  16. Weight forwarding
//  17. Randomized MAC testing
//  18. INT32 positive overflow / wraparound
//  19. INT32 negative overflow / wraparound
//
// Expected arithmetic behavior:
//   signed two's-complement wraparound
// ============================================================================

module pe_tb;

    localparam int DATA_W = 8;
    localparam int ACC_W  = 32;

    localparam time CLK_PERIOD = 10ns;

    // ========================================================================
    // DUT signals
    // ========================================================================

    logic clk;
    logic rst_n;

    logic enable;
    logic clear_acc;

    logic signed [DATA_W-1:0] activation_in;
    logic signed [DATA_W-1:0] weight_in;

    logic signed [DATA_W-1:0] activation_out;
    logic signed [DATA_W-1:0] weight_out;

    logic signed [ACC_W-1:0] accumulator;

    // ========================================================================
    // Reference model
    // ========================================================================

    logic signed [ACC_W-1:0] expected_acc;

    integer tests_run;
    integer tests_failed;

    // ========================================================================
    // DUT
    // ========================================================================

    pe #(
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),

        .enable         (enable),
        .clear_acc      (clear_acc),

        .activation_in  (activation_in),
        .weight_in      (weight_in),

        .activation_out (activation_out),
        .weight_out     (weight_out),

        .accumulator    (accumulator)
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
    // Check helper
    // ========================================================================

    task automatic check_value(
        input string name,
        input logic signed [ACC_W-1:0] actual,
        input logic signed [ACC_W-1:0] expected
    );

        tests_run++;

        if (actual !== expected) begin

            $error(
                "%s FAILED: expected %0d, got %0d",
                name,
                expected,
                actual
            );

            tests_failed++;

        end
        else begin

            $display(
                "[PASS] %-35s expected=%0d actual=%0d",
                name,
                expected,
                actual
            );

        end

    endtask

    // ========================================================================
    // Forwarding checker
    // ========================================================================

    task automatic check_forwarding(
        input logic signed [DATA_W-1:0] expected_activation,
        input logic signed [DATA_W-1:0] expected_weight
    );

        tests_run++;

        if ((activation_out !== expected_activation) ||
            (weight_out     !== expected_weight)) begin

            $error(
                "Forwarding FAILED: A expected=%0d got=%0d, W expected=%0d got=%0d",
                expected_activation,
                activation_out,
                expected_weight,
                weight_out
            );

            tests_failed++;

        end
        else begin

            $display(
                "[PASS] Forwarding: activation=%0d weight=%0d",
                activation_out,
                weight_out
            );

        end

    endtask

    // ========================================================================
    // Perform one MAC
    // ========================================================================

    task automatic do_mac(
        input logic signed [DATA_W-1:0] activation,
        input logic signed [DATA_W-1:0] weight,
        input string test_name
    );

        logic signed [(2*DATA_W)-1:0] product;

        begin

            @(negedge clk);

            enable        = 1'b1;
            clear_acc     = 1'b0;

            activation_in = activation;
            weight_in     = weight;

            product = activation * weight;

            // Match fixed-width hardware arithmetic.
            expected_acc = expected_acc + product;

            @(posedge clk);
            #1;

            check_value(
                test_name,
                accumulator,
                expected_acc
            );

            check_forwarding(
                activation,
                weight
            );

        end

    endtask

    // ========================================================================
    // Clear accumulator
    // ========================================================================

    task automatic clear_accumulator();

        begin

            @(negedge clk);

            enable    = 1'b0;
            clear_acc = 1'b1;

            @(posedge clk);
            #1;

            expected_acc = '0;

            check_value(
                "Clear accumulator",
                accumulator,
                expected_acc
            );

            @(negedge clk);

            clear_acc = 1'b0;

        end

    endtask

    // ========================================================================
    // Main test sequence
    // ========================================================================

    integer i;

    logic signed [DATA_W-1:0] random_activation;
    logic signed [DATA_W-1:0] random_weight;

    initial begin

        // --------------------------------------------------------------------
        // Initialization
        // --------------------------------------------------------------------

        tests_run    = 0;
        tests_failed = 0;

        rst_n         = 1'b0;

        enable        = 1'b0;
        clear_acc     = 1'b0;

        activation_in = '0;
        weight_in     = '0;

        expected_acc  = '0;

        // ====================================================================
        // TEST 1: RESET
        // ====================================================================

        $display("\n========================================");
        $display("TEST 1: RESET");
        $display("========================================");

        repeat (2) @(posedge clk);

        #1;

        check_value(
            "Accumulator after reset",
            accumulator,
            32'sd0
        );

        if ((activation_out !== 0) ||
            (weight_out !== 0)) begin

            $error("Reset forwarding outputs FAILED");
            tests_failed++;

        end
        else begin

            $display("[PASS] Forwarding outputs reset to zero");

        end

        tests_run++;

        // Release reset.

        @(negedge clk);
        rst_n = 1'b1;

        // ====================================================================
        // TEST 2: BASIC POSITIVE MAC
        // ====================================================================

        $display("\n========================================");
        $display("TEST 2: BASIC POSITIVE MAC");
        $display("========================================");

        do_mac(
            8'sd5,
            8'sd4,
            "5 * 4"
        );

        // Expected:
        //
        // 5 * 4 = 20

        // ====================================================================
        // TEST 3: POSITIVE x NEGATIVE
        // ====================================================================

        clear_accumulator();

        do_mac(
            8'sd7,
            -8'sd3,
            "7 * -3"
        );

        // Expected:
        //
        // -21

        // ====================================================================
        // TEST 4: NEGATIVE x POSITIVE
        // ====================================================================

        clear_accumulator();

        do_mac(
            -8'sd8,
            8'sd6,
            "-8 * 6"
        );

        // Expected:
        //
        // -48

        // ====================================================================
        // TEST 5: NEGATIVE x NEGATIVE
        // ====================================================================

        clear_accumulator();

        do_mac(
            -8'sd9,
            -8'sd7,
            "-9 * -7"
        );

        // Expected:
        //
        // 63

        // ====================================================================
        // TEST 6: ZERO ACTIVATION
        // ====================================================================

        clear_accumulator();

        do_mac(
            8'sd0,
            8'sd100,
            "0 * 100"
        );

        // ====================================================================
        // TEST 7: ZERO WEIGHT
        // ====================================================================

        clear_accumulator();

        do_mac(
            8'sd100,
            8'sd0,
            "100 * 0"
        );

        // ====================================================================
        // TEST 8: INT8 MAXIMUM
        //
        // 127 * 127 = 16129
        // ====================================================================

        clear_accumulator();

        do_mac(
            8'sd127,
            8'sd127,
            "INT8_MAX * INT8_MAX"
        );

        // ====================================================================
        // TEST 9: INT8 MINIMUM
        //
        // -128 * -128 = 16384
        // ====================================================================

        clear_accumulator();

        do_mac(
            -8'sd128,
            -8'sd128,
            "INT8_MIN * INT8_MIN"
        );

        // ====================================================================
        // TEST 10: INT8 MIN x INT8 MAX
        //
        // -128 * 127 = -16256
        // ====================================================================

        clear_accumulator();

        do_mac(
            -8'sd128,
            8'sd127,
            "INT8_MIN * INT8_MAX"
        );

        // ====================================================================
        // TEST 11: MULTI-CYCLE ACCUMULATION
        //
        // 2*4 + (-3)*5 + 6*(-2)
        //
        // = 8 - 15 - 12
        // = -19
        // ====================================================================

        $display("\n========================================");
        $display("TEST 11: MULTI-CYCLE ACCUMULATION");
        $display("========================================");

        clear_accumulator();

        do_mac(
            8'sd2,
            8'sd4,
            "Accumulation step 1"
        );

        do_mac(
            -8'sd3,
            8'sd5,
            "Accumulation step 2"
        );

        do_mac(
            8'sd6,
            -8'sd2,
            "Accumulation step 3"
        );

        check_value(
            "Final accumulated result",
            accumulator,
            -32'sd19
        );

        // ====================================================================
        // TEST 12: ENABLE = 0
        //
        // Accumulator and forwarding registers should hold their values.
        // ====================================================================

        $display("\n========================================");
        $display("TEST 12: ENABLE HOLD");
        $display("========================================");

        @(negedge clk);

        enable        = 1'b0;
        clear_acc     = 1'b0;

        activation_in = 8'sd100;
        weight_in     = 8'sd100;

        @(posedge clk);
        #1;

        check_value(
            "Accumulator holds when enable=0",
            accumulator,
            expected_acc
        );

        check_forwarding(
            8'sd6,
            -8'sd2
        );

        // ====================================================================
        // TEST 13: CLEAR ACCUMULATOR
        // ====================================================================

        $display("\n========================================");
        $display("TEST 13: CLEAR ACCUMULATOR");
        $display("========================================");

        clear_accumulator();

        // ====================================================================
        // TEST 14: CLEAR PRIORITY OVER ENABLE
        // ====================================================================

        $display("\n========================================");
        $display("TEST 14: CLEAR PRIORITY");
        $display("========================================");

        // First create a nonzero accumulator.

        do_mac(
            8'sd10,
            8'sd10,
            "Setup accumulator"
        );

        @(negedge clk);

        enable        = 1'b1;
        clear_acc     = 1'b1;

        activation_in = 8'sd50;
        weight_in     = 8'sd50;

        @(posedge clk);
        #1;

        expected_acc = '0;

        check_value(
            "clear_acc overrides MAC",
            accumulator,
            32'sd0
        );

        // Forwarding should STILL occur because enable is asserted.

        check_forwarding(
            8'sd50,
            8'sd50
        );

        @(negedge clk);

        enable    = 1'b0;
        clear_acc = 1'b0;

        // ====================================================================
        // TEST 15: MAXIMUM POSITIVE/NEGATIVE FORWARDING
        // ====================================================================

        $display("\n========================================");
        $display("TEST 15: FORWARDING EDGE VALUES");
        $display("========================================");

        do_mac(
            8'sd127,
            -8'sd128,
            "Forward INT8 boundaries"
        );

        // ====================================================================
        // TEST 16: RANDOMIZED MAC TESTING
        // ====================================================================

        $display("\n========================================");
        $display("TEST 16: RANDOMIZED TESTING");
        $display("========================================");

        clear_accumulator();

        for (i = 0; i < 100; i = i + 1) begin

            random_activation = $signed($urandom_range(0, 255));
            random_weight     = $signed($urandom_range(0, 255));

            do_mac(
                random_activation,
                random_weight,
                $sformatf("Random MAC %0d", i)
            );

        end

        // ====================================================================
        // TEST 17: REPEATED MAX POSITIVE ACCUMULATION
        //
        // Stress accumulation with large products.
        // ====================================================================

        $display("\n========================================");
        $display("TEST 17: LARGE POSITIVE ACCUMULATION");
        $display("========================================");

        clear_accumulator();

        for (i = 0; i < 32; i = i + 1) begin

            do_mac(
                8'sd127,
                8'sd127,
                $sformatf("Positive stress %0d", i)
            );

        end

        // ====================================================================
        // TEST 18: REPEATED NEGATIVE ACCUMULATION
        // ====================================================================

        $display("\n========================================");
        $display("TEST 18: LARGE NEGATIVE ACCUMULATION");
        $display("========================================");

        clear_accumulator();

        for (i = 0; i < 32; i = i + 1) begin

            do_mac(
                -8'sd128,
                8'sd127,
                $sformatf("Negative stress %0d", i)
            );

        end

        // ====================================================================
        // TEST 19: INT32 POSITIVE OVERFLOW
        //
        // Force the DUT accumulator close to INT32_MAX so that one legal
        // INT8 MAC crosses the signed 32-bit boundary.
        //
        // Expected behavior for this PE:
        //     two's-complement wraparound
        // ====================================================================

        $display("\n========================================");
        $display("TEST 19: INT32 POSITIVE OVERFLOW");
        $display("========================================");

        @(negedge clk);

        enable    = 1'b0;
        clear_acc = 1'b0;

        // Testbench-only hierarchical initialization.
        dut.accumulator = 32'sh7FFF_FFF0;
        expected_acc    = 32'sh7FFF_FFF0;

        do_mac(
            8'sd2,
            8'sd16,
            "Positive INT32 overflow"
        );

        // ====================================================================
        // TEST 20: INT32 NEGATIVE OVERFLOW
        // ====================================================================

        $display("\n========================================");
        $display("TEST 20: INT32 NEGATIVE OVERFLOW");
        $display("========================================");

        @(negedge clk);

        enable    = 1'b0;
        clear_acc = 1'b0;

        dut.accumulator = 32'sh8000_0010;
        expected_acc    = 32'sh8000_0010;

        do_mac(
            -8'sd2,
            8'sd16,
            "Negative INT32 overflow"
        );

        // ====================================================================
        // Final reset test
        // ====================================================================

        $display("\n========================================");
        $display("FINAL RESET TEST");
        $display("========================================");

        @(negedge clk);

        rst_n = 1'b0;

        #1;

        expected_acc = '0;

        check_value(
            "Asynchronous reset accumulator",
            accumulator,
            32'sd0
        );

        tests_run++;

        if ((activation_out !== '0) ||
            (weight_out !== '0)) begin

            $error("Asynchronous reset outputs FAILED");
            tests_failed++;

        end
        else begin

            $display("[PASS] Asynchronous reset outputs");
        end

        // ====================================================================
        // Summary
        // ====================================================================

        $display("\n");
        $display("========================================");
        $display("           PE TEST SUMMARY");
        $display("========================================");
        $display("Tests run    : %0d", tests_run);
        $display("Tests passed : %0d", tests_run - tests_failed);
        $display("Tests failed : %0d", tests_failed);
        $display("========================================");

        if (tests_failed == 0) begin

            $display("");
            $display("  ALL PE TESTS PASSED");
            $display("");

        end
        else begin

            $fatal(
                1,
                "%0d PE tests failed",
                tests_failed
            );

        end

        $finish;

    end

endmodule