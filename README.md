A parameterized neural-network accelerator implemented in SystemVerilog using a pipelined systolic array for high-throughput matrix multiplication. The design targets neural-network inference workloads and uses parallel multiply-accumulate (MAC) processing elements, on-chip buffers, and a hardware controller for tiled matrix operations.

Architecture

The accelerator is designed around the matrix multiplication operation:

C = A \times B

For neural-network inference, this corresponds to operations such as:

Y = XW + b

where:

* X represents input activations
* W represents neural-network weights
* b represents bias values
* Y represents output activations

The initial architecture uses an 8 × 8 systolic array containing 64 processing elements (PEs).

Each PE performs a multiply-accumulate operation:

psum_{new} = psum_{old} + (activation \times weight)

The design uses INT8 activations and weights with INT32 accumulation.
