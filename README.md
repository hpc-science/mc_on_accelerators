# Monte Carlo Cross Section Lookup Kernel 

This repository contains several implementations of the continuous energy Monte Carlo cross section lookup kernel.


## 1. CSL Full Implementation

This implementaiton is a full 2D decomposition across the many processing elements (PEs) of the WSE-2. It is written in CSL (with host-side code written in python). It is available in the `csl/full_implementation` directory. Instructions for compiling and running this code are available in the README.md file in that directory.

In general, this is the version of the code that was used to support the publication on this topic linked below.

## 2. CSL Row Reduction Study

This implementation, available in the `csl/row_reduce_study` folder, only performs a single phase of the larger cross section lookup algorithm. It constitutes an alternative to the "round-robin" algorithm used in the full implementation. Generally, the row-reduce implementation was found to be slower than the round-robin, so it was not extended to include support for the sorting or load balancing phases of the simulation. Instructions for compiling and running this code are available in the `csl/row_reduce_study` directory.

This code was used to generate results for the "alternative algorithms" section of the publication on this topic linked below.

## 3. CUDA Full Implementation

To give performance results on the WSE-2 a baseline for comparison, a CUDA version for use with NVIDIA GPUs was also implemented. Those code is available in the `cuda` folder. Instructions for compiling and running this code are available in the README.md file in that directory.

This code was used to generate results on an A100 GPU for the baseline comparison section of the publication on this topic linked below.

## Reference

Those wishing to reference this work or code within this repository should use the following reference:

## Source Code Contributors

- John Tramm
- Bryce Allen
