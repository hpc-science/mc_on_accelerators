# CUDA Cross Section Lookup Implementation

This code implements a cross section lookup kernel in CUDA to serve as a baseline for comparison to the CSL version. It has many similarities to the CUDA implementation of XSBench, but there are a few small differences that were needed to ensure an apples-to-apples comparison.

A number of optimizations were made to ensure the kernel was running as efficiently as possible.

## Settings

Settings to control which optimizations are being used are available by commenting out a number of macro definitions in the main.cu file:

- `USE_F16`: Replaces the default 32-bit division operation in the interpolation routine with a 16-bit operation.
- `USE_LCG`: Uses stochastic interpolation via a simple custom LCG PRNG.
- `USE_CURAND`: Uses stochastic interpolation via the CUDA CURAND optimized library.
- `USE_SORT`: Enables particle pre-sorting by energy. This is by far the most impactful optimzation.
- `USE_UEG`: Generates and uses a unionized energy grid for removing the need for per-nuclide binary searches, though at the cost of significant more memory usage.

## Compiling

Compilation can be done via `make` (with a default SM target of 80). This can be edited in the Makefile.

## Running

Running can be done by `./a.out <n_particles>`, where `n_particles` is typically set at 100000000 (100 million) so as to ensure the GPU is well saturated.
