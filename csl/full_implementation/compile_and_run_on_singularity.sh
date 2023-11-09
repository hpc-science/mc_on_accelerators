#!/bin/bash
set -xe

# This script is where you can edit any run parameters (e.g., # particles, nuclides) such that
# both the compiled CSL and host python will know about the settings.

# This script also compiles the CSL and calls the host python code to launch the kernel on
# simulator or device.

# Define PE grid dimensions
WIDTH=2 # The width also affects the number of nuclides. E.g., total nuclides = WIDTH * NNUCLIDES)
HEIGHT=2

# Define Cross Section lookup parameters
NPARTICLES=5 # starting particles per PE
NNUCLIDES=1 # this is the number of nuclides per PE in a row (e.g., total nuclides = WIDTH * NNUCLIDES)
NGRIDPOINTS=10 # Number of gridpoints per PE (e.g., the number of gridpoints in that energy band)
NXS=5 # Number of XS lookups. Should always be 5.

TILE_WIDTH=1
TILE_HEIGHT=1

# This increases the length of sevearl device particle buffers, so as
# to accomodate the case where the random distribution of particles results in
# some PEs needing to process more particles than they started with. A value 
# of "1" indicates that perfect load balancing is applied, while a value of
# e.g., "5" means that a single PE has room to store 5x more particles than it
# started with.
PARTICLE_BUFFER_MULTIPLIER=3

MODE=singularity

# Invoke csl compiler, passing above parameters
python compile.py --mode $MODE ./device_layout.csl \
    --height $HEIGHT --width $WIDTH \
    --tile-height $TILE_HEIGHT --tile-width $TILE_WIDTH \
    --particles $NPARTICLES \
    --xs $NXS --nuclides $NNUCLIDES \
    --gp $NGRIDPOINTS \
    --particle-buffer-multiplier $PARTICLE_BUFFER_MULTIPLIER

# Invoke python code (above parameters will be read from compiled CSL file by python library)
cs_python host_code.py --mode $MODE
