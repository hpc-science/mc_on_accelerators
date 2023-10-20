#!/bin/bash
set -xe

# This script is where you can edit any run parameters (e.g., # particles, nuclides) such that
# both the compiled CSL and host python will know about the settings.

# This script also compiles the CSL and calls the host python code to launch the kernel on
# simulator or device.

# Define PE grid dimensions
WIDTH=1 # The width also affects the number of nuclides. E.g., total nuclides = WIDTH * NNUCLIDES)
HEIGHT=1

# Define Cross Section lookup parameters
NPARTICLES=5 # starting particles per PE
NNUCLIDES=2 # this is the number of nuclides per PE in a row (e.g., total nuclides = WIDTH * NNUCLIDES)
NGRIDPOINTS=2 # Number of gridpoints per PE (e.g., the number of gridpoints in that energy band)
NXS=1 # Number of XS lookups. Should always be 5.

TILE_WIDTH=1
TILE_HEIGHT=1

# This increases the length of sevearl device particle buffers, so as
# to accomodate the case where the random distribution of particles results in
# some PEs needing to process more particles than they started with. A value 
# of "1" indicates that perfect load balancing is applied, while a value of
# e.g., "5" means that a single PE has room to store 5x more particles than it
# started with.
PARTICLE_BUFFER_MULTIPLIER=3

# The 7 and 2 here are fixed numbers needed by CSL if using memcpy.
FABRIC_WIDTH=$(($WIDTH * $TILE_WIDTH + 7))
FABRIC_HEIGHT=$(($HEIGHT * $TILE_HEIGHT + 2))

# Invoke csl compiler, passing above parameters
cslc device_layout.csl --fabric-dims=${FABRIC_WIDTH},${FABRIC_HEIGHT} --fabric-offsets=4,1 --params=width:${WIDTH},height:${HEIGHT},n_starting_particles_per_pe:${NPARTICLES},n_nuclides:${NNUCLIDES},n_gridpoints_per_nuclide:${NGRIDPOINTS},n_xs:${NXS},particle_buffer_multiplier:${PARTICLE_BUFFER_MULTIPLIER},tile_width:${TILE_WIDTH},tile_height:${TILE_HEIGHT} --memcpy --channels=1 -o out

# Invoke python code (above parameters will be read from compiled CSL file by python library)
cs_python host_code.py --name out
