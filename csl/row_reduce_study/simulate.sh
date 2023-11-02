#!/usr/bin/env bash

set -e

WIDTH=5

N_PARTICLES_PER_ROW=10
N_XS=5
N_GRID_POINTS_PER_NUCLIDE=10

if [ $# -gt 0 ]; then
  N_PARTICLES_PER_ROW=$1
  shift
  if [ $# -gt 0 ]; then
    WIDTH=$1
  fi
  shift
fi

# one nuclide per column
N_NUCLIDES=$((1 * WIDTH))

HEIGHT=1

N_BATCHES=1

# nuclide density: N_NUCLIDES
# nuclide energy grids: N_NUCLIDES * N_GRID_POINTS_PER_NUCLIDE
# nuclide xs data: N_NUCLIDES * N_GRID_POINTS_PER_NUCLIDE * N_XS
nuclide_data_count=$((N_NUCLIDES * (1 + N_GRID_POINTS_PER_NUCLIDE * (N_XS + 1))))
# particles: N_PARTICLES_PER_ROW
# particles_xs: N_PARTICLES_PER_ROW * N_XS
bytes_per_pe=$(((nuclide_data_count / WIDTH + (N_XS + 1) * N_PARTICLES_PER_ROW) * 4))
echo "estimated KB per PE: $((bytes_per_pe / 1024))"

set -x
python compile.py --mode singularity ./layout.csl \
    --height $HEIGHT --width $WIDTH \
    --particles-per-row $N_PARTICLES_PER_ROW \
    --particle-batches $N_BATCHES \
    --xs $N_XS --nuclides $N_NUCLIDES \
    --grid-points-per-nuclide $N_GRID_POINTS_PER_NUCLIDE
cs_python host.py --mode singularity
set +x
