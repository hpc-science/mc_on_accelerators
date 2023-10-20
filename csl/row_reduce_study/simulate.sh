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
FAB_DIMS="$((WIDTH + 7)),$((HEIGHT+2))"
FAB_OFFSETS="4,1"

# nuclide density: N_NUCLIDES
# nuclide energy grids: N_NUCLIDES * N_GRID_POINTS_PER_NUCLIDE
# nuclide xs data: N_NUCLIDES * N_GRID_POINTS_PER_NUCLIDE * N_XS
nuclide_data_count=$((N_NUCLIDES * (1 + N_GRID_POINTS_PER_NUCLIDE * (N_XS + 1))))
# particles: N_PARTICLES_PER_ROW
# particles_xs: N_PARTICLES_PER_ROW * N_XS
bytes_per_pe=$(((nuclide_data_count / WIDTH + (N_XS + 1) * N_PARTICLES_PER_ROW) * 4))
echo "estimated KB per PE: $((bytes_per_pe / 1024))"

set -x
cslc ./layout.csl --fabric-dims=$FAB_DIMS --fabric-offsets=$FAB_OFFSETS \
  --params=N_PARTICLES_PER_ROW:$N_PARTICLES_PER_ROW,N_XS:$N_XS,N_NUCLIDES:$N_NUCLIDES,N_GRID_POINTS_PER_NUCLIDE:$N_GRID_POINTS_PER_NUCLIDE,width:$WIDTH \
  -o out --memcpy --channels 1
cs_python host.py --name out
set +x
