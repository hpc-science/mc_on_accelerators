#!/bin/bash
set -xe

TILE_WIDTH=6
TILE_HEIGHT=11
WIDTH=125
HEIGHT=90
GP=111
P=30
N=2
XS=5

csctl get jobs
python app-compile.py --mode=cs2 device_layout.csl --particles=$P --width=$WIDTH --gp=$GP --height=$HEIGHT --nuclides=$N --tile-width=$TILE_WIDTH --tile-height=$TILE_HEIGHT --xs=$XS
csctl get jobs
python host_code.py --mode=cs2
