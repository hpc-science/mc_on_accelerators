#!/bin/bash
set -xe

TILE_WIDTH=1
TILE_HEIGHT=1
WIDTH=50
HEIGHT=62
GP=161
P=20
N=5
XS=5

csctl get jobs
python app-compile.py --mode=cs2 device_layout.csl --particles=$P --width=$WIDTH --gp=$GP --height=$HEIGHT --nuclides=$N --tile-width=$TILE_WIDTH --tile-height=$TILE_HEIGHT --xs=$XS
csctl get jobs
python host_code.py --mode=cs2
