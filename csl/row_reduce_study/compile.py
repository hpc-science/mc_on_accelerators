import os
import json
import argparse
import sys
from pathlib import Path


def parse_args():
    from sdkwrapper import add_standard_compile_args

    # three modes are possible - the singularity simulator, the app sdk
    # simulator, and the app sdk submit to real CS-2
    parser = argparse.ArgumentParser()
    add_standard_compile_args(parser)
    parser.add_argument("-w", "--width", type=int, default=10)
    parser.add_argument("-e", "--height", type=int, default=2)
    parser.add_argument("-n", "--nuclides", type=int, default=10)
    parser.add_argument("-p", "--particles-per-row", type=int, default=100)
    parser.add_argument("-b", "--particle-batches", type=int, default=1)
    parser.add_argument("-x", "--xs", type=int, default=5)
    parser.add_argument("-g", "--grid-points-per-nuclide", type=int, default=10)
    return parser.parse_args()


def main():
    from sdkwrapper import compile
    args = parse_args()

    # On CS-2, max size is (750, 994)

    # TODO: make these command line option in parse_args
    params_dict = dict(
        width=args.width,
        height=args.height,
        N_NUCLIDES=args.nuclides,
        N_PARTICLES_PER_ROW=args.particles_per_row,
        N_PARTICLE_BATCHES=args.particle_batches,
        N_XS=args.xs,
        N_GRID_POINTS_PER_NUCLIDE=args.grid_points_per_nuclide,
    )

    print("mode           ", args.mode)
    print("size           ", (args.width, args.height))
    print("total particles", args.height * args.particles_per_row)
    print("nuclides       ", args.nuclides)
    print("particles / row", args.particles_per_row)
    print("xs             ", args.xs)
    print("grid pts / nucl", args.grid_points_per_nuclide)

    src_dir, layout_file = os.path.split(args.layout_file)
    if not src_dir:
        src_dir = "./"
    artifact_id = compile(args.mode, src_dir, layout_file, params_dict,
                          debug_only=args.debug)
    if args.debug:
        print("Debug mode, not executing compiler")
        return
    print("compile artifact:", artifact_id)


if __name__ == "__main__":
    main()
