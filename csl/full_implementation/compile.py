import os
import json
import argparse


def parse_args():
    from sdkwrapper import add_standard_compile_args

    # three modes are possible - the singularity simulator, the app sdk
    # simulator, and the app sdk submit to real CS-2
    parser = argparse.ArgumentParser()
    add_standard_compile_args(parser)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--nuclides", type=int)
    parser.add_argument("--gp", type=int)
    parser.add_argument("--tile-width", type=int)
    parser.add_argument("--tile-height", type=int)
    parser.add_argument("--xs", type=int, default=5)
    parser.add_argument("--particle-buffer-multiplier", type=int, default=3)
    return parser.parse_args()


def main():
    from sdkwrapper import compile
    args = parse_args()

    params_dict = dict(
        width=args.width,
        height=args.height,
        n_starting_particles_per_pe=args.particles,
        n_nuclides=args.nuclides,
        n_gridpoints_per_nuclide=args.gp,
        n_xs=args.xs,
        particle_buffer_multiplier=args.particle_buffer_multiplier,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )

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
