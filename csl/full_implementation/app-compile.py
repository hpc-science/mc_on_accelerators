import os
import json
import argparse

from cerebras_appliance.sdk import SdkCompiler


ARTIFACT_ID_FILENAME = "artifact_id.json"


def parse_args():
    # three modes are possible - the singularity simulator, the app sdk
    # simulator, and the app sdk submit to real CS-2
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["appsim", "cs2"],
                        help="Run mode (, appsim, cs2)",
                        default="appsim")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("layout_file")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--nuclides", type=int)
    parser.add_argument("--gp", type=int)
    parser.add_argument("--tile-width", type=int)
    parser.add_argument("--tile-height", type=int)
    parser.add_argument("--xs", type=int, default=5)
    return parser.parse_args()


def get_params_str(param_dict):
    return ",".join([":".join([k, str(v)]) for k, v in param_dict.items()])


def compile(src_dir, master_layout_file, params_dict, cs2=False, debug_only=False):
    if cs2:
        fab_dims = "757,996"
    else:
        fab_dims = (str(params_dict["width"] + 7)
                    + "," + str(params_dict["height"] + 2))
    # assume memcpy will always be used
    fab_offsets = "4,1"

    compile_data = { "params": params_dict }
    params_str = get_params_str(compile_data["params"])
    cmd = f"--arch wse2 --fabric-dims={fab_dims} --fabric-offsets={fab_offsets} --params={params_str} -o latest --memcpy --channels=1"

    print("compile options: ", cmd)

    if debug_only:
        return None

    compiler = SdkCompiler()
    artifact_id = compiler.compile(src_dir, master_layout_file, cmd)
    os.mkdir(artifact_id)
    with open(os.path.join(artifact_id, "out.json"), "w") as f:
        json.dump(compile_data, f)

    return artifact_id


def main():
    args = parse_args()

    # TODO: make these command line option in parse_args
    params_dict = dict(
        width=args.width,
        height=args.height,
        n_starting_particles_per_pe=args.particles,
        n_nuclides=args.nuclides,
        n_gridpoints_per_nuclide=args.gp,
        n_xs=args.xs,
        particle_buffer_multiplier=3,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )

    src_dir, layout_file = os.path.split(args.layout_file)
    if not src_dir:
        src_dir = "./"
    artifact_id = compile(src_dir, layout_file, params_dict,
                          cs2=(args.mode == "cs2"), debug_only=args.debug)
    if args.debug:
        print("Debug mode, not executing compiler")
        return
    print("compile artifact:", artifact_id)

    print(f"dumping artifact_id to file {ARTIFACT_ID_FILENAME}")
    with open(ARTIFACT_ID_FILENAME, "w") as write_file:
        json.dump(artifact_id, write_file)


if __name__ == "__main__":
    main()
