"""
Usage:

    compile.py --mode=singularity|appsim|cs2 \
            --width=10 --height=7 -n 100 \
            layout.csl
"""
import os
import argparse

from sdkwrapper import compile, add_standard_compile_args


def parse_args():
    # three modes are possible - the singularity simulator, the app sdk
    # simulator, and the app sdk submit to real CS-2
    parser = argparse.ArgumentParser()
    add_standard_compile_args(parser)
    parser.add_argument("--width", type=int, default=10,
                        help="fabric width (not including memcpy overhead)")
    parser.add_argument("--height", type=int, default=2,
                        help="fabric height (not including memcpy overhead)")
    # TODO: add real app specific args
    parser.add_argument("-n", type=int, default=1024,
                        help="size of data array")
    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: add app specific args
    params_dict = dict(
        width=args.width,
        height=args.height,
        n=args.n,
    )

    print("mode ", args.mode)

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
