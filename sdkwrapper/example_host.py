#!/usr/bin/env python

import argparse
import numpy as np

from sdkwrapper import SdkWrapper, add_mode_arg


def parse_args():
    parser = argparse.ArgumentParser()
    # adds --mode singularity|appsim|cs2 arg that maps to SdkMode enum, passed
    # to SdkWrapper constructor
    add_mode_arg(parser)
    # Note: if name is blank, the artifact_id will be read from the standard json file
    # according to mode, or for singularity, the "out" folder will be used as the name
    parser.add_argument("--name",
                        help="singularity name OR artifact id OR artifact id json file path")
    # TODO: add app specific args
    return parser.parse_args()


def run_app(mode, name, n):
    # constructs appropriate SdkRuntime for the specified mode, and reads compile data
    # from the out.json produced by the sdkwrapper.compile function, called in
    # example-compiler.py
    sdk = SdkWrapper(mode, name)

    # compile parameters
    n = int(sdk.compile_data["params"]["N"])

    # Number of PEs in program
    width = int(sdk.compile_data["params"]["width"])
    height = int(sdk.compile_data["params"]["height"])
    n_pe = width * height

    indata = np.zeros(n, dtype=np.float32)

    # Get symbols
    indata_symbol = sdk.get_id("indata")
    outdata_symbol = sdk.get_id("outdata")

    # Load and run the program, using methods appropriate to SdkRuntime type
    sdk.start()

    # Broadcast particles to each PE in the row
    sdk.memcpy_h2d(
        indata_symbol,
        indata,
        0,
        0,
        width,
        height,
        n / n_pe,
        streaming=False,
        # Note: must use sdk member for Memcpy* enums, imports are different
        # depending on mode
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # Launch the main compute function on device (passed to underlying SdkRuntime)
    sdk.launch("compute", nonblock=False)

    # copy result back to host
    outdata = np.empty(n_data, dtype=np.float32)
    sdk.memcpy_d2h(
        outdata,
        outdata_symbol,
        0,
        0,
        width,
        height,
        n / n_pe,
        streaming=False,
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # Stop the program
    sdk.stop()

    if mode.is_sim:
        # TODO: do something with debug_mod
        pass


def main():
    args = parse_args()
    run_app(mode=args.mode, name=args.name, n=args.n)


if __name__ == "__main__":
    main()
