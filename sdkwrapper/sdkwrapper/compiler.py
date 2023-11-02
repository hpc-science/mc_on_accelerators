
import os
import json
import subprocess

from .mode import SdkMode, add_mode_arg


MEMCPY_OFFSET_X = 4
MEMCPY_OFFSET_Y = 1
MEMCPY_WIDTH = 7
MEMCPY_HEIGHT = 2
CS2_DIMS = (757, 996)


def get_max_usable_size(memcpy=True):
    if memcpy:
        return (CS2_DIMS[0] - MEMCPY_WIDTH,
                CS2_DIMS[1] - MEMCPY_HEIGHT)
    else:
        return CS2_DIMS


def add_standard_compile_args(parser):
    add_mode_arg(parser)
    parser.add_argument("--debug", action="store_true",
                        help="Don't execute, just show command")
    parser.add_argument("--artifact-file",
                        help="alternate json file path")
    parser.add_argument("layout_file")
    return parser


def get_params_str(param_dict):
    return ",".join([":".join([k, str(v)]) for k, v in param_dict.items()])


def compile(mode, src_dir, master_layout_file, params_dict, debug_only=False,
            cs2_dims=CS2_DIMS, memcpy=True, channels=1,
            artifact_id_filename=None, artifact_dir=None, out_name="out",
            arch="wse2"):
    """
    Output directory is either artifact id (for CS2 and app sim) or out_name
    for singularity, and will contain "out.json" file with build parameters,
    which is read by SdkWrapper.

    params_dict must contain 'width' and 'height', containing required size
    not including memcpy overhead.

    The artifact id, which is needed when running, will be saved to a json file,
    by default "artifact_id_{cs2,sim}.json" depending on mode. This is read by
    default (also based on mode) by the sdkwrapper.
    """
    width = params_dict["width"]
    height = params_dict["height"]
    if memcpy:
        width += MEMCPY_WIDTH
        height += MEMCPY_HEIGHT

    if mode == SdkMode.CS2:
        fab_dims = ",".join(str(n) for n in cs2_dims)
        if width > cs2_dims[0] or height > cs2_dims[1]:
            raise ValueError(f"Requested size does not fit on CS-2: ({width}, {height}) > {cs2_dims}")
    else:
        fab_dims = ",".join(str(n) for n in (width, height))

    # assume memcpy will always be used
    if memcpy:
        fab_offsets = "4,1"
    else:
        fab_offsets = "0,0"

    compile_data = {"params": params_dict}
    params_str = get_params_str(compile_data["params"])
    cmd = f"--fabric-dims={fab_dims} --fabric-offsets={fab_offsets} --params={params_str} --channels={channels}"
    if memcpy:
        cmd += " --memcpy"

    if mode.is_app_sdk:
        cmd = f"--arch {arch} -o latest {cmd}"
    else:
        layout_file_path = os.path.join(src_dir, master_layout_file)
        cmd = f"cslc -o {out_name} {cmd} {layout_file_path}"

    print("compile command: ", cmd)

    if debug_only:
        return None

    if mode.is_app_sdk:
        from cerebras_appliance.sdk import SdkCompiler
        compiler = SdkCompiler()
        artifact_id = compiler.compile(src_dir, master_layout_file, cmd)
        if artifact_dir is None:
            artifact_dir = mode.get_default_artifact_dir(artifact_id)
        if os.path.exists(artifact_dir):
            print(f"ERR: artifact directory '{artifact_dir}' exists")
            return None
        os.mkdir(artifact_dir)
        with open(os.path.join(artifact_dir, "out.json"), "w") as f:
            json.dump(compile_data, f)
        if artifact_id_filename is None:
            artifact_id_filename = mode.default_artifact_file
        print(f"dumping artifact_id to file {artifact_id_filename}")
        with open(artifact_id_filename, "w") as write_file:
            json.dump(artifact_id, write_file)
    else:
        try:
            stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                             shell=True)
        except subprocess.CalledProcessError as e:
            print("ERR: cslc exit code: ", e.returncode)
            print("cslc error:")
            print(e.output.decode("utf8"))
            return None
        else:
            output = stdout.decode("utf8").strip()
            if output:
                print("cslc output:")
                print(output)
            artifact_id = out_name

    return artifact_id
