from enum import Enum


class SdkMode(Enum):
    CS2 = 1
    SIM_APP = 2
    SIM_SINGULARITY = 3

    @property
    def is_app_sdk(self):
        return (self in (self.CS2, self.SIM_APP))

    @property
    def is_sim(self):
        return (self in (self.SIM_APP, self.SIM_SINGULARITY))

    @property
    def default_artifact_file(self):
        """Used by compiler and sdk wrapper for consistent naming."""
        if self.is_sim:
            return "artifact_id_sim.json"
        else:
            return "artifact_id_cs2.json"

    def get_default_artifact_dir(self, artifact_id_or_name):
        """Used by compiler and sdk wrapper for consistent naming. Allow build
        for app sim and cs2 to coexist."""
        simcode = ""
        if self == SdkMode.CS2:
            simcode = "_cs2"
        elif self == SdkMode.SIM_APP:
            simcode = "_sim"
        return artifact_id_or_name + simcode


def _mode_arg(s):
    if isinstance(s, SdkMode):
        return s

    sup = s.upper()
    try:
        return SdkMode[sup]
    except KeyError:
        pass

    if sup == "SINGULARITY":
        return SdkMode.SIM_SINGULARITY
    elif sup == "APPSIM":
        return SdkMode.SIM_APP
    else:
        raise ValueError("Unknown mode value: " + s)


def add_mode_arg(parser):
    # three modes are possible - the singularity simulator, the app sdk
    # simulator, and the app sdk submit to real CS-2
    parser.add_argument("--mode", type=_mode_arg,
                        help="Run mode (singularity, appsim, cs2)",
                        default=SdkMode.SIM_SINGULARITY)
    return parser
