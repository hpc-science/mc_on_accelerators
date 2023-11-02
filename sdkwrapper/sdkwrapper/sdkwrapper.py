import json
import os.path


class SdkWrapper(object):
    def __init__(self, mode, app_name, artifact_file_path=None,
                 artifact_dir=None):
        self.mode = mode
        self.is_app_sdk = mode.is_app_sdk
        if self.is_app_sdk:
            from cerebras_appliance.sdk import (
                    SdkRuntime,
            )

            from cerebras_appliance.pb.sdk.sdk_common_pb2 import (
                    MemcpyDataType,
                    MemcpyOrder,
            )
            from cerebras_appliance.sdk.debug_util import debug_util
        else:
            from cerebras.sdk.runtime.sdkruntimepybind import (
                SdkRuntime,
                MemcpyDataType,
                MemcpyOrder,
            )  # pylint: disable=no-name-in-module
            from cerebras.sdk.debug.debug_util import debug_util

        # for use by application; SdkRuntime can be internal only, since this
        # class can construct the runner instance
        self.MemcpyDataType = MemcpyDataType
        self.MemcpyOrder = MemcpyOrder

        if self.is_app_sdk:
            # For the app sdk, the "name" is the artifact id or a path to a
            # json file containing the artifact id
            if app_name is None:
                app_name = mode.default_artifact_file
            if app_name.endswith(".json"):
                with open(app_name, "r") as f:
                    hashstr = json.load(f)
                    app_name = hashstr
        elif app_name is None:
            app_name = "out"

        if artifact_dir is None:
            artifact_dir = mode.get_default_artifact_dir(app_name)
        compile_data_path = os.path.join(artifact_dir, "out.json")
        # Get matrix dimensions from compile metadata
        with open(compile_data_path, encoding="utf-8") as json_file:
            self.compile_data = json.load(json_file)

        self.debug_mod = None

        if self.is_app_sdk:
            self.runner = SdkRuntime(app_name, simulator=mode.is_sim)
            if mode.is_sim:
                self.debug_mod = debug_util(app_name, self.runner)
        else:
            self.runner = SdkRuntime(app_name)
            self.debug_mod = debug_util(app_name)

    def start(self):
        if self.is_app_sdk:
            self.runner.start()
        else:
            self.runner.load()
            self.runner.run()

    def launch(self, *args, **kwargs):
        return self.runner.launch(*args, **kwargs)

    def stop(self):
        return self.runner.stop()

    def get_id(self, *args):
        return self.runner.get_id(*args)

    def memcpy_h2d(self, *args, **kwargs):
        return self.runner.memcpy_h2d(*args, **kwargs)

    def memcpy_d2h(self, *args, **kwargs):
        return self.runner.memcpy_d2h(*args, **kwargs)
