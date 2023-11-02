# sdkwrapper

Python package to simplify compiling and running on singularity, cs2, and app simulator
with minimal modification to code, i.e. provides a common interface.

# Setup

For the app sdk, simply add the sdkwrapper top directory to PYTHONPATH:

    export PYTHONPATH=/path/to/mc-on-cerebras/sdkwrapper

For singularity, the cs\_python wrapper provided by Cerebras only binds /tmp and the
current working directory, so if in an app subdirectory of the repository, the
sdkwrapper directory is inaccessible to the container.

Workarounds:

1. Use modified cs\_python script:

    cd /path/to/cerebras-sdk
    mv -iv cs_python cs_python.dist
    ln -s /path/to/mc-on-cerebras/sdkwrapper/cs_python .

2. Copy the module subdirectory to the app subdirectory. Note that the path has changed
   since the initial commit, it was moved to a subdirectory to better organize project
   files like setup.py and README.md and the actual python module:

    cd my-xs-impl
    cp -R ../sdkwrapper/sdkwrapper .
    # OR
    rsync -va ../sdkwrapper/sdkwrapper/ ./sdkwrapper/

I prefer (1), to avoid the possibility of modifying the wrong version of
sdkwrapper and loosing changes.

# Usage

See [example\_compile.py](example_compile.py).

For singularity with a locally installed sdk (Note: use host python, not cs\_python):

    $ python example_compile.py --mode singularity \
        --width 23 --height 7 -n 1024 \
         layout.csl

This will call cslc, which creates an "out/out.json" file containing the build
parameters, that are read by the SdkWrapper class when running:

    $ cs_python example_host.py --mode singularity

The steps are the same for running on a CS-2 login node, but use mode "appsim" or "cs2",
and use "python" from the virtualenv instead of "cs\_python".

Note: the compilation must be done separately for each, and they will use
different directories so they can coexist without interference.
