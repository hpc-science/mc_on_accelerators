# Full XS Lookup Implementation in CSL

## Requirements

There are two ways to run this code:

1. Via the Cerebras SDK hardware simulator
2. On Cerebras hardware (e.g., a CS-2 system featuring a WSE-2 chip).

## Source Code Layout

A basic CSL program like this is composed of 3 basic files + a script to compile/run things:

1. `host_cody.py`: This is host-side python code that will utilize the Cerebras API to transfer data arrays and invoke kernels on the device.
2. `device_code.csl`: This is the device kernel-code (written in the CSL language) that will be run on the device. Functions within this file can be invoked remotely by the host via the python API.
3. `device_layout.csl`: This CSL file defines which PE's we will use on the CS2 machine. The CS2 has 850k PEs and the user is allowed to run on a user-specified subset of them.
4. `compile_and_run_on_device.sh`: This is a short host script that invokes the CSL compiler and launches the simulator job using the Cerebras SDK. In the "multi PE" version, this is where you can specify the number of PEs you want to run on via the "width" and "height" grid parameters.

## How to adjust settings

- Presently, the repository is configured for running on a real CS-2 system, not the simulator. If wishing to run on the simulator, some lines of code in the `host_code.py` file will need to be adjusted so the appropriate python library files are loaded.

- To adjust load balancing assumptions, set the `ASSUME_PERFECT_LOAD_BALANCE` variable at the top of the `host_code.py` file.


- To enable validation of outputted results, set the `VALIDATE_RESULTS = True` variable near the top of the `host_code.py` file. Currently, you will also need to uncomment several blocks of code in the file as well to enable full validation. Note that stochastic interpolation should be disabled when validating results, as it utilizes CSL's hardware PRNG, so results cannot be reproduced on the host for validation. Also note that validation can take significant additional time, as reference lookups are generated in serial by the host python code, which is tens of thousands (or perhaps millions) of times slower than the CS-2 kernel itself. Validation of full machine problems (without stochastic interpolation) can take up to an hour.

- To enable the use of stochastic interpolation, FP16 interpolation division, and DSD vectorization in the inner loop, adjust the following settings in the `device_code.csl` file:

```
var enable_diffusion : bool = true;
const use_rng_interp: bool = true;
const use_dsd_interp: bool = true;
```

- Other settings are set via arguments passed to the compiler. These can be easily adjusted in the `compile_and_run_on_hardware.sh` script. The following variables are accessible there:

	- `TILE_WIDTH`: The number of tiles in the x-dimension
	- `TILE_HEIGHT`: The number of tiles in the y-dimension
	- `WIDTH`: The number of PEs in the x-dimension inside each tile
	- `HEIGHT`: The number of PEs in the y-dimension inside each tile
	- `GP`: The number of energy gridpoints per nuclide
	- `P`: The number of initial starting particles per PE. Total number of particles is this value multiplied by `WIDTH x HEIGHT x TILE_WIDTH x TILE_HEIGHT`
	- `N`: The number of nuclides per PE. The total number of unique nuclides in the simulation is this value multiplied by `WIDTH`. Nuclide data is replicated across tiles.
	- `XS`: The number of cross section reaction channels per energy gridpoint.

## How to run

A full-machine optimized example can be run on the Cerebras simulator as: `./compile_and_run_on_hardware.sh`. A number of files are outputted after running, and can be cleaned up via the `./cleanup.sh` script. The script will follow multiple stages. In the first, a compile job will be submitted to the CS-2, which may require queueing. In the second stage, the job is actually run, which also may require queueing.
