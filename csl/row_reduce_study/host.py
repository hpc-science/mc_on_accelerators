#!/usr/bin/env python

import argparse
import numpy as np

from sdkwrapper import SdkWrapper, SdkMode, add_mode_arg


class LCG(object):
    m = 4294967291
    a = 1588635695
    c = 12345

    def __init__(self, start_seed):
        self.seed = start_seed

    def next_float(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed / float(self.m)


class XSTable(object):
    def __init__(self, n_nuclides, n_grid_points_per_nuclide, n_xs, lcg):
        self.n_nuclides = n_nuclides
        self.n_grid_points_per_nuclide = n_grid_points_per_nuclide
        self.n_xs = n_xs
        self.lcg = lcg

        self.nuclide_energy_grids = np.empty(
            (n_nuclides, n_grid_points_per_nuclide), dtype=np.float32
        )
        self.nuclide_densities = np.empty(n_nuclides, dtype=np.float32)
        self.nuclide_xs_data = np.empty(
            (n_nuclides, n_grid_points_per_nuclide, n_xs), dtype=np.float32
        )
        self._generate_random_data()

    def _generate_random_data(self):
        for nuc in range(self.n_nuclides):
            for gridp in range(self.n_grid_points_per_nuclide):
                self.nuclide_energy_grids[nuc, gridp] = self.lcg.next_float()
                for xs in range(self.n_xs):
                    self.nuclide_xs_data[nuc, gridp, xs] = self.lcg.next_float()
            self.nuclide_densities[nuc] = self.lcg.next_float()
        # sort along second axis, i.e. within each nuclide
        self.nuclide_energy_grids.sort(axis=1)


def parse_args():
    # three modes are possible - the singularity simulator, the app sdk
    # simulator, and the app sdk submit to real CS-2
    parser = argparse.ArgumentParser()
    add_mode_arg(parser)
    #parser.add_argument("--cmaddr", help="IP:port for CS system")
    parser.add_argument("--name",
                        help="singularity name OR artifact id OR artifact id json file path")
    parser.add_argument("--seed", help="Path to file ending in .json, or id itself",
                        type=int, default=42)
    return parser.parse_args()


def run_app(seed, mode, name):
    sdk = SdkWrapper(mode, name)

    # compile parameters
    n_particles = int(sdk.compile_data["params"]["N_PARTICLES_PER_ROW"])
    n_particle_batches = int(sdk.compile_data["params"]["N_PARTICLE_BATCHES"])
    n_xs = int(sdk.compile_data["params"]["N_XS"])
    n_grid_points_per_nuclide = int(sdk.compile_data["params"]["N_GRID_POINTS_PER_NUCLIDE"])
    n_nuclides = int(sdk.compile_data["params"]["N_NUCLIDES"])

    # Number of PEs in program
    width = int(sdk.compile_data["params"]["width"])
    height = int(sdk.compile_data["params"]["height"])
    n_pe = width * height

    assert n_nuclides % width == 0
    n_nuclides_per_col = n_nuclides // width

    lcg = LCG(seed)
    xstable = XSTable(n_nuclides, n_grid_points_per_nuclide, n_xs, lcg)
    particles = np.zeros(n_particles, dtype=np.float32)

    # TODO: use a parallel random algorithm
    for i in range(n_particles):
        particles[i] = lcg.next_float()

    # rng = np.random.default_rng(seed)
    # particles = rng.random(n_particles, dtype=np.float32)

    # Get symbols
    P_symbol = sdk.get_id("P")
    Pxs_symbol = sdk.get_id("Pxs")
    D_symbol = sdk.get_id("D")
    XS_symbol = sdk.get_id("XS")
    NRG_symbol = sdk.get_id("NRG")

    # Load and run the program
    sdk.start()

    bytes_per_pe = 0

    # Broadcast particles to each PE. In real application, each row would
    # have different particles for that energy band, but here we use the
    # same particles everywhere for benchmarking
    sdk.memcpy_h2d(
        P_symbol,
        np.broadcast_to(particles, (height, width, n_particles)).copy(),
        0,
        0,
        width,
        height,
        n_particles,
        streaming=False,
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # 32 bit values
    bytes_per_pe += n_particles * 4

    # distribute same nuclide specific density across each row
    sdk.memcpy_h2d(
        D_symbol,
        np.broadcast_to(xstable.nuclide_densities,
                        (height, xstable.nuclide_densities.size)).copy(),
        0,
        0,
        width,
        height,
        xstable.nuclide_densities.size // width,
        streaming=False,
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    bytes_per_pe += xstable.nuclide_densities.size // width * 4

    # distribute same nuclide specific xs data across each row
    xsdata_bcast_shape = [height] + list(xstable.nuclide_xs_data.shape)
    sdk.runner.memcpy_h2d(
        XS_symbol,
        np.broadcast_to(xstable.nuclide_xs_data, xsdata_bcast_shape).copy(),
        0,
        0,
        width,
        height,
        xstable.nuclide_xs_data.size // width,
        streaming=False,
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    bytes_per_pe += xstable.nuclide_xs_data.size // width * 4

    # distribute same nuclide specific energy grid data across each row
    grids_bcast_shape = [height] + list(xstable.nuclide_energy_grids.shape)
    sdk.runner.memcpy_h2d(
        NRG_symbol,
        np.broadcast_to(xstable.nuclide_energy_grids,
                        grids_bcast_shape).copy(),
        0,
        0,
        width,
        height,
        xstable.nuclide_energy_grids.size // width,
        streaming=False,
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    bytes_per_pe += xstable.nuclide_energy_grids.size // width * 4

    # Launch the main compute function on device
    sdk.launch("compute", nonblock=False)

    # Copy computed xs data back from right PE(s)
    particles_xs = np.empty((height, n_particles, n_xs), dtype=np.float32)
    sdk.memcpy_d2h(
        particles_xs,
        Pxs_symbol,
        width-1,
        0,
        1,
        height,
        n_particles * n_xs,
        streaming=False,
        order=sdk.MemcpyOrder.ROW_MAJOR,
        data_type=sdk.MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    bytes_per_pe += n_particles * n_xs * 4

    # Copy timestamps from device
    # Note: for some reason these need to be 32 bit, even though on device
    # they are 16
    def get_device_ts(n_per_pe, name):
        """Get numpy array of timestamps (as uint32) from a device array
        of the specified name, with dimensions [n_pe, n_per_pe]. Note that
        Device array has 3 elements per time stamp because of the odd format
        used on device, this converts them to be more convenient to manipulate
        in Python."""
        elements_per_pe = 3 * n_per_pe
        raw_arr = np.zeros((n_pe, n_per_pe, 3), dtype=np.uint32)
        # use signed for converted host copy so if timestamps are unexpected
        # order, we can see the negatives when doing subtraction.
        ts_arr = np.zeros((n_pe, n_per_pe), dtype=np.int64)
        symbol = sdk.get_id(name)
        if symbol is None:
            raise ValueError(f"Symbol not found: '{name}'")
        sdk.memcpy_d2h(raw_arr, symbol, 0, 0, width, height, elements_per_pe,
                       streaming=False, order=sdk.MemcpyOrder.ROW_MAJOR,
                       data_type=sdk.MemcpyDataType.MEMCPY_16BIT,
                       nonblock=False)
        # convert to numpy array of timestamps (note: not sure how to vectorize
        # this when it needs to take 3 elements at once)
        for pe in range(n_pe):
            for ts_i in range(n_per_pe):
                ts_arr[pe, ts_i] = ts_device2host(raw_arr[pe, ts_i])
        return ts_arr

    ts_start = get_device_ts(1, "ts_start")
    ts_done = get_device_ts(1, "ts_done")
    ts_xsstart = get_device_ts(n_particle_batches, "ts_xsstart")
    ts_xsdone = get_device_ts(n_particle_batches, "ts_xsdone")
    ts_reducestart = get_device_ts(n_particle_batches, "ts_reducestart")
    ts_reducedone = get_device_ts(n_particle_batches, "ts_reducedone")

    # Stop the program
    sdk.stop()

    # each row should be the same, so just calculate once
    particles_xs_host = np.zeros((n_particles, n_xs), dtype=np.float32)
    host_calculate(xstable, particles, particles_xs_host)

    max_print_xs = min(5, n_xs)
    max_print_p = min(5, n_particles)
    print("bytes per pe ", bytes_per_pe / 1024, " K")
    print("xs host       ",
          np.squeeze(particles_xs_host[:max_print_xs, :max_print_p]))
    print("xs device[ 0] ",
          np.squeeze(particles_xs[0, :max_print_xs, :max_print_p]))
    print("xs device[-1] ",
          np.squeeze(particles_xs[-1, :max_print_xs, :max_print_p]))

    for i in range(height):
        # Note: squeezing host array handles case where particles
        # per row = 1, so per row arrays are just single dim, n_xs len
        np.testing.assert_allclose(np.squeeze(particles_xs[i, :, :]),
                                   np.squeeze(particles_xs_host),
                                   atol=0.01, rtol=0)
    print("SUCCESS!")

    # Total time, single timestamp for each PE
    tmin, tmax, tavg = device_cycle_stats(ts_start, ts_done)
    print("[dev] tot cycles (min/max/avg): ",
          tmin[0], tmax[0], tavg[0])

    # Time for first xs calculation, where there is no concurrency with reduce
    xsmin, xsmax, xsavg = device_cycle_stats(ts_xsstart[0:1], ts_xsdone[0:1])
    print("[dev] xs cycles [0] (min/max/avg): ",
          xsmin[0], xsmax[0], xsavg[0])

    if n_particle_batches > 1:
        # Check if reduce runs concurrently with xs after the first xs, so need
        # to offset the arrays in the second per-pe / time dimension
        rxmin, rxmax, _ = device_cycle_stats(ts_reducestart[:, :-1],
                                             ts_xsstart[:, 1:], avg=False)
        print("[dev] reduce->xs start delay cycles (min/max): ",
              rxmin, rxmax)

        # also check end times, we expect xs to be done before reduce
        xrmin, xrmax, _ = device_cycle_stats(ts_xsdone[:, 1:],
                                             ts_reducedone[:, :-1], avg=False)
        print("[dev] xs->reduce done delay cycles (min/max): ",
              xrmin, xrmax)
    else:
        # with only one batch, xs always finishes first before the single
        # reduce phase can start
        rxmin, rxmax, _ = device_cycle_stats(ts_xsstart, ts_reducestart,
                                             avg=False)
        print("[dev] xs->reduce start delay cycles (min/max): ",
              rxmin, rxmax)

        # also check end times, we expect xs to be done before reduce
        xrmin, xrmax, _ = device_cycle_stats(ts_xsdone, ts_reducedone,
                                             avg=False)
        print("[dev] xs->reduce done delay cycles (min/max): ",
              xrmin, xrmax)


def host_calculate(table, particles, particles_xs):
    for i in range(particles.size):
        energy = particles[i]
        for j in range(table.n_nuclides):
            density = table.nuclide_densities[j]
            lower_idx = iterative_binary_search(
                table.nuclide_energy_grids, table.n_grid_points_per_nuclide,
                j, energy
            )
            e_low = table.nuclide_energy_grids[j, lower_idx]
            e_high = table.nuclide_energy_grids[j, lower_idx + 1]
            f = (e_high - energy) / (e_high - e_low)
            for k in range(table.n_xs):
                xs_low = table.nuclide_xs_data[j, lower_idx, k]
                xs_high = table.nuclide_xs_data[j, lower_idx + 1, k]
                # NOTE: 0 index is particle energy, so xs are offset by 1
                particles_xs[i, k] += density * (xs_high - f * (xs_high - xs_low))


def iterative_binary_search(nrg_grid, n_grid_points_per_nuclide, nuclide, energy):
    lower_idx = 0
    upper_idx = n_grid_points_per_nuclide - 1
    length = upper_idx - lower_idx

    while length > 1:
        current_idx = lower_idx + (length // 2)

        if nrg_grid[nuclide, current_idx] > energy:
            upper_idx = current_idx
        else:
            lower_idx = current_idx
        length = upper_idx - lower_idx

    return lower_idx


def trace_stats(debug_mod, width, height):
    n_pe = width * height
    X_OFFSET = 4
    Y_OFFSET = 1
    xs_min_cycles = None
    xs_max_cycles = 0
    xs_avg_cycles = 0.0
    com_min_cycles = None
    com_max_cycles = 0
    com_avg_cycles = 0.0
    total_min_cycles = None
    total_max_cycles = 0
    total_avg_cycles = 0.0
    for x in range(X_OFFSET, X_OFFSET + width):
        for y in range(Y_OFFSET, Y_OFFSET + height):
            trace_output = debug_mod.read_trace(x, y, "trace")
            cycles = trace_output[1] - trace_output[0]
            xs_avg_cycles += cycles / float(n_pe)
            if xs_min_cycles is None or cycles < xs_min_cycles:
                xs_min_cycles = cycles
            if cycles > xs_max_cycles:
                xs_max_cycles = cycles
            cycles = trace_output[2] - trace_output[1]
            com_avg_cycles += cycles / float(n_pe)
            if com_min_cycles is None or cycles < com_min_cycles:
                com_min_cycles = cycles
            if cycles > com_max_cycles:
                com_max_cycles = cycles
            cycles = trace_output[2] - trace_output[0]
            total_avg_cycles += cycles / float(n_pe)
            if total_min_cycles is None or cycles < total_min_cycles:
                total_min_cycles = cycles
            if cycles > total_max_cycles:
                total_max_cycles = cycles
    return (xs_min_cycles, xs_max_cycles, xs_avg_cycles,
            com_min_cycles, com_max_cycles, com_avg_cycles,
            total_min_cycles, total_max_cycles, total_avg_cycles)


def device_cycle_stats(ts_start, ts_done, avg=True):
    """Calculate stats across PEs between arrays of start and end times.
    Assumes axis=0 is for PEs"""
    assert ts_start.shape == ts_done.shape
    cycle_times = ts_done - ts_start
    # compute stats across PEs
    min_cycles = np.min(cycle_times, axis=0)
    max_cycles = np.max(cycle_times, axis=0)
    if avg:
        avg_cycles = np.average(cycle_times, axis=0)
    else:
        avg_cycles = np.zeros_like(min_cycles)
    return min_cycles, max_cycles, avg_cycles


#####################################################################
# Timestamp decoding functions
#####################################################################
def ts_device2host(words):
    """Convert 3 word device timestamp to a single number. Converts
    to signed (int64), so subtraction operations don't result in
    misleading results."""
    words = words.astype(np.int64)
    return words[0] + (words[1] << 16) + (words[2] << 32)


def main():
    args = parse_args()
    run_app(seed=args.seed, mode=args.mode, name=args.name)


if __name__ == "__main__":
    main()
