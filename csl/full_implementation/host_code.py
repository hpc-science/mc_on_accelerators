#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
import time

from sdkwrapper import SdkWrapper, SdkMode, add_mode_arg

# For Singularlity-Based SDK Simulator:
#from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
#from cerebras.sdk.debug.debug_util import debug_util # For simulator-only debug performance timer

class Particle:
    def __init__(self, energy, xs):
        self.energy = energy
        self.xs = xs

    def __repr__(self) -> str:
        return f"{type(self).__name__}(energy={self.energy}, xs={self.xs})"

# This variable controls whether or not the starting distribution of
# particles will be fully random or controlled so as to
# ensure that after sorting completes after Phase 1, all PE's will
# hold the same number of particles (i.e., ideal load balancing).
# In both cases, the sorting stage will be more or less equally
# random, with particles needing to move around, but the difference
# is that when this value is (true), we have fixed the sampling
# such that particles will end up with an even distribution.
ASSUME_PERFECT_LOAD_BALANCE = False

# Not going to work correctly with stochastic interpolation
VALIDATE_RESULTS = True

#####################################################################
# I/O helper functions
#####################################################################

def print_line():
    buf = ''
    buf = buf.center(80,'=')
    print(buf)

def print_border(message):
    print_line()
    message = message.center(80)
    print(message)
    print_line()

def print_logo():
    print_line()
    print("                   __   __ ___________                 _                        \n"
	"                   \\ \\ / //  ___| ___ \\               | |                       \n"
	"                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
	"                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
	"                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
	"                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n")
    print("Cerebras CS2 Implementation".center(80))
    print()
    print("Developed at Argonne National Laboratory".center(80))
    print()

#####################################################################
# Host/Device Data Transfer Helper Functions
#####################################################################

# These two functions call the cerebras memcpy function for moving
# data arrays to/from device. They are useful for reducing the amount
# of repeated boilerplate. Currently, they just copy the arrays exactly
# to/from the single PE in the problem
def copy_array_host_to_device(arr, name, sdk, width, height, starting_col=0, starting_row=0):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size / (width * height))
    handle = sdk.memcpy_h2d(symbol, arr, starting_col, starting_row, width, height, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.ROW_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return handle

def copy_array_host_to_device_COL(arr, name, sdk, width, height):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size / (width * height))
    handle = sdk.memcpy_h2d(symbol, arr, 0, 0, width, height, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.COL_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return handle

def copy_array_device_to_host(arr, name, sdk, width, height, starting_col=0, starting_row=0):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size / (width * height))
    handle = sdk.memcpy_d2h(arr, symbol, starting_col, starting_row, width, height, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.ROW_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return handle

def copy_array_device_to_host_16_bit(arr, name, sdk, width, height, starting_col=0, starting_row=0):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size / (width * height))
    handle = sdk.memcpy_d2h(arr, symbol, starting_col, starting_row, width, height, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.ROW_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_16BIT, nonblock=False)
    return handle

def copy_array_host_to_device_single_row(arr, name, sdk, width, row, starting_col=0, starting_row=0):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size / width)
    handle = sdk.memcpy_h2d(symbol, arr, starting_col, starting_row + row, width, 1, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.ROW_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return handle

def copy_array_host_to_device_single_column(arr, name, sdk, height, column, starting_col=0, starting_row=0):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size / height)
    handle = sdk.memcpy_h2d(symbol, arr, starting_col + column, starting_row, 1, height, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.ROW_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return handle

def copy_array_host_to_device_single_PE(arr, name, sdk, row, column):
    symbol = sdk.get_id(name)
    elements_per_pe = np.int32(arr.size)
    handle = sdk.memcpy_h2d(symbol, arr, column, row, 1, 1, elements_per_pe, streaming=False,
            order=sdk.MemcpyOrder.ROW_MAJOR, data_type=sdk.MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    return handle

#####################################################################
# Timestamp decoding functions
#####################################################################
def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def subtract_timestamps(words, a, b):
    #return make_u48(words[3:]) - make_u48(words[0:3])
    return make_u48(words[(b*3):(b*3+3)]) - make_u48(words[(a*3):(a*3+3)])

#####################################################################
# Host functions for generating the reference XS solutions.
#####################################################################

# Note: these are not needed for running on the CS2 at all, they are
# just used for checking to ensure the final CS2 solution is correct.

# Binary search for finding the lower bounding index in the energy grid
def iterative_binary_search(arr, quarry):
    n = arr.size
    lowerLimit = 0
    upperLimit = n-1
    examinationPoint = 0
    length = upperLimit - lowerLimit;

    while length > 1 :
        examinationPoint = lowerLimit + ( length // 2 );

        if arr[examinationPoint] > quarry:
            upperLimit = examinationPoint
        else:
            lowerLimit = examinationPoint

        length = upperLimit - lowerLimit

    return lowerLimit


# Performs the same lookup operation that will occur on the Cerebras,
# so as to have a basis for checking for correctness.
def calculate_xs_reference(n_starting_particles_per_pe, n_nuclides, n_gridpoints_per_nuclide, n_xs, nuclide_energy_grids, nuclide_xs_data, densities, particle_e, width, height) :
    particle_xs_expected = np.zeros(n_starting_particles_per_pe * n_xs, dtype=np.float32)
    for p in range(n_starting_particles_per_pe):
        e = particle_e[p]
        for n in range(n_nuclides):
            # Binary search
            low = n*n_gridpoints_per_nuclide
            high = low + n_gridpoints_per_nuclide
            lower_index = iterative_binary_search(nuclide_energy_grids[low:high], e)
 
            # Interpolation factor
            e_lower = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index]
            e_higher = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index + 1]
            f = (e_higher - e) / (e_higher - e_lower)
 
            # Lookup xs data, interpolate, and store to particle array
            for xs in range(n_xs):
                xs_lower = nuclide_xs_data[n * n_gridpoints_per_nuclide * n_xs + lower_index * n_xs + xs]
                xs_higher = nuclide_xs_data[n * n_gridpoints_per_nuclide * n_xs + (lower_index + 1) * n_xs + xs]
                particle_xs_expected[p*n_xs + xs] += densities[n] * (xs_higher - f * (xs_higher - xs_lower) )

    return particle_xs_expected

#####################################################################
# Generation of randomized XS data
#####################################################################

# Allocates and initializes XS data arrays. This function 
# assumes that the same XS data will be copied to all PE's, for
# studying more ideal cases.
def init_xs_data_replicated(n_nuclides, n_gridpoints_per_nuclide, n_xs, seed):
    # This is a 2D array stored in 1D, containing the energy gridpoints that are represented
    # for each nuclide. Has dimensions [n_nuclides, n_gridpoints_per_nuclide]
    nuclide_energy_grids   = np.random.random(n_nuclides * n_gridpoints_per_nuclide).astype(np.float32)

    # This is a 3D array stored in 1D, containing the XS data for each nuclide and energy level.
    # Has dimensions [n_nuclides, n_gridpoints_per_nuclide, n_xs]
    nuclide_xs_data        = np.random.random(n_nuclides * n_gridpoints_per_nuclide * n_xs).astype(np.float32)

    # This is a 1D array containing the isotopic density for each nuclide.
    # Has length n_nuclides.
    densities              = np.random.random(n_nuclides).astype(np.float32)

    # Sort nuclide grids
    for i in range(n_nuclides):
        low = i*n_gridpoints_per_nuclide
        high = low + n_gridpoints_per_nuclide
        nuclide_energy_grids[low:high] = np.sort(nuclide_energy_grids[low:high])

    return nuclide_energy_grids, nuclide_xs_data, densities, seed

# Allocates and initializes XS data arrays. This function generates
# unique XS data for each PE, which is more realistic.
def init_xs_data_unique(n_nuclides, n_gridpoints_per_nuclide, n_xs, width, height):
    # This is a 2D array stored in 1D, containing the energy gridpoints that are represented
    # for each nuclide. Has dimensions [n_nuclides, n_gridpoints_per_nuclide]
    # To simplify our decomposition routines slightly, we assume all nuclides have the same energy grid,
    # which does not impact performance but makes our indexing job a lot easier
    nuclide_energy_grids = np.zeros(n_gridpoints_per_nuclide * n_nuclides * width * height, dtype=np.float32)
    # I.e., we just sample n gridpoints once, then assign the same grid to all nuclides
    #egrid = np.random.random(n_gridpoints_per_nuclide*height).astype(np.float32)

    # Using an even distribution of gridpoints makes more sense, as this is reduces unrealistic variance in the row widths
    egrid = np.linspace(0.0, 1.0, n_gridpoints_per_nuclide * height, dtype=np.float32)
    # Sort nuclide energy grid
    egrid = np.sort(egrid)
    
    # assign replicated nuclide energy grid
    for i in range(n_nuclides * width):
        low = i*n_gridpoints_per_nuclide*height
        high = low + n_gridpoints_per_nuclide * height
        nuclide_energy_grids[low:high] = egrid

    #print("egrid")
    #print(egrid)
    #print("nuclide_energy_grids")
    #print(nuclide_energy_grids)


    # This is a 3D array stored in 1D, containing the XS data for each nuclide and energy level.
    # Has dimensions [n_nuclides, n_gridpoints_per_nuclide, n_xs]
    nuclide_xs_data = np.random.random(n_nuclides * width * n_gridpoints_per_nuclide * height * n_xs).astype(np.float32)

    # This is a 1D array containing the isotopic density for each nuclide.
    # Has length n_nuclides.
    densities              = np.random.random(n_nuclides * width).astype(np.float32)

    return nuclide_energy_grids, nuclide_xs_data, densities

# The below two functions are needed for mapping the generate XS data into the format
# that is expected for the host -> device mempy function. This logic is tricky due to the way
# we are decomposing memory. Specifically, each row represents a single energy band, and
# each column represents a specific nuclide. As each nuclide has many gridpoints within
# each energy level, it becomes a fairly complex mapping between the host array and the device
# one. So, this mappiong and convolutiion function re-organizes the XS data into the correct
# format.
def make_mapper(entries, width, height):
    low = 0
    high = 0
    mapper = {}
    for row in range(height):
        for col in range(width):
            low = high
            high += entries
            mapper[row,col] = [low,high]
            #print("mapper[%d,%d] = [low = %d, high = %d]" % (row, col,low, high))
    return mapper

def convolute_xs_data_for_memcpy_fast(n_nuclides, n_gridpoints_per_nuclide, n_xs, width, height):
    #print("nuclide XS data length = %d" % nuclide_xs_data.size)
    conv = np.zeros(n_nuclides * width * height * n_gridpoints_per_nuclide * n_xs, dtype=np.float32)
    low = 0
    high = 0
    mapper = make_mapper(n_nuclides * n_gridpoints_per_nuclide * n_xs, width, height)
    for col in range(width):
        for row in range(height):
            low = high
            high += n_gridpoints_per_nuclide * n_xs * n_nuclides
            arr = nuclide_xs_data[low:high]
            dest_low, dest_high = mapper[row,col]
            conv[dest_low:dest_high] = arr
    #print(nuclide_xs_data)
    #print(conv)


    return conv


#####################################################################
# Generation of randomized starting particle buffers
#####################################################################

# Allocates and initializes particle arrays. This is the simplified
# version that makes no assumptions on energy spacing etc.
def init_particle_data(n_starting_particles_per_pe, n_xs, seed):
    particle_e =           np.random.random(n_starting_particles_per_pe).astype(np.float32)
    particle_xs          = np.zeros(n_starting_particles_per_pe * n_xs, dtype=np.float32)
    
    return particle_e, particle_xs, seed

# This is the most complex version, that samples particle energies randomly, and enforces
# TWO conditions on the distribution. The first condition is that particles must be
# sample between energy bands that are row-decomposed, such that no particle
# ends up in the space between what is held by adjacent PE's. We could have removed
# this by adding ghost rows to the PE's, which wouldn't affect performance or anything,
# but would greatly complicate the mapping of data to PEs and would be a big pain
# for something that is irrelevant for performance. The second major condition is that
# in each column, an equal number of particles will be sampled for each row (band), and
# then shuffled. This means that the particles will still have to be sorted within the column,
# but they will ultimately end up sorted such that ideal load balancing occurs. This strategy
# is only used if the "ASSUME_PERFECT_LOAD_BALANCE" option at the top of the file is set.
def init_particle_data_load_balanced(n_starting_particles_per_pe, n_xs, width, height, nuclide_energy_grids, n_gridpoints_per_nuclide, n_nuclides, neg_2D):
    n_particles = n_starting_particles_per_pe*width*height
    particle_e =           np.zeros(n_particles, dtype=np.float32)
    particle_xs          = np.zeros(n_particles * n_xs, dtype=np.float32)

    # Reshape the 1D particle energy array into a 2D matrix for easier accessing of rows/columns
    # Notably, if there are multiple starting particles per PE, then the number of columsn will increase
    # by that factor, but the number of rows will stay the same.
    particle_e = np.reshape(particle_e, (height, width * n_starting_particles_per_pe))

    # initialize each row to have particles that are randomly sampled within the band
    for row in range(height):
        #val_low = nuclide_energy_grids[n_gridpoints_per_nuclide * n_nuclides * row]
        val_low  = neg_2D[0,row*n_gridpoints_per_nuclide]
        val_high = neg_2D[0,row*n_gridpoints_per_nuclide + n_gridpoints_per_nuclide - 1]
        #print("sampling particles for row %d between %.2f and %.2f" %(row, val_low, val_high))
        particle_e[row,:] = np.random.uniform(val_low, val_high,n_starting_particles_per_pe * width).astype(np.float32)
    
    # At this point, each column is perfectly load balanced already, but the distribution of
    # particles in the column is non-random. Thus, if left alone, the sorting stage would not be
    # needed. Thus, we shuffle each column of the particle energy matrix independently, so that
    # the load balance is retained, but the particles still require sorting, which is a more realistic
    # assumption.
    for col in range(width):
        np.random.shuffle(particle_e[:,col])

    # Convert the particle matrix back to a 1D array
    particle_e = particle_e.ravel()

    return particle_e, particle_xs

# This function is a more realistic case to the one above (read its documentation first)
# This function is used when "ASSUME_PERFECT_LOAD_BALANCE" is set to false.
# This is a more realistic sampling of particles that does not guarantee that all PE's will
# end up with the same number of particles to process. This is not a big deal if there
# are many particles per PE, but if there is only 1 starting particle per PE, random noise
# can mean that some PE's get many particles and others none, and the load imbalance factor
# becomes very high. However, similar to he above function, this one does enforce that
# particles are sampled within energy bands, such that no particle is sampled between
# the space covered by adjacent PEs (which would result in incorrect results, in the absence
# of ghost cells, which are not worth implementing given that they would not affect performance).
# This function also gives a load balance report, such that the user may set the
# PARTICLE_BUFFER_MULTIPLIER factor in the shell launch script to accomodate whichever
# PE has the largest number of particles. E.g., if you ran 10 starting particles per PE,
# and the report said the max number of particles per PE after sorting would be 23, you
# would want to set PARTICLE_BUFFER_MULTIPLER to 3, so as to set the particle buffers
# to have enough space for 30 particles.
def init_particle_data_in_bands(n_starting_particles_per_pe, n_xs, width, height, nuclide_energy_grids, n_gridpoints_per_nuclide, neg_2D, particle_buffer_multiplier):
    n_particles = n_starting_particles_per_pe*width*height
    particle_e =           np.random.random(n_particles).astype(np.float32)
    particle_xs          = np.zeros(n_particles * n_xs, dtype=np.float32)

    # Compute bounds
    bounds = []
    
    for row in range(height):
        val_low  = neg_2D[0,row*n_gridpoints_per_nuclide]
        val_high = neg_2D[0,row*n_gridpoints_per_nuclide + n_gridpoints_per_nuclide - 1]
        bounds.append([val_low, val_high])

    # sample particles randomly between bounds (assuring approx even startind distro)
    sampled_bands = np.zeros((height, width), dtype = int)

    cur_col = 0
    for p in range(n_particles):
        row = np.random.randint(height)
        sampled_bands[row][cur_col] += 1
        band = bounds[row]
        val = np.random.uniform(band[0], band[1])
        particle_e[p] = val
        if (p+1) % n_starting_particles_per_pe == 0:
            cur_col += 1
        if (cur_col > width-1):
            cur_col = 0

    # For smaller cases, it may be useful to visualize the load balance via diagram.
    # We leave it commented out to prevent spam for full CS2 problems
    #print("Load balance diagram:")
    #print(sampled_bands)
    print("Load Balance Min: ",sampled_bands.min(), ", Max: ", sampled_bands.max(), " Avg: ", np.average(sampled_bands))

    assert sampled_bands.max() <= n_starting_particles_per_pe * particle_buffer_multiplier

    return particle_e, particle_xs

# Helper function
def sample_row_particles(low_e, high_e, n_particles):
    results = np.random.uniform(low_e, high_e,n_particles).astype(np.float32)
    return results

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

#####################################################################
# Actual Host CSL Program Begins Here
#####################################################################

# Read arguments
args = parse_args()
mode = args.mode
name = args.name
sdk = SdkWrapper(mode, name)

# Set the starting seed
np.random.seed(1337)

# Get pararms defined in .sh script that show up in compiled CSL module
width  = int(sdk.compile_data['params']['width'])
height = int(sdk.compile_data['params']['height'])

tile_width  = int(sdk.compile_data['params']['tile_width'])
tile_height = int(sdk.compile_data['params']['tile_height'])

# Number of particles to simulate (one lookup per particle)
n_starting_particles_per_pe = int(sdk.compile_data['params']['n_starting_particles_per_pe'])

# Number of nuclides in the problem
n_nuclides = int(sdk.compile_data['params']['n_nuclides'])
# Number of energy gridpoints per problem
n_gridpoints_per_nuclide = int(sdk.compile_data['params']['n_gridpoints_per_nuclide'])

# Number of cross section reaction channels stored
# E.g., total, fission, nu-fission, elastic, absorption
n_xs = int(sdk.compile_data['params']['n_xs'])

particle_buffer_multiplier = int(sdk.compile_data['params']['particle_buffer_multiplier'])

# Compute Total Number of PE's
n_PE = width * height
n_PE_total = width * height * tile_width * tile_height

# PRNG seed
seed = np.uint32(42)

######################################
# Input Summary
######################################
print_logo()
print_border("Input Summary")
print("Tile Grid Width                    =", tile_width)
print("Tile Grid Height                   =", tile_height)
print("Per-tile PE Grid Width             =", width)
print("Per-tile PE Grid Height            =", height)
print("Number of PEs per tile             =", n_PE)
print("Number of PEs total                =", n_PE_total)
print("Particles per PE                   =", n_starting_particles_per_pe)
print("Particles per tile                 =", n_starting_particles_per_pe * n_PE)
print("Particles total                    =", n_starting_particles_per_pe * n_PE_total)
print("Nuclides (per column)              =", n_nuclides)
print("Nuclides (total per tile)          =", n_nuclides * width)
print("Energy Gridpoints (per row)        =", n_gridpoints_per_nuclide)
print("Energy Gridpoints (total per tile) =", n_gridpoints_per_nuclide * height)
print("Number of XS reaction channels     =", n_xs)

######################################
# Initialize XS data and particle arrays on host.
# Also compute reference solutions
######################################

print_border("Initialization")

# Initialize cross section data arrays (unique for each PE)
print("Initializing XS data on host...")

# Convert things to 2D/3D nicely
nuclide_energy_grids, nuclide_xs_data, densities = init_xs_data_unique(n_nuclides, n_gridpoints_per_nuclide, n_xs, width, height)
neg_2D = np.reshape(nuclide_energy_grids, (n_nuclides * width, n_gridpoints_per_nuclide * height))

# If we wanted instead to have the same XS data on all PE's, we could use the below function
#nuclide_energy_grids, nuclide_xs_data, densities, seed = init_xs_data_replicated(n_nuclides, n_gridpoints_per_nuclide, n_xs, seed)

# Initialize particles
print("Initializing particles on host...")

particle_e = []
particle_xs = []

# Initialize Particle buffers
if ASSUME_PERFECT_LOAD_BALANCE:
    particle_e, particle_xs = init_particle_data_load_balanced(n_starting_particles_per_pe, n_xs, width, height, nuclide_energy_grids, n_gridpoints_per_nuclide, n_nuclides, neg_2D)
else:
    particle_e, particle_xs = init_particle_data_in_bands(n_starting_particles_per_pe, n_xs, width, height, nuclide_energy_grids, n_gridpoints_per_nuclide, neg_2D, particle_buffer_multiplier)

# Compute python reference solution on the host
particle_xs_expected = []
reference_particles = []
#if VALIDATE_RESULTS:
if False:
    print("Computing reference solution on host...")
    particle_xs_expected = calculate_xs_reference(n_starting_particles_per_pe*width*height, n_nuclides*width, n_gridpoints_per_nuclide*height, n_xs, nuclide_energy_grids, nuclide_xs_data, densities, particle_e, width, height)

    # Build reference particle objects, and sort them in energy for later comparison to CS2 solution
    for p in range(particle_e.size):
        reference_particles.append(Particle(particle_e[p], particle_xs_expected[p*n_xs:p*n_xs + n_xs]))

    reference_particles.sort(key=lambda p: p.energy)



xs_3D = np.reshape(nuclide_xs_data, (n_nuclides * width, n_gridpoints_per_nuclide * height, n_xs))
#print("nuclide grid in 2d")
#print(neg_2D)
#print("xs grid in 3d")
#print(xs_3D)

######################################
# Transfer all XS and particle data from Host ->Device
######################################

print_border("Host -> Device Data Migration")

# Construct a runner using SdkRuntime
#runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

print("Packing particles into contiguous buffer for transfer...")
# Assemble particles into contiguous storage
par_struct_sz = n_xs + 1;
par_len = par_struct_sz * particle_e.size
particle_contiguous = np.zeros(par_len, dtype=np.float32)

# Pack particles into contiguous buffer
for p in range(particle_e.size):
    particle_offset = p * par_struct_sz
    particle_contiguous[particle_offset] = particle_e[p] # Unsorted
    #particle_contiguous[particle_offset] = reference_particles[p].energy # Sorted (for testing)
    for xs in range(n_xs):
        particle_contiguous[particle_offset + 1 + xs] = particle_xs[p * n_xs + xs]
        
print("Reformatting nuclide XS grids...")
conv_nuclide_xs =  convolute_xs_data_for_memcpy_fast(n_nuclides, n_gridpoints_per_nuclide, n_xs, width, height)

print("Starting Cerebras Runner...")
# Load and run the program
sdk.start()

handles = []

#print("nuclide e grid")
#print(nuclide_energy_grids)


print("Transferring tiles to device...")
for tile_row in range(tile_height):
    for tile_col in range(tile_width):
        print("Transferring tile (", tile_row,",",tile_col,")...")
        starting_col = tile_col * width
        starting_row = tile_row * height

        # Copy all XS and particle data to the device
        #print("  Transferring nuclide energy grids from host -> device...")
        #for row in range(height):
        #    low = row * n_gridpoints_per_nuclide
        #    #high = low + n_gridpoints_per_nuclide * n_nuclides
        #    high = low + n_gridpoints_per_nuclide
        #    print("trasnferring NEGs to device. Doing single row %d" % row)
        #    print("low, high = %d, %d" % (low, high))
        #    print(nuclide_energy_grids[low:high])
        #    handle = copy_array_host_to_device_single_row(np.tile(nuclide_energy_grids[low:high], width*n_nuclides), 'nuclide_energy_grids', sdk, width, row, starting_col, starting_row)
        #    handles.append(handle)

        print("  Transferring nuclide XS grids from host -> device...")
        #handle = copy_array_host_to_device(conv_nuclide_xs, 'nuclide_xs_data', sdk, width, height, starting_col, starting_row)
        #handles.append(handle)
        #xs_buf_size = height * width * n_nuclides * n_gridpoints_per_nuclide * xs
        #e_buf_size = height * width * n_nuclides * n_gridpoints_per_nuclide
        #xs_buf = np.zeros(xs_buf_size, dtype=npfloat32)
        #e_buf = np.zeros(e_buf_size, dtype=npfloat32)
        xs_buf = np.array([],dtype=np.float32)
        e_buf = np.array([],dtype=np.float32)
        for row in range(height):
            #for col in range(width):

            # What XS data do I need?
            # I have a 3D array, of length (n total nuclides x n tot energy levels x xs)
            # For each PE, I just want basically a 3D slice of size (n nuclides x n energy levels x xs)
            #nlow = col*n_nuclides
            #nhigh = nlow + n_nuclides
            elow = row * n_gridpoints_per_nuclide
            ehigh = elow + n_gridpoints_per_nuclide
            data = xs_3D[:, elow:ehigh, :]
            data = np.ravel(data)
            #print("Assigning PE row = %d, col = %d" %(row, col))
            #print(data)
            xs_buf = np.append(xs_buf,data)
            #handle = copy_array_host_to_device(data, 'nuclide_xs_data', sdk, 1, 1, starting_col+col, starting_row+row)
            #handles.append(handle)

            # what energy data do I need?
            # I have a 2D array, of length (n total nuclides x n tot energy levels)
            # For each PE, I just want basically a 2D slice of size (n_nuclides * n_energy_levels)
            edata = neg_2D[:, elow:ehigh]
            edata = np.ravel(edata)
            #print("energy data:")
            #print(edata)
            e_buf = np.append(e_buf, edata)
            #handle = copy_array_host_to_device(edata, 'nuclide_energy_grids', sdk, 1, 1, starting_col+col, starting_row+row)
            #handles.append(handle)

        
        handle = copy_array_host_to_device(xs_buf, 'nuclide_xs_data', sdk, width, height, starting_col, starting_row)
        handles.append(handle)
        handle = copy_array_host_to_device(e_buf, 'nuclide_energy_grids', sdk, width, height, starting_col, starting_row)
        handles.append(handle)


        print("  Transferring nuclide density arrays from host -> device...")
        # A slice of the density array is shared for everyone in a given column (nuclide),
        # as densities do not vary by energy level. If there is only 1 nuclide per column,
        # then this function will actually just copy 1 value to each PE.
        for col in range(width):
            start = n_nuclides * col
            stop = start + n_nuclides
            handle = copy_array_host_to_device_single_column(np.tile(densities[start:stop], height), 'densities', sdk, height, col, starting_col, starting_row)
            handles.append(handle)


        print("  Transferring contiguous particle arrays from host -> device...")
        handle = copy_array_host_to_device(particle_contiguous, 'particle', sdk, width, height, starting_col, starting_row)
        handles.append(handle)

print("Blocking for host -> device transfers to complete...")
#for handle in handles:
#    runner.task_wait(handle)

print("All host -> device transfers to completed!")

######################################
# Launch kernel on device
######################################
print_border("Simulation")

# Start python host timer
time_start = time.time()

# Run XS lookup kernel on the device
print("Executing XS lookup kernel on device...")
sdk.launch('start_simulation', nonblock=False)

# Stop python host timer
time_stop = time.time()
total_time = time_stop-time_start
print("Host Runtime = %.2e [seconds]" % total_time)

######################################
# Transfer data back to the host
######################################


# Note: the "*3" is to allow extra particles per rank to be return, e.g., if there is not perfect load balancing
particle_contiguous_return = np.zeros((tile_height, tile_width, par_len*particle_buffer_multiplier), dtype=np.float32)

if VALIDATE_RESULTS:
    handles = []
    for tile_row in range(tile_height):
        for tile_col in range(tile_width):
            starting_col = tile_col * width
            starting_row = tile_row * height

            # Copy results (particle XS data) back to the host for validation
            handle = copy_array_device_to_host(particle_contiguous_return[tile_row, tile_col,:], 'particle_finished', sdk, width, height, starting_col, starting_row)
            handles.append(handle)
            if(tile_row == 0 and tile_col == 0):
                print("Kernel complete!")
            print("Tranfser of particle XS data results from device -> host on tile (",tile_row,",",tile_col,") complete.")

    print("Blocking for device -> host transfers to finish...")
    #for handle in handles:
    #    runner.task_wait(handle)
    print("Device -> host transfers complete!")


# Copy timestamp array back to the host
# Note that each device-side timestamp is 48 bits, stored in the form of an array of 16-bit uints.
# To store three timestamps, we are concatenating them together into an array of 9. One would
# think that the host array should also be of type uint16, but it turns out it requires the host
# array to be 32 bits.
timestamps = np.zeros(tile_height*tile_width*9*n_PE, dtype=np.uint32)
print("Transferring all hardware counter data from device -> host...")
handle = copy_array_device_to_host_16_bit(timestamps, 'timestamps', sdk, width*tile_width, height*tile_height)
#runner.task_wait(handle)

# Stop the program
print("Terminating program runner...")
sdk.stop()

######################################
# Create Particle objects to sort
######################################


######################################
# Read simulator trace and hardware counters to get cycle counts
######################################

print("Reading simulator traces...")
#Read the simulator trace (only works in simulator, not on real CS2 hardware)
#debug_mod = debug_util(args.name)
#trace_cycles = []
#sort_cycles = []
#round_robin_cycles = []
#for y in range(1, 1 + height):
#    for x in range(4, 4 + width):
#        #trace_output = debug_mod.read_trace(x, y, 'my_trace')
#        #trace_cycles.append( trace_output[2] - trace_output[0] )
#        #sort_cycles.append( trace_output[1] - trace_output[0] )
#        #round_robin_cycles.append( trace_output[2] - trace_output[1] )
#
#       # Note that reading printf trace output will fail if
#       # there are no printf statements, so this is commented out
#
#        printf_output = debug_mod.read_trace(x,y,'printf')
#        print("(",y-1,",",x-4,")")
#        print(printf_output)

#trace_cycles = max(trace_cycles)
#sort_cycles = max(sort_cycles)
#round_robin_cycles = max(round_robin_cycles)

# Read the hardware timestamps (works on both simulator and real hardware)
print("Reading CS2 hardware counters...")
cycles = []
sorting = []
robin = []
for i in range(n_PE_total):
    low = i * 9
    high = low + 9

    cycles.append( subtract_timestamps(timestamps[low:high], 0, 2) )
    sorting.append( subtract_timestamps(timestamps[low:high], 0, 1) )
    robin.append( subtract_timestamps(timestamps[low:high], 1, 2) )

cycles = max(cycles)
sort_cycles= max(sorting)
round_robin_cycles = max(robin)

######################################
# Output Performance Data
######################################
print_border("Performance Results")

# Compute some FOMs
seconds = cycles / 0.85e9
lps = n_starting_particles_per_pe * width * height * tile_width * tile_height / seconds
lps_per_pe = lps / (width*height*tile_width*tile_height)
total_pe_count = (757-7) * (996-2)
whole_machine_lps = lps_per_pe * total_pe_count

print("PEs used per tile:                             %d" % n_PE)
print("PEs used total:                                %d" % n_PE_total)
print("Cycle count (via tsc.get_timestamp):           %d [max cycles per PE]" % cycles)
#print("Simulated cycle count (via trace_timestamp):   %d [max cycles per PE]" % trace_cycles)
print("Phase I (Sorting):                             %d [max cycles per PE]" % sort_cycles)
print("Phase II (Round-Robin):                        %d [max cycles per PE]" % round_robin_cycles)
print("Assumed clockrate:                             850 [MHz]")
print("Physical runtime:                              %.2e [seconds]"   % seconds)
print("Physical lookups/s:                            %.2e [lookups/s] on %d PEs" % (lps, n_PE_total))
print("Est. physical lookups/s if scaled to full CS2: %.2e [lookups/s] on %d PEs" % (whole_machine_lps, total_pe_count))

######################################
# Validate Results
######################################
if VALIDATE_RESULTS:
    print_border("Validation")
    for tile_row in range(tile_height):
        for tile_col in range(tile_width):
            print("Unpacking particles and validating tile (",tile_row,",",tile_col,")...")
            starting_col = tile_col * width
            starting_row = tile_row * height

            print("  Unpacking particles from contiguous buffer...")
            # unpack contiguous buffer
            p = 0
            pe_particles = np.zeros((height, width))
            for possible_p in range(particle_e.size * particle_buffer_multiplier):
                row = possible_p // (n_starting_particles_per_pe * width * particle_buffer_multiplier)
                col = possible_p % (n_starting_particles_per_pe * width * particle_buffer_multiplier)
                col = col // (n_starting_particles_per_pe * particle_buffer_multiplier)
                particle_offset = possible_p * par_struct_sz
                possible_e = particle_contiguous_return[tile_row, tile_col, particle_offset]
                if possible_e == 0.0:
                    continue
                particle_e[p] = particle_contiguous_return[tile_row, tile_col, particle_offset];
                for xs in range(n_xs):
                    particle_xs[p * n_xs + xs] = particle_contiguous_return[tile_row, tile_col, particle_offset + 1 + xs]
                p += 1
                pe_particles[row,col] += 1
            print(pe_particles)
            print("Maximum load = %d" % pe_particles.max())


            print("  Unpacking particles from contiguous buffer complete! Number of particles: ", p)

            print("  Reorganizing particles and sorting for comparison...")

            experimental_particles = []

            for p in range(particle_e.size):
                experimental_particles.append(Particle(particle_e[p], particle_xs[p*n_xs:p*n_xs + n_xs]))

            experimental_particles.sort(key=lambda p: p.energy)
            reference_particles.sort(key=lambda p: p.energy)
    
            print("  Validating...")

            #print("egrids:")
            #print(nuclide_energy_grids)
            #print("xs data:")
            #print(nuclide_xs_data)
            #print("densities")
            #print(densities)
            for p in range(particle_e.size):
                #print("Particle %d:" % p)
                #print("Energy Reference = %.3f Actual = %.3f" %(reference_particles[p].energy, experimental_particles[p].energy))
                #print("Reference XS:")
                #print(reference_particles[p].xs)
                #print("Exerpimental XS:")
                #print(experimental_particles[p].xs)
                np.testing.assert_allclose(reference_particles[p].energy, experimental_particles[p].energy, atol=0.01, rtol=0)
                np.testing.assert_allclose(reference_particles[p].xs, experimental_particles[p].xs, atol=0.01, rtol=0)
        
            print("  CALCULATIONS CORRECT")

    #print("REFERENCE PARTICLE OBJECTS")
    #print(reference_particles)
    #print("EXPERIMENTAL PARTICLE OBJECTS")
    #print(experimental_particles)

    # Ensure that the result matches our expectation
    #for p in range(particle_e.size):
    #    np.testing.assert_allclose(reference_particles[p].energy, experimental_particles[p].energy, atol=0.01, rtol=0)
    #    np.testing.assert_allclose(reference_particles[p].xs, experimental_particles[p].xs, atol=0.01, rtol=0)

else:
    print("Reference host solution not generated. Validation not attempted.")

print_line()
