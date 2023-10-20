#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <omp.h>
#include <thrust/sort.h>
#include <assert.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

//#define USE_F16
//#define USE_LCG
#define USE_CURAND
#define USE_SORT
#define USE_UEG

// The number of reaction channels stored for each energy level
// (e.g., total, elastic scattering, fission, nu-fission, absorption)
#define NXS 5

// Minimal particle abstraction
typedef struct{
  float energy; // Energy parameter that defines the lookup
  float macro_xs[NXS]; // Macroscopic XS data that defines the output data, to be computed by lookup kernel
} Particle;

struct Cmp {
  __host__ __device__
  bool operator()(const Particle& p1, const Particle& p2) {
      return p1.energy < p2.energy;
  }
};

__host__ __device__ float LCG_random_float(uint32_t* seed);
int compare(const void * a, const void * b);
__device__ int iterative_binary_search(float* A, int n, float quarry);
void init_grids(uint32_t* seed, int n_nuclides, int n_gridpoints_per_nuclide, float** neg, float** nxd, float** conc);
void init_particles(uint32_t* seed, int n_particles, Particle** particles);
void input_summary(uint32_t seed, int n_particles, int n_nuclides, int n_gridpoints_per_nuclide);

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void calculate_xs_kernel(int n_particles, int n_nuclides, int n_gridpoints_per_nuclide, float * nuclide_energy_grids, float * nuclide_xs_data, float * densities, Particle* particles, float * ueg_energy, int * ueg_idx, curandStatePhilox4_32_10_t *state) {
  int p = blockIdx.x*blockDim.x + threadIdx.x;
  #ifdef USE_CURAND
  curandStatePhilox4_32_10_t localState = state[p];
  #endif
  #ifdef USE_LCG
  uint32_t seed = p;
  #endif

  if (p >= n_particles)
    return;

  float e = particles[p].energy;
  #ifdef USE_UEG
  int uidx = iterative_binary_search( ueg_energy, n_nuclides * n_gridpoints_per_nuclide, e);
  #endif
  for (int n = 0; n < n_nuclides; n++) { 
    // Binary search for location on energy grid
    // The particle's energy will almost always fall between two energy points
    #ifdef USE_UEG
    int ueg_lower_index;
    if( ueg_idx[uidx * n_nuclides + n] == n_gridpoints_per_nuclide - 1 )
			ueg_lower_index = ueg_idx[uidx * n_nuclides + n] - 1;
		else
			ueg_lower_index = ueg_idx[uidx * n_nuclides + n];
    int lower_index = ueg_lower_index;
    
    #else
    int lower_index = iterative_binary_search( &nuclide_energy_grids[n * n_gridpoints_per_nuclide], n_gridpoints_per_nuclide, e);
    #endif

    // Compute Interpolation Factor
    float e_lower = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index];
    float e_higher = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index + 1];
    #if defined(USE_LCG) || defined(USE_CURAND)
      #ifdef USE_CURAND
        float sample = e_lower + curand_uniform(&localState) *(e_higher-e_lower);
      #else
        float sample = e_lower + LCG_random_float(&seed) *(e_higher-e_lower);
      #endif
      if (e > sample ) {
        lower_index++;
      }
    #else
      #ifdef USE_FP16
        half left = e_higher - e;
        half right = e_eight - e_lower;
        float f = __hdiv(left, right);
      #else
        float f = (e_higher - e) / (e_higher - e_lower);
      #endif
    #endif

    for (int xs = 0; xs < NXS; xs++ ) {
      float xs_lower  = nuclide_xs_data[n * n_gridpoints_per_nuclide * NXS +  lower_index      * NXS + xs];
      #if defined(USE_LCG) || defined(USE_CURAND)
        particles[p].macro_xs[xs] += densities[n] * xs_lower;
      #else
        float xs_higher = nuclide_xs_data[n * n_gridpoints_per_nuclide * NXS + (lower_index + 1) * NXS + xs];
        particles[p].macro_xs[xs] += densities[n] * (xs_higher - f * (xs_higher - xs_lower) );
      #endif
    }
  }
}

__global__ void calculate_xs_kernel_LCG(int n_particles, int n_nuclides, int n_gridpoints_per_nuclide, float * nuclide_energy_grids, float * nuclide_xs_data, float * densities, Particle* particles, float * ueg_energy, int * ueg_idx) {
  int p = blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t seed = p;
  if (p >= n_particles)
    return;
  float e = particles[p].energy;
  int uidx = iterative_binary_search( ueg_energy, n_nuclides * n_gridpoints_per_nuclide, e);
  for (int n = 0; n < n_nuclides; n++) { 
    // Binary search for location on energy grid
    // The particle's energy will almost always fall between two energy points
    int ueg_lower_index;
    if( ueg_idx[uidx * n_nuclides + n] == n_gridpoints_per_nuclide - 1 )
			ueg_lower_index = ueg_idx[uidx * n_nuclides + n] - 1;
		else
			ueg_lower_index = ueg_idx[uidx * n_nuclides + n];
    int lower_index = ueg_lower_index;

    // Compute Interpolation Factor
    float e_lower = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index];
    float e_higher = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index + 1];
    //float f = (e_higher - e) / (e_higher - e_lower);
    float sample = e_lower + LCG_random_float(&seed) *(e_higher-e_lower);
    if (e > sample ) {
      lower_index++;
    }

    for (int xs = 0; xs < NXS; xs++ ) {
      float xs_lower  = nuclide_xs_data[n * n_gridpoints_per_nuclide * NXS +  lower_index      * NXS + xs];
      //float xs_higher = nuclide_xs_data[n * n_gridpoints_per_nuclide * NXS + (lower_index + 1) * NXS + xs];
      particles[p].macro_xs[xs] += densities[n] * xs_lower;
    }
  }
}

__global__ void calculate_xs_kernel_og(int n_particles, int n_nuclides, int n_gridpoints_per_nuclide, float * nuclide_energy_grids, float * nuclide_xs_data, float * densities, Particle* particles, float * ueg_energy, int * ueg_idx) {
  int p = blockIdx.x*blockDim.x + threadIdx.x;
  if (p >= n_particles)
    return;
  float e = particles[p].energy;
  int uidx = iterative_binary_search( ueg_energy, n_nuclides * n_gridpoints_per_nuclide, e);
  for (int n = 0; n < n_nuclides; n++) { 
    // Binary search for location on energy grid
    // The particle's energy will almost always fall between two energy points
    //int lower_index = iterative_binary_search( &nuclide_energy_grids[n * n_gridpoints_per_nuclide], n_gridpoints_per_nuclide, e);
    int ueg_lower_index;
    if( ueg_idx[uidx * n_nuclides + n] == n_gridpoints_per_nuclide - 1 )
			ueg_lower_index = ueg_idx[uidx * n_nuclides + n] - 1;
		else
			ueg_lower_index = ueg_idx[uidx * n_nuclides + n];
    int lower_index = ueg_lower_index;
    
    //int ueg_lower_index = ueg_idx[ui];
    //if( lower_index != ueg_lower_index )
    //  printf("true lower index = %d, ueg lower index = %d\n", lower_index, ueg_lower_index);

    // Compute Interpolation Factor
    float e_lower = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index];
    float e_higher = nuclide_energy_grids[n * n_gridpoints_per_nuclide + lower_index + 1];
    float f = (e_higher - e) / (e_higher - e_lower);

    for (int xs = 0; xs < NXS; xs++ ) {
      float xs_lower  = nuclide_xs_data[n * n_gridpoints_per_nuclide * NXS +  lower_index      * NXS + xs];
      float xs_higher = nuclide_xs_data[n * n_gridpoints_per_nuclide * NXS + (lower_index + 1) * NXS + xs];
      particles[p].macro_xs[xs] += densities[n] * (xs_higher - f * (xs_higher - xs_lower) );
    }
  }
}

int main(int argc, char* argv[])
{
  // Simulation parameters
  uint32_t seed = 42;
  //int n_particles = 750.0*950.0/50.0;
  //int n_particles = 750.0*950.0;
  int n_particles = 100000000;
  int n_nuclides = 250;
  int n_gridpoints_per_nuclide = 10000;
  assert( argc == 2 );
  n_particles = atoi(argv[1]);

  // Print input summary
  input_summary(seed, n_particles, n_nuclides, n_gridpoints_per_nuclide);

  // 2D array (stored in 1D) with dimensions of size [n_nuclides, n_gridpoints_per_nuclide].
  // Each row constitutes the energy grid for a single nuclide.
  // All nuclides are assumed to have equal length, though each contain randomized (but sorted) energy levels.
  float* nuclide_energy_grids;

  // 3D array (stored in 1D) with dimensions of size [n_nuclides, n_gridpoints_per_nuclide, NXS].
  // Each inner 2D slice represents cross section data for a single nuclide, by energy level, then by
  // reaction channel in the inner dimension.
  float* nuclide_xs_data;

  // 1D array with length n_nuclides.
  // Each entry represents the density of each nuclide present in the material
  float* densities;
  
  // 1D array of lightweight particle objects with length n_particles.
  // Each particle represents a computational task, where the particle's energy is
  // used as input to the cross section lookup kernel, and the kernel will fill in
  // the particle's macroscopic XS data array.
  Particle* particles;

  // Generates synthetic cross section data sets
  init_grids(&seed, n_nuclides, n_gridpoints_per_nuclide, &nuclide_energy_grids, &nuclide_xs_data, &densities);

  // Generates UEG
  int n_gridpoints_total = n_gridpoints_per_nuclide * n_nuclides;
  float* ueg_energy;
  int* ueg_idx;
  #ifdef USE_UEG
  ueg_energy = (float*) malloc(n_gridpoints_total * sizeof(float));

  for( int i = 0; i < n_gridpoints_total; i++ )
  {
    ueg_energy[i] = nuclide_energy_grids[i];
  }

  // Sort UEG energy grid
  qsort(ueg_energy, n_gridpoints_total, sizeof(float), compare);

  // Generate double index grid
  int * idx_low = (int *) calloc( n_nuclides, sizeof(int));
  float * energy_high = (float *) malloc( n_nuclides * sizeof(float));
  
  int n_idx_total = n_gridpoints_total * n_nuclides;

  ueg_idx = (int*) malloc(n_idx_total * sizeof(int));

  for( int i = 0; i < n_nuclides; i++ )
			energy_high[i] = nuclide_energy_grids[i * n_gridpoints_per_nuclide + 1];

  for( int e = 0; e < n_gridpoints_total; e++ )
  {
    double unionized_energy = ueg_energy[e];
    for( int i = 0; i < n_nuclides; i++ )
    {
      if( unionized_energy < energy_high[i]  )
        ueg_idx[e * n_nuclides + i] = idx_low[i];
      else if( idx_low[i] == n_gridpoints_per_nuclide - 2 )
        ueg_idx[e * n_nuclides + i] = idx_low[i];
      else
      {
        idx_low[i]++;
        ueg_idx[e * n_nuclides + i] = idx_low[i];
        energy_high[i] = nuclide_energy_grids[i * n_gridpoints_per_nuclide + idx_low[i] + 1];
      }
    }
  }
  #endif

  
  // Generates randomized particles
  init_particles(&seed, n_particles, &particles);

  // Copy data to device
  float* d_nuclide_energy_grids;
  float* d_nuclide_xs_data;
  float* d_densities;
  Particle* d_particles;
  float* d_ueg_energy;
  int* d_ueg_idx;

  size_t sz = n_nuclides * n_gridpoints_per_nuclide * sizeof(float);
  cudaMalloc(&d_nuclide_energy_grids, sz);
  cudaMemcpy(d_nuclide_energy_grids, nuclide_energy_grids, sz, cudaMemcpyHostToDevice);
  
  sz = n_nuclides * n_gridpoints_per_nuclide * NXS * sizeof(float);
  cudaMalloc(&d_nuclide_xs_data, sz);
  cudaMemcpy(d_nuclide_xs_data, nuclide_xs_data, sz, cudaMemcpyHostToDevice);
  
  sz = n_nuclides * sizeof(float);
  cudaMalloc(&d_densities, sz);
  cudaMemcpy(d_densities, densities, sz, cudaMemcpyHostToDevice);
  
  sz = n_particles * sizeof(Particle);
  cudaMalloc(&d_particles, sz);
  cudaMemcpy(d_particles, particles, sz, cudaMemcpyHostToDevice);
  
  #ifdef USE_UEG
  sz = n_gridpoints_total * sizeof(float);
  cudaMalloc(&d_ueg_energy, sz);
  cudaMemcpy(d_ueg_energy, ueg_energy, sz, cudaMemcpyHostToDevice);
  
  sz = n_idx_total * sizeof(int);
  cudaMalloc(&d_ueg_idx, sz);
  cudaMemcpy(d_ueg_idx, ueg_idx, sz, cudaMemcpyHostToDevice);
  #endif

  curandStatePhilox4_32_10_t * d_states;
  #ifdef USE_CURAND
  sz = n_particles * sizeof(curandStatePhilox4_32_10_t);
  cudaMalloc(&d_states, sz);
  #endif
  
  printf("=====================================================================\n");
  printf("Simulation\n");
  printf("=====================================================================\n");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEvent_t sort_start, sort_stop;
  cudaEventCreate(&sort_start);
  cudaEventCreate(&sort_stop);

  printf("Sorting on Thrust...\n");
  
  cudaEventRecord(sort_start);
  #ifdef USE_SORT
  thrust::sort(thrust::device, d_particles, d_particles + n_particles, Cmp());
  #endif
  cudaEventRecord(sort_stop);
  cudaEventSynchronize(sort_stop);
  
  printf("Running simulation kernel on GPU...\n");

  int nthreads = 256;
  int nblocks = ceil((double) n_particles / (double) nthreads);

  #ifdef USE_CURAND
  setup_kernel<<<nblocks,nthreads>>>(d_states);
  #endif

  cudaEventRecord(start);
  calculate_xs_kernel<<<nblocks, nthreads>>>(n_particles, n_nuclides, n_gridpoints_per_nuclide, d_nuclide_energy_grids, d_nuclide_xs_data, d_densities, d_particles, d_ueg_energy, d_ueg_idx, d_states);
  cudaEventRecord(stop);

  // Copy data back from device
  sz = n_particles * sizeof(Particle);
  cudaMemcpy(particles, d_particles, sz, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double time = milliseconds / 1000.0;

  float sort_milliseconds = 0;
  cudaEventElapsedTime(&sort_milliseconds, sort_start, sort_stop);
  double sort_time = sort_milliseconds / 1000.0;

  printf("Simulation kernel complete.\n");

  printf("=====================================================================\n");
  printf("Results and Validation\n");
  printf("=====================================================================\n");

  printf("Sort runtime:                              %.4le [seconds]\n", sort_time);
  printf("XS kernel runtime:                         %.4le [seconds]\n", time);
  printf("Total      runtime:                        %.4le [seconds]\n", time+sort_time);
  printf("Lookups/sec:                               %.4le [lookups/s]\n", n_particles / (time+sort_time));

  // Kernel is finished. Now we reduce the simulation results and validate them
  // against a known checksum.

  // For verification, and to prevent the compiler from optimizing
  // all work out, we check the returned macro_xs array for each
  // particle to find its maximum value index, then increment the verification
  // value by that index. This validation strategy is not physically meaningful,
  // but is a reproducible way of ensuring correctness that is not highly
  // sensitive to floating point roundoff error. We can transfer all particle data
  // slowly back to the host, or could theoretically use reduction hardware
  // to accomplish this task.

  printf("Running verification reduction to generate checksum...\n");
  uint32_t checksum = 0;
  #pragma omp parallel for reduction(+:checksum)
  for (int p = 0; p < n_particles; p++) {
    float max = -1.0;
    int max_idx = 0;
    for (int xs = 0; xs < NXS; xs++) {
      if (particles[p].macro_xs[xs] > max) {
        max = particles[p].macro_xs[xs];
        max_idx = xs;
      }
    }
    checksum += max_idx;
  }

  printf("Validation checksum:                       %d\n", checksum);
  printf("=====================================================================\n");
}

// LCG params taken from l'Ecuyer
// https://www.ams.org/journals/mcom/1999-68-225/S0025-5718-99-00996-5/S0025-5718-99-00996-5.pdf
__host__ __device__ float LCG_random_float(uint32_t* seed)
{
  const uint32_t m = 4294967291; // 2^32 - 5
  const uint32_t a = 1588635695;
  const uint32_t c = 12345;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

// FP32 comparator function
int compare(const void * a, const void * b)
{
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}

// FP32 lower bound binary search function
// Returns the lower bound index of the value in array "A" that
// is immediately below the quarry value
__device__ int iterative_binary_search(float* A, int n, float quarry)
{
  int lowerLimit = 0;
  int upperLimit = n-1;
  int examinationPoint;
  int length = upperLimit - lowerLimit;

  while( length > 1 )
  {
    examinationPoint = lowerLimit + ( length / 2 );

    if( A[examinationPoint] > quarry )
      upperLimit = examinationPoint;
    else
      lowerLimit = examinationPoint;

    length = upperLimit - lowerLimit;
  }

  return lowerLimit;
}

void init_grids(uint32_t* seed, int n_nuclides, int n_gridpoints_per_nuclide, float** neg, float** nxd, float** conc) {

  // 2D array (stored in 1D) with dimensions of size [n_nuclides, n_gridpoints_per_nuclide].
  // Each row constitutes the energy grid for a single nuclide.
  // All nuclides are assumed to have equal length, though each contain randomized (but sorted) energy levels.
  int nuclide_energy_grids_len = n_nuclides * n_gridpoints_per_nuclide;
  float* nuclide_energy_grids = (float *) malloc(nuclide_energy_grids_len * sizeof(float));

  printf("Allocating nuclide energy array of size:   %.2le [bytes]\n", (double) (nuclide_energy_grids_len * sizeof(float)));

  // 3D array (stored in 1D) with dimensions of size [n_nuclides, n_gridpoints_per_nuclide, NXS].
  // Each inner 2D slice represents cross section data for a single nuclide, by energy level, then by
  // reaction channel in the inner dimension.
  int nuclide_xs_data_len = n_nuclides * n_gridpoints_per_nuclide * NXS;
  float* nuclide_xs_data = (float*) malloc(nuclide_xs_data_len * sizeof(float));

  printf("Allocating nuclide XS data of size:        %.2le [bytes]\n", (double) (nuclide_xs_data_len * sizeof(float)));

  // 1D array with length n_nuclides.
  // Each entry represents the density of each nuclide present in the material
  float* densities = (float*) malloc(n_nuclides * sizeof(float));

  printf("Allocating nuclide densities data of size: %.2le [bytes]\n", (double) (n_nuclides * sizeof(float)));

  printf("Initializing energy grids...\n");
  // Initialize energy grids
  for (int n = 0; n < n_nuclides; n++ ) {

    // Randomly sample all energy points for the nuclide
    for (int e = 0; e < n_gridpoints_per_nuclide; e++) {
      nuclide_energy_grids[n * n_gridpoints_per_nuclide + e] = LCG_random_float(seed);
    }
      
    // Sort energy grid for the nuclide
    qsort(nuclide_energy_grids + n*n_gridpoints_per_nuclide, n_gridpoints_per_nuclide, sizeof(float), compare);

    // If using infinite precision arithmetic, we would just sort the energy array and move on.
    // However, as the vagaries of FP32 PRNG dictate that some samples may be repeated occasionally,
    // we must reject duplicate samples and then resort again.
    while (1) {
      int duplicates_found = 0;
      // Iterate through sorted array and check for adjacent neighbors being the same
      for (int e = 1; e < n_gridpoints_per_nuclide; e++) {
        // If a duplicate entry is found, resample it.
        if (nuclide_energy_grids[n * n_gridpoints_per_nuclide + e] == nuclide_energy_grids[n * n_gridpoints_per_nuclide + e - 1] ) {
          nuclide_energy_grids[n * n_gridpoints_per_nuclide + e] = LCG_random_float(seed);
          duplicates_found++;
        }
      }
      // If a new value was sampled, then we need to resort. Otherwise, we can move on to next nuclide.
      if (duplicates_found) {
        qsort(nuclide_energy_grids + n*n_gridpoints_per_nuclide, n_gridpoints_per_nuclide, sizeof(float), compare);
      } else {
        break;
      }
    }
  }

  printf("Initializing XS data...\n");
  // Initialize nuclide XS datasets
  for (int i = 0; i < n_nuclides * n_gridpoints_per_nuclide * NXS; i++) {
    nuclide_xs_data[i] = LCG_random_float(seed);
  }

  printf("Initializing densities...\n");
  // Initialize densities
  for (int n = 0; n < n_nuclides; n++) {
    densities[n] = LCG_random_float(seed);
  }

  // Pass arrays back to caller
  *neg = nuclide_energy_grids;
  *nxd = nuclide_xs_data;
  *conc = densities;
}

void init_particles(uint32_t* seed, int n_particles, Particle** par) {
  printf("Initializing particles...\n");
  
  // Initialize randomized particle energies, set macro XS data to zero
  Particle* particles = (Particle*) malloc(n_particles * sizeof(Particle));
  
  double total_kb = n_particles * sizeof(Particle);
  printf("Allocating particle data of size:          %.2le [bytes]\n", total_kb);

  for (int p = 0; p < n_particles; p++) {
    particles[p].energy = LCG_random_float(seed);
    for( int xs = 0; xs < NXS; xs++ ) {
      particles[p].macro_xs[xs] = 0.0;
    }
  }

  // Pass array back to caller
  *par = particles;
}

void input_summary(uint32_t seed, int n_particles, int n_nuclides, int n_gridpoints_per_nuclide) {
  printf("=====================================================================\n");
  printf("Input Summary\n");
  printf("=====================================================================\n");
  printf("Seed:                                      %d\n", seed);
  printf("Particles (xs lookup work items):          %d\n", n_particles);
  printf("Number of nuclides in material:            %d\n", n_nuclides);
  printf("Number of energy gridpoints per material:  %d\n", n_gridpoints_per_nuclide);
  double kb_xs_total = (n_nuclides * n_gridpoints_per_nuclide
      + n_nuclides * n_gridpoints_per_nuclide * NXS
      + n_nuclides ) * sizeof(float);
  printf("Estimated XS data size:                    %.2le [bytes]\n", kb_xs_total);
  double kb_particle_total = n_particles * sizeof(Particle);
  printf("Estimated particle data size:              %.2le [bytes]\n", kb_particle_total);
  printf("=====================================================================\n");
  printf("Initialization\n");
  printf("=====================================================================\n");
}
