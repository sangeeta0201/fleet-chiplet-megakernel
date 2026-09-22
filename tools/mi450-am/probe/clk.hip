#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_wall(unsigned long long *o){ o[0] = wall_clock64(); }
__global__ void k_cyc (unsigned long long *o){ o[1] = __builtin_readcyclecounter(); }
__global__ void k_clk (unsigned long long *o){ o[2] = clock64(); }
