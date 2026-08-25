#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>
#include "mirage/persistent_kernel/arch_traits.cuh"
// Does the async global->LDS copy actually land, and does the asynccnt wait
// actually gate it? 32 lanes x 16B = 512B moved HBM->LDS without staging
// through VGPRs.
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
__global__ __launch_bounds__(32) void k(int const*g,int*out){
#if defined(MIRAGE_ARCH_GFX1250)
  __shared__ int lds[32*4];
  int lane=threadIdx.x;
  using v_t = mirage::arch::async_copy_vec_t;
  mirage::arch::async_copy_b128((v_t const*)(g+lane*4),(v_t*)(lds+lane*4));
  mirage::arch::async_copy_wait_all();   // s_wait_asynccnt(0)
  __syncthreads();
  for(int i=0;i<4;++i) out[lane*4+i]=lds[lane*4+i];
#endif
}
#endif
int main(){
  const int N=128;
  std::vector<int> h(N); for(int i=0;i<N;++i) h[i]=i*7+1;
  int*dg,*dO; hipMalloc(&dg,N*4); hipMalloc(&dO,N*4);
  hipMemcpy(dg,h.data(),N*4,hipMemcpyHostToDevice);
  std::vector<int> poison(N,-99999);
  hipMemcpy(dO,poison.data(),N*4,hipMemcpyHostToDevice);
  hipLaunchKernelGGL(k,dim3(1),dim3(32),0,0,dg,dO);
  hipError_t e=hipDeviceSynchronize();
  if(e!=hipSuccess){printf("FAIL: %s\n",hipGetErrorString(e));return 1;}
  std::vector<int> got(N); hipMemcpy(got.data(),dO,N*4,hipMemcpyDeviceToHost);
  int bad=0,untouched=0;
  for(int i=0;i<N;++i){ if(got[i]==-99999)++untouched; if(got[i]!=h[i])++bad; }
  printf("untouched=%d  mismatches=%d of %d\n",untouched,bad,N);
  printf("first 6: "); for(int i=0;i<6;++i) printf("%d(want %d) ",got[i],h[i]); printf("\n");
  if(untouched||bad){printf("FAIL: async global->LDS copy did not land correctly\n");return 1;}
  printf("PASS: global_load_async_to_lds_b128 + s_wait_asynccnt(0) correct\n");
  return 0;
}
