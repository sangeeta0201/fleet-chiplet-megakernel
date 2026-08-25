#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>
#include "mirage/persistent_kernel/tasks/mi450/gemm_handtuned_mi450.cuh"
constexpr int M=16,K=512,N=64;
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
__global__ __launch_bounds__(128) void gk(void const*a,void const*b,void const*r,void*o,int nt,bool ra){
#if defined(MIRAGE_ARCH_GFX1250)
  kernel::mi450::gemm_handtuned_wmma<M,K>(a,b,r,o,nt,ra,N,0,N);
#endif
}
#endif
int main(){
  std::vector<__bf16> hA(M*K),hB(N*K),hR(M*N);
  for(int i=0;i<M*K;++i) hA[i]=(__bf16)(((i*37)%17-8)*0.125f);
  for(int i=0;i<N*K;++i) hB[i]=(__bf16)(((i*53)%13-6)*0.25f);
  for(int i=0;i<M*N;++i) hR[i]=(__bf16)(((i*11)%7-3)*0.5f);
  std::vector<float> ref(M*N,0.f);
  for(int m=0;m<M;++m)for(int n=0;n<N;++n){float a=0;for(int k=0;k<K;++k)a+=(float)hA[m*K+k]*(float)hB[n*K+k];ref[m*N+n]=a+(float)hR[m*N+n];}
  __bf16*dA,*dB,*dR,*dO;
  hipMalloc(&dA,M*K*2);hipMalloc(&dB,N*K*2);hipMalloc(&dR,M*N*2);hipMalloc(&dO,M*N*2);
  hipMemcpy(dA,hA.data(),M*K*2,hipMemcpyHostToDevice);
  hipMemcpy(dB,hB.data(),N*K*2,hipMemcpyHostToDevice);
  hipMemcpy(dR,hR.data(),M*N*2,hipMemcpyHostToDevice);
  // Poison the output so "kernel never wrote" is distinguishable from "wrote zeros".
  std::vector<__bf16> poison(M*N,(__bf16)-99999.f);
  hipMemcpy(dO,poison.data(),M*N*2,hipMemcpyHostToDevice);
  hipLaunchKernelGGL(gk,dim3(1),dim3(128),0,0,dA,dB,dR,dO,M,true);
  hipDeviceSynchronize();
  std::vector<__bf16> hO(M*N); hipMemcpy(hO.data(),dO,M*N*2,hipMemcpyDeviceToHost);
  int untouched=0, nonzero=0;
  for(int i=0;i<M*N;++i){ float g=(float)hO[i]; if(g==-99999.f)++untouched; if(g!=0.f)++nonzero; }
  printf("untouched(poison remains)=%d  nonzero=%d  of %d\n",untouched,nonzero,M*N);
  printf("first 8 got vs want:\n");
  for(int i=0;i<8;++i) printf("  [%d,%d] got %12.4f want %12.4f  diff %.6f\n",i/N,i%N,(float)hO[i],ref[i],(float)hO[i]-ref[i]);
  printf("corner [15,63] got %.4f want %.4f\n",(float)hO[15*N+63],ref[15*N+63]);
  return 0;
}
