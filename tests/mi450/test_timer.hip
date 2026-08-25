#include <cstdio>
#include <hip/hip_runtime.h>
#include "mirage/persistent_kernel/profiler.h"
__global__ void k(unsigned long long*o){
  if(threadIdx.x) return;
  o[0]=mirage::arch::realtime_ticks();
  tb::sleep_cycles(1000);            // must terminate: exercises the loop
  o[1]=mirage::arch::realtime_ticks();
  o[2]=tb::get_timestamp();
}
int main(){
  unsigned long long*d;hipMalloc(&d,24);hipMemset(d,0,24);
  hipLaunchKernelGGL(k,dim3(1),dim3(32),0,0,d);
  hipError_t e=hipDeviceSynchronize();
  if(e!=hipSuccess){printf("FAIL: %s\n",hipGetErrorString(e));return 1;}
  unsigned long long h[3];hipMemcpy(h,d,24,hipMemcpyDeviceToHost);
  printf("t0=%llu\nt1=%llu\ndelta=%lld ticks\nget_timestamp=%llu\n",h[0],h[1],(long long)(h[1]-h[0]),h[2]);
  if(h[0]==0){printf("FAIL: clock read zero\n");return 1;}
  if(h[1]<h[0]){printf("FAIL: clock went backwards\n");return 1;}
  if((long long)(h[1]-h[0])<1000){printf("FAIL: sleep_cycles(1000) advanced only %lld ticks\n",(long long)(h[1]-h[0]));return 1;}
  printf("PASS: clock monotonic, sleep_cycles terminates and advances >=1000 ticks\n");
  return 0;
}
