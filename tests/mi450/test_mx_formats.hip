#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>
#include "mirage/persistent_kernel/tasks/mi450/mx_wmma_mi450.cuh"
using namespace kernel::mi450;
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
__global__ __launch_bounds__(32) void mk(void const*ap,void const*bp,float*out,unsigned sa,unsigned sb){
#if defined(MIRAGE_ARCH_GFX1250)
  int lane=threadIdx.x;
  mx_i32x16_t a=((mx_i32x16_t const*)ap)[lane];
  mx_i32x16_t b=((mx_i32x16_t const*)bp)[lane];
  mx_f32x8_t acc={0,0,0,0,0,0,0,0};
  acc=_gang_wmma_f4xf8(a,b,acc,(int)sa,(int)sb);
  for(int i=0;i<8;++i) out[lane*8+i]=acc[i];
#endif
}
#endif
static std::vector<float> run(uint8_t av,uint8_t bv,unsigned sa,unsigned sb){
  std::vector<uint8_t> hA(32*64,av),hB(32*64,bv);
  uint8_t*dA,*dB; float*dO;
  hipMalloc(&dA,hA.size());hipMalloc(&dB,hB.size());hipMalloc(&dO,256*4);
  hipMemcpy(dA,hA.data(),hA.size(),hipMemcpyHostToDevice);
  hipMemcpy(dB,hB.data(),hB.size(),hipMemcpyHostToDevice);
  hipMemset(dO,0,256*4);
  hipLaunchKernelGGL(mk,dim3(1),dim3(32),0,0,dA,dB,dO,sa,sb);
  hipDeviceSynchronize();
  std::vector<float> o(256); hipMemcpy(o.data(),dO,256*4,hipMemcpyDeviceToHost);
  hipFree(dA);hipFree(dB);hipFree(dO);
  return o;
}
int main(){
  hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
  printf("device: %s (%s)\n",p.name,p.gcnArchName);
  int fails=0;
  // TEST 1: constant fill. Every A nibble = E2M1 code 2 = 1.0; every B byte =
  // E4M3 0x38 = 1.0. Then EVERY output element must be exactly K=128,
  // regardless of how lanes map to elements. Layout-independent.
  {
    auto o=run(0x22,0x38,127,127);
    int bad=0; for(float v:o) if(v!=128.0f) ++bad;
    printf("T1 constant-fill (A=1.0,B=1.0,K=128): out[0]=%.4f expect 128.0  mismatches=%d/256\n",o[0],bad);
    if(bad){printf("   -> FAIL: format immediates decode to the wrong numeric interpretation\n");++fails;}
  }
  // TEST 2: E2M1 value check. A nibble code 5 = 3.0 -> every output = 3*128.
  {
    auto o=run(0x55,0x38,127,127);
    int bad=0; for(float v:o) if(v!=384.0f) ++bad;
    printf("T2 A=3.0 (E2M1 code 5): out[0]=%.4f expect 384.0  mismatches=%d/256\n",o[0],bad);
    if(bad){printf("   -> FAIL: A is not being decoded as FP4 E2M1\n");++fails;}
  }
  // TEST 3: E4M3 value check. B byte 0x40 = 2.0 -> every output = 2*128.
  {
    auto o=run(0x22,0x40,127,127);
    int bad=0; for(float v:o) if(v!=256.0f) ++bad;
    printf("T3 B=2.0 (E4M3 0x40): out[0]=%.4f expect 256.0  mismatches=%d/256\n",o[0],bad);
    if(bad){printf("   -> FAIL: B is not being decoded as FP8 E4M3\n");++fails;}
  }
  // TEST 4: E8M0 scale semantics. scale byte 129 = 2^2, so output must be
  // exactly 4x the 2^0 case. Confirms the scale format immediate.
  {
    auto o=run(0x22,0x38,129,127);
    int bad=0; for(float v:o) if(v!=512.0f) ++bad;
    printf("T4 scale_a=2^2: out[0]=%.4f expect 512.0  mismatches=%d/256\n",o[0],bad);
    if(bad){printf("   -> FAIL: scale not decoded as E8M0\n");++fails;}
  }
  // TEST 5: negative FP4. code 10 = -1.0 -> every output = -128.
  {
    auto o=run(0xAA,0x38,127,127);
    int bad=0; for(float v:o) if(v!=-128.0f) ++bad;
    printf("T5 A=-1.0 (E2M1 code 10): out[0]=%.4f expect -128.0  mismatches=%d/256\n",o[0],bad);
    if(bad){printf("   -> FAIL: sign bit of FP4 mishandled\n");++fails;}
  }
  printf(fails?"\nFAIL: %d/5 checks failed\n":"\nPASS: all 5 checks exact\n",fails);
  return fails?1:0;
}
