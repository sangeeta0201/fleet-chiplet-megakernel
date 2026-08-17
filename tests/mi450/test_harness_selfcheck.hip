#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
constexpr int M=16, K=512, N=64;
typedef short v4s __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
// gfx950 MFMA reference GEMM with the SAME semantics: C[m][n]=sum_k A[m][k]*B[n][k]
__global__ __launch_bounds__(256) void mfma_gemm(const __bf16* A, const __bf16* B, float* O){
  int lane = threadIdx.x % 64, warp = threadIdx.x / 64;
  int m_idx = lane % 16, koff = (lane/16)*4;
  v4f acc = {0,0,0,0};
  for (int k0=0; k0<K; k0+=16){
    v4s a,b;
    for(int i=0;i<4;i++){
      a[i] = *(const short*)&A[m_idx*K + k0 + koff + i];
      b[i] = *(const short*)&B[(warp*16 + m_idx)*K + k0 + koff + i];
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,b,acc,0,0,0);
  }
  // MFMA 16x16x16 C layout: lane l holds C[(l/16)*4 + i][l%16]
  for(int i=0;i<4;i++){
    int row = (lane/16)*4 + i, col = warp*16 + (lane%16);
    if(row<M && col<N) O[row*N+col] = acc[i];
  }
}
int main(){
  std::vector<__bf16> hA(M*K), hB(N*K);
  for(int i=0;i<M*K;i++) hA[i]=(__bf16)(((i*37)%17-8)*0.125f);
  for(int i=0;i<N*K;i++) hB[i]=(__bf16)(((i*53)%13-6)*0.25f);
  std::vector<float> ref(M*N,0.f);
  for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float a=0; for(int k=0;k<K;k++) a+=(float)hA[m*K+k]*(float)hB[n*K+k]; ref[m*N+n]=a; }
  __bf16 *dA,*dB; float* dO;
  hipMalloc(&dA,M*K*2); hipMalloc(&dB,N*K*2); hipMalloc(&dO,M*N*4);
  hipMemcpy(dA,hA.data(),M*K*2,hipMemcpyHostToDevice);
  hipMemcpy(dB,hB.data(),N*K*2,hipMemcpyHostToDevice);
  hipMemset(dO,0,M*N*4);
  hipLaunchKernelGGL(mfma_gemm,dim3(1),dim3(256),0,0,dA,dB,dO);
  hipDeviceSynchronize();
  std::vector<float> hO(M*N); hipMemcpy(hO.data(),dO,M*N*4,hipMemcpyDeviceToHost);
  int bad=0; float worst=0;
  for(int i=0;i<M*N;i++){ float d=fabsf(hO[i]-ref[i])/fmaxf(fabsf(ref[i]),1.f); if(d>worst)worst=d; if(d>0.03f)bad++; }
  printf("harness self-check on gfx950 MFMA: worst rel=%.5f  bad=%d/%d  %s\n", worst,bad,M*N, bad?"FAIL":"PASS");
  return bad?1:0;
}
