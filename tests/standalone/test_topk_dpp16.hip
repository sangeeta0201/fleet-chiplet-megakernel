// Equivalence: 16-lane TopK xor butterfly vs DPP row_shl + __shfl(*,0,16).
// Mirrors moe_topk_softmax_mi300.cuh MPK_TOPK_DPP_REDUCE.
//
//   hipcc -O2 --offload-arch=gfx950 test_topk_dpp16.hip -o /tmp/t_dpp16
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

__device__ __forceinline__ void dpp16_max(float &mx) {
  float mx_p = mx;
  asm volatile("s_nop 1\n"
               "v_mov_b32_dpp %[mx_p], %[mx] row_shl:8 row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %[mx], %[mx], %[mx_p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[mx_p], %[mx] row_shl:4 row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %[mx], %[mx], %[mx_p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[mx_p], %[mx] row_shl:2 row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %[mx], %[mx], %[mx_p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[mx_p], %[mx] row_shl:1 row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %[mx], %[mx], %[mx_p]"
               : [mx] "+v"(mx), [mx_p] "+v"(mx_p));
}

__device__ __forceinline__ void dpp16_sum(float &sum) {
  float sum_p = sum;
  asm volatile(
      "s_nop 1\n"
      "v_mov_b32_dpp %[sum_p], %[sum] row_shl:8 row_mask:0xf bank_mask:0xf "
      "bound_ctrl:1\n"
      "v_add_f32 %[sum], %[sum], %[sum_p]\n"
      "s_nop 1\n"
      "v_mov_b32_dpp %[sum_p], %[sum] row_shl:4 row_mask:0xf bank_mask:0xf "
      "bound_ctrl:1\n"
      "v_add_f32 %[sum], %[sum], %[sum_p]\n"
      "s_nop 1\n"
      "v_mov_b32_dpp %[sum_p], %[sum] row_shl:2 row_mask:0xf bank_mask:0xf "
      "bound_ctrl:1\n"
      "v_add_f32 %[sum], %[sum], %[sum_p]\n"
      "s_nop 1\n"
      "v_mov_b32_dpp %[sum_p], %[sum] row_shl:1 row_mask:0xf bank_mask:0xf "
      "bound_ctrl:1\n"
      "v_add_f32 %[sum], %[sum], %[sum_p]"
      : [sum] "+v"(sum), [sum_p] "+v"(sum_p));
}

__device__ __forceinline__ void dpp16_argmax(float &mx, int &ex) {
  float om = mx;
  int oe = ex;
  asm volatile(
      "s_nop 1\n"
      "v_mov_b32_dpp %[om], %[mx] row_shl:8 row_mask:0xf bank_mask:0xf\n"
      "v_mov_b32_dpp %[oe], %[ex] row_shl:8 row_mask:0xf bank_mask:0xf\n"
      "v_cmp_gt_f32 vcc, %[om], %[mx]\n"
      "v_cndmask_b32 %[mx], %[mx], %[om], vcc\n"
      "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
      "s_nop 1\n"
      "v_mov_b32_dpp %[om], %[mx] row_shl:4 row_mask:0xf bank_mask:0xf\n"
      "v_mov_b32_dpp %[oe], %[ex] row_shl:4 row_mask:0xf bank_mask:0xf\n"
      "v_cmp_gt_f32 vcc, %[om], %[mx]\n"
      "v_cndmask_b32 %[mx], %[mx], %[om], vcc\n"
      "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
      "s_nop 1\n"
      "v_mov_b32_dpp %[om], %[mx] row_shl:2 row_mask:0xf bank_mask:0xf\n"
      "v_mov_b32_dpp %[oe], %[ex] row_shl:2 row_mask:0xf bank_mask:0xf\n"
      "v_cmp_gt_f32 vcc, %[om], %[mx]\n"
      "v_cndmask_b32 %[mx], %[mx], %[om], vcc\n"
      "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
      "s_nop 1\n"
      "v_mov_b32_dpp %[om], %[mx] row_shl:1 row_mask:0xf bank_mask:0xf\n"
      "v_mov_b32_dpp %[oe], %[ex] row_shl:1 row_mask:0xf bank_mask:0xf\n"
      "v_cmp_gt_f32 vcc, %[om], %[mx]\n"
      "v_cndmask_b32 %[mx], %[mx], %[om], vcc\n"
      "v_cndmask_b32 %[ex], %[ex], %[oe], vcc"
      : [mx] "+v"(mx), [ex] "+v"(ex), [om] "+v"(om), [oe] "+v"(oe)
      :
      : "vcc");
}

// mask_hi: if true, only lanes 0-15 of each wave run (bs=1 TopK shape).
// mm[0]=max all, mm[1]=max lane0, mm[2]=sum all, mm[3]=arg all, mm[4]=arg lane0
__global__ void compare(float const *__restrict__ vals,
                        int const *__restrict__ ids,
                        int n_groups,
                        int mask_hi,
                        unsigned *__restrict__ mm) {
  int const lane = threadIdx.x & 63;
  int const wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;
  if (wave >= n_groups)
    return;
  if (mask_hi && lane >= 16)
    return;

  int const g = lane & 15;
  float x = vals[wave * 16 + g];
  int id = ids[wave * 16 + g];

  float bx = x, dx = x;
  for (int mask = 8; mask > 0; mask /= 2) {
    float other = __shfl_xor(bx, mask, 16);
    asm volatile("v_max_f32 %[mx], %[mx], %[ot]"
                 : [mx] "+v"(bx)
                 : [ot] "v"(other));
  }
  dpp16_max(dx);
  dx = __shfl(dx, 0, 16);
  if (__float_as_uint(bx) != __float_as_uint(dx)) {
    atomicAdd(mm + 0, 1u);
    if (g == 0)
      atomicAdd(mm + 1, 1u);
  }

  float bs = x, ds = x;
  for (int mask = 8; mask > 0; mask /= 2) {
    bs += __shfl_xor(bs, mask, 16);
  }
  dpp16_sum(ds);
  ds = __shfl(ds, 0, 16);
  if (__float_as_uint(bs) != __float_as_uint(ds))
    atomicAdd(mm + 2, 1u);

  float bm = x, dm = x;
  int bi = id, di = id;
  for (int mask = 8; mask > 0; mask /= 2) {
    float om = __shfl_xor(bm, mask, 16);
    int oe = __shfl_xor(bi, mask, 16);
    asm volatile("v_cmp_gt_f32 vcc, %[om], %[mv]\n"
                 "v_cndmask_b32 %[mv], %[mv], %[om], vcc\n"
                 "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
                 : [mv] "+v"(bm), [ex] "+v"(bi)
                 : [om] "v"(om), [oe] "v"(oe)
                 : "vcc");
  }
  dpp16_argmax(dm, di);
  dm = __shfl(dm, 0, 16);
  di = __shfl(di, 0, 16);
  if (__float_as_uint(bm) != __float_as_uint(dm) || bi != di) {
    atomicAdd(mm + 3, 1u);
    if (g == 0)
      atomicAdd(mm + 4, 1u);
  }
}

int main() {
  constexpr int N = 4096;
  float *h = new float[N * 16];
  int *hid = new int[N * 16];
  for (int g = 0; g < N; ++g) {
    for (int i = 0; i < 16; ++i) {
      // Finite logits, including negatives and exact ties.
      int k = (g * 16 + i);
      float f = ((k % 23) - 11) * 0.25f;
      if ((g & 3) == 0 && (i & 1) == 0)
        f = -1.0f; // ties
      h[g * 16 + i] = f;
      hid[g * 16 + i] = i;
    }
  }
  float *d;
  int *did;
  unsigned *mm;
  (void)hipMalloc(&d, N * 16 * sizeof(float));
  (void)hipMalloc(&did, N * 16 * sizeof(int));
  (void)hipMalloc(&mm, 10 * sizeof(unsigned));
  (void)hipMemcpy(d, h, N * 16 * sizeof(float), hipMemcpyHostToDevice);
  (void)hipMemcpy(did, hid, N * 16 * sizeof(int), hipMemcpyHostToDevice);
  (void)hipMemset(mm, 0, 10 * sizeof(unsigned));

  compare<<<N, 64>>>(d, did, N, 0, mm + 0);
  compare<<<N, 64>>>(d, did, N, 1, mm + 5);
  (void)hipDeviceSynchronize();
  unsigned hmm[10];
  (void)hipMemcpy(hmm, mm, 10 * sizeof(unsigned), hipMemcpyDeviceToHost);
  printf("full-wave  max=%u max_l0=%u sum=%u arg=%u arg_l0=%u\n",
         hmm[0], hmm[1], hmm[2], hmm[3], hmm[4]);
  printf("mask-hi16  max=%u max_l0=%u sum=%u arg=%u arg_l0=%u\n",
         hmm[5], hmm[6], hmm[7], hmm[8], hmm[9]);
  unsigned tot = 0;
  for (int i = 0; i < 10; ++i)
    tot += hmm[i];
  return tot ? 1 : 0;
}
