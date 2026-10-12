/* MPK_TOPK_2L bit-exactness test.
 *
 * Runs the fused router's TopK (gang_rmsnorm_topk_detail::topk_ll_noinline,
 * the K-split + MPK_ROUTE_LL path) as eight blocks, one per XCD, on seeded
 * partial logits, sums of squares and correction biases laid out exactly as
 * the router tiles leave them, and dumps every routing output of every XCD's
 * copy. Build it twice, -DMPK_TOPK_2L=0 and -DMPK_TOPK_2L=1, and cmp the two
 * dumps: the two-level TopK must reproduce the one-level one bit for bit.
 *
 * The cases include forced ties (logits and biases repeated across experts,
 * so keys collide in their 24 score bits and the id decides) and run on
 * successive epochs, so each case's words sit beside the previous case's
 * stale ones.
 *
 * Usage: test_topk_2l <dump file>; exit 0 when all eight copies agree.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <random>
#include <vector>

#ifndef MPK_NT
#define MPK_NT (blockDim.x)
#endif
// The worker-state tracers the persistent kernel defines; no-ops here.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
#define MPK_WS_MARK(code, aux) ((void)0)
#define MPK_WS_PHASE(phase, layer, xcd) ((void)0)
#define MPK_WS_WAIT_BEGIN(barrier_id, expected) ((void)0)
#define MPK_WS_WAIT_TICK(observed, spins) ((void)0)
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3) ((void)0)
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

static constexpr int NE = 256;
static constexpr int K = 8;
static constexpr int RLL_INTS = MPK_RLL_WORD_LINES * 16 + 8 * MPK_RLL_XCD_INTS;
static constexpr int KS_INTS = 8 * MPK_XSPLIT_XCD_INTS;

__global__ void __launch_bounds__(256, 1)
    topk_entry(int *rll, unsigned epoch, void *bias, int const *ks_all) {
  kernel::gang_rmsnorm_topk_detail::topk_ll_noinline<__hip_bfloat16, NE, K,
                                                      true>(
      rll, (int)blockIdx.x, epoch, bias, 1, true, 2.5f, 1, ks_all, 6144.0f);
}

#define CK(x)                                                                  \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__);         \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

static unsigned short f2bf(float f) {
  unsigned u;
  memcpy(&u, &f, 4);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s <dump>\n", argv[0]);
    return 2;
  }
  FILE *out = fopen(argv[1], "wb");
  int *d_rll, *d_ks;
  void *d_bias;
  CK(hipMalloc(&d_rll, RLL_INTS * 4));
  CK(hipMalloc(&d_ks, KS_INTS * 4));
  CK(hipMalloc(&d_bias, NE * 2));
  CK(hipMemset(d_rll, 0, RLL_INTS * 4));
  CK(hipMemset(d_ks, 0, KS_INTS * 4));
  std::vector<int> ks(KS_INTS, 0), rll(RLL_INTS);
  std::vector<unsigned short> bias(NE);
  constexpr int CASES = 400;
  int bad = 0;
  for (int c = 0; c < CASES; c++) {
    std::mt19937 rng(1234 + c);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    unsigned const epoch = (unsigned)(c + 1);
    int const mode = c % 4;  // 0 plain, 1 tied logits, 2 tied biases, 3 both
    float part[NE][16], ssq[16];
    for (int t = 0; t < 16; t++) {
      ssq[t] = 0.5f + std::fabs(nd(rng)) * 40.0f;
    }
    for (int e = 0; e < NE; e++) {
      int const src = (mode & 1) ? (e % 37) : e;
      std::mt19937 r2(99991 * (c + 1) + src);
      std::normal_distribution<float> n2(0.0f, 1.0f);
      for (int t = 0; t < 16; t++) {
        part[e][t] = n2(r2) * 3.0f;
      }
      int const bsrc = (mode & 2) ? (e % 5) : e;
      std::mt19937 r3(7777 * (c + 1) + bsrc);
      std::normal_distribution<float> n3(0.0f, 1.0f);
      bias[e] = f2bf(n3(r3) * 0.3f);
    }
    for (int x = 0; x < 8; x++) {
      int *xb = ks.data() + x * MPK_XSPLIT_XCD_INTS;
      for (int t = 0; t < 16; t++) {
        xb[t] = (int)epoch;  // the normed flags
      }
      unsigned long long *blk = reinterpret_cast<unsigned long long *>(
          xb + MPK_XSPLIT_FLAG_LINES * 16);
      for (int t = 0; t < 16; t++) {
        for (int w = 0; w < 32; w++) {
          unsigned u;
          memcpy(&u, &part[x * 32 + w][t], 4);
          blk[t * 64 + w] = ((unsigned long long)epoch << 32) | u;
        }
        unsigned u;
        memcpy(&u, &ssq[t], 4);
        blk[t * 64 + 32] = ((unsigned long long)epoch << 32) | u;
      }
    }
    CK(hipMemcpy(d_ks, ks.data(), KS_INTS * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_bias, bias.data(), NE * 2, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(topk_entry, dim3(8), dim3(256), 0, 0, d_rll, epoch,
                       d_bias, d_ks);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(rll.data(), d_rll, RLL_INTS * 4, hipMemcpyDeviceToHost));
#if MPK_TOPK_2L
    if (c == 0 && getenv("TK2L_DEBUG")) {
      std::vector<int> kd(KS_INTS);
      CK(hipMemcpy(kd.data(), d_ks, KS_INTS * 4, hipMemcpyDeviceToHost));
      for (int x = 0; x < 8; x++) {
        unsigned long long const *cd = reinterpret_cast<unsigned long long const *>(
            kd.data() + x * MPK_XSPLIT_XCD_INTS + MPK_XSPLIT_FLAG_LINES * 16) + 16 * 64;
        printf("xcd %d cands:", x);
        for (int r = 0; r < 8; r++) {
          printf(" [e%u k%08x p%04x ep%u]", (unsigned)(cd[2 * r] & 0xFF),
                 (unsigned)cd[2 * r], (unsigned)(cd[2 * r + 1] & 0xFFFF),
                 (unsigned)(cd[2 * r] >> 32));
        }
        printf("\n");
      }
    }
#endif
    unsigned ref[2 * (K + 1) + (K + 1) + 2];
    for (int x = 0; x < 8; x++) {
      int const *xb = rll.data() + MPK_RLL_WORD_LINES * 16 + x * MPK_RLL_XCD_INTS;
      unsigned long long const *llr = reinterpret_cast<unsigned long long const *>(
          xb + 32 + MPK_RLL_IDS_LINES * 16);
      unsigned v[2 * (K + 1) + (K + 1) + 2];
      int n = 0;
      for (int i = 0; i <= K; i++) {
        if ((unsigned)(llr[i] >> 32) != epoch ||
            (unsigned)(llr[16 + i] >> 32) != epoch) {
          printf("case %d xcd %d slot %d: stale routing word\n", c, x, i);
          bad++;
        }
        v[n++] = (unsigned)llr[i];
        v[n++] = (unsigned)llr[16 + i];
      }
      for (int i = 0; i <= K; i++) {
        v[n++] = (unsigned)xb[16 + i];
      }
      v[n++] = (unsigned)xb[32 + K];
      v[n++] = (unsigned)xb[32 + NE + 1];
      if (x == 0) {
        memcpy(ref, v, sizeof(v));
        fwrite(v, sizeof(v), 1, out);
      } else if (memcmp(ref, v, sizeof(v)) != 0) {
        printf("case %d: xcd %d copy differs from xcd 0\n", c, x);
        bad++;
      }
    }
  }
  fclose(out);
  printf("%s: %d cases, %d problems (MPK_TOPK_2L=%d)\n",
         bad ? "FAIL" : "PASS", CASES, bad, (int)MPK_TOPK_2L);
  return bad ? 1 : 0;
}
