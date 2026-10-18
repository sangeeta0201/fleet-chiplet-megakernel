#pragma once
// MPK_EAGLE3: the LMSYS EAGLE3 draft head for gpt-oss-120b, run on every worker at the end of the LM-head
// task (MPK_SPEC_VSLOT passes). Each pass extends the draft's KV cache over the rows whose target features
// are final (every slot of a prefill pass, the accepted slots of a verify pass); when the next pass is a
// verify pass, the last row also runs attention + MLP + LM head and the next draft token is stored in
// g_e3.draft_out for prepare_next_batch.
//
// Target features (EAGLE3 aux layers 1, 17, 33) need no capture: under MPK_LAYER_RING the hidden state after
// layer L is attn_proj_out ring copy L plus the four slots of moe_workspace_f32 ring copy L+1, each written
// once per pass and still intact when the LM head runs.
//
// The ring copies are read at system scope (rewritten every pass). Scratch vectors are written
// write-through once per pass before any read and read with plain loads: at QKV_BATCH_SIZE > 1 every
// worker invalidates L2 (wave 0) and its vL1 at each layer boundary, so no copy from an earlier pass
// reaches the draft phase. KV rows are written once (only accepted positions are extended).

#ifndef MPK_MULTI_DEV  // MDEV_SPEC: a shared buffer (persistent_kernel.cuh) under MPK_MULTI_DEV
extern __device__ __attribute__((aligned(16))) int g_spec_state[4];
#endif
#ifdef MPK_E3_STATE1
// g_spec_state is written once per pass by prepare_next_batch; every LM-head worker reads it. One read per
// worker (two 64-bit system-scope loads), passed to both draft functions by value.
struct E3St {
  int st0, n, s, kind;
};
__device__ __forceinline__ E3St e3_state() {
  unsigned long long *const p = (unsigned long long *)(g_spec_state);
#ifdef MPK_E3_STATE_PLAIN
  // The task's dependency check ran an agent-scope acquire on this XCD after prepare_next_batch's release.
  // Not volatile: on gfx940+ a volatile access gets sc0 sc1, i.e. the same system-scope load.
  unsigned long long const a = p[0];
  unsigned long long const b = p[1];
#else
  unsigned long long const a = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  unsigned long long const b = __hip_atomic_load(p + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
  return E3St{(int)(unsigned)a, (int)(unsigned)(a >> 32), (int)(unsigned)b, (int)(unsigned)(b >> 32)};
}
#define E3_SPEC(i) ((i) == 0 ? e3st.st0 : (i) == 1 ? e3st.n : (i) == 2 ? e3st.s : e3st.kind)
#else
#define E3_SPEC(i) __hip_atomic_load((int *)&g_spec_state[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
#endif
#ifndef MPK_MULTI_DEV
extern __device__ long long g_spec_prev_in[MPK_MAX_NUM_BATCHED_TOKENS];
#endif
#ifdef MPK_E3_CONF  // persistent_kernel.cuh (host-set on every device)
extern __device__ float g_spec_conf;
extern __device__ int g_spec_kmin;
#ifdef MPK_E3_CONF_D
extern __device__ float g_spec_confd;
#endif
#endif

namespace kernel {

constexpr int E3_H = 2880;
constexpr int E3_HP = 2944;  // padded hidden row of the target's ring buffers
constexpr int E3_FEAT = 3 * E3_H;
constexpr int E3_IN2 = 2 * E3_H;
constexpr int E3_NQ = 32, E3_NKV = 8, E3_HD = 128;
constexpr int E3_FF = 16384, E3_VOCAB = 32000;
// MPK_E3_INLINE: the draft's entry points inlined into the LM-head kernel (no callee-saved register save/restore).
#ifdef MPK_E3_INLINE
#define MPK_E3_CALL __forceinline__
#else
#define MPK_E3_CALL __noinline__
#endif
constexpr int E3_R = MPK_MAX_NUM_BATCHED_TOKENS;  // extend rows per pass
static_assert(E3_R <= 8, "MPK_EAGLE3: scratch sized for 8 rows (the block <= 4 layout for 4)");
// g_aid_nc_rep (first 64 KB cleared per launch): 8 per-die arrival lines, the global line, the final-reduce
// counter. g_aid_flag_rep (2 MiB cleared per launch): the release generation.
constexpr int E3_NC_INTS = 14336;
constexpr int E3_FLAG_INTS = 516096;
constexpr int E3_REP_INTS = 327680;  // scratch vectors, one copy per AID replica
#ifdef MPK_AID_SPLIT_ATTNOUT
#error "MPK_EAGLE3 keeps its scratch in the replica window MPK_AID_SPLIT_ATTNOUT uses"
#endif
// scratch (float offsets into g_e3.vec)
#if MPK_MAX_NUM_BATCHED_TOKENS > 4
// past block 4: 8 extend rows; base 163840 floats (demo.py follows -DMPK_VTN_T)
constexpr int E3_V_AUX = 0;       // target features, 8 rows of E3_FEAT
constexpr int E3_V_H = 69120;     // fc outputs, 8 rows of E3_H
constexpr int E3_V_Q = 92160;     // last row's q after RoPE
constexpr int E3_V_A = 96256;     // attention output
constexpr int E3_V_R = 100352;    // o_proj + residual
constexpr int E3_V_F = 104448;    // silu(gate) * up
constexpr int E3_V_O = 120832;    // down + residual
constexpr int E3_V_PV = 124928;   // LM-head partial values (per worker)
constexpr int E3_V_PI = 125184;   // LM-head partial indices (int bits)
constexpr int E3_V_YV = 125440;   // target argmax partials copied by the LM head: [8][256] bf16 bits
constexpr int E3_V_YI = 127488;   // [8][256] indices
constexpr int E3_V_AP = 129792;   // MPK_E3_ATTN_SPLIT2: [32][SP][130] partial out, max, sum
constexpr int E3_V_FLOATS = 163840;
constexpr int E3_V_SPAN = 160000;  // MPK_EAGLE3_TS: 3 stamps per worker
static_assert(E3_V_AUX + 8 * E3_FEAT <= E3_V_H && E3_V_H + 8 * E3_H <= E3_V_Q, "block > 4 scratch");
#if !defined(MPK_E3_PRO4) || !defined(MPK_E3_Q8M)
#error "MPK_EAGLE3 past block 4 needs MPK_E3_PRO4 + MPK_E3_Q8M (row groups of 4 in stage 2)"
#endif
#else
constexpr int E3_V_AUX = 0;      // target features, E3_R rows of E3_FEAT
constexpr int E3_V_H = 36864;    // fc outputs, E3_R rows of E3_H
constexpr int E3_V_Q = 49152;    // last row's q after RoPE
constexpr int E3_V_A = 53248;    // attention output
constexpr int E3_V_R = 57344;    // o_proj + residual
constexpr int E3_V_F = 61440;    // silu(gate) * up
constexpr int E3_V_O = 77824;    // down + residual
constexpr int E3_V_PV = 81920;   // LM-head partial values (per worker)
constexpr int E3_V_PI = 82176;   // LM-head partial indices (int bits)
constexpr int E3_V_YV = 82432;   // target argmax partials copied by the LM head: [4][256] bf16 bits
constexpr int E3_V_YI = 83456;   // [4][256] indices
constexpr int E3_V_AP = 98304;      // MPK_E3_ATTN_SPLIT2: [32][SP][130] partial out, max, sum
constexpr int E3_V_FLOATS = 131072;
constexpr int E3_V_SPAN = 90000;  // MPK_EAGLE3_TS: 3 stamps per worker
static_assert(E3_REP_INTS + E3_V_FLOATS <= 458752, "EAGLE3 scratch runs into the EVCTR mirrors");
#endif
// MPK_E3_CHAIN=K: K chained drafts per pass. Step 0 is the draft from the last accepted row; step j >= 1 feeds
// draft j's token and step j-1's output hidden at the next draft position (its k / v row is speculative: the
// next pass rewrites it from target features). Readers use plain loads of scratch written once per pass, so
// each step >= 1 has its own copy of the single-row regions (q, attention partials, r2, act, out, LM partials).
#ifndef MPK_E3_CHAIN
#define MPK_E3_CHAIN 1
#endif
constexpr int E3_K = MPK_E3_CHAIN;
static_assert(E3_K >= 1 && E3_K <= 7, "MPK_E3_CHAIN: 1..7 drafts (draft_out slots 0..6, the step marker in 7)");
#if defined(MPK_E3_ATTN_GQA) && defined(MPK_E3_ATTN_SPLIT2)
#error "MPK_E3_ATTN_GQA replaces MPK_E3_ATTN_SPLIT2"
#endif
#if MPK_E3_CHAIN > 1
#if !defined(MPK_E3_Q8M) || !defined(MPK_E3_Q4M) || \
    (!defined(MPK_E3_ATTN_SPLIT2) && !defined(MPK_E3_ATTN_GQA)) || defined(MPK_E3_PRO3) || defined(MPK_E3_PIPE)
#error "MPK_E3_CHAIN > 1 needs MPK_E3_Q8M + MPK_E3_Q4M + MPK_E3_ATTN_SPLIT2 or _GQA (and not MPK_E3_PRO3 / MPK_E3_PIPE)"
#endif
#endif
#if (defined(MPK_E3_IDLE_SKIP) || defined(MPK_E3_PRO4) || defined(MPK_E3_QKV_AID)) && !defined(MPK_E3_Q8M)
#error "MPK_E3_IDLE_SKIP / MPK_E3_PRO4 / MPK_E3_QKV_AID build on MPK_E3_Q8M"
#endif
#if defined(MPK_E3_PRO4) && defined(MPK_E3_PRO3)
#error "MPK_E3_PRO4 replaces MPK_E3_PRO3"
#endif
constexpr int E3_V_CH0 = E3_V_FLOATS, E3_CS = 65536;  // chain step j >= 1: [E3_V_CH0 + (j - 1) * E3_CS, +E3_CS)
constexpr int E3_CQ = 0, E3_CR = 4096, E3_CF = 8192, E3_CO = 24576, E3_CPV = 28672, E3_CPI = 28928, E3_CAP = 29184;
constexpr int E3_CA = 58368;  // MPK_E3_ATTN_GQA: the step's attention output
static_assert(E3_CAP + E3_NQ * 7 * 130 <= E3_CA && E3_CA + E3_NQ * E3_HD <= E3_CS, "chain step block");
constexpr int E3_V_TOTAL = E3_V_CH0 + (E3_K - 1) * E3_CS;
// MPK_E3_CONF: per-worker LM-head sums of exp(v - max) (step 0 after the target argmax partials, steps >= 1 after
// the step block's attention output), and the drafts' probabilities in draft_out[E3_CONF_SLOT ..] (16 slots).
constexpr int E3_V_PS = E3_R <= 4 ? 84480 : 129536, E3_CPS = 62464, E3_CONF_SLOT = 8;
#if defined(MPK_TAIL_NO111) || defined(MPK_TAIL_NO111_CHECK)
// MPK_TAIL_NO111: each row's target argmax, with argmax_reduce's (task 111's) tie order, published by worker 0 in
// draft_out[E3_TGT_SLOT + j] (demo.py allocates 32 slots); the virtual-slot prepare reads it there, so task 111
// leaves the graph. _CHECK keeps task 111 and also publishes the lower-worker tie rule's argmax at E3_TGT_SLOT + 8.
constexpr int E3_TGT_SLOT = 16;
static_assert(E3_CONF_SLOT + E3_K <= E3_TGT_SLOT && E3_TGT_SLOT + 8 + E3_R <= 32, "target argmax slots in draft_out");
#if !defined(MPK_E3_ACC1)
#error "MPK_TAIL_NO111 builds on MPK_E3_ACC1"
#endif
// argmax_reduce over partial p (one per thread, 4 waves): a wave's shfl_down tournament keeps the lane with the
// smallest 6-bit-reversed index on a tie, and wave 0's tournament over the 4 wave winners the smallest 2-bit-reversed
// wave. So its winner is the max value with the smallest key(p).
__device__ __forceinline__ int e3_k111(int p) {
  return (int)(((__brev((unsigned)(p >> 6)) >> 30) << 6) | (__brev((unsigned)(p & 63)) >> 26));
}
#endif
static_assert(E3_V_YI + (E3_R <= 4 ? 4 : 8) * 256 <= E3_V_PS && E3_V_PS + 256 <= E3_V_AP, "step 0 LM sum partials");
static_assert(E3_CA + E3_NQ * E3_HD <= E3_CPS && E3_CPS + 256 <= E3_CS, "chain step LM sum partials");
static_assert(E3_CONF_SLOT + E3_K <= 16, "probability slots inside draft_out (demo.py allocates 16)");
#if defined(MPK_E3_CONF) && (!defined(MPK_E3_Q4M) || defined(MPK_E3_PIPE))
#error "MPK_E3_CONF: the MPK_E3_Q4M LM head"
#endif
// MPK_E3_ATTN_GQA partials, one block per chain step after the step blocks: [group][split][q head][128 out, max, sum].
#ifdef MPK_E3_ATTN_GQA
constexpr int E3_GQA_SZ = E3_NKV * MPK_E3_ATTN_GQA * (E3_NQ / E3_NKV) * 130;
#else
constexpr int E3_GQA_SZ = 0;
#endif
constexpr int E3_V_GQA0 = E3_V_TOTAL;
static_assert(E3_V_TOTAL + E3_K * E3_GQA_SZ <= 1048576, "EAGLE3 scratch (demo.py sizes it from the same terms)");
__device__ __forceinline__ int e3_reg(int base0, int j, int coff) {
  return j == 0 ? base0 : E3_V_CH0 + (j - 1) * E3_CS + coff;
}

struct MpkEagle3 {
  unsigned short const *fc, *wq, *wk, *wv, *wo, *wg, *wu, *wd, *wlm;
  unsigned short const *n_hidden, *n_input, *n_post, *n_final, *embed;
  long long const *d2t;
  unsigned short *kc, *vc;  // [maxpos][8][128] bf16
  float *vec;
  long long *draft_out;     // [0] next draft, [1] pass start step it was made in
  unsigned short const *apo[3];
  float const *ws[3];
  long long const *tokens;  // request 0's token row
  int const *prompt_len;
  int ws_stride, maxpos, nworkers, enabled;
  // MPK_EAGLE3_FP8: per-row scales of fc, wq, wk, wv, wo, wg, wu, wd, wlm (weights then e4m3).
  float const *sfc, *swq, *swk, *swv, *swo, *swg, *swu, *swd, *swlm;
};
__constant__ MpkEagle3 g_e3;

__device__ __forceinline__ float e3_bf(unsigned b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ unsigned e3_tobf(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float e3_rbf(float f) { return e3_bf(e3_tobf(f)); }
__device__ __forceinline__ unsigned e3_ld_sys_u32(void const *p) {
  return __hip_atomic_load((unsigned const *)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float e3_ld_sys_f32(float const *p) {
  return __uint_as_float(e3_ld_sys_u32(p));
}
__device__ __forceinline__ long long e3_ld_sys_i64(long long const *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void e3_st_sys_u32(void *p, unsigned v) {
  __hip_atomic_store((unsigned *)p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void e3_st_sys_f32(float *p, float v) { e3_st_sys_u32(p, __float_as_uint(v)); }

// Scratch: one write-through store; readers use plain loads (see the header note on why that is sound).
__device__ __forceinline__ void e3_st2(int off, unsigned v) { e3_st_sys_u32(g_e3.vec + off, v); }
__device__ __forceinline__ void e3_st2f(int off, float v) { e3_st2(off, __float_as_uint(v)); }
#ifdef MPK_E3_FEATH
// bf16 bits b as element i of the bf16 array at float offset off of the scratch (write-through, as e3_st2)
__device__ __forceinline__ void e3_st2h(int off, int i, unsigned b) {
  unsigned short *const p = reinterpret_cast<unsigned short *>(g_e3.vec + off) + i;
  asm volatile("global_store_short %0, %1, off sc0 sc1" ::"v"(p), "v"(b) : "memory");
}
#endif
__device__ __forceinline__ float const *e3_rep_local() { return g_e3.vec; }

__device__ __forceinline__ float e3_wave_sum(float v) {
#ifdef MPK_E3_DPPSUM
  // Valid in lane 0 only, which is all any caller reads: the shuffles' 32/16/8/4/2/1 tree on
  // permlane32/16 swaps and row_shl DPP adds (see rmsnorm_wave_sum_to_lane_zero), bit-identical at lane 0.
  float peer = v;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "v_mov_b32_e32 %[p], %[s]\n"
               "s_nop 1\n"
               "v_permlane16_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1"
               : [s] "+v"(v), [p] "+v"(peer));
  return v;
#else
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    v += __shfl_xor(v, o, 64);
  }
  return v;
#endif
}
__device__ __forceinline__ float e3_wave_max(float v) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    v = fmaxf(v, __shfl_xor(v, o, 64));
  }
  return v;
}
// Block-wide sum / max over 256 threads; red holds 4 floats.
__device__ __forceinline__ float e3_block_sum(float v, float *red) {
  v = e3_wave_sum(v);
  __syncthreads();
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = v;
  __syncthreads();
  return red[0] + red[1] + red[2] + red[3];
}
__device__ __forceinline__ float e3_block_max(float v, float *red) {
  v = e3_wave_max(v);
  __syncthreads();
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = v;
  __syncthreads();
  return fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
}
// The same for four values per thread (the four q heads of a KV group); red holds 16 floats.
__device__ __forceinline__ void e3_block_max4(float (&v)[4], float *red) {
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = e3_wave_max(v[k]);
  __syncthreads();
  if ((threadIdx.x & 63) == 0)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[(threadIdx.x >> 6) * 4 + k] = v[k];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = fmaxf(fmaxf(red[k], red[4 + k]), fmaxf(red[8 + k], red[12 + k]));
}
__device__ __forceinline__ void e3_block_sum4(float (&v)[4], float *red) {
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = e3_wave_sum(v[k]);
  __syncthreads();
  if ((threadIdx.x & 63) == 0)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[(threadIdx.x >> 6) * 4 + k] = v[k];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = red[k] + red[4 + k] + red[8 + k] + red[12 + k];
}

// Block argmax: every thread holds (v, key); returns in all threads the key of the max value, ties to
// the lower key. red holds 8 floats.
__device__ __forceinline__ int e3_block_argmax(float v, int key, float *red) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    float const ov = __shfl_xor(v, o, 64);
    int const ok = __shfl_xor(key, o, 64);
    if (ov > v || (ov == v && ok < key)) {
      v = ov;
      key = ok;
    }
  }
  __syncthreads();
  if ((threadIdx.x & 63) == 0) {
    red[(threadIdx.x >> 6) * 2] = v;
    red[(threadIdx.x >> 6) * 2 + 1] = __int_as_float(key);
  }
  __syncthreads();
  float bv = red[0];
  int bk = __float_as_int(red[1]);
#pragma unroll
  for (int w4 = 1; w4 < 4; ++w4) {
    float const ov = red[w4 * 2];
    int const ok = __float_as_int(red[w4 * 2 + 1]);
    if (ov > bv || (ov == bv && ok < bk)) {
      bv = ov;
      bk = ok;
    }
  }
  __syncthreads();
  return bk;
}

#ifdef MPK_E3_CONF
// e3_block_argmax that also returns the max value (every thread).
__device__ __forceinline__ int e3_block_argmax_v(float v, int key, float *red, float &vmax) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    float const ov = __shfl_xor(v, o, 64);
    int const ok = __shfl_xor(key, o, 64);
    if (ov > v || (ov == v && ok < key)) {
      v = ov;
      key = ok;
    }
  }
  __syncthreads();
  if ((threadIdx.x & 63) == 0) {
    red[(threadIdx.x >> 6) * 2] = v;
    red[(threadIdx.x >> 6) * 2 + 1] = __int_as_float(key);
  }
  __syncthreads();
  float bv = red[0];
  int bk = __float_as_int(red[1]);
#pragma unroll
  for (int w4 = 1; w4 < 4; ++w4) {
    float const ov = red[w4 * 2];
    int const ok = __float_as_int(red[w4 * 2 + 1]);
    if (ov > bv || (ov == bv && ok < bk)) {
      bv = ov;
      bk = ok;
    }
  }
  __syncthreads();
  vmax = bv;
  return bk;
}
// The draft argmax's softmax probability from the per-worker partials (max m, sum sw of exp(v - m); sw = 0 for no
// rows): 1 / sum_w sw exp(m - M), M the global max. All threads (one worker's partial each); the same value on
// every worker.
__device__ __forceinline__ float e3_conf_p(float m, float sw, float M, float *red) {
  float const t = sw > 0.f ? sw * __expf(m - M) : 0.f;
  return 1.f / e3_block_sum(t, red);
}
#endif

// All nworkers workers: per-die NC counter, the die's last arrival bumps the global counter, the global last
// arrival publishes `gen` into both AID flag replicas; every worker polls the replica of its own physical
// AID (an RW replica line is coherent only inside its AID).
#ifdef MPK_EAGLE3_TS
// LM-head task, worker 0: entry, after the draft features, before the draft (10 ns ticks).
__device__ unsigned long long g_e3lm_ts[3];
#ifdef MPK_LMT_TS
__device__ unsigned long long g_lmt_ts[8];  // LM head: norm, quant, group 0, tiles, epilogue
#endif
#endif
__device__ __forceinline__ void e3_barrier(int gen, int die) {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    int const per_die = g_e3.nworkers / MPK_NUM_XCDS;
    int *const xc = g_aid_nc_rep[die >> 2] + E3_NC_INTS + die * 64;
    int const old = __hip_atomic_fetch_add(xc, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#ifdef MPK_E3_FLATBAR
    static_assert(MPK_NUM_XCDS == 8, "one slot per die, 8 slots per batched poll");
    if (old % per_die == per_die - 1) {
      e3_st_sys_u32(g_aid_flag_rep[0] + E3_FLAG_INTS + 16 * die, (unsigned)gen);
      e3_st_sys_u32(g_aid_flag_rep[1] + E3_FLAG_INTS + 16 * die, (unsigned)gen);
    }
    unsigned xcc;
    MPK_GET_XCC_ID32(xcc);
    int *const fl = g_aid_flag_rep[(xcc & 7u) >> 2] + E3_FLAG_INTS;
    while (ld_aid_min8_s32(fl) < gen) {
      __builtin_amdgcn_s_sleep(1);
    }
#else
    if (old % per_die == per_die - 1) {
      int *const gc = g_aid_nc_rep[0] + E3_NC_INTS + 8 * 64;
      int const gold = __hip_atomic_fetch_add(gc, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (gold % MPK_NUM_XCDS == MPK_NUM_XCDS - 1) {
        e3_st_sys_u32(g_aid_flag_rep[0] + E3_FLAG_INTS, (unsigned)gen);
        e3_st_sys_u32(g_aid_flag_rep[1] + E3_FLAG_INTS, (unsigned)gen);
      }
    }
    unsigned xcc;
    MPK_GET_XCC_ID32(xcc);
    int const *const fl = g_aid_flag_rep[(xcc & 7u) >> 2] + E3_FLAG_INTS;
    while ((int)e3_ld_sys_u32(fl) < gen) {
      __builtin_amdgcn_s_sleep(1);
    }
#endif
  }
  __syncthreads();
}

// Rows [o0, o1) of N for worker wid of W. MPK_E3_AID_ROWS: rows [0, N/2) on dies 0-3 (wid < W/2),
// [N/2, N) on dies 4-7, matching the draft weights' midpoint-cut BOs (MPK_E3_OWN_BO).
__device__ __forceinline__ void e3_split(int N, int wid, int W, int &o0, int &o1) {
#ifdef MPK_E3_AID_ROWS
  int const WH = W >> 1, NH = N >> 1;
  int const aid = wid >= WH ? 1 : 0;
  int const per = (NH + WH - 1) / WH;
  o0 = aid * NH + min(NH, (wid - aid * WH) * per);
  o1 = aid * NH + min(NH, (wid - aid * WH) * per + per);
#else
  int const per = (N + W - 1) / W;
  o0 = wid * per;
  o1 = min(N, o0 + per);
#endif
}

// Wave-level dots of NW bf16 weight rows against R f32 LDS rows of length K (row stride xs floats).
template <int K, int NW, int R>
__device__ __forceinline__ void e3_dots(unsigned short const *const (&w)[NW], float const *x, int xs,
                                        float (&acc)[NW][R]) {
  static_assert(K % 8 == 0, "16-byte weight chunks");
  constexpr int NC = K / 8;
  int const lane = threadIdx.x & 63;
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
#pragma unroll 4
  for (int c = lane; c < NC; c += 64) {
    float xv[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float4 const p = *reinterpret_cast<float4 const *>(x + r * xs + c * 8);
      float4 const q = *reinterpret_cast<float4 const *>(x + r * xs + c * 8 + 4);
      xv[r][0] = p.x; xv[r][1] = p.y; xv[r][2] = p.z; xv[r][3] = p.w;
      xv[r][4] = q.x; xv[r][5] = q.y; xv[r][6] = q.z; xv[r][7] = q.w;
    }
#pragma unroll
    for (int a = 0; a < NW; ++a) {
      uint4 const v = reinterpret_cast<uint4 const *>(w[a])[c];
      float const f[8] = {e3_bf(v.x & 0xFFFFu), e3_bf(v.x >> 16), e3_bf(v.y & 0xFFFFu), e3_bf(v.y >> 16),
                          e3_bf(v.z & 0xFFFFu), e3_bf(v.z >> 16), e3_bf(v.w & 0xFFFFu), e3_bf(v.w >> 16)};
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[a][r] += f[e] * xv[r][e];
    }
  }
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] = e3_wave_sum(acc[a][r]);
}

#ifdef MPK_EAGLE3_FP8
// e4m3 rows: 16 weights per 16-byte chunk; the caller applies each row's scale to its sum.
template <int K, int NW, int R>
__device__ __forceinline__ void e3_dots8(unsigned char const *const (&w)[NW], float const *x, int xs,
                                         float (&acc)[NW][R]) {
  static_assert(K % 16 == 0, "16-byte weight chunks");
  constexpr int NC = K / 16;
  int const lane = threadIdx.x & 63;
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
#ifdef MPK_E3_LOADS_FIRST
  constexpr int IT = (NC + 63) / 64;
  constexpr int BI = (NW * IT <= 24) ? IT : ((24 / NW) > 0 ? (24 / NW) : 1);
#pragma unroll
  for (int i0 = 0; i0 < IT; i0 += BI) {
    uint4 vv[BI][NW];
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      int const c = lane + (i0 + i) * 64;
      bool const ok = (i0 + i) < IT && c < NC;
#pragma unroll
      for (int a = 0; a < NW; ++a) {
        vv[i][a] = ok ? reinterpret_cast<uint4 const *>(w[a])[c] : uint4{0u, 0u, 0u, 0u};
      }
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      int const c = lane + (i0 + i) * 64;
      if ((i0 + i) >= IT || c >= NC) continue;
#ifdef MPK_E3_NOFMA
#pragma unroll
      for (int a = 0; a < NW; ++a) {
        uint4 const v = vv[i][a];
        acc[a][0] += __uint_as_float((v.x ^ v.y ^ v.z ^ v.w) & 0x007fffffu);
      }
      continue;
#endif
#ifdef MPK_E3_DOT2
      typedef __bf16 e3bf2 __attribute__((ext_vector_type(2)));
      e3bf2 xp[R][8];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float4 const p = *reinterpret_cast<float4 const *>(x + r * xs + c * 16 + q * 4);
          xp[r][q * 2 + 0] = e3bf2{(__bf16)p.x, (__bf16)p.y};
          xp[r][q * 2 + 1] = e3bf2{(__bf16)p.z, (__bf16)p.w};
        }
#pragma unroll
      for (int a = 0; a < NW; ++a) {
        uint4 const v = vv[i][a];
        unsigned const d[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          e3bf2 const wlo = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(d[q], 1.0f, false);
          e3bf2 const whi = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(d[q], 1.0f, true);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            acc[a][r] = __builtin_amdgcn_fdot2_f32_bf16(wlo, xp[r][q * 2 + 0], acc[a][r], false);
            acc[a][r] = __builtin_amdgcn_fdot2_f32_bf16(whi, xp[r][q * 2 + 1], acc[a][r], false);
          }
        }
      }
      continue;
#endif
      float xv[R][16];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float4 const p = *reinterpret_cast<float4 const *>(x + r * xs + c * 16 + q * 4);
          xv[r][q * 4 + 0] = p.x; xv[r][q * 4 + 1] = p.y; xv[r][q * 4 + 2] = p.z; xv[r][q * 4 + 3] = p.w;
        }
#pragma unroll
      for (int a = 0; a < NW; ++a) {
        uint4 const v = vv[i][a];
        unsigned const d[4] = {v.x, v.y, v.z, v.w};
        float f[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          auto const lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)d[q], false);
          auto const hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)d[q], true);
          f[q * 4 + 0] = lo[0]; f[q * 4 + 1] = lo[1]; f[q * 4 + 2] = hi[0]; f[q * 4 + 3] = hi[1];
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int e = 0; e < 16; ++e) acc[a][r] += f[e] * xv[r][e];
      }
    }
  }
#else
#pragma unroll 2
  for (int c = lane; c < NC; c += 64) {
    float xv[R][16];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float4 const p = *reinterpret_cast<float4 const *>(x + r * xs + c * 16 + q * 4);
        xv[r][q * 4 + 0] = p.x; xv[r][q * 4 + 1] = p.y; xv[r][q * 4 + 2] = p.z; xv[r][q * 4 + 3] = p.w;
      }
#pragma unroll
    for (int a = 0; a < NW; ++a) {
      uint4 const v = reinterpret_cast<uint4 const *>(w[a])[c];
      unsigned const d[4] = {v.x, v.y, v.z, v.w};
      float f[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        auto const lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)d[q], false);
        auto const hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)d[q], true);
        f[q * 4 + 0] = lo[0]; f[q * 4 + 1] = lo[1]; f[q * 4 + 2] = hi[0]; f[q * 4 + 3] = hi[1];
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[a][r] += f[e] * xv[r][e];
    }
  }
#endif
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] = e3_wave_sum(acc[a][r]);
}
#endif

#ifdef MPK_E3_FP4
#ifndef MPK_EAGLE3_FP8
#error "MPK_E3_FP4 replaces four of the MPK_EAGLE3_FP8 projections"
#endif
#if defined(MPK_E3_PIPE) || defined(MPK_E3_LMG8)
#error "MPK_E3_FP4 has no MPK_E3_PIPE / MPK_E3_LMG8 path"
#endif
// MXFP4 rows: e2m1 values, value 2j in the low nibble of byte j; one E8M0 scale per 32 values. Lane l takes
// 16-value (8-byte) chunks c = l + 64 i; a row's scales are stored lane-pair major, byte (l >> 1) * ITP + i
// = the scale of block (l >> 1) + 32 i, which covers chunk l + 64 i.
template <int K>
struct E3Q4 {
  static constexpr int NC = K / 16;
  static constexpr int IT = (NC + 63) / 64;
  static constexpr int ITP = (IT + 3) & ~3;
  static constexpr int SROW = 32 * ITP;  // scale bytes per row
};
template <int K, int NW, int R>
__device__ __forceinline__ void e3_dots4(unsigned char const *const (&w)[NW], unsigned char const *const (&sc)[NW],
                                         float const *x, int xs, float (&acc)[NW][R]) {
  using Q = E3Q4<K>;
  static_assert(K % 32 == 0, "32-value blocks");
  constexpr int NC = Q::NC, IT = Q::IT, ITP = Q::ITP;
  typedef __bf16 e3bf2 __attribute__((ext_vector_type(2)));
  int const lane = threadIdx.x & 63;
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] = 0.f;
  unsigned sw[NW][ITP / 4];
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int j = 0; j < ITP / 4; ++j) sw[a][j] = reinterpret_cast<unsigned const *>(sc[a] + (lane >> 1) * ITP)[j];
  constexpr int BI = (NW * IT <= 24) ? IT : ((24 / NW) > 0 ? (24 / NW) : 1);
#pragma unroll
  for (int i0 = 0; i0 < IT; i0 += BI) {
    uint2 vv[BI][NW];
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      int const c = lane + (i0 + i) * 64;
      bool const ok = (i0 + i) < IT && c < NC;
#pragma unroll
      for (int a = 0; a < NW; ++a) vv[i][a] = ok ? reinterpret_cast<uint2 const *>(w[a])[c] : uint2{0u, 0u};
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      int const it = i0 + i;
      int const c = lane + it * 64;
      if (it >= IT || c >= NC) continue;
      e3bf2 xp[R][8];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float4 const p = *reinterpret_cast<float4 const *>(x + r * xs + c * 16 + q * 4);
          xp[r][q * 2 + 0] = e3bf2{(__bf16)p.x, (__bf16)p.y};
          xp[r][q * 2 + 1] = e3bf2{(__bf16)p.z, (__bf16)p.w};
        }
#pragma unroll
      for (int a = 0; a < NW; ++a) {
        float const sv = __uint_as_float(((sw[a][it >> 2] >> (8 * (it & 3))) & 0xffu) << 23);
        unsigned const d[2] = {vv[i][a].x, vv[i][a].y};
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          e3bf2 const w0 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(d[q], sv, 0);
          e3bf2 const w1 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(d[q], sv, 1);
          e3bf2 const w2 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(d[q], sv, 2);
          e3bf2 const w3 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(d[q], sv, 3);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            acc[a][r] = __builtin_amdgcn_fdot2_f32_bf16(w0, xp[r][q * 4 + 0], acc[a][r], false);
            acc[a][r] = __builtin_amdgcn_fdot2_f32_bf16(w1, xp[r][q * 4 + 1], acc[a][r], false);
            acc[a][r] = __builtin_amdgcn_fdot2_f32_bf16(w2, xp[r][q * 4 + 2], acc[a][r], false);
            acc[a][r] = __builtin_amdgcn_fdot2_f32_bf16(w3, xp[r][q * 4 + 3], acc[a][r], false);
          }
        }
      }
    }
  }
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] = e3_wave_sum(acc[a][r]);
}
#endif

#ifdef MPK_E3_Q4M  // MPK_E3_Q4M v2: K-major tiles
#ifndef MPK_EAGLE3_FP8
#error "MPK_E3_Q4M replaces four of the MPK_EAGLE3_FP8 projections"
#endif
#if defined(MPK_E3_PIPE) || defined(MPK_E3_LMG8) || defined(MPK_E3_FP4)
#error "MPK_E3_Q4M has no MPK_E3_PIPE / MPK_E3_LMG8 / MPK_E3_FP4 path"
#endif
typedef int __attribute__((ext_vector_type(4))) e3_i32x4;
typedef int __attribute__((ext_vector_type(8))) e3_i32x8;
typedef float __attribute__((ext_vector_type(4))) e3_f32x4;
typedef short __attribute__((ext_vector_type(2))) e3_fp8x2;
constexpr int E3_HQ = 2944;  // E3_H padded to the 128-wide MFMA K tile
// K-major tile strides: data K tiles * 1 KB, scales ceil(K tiles / 4) * 256 B
template <int KP> struct E3QT {
  static constexpr int KT = KP / 128;
  static constexpr int DATA = KT * 1024;
  static constexpr int SC = (KT + 3) / 4 * 256;
};
// N items over W workers, sizes differing by at most one.
__device__ __forceinline__ void e3_bsplit(int N, int wid, int W, int &t0, int &t1) {
  int const base = N / W, rem = N % W;
  t0 = wid * base + min(wid, rem);
  t1 = t0 + base + (wid < rem ? 1 : 0);
}
// MPK_E3_Q4M v5. x (LDS f32, K values, K % 8 == 0) -> e4m3 q8 (LDS, KP bytes, zero padded) with one E8M0
// scale per 128 values in qs, the target's MFMA activation format. 8 values per lane: a 128-value block is one
// 16-lane DPP row, its amax 4 DPP max steps. All threads; ends in a barrier.
__device__ __forceinline__ float e3_dpp_max(float v, int ctrl_sel) {
  int const u = __float_as_int(v);
  int r;
  switch (ctrl_sel) {  // DPP controls must be compile-time constants
    case 0: r = __builtin_amdgcn_mov_dpp(u, 0xB1, 0xf, 0xf, false); break;   // quad_perm [1,0,3,2]
    case 1: r = __builtin_amdgcn_mov_dpp(u, 0x4E, 0xf, 0xf, false); break;   // quad_perm [2,3,0,1]
    case 2: r = __builtin_amdgcn_mov_dpp(u, 0x141, 0xf, 0xf, false); break;  // row_half_mirror
    default: r = __builtin_amdgcn_mov_dpp(u, 0x140, 0xf, 0xf, false); break; // row_mirror
  }
  return fmaxf(v, __int_as_float(r));
}
// R rows of x (row stride xs floats, K valid values each) -> q8 [r][KP] e4m3 + qs [r][KP / 128] E8M0. Rows
// flattened into one pass; 4 chunks of 8 values per thread per batch, all LDS reads of a batch issued first.
// All threads; ends in a barrier.
template <int K, int KP>
__device__ __forceinline__ void e3_q8act_r(float const *__restrict__ x, int xs, int R, unsigned char *__restrict__ q8,
                                           unsigned char *__restrict__ qs) {
  static_assert(K % 8 == 0 && KP % 128 == 0, "8-value lanes, 128-value blocks");
  constexpr int NL = KP / 8, B = 4;
  int const tid = threadIdx.x, tot = R * NL;
#pragma unroll 1
  for (int b0 = 0; b0 < tot; b0 += 256 * B) {
    float4 v[B][2];
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      bool const ok = l < tot && k < K;
      v[j][0] = ok ? *reinterpret_cast<float4 const *>(x + r * xs + k) : make_float4(0.f, 0.f, 0.f, 0.f);
      v[j][1] = ok ? *reinterpret_cast<float4 const *>(x + r * xs + k + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      float4 const v0 = v[j][0], v1 = v[j][1];
      float amax = fmaxf(fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w))),
                         fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
      amax = e3_dpp_max(amax, 0);
      amax = e3_dpp_max(amax, 1);
      amax = e3_dpp_max(amax, 2);
      amax = e3_dpp_max(amax, 3);
      unsigned se = 0;
      if (amax != 0.f) {
        unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
        se = min(255u, ((u >> 23) & 0xffu) + ((u & 0x7fffffu) ? 1u : 0u));
      }
      float const sf = se ? __uint_as_float(se << 23) : 1.0f;
      if (l < tot) {
        e3_fp8x2 p0 = {0, 0}, p1 = {0, 0};
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.x, v0.y, sf, false);
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.z, v0.w, sf, true);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.x, v1.y, sf, false);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.z, v1.w, sf, true);
        *reinterpret_cast<e3_fp8x2 *>(q8 + r * KP + k) = p0;
        *reinterpret_cast<e3_fp8x2 *>(q8 + r * KP + k + 4) = p1;
        if ((l & 15) == 0) qs[r * (KP / 128) + (k >> 7)] = (unsigned char)se;
      }
    }
  }
  __syncthreads();
}
template <int K, int KP>
__device__ __forceinline__ void e3_q8act(float const *x, unsigned char *q8, unsigned char *qs) {
  e3_q8act_r<K, KP>(x, 0, 1, q8, qs);
}
#ifdef MPK_E3_FEATH
// e3_q8act_r on R bf16 rows of global x (row stride xs elements), loaded straight into registers (8 values = 16
// bytes per lane, B chunks per thread per batch, all of a batch's loads issued first): the same values, the same
// arithmetic, so the same e4m3 rows and scales. All threads; ends in a barrier.
template <int K, int KP, int B>
__device__ __forceinline__ void e3_q8g_bf(unsigned short const *__restrict__ x, int xs, int R,
                                          unsigned char *__restrict__ q8, unsigned char *__restrict__ qs) {
  static_assert(K % 8 == 0 && KP % 128 == 0, "8-value lanes, 128-value blocks");
  constexpr int NL = KP / 8;
  int const tid = threadIdx.x, tot = R * NL;
#pragma unroll 1
  for (int b0 = 0; b0 < tot; b0 += 256 * B) {
    uint4 v[B];
#pragma unroll
    for (int j = 0; j < B; ++j) {  // branch-free: clamped address, then select
      int const l = b0 + j * 256 + tid, lc = min(l, tot - 1);
      int const r = lc / NL, k0 = 8 * (lc - r * NL), k = min(k0, K - 8);
      uint4 const t = *reinterpret_cast<uint4 const *>(x + (size_t)r * xs + k);
      v[j] = (l < tot && k0 < K) ? t : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      float4 const v0 = make_float4(e3_bf(v[j].x & 0xFFFFu), e3_bf(v[j].x >> 16), e3_bf(v[j].y & 0xFFFFu),
                                    e3_bf(v[j].y >> 16));
      float4 const v1 = make_float4(e3_bf(v[j].z & 0xFFFFu), e3_bf(v[j].z >> 16), e3_bf(v[j].w & 0xFFFFu),
                                    e3_bf(v[j].w >> 16));
      float amax = fmaxf(fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w))),
                         fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
      amax = e3_dpp_max(amax, 0);
      amax = e3_dpp_max(amax, 1);
      amax = e3_dpp_max(amax, 2);
      amax = e3_dpp_max(amax, 3);
      unsigned se = 0;
      if (amax != 0.f) {
        unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
        se = min(255u, ((u >> 23) & 0xffu) + ((u & 0x7fffffu) ? 1u : 0u));
      }
      float const sf = se ? __uint_as_float(se << 23) : 1.0f;
      if (l < tot) {
        e3_fp8x2 p0 = {0, 0}, p1 = {0, 0};
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.x, v0.y, sf, false);
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.z, v0.w, sf, true);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.x, v1.y, sf, false);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.z, v1.w, sf, true);
        *reinterpret_cast<e3_fp8x2 *>(q8 + r * KP + k) = p0;
        *reinterpret_cast<e3_fp8x2 *>(q8 + r * KP + k + 4) = p1;
        if ((l & 15) == 0) qs[r * (KP / 128) + (k >> 7)] = (unsigned char)se;
      }
    }
  }
  __syncthreads();
}
#endif
// MPK_E3_Q4M v4. NT 16-row MXFP4 tiles against the e4m3 activation in LDS. K-major tiles: wr[t] = the tile's
// data (K tile kt, lane l: 16 bytes at (kt * 64 + l) * 16), sr[t] = its scales (byte ((kt >> 2) * 64 + l) * 4
// + (kt & 3)); lane l = (l >> 4) K group, (l & 15) row. NKT 128-wide K tiles in groups of D: the loop body
// computes group gi and loads group gi + 1 into the same registers (no branches, no copies); the last group and
// the NKT % D tail are static. Result: acc[t][i] in the lanes with (lane & 15) == 0 is row (lane >> 4) * 4 + i.
#ifdef MPK_E3_WPF
// MPK_E3_WPF: the first D K tiles of up to 3 MXFP4 tiles, loaded ahead (e3_pre4) and handed to e3_mgemv<..., true>.
template <int D>
struct E3Pre4 {
  e3_i32x4 a[D][3];
  int sa[D][3];
};
template <int D, int NT>
__device__ __forceinline__ void e3_pre4(unsigned char const *const (&wr)[3], unsigned char const *const (&sr)[3],
                                        E3Pre4<D> &p) {
  int const lane = threadIdx.x & 63;
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      p.a[d][t] = *reinterpret_cast<e3_i32x4 const *>(wr[t] + (d * 64 + lane) * 16);
      p.sa[d][t] = sr[t][((d >> 2) * 64 + lane) * 4 + (d & 3)];
    }
}
#endif
template <int NKT, int NT, int D, bool PRE = false, int DP = 4>
__device__ __forceinline__ void e3_mgemv(unsigned char const *const (&wr)[3], unsigned char const *const (&sr)[3],
                                         unsigned char const *q8, unsigned char const *qs, e3_f32x4 (&acc)[3]
#ifdef MPK_E3_WPF
                                         , void const *prev = nullptr
#endif
                                         ) {
  static_assert(NKT >= D, "at least one full group");
  constexpr int G = NKT / D, TL = NKT % D;
  int const lane = threadIdx.x & 63, g = lane >> 4;
  e3_i32x4 a[D][NT];
  int sa[D][NT];
  auto ld = [&](int kt, int d, int t) {
    a[d][t] = *reinterpret_cast<e3_i32x4 const *>(wr[t] + (kt * 64 + lane) * 16);
    sa[d][t] = sr[t][((kt >> 2) * 64 + lane) * 4 + (kt & 3)];
  };
  auto mma = [&](int kt, int d) {
    e3_i32x4 const blo = *reinterpret_cast<e3_i32x4 const *>(q8 + kt * 128 + g * 16);
    e3_i32x4 const bhi = *reinterpret_cast<e3_i32x4 const *>(q8 + kt * 128 + 64 + g * 16);
    e3_i32x8 const b = {blo[0], blo[1], blo[2], blo[3], bhi[0], bhi[1], bhi[2], bhi[3]};
    int sb;  // VGPR on purpose: an SGPR scale operand reads as 0 (mfma_fp4_test.hip); nop = VALU -> MFMA states
    asm volatile("v_mov_b32 %0, %1\n\ts_nop 2" : "=v"(sb) : "v"((int)qs[kt]));
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      e3_i32x8 const av = {a[d][t][0], a[d][t][1], a[d][t][2], a[d][t][3], 0, 0, 0, 0};
      acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, b, acc[t], 4, 0, 0, sa[d][t], 0, sb);
    }
  };
#pragma unroll
  for (int t = 0; t < NT; ++t) acc[t] = e3_f32x4{0.f, 0.f, 0.f, 0.f};
#ifdef MPK_E3_WPF
  if constexpr (PRE) {
    static_assert(DP == D, "the preload holds D K tiles");
    E3Pre4<D> const *const pre = static_cast<E3Pre4<D> const *>(prev);
#pragma unroll
    for (int d = 0; d < D; ++d)
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        a[d][t] = pre->a[d][t];
        sa[d][t] = pre->sa[d][t];
      }
  } else
#endif
  {
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int t = 0; t < NT; ++t) ld(d, d, t);
  }
#pragma unroll 1
  for (int gi = 0; gi < G - 1; ++gi) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      int const kt = gi * D + d;
      mma(kt, d);
#pragma unroll
      for (int t = 0; t < NT; ++t) ld(kt + D, d, t);
    }
  }
#pragma unroll
  for (int d = 0; d < D; ++d) {  // last full group, loading the tail
    mma((G - 1) * D + d, d);
    if (d < TL)
#pragma unroll
      for (int t = 0; t < NT; ++t) ld(G * D + d, d, t);
  }
#pragma unroll
  for (int d = 0; d < TL; ++d) mma(G * D + d, d);
}
// nv (1..3) tiles: one instantiation each, picked outside the loop.
template <int NKT>
__device__ __forceinline__ void e3_mgemv_nv(unsigned char const *const (&wr)[3], unsigned char const *const (&sr)[3],
                                            unsigned char const *q8, unsigned char const *qs, e3_f32x4 (&acc)[3],
                                            int nv) {
  if (nv >= 3)
    e3_mgemv<NKT, 3, 4>(wr, sr, q8, qs, acc);
  else if (nv == 2)
    e3_mgemv<NKT, 2, 4>(wr, sr, q8, qs, acc);
  else if (nv == 1)
    e3_mgemv<NKT, 1, 4>(wr, sr, q8, qs, acc);
}
#ifdef MPK_E3_WPF
template <int NKT>
__device__ __forceinline__ void e3_mgemv_nv_pre(unsigned char const *const (&wr)[3],
                                                unsigned char const *const (&sr)[3], unsigned char const *q8,
                                                unsigned char const *qs, e3_f32x4 (&acc)[3], int nv,
                                                E3Pre4<4> const &pre) {
  if (nv >= 3)
    e3_mgemv<NKT, 3, 4, true>(wr, sr, q8, qs, acc, &pre);
  else if (nv == 2)
    e3_mgemv<NKT, 2, 4, true>(wr, sr, q8, qs, acc, &pre);
  else if (nv == 1)
    e3_mgemv<NKT, 1, 4, true>(wr, sr, q8, qs, acc, &pre);
}
#endif
#endif

#ifdef MPK_E3_Q8M
#ifndef MPK_E3_Q4M
#error "MPK_E3_Q8M builds on MPK_E3_Q4M"
#endif
static_assert(E3_R <= 8, "MPK_E3_Q8M: the verify rows are MFMA columns 0..7 (fc / q-k-v reductions sized by E3_R)");
// e4m3-weight MFMA over NT 16-row tiles: wr[t] = the tile's first K tile (fragment kt at kt * 2048: half 0 then
// half 1, 1 KB each in lane order). The activation rows are columns: lane column n reads row min(n, R - 1) of
// q8 (row stride qst bytes) and its scale from qs (row stride qsst). Weight scale 1.0 (the per-row fp8 scale
// is applied by the caller). NKT K tiles in groups of D, branch-free loop as in e3_mgemv.
#ifdef MPK_E3_WPF
struct E3Pre8 {  // MPK_E3_WPF: the first 4 K tiles of one e4m3 tile (o_proj)
  e3_i32x4 lo[4], hi[4];
};
__device__ __forceinline__ void e3_pre8(unsigned char const *w, E3Pre8 &p) {
  int const lane = threadIdx.x & 63;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    p.lo[d] = *reinterpret_cast<e3_i32x4 const *>(w + d * 2048 + lane * 16);
    p.hi[d] = *reinterpret_cast<e3_i32x4 const *>(w + d * 2048 + 1024 + lane * 16);
  }
}
#endif
template <int NKT, int NT, int D, bool PRE = false>
__device__ __forceinline__ void e3_m8gemv(unsigned char const *const (&wr)[2], unsigned char const *q8, int qst,
                                          unsigned char const *qs, int qsst, int R, e3_f32x4 (&acc)[2]
#ifdef MPK_E3_WPF
                                          , void const *prev = nullptr
#endif
                                          ) {
  static_assert(NKT >= D && NT <= 2, "at least one group, at most 2 tiles");
  constexpr int G = NKT / D, TL = NKT % D;
  int const lane = threadIdx.x & 63, g = lane >> 4, rs = min(lane & 15, R - 1);
  unsigned char const *const qb = q8 + rs * qst;
  unsigned char const *const qsb = qs + rs * qsst;
  int sa1;  // weight scale E8M0 127 = 1.0 in a VGPR (an SGPR scale operand reads as 0)
  asm volatile("v_mov_b32 %0, 127\n\ts_nop 2" : "=v"(sa1));
  e3_i32x4 lo[D][NT], hi[D][NT];
  auto ld = [&](int kt, int d, int t) {
    lo[d][t] = *reinterpret_cast<e3_i32x4 const *>(wr[t] + kt * 2048 + lane * 16);
    hi[d][t] = *reinterpret_cast<e3_i32x4 const *>(wr[t] + kt * 2048 + 1024 + lane * 16);
  };
  auto mma = [&](int kt, int d) {
    e3_i32x4 const blo = *reinterpret_cast<e3_i32x4 const *>(qb + kt * 128 + g * 16);
    e3_i32x4 const bhi = *reinterpret_cast<e3_i32x4 const *>(qb + kt * 128 + 64 + g * 16);
    e3_i32x8 const b = {blo[0], blo[1], blo[2], blo[3], bhi[0], bhi[1], bhi[2], bhi[3]};
    int sb;
    asm volatile("v_mov_b32 %0, %1\n\ts_nop 2" : "=v"(sb) : "v"((int)qsb[kt]));
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      e3_i32x8 const av = {lo[d][t][0], lo[d][t][1], lo[d][t][2], lo[d][t][3],
                           hi[d][t][0], hi[d][t][1], hi[d][t][2], hi[d][t][3]};
      acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, b, acc[t], 0, 0, 0, sa1, 0, sb);
    }
  };
#pragma unroll
  for (int t = 0; t < NT; ++t) acc[t] = e3_f32x4{0.f, 0.f, 0.f, 0.f};
#ifdef MPK_E3_WPF
  if constexpr (PRE) {
    static_assert(D == 4 && NT == 1, "E3Pre8: 4 K tiles of one tile");
    E3Pre8 const *const pre = static_cast<E3Pre8 const *>(prev);
#pragma unroll
    for (int d = 0; d < D; ++d) {
      lo[d][0] = pre->lo[d];
      hi[d][0] = pre->hi[d];
    }
  } else
#endif
  {
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int t = 0; t < NT; ++t) ld(d, d, t);
  }
#pragma unroll 1
  for (int gi = 0; gi < G - 1; ++gi) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      int const kt = gi * D + d;
      mma(kt, d);
#pragma unroll
      for (int t = 0; t < NT; ++t) ld(kt + D, d, t);
    }
  }
#pragma unroll
  for (int d = 0; d < D; ++d) {
    mma((G - 1) * D + d, d);
    if (d < TL)
#pragma unroll
      for (int t = 0; t < NT; ++t) ld(G * D + d, d, t);
  }
#pragma unroll
  for (int d = 0; d < TL; ++d) mma(G * D + d, d);
}
// 180 row tiles of a 2880-row matrix on AID-local workers: AID0's half on workers 0..89, AID1's on W/2 .. W/2 + 89.
__device__ __forceinline__ int e3_tile180(int wid, int W) {
  constexpr int TH = E3_H / 16 / 2;
  int const WH = W >> 1;
  return WH >= TH ? (wid < WH ? (wid < TH ? wid : -1) : (wid - WH < TH ? TH + wid - WH : -1))
                  : (wid < 2 * TH ? wid : -1);
}
#endif

// One projection row: bf16 [rows][K], or e4m3 [rows][K] plus a per-row scale under MPK_EAGLE3_FP8.
struct E3Row {
  void const *p;
  float s;
};
__device__ __forceinline__ E3Row e3_row(unsigned short const *base, float const *sc, size_t row, int K) {
#ifdef MPK_EAGLE3_FP8
  return E3Row{(unsigned char const *)base + row * (size_t)K, sc[row]};
#else
  (void)sc;
  return E3Row{base + row * (size_t)K, 1.f};
#endif
}
template <int K, int NW, int R>
__device__ __forceinline__ void e3_dotsr(E3Row const (&w)[NW], float const *x, int xs, float (&acc)[NW][R]) {
#ifdef MPK_EAGLE3_FP8
  unsigned char const *wp[NW];
#pragma unroll
  for (int a = 0; a < NW; ++a) wp[a] = (unsigned char const *)w[a].p;
  unsigned char const *const (&wc)[NW] = wp;
  e3_dots8<K, NW, R>(wc, x, xs, acc);
#pragma unroll
  for (int a = 0; a < NW; ++a)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[a][r] *= w[a].s;
#else
  unsigned short const *wp[NW];
#pragma unroll
  for (int a = 0; a < NW; ++a) wp[a] = (unsigned short const *)w[a].p;
  unsigned short const *const (&wc)[NW] = wp;
  e3_dots<K, NW, R>(wc, x, xs, acc);
#endif
}

// RMSNorm of an f32 LDS row of E3_H (bf16 weight) into dst, rounded to bf16 as the reference does.
__device__ __forceinline__ void e3_rms_row(float const *src, float *dst, unsigned short const *wt,
                                           float *red) {
#ifdef MPK_E3_RMS_FAST
  constexpr int NCW = E3_H / 8;  // 16-byte gamma chunks
  static_assert(E3_H % 8 == 0, "gamma in 16-byte chunks");
  int const c0 = threadIdx.x, c1 = threadIdx.x + blockDim.x;
  uint4 const gw0 = c0 < NCW ? reinterpret_cast<uint4 const *>(wt)[c0] : make_uint4(0u, 0u, 0u, 0u);
  uint4 const gw1 = c1 < NCW ? reinterpret_cast<uint4 const *>(wt)[c1] : make_uint4(0u, 0u, 0u, 0u);
#endif
  float ss = 0.f;
  for (int i = threadIdx.x; i < E3_H; i += blockDim.x) ss += src[i] * src[i];
  float const inv = rsqrtf(e3_block_sum(ss, red) / (float)E3_H + 1e-5f);
#ifdef MPK_E3_RMS_FAST
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    int const c = u ? c1 : c0;
    if (c < NCW) {
      uint4 const g4 = u ? gw1 : gw0;
      unsigned const w[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int const i = c * 8 + e;
        unsigned const b = (e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xFFFFu);
        dst[i] = e3_rbf(src[i] * inv * e3_bf(b));
      }
    }
  }
#else
  for (int i = threadIdx.x; i < E3_H; i += blockDim.x) dst[i] = e3_rbf(src[i] * inv * e3_bf(wt[i]));
#endif
  __syncthreads();
}

// Copy n floats of the local scratch replica into LDS (n % 4 == 0). All threads.
constexpr int E3_LDU = 8;  // 16-byte loads in flight per thread in the copy loops
__device__ __forceinline__ void e3_load_rep(float *dst, float const *rv, int off, int n) {
  float4 const *const src = reinterpret_cast<float4 const *>(rv + off);
  float4 *const d4 = reinterpret_cast<float4 *>(dst);
  int const nc = n / 4;
  for (int c0 = threadIdx.x; c0 < nc; c0 += blockDim.x * E3_LDU) {
    float4 v[E3_LDU];
#pragma unroll
    for (int u = 0; u < E3_LDU; ++u) {
      int const c = c0 + u * blockDim.x;
      v[u] = c < nc ? src[c] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < E3_LDU; ++u) {
      int const c = c0 + u * blockDim.x;
      if (c < nc) d4[c] = v[u];
    }
  }
  __syncthreads();
}

// One bf16 row of E3_H (read-only weights) into LDS as f32. All threads.
__device__ __forceinline__ void e3_load_bf_row(float *dst, unsigned short const *row) {
  constexpr int NC = E3_H / 8;
  uint4 const *const src = reinterpret_cast<uint4 const *>(row);
  uint4 v[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    int const c = threadIdx.x + u * blockDim.x;
    v[u] = c < NC ? src[c] : make_uint4(0u, 0u, 0u, 0u);
  }
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    int const c = threadIdx.x + u * blockDim.x;
    if (c < NC) {
      float *const d = dst + c * 8;
      d[0] = e3_bf(v[u].x & 0xFFFFu); d[1] = e3_bf(v[u].x >> 16);
      d[2] = e3_bf(v[u].y & 0xFFFFu); d[3] = e3_bf(v[u].y >> 16);
      d[4] = e3_bf(v[u].z & 0xFFFFu); d[5] = e3_bf(v[u].z >> 16);
      d[6] = e3_bf(v[u].w & 0xFFFFu); d[7] = e3_bf(v[u].w >> 16);
    }
  }
  __syncthreads();
}

// Rows [o0, o1) of a projection (row length K) against R LDS rows: each wave takes E3_GROWS rows at a
// time, out(o, acc[r]) in lane 0, rows in increasing order within a wave.
constexpr int E3_GROWS = 4;
template <int K, int R, int KD = K, typename Out>
__device__ __forceinline__ void e3_gemv4(unsigned short const *wb, float const *sc, float const *x, int xs,
                                         int o0, int o1, Out &&out) {
  int const wave = threadIdx.x >> 6;
  for (int ob = o0 + wave * E3_GROWS; ob < o1; ob += 4 * E3_GROWS) {
    E3Row wr[E3_GROWS];
#pragma unroll
    for (int a = 0; a < E3_GROWS; ++a) wr[a] = e3_row(wb, sc, (size_t)min(ob + a, o1 - 1), K);
    float acc[E3_GROWS][R];
    e3_dotsr<KD, E3_GROWS, R>(wr, x, xs, acc);
    if ((threadIdx.x & 63) == 0) {
#pragma unroll
      for (int a = 0; a < E3_GROWS; ++a)
        if (ob + a < o1) out(ob + a, acc[a]);
    }
  }
}
#ifdef MPK_E3_FP4
// e3_gemv4 over MXFP4 rows: wb = e2m1 data [rows][K / 2], sc = E8M0 scales [rows][E3Q4<K>::SROW].
template <int K, int R, typename Out>
__device__ __forceinline__ void e3_gemv4q(unsigned short const *wb, float const *sc, float const *x, int xs,
                                          int o0, int o1, Out &&out) {
  int const wave = threadIdx.x >> 6;
  for (int ob = o0 + wave * E3_GROWS; ob < o1; ob += 4 * E3_GROWS) {
    unsigned char const *wp[E3_GROWS], *sp[E3_GROWS];
#pragma unroll
    for (int a = 0; a < E3_GROWS; ++a) {
      size_t const row = (size_t)min(ob + a, o1 - 1);
      wp[a] = (unsigned char const *)wb + row * (size_t)(K / 2);
      sp[a] = (unsigned char const *)sc + row * (size_t)E3Q4<K>::SROW;
    }
    unsigned char const *const (&wc)[E3_GROWS] = wp;
    unsigned char const *const (&scc)[E3_GROWS] = sp;
    float acc[E3_GROWS][R];
    e3_dots4<K, E3_GROWS, R>(wc, scc, x, xs, acc);
    if ((threadIdx.x & 63) == 0) {
#pragma unroll
      for (int a = 0; a < E3_GROWS; ++a)
        if (ob + a < o1) out(ob + a, acc[a]);
    }
  }
}
#endif
#ifdef MPK_E3_LMG8
// e3_gemv4 with G rows per wave pass (more loads in flight for short rows).
template <int K, int R, int G, typename Out>
__device__ __forceinline__ void e3_gemvg(unsigned short const *wb, float const *sc, float const *x, int xs,
                                         int o0, int o1, Out &&out) {
  int const wave = threadIdx.x >> 6;
  for (int ob = o0 + wave * G; ob < o1; ob += 4 * G) {
    E3Row wr[G];
#pragma unroll
    for (int a = 0; a < G; ++a) wr[a] = e3_row(wb, sc, (size_t)min(ob + a, o1 - 1), K);
    float acc[G][R];
    e3_dotsr<K, G, R>(wr, x, xs, acc);
    if ((threadIdx.x & 63) == 0) {
#pragma unroll
      for (int a = 0; a < G; ++a)
        if (ob + a < o1) out(ob + a, acc[a]);
    }
  }
}
#endif

#ifdef MPK_E3_PIPE
#ifndef MPK_EAGLE3_FP8
#error "MPK_E3_PIPE streams e4m3 rows (MPK_EAGLE3_FP8)"
#endif
// Groups g0 + wave, g0 + wave + 4, ... < g1 of NW e4m3 rows each (rowp(g, a) -> E3Row, row length K)
// against one f32 LDS row x; out(g, acc[NW]) in lane 0 with each row's scale applied. Group g+1's
// loads are in flight while group g computes.
template <int K, int NW, typename RowP, typename Out>
__device__ __forceinline__ void e3_gemv_pipe(int g0, int g1, RowP &&rowp, float const *x, Out &&out) {
  static_assert(K % 16 == 0, "16-byte weight chunks");
  constexpr int NC = K / 16;
  constexpr int IT = (NC + 63) / 64;
  static_assert(NW * IT <= 24, "two groups of loads in flight");
  int const lane = threadIdx.x & 63;
  int const wave = threadIdx.x >> 6;
  uint4 va[IT][NW], vb[IT][NW];
  float sa[NW], sb[NW];
  E3Row const r0 = rowp(g0, 0);  // a finished wave's dummy group re-reads this row (cache hit)
  auto issue = [&](int g, uint4 (&v)[IT][NW], float (&sc)[NW]) {
    bool const live = g < g1;
#pragma unroll
    for (int a = 0; a < NW; ++a) {
      E3Row const r = live ? rowp(g, a) : r0;
      sc[a] = r.s;
      uint4 const *const p = reinterpret_cast<uint4 const *>(r.p);
#pragma unroll
      for (int i = 0; i < IT; ++i) {
        int const c = lane + i * 64;
        v[i][a] = p[c < NC ? c : NC - 1];
      }
    }
  };
  auto compute = [&](int g, uint4 const (&v)[IT][NW], float const (&sc)[NW]) {
    float acc[NW];
#pragma unroll
    for (int a = 0; a < NW; ++a) acc[a] = 0.f;
#pragma unroll
    for (int i = 0; i < IT; ++i) {
      int const c = lane + i * 64;
      if (c >= NC) continue;
      float xv[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float4 const p4 = *reinterpret_cast<float4 const *>(x + c * 16 + q * 4);
        xv[q * 4 + 0] = p4.x; xv[q * 4 + 1] = p4.y; xv[q * 4 + 2] = p4.z; xv[q * 4 + 3] = p4.w;
      }
#pragma unroll
      for (int a = 0; a < NW; ++a) {
        unsigned const d[4] = {v[i][a].x, v[i][a].y, v[i][a].z, v[i][a].w};
        float f[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          auto const lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)d[q], false);
          auto const hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)d[q], true);
          f[q * 4 + 0] = lo[0]; f[q * 4 + 1] = lo[1]; f[q * 4 + 2] = hi[0]; f[q * 4 + 3] = hi[1];
        }
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[a] += f[e] * xv[e];
      }
    }
#pragma unroll
    for (int a = 0; a < NW; ++a) acc[a] = e3_wave_sum(acc[a]) * sc[a];
    if (lane == 0 && g < g1) out(g, acc);
  };
  int g = g0 + wave;
  issue(g, va, sa);
  for (; g < g1; g += 8) {
    issue(g + 4, vb, sb);
    compute(g, va, sa);
    if (g + 4 >= g1) break;
    issue(g + 8, va, sa);
    compute(g + 4, vb, sb);
  }
}
#endif

#ifdef MPK_E3_ATTN_GQA
// Stage 3 under MPK_E3_ATTN_GQA: one worker per (KV group, position split); each K / V row is read once for the
// group's four q heads, and the last split of a group to finish combines the group's partials into the attention
// output. noinline: inlined, its K / V loops raised the whole draft function to 256 VGPR + 120 AGPR + 800 B scratch.
__device__ __noinline__ void e3_attn_gqa(int const P, int const QOFF, int const GOFF, int const AOFF, int const wid,
                                         float const *const rv) {
  constexpr int SPG = MPK_E3_ATTN_GQA;  // position splits per KV group
  constexpr int QPG = E3_NQ / E3_NKV;   // q heads per KV group
  static_assert(QPG == 4 && E3_NKV * SPG <= 248, "MPK_E3_ATTN_GQA: one worker per (KV group, split), 4 q heads each");
  // From the symbol, not a parameter: through a generic pointer every LDS access would be a FLAT op.
  extern __shared__ char _rnlm_smem[];
  float *const lds = reinterpret_cast<float *>(_rnlm_smem);
  int const tid = threadIdx.x;
  int const g = wid / SPG, sp = wid % SPG;
  int const len = (P + 1 + SPG - 1) / SPG;
  int const t0 = sp * len, t1 = min(P + 1, t0 + len);
  int const lenp = (len + 63) & ~63;
  float *const qv = lds;                     // [4][128]
  float *const sc = lds + QPG * E3_HD;       // [4][lenp] scores, then exp weights
  float *const pv = sc + QPG * lenp;         // [4][16][128] partial V sums
  float *const red = pv + QPG * 16 * E3_HD;  // 16
  for (int i = tid; i < QPG * E3_HD; i += blockDim.x) qv[i] = rv[QOFF + QPG * g * E3_HD + i];
  __syncthreads();
  float const scale = rsqrtf((float)E3_HD);
  float lm[QPG] = {-3.0e38f, -3.0e38f, -3.0e38f, -3.0e38f};
  for (int t = t0 + tid; t < t1; t += blockDim.x) {
    uint4 const *kr = reinterpret_cast<uint4 const *>(g_e3.kc + ((size_t)t * E3_NKV + g) * E3_HD);
    uint4 kv[E3_HD / 8];
#pragma unroll
    for (int c = 0; c < E3_HD / 8; ++c) kv[c] = kr[c];
    float d[QPG] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < E3_HD / 8; ++c) {
      uint4 const v = kv[c];
      float const k8[8] = {e3_bf(v.x & 0xFFFFu), e3_bf(v.x >> 16), e3_bf(v.y & 0xFFFFu), e3_bf(v.y >> 16),
                           e3_bf(v.z & 0xFFFFu), e3_bf(v.z >> 16), e3_bf(v.w & 0xFFFFu), e3_bf(v.w >> 16)};
#pragma unroll
      for (int hq = 0; hq < QPG; ++hq) {
        float const *qq = qv + hq * E3_HD + c * 8;
        d[hq] += k8[0] * qq[0] + k8[1] * qq[1] + k8[2] * qq[2] + k8[3] * qq[3] + k8[4] * qq[4] +
                 k8[5] * qq[5] + k8[6] * qq[6] + k8[7] * qq[7];
      }
    }
#pragma unroll
    for (int hq = 0; hq < QPG; ++hq) {
      float const s = d[hq] * scale;
      sc[hq * lenp + t - t0] = s;
      lm[hq] = fmaxf(lm[hq], s);
    }
  }
  e3_block_max4(lm, red);
  float ls[QPG] = {0.f, 0.f, 0.f, 0.f};
  for (int t = t0 + tid; t < t1; t += blockDim.x) {
#pragma unroll
    for (int hq = 0; hq < QPG; ++hq) {
      float const e = expf(sc[hq * lenp + t - t0] - lm[hq]);
      sc[hq * lenp + t - t0] = e;
      ls[hq] += e;
    }
  }
  e3_block_sum4(ls, red);
  int const pg = tid >> 4, dc = tid & 15;  // 16 position groups x 16 lanes of 8 dims
  float a8[QPG][8];
#pragma unroll
  for (int hq = 0; hq < QPG; ++hq)
#pragma unroll
    for (int e = 0; e < 8; ++e) a8[hq][e] = 0.f;
#pragma unroll 8
  for (int t = t0 + pg; t < t1; t += 16) {
    uint4 const v = *reinterpret_cast<uint4 const *>(g_e3.vc + ((size_t)t * E3_NKV + g) * E3_HD + dc * 8);
    float const v8[8] = {e3_bf(v.x & 0xFFFFu), e3_bf(v.x >> 16), e3_bf(v.y & 0xFFFFu), e3_bf(v.y >> 16),
                         e3_bf(v.z & 0xFFFFu), e3_bf(v.z >> 16), e3_bf(v.w & 0xFFFFu), e3_bf(v.w >> 16)};
#pragma unroll
    for (int hq = 0; hq < QPG; ++hq) {
      float const pt = sc[hq * lenp + t - t0];
#pragma unroll
      for (int e = 0; e < 8; ++e) a8[hq][e] += pt * v8[e];
    }
  }
#pragma unroll
  for (int hq = 0; hq < QPG; ++hq)
#pragma unroll
    for (int e = 0; e < 8; ++e) pv[(hq * 16 + pg) * E3_HD + dc * 8 + e] = a8[hq][e];
  __syncthreads();
  float *const part = g_e3.vec + GOFF + (g * SPG + sp) * QPG * 130;
  for (int i = tid; i < QPG * E3_HD; i += blockDim.x) {
    int const hq = i >> 7, dd = i & 127;
    float sum = 0.f;
#pragma unroll
    for (int q = 0; q < 16; ++q) sum += pv[(hq * 16 + q) * E3_HD + dd];
    e3_st_sys_f32(part + hq * 130 + dd, sum);
  }
  if (tid < QPG) {
    e3_st_sys_f32(part + tid * 130 + 128, lm[tid]);
    e3_st_sys_f32(part + tid * 130 + 129, ls[tid]);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    int *const cnt = g_aid_nc_rep[0] + E3_NC_INTS + (10 + g) * 64;
    int const c = __hip_atomic_fetch_add(cnt, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    red[0] = __int_as_float(c % SPG == SPG - 1 ? 1 : 0);
  }
  __syncthreads();
  if (__float_as_int(red[0]) == 1) {
    // Every partial load of a thread is issued before any arithmetic (they come from memory, not this CU's cache).
    static_assert(QPG * E3_HD == 2 * 256, "two outputs per thread");
    float const *const gp = rv + GOFF + g * SPG * QPG * 130;
    float *const fac = sc;  // [4][SPG]
    if (tid < QPG) {
      float mx[SPG], sm[SPG];
#pragma unroll
      for (int q = 0; q < SPG; ++q) {
        mx[q] = gp[(q * QPG + tid) * 130 + 128];
        sm[q] = gp[(q * QPG + tid) * 130 + 129];
      }
      float M = -3.0e38f;
#pragma unroll
      for (int q = 0; q < SPG; ++q) M = fmaxf(M, mx[q]);
      float L = 0.f;
#pragma unroll
      for (int q = 0; q < SPG; ++q) L += sm[q] * expf(mx[q] - M);
#pragma unroll
      for (int q = 0; q < SPG; ++q) fac[tid * SPG + q] = expf(mx[q] - M) / L;
    }
    int const hq0 = tid >> 7, hq1 = hq0 + 2, dd = tid & 127;
    float o0v[SPG], o1v[SPG];
#pragma unroll
    for (int q = 0; q < SPG; ++q) {
      o0v[q] = gp[(q * QPG + hq0) * 130 + dd];
      o1v[q] = gp[(q * QPG + hq1) * 130 + dd];
    }
    __syncthreads();
    float o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int q = 0; q < SPG; ++q) {
      o0 += o0v[q] * fac[hq0 * SPG + q];
      o1 += o1v[q] * fac[hq1 * SPG + q];
    }
    e3_st2f(AOFF + (QPG * g + hq0) * E3_HD + dd, e3_rbf(o0));
    e3_st2f(AOFF + (QPG * g + hq1) * E3_HD + dd, e3_rbf(o1));
  }
}
#endif

#ifdef MPK_E3_PRO4
// MPK_E3_PRO4: stage 2's x2 rows [norm(embed(tok_r)), norm(h_r)] for r < Rr (NR >= Rr compiled rows): every row's
// raw embed (bf16 -> f32) and hidden values loaded in one batch straight into x2, the 2 Rr sums of squares in
// e3_rms_row's per-thread order, one fused block reduction in e3_block_sum's association, then e3_rms_row's
// normalize in place. Bit-identical to the row loop (2 Rr load round trips + 2 Rr block reductions before).
template <int NR>
__device__ __forceinline__ void e3_pro4(float *x2, float *red8, float const *rv, int hsrc, int Rr,
                                        long long const *tok) {
  static_assert(NR >= 1 && NR <= 4, "8 sums");
  constexpr int NB = E3_H / 8, NF = E3_H / 4;  // 360 bf16 chunks / 720 float4 per row
  int const tid = threadIdx.x;
#ifdef MPK_E3_PRO5
  {
    // row bases first (one LDS round), then every load; raw rows stored unconditionally (rows >= Rr hold row Rr - 1,
    // chunks past a row's end rewrite its last chunk with the same bytes) so no load is sunk into a branch
    uint4 const *er[NR];
    float4 const *hr[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      int const rl = min(r, Rr - 1);
      er[r] = reinterpret_cast<uint4 const *>(g_e3.embed + (size_t)tok[rl] * E3_H);
      hr[r] = reinterpret_cast<float4 const *>(rv + hsrc + rl * E3_H);
    }
    uint4 gi5[2];  // n_input, 8-value chunks c = tid + 256 u (the embed half's lane mapping)
    uint2 gh5[3];  // n_hidden, 4-value chunks c = tid + 256 u (the hidden half's lane mapping)
#pragma unroll
    for (int u = 0; u < 2; ++u) gi5[u] = reinterpret_cast<uint4 const *>(g_e3.n_input)[min(tid + 256 * u, NB - 1)];
#pragma unroll
    for (int u = 0; u < 3; ++u) gh5[u] = reinterpret_cast<uint2 const *>(g_e3.n_hidden)[min(tid + 256 * u, NF - 1)];
    uint4 em[NR][2];
    float4 hd[NR][3];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
#pragma unroll
      for (int u = 0; u < 2; ++u) em[r][u] = er[r][min(tid + 256 * u, NB - 1)];
#pragma unroll
      for (int u = 0; u < 3; ++u) hd[r][u] = hr[r][min(tid + 256 * u, NF - 1)];
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      float *const xr = x2 + r * E3_IN2;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        int const c = min(tid + 256 * u, NB - 1);
        uint4 const v = em[r][u];
        *reinterpret_cast<float4 *>(xr + c * 8) =
            make_float4(e3_bf(v.x & 0xFFFFu), e3_bf(v.x >> 16), e3_bf(v.y & 0xFFFFu), e3_bf(v.y >> 16));
        *reinterpret_cast<float4 *>(xr + c * 8 + 4) =
            make_float4(e3_bf(v.z & 0xFFFFu), e3_bf(v.z >> 16), e3_bf(v.w & 0xFFFFu), e3_bf(v.w >> 16));
      }
#pragma unroll
      for (int u = 0; u < 3; ++u) {
        int const c = min(tid + 256 * u, NF - 1);
        *reinterpret_cast<float4 *>(xr + E3_H + c * 4) = hd[r][u];
      }
    }
    __syncthreads();
    float ss[2 * NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      float a = 0.f, b = 0.f;
      if (r < Rr) {
        float const *const xr = x2 + r * E3_IN2;
        for (int i = tid; i < E3_H; i += 256) a += xr[i] * xr[i];
        for (int i = tid; i < E3_H; i += 256) b += xr[E3_H + i] * xr[E3_H + i];
      }
      ss[2 * r] = a;
      ss[2 * r + 1] = b;
    }
#pragma unroll
    for (int k = 0; k < 2 * NR; ++k) ss[k] = e3_wave_sum(ss[k]);
    if ((tid & 63) == 0)
#pragma unroll
      for (int k = 0; k < 2 * NR; ++k) red8[(tid >> 6) * 8 + k] = ss[k];
    __syncthreads();
    // normalize from the registers (the raw values every thread stored), the expression of e3_rms_row
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      if (r < Rr) {
        float const ie =
            rsqrtf((red8[2 * r] + red8[8 + 2 * r] + red8[16 + 2 * r] + red8[24 + 2 * r]) / (float)E3_H + 1e-5f);
        float const ih =
            rsqrtf((red8[2 * r + 1] + red8[9 + 2 * r] + red8[17 + 2 * r] + red8[25 + 2 * r]) / (float)E3_H + 1e-5f);
        float *const xr = x2 + r * E3_IN2;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          int const c = tid + 256 * u;
          if (c < NB) {
            unsigned const wv[4] = {em[r][u].x, em[r][u].y, em[r][u].z, em[r][u].w};
            unsigned const wg[4] = {gi5[u].x, gi5[u].y, gi5[u].z, gi5[u].w};
            float o[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              unsigned const bv = (e & 1) ? (wv[e >> 1] >> 16) : (wv[e >> 1] & 0xFFFFu);
              unsigned const bg = (e & 1) ? (wg[e >> 1] >> 16) : (wg[e >> 1] & 0xFFFFu);
              o[e] = e3_rbf(e3_bf(bv) * ie * e3_bf(bg));
            }
            *reinterpret_cast<float4 *>(xr + c * 8) = make_float4(o[0], o[1], o[2], o[3]);
            *reinterpret_cast<float4 *>(xr + c * 8 + 4) = make_float4(o[4], o[5], o[6], o[7]);
          }
        }
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          int const c = tid + 256 * u;
          if (c < NF) {
            float const hv[4] = {hd[r][u].x, hd[r][u].y, hd[r][u].z, hd[r][u].w};
            unsigned const wg[2] = {gh5[u].x, gh5[u].y};
            float o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              unsigned const bg = (e & 1) ? (wg[e >> 1] >> 16) : (wg[e >> 1] & 0xFFFFu);
              o[e] = e3_rbf(hv[e] * ih * e3_bf(bg));
            }
            *reinterpret_cast<float4 *>(xr + E3_H + c * 4) = make_float4(o[0], o[1], o[2], o[3]);
          }
        }
      }
    }
    __syncthreads();
    return;
  }
#endif
  uint4 gi[2], gh[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    int const c = min(tid + 256 * u, NB - 1);
    gi[u] = reinterpret_cast<uint4 const *>(g_e3.n_input)[c];
    gh[u] = reinterpret_cast<uint4 const *>(g_e3.n_hidden)[c];
  }
  uint4 em[NR][2];
  float4 hd[NR][3];
#pragma unroll
  for (int r = 0; r < NR; ++r) {  // branch-free: rows past Rr re-load the last live row
    int const rl = min(r, Rr - 1);
    uint4 const *const er = reinterpret_cast<uint4 const *>(g_e3.embed + (size_t)tok[rl] * E3_H);
    float4 const *const hr = reinterpret_cast<float4 const *>(rv + hsrc + rl * E3_H);
#pragma unroll
    for (int u = 0; u < 2; ++u) em[r][u] = er[min(tid + 256 * u, NB - 1)];
#pragma unroll
    for (int u = 0; u < 3; ++u) hd[r][u] = hr[min(tid + 256 * u, NF - 1)];
  }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    if (r < Rr) {
      float *const xr = x2 + r * E3_IN2;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        int const c = tid + 256 * u;
        if (c < NB) {
          uint4 const v = em[r][u];
          *reinterpret_cast<float4 *>(xr + c * 8) =
              make_float4(e3_bf(v.x & 0xFFFFu), e3_bf(v.x >> 16), e3_bf(v.y & 0xFFFFu), e3_bf(v.y >> 16));
          *reinterpret_cast<float4 *>(xr + c * 8 + 4) =
              make_float4(e3_bf(v.z & 0xFFFFu), e3_bf(v.z >> 16), e3_bf(v.w & 0xFFFFu), e3_bf(v.w >> 16));
        }
      }
#pragma unroll
      for (int u = 0; u < 3; ++u) {
        int const c = tid + 256 * u;
        if (c < NF) *reinterpret_cast<float4 *>(xr + E3_H + c * 4) = hd[r][u];
      }
    }
  }
  __syncthreads();
  float ss[2 * NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    float a = 0.f, b = 0.f;
    if (r < Rr) {
      float const *const xr = x2 + r * E3_IN2;
      for (int i = tid; i < E3_H; i += 256) a += xr[i] * xr[i];
      for (int i = tid; i < E3_H; i += 256) b += xr[E3_H + i] * xr[E3_H + i];
    }
    ss[2 * r] = a;
    ss[2 * r + 1] = b;
  }
#pragma unroll
  for (int k = 0; k < 2 * NR; ++k) ss[k] = e3_wave_sum(ss[k]);
  if ((tid & 63) == 0)
#pragma unroll
    for (int k = 0; k < 2 * NR; ++k) red8[(tid >> 6) * 8 + k] = ss[k];
  __syncthreads();
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    if (r < Rr) {
      float const ie =
          rsqrtf((red8[2 * r] + red8[8 + 2 * r] + red8[16 + 2 * r] + red8[24 + 2 * r]) / (float)E3_H + 1e-5f);
      float const ih =
          rsqrtf((red8[2 * r + 1] + red8[9 + 2 * r] + red8[17 + 2 * r] + red8[25 + 2 * r]) / (float)E3_H + 1e-5f);
      float *const xr = x2 + r * E3_IN2;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        int const c = tid + 256 * u;
        if (c < NB) {
          unsigned const wi[4] = {gi[u].x, gi[u].y, gi[u].z, gi[u].w};
          unsigned const wh[4] = {gh[u].x, gh[u].y, gh[u].z, gh[u].w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            int const i = c * 8 + e;
            unsigned const bi = (e & 1) ? (wi[e >> 1] >> 16) : (wi[e >> 1] & 0xFFFFu);
            unsigned const bh = (e & 1) ? (wh[e >> 1] >> 16) : (wh[e >> 1] & 0xFFFFu);
            xr[i] = e3_rbf(xr[i] * ie * e3_bf(bi));
            xr[E3_H + i] = e3_rbf(xr[E3_H + i] * ih * e3_bf(bh));
          }
        }
      }
    }
  }
  __syncthreads();
}
#endif

// Stage 0, run at the start of the LM-head task: target features of this pass's n slots into the
// scratch replicas. The ring copies it reads are final once the last layer has run.
#ifdef MPK_E3_STATE1
__device__ MPK_E3_CALL void mpk_eagle3_features(int wid, E3St const e3st) {
#else
__device__ MPK_E3_CALL void mpk_eagle3_features(int wid) {
#endif
  if (!g_e3.enabled || E3_SPEC(0) != 1) {
    return;
  }
  int const tid = threadIdx.x;
  int const W = g_e3.nworkers;
  int const n = E3_SPEC(1);
#ifdef MPK_EAGLE3_TS
  if (tid == 0 && E3_SPEC(2) ==
                      *g_e3.prompt_len + 20) {
    e3_st_sys_u32(g_e3.vec + E3_V_SPAN + 3 * wid, (unsigned)__builtin_amdgcn_s_memrealtime());
  }
#endif
  // ---- stage 0: target features of the candidate rows (all n slots; rows beyond R are ignored later)
  {
    int const total = n * E3_FEAT;
    int const per = (total + W - 1) / W;
    int const e0 = wid * per, e1 = min(total, e0 + per);
    for (int e = e0 + tid; e < e1; e += blockDim.x) {
      int const r = e / E3_FEAT, i = e - r * E3_FEAT;
      int const L = i / E3_H, c = i - L * E3_H;
      int const ai = r * E3_HP + c;
      unsigned const wa = e3_ld_sys_u32(g_e3.apo[L] + (ai & ~1));
      float v = e3_bf((ai & 1) ? (wa >> 16) : (wa & 0xFFFFu));
      float const *wsr = g_e3.ws[L] + (size_t)r * g_e3.ws_stride + c;
#pragma unroll
      for (int k = 0; k < 4; ++k) v += e3_ld_sys_f32(wsr + k * E3_HP);
#ifdef MPK_E3_FEATH
      e3_st2h(E3_V_AUX, e, e3_tobf(v));
#else
      e3_st2f(E3_V_AUX + e, e3_rbf(v));
#endif
    }
  }
}

// wid: this worker's index 0..nworkers-1; die: its logical XCD (wid / workers_per_xcd).
#ifdef MPK_E3_STATE1
__device__ MPK_E3_CALL void mpk_eagle3_phase(int wid, int die, E3St const e3st) {
#else
__device__ MPK_E3_CALL void mpk_eagle3_phase(int wid, int die) {
#endif
  if (!g_e3.enabled || E3_SPEC(0) != 1) {
    return;
  }
  extern __shared__ char _rnlm_smem[];
  float *const lds = reinterpret_cast<float *>(_rnlm_smem);
  __shared__ float e3_red[8];
  __shared__ long long e3_tok[E3_R];
  __shared__ int e3_pos[E3_R];
  int const tid = threadIdx.x;
  int const W = g_e3.nworkers;
  float const *const rv = e3_rep_local();
  int const n = E3_SPEC(1);
  int const s = E3_SPEC(2);
  int const kind = E3_SPEC(3);
  int const prompt_len = *g_e3.prompt_len;
  constexpr int E3_GS = E3_K <= 4 ? 32 : 64;  // barrier generations per pass
  int const gen0 = (s + 1) * E3_GS;  // a pass uses gen0 + 1 .. gen0 + 2 + 6 * E3_K
  static_assert(2 + 6 * E3_K < E3_GS, "barrier generations per pass");
#ifdef MPK_E3_AID_ROWS_CHECK
  if (tid == 0 && (s == prompt_len + 20 || s == prompt_len + 21)) {
    unsigned xcc_c;
    MPK_GET_XCC_ID32(xcc_c);
    printf("[E3XCC] wid %d die %d xcc %u%s\n", wid, die, xcc_c & 7u, (int)(xcc_c & 7u) == die ? "" : " MISMATCH");
  }
#endif
#ifdef MPK_EAGLE3_TS
  if (tid == 0 && s == prompt_len + 20) {
    e3_st_sys_u32(g_e3.vec + E3_V_SPAN + 3 * wid + 1, (unsigned)__builtin_amdgcn_s_memrealtime());
  }
#endif
#ifdef MPK_EAGLE3_TS
#ifdef MPK_E3TS_EVERY
  constexpr int E3TS_N = 64;
#else
  constexpr int E3TS_N = 56;
#endif
  unsigned long long e3ts[E3TS_N];
  int e3n = 0;
#ifndef MPK_E3TS_W
#define MPK_E3TS_W 0
#endif
#define E3TS() if (wid == MPK_E3TS_W && tid == 0 && e3n < E3TS_N) e3ts[e3n++] = __builtin_amdgcn_s_memrealtime()
#if defined(MPK_E3TS_EVERY)
#define E3TS_AT_POS (kind == 1 && s > prompt_len && (s % (MPK_E3TS_EVERY)) < 4)
#elif defined(MPK_E3TS_AT)
#define E3TS_AT_POS (s >= (MPK_E3TS_AT) && s < (MPK_E3TS_AT) + 3)
#else
#define E3TS_AT_POS (s == 400 || s == prompt_len + 20 || s == prompt_len + 21)
#endif
#ifdef MPK_LMT_TS
#define E3TS_LMT() printf("[LMTTS] s=%d norm %llu quant %llu g0 %llu tiles %llu epi %llu nsum %llu nred %llu nst %llu\n", s, \
    g_lmt_ts[0] - g_e3lm_ts[1], g_lmt_ts[1] - g_lmt_ts[0], g_lmt_ts[2] - g_lmt_ts[1],            \
    g_lmt_ts[3] - g_lmt_ts[2], g_lmt_ts[4] - g_lmt_ts[3], g_lmt_ts[5] - g_e3lm_ts[1],            \
    g_lmt_ts[6] - g_lmt_ts[5], g_lmt_ts[7] - g_lmt_ts[6])
#else
#define E3TS_LMT()
#endif
#define E3TS_DUMP(tag)                                                                                  \
  if (wid == MPK_E3TS_W && tid == 0 && E3TS_AT_POS) {                                                              \
    printf("[E3TS] %s s=%d kind=%d n=%d R=%d lm(10ns): feat %llu lmhead %llu gap %llu | dt(10ns):", tag, s, kind, n, R, \
           g_e3lm_ts[1] - g_e3lm_ts[0], g_e3lm_ts[2] - g_e3lm_ts[1], e3ts[0] - g_e3lm_ts[2]);      \
    for (int q = 1; q < e3n; ++q) printf(" %llu", e3ts[q] - e3ts[q - 1]);                                 \
    printf(" total %llu\n", e3ts[e3n - 1] - e3ts[0]);                                                     \
    E3TS_LMT();                                                                                   \
  }
#else
#define E3TS()
#define E3TS_DUMP(tag)
#endif
  E3TS();
  // Rows whose features are final this pass and the next token each one sees.
  int R = 0;
  bool full = false;
  if (kind == 0) {
    R = n;
    full = s + n >= prompt_len;
  } else {
    full = true;
  }

  E3TS();
  e3_barrier(gen0 + 1, die);
  E3TS();

  // ---- acceptance (verify pass) and the next tokens, from the target argmax partials the LM head copied
  __shared__ long long e3_y[E3_R];
#ifdef MPK_E3_ACC1
  long long pin[E3_R];  // thread 0: the drafts this verify pass checked (plain loads, as MPK_E3_STATE_PLAIN)
  if (tid == 0) {
    long long const *const pvi = (long long const *)(g_spec_prev_in);
#pragma unroll
    for (int j = 1; j < E3_R; ++j) pin[j] = pvi[j];
  }
  {  // per row j: argmax over the W target partials (ties to the lower worker), the winner's index as payload
    float vv[E3_R];
    int ky[E3_R];
    unsigned ix[E3_R];
#pragma unroll
    for (int j = 0; j < E3_R; ++j) {
      bool const ok = j < n && tid < W;
      vv[j] = ok ? e3_bf(__float_as_uint(rv[E3_V_YV + j * 256 + tid])) : -3.0e38f;
      ix[j] = ok ? __float_as_uint(rv[E3_V_YI + j * 256 + tid]) : 0u;
#if defined(MPK_TAIL_NO111) || defined(MPK_TAIL_NO111_CHECK)
      ky[j] = tid < W ? e3_k111(tid) : 0x7fffffff;
#else
      ky[j] = tid < W ? tid : 0x7fffffff;
#endif
    }
#pragma unroll
    for (int o = 32; o >= 1; o >>= 1) {
#pragma unroll
      for (int j = 0; j < E3_R; ++j) {
        float const ov = __shfl_xor(vv[j], o, 64);
        int const ok = __shfl_xor(ky[j], o, 64);
        unsigned const oi = (unsigned)__shfl_xor((int)ix[j], o, 64);
        if (ov > vv[j] || (ov == vv[j] && ok < ky[j])) {
          vv[j] = ov;
          ky[j] = ok;
          ix[j] = oi;
        }
      }
    }
    float *const ar = lds;  // [4 waves][E3_R][3]
    if ((tid & 63) == 0)
#pragma unroll
      for (int j = 0; j < E3_R; ++j) {
        ar[((tid >> 6) * E3_R + j) * 3] = vv[j];
        ar[((tid >> 6) * E3_R + j) * 3 + 1] = __int_as_float(ky[j]);
        ar[((tid >> 6) * E3_R + j) * 3 + 2] = __uint_as_float(ix[j]);
      }
    __syncthreads();
    if (tid < n) {
      int const j = tid;
      float bv = ar[j * 3];
      int bk = __float_as_int(ar[j * 3 + 1]);
      unsigned bi = __float_as_uint(ar[j * 3 + 2]);
      for (int w4 = 1; w4 < 4; ++w4) {
        float const ov = ar[(w4 * E3_R + j) * 3];
        int const ok = __float_as_int(ar[(w4 * E3_R + j) * 3 + 1]);
        if (ov > bv || (ov == bv && ok < bk)) {
          bv = ov;
          bk = ok;
          bi = __float_as_uint(ar[(w4 * E3_R + j) * 3 + 2]);
        }
      }
      e3_y[j] = (long long)bi;
#if defined(MPK_TAIL_NO111) || defined(MPK_TAIL_NO111_CHECK)
      if (wid == 0) {
        __hip_atomic_store(&g_e3.draft_out[E3_TGT_SLOT + j], (long long)bi, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
#endif
    }
  }
#else
  for (int j = 0; j < n; ++j) {
    float const v = tid < W ? e3_bf(__float_as_uint(rv[E3_V_YV + j * 256 + tid])) : -3.0e38f;
    int const wk = e3_block_argmax(v, tid < W ? tid : 0x7fffffff, e3_red);
    if (tid == 0) e3_y[j] = (long long)__float_as_uint(rv[E3_V_YI + j * 256 + wk]);
  }
#endif
#ifdef MPK_TAIL_NO111_CHECK
  for (int j = 0; j < n; ++j) {  // the lower-worker tie rule: the check's control
    float const v = tid < W ? e3_bf(__float_as_uint(rv[E3_V_YV + j * 256 + tid])) : -3.0e38f;
    int const wk = e3_block_argmax(v, tid < W ? tid : 0x7fffffff, e3_red);
    if (tid == 0 && wid == 0) {
      __hip_atomic_store(&g_e3.draft_out[E3_TGT_SLOT + 8 + j], (long long)__float_as_uint(rv[E3_V_YI + j * 256 + wk]),
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
#endif
  __syncthreads();
  if (tid == 0) {
    long long y[E3_R];
    for (int j = 0; j < n; ++j) y[j] = e3_y[j];
    int rr = n;
    if (kind == 1) {
      int a = 0;
#if defined(MPK_E3_ACC1)
#pragma unroll
      for (int j = 0; j < E3_R - 1; ++j)
        if (a == j && a < n - 1 && pin[j + 1] == e3_y[j]) ++a;
#elif defined(MPK_E3_STATE_PLAIN)
      while (a < n - 1 && g_spec_prev_in[a + 1] == y[a]) ++a;
#else
      while (a < n - 1 && e3_ld_sys_i64((long long const *)&g_spec_prev_in[a + 1]) == y[a]) ++a;
#endif
      rr = a + 1;
    }
    for (int j = 0; j < rr; ++j) {
      int const p = s + j;
      e3_pos[j] = p;
      long long t;
      if (kind == 0 && p + 1 < prompt_len) {
        t = e3_ld_sys_i64(g_e3.tokens + p + 1);
      } else {
        t = y[j];
      }
      e3_tok[j] = t < 0 ? 0 : t;
    }
    e3_red[7] = __int_as_float(rr);
  }
  __syncthreads();
  R = __float_as_int(e3_red[7]);
  if (R < 1 || e3_pos[R - 1] >= g_e3.maxpos) return;  // same decision on every worker

  // ---- stage 1: h = fc(features), all R rows
  {
    float *const x = lds;  // MPK_E3_Q8M: XR rows of E3_FEAT per round; otherwise E3_R rows
#ifdef MPK_E3_Q8M
    {
      constexpr int KP = 8704, KT = KP / 128, KQ = KT / 4;  // 68 K tiles, 17 per wave
      constexpr int XR = E3_R < 2 ? E3_R : 2;  // f32 feature rows staged per round (4 rows exceed LDS)
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + XR * E3_FEAT + 128);
      unsigned char *const qs = q8 + E3_R * KP;
      static_assert(E3_R * KT <= 1024, "fc scale rows");
      float *const s_red = reinterpret_cast<float *>(qs + (E3_R * KT <= 512 ? 512 : 1024));  // [4 waves][16][E3_R]
#ifdef MPK_E3_IDLE_SKIP
      int const dt = e3_tile180(wid, W);
      if (dt >= 0)  // workers without a row tile skip the load + quantize
#endif
#ifdef MPK_E3_FEATH
      e3_q8g_bf<E3_FEAT, KP, E3_R <= 4 ? 17 : 9>(reinterpret_cast<unsigned short const *>(rv + E3_V_AUX), E3_FEAT, R,
                                                q8, qs);
#else
      for (int r0 = 0; r0 < R; r0 += XR) {
        int const nr = min(XR, R - r0);
        e3_load_rep(x, rv, E3_V_AUX + r0 * E3_FEAT, nr * E3_FEAT);
        e3_q8act_r<E3_FEAT, KP>(x, E3_FEAT, nr, q8 + r0 * KP, qs + r0 * KT);
      }
#endif
      E3TS();  // fc-load
      E3TS();  // fc: quantized (diagnostic stamp)
#ifndef MPK_E3_IDLE_SKIP
      int const dt = e3_tile180(wid, W);
#endif
      int const wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
      if (dt >= 0) {
        unsigned char const *const w0 = (unsigned char const *)g_e3.fc + ((size_t)dt * KT + wave * KQ) * 2048;
        unsigned char const *wr[2] = {w0, w0};
        e3_f32x4 acc[2];
        e3_m8gemv<KQ, 1, 4>(wr, q8 + wave * KQ * 128, KP, qs + wave * KQ, KT, R, acc);
        if (col < E3_R)
#pragma unroll
          for (int i = 0; i < 4; ++i) s_red[((wave * 16) + gq * 4 + i) * E3_R + col] = acc[0][i];
      }
      __syncthreads();
      if (dt >= 0 && tid < 16 * E3_R) {
        int const row = tid / E3_R, r = tid - row * E3_R, o = dt * 16 + row;
        if (r < R) {
          float const v = (s_red[(row) * E3_R + r] + s_red[(16 + row) * E3_R + r]) +
                          (s_red[(32 + row) * E3_R + r] + s_red[(48 + row) * E3_R + r]);
          e3_st2f(E3_V_H + r * E3_H + o, e3_rbf(v * g_e3.sfc[o]));
        }
      }
    }
#else
    e3_load_rep(x, rv, E3_V_AUX, R * E3_FEAT);
    E3TS();
    for (int i = R * E3_FEAT + tid; i < E3_R * E3_FEAT; i += blockDim.x) x[i] = 0.f;
    __syncthreads();
    E3TS(); // fc-load
    int o0, o1;
    e3_split(E3_H, wid, W, o0, o1);
    e3_gemv4<E3_FEAT, E3_R>(g_e3.fc, g_e3.sfc, x, E3_FEAT, o0, o1, [&](int o, auto const &acc) {
      for (int r = 0; r < R; ++r) e3_st2f(E3_V_H + r * E3_H + o, e3_rbf(acc[r]));
    });
#endif
  }
  E3TS();
  e3_barrier(gen0 + 2, die);
  E3TS();

  int gen = gen0 + 2;  // running barrier generation
  int const P0 = e3_pos[R - 1];
#ifdef MPK_E3_CONF
  float e3_pr = 1.f;  // running product of the drafts' probabilities (the same on every worker)
#endif
#if MPK_E3_CHAIN > 1
  for (int jc = 0; jc < E3_K; ++jc) {
#else
  // No loop for one draft: a one-trip loop around this body measured 256 AGPR + 1.8 KB scratch (vs 32 AGPR,
  // 0.37 KB) and +70 us per pass.
  {
  constexpr int jc = 0;
#endif
  // Step jc: Rr rows; their hidden inputs at HSRC (fc outputs for step 0, step jc-1's out after), and this
  // step's copies of the single-row regions.
  int const Rr = jc == 0 ? R : 1;
  int const HSRC = jc == 0 ? E3_V_H : e3_reg(E3_V_O, jc - 1, E3_CO);
  int const QOFF = e3_reg(E3_V_Q, jc, E3_CQ), APOFF = e3_reg(E3_V_AP, jc, E3_CAP), ROFF = e3_reg(E3_V_R, jc, E3_CR);
  int const FOFF = e3_reg(E3_V_F, jc, E3_CF), OOFF = e3_reg(E3_V_O, jc, E3_CO);
  int const PVOFF = e3_reg(E3_V_PV, jc, E3_CPV), PIOFF = e3_reg(E3_V_PI, jc, E3_CPI);
#ifdef MPK_E3_ATTN_GQA
  int const AOFF = e3_reg(E3_V_A, jc, E3_CA), GOFF = E3_V_GQA0 + jc * E3_GQA_SZ;
#endif
#if MPK_E3_CHAIN > 1
  if (jc > 0) {
    // draft jc = argmax of step jc-1's per-worker LM partials (every worker reduces them: no publish hop)
    int const pv0 = e3_reg(E3_V_PV, jc - 1, E3_CPV), pi0 = e3_reg(E3_V_PI, jc - 1, E3_CPI);
    int const pi = tid < W ? (int)__float_as_uint(rv[pi0 + tid]) : -1;
    float const pvv = (tid < W && pi >= 0) ? rv[pv0 + tid] : -3.0e38f;
#ifdef MPK_E3_CONF
    float const psv = (tid < W && pi >= 0) ? rv[e3_reg(E3_V_PS, jc - 1, E3_CPS) + tid] : 0.f;
    float Mx;
    int const fi = e3_block_argmax_v(pvv, pi >= 0 ? pi : 0x7fffffff, e3_red, Mx);
    float const pj = e3_conf_p(pvv, psv, Mx, e3_red);
    e3_pr *= pj;
#ifdef MPK_E3_CONF_D
    bool const stop = g_spec_conf > 0.f && jc + 1 > g_spec_kmin && e3_pr < g_spec_confd;
#else
    bool const stop = g_spec_conf > 0.f && jc + 1 > g_spec_kmin && e3_pr < g_spec_conf;
#endif
#else
    int const fi = e3_block_argmax(pvv, pi >= 0 ? pi : 0x7fffffff, e3_red);
#endif
    if (tid == 0) {
      long long const d = (long long)fi + g_e3.d2t[fi];
      e3_tok[0] = d;
      e3_pos[0] = P0 + jc;
      if (wid == 0) __hip_atomic_store(&g_e3.draft_out[jc - 1], d, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#ifdef MPK_E3_CONF
      if (wid == 0) {
        __hip_atomic_store(&g_e3.draft_out[E3_CONF_SLOT + jc - 1], (long long)__float_as_uint(pj), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
        if (stop)  // drafts jc + 1 .. E3_K are not made: probability 0 cuts the verify prefix before them
          for (int q = jc; q < E3_K; ++q)
            __hip_atomic_store(&g_e3.draft_out[E3_CONF_SLOT + q], 0ll, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
#endif
    }
    __syncthreads();
#ifdef MPK_E3_CONF
    if (stop) break;  // same decision on every worker
#endif
    if (P0 + jc >= g_e3.maxpos) break;  // same decision on every worker
  }
#endif

  // ---- stage 2: x2 = [norm(embed(tok)), norm(h)]; q / k / v with RoPE; k, v into the draft cache
  {
    constexpr int E3_X2R = E3_R <= 4 ? E3_R : 4;  // f32 x2 rows in LDS (past block 4: groups of 4 rows)
    float *const x2 = lds;                    // E3_X2R rows of E3_IN2
    float *const tmp = lds + E3_X2R * E3_IN2;  // E3_H (past block 4: 64 floats, the PRO4 reduction)
#ifdef MPK_E3_PRO3
    {
      static_assert(E3_R <= 2, "MPK_E3_PRO3: 4-value reduction");
      constexpr int NB = E3_H / 8, NF = E3_H / 4;  // 16-byte chunks: bf16 rows 360, f32 rows 720
      float *const red4 = tmp;                     // [4 waves][4]
      uint4 em[E3_R][2], gi[2];
      float4 hd[E3_R][3];
      uint2 gh[3];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        int const c = tid + 256 * u;
        gi[u] = c < NB ? reinterpret_cast<uint4 const *>(g_e3.n_input)[c] : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int u = 0; u < 3; ++u) {
        int const c = tid + 256 * u;
        gh[u] = c < NF ? reinterpret_cast<uint2 const *>(g_e3.n_hidden)[c] : make_uint2(0u, 0u);
      }
#pragma unroll
      for (int r = 0; r < E3_R; ++r) {
        uint4 const *const er = reinterpret_cast<uint4 const *>(g_e3.embed + (size_t)e3_tok[r] * E3_H);
        float4 const *const hr = reinterpret_cast<float4 const *>(rv + HSRC + r * E3_H);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          int const c = tid + 256 * u;
          em[r][u] = (r < Rr && c < NB) ? er[c] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          int const c = tid + 256 * u;
          hd[r][u] = (r < Rr && c < NF) ? hr[c] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
      }
      float ss[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < E3_R; ++r) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          unsigned const w[4] = {em[r][u].x, em[r][u].y, em[r][u].z, em[r][u].w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float const f = e3_bf((e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xFFFFu));
            ss[2 * r] += f * f;
          }
        }
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          float4 const h = hd[r][u];
          ss[2 * r + 1] += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
        }
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) ss[k] = e3_wave_sum(ss[k]);  // lane 0
      if ((tid & 63) == 0)
#pragma unroll
        for (int k = 0; k < 4; ++k) red4[(tid >> 6) * 4 + k] = ss[k];
      __syncthreads();
      float inv[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        inv[k] = rsqrtf((red4[k] + red4[4 + k] + red4[8 + k] + red4[12 + k]) / (float)E3_H + 1e-5f);
#pragma unroll
      for (int r = 0; r < E3_R; ++r) {
        float *const xr = x2 + r * E3_IN2;
        bool const live = r < Rr;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          int const c = tid + 256 * u;
          if (c < NB) {
            unsigned const w[4] = {em[r][u].x, em[r][u].y, em[r][u].z, em[r][u].w};
            unsigned const g[4] = {gi[u].x, gi[u].y, gi[u].z, gi[u].w};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              unsigned const fb = (e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xFFFFu);
              unsigned const gb = (e & 1) ? (g[e >> 1] >> 16) : (g[e >> 1] & 0xFFFFu);
              xr[c * 8 + e] = live ? e3_rbf(e3_bf(fb) * inv[2 * r] * e3_bf(gb)) : 0.f;
            }
          }
        }
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          int const c = tid + 256 * u;
          if (c < NF) {
            float const hv[4] = {hd[r][u].x, hd[r][u].y, hd[r][u].z, hd[r][u].w};
            unsigned const g[2] = {gh[u].x, gh[u].y};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              unsigned const gb = (e & 1) ? (g[e >> 1] >> 16) : (g[e >> 1] & 0xFFFFu);
              xr[E3_H + c * 4 + e] = live ? e3_rbf(hv[e] * inv[2 * r + 1] * e3_bf(gb)) : 0.f;
            }
          }
        }
      }
      __syncthreads();
    }
#elif defined(MPK_E3_PRO4)
#if MPK_MAX_NUM_BATCHED_TOKENS <= 4
    if (Rr == 1) e3_pro4<1>(x2, tmp, rv, HSRC, 1, e3_tok);
    else e3_pro4<E3_R>(x2, tmp, rv, HSRC, Rr, e3_tok);
#endif  // past block 4: in row groups with the quantize, below
#else
    for (int r = 0; r < E3_R; ++r) {
      if (r < Rr) {
        e3_load_bf_row(tmp, g_e3.embed + (size_t)e3_tok[r] * E3_H);
        e3_rms_row(tmp, x2 + r * E3_IN2, g_e3.n_input, e3_red);
        e3_load_rep(tmp, rv, HSRC + r * E3_H, E3_H);
        e3_rms_row(tmp, x2 + r * E3_IN2 + E3_H, g_e3.n_hidden, e3_red);
      } else {
        for (int i = tid; i < E3_IN2; i += blockDim.x) x2[r * E3_IN2 + i] = 0.f;
        __syncthreads();
      }
    }
#endif
    E3TS();
#ifdef MPK_E3_Q8M
    {
      // 384 tiles (48 heads x 8), 1-2 per worker; K (45 tiles) split 12 / 11 / 11 / 11 over the 4 waves.
      constexpr int KT = E3_IN2 / 128;
      constexpr int X2E = E3_X2R * E3_IN2 + (E3_R <= 4 ? E3_H : 64);  // x2 + tmp
      constexpr int Q8O = X2E > 16384 ? ((X2E + 63) & ~63) : 16384;
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + Q8O);
      unsigned char *const qs = q8 + E3_R * E3_IN2;
      static_assert(E3_R * KT <= 512, "q/k/v scale rows");
      float *const s_red = reinterpret_cast<float *>(qs + (E3_R * KT <= 256 ? 256 : 512));  // [4 waves][2][16][E3_R]
#if MPK_MAX_NUM_BATCHED_TOKENS > 4
      for (int r0 = 0; r0 < Rr; r0 += 4) {  // MPK_E3_PRO4 rows of a group into x2, then their e4m3 rows
        int const nr = min(4, Rr - r0);
        if (nr == 1) e3_pro4<1>(x2, tmp, rv, HSRC + r0 * E3_H, 1, e3_tok + r0);
        else e3_pro4<4>(x2, tmp, rv, HSRC + r0 * E3_H, nr, e3_tok + r0);
        e3_q8act_r<E3_IN2, E3_IN2>(x2, E3_IN2, nr, q8 + r0 * E3_IN2, qs + r0 * KT);
      }
#else
      e3_q8act_r<E3_IN2, E3_IN2>(x2, E3_IN2, Rr, q8, qs);
#endif
      int t0, t1;
#ifdef MPK_E3_QKV_AID
      // each device's workers take the tiles in its own AID half (MPK_E3_OWN_BO midpoint cut): device d q tiles
      // [128 d, 128 d + 128), k and v tiles [32 d, 32 d + 32) of the concatenated (q 256, k 64, v 64) tiles
      int const qd = wid >= (W >> 1) ? 1 : 0;
      e3_bsplit((E3_NQ + 2 * E3_NKV) * 4, wid - qd * (W >> 1), qd ? W - (W >> 1) : (W >> 1), t0, t1);
      auto tmap = [&](int u) -> int {
        return u < E3_NQ * 4 ? qd * E3_NQ * 4 + u
                             : (u < (E3_NQ + E3_NKV) * 4
                                    ? E3_NQ * 8 + qd * E3_NKV * 4 + (u - E3_NQ * 4)
                                    : (E3_NQ + E3_NKV) * 8 + qd * E3_NKV * 4 + (u - (E3_NQ + E3_NKV) * 4));
      };
#else
      e3_bsplit((E3_NQ + 2 * E3_NKV) * 8, wid, W, t0, t1);
      auto tmap = [](int u) -> int { return u; };
#endif
      int const nt = t1 - t0;
      int const Ta = tmap(t0), Tb = tmap(t0 + (nt > 1 ? 1 : 0));
      int const wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
      int const kt0 = wave == 0 ? 0 : 12 + 11 * (wave - 1);
      auto tbase = [&](int T) -> unsigned char const * {
        int const head = T >> 3;
        unsigned short const *const m = head < E3_NQ ? g_e3.wq : head < E3_NQ + E3_NKV ? g_e3.wk : g_e3.wv;
        int const tl = head < E3_NQ ? T : head < E3_NQ + E3_NKV ? T - E3_NQ * 8 : T - (E3_NQ + E3_NKV) * 8;
        return (unsigned char const *)m + ((size_t)tl * KT + kt0) * 2048;
      };
      if (nt > 0) {
        unsigned char const *wr[2] = {tbase(Ta), tbase(Tb)};
        unsigned char const *const qw = q8 + kt0 * 128;
        unsigned char const *const qsw = qs + kt0;
        e3_f32x4 acc[2];
        if (wave == 0) {
          if (nt > 1) e3_m8gemv<12, 2, 4>(wr, qw, E3_IN2, qsw, KT, Rr, acc);
          else e3_m8gemv<12, 1, 4>(wr, qw, E3_IN2, qsw, KT, Rr, acc);
        } else {
          if (nt > 1) e3_m8gemv<11, 2, 4>(wr, qw, E3_IN2, qsw, KT, Rr, acc);
          else e3_m8gemv<11, 1, 4>(wr, qw, E3_IN2, qsw, KT, Rr, acc);
        }
        if (col < E3_R)
#pragma unroll
          for (int t = 0; t < 2; ++t)
            if (t < nt)
#pragma unroll
              for (int i = 0; i < 4; ++i) s_red[(((wave * 2 + t) * 16) + gq * 4 + i) * E3_R + col] = acc[t][i];
      }
      __syncthreads();
      float const lth = logf(150000.0f);
      // one thread per (tile, dim pair p, row r): dims d0 = 8j + 2p, d0 + 1, d0 + 64, d0 + 65 (tile rows
      // 2p, 2p + 1, 2p + 8, 2p + 9), the unit the dot path computes.
      if (tid < nt * 4 * E3_R) {
        int const t = tid / (4 * E3_R), p = (tid / E3_R) & 3, r = tid % E3_R;
        if (r < Rr) {
          int const T = t ? Tb : Ta, head = T >> 3, j = T & 7;
          int hl;
          float const *sc;
          if (head < E3_NQ) {
            sc = g_e3.swq + (size_t)T * 16;
            hl = head;
          } else if (head < E3_NQ + E3_NKV) {
            sc = g_e3.swk + (size_t)(T - E3_NQ * 8) * 16;
            hl = head - E3_NQ;
          } else {
            sc = g_e3.swv + (size_t)(T - (E3_NQ + E3_NKV) * 8) * 16;
            hl = head - E3_NQ - E3_NKV;
          }
          auto red = [&](int row) {
            float const *const b = s_red + ((t * 16) + row) * E3_R + r;
            return e3_rbf(((b[0] + b[2 * 16 * E3_R]) + (b[4 * 16 * E3_R] + b[6 * 16 * E3_R])) * sc[row]);
          };
          float v0 = red(2 * p), v1 = red(2 * p + 1), v2 = red(2 * p + 8), v3 = red(2 * p + 9);
          int const d0 = 8 * j + 2 * p;
          if (head < E3_NQ + E3_NKV) {  // RoPE: dim i pairs with i + 64
            float const rf0 = expf(-(float)(2 * d0) / (float)E3_HD * lth);
            float const rf1 = expf(-(float)(2 * (d0 + 1)) / (float)E3_HD * lth);
            float const pp = (float)e3_pos[r];
            float s0, c0, s1, c1;
            sincosf(pp * rf0, &s0, &c0);
            sincosf(pp * rf1, &s1, &c1);
            float const a0 = v0 * c0 - v2 * s0, a2 = v2 * c0 + v0 * s0;
            float const a1 = v1 * c1 - v3 * s1, a3 = v3 * c1 + v1 * s1;
            v0 = e3_rbf(a0); v2 = e3_rbf(a2); v1 = e3_rbf(a1); v3 = e3_rbf(a3);
          }
          if (head < E3_NQ) {
            if (r == Rr - 1) {
              int const q = QOFF + head * E3_HD;
              e3_st2f(q + d0, v0);
              e3_st2f(q + d0 + 1, v1);
              e3_st2f(q + d0 + 64, v2);
              e3_st2f(q + d0 + 65, v3);
            }
          } else {
            unsigned short *const cache = head < E3_NQ + E3_NKV ? g_e3.kc : g_e3.vc;
            unsigned short *const crow = cache + ((size_t)e3_pos[r] * E3_NKV + hl) * E3_HD;
            e3_st_sys_u32(crow + d0, e3_tobf(v0) | (e3_tobf(v1) << 16));
            e3_st_sys_u32(crow + d0 + 64, e3_tobf(v2) | (e3_tobf(v3) << 16));
          }
        }
      }
    }
#else
    // Unit u = head u/32, dims {2m, 2m+1, 2m+64, 2m+65} with m = u%32; heads 0-31 q, 32-39 k, 40-47 v.
    constexpr int NU = (E3_NQ + 2 * E3_NKV) * 32;
    int u0, u1;
    e3_split(NU, wid, W, u0, u1);
    float const lth = logf(150000.0f);
    for (int u = u0 + (tid >> 6); u < u1; u += 4) {
#ifdef MPK_E3_AID_ROWS
      // u = aid * NU/2 + local unit: q heads aid*16.., k heads aid*4.., v heads aid*4.. (first or second
      // half of wq / wk / wv).
      int const ua = u / (NU / 2), ul = u - ua * (NU / 2), hh = ul >> 5;
      int const head = hh < E3_NQ / 2 ? ua * (E3_NQ / 2) + hh
                       : hh < E3_NQ / 2 + E3_NKV / 2 ? E3_NQ + ua * (E3_NKV / 2) + (hh - E3_NQ / 2)
                                                      : E3_NQ + E3_NKV + ua * (E3_NKV / 2) +
                                                            (hh - E3_NQ / 2 - E3_NKV / 2);
      int const m = ul & 31;
#else
      int const head = u >> 5, m = u & 31;
#endif
      unsigned short const *base;
      float const *bsc;
      int hl;
      if (head < E3_NQ) {
        base = g_e3.wq;
        bsc = g_e3.swq;
        hl = head;
      } else if (head < E3_NQ + E3_NKV) {
        base = g_e3.wk;
        bsc = g_e3.swk;
        hl = head - E3_NQ;
      } else {
        base = g_e3.wv;
        bsc = g_e3.swv;
        hl = head - E3_NQ - E3_NKV;
      }
      int const d0 = 2 * m;
      E3Row const wr[4] = {e3_row(base, bsc, (size_t)(hl * E3_HD + d0), E3_IN2),
                           e3_row(base, bsc, (size_t)(hl * E3_HD + d0 + 1), E3_IN2),
                           e3_row(base, bsc, (size_t)(hl * E3_HD + d0 + 64), E3_IN2),
                           e3_row(base, bsc, (size_t)(hl * E3_HD + d0 + 65), E3_IN2)};
      float acc[4][E3_R];
      e3_dotsr<E3_IN2, 4, E3_R>(wr, x2, E3_IN2, acc);
      float const rf0 = expf(-(float)(2 * d0) / (float)E3_HD * lth);
      float const rf1 = expf(-(float)(2 * (d0 + 1)) / (float)E3_HD * lth);
      if ((tid & 63) == 0) {
        for (int r = 0; r < Rr; ++r) {
          float v0 = e3_rbf(acc[0][r]), v1 = e3_rbf(acc[1][r]), v2 = e3_rbf(acc[2][r]), v3 = e3_rbf(acc[3][r]);
          if (head < E3_NQ + E3_NKV) {  // RoPE: dim i pairs with i + 64
            float const p = (float)e3_pos[r];
            float s0, c0, s1, c1;
            sincosf(p * rf0, &s0, &c0);
            sincosf(p * rf1, &s1, &c1);
            float const a0 = v0 * c0 - v2 * s0, a2 = v2 * c0 + v0 * s0;
            float const a1 = v1 * c1 - v3 * s1, a3 = v3 * c1 + v1 * s1;
            v0 = e3_rbf(a0); v2 = e3_rbf(a2); v1 = e3_rbf(a1); v3 = e3_rbf(a3);
          }
          if (head < E3_NQ) {
            if (r == Rr - 1) {
              int const q = QOFF + head * E3_HD;
              e3_st2f(q + d0, v0);
              e3_st2f(q + d0 + 1, v1);
              e3_st2f(q + d0 + 64, v2);
              e3_st2f(q + d0 + 65, v3);
            }
          } else {
            unsigned short *const cache = head < E3_NQ + E3_NKV ? g_e3.kc : g_e3.vc;
            unsigned short *const row = cache + ((size_t)e3_pos[r] * E3_NKV + hl) * E3_HD;
            e3_st_sys_u32(row + d0, e3_tobf(v0) | (e3_tobf(v1) << 16));
            e3_st_sys_u32(row + d0 + 64, e3_tobf(v2) | (e3_tobf(v3) << 16));
          }
        }
      }
    }
#endif
  }
  E3TS();
  if (!full) {
    E3TS_DUMP("pre");
    return;
  }
  e3_barrier(++gen, die);
  E3TS();

  // ---- stage 3: attention of the last row over draft positions 0..P (one q head per worker)
  int const P = e3_pos[Rr - 1];
#if defined(MPK_E3_ATTN_GQA)
  constexpr int E3_GX = 0;
  if (wid < E3_NKV * MPK_E3_ATTN_GQA) e3_attn_gqa(P, QOFF, GOFF, AOFF, wid, rv);
#elif defined(MPK_E3_ATTN_SPLIT2)
  constexpr int E3_GX = 0;
  {
    constexpr int SP = MPK_E3_ATTN_SPLIT2;  // workers per q head
    static_assert(E3_NQ * SP <= 248, "one worker per (head, split)");
    static_assert(E3_V_AP + E3_NQ * SP * 130 <= E3_V_FLOATS, "partial region");
    if (wid < E3_NQ * SP) {
      int const h = wid / SP, sp = wid % SP, g = h / (E3_NQ / E3_NKV);
      int const len = (P + 1 + SP - 1) / SP;
      int const t0 = sp * len, t1 = min(P + 1, t0 + len);
      float *const qv = lds;                       // 128
      float *const sc = lds + 128;                 // len scores
      float *const pv = sc + ((len + 63) & ~63);   // [16][128] partial V sums
      if (tid < E3_HD) qv[tid] = rv[QOFF + h * E3_HD + tid];
      __syncthreads();
      float const scale = rsqrtf((float)E3_HD);
      float lmax = -3.0e38f;
      for (int t = t0 + tid; t < t1; t += blockDim.x) {
        uint4 const *kr = reinterpret_cast<uint4 const *>(g_e3.kc + ((size_t)t * E3_NKV + g) * E3_HD);
        uint4 kv[E3_HD / 8];
#pragma unroll
        for (int c = 0; c < E3_HD / 8; ++c) kv[c] = kr[c];
        float d = 0.f;
#pragma unroll
        for (int c = 0; c < E3_HD / 8; ++c) {
          uint4 const v = kv[c];
          float const *qq = qv + c * 8;
          d += e3_bf(v.x & 0xFFFFu) * qq[0] + e3_bf(v.x >> 16) * qq[1] + e3_bf(v.y & 0xFFFFu) * qq[2] +
               e3_bf(v.y >> 16) * qq[3] + e3_bf(v.z & 0xFFFFu) * qq[4] + e3_bf(v.z >> 16) * qq[5] +
               e3_bf(v.w & 0xFFFFu) * qq[6] + e3_bf(v.w >> 16) * qq[7];
        }
        d *= scale;
        sc[t - t0] = d;
        lmax = fmaxf(lmax, d);
      }
      float const mx = e3_block_max(lmax, e3_red);
      float lsum = 0.f;
      for (int t = t0 + tid; t < t1; t += blockDim.x) {
        float const e = expf(sc[t - t0] - mx);
        sc[t - t0] = e;
        lsum += e;
      }
      float const tot = e3_block_sum(lsum, e3_red);
      int const pg = tid >> 4, dc = tid & 15;  // 16 position groups x 16 lanes of 8 dims
      float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 16
      for (int t = t0 + pg; t < t1; t += 16) {
        uint4 const v = *reinterpret_cast<uint4 const *>(g_e3.vc + ((size_t)t * E3_NKV + g) * E3_HD + dc * 8);
        float const pt = sc[t - t0];
        a8[0] += pt * e3_bf(v.x & 0xFFFFu); a8[1] += pt * e3_bf(v.x >> 16);
        a8[2] += pt * e3_bf(v.y & 0xFFFFu); a8[3] += pt * e3_bf(v.y >> 16);
        a8[4] += pt * e3_bf(v.z & 0xFFFFu); a8[5] += pt * e3_bf(v.z >> 16);
        a8[6] += pt * e3_bf(v.w & 0xFFFFu); a8[7] += pt * e3_bf(v.w >> 16);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) pv[pg * E3_HD + dc * 8 + e] = a8[e];
      __syncthreads();
      float *const part = g_e3.vec + APOFF + (h * SP + sp) * 130;
      if (tid < E3_HD) {
        float sum = 0.f;
#pragma unroll
        for (int q = 0; q < 16; ++q) sum += pv[q * E3_HD + tid];
        e3_st_sys_f32(part + tid, sum);
      }
      if (tid == 0) {
        e3_st_sys_f32(part + 128, mx);
        e3_st_sys_f32(part + 129, tot);
      }
    }
  }
#elif defined(MPK_E3_ATTN_SPLIT)
  {
    constexpr int SP = 4;  // workers per q head
    static_assert(E3_NQ * SP * E3_HD <= E3_V_O - E3_V_F, "partial outputs in the silu(gate)*up region");
    static_assert(E3_NQ * SP * 2 <= E3_V_PI - E3_V_PV, "partial max / sum in the LM-head partial region");
    if (wid < E3_NQ * SP) {
      int const h = wid / SP, sp = wid % SP, g = h / (E3_NQ / E3_NKV);
      int const len = (P + 1 + SP - 1) / SP;
      int const t0 = sp * len, t1 = min(P + 1, t0 + len);
      float *const qv = lds;           // 128
      float *const sc = lds + 128;     // len scores
      float *const pv = sc + ((len + 63) & ~63);  // [16][128] partial V sums
      if (tid < E3_HD) qv[tid] = rv[QOFF + h * E3_HD + tid];
      __syncthreads();
      float const scale = rsqrtf((float)E3_HD);
      float lmax = -3.0e38f;
      for (int t = t0 + tid; t < t1; t += blockDim.x) {
        uint4 const *kr = reinterpret_cast<uint4 const *>(g_e3.kc + ((size_t)t * E3_NKV + g) * E3_HD);
        float d = 0.f;
#pragma unroll 4
        for (int c = 0; c < E3_HD / 8; ++c) {
          uint4 const v = kr[c];
          float const *qq = qv + c * 8;
          d += e3_bf(v.x & 0xFFFFu) * qq[0] + e3_bf(v.x >> 16) * qq[1] + e3_bf(v.y & 0xFFFFu) * qq[2] +
               e3_bf(v.y >> 16) * qq[3] + e3_bf(v.z & 0xFFFFu) * qq[4] + e3_bf(v.z >> 16) * qq[5] +
               e3_bf(v.w & 0xFFFFu) * qq[6] + e3_bf(v.w >> 16) * qq[7];
        }
        d *= scale;
        sc[t - t0] = d;
        lmax = fmaxf(lmax, d);
      }
      float const mx = e3_block_max(lmax, e3_red);
      float lsum = 0.f;
      for (int t = t0 + tid; t < t1; t += blockDim.x) {
        float const e = expf(sc[t - t0] - mx);
        sc[t - t0] = e;
        lsum += e;
      }
      float const tot = e3_block_sum(lsum, e3_red);
      int const pg = tid >> 4, dc = tid & 15;  // 16 position groups x 16 lanes of 8 dims
      float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int t = t0 + pg; t < t1; t += 16) {
        uint4 const v = *reinterpret_cast<uint4 const *>(g_e3.vc + ((size_t)t * E3_NKV + g) * E3_HD + dc * 8);
        float const pt = sc[t - t0];
        a8[0] += pt * e3_bf(v.x & 0xFFFFu); a8[1] += pt * e3_bf(v.x >> 16);
        a8[2] += pt * e3_bf(v.y & 0xFFFFu); a8[3] += pt * e3_bf(v.y >> 16);
        a8[4] += pt * e3_bf(v.z & 0xFFFFu); a8[5] += pt * e3_bf(v.z >> 16);
        a8[6] += pt * e3_bf(v.w & 0xFFFFu); a8[7] += pt * e3_bf(v.w >> 16);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) pv[pg * E3_HD + dc * 8 + e] = a8[e];
      __syncthreads();
      if (tid < E3_HD) {
        float sum = 0.f;
#pragma unroll
        for (int q = 0; q < 16; ++q) sum += pv[q * E3_HD + tid];
        e3_st2f(E3_V_F + (h * SP + sp) * E3_HD + tid, sum);
      }
      if (tid == 0) {
        e3_st2f(E3_V_PV + (h * SP + sp) * 2, mx);
        e3_st2f(E3_V_PV + (h * SP + sp) * 2 + 1, tot);
      }
    }
    E3TS();
    e3_barrier(++gen, die);
    if (wid < E3_NQ && tid < E3_HD) {
      int const h = wid;
      float ms[SP], ls[SP];
#pragma unroll
      for (int q = 0; q < SP; ++q) {
        ms[q] = __uint_as_float(e3_ld_sys_u32((unsigned const *)(g_e3.vec + E3_V_PV + (h * SP + q) * 2)));
        ls[q] = __uint_as_float(e3_ld_sys_u32((unsigned const *)(g_e3.vec + E3_V_PV + (h * SP + q) * 2 + 1)));
      }
      float m = ms[0];
#pragma unroll
      for (int q = 1; q < SP; ++q) m = fmaxf(m, ms[q]);
      float l = 0.f, o = 0.f;
#pragma unroll
      for (int q = 0; q < SP; ++q) {
        float const f = expf(ms[q] - m);
        l += ls[q] * f;
        o += __uint_as_float(e3_ld_sys_u32((unsigned const *)(g_e3.vec + E3_V_F + (h * SP + q) * E3_HD + tid))) * f;
      }
      e3_st2f(E3_V_A + h * E3_HD + tid, e3_rbf(o / l));
    }
  }
  constexpr int E3_GX = 1;
#else
  constexpr int E3_GX = 0;
  if (wid < E3_NQ) {
    int const h = wid, g = h / (E3_NQ / E3_NKV);
    float *const qv = lds;           // 128
    float *const sc = lds + 128;     // P + 1 scores
    float *const pv = sc + ((P + 1 + 63) & ~63);  // [16][128] partial V sums
    if (tid < E3_HD) qv[tid] = rv[QOFF + h * E3_HD + tid];
    __syncthreads();
    float const scale = rsqrtf((float)E3_HD);
    float lmax = -3.0e38f;
#ifdef MPK_E3_ATTN_ILP
    for (int t = tid; t <= P; t += 2 * blockDim.x) {
      int const t2 = t + blockDim.x;
      bool const has2 = t2 <= P;
      uint4 const *kr = reinterpret_cast<uint4 const *>(g_e3.kc + ((size_t)t * E3_NKV + g) * E3_HD);
      uint4 const *kr2 = reinterpret_cast<uint4 const *>(g_e3.kc + ((size_t)(has2 ? t2 : t) * E3_NKV + g) * E3_HD);
      uint4 v1[E3_HD / 8], v2[E3_HD / 8];
#pragma unroll
      for (int c = 0; c < E3_HD / 8; ++c) {
        v1[c] = kr[c];
        v2[c] = kr2[c];
      }
      float d = 0.f, d2 = 0.f;
#pragma unroll
      for (int c = 0; c < E3_HD / 8; ++c) {
        uint4 const v = v1[c];
        uint4 const w = v2[c];
        float const *qq = qv + c * 8;
        d += e3_bf(v.x & 0xFFFFu) * qq[0] + e3_bf(v.x >> 16) * qq[1] + e3_bf(v.y & 0xFFFFu) * qq[2] +
             e3_bf(v.y >> 16) * qq[3] + e3_bf(v.z & 0xFFFFu) * qq[4] + e3_bf(v.z >> 16) * qq[5] +
             e3_bf(v.w & 0xFFFFu) * qq[6] + e3_bf(v.w >> 16) * qq[7];
        d2 += e3_bf(w.x & 0xFFFFu) * qq[0] + e3_bf(w.x >> 16) * qq[1] + e3_bf(w.y & 0xFFFFu) * qq[2] +
              e3_bf(w.y >> 16) * qq[3] + e3_bf(w.z & 0xFFFFu) * qq[4] + e3_bf(w.z >> 16) * qq[5] +
              e3_bf(w.w & 0xFFFFu) * qq[6] + e3_bf(w.w >> 16) * qq[7];
      }
      d *= scale;
      sc[t] = d;
      lmax = fmaxf(lmax, d);
      if (has2) {
        d2 *= scale;
        sc[t2] = d2;
        lmax = fmaxf(lmax, d2);
      }
    }
#else
    for (int t = tid; t <= P; t += blockDim.x) {
      uint4 const *kr = reinterpret_cast<uint4 const *>(g_e3.kc + ((size_t)t * E3_NKV + g) * E3_HD);
      float d = 0.f;
#pragma unroll 4
      for (int c = 0; c < E3_HD / 8; ++c) {
        uint4 const v = kr[c];
        float const *qq = qv + c * 8;
        d += e3_bf(v.x & 0xFFFFu) * qq[0] + e3_bf(v.x >> 16) * qq[1] + e3_bf(v.y & 0xFFFFu) * qq[2] +
             e3_bf(v.y >> 16) * qq[3] + e3_bf(v.z & 0xFFFFu) * qq[4] + e3_bf(v.z >> 16) * qq[5] +
             e3_bf(v.w & 0xFFFFu) * qq[6] + e3_bf(v.w >> 16) * qq[7];
      }
      d *= scale;
      sc[t] = d;
      lmax = fmaxf(lmax, d);
    }
#endif
    float const mx = e3_block_max(lmax, e3_red);
    float lsum = 0.f;
    for (int t = tid; t <= P; t += blockDim.x) {
      float const e = expf(sc[t] - mx);
      sc[t] = e;
      lsum += e;
    }
    float const tot = e3_block_sum(lsum, e3_red);
    int const pg = tid >> 4, dc = tid & 15;  // 16 position groups x 16 lanes of 8 dims
    float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#ifdef MPK_E3_ATTN_ILP
#pragma unroll 16
#else
#pragma unroll 4
#endif
    for (int t = pg; t <= P; t += 16) {
      uint4 const v = *reinterpret_cast<uint4 const *>(g_e3.vc + ((size_t)t * E3_NKV + g) * E3_HD + dc * 8);
      float const pt = sc[t];
      a8[0] += pt * e3_bf(v.x & 0xFFFFu); a8[1] += pt * e3_bf(v.x >> 16);
      a8[2] += pt * e3_bf(v.y & 0xFFFFu); a8[3] += pt * e3_bf(v.y >> 16);
      a8[4] += pt * e3_bf(v.z & 0xFFFFu); a8[5] += pt * e3_bf(v.z >> 16);
      a8[6] += pt * e3_bf(v.w & 0xFFFFu); a8[7] += pt * e3_bf(v.w >> 16);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) pv[pg * E3_HD + dc * 8 + e] = a8[e];
    __syncthreads();
    if (tid < E3_HD) {
      float sum = 0.f;
#pragma unroll
      for (int q = 0; q < 16; ++q) sum += pv[q * E3_HD + tid];
      e3_st2f(E3_V_A + h * E3_HD + tid, e3_rbf(sum / tot));
    }
  }
#endif
  E3TS();
  e3_barrier(++gen, die);
  E3TS();

  // ---- stage 4: r2 = o_proj(attn) + h (last row)
  {
    float *const x = lds;  // 4096
#ifdef MPK_E3_IDLE_SKIP
    if (e3_tile180(wid, W) >= 0) {  // workers without an o_proj row tile skip the combine + quantize
#endif
#ifdef MPK_E3_WPF
#if !defined(MPK_E3_IDLE_SKIP) || !defined(MPK_E3_CMB1) || !defined(MPK_E3_ATTN_SPLIT2) || !defined(MPK_E3_Q8M)
#error "MPK_E3_WPF: the o_proj preload builds on MPK_E3_IDLE_SKIP + MPK_E3_CMB1 + MPK_E3_ATTN_SPLIT2 + MPK_E3_Q8M"
#endif
    E3Pre8 opre;
#endif
#if defined(MPK_E3_ATTN_GQA)
    e3_load_rep(x, rv, AOFF, E3_NQ * E3_HD);
#elif defined(MPK_E3_ATTN_SPLIT2)
    {
      constexpr int SP = MPK_E3_ATTN_SPLIT2;
#ifdef MPK_E3_CMB1
      constexpr int PER = E3_NQ * E3_HD / 256;  // 16 outputs per thread
      float ov[PER][SP];
#pragma unroll
      for (int k = 0; k < PER; ++k) {
        int const i = tid + k * 256;
        float const *const pi = rv + APOFF + (i >> 7) * SP * 130 + (i & 127);
#pragma unroll
        for (int q = 0; q < SP; ++q) ov[k][q] = pi[q * 130];
      }
#ifdef MPK_E3_WPF
      {
        constexpr int KT = E3_NQ * E3_HD / 128, KQ = KT / 4;
        e3_pre8((unsigned char const *)g_e3.wo + ((size_t)e3_tile180(wid, W) * KT + (tid >> 6) * KQ) * 2048, opre);
      }
#endif
#endif
      float *const fac = lds + E3_NQ * E3_HD;  // [32][SP] exp(m_sp - M) / L
      if (tid < E3_NQ) {
        float const *const ph = rv + APOFF + tid * SP * 130;
        float m[SP], l[SP];
        float M = -3.0e38f;
#pragma unroll
        for (int q = 0; q < SP; ++q) {
          m[q] = ph[q * 130 + 128];
          l[q] = ph[q * 130 + 129];
          M = fmaxf(M, m[q]);
        }
        float L = 0.f;
#pragma unroll
        for (int q = 0; q < SP; ++q) L += l[q] * expf(m[q] - M);
#pragma unroll
        for (int q = 0; q < SP; ++q) fac[tid * SP + q] = expf(m[q] - M) / L;
      }
      __syncthreads();
#ifndef MPK_E3_CMB1
      constexpr int PER = E3_NQ * E3_HD / 256;  // 16 outputs per thread
      float ov[PER][SP];
#pragma unroll
      for (int k = 0; k < PER; ++k) {
        int const i = tid + k * 256;
        float const *const pi = rv + APOFF + (i >> 7) * SP * 130 + (i & 127);
#pragma unroll
        for (int q = 0; q < SP; ++q) ov[k][q] = pi[q * 130];
      }
#endif
#pragma unroll
      for (int k = 0; k < PER; ++k) {
        int const i = tid + k * 256;
        float o = 0.f;
#pragma unroll
        for (int q = 0; q < SP; ++q) o += ov[k][q] * fac[(i >> 7) * SP + q];
        x[i] = e3_rbf(o);
      }
      __syncthreads();
    }
#else
    e3_load_rep(x, rv, E3_V_A, E3_NQ * E3_HD);
#endif
    E3TS();
#ifdef MPK_E3_Q8M
    {
      constexpr int KP = E3_NQ * E3_HD, KT = KP / 128, KQ = KT / 4;  // 32 K tiles, 8 per wave
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + 16384);
      unsigned char *const qs = q8 + KP;
      float *const s_red = reinterpret_cast<float *>(qs + 256);  // [4 waves][16][2]
      e3_q8act<KP, KP>(x, q8, qs);
      int const dt = e3_tile180(wid, W);
      int const wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
      if (dt >= 0) {
        unsigned char const *const w0 = (unsigned char const *)g_e3.wo + ((size_t)dt * KT + wave * KQ) * 2048;
        unsigned char const *wr[2] = {w0, w0};
        e3_f32x4 acc[2];
#ifdef MPK_E3_WPF
        e3_m8gemv<KQ, 1, 4, true>(wr, q8 + wave * KQ * 128, KP, qs + wave * KQ, KT, 1, acc, &opre);
#else
        e3_m8gemv<KQ, 1, 4>(wr, q8 + wave * KQ * 128, KP, qs + wave * KQ, KT, 1, acc);
#endif
        if (col == 0)
#pragma unroll
          for (int i = 0; i < 4; ++i) s_red[((wave * 16) + gq * 4 + i) * 2] = acc[0][i];
      }
      __syncthreads();
      if (dt >= 0 && tid < 16) {
        int const o = dt * 16 + tid;
        float const v = (s_red[tid * 2] + s_red[(16 + tid) * 2]) + (s_red[(32 + tid) * 2] + s_red[(48 + tid) * 2]);
        float const res = rv[HSRC + (Rr - 1) * E3_H + o];
        e3_st2f(ROFF + o, e3_rbf(e3_rbf(v * g_e3.swo[o]) + res));
      }
    }
#else
    int o0, o1;
    e3_split(E3_H, wid, W, o0, o1);
    e3_gemv4<E3_NQ * E3_HD, 1>(g_e3.wo, g_e3.swo, x, 0, o0, o1, [&](int o, auto const &acc) {
      float const res = rv[HSRC + (Rr - 1) * E3_H + o];
      e3_st2f(ROFF + o, e3_rbf(e3_rbf(acc[0]) + res));
    });
#endif
#ifdef MPK_E3_IDLE_SKIP
    }
#endif
  }
  E3TS();
  e3_barrier(++gen, die);
  E3TS();

  // ---- stage 5: act = silu(gate(m)) * up(m), m = post_attention_layernorm(r2)
  {
    float *const r2 = lds;
    float *const mm = lds + E3_H;
    e3_load_rep(r2, rv, ROFF, E3_H);
#ifdef MPK_E3_WPF
    static_assert(E3_FF / 16 >= 248, "every worker has a gate/up tile pair");
    int gt0, gt1;
    e3_bsplit(E3_FF / 16, wid, W, gt0, gt1);
    unsigned char const *gwr[3], *gsr[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      int const rt = min((tid >> 6) + 4 * j, max(2 * (gt1 - gt0) - 1, 0));
      size_t const tile = (size_t)(gt0 + (rt >> 1));
      gwr[j] = (unsigned char const *)((rt & 1) ? g_e3.wu : g_e3.wg) + tile * E3QT<E3_HQ>::DATA;
      gsr[j] = (unsigned char const *)((rt & 1) ? g_e3.swu : g_e3.swg) + tile * E3QT<E3_HQ>::SC;
    }
    E3Pre4<4> gpre;
    e3_pre4<4, 3>(gwr, gsr, gpre);
#endif
    E3TS();
    e3_rms_row(r2, mm, g_e3.n_post, e3_red);
    E3TS();
    int o0, o1;
    e3_split(E3_FF, wid, W, o0, o1);
    auto act = [&](int o, float g, float u) {
      float const gg = e3_rbf(g), uu = e3_rbf(u);
      float const sg = e3_rbf(gg / (1.f + expf(-gg)));
      e3_st2f(FOFF + o, e3_rbf(sg * uu));
    };
#ifdef MPK_E3_PIPE
    // Two outputs (gate + up rows) per group, groups pipelined per wave.
#ifndef MPK_E3_PIPE_GU
#define MPK_E3_PIPE_GU 2  // outputs (gate + up row pairs) per pipelined group
#endif
    constexpr int GU = MPK_E3_PIPE_GU;
    e3_gemv_pipe<E3_H, 2 * GU>(
        0, (o1 - o0 + GU - 1) / GU,
        [&](int g, int a) {
          size_t const oc = (size_t)min(o0 + GU * g + (a >> 1), o1 - 1);
          return (a & 1) ? e3_row(g_e3.wu, g_e3.swu, oc, E3_H) : e3_row(g_e3.wg, g_e3.swg, oc, E3_H);
        },
        mm, [&](int g, float const (&acc)[2 * GU]) {
#pragma unroll
          for (int j = 0; j < GU; ++j) {
            int const o = o0 + GU * g + j;
            if (o < o1) act(o, acc[2 * j], acc[2 * j + 1]);
          }
        });
#elif defined(MPK_E3_Q4M)
    {
      // 16-output tiles split evenly over the workers; a worker's gate and up row tiles over its 4 waves.
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + 16384);
      unsigned char *const qs = q8 + 16384;
      float *const s_gu = reinterpret_cast<float *>(qs + 128);  // [row tile][16]
      e3_q8act<E3_H, E3_HQ>(mm, q8, qs);
      int t0, t1;
      e3_bsplit(E3_FF / 16, wid, W, t0, t1);
      int const nrt = 2 * (t1 - t0), wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
      unsigned char const *wr[3], *sr[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        int const rt = min(wave + 4 * j, max(nrt - 1, 0));
        size_t const tile = (size_t)(t0 + (rt >> 1));
        wr[j] = (unsigned char const *)((rt & 1) ? g_e3.wu : g_e3.wg) + tile * E3QT<E3_HQ>::DATA;
        sr[j] = (unsigned char const *)((rt & 1) ? g_e3.swu : g_e3.swg) + tile * E3QT<E3_HQ>::SC;
      }
      int const nv = nrt > wave ? (nrt - wave + 3) / 4 : 0;
      e3_f32x4 acc[3];
#ifdef MPK_E3_WPF
      e3_mgemv_nv_pre<E3_HQ / 128>(wr, sr, q8, qs, acc, nv, gpre);
#else
      e3_mgemv_nv<E3_HQ / 128>(wr, sr, q8, qs, acc, nv);
#endif
      if (col == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
          if (j < nv)
#pragma unroll
            for (int i = 0; i < 4; ++i) s_gu[(wave + 4 * j) * 16 + gq * 4 + i] = acc[j][i];
      }
      __syncthreads();
      for (int q = tid; q < (t1 - t0) * 16; q += 256)
        act(t0 * 16 + q, s_gu[(2 * (q >> 4)) * 16 + (q & 15)], s_gu[(2 * (q >> 4) + 1) * 16 + (q & 15)]);
    }
#else
    for (int ob = o0 + (tid >> 6) * 4; ob < o1; ob += 16) {  // four outputs (gate + up rows) per wave pass
#ifdef MPK_E3_FP4
      unsigned char const *wp[8], *sp[8];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        size_t const oc = (size_t)min(ob + a, o1 - 1);
        wp[2 * a] = (unsigned char const *)g_e3.wg + oc * (size_t)(E3_H / 2);
        wp[2 * a + 1] = (unsigned char const *)g_e3.wu + oc * (size_t)(E3_H / 2);
        sp[2 * a] = (unsigned char const *)g_e3.swg + oc * (size_t)E3Q4<E3_H>::SROW;
        sp[2 * a + 1] = (unsigned char const *)g_e3.swu + oc * (size_t)E3Q4<E3_H>::SROW;
      }
      unsigned char const *const (&wc)[8] = wp;
      unsigned char const *const (&scc)[8] = sp;
      float acc[8][1];
      e3_dots4<E3_H, 8, 1>(wc, scc, mm, 0, acc);
#else
      E3Row wr[8];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        size_t const oc = (size_t)min(ob + a, o1 - 1);
        wr[2 * a] = e3_row(g_e3.wg, g_e3.swg, oc, E3_H);
        wr[2 * a + 1] = e3_row(g_e3.wu, g_e3.swu, oc, E3_H);
      }
      float acc[8][1];
#ifdef MPK_E3_HALFK
      e3_dotsr<E3_H / 2, 8, 1>(wr, mm, 0, acc);
#else
      e3_dotsr<E3_H, 8, 1>(wr, mm, 0, acc);
#endif
#endif
      if ((tid & 63) == 0) {
#pragma unroll
        for (int a = 0; a < 4; ++a)
          if (ob + a < o1) act(ob + a, acc[2 * a][0], acc[2 * a + 1][0]);
      }
    }
#endif
  }
  E3TS();
  e3_barrier(++gen, die);
  E3TS();

  // ---- stage 6: out = down(act) + r2
  {
    float *const x = lds;  // 16384
#ifdef MPK_E3_IDLE_SKIP
    if (e3_tile180(wid, W) >= 0) {  // workers without a down row tile skip the load + quantize
#endif
    e3_load_rep(x, rv, FOFF, E3_FF);
#ifdef MPK_E3_WPF
#ifndef MPK_E3_IDLE_SKIP
#error "MPK_E3_WPF: the down preload assumes MPK_E3_IDLE_SKIP (workers with a row tile only)"
#endif
    E3Pre4<8> dpre;
    {
      constexpr int KQ = E3_FF / 128 / 4;
      int const dtp = e3_tile180(wid, W);
      unsigned char const *const w0 = (unsigned char const *)g_e3.wd + (size_t)dtp * E3QT<E3_FF>::DATA + (tid >> 6) * KQ * 1024;
      unsigned char const *const s0 = (unsigned char const *)g_e3.swd + (size_t)dtp * E3QT<E3_FF>::SC + (tid >> 6) * (KQ / 4) * 256;
      unsigned char const *pw[3] = {w0, w0, w0};
      unsigned char const *ps[3] = {s0, s0, s0};
      e3_pre4<8, 1>(pw, ps, dpre);
    }
#endif
    E3TS();
    int o0, o1;
    e3_split(E3_H, wid, W, o0, o1);
#ifdef MPK_E3_Q4M
    {
      // 180 row tiles: AID0's half on workers 0..89, AID1's half on workers W/2 .. W/2 + 89 (the own-BO cut);
      // each worker's 4 waves split K and sum through LDS.
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + 16384);
      unsigned char *const qs = q8 + 16384;
      float *const s_red = reinterpret_cast<float *>(qs + 128);  // [4 waves][16]
      e3_q8act<E3_FF, E3_FF>(x, q8, qs);
      constexpr int TH = E3_H / 16 / 2;
      constexpr int KQ = E3_FF / 128 / 4;  // K tiles per wave
      int const WH = W >> 1;
      int const dt = WH >= TH ? (wid < WH ? (wid < TH ? wid : -1) : (wid - WH < TH ? TH + wid - WH : -1))
                              : (wid < 2 * TH ? wid : -1);
      int const wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
      if (dt >= 0) {
        static_assert(KQ % 4 == 0, "a wave's K range starts on a scale group");
        unsigned char const *const w0 = (unsigned char const *)g_e3.wd + (size_t)dt * E3QT<E3_FF>::DATA + wave * KQ * 1024;
        unsigned char const *const s0 = (unsigned char const *)g_e3.swd + (size_t)dt * E3QT<E3_FF>::SC + wave * (KQ / 4) * 256;
        unsigned char const *wr[3] = {w0, w0, w0};
        unsigned char const *sr[3] = {s0, s0, s0};
        e3_f32x4 acc[3];
#ifdef MPK_E3_WPF
        e3_mgemv<KQ, 1, 8, true, 8>(wr, sr, q8 + wave * KQ * 128, qs + wave * KQ, acc, &dpre);
#else
        e3_mgemv<KQ, 1, 8>(wr, sr, q8 + wave * KQ * 128, qs + wave * KQ, acc);
#endif
        if (col == 0)
#pragma unroll
          for (int i = 0; i < 4; ++i) s_red[wave * 16 + gq * 4 + i] = acc[0][i];
      }
      __syncthreads();
      if (dt >= 0 && tid < 16) {
        int const o = dt * 16 + tid;
        float const sum = (s_red[tid] + s_red[16 + tid]) + (s_red[32 + tid] + s_red[48 + tid]);
        e3_st2f(OOFF + o, e3_rbf(e3_rbf(sum) + rv[ROFF + o]));
      }
    }
#else
#ifdef MPK_E3_HALFK
    e3_gemv4<E3_FF, 1, E3_FF / 2>(g_e3.wd, g_e3.swd, x, 0, o0, o1, [&](int o, auto const &acc) {
#elif defined(MPK_E3_FP4)
    e3_gemv4q<E3_FF, 1>(g_e3.wd, g_e3.swd, x, 0, o0, o1, [&](int o, auto const &acc) {
#else
    e3_gemv4<E3_FF, 1>(g_e3.wd, g_e3.swd, x, 0, o0, o1, [&](int o, auto const &acc) {
#endif
      float const res = rv[ROFF + o];
      e3_st2f(OOFF + o, e3_rbf(e3_rbf(acc[0]) + res));
    });
#endif
#ifdef MPK_E3_IDLE_SKIP
    }
#endif
  }
  E3TS();
  e3_barrier(++gen, die);
  E3TS();

  // ---- stage 7: logits = lm_head(norm(out)); per-worker argmax. The last step's last arrival reduces and
  // publishes; earlier steps end in a barrier and the next step reduces the partials on every worker.
  {
    float *const ov = lds;
    float *const fv = lds + E3_H;
    float *const wbest = lds + 2 * E3_H;  // 4 waves x (value, index)
    e3_load_rep(ov, rv, OOFF, E3_H);
#ifdef MPK_E3_WPF
    int lt0, lt1;
    e3_bsplit(E3_VOCAB / 16, wid, W, lt0, lt1);
    unsigned char const *lwr[3], *lsr[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      size_t const tile = (size_t)min(lt0 + (tid >> 6) + 4 * j, max(lt1 - 1, 0));
      lwr[j] = (unsigned char const *)g_e3.wlm + tile * E3QT<E3_HQ>::DATA;
      lsr[j] = (unsigned char const *)g_e3.swlm + tile * E3QT<E3_HQ>::SC;
    }
    E3Pre4<4> lpre;
    e3_pre4<4, 3>(lwr, lsr, lpre);
#endif
    E3TS();
    e3_rms_row(ov, fv, g_e3.n_final, e3_red);
    E3TS();
    int o0, o1;
    e3_split(E3_VOCAB, wid, W, o0, o1);
    float best = -3.0e38f;
    int bidx = -1;
#ifdef MPK_E3_CONF
    float se = 0.f;  // sum of exp(v - best) over the lane's values
#endif
#ifdef MPK_E3_PIPE
#ifndef MPK_E3_PIPE_LM
#define MPK_E3_PIPE_LM 4  // rows per pipelined group
#endif
    constexpr int LMG = MPK_E3_PIPE_LM;
    e3_gemv_pipe<E3_H, LMG>(
        0, (o1 - o0 + LMG - 1) / LMG,
        [&](int g, int a) { return e3_row(g_e3.wlm, g_e3.swlm, (size_t)min(o0 + LMG * g + a, o1 - 1), E3_H); },
        fv, [&](int g, float const (&acc)[LMG]) {
#pragma unroll
          for (int a = 0; a < LMG; ++a) {
            int const o = o0 + LMG * g + a;
            if (o < o1 && acc[a] > best) {  // a wave's rows come in increasing order: ties keep the lower index
              best = acc[a];
              bidx = o;
            }
          }
        });
#elif defined(MPK_E3_Q4M)
    {
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + 16384);
      unsigned char *const qs = q8 + 16384;
      e3_q8act<E3_H, E3_HQ>(fv, q8, qs);
      int t0, t1;
      e3_bsplit(E3_VOCAB / 16, wid, W, t0, t1);
      int const wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
      unsigned char const *wr[3], *sr[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        size_t const tile = (size_t)min(t0 + wave + 4 * j, max(t1 - 1, 0));
        wr[j] = (unsigned char const *)g_e3.wlm + tile * E3QT<E3_HQ>::DATA;
        sr[j] = (unsigned char const *)g_e3.swlm + tile * E3QT<E3_HQ>::SC;
      }
      int const nt = t1 - t0, nv = nt > wave ? (nt - wave + 3) / 4 : 0;
      e3_f32x4 acc[3];
#ifdef MPK_E3_WPF
      e3_mgemv_nv_pre<E3_HQ / 128>(wr, sr, q8, qs, acc, nv, lpre);
#else
      e3_mgemv_nv<E3_HQ / 128>(wr, sr, q8, qs, acc, nv);
#endif
      if (col == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
          if (j < nv)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              int const o = (t0 + wave + 4 * j) * 16 + gq * 4 + i;
              float const v = acc[j][i];
#ifdef MPK_E3_CONF
              {
                float const nb = fmaxf(best, v);
                se = se * __expf(best - nb) + __expf(v - nb);
              }
#endif
              if (v > best || (v == best && o < bidx)) {  // ties keep the lower index
                best = v;
                bidx = o;
              }
            }
      }
#pragma unroll
      for (int m = 16; m <= 32; m <<= 1) {  // the 4 result lanes (0, 16, 32, 48) -> lane 0
        float const ob = __shfl_xor(best, m);
        int const oi = __shfl_xor(bidx, m);
#ifdef MPK_E3_CONF
        {
          float const os = __shfl_xor(se, m);
          float const nb = fmaxf(best, ob);
          se = se * __expf(best - nb) + os * __expf(ob - nb);
        }
#endif
        if (ob > best || (ob == best && oi >= 0 && (bidx < 0 || oi < bidx))) {
          best = ob;
          bidx = oi;
        }
      }
    }
#else
#ifdef MPK_E3_LMG8
    e3_gemvg<E3_H, 1, 8>(g_e3.wlm, g_e3.swlm, fv, 0, o0, o1, [&](int o, auto const &acc) {
#elif defined(MPK_E3_HALFK)
    e3_gemv4<E3_H, 1, E3_H / 2>(g_e3.wlm, g_e3.swlm, fv, 0, o0, o1, [&](int o, auto const &acc) {
#elif defined(MPK_E3_FP4)
    e3_gemv4q<E3_H, 1>(g_e3.wlm, g_e3.swlm, fv, 0, o0, o1, [&](int o, auto const &acc) {
#else
    e3_gemv4<E3_H, 1>(g_e3.wlm, g_e3.swlm, fv, 0, o0, o1, [&](int o, auto const &acc) {
#endif
      if (acc[0] > best) {  // a wave's rows come in increasing order, so ties keep the lower index
        best = acc[0];
        bidx = o;
      }
    });
#endif
    if ((tid & 63) == 0) {
      wbest[(tid >> 6) * 2] = best;
      wbest[(tid >> 6) * 2 + 1] = __int_as_float(bidx);
#ifdef MPK_E3_CONF
      wbest[8 + (tid >> 6)] = se;
#endif
    }
    __syncthreads();
    if (tid == 0) {
      float bv = -3.0e38f;
      int bi = -1;
#ifdef MPK_E3_CONF
      float bm = -3.0e38f, bs = 0.f;
#endif
      for (int w4 = 0; w4 < 4; ++w4) {
        float const v = wbest[w4 * 2];
        int const i = __float_as_int(wbest[w4 * 2 + 1]);
#ifdef MPK_E3_CONF
        if (i >= 0) {
          float const nb = fmaxf(bm, v);
          bs = bs * __expf(bm - nb) + wbest[8 + w4] * __expf(v - nb);
          bm = nb;
        }
#endif
        if (i >= 0 && (v > bv || (v == bv && i < bi))) {
          bv = v;
          bi = i;
        }
      }
      e3_st2f(PVOFF + wid, bv);
      e3_st2(PIOFF + wid, (unsigned)bi);
#ifdef MPK_E3_CONF
      e3_st2f(e3_reg(E3_V_PS, jc, E3_CPS) + wid, bs);  // relative to bm == bv
#endif
      if (jc == E3_K - 1) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        int *const cnt = g_aid_nc_rep[0] + E3_NC_INTS + 9 * 64;
        int const c = __hip_atomic_fetch_add(cnt, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        e3_red[7] = __int_as_float(c % W == W - 1 ? 1 : 0);
      }
    }
    __syncthreads();
    if (jc == E3_K - 1 && __float_as_int(e3_red[7]) == 1) {  // the last arrival's block reduces the W partials
      int const pi = tid < W ? (int)__float_as_uint(rv[PIOFF + tid]) : -1;
      float const pvv = (tid < W && pi >= 0) ? rv[PVOFF + tid] : -3.0e38f;
#ifdef MPK_E3_CONF
      float const psv = (tid < W && pi >= 0) ? rv[e3_reg(E3_V_PS, jc, E3_CPS) + tid] : 0.f;
      float Mx;
      int const fi = e3_block_argmax_v(pvv, pi >= 0 ? pi : 0x7fffffff, e3_red, Mx);
      float const pj = e3_conf_p(pvv, psv, Mx, e3_red);
#else
      int const fi = e3_block_argmax(pvv, pi >= 0 ? pi : 0x7fffffff, e3_red);
#endif
      if (tid == 0) {
        long long const d = (long long)fi + g_e3.d2t[fi];
#ifdef MPK_E3_CONF
        __hip_atomic_store(&g_e3.draft_out[E3_CONF_SLOT + jc], (long long)__float_as_uint(pj), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#endif
        __hip_atomic_store(&g_e3.draft_out[jc], d, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&g_e3.draft_out[7], (long long)s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  }
  if (jc < E3_K - 1) {
    E3TS();
    e3_barrier(++gen, die);  // step jc + 1 reduces every worker's partials
  }
  }  // chain steps
#ifdef MPK_EAGLE3_TS
  if (tid == 0 && s == prompt_len + 20) {
    e3_st_sys_u32(g_e3.vec + E3_V_SPAN + 3 * wid + 2, (unsigned)__builtin_amdgcn_s_memrealtime());
  }
#endif
  E3TS();
  E3TS_DUMP("full");
}

}  // namespace kernel

