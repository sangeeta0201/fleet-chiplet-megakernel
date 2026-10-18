#pragma once
// MPK_DFLASH: the z-lab DFlash block drafter for gpt-oss-120b (8 Qwen3 layers; the target features of layers
// 1, 9, 17, 25, 33 enter every layer as extra K/V), run on every worker at the end of the LM-head task
// (MPK_SPEC_VSLOT passes). Each pass appends the context K/V of the rows whose target features are final
// (every slot of a prefill pass, the accepted slots of a verify pass). When the next pass is a verify pass,
// the block (the next anchor token + DF_B - 1 mask tokens, attending to the context and bidirectionally to
// itself) runs through the 8 layers and the target LM head; the DF_B - 1 drafts go to g_df.draft_out for
// prepare_next_batch.
//
// Target features need no capture: under MPK_LAYER_RING the hidden state after layer L is attn_proj_out ring
// copy L plus the four slots of moe_workspace_f32 ring copy L + 1 (read at system scope).
// Scratch: every location is written once per pass with write-through stores before any read (one region per
// draft layer), so plain loads are sound; the scheduler's per-pass agent-scope acquires drop the lines of
// earlier passes. Partials read right after an atomic use system-scope loads. Context K/V rows are written
// once: only committed positions are appended.
#ifdef MPK_EAGLE3
#error "MPK_DFLASH and MPK_EAGLE3 share the NC barrier lines and the release slot"
#endif
#ifndef MPK_SPEC_VSLOT
#error "MPK_DFLASH drafts for MPK_SPEC_VSLOT verify passes"
#endif

#ifndef MPK_MULTI_DEV
extern __device__ __attribute__((aligned(16))) int g_spec_state[4];
extern __device__ long long g_spec_prev_in[MPK_MAX_NUM_BATCHED_TOKENS];
#endif

namespace kernel {

constexpr int DF_H = 2880;
constexpr int DF_HP = 2944;  // padded hidden row of the target's ring buffers = DF_H padded to 128
constexpr int DF_NTL = 5;    // target layers
constexpr int DF_FEAT = DF_NTL * DF_H;
constexpr int DF_FEATP = 14464;
constexpr int DF_NL = 8;
constexpr int DF_NQ = 64, DF_NKV = 8, DF_HD = 64, DF_GQ = DF_NQ / DF_NKV;
constexpr int DF_QD = DF_NQ * DF_HD;
constexpr int DF_KVD = DF_NKV * DF_HD;
constexpr int DF_FF = 7680;
constexpr int DF_VOCAB = 201088;
constexpr int DF_B = MPK_MAX_NUM_BATCHED_TOKENS;  // block = anchor + DF_B - 1 masks; also rows per pass
static_assert(DF_B == 4 || DF_B == 6 || DF_B == 8, "MPK_DFLASH: blocks of 4, 6 or 8 tokens");
// GEMV output columns: the MFMA computes 16 activation columns per tile, df_round8 keeps the first DF_RC (a power of
// two >= DF_B), and a GEMV epilogue maps DF_TPT = 16 * DF_RC threads per 16-row tile as (row, column). DF_RC = 4 is
// the block-4 layout.
#if MPK_MAX_NUM_BATCHED_TOKENS > 4
#define DF_SINL __noinline__
#else
#define DF_SINL __forceinline__
#endif
constexpr int DF_RC = DF_B <= 4 ? 4 : 8;
constexpr int DF_RCS = DF_RC == 4 ? 2 : 3;  // log2(DF_RC)
constexpr int DF_TPT = 16 * DF_RC;
static_assert(2 * DF_TPT <= 256, "a round's two tiles in one pass of the 256 threads");
// Bytes reserved after a DF_B x DF_KT_H scale block (q8 rows of DF_HP) before the next LDS region.
constexpr int DF_QSP = DF_B <= 4 ? 128 : 256;
static_assert(DF_B * (DF_HP / 128) <= DF_QSP && DF_B * (DF_QD / 128) <= DF_QSP, "scale rows fit their reserve");
constexpr long long DF_MASK = 200000;
constexpr int DF_CONF_SLOT = 9;  // draft_out[9 ..]: the drafts' softmax probabilities (demo.py allocates 16 slots)
static_assert(DF_CONF_SLOT + DF_B - 1 <= 16, "probability slots inside draft_out");
#ifndef MPK_DF_SP
#define MPK_DF_SP 3
#endif
constexpr int DF_SP = MPK_DF_SP;  // workers per q head in the attention (context split)
static_assert(DF_NQ * DF_SP <= 248, "one worker per (q head, context split)");
constexpr int DF_KT_H = DF_HP / 128;        // 23
constexpr int DF_KT_FEAT = DF_FEATP / 128;  // 113
constexpr int DF_KT_Q = DF_QD / 128;        // 32
constexpr int DF_KT_FF = DF_FF / 128;       // 60
constexpr int DF_T_H = DF_H / 16;           // 180 row tiles
constexpr int DF_T_Q = DF_QD / 16;          // 256
constexpr int DF_T_KV = DF_KVD / 16;        // 32
constexpr int DF_T_QKV = DF_T_Q + 2 * DF_T_KV;
constexpr int DF_T_FF = DF_FF / 16;         // 480
constexpr int DF_T_V = DF_VOCAB / 16;       // 12568
constexpr int DF_F8 = 2048;                 // e4m3 fragment (16 rows x 128 K)
constexpr int DF_F4 = 1024;                 // MXFP4 fragment
constexpr int DF_SC4 = (DF_KT_H + 3) / 4 * 256;  // MXFP4 scale bytes per LM-head tile
static_assert(DF_T_H % 2 == 0 && DF_T_Q % 2 == 0 && DF_T_KV % 2 == 0 && DF_T_FF % 2 == 0 && DF_T_V % 2 == 0,
              "halves: even tile counts");
// g_aid_nc_rep (first 64 KB cleared per launch): 8 per-die lines, the global line, the LM-reduce counter, one
// counter per q head. g_aid_flag_rep (2 MiB cleared per launch): the release generation.
constexpr int DF_NC_INTS = 14336;
constexpr int DF_NC_HEAD = DF_NC_INTS + 10 * 64;
static_assert(DF_NC_HEAD + DF_NQ * 16 <= 16384, "inside the 64 KB cleared per launch");
constexpr int DF_FLAG_INTS = 516096;
constexpr int DF_FLAG_DIE = DF_FLAG_INTS + 64;  // per-die release slots, one 256-byte line each
static_assert(DF_FLAG_DIE + 8 * 64 <= 524288, "inside the 2 MiB flag replica");

// scratch (float offsets into g_df.vec), 64-float aligned regions
constexpr int df_al64(int x) { return (x + 63) & ~63; }
constexpr int DF_NR = DF_B < 4 ? 4 : DF_B;                    // rows of the per-worker partial arrays
constexpr int DF_V_FEAT = 0;                                  // [DF_B][DF_FEAT] target features
constexpr int DF_V_YV = df_al64(DF_V_FEAT + DF_B * DF_FEAT);  // [DF_NR][256] target argmax partials (bf16 bits)
constexpr int DF_V_YI = DF_V_YV + DF_NR * 256;                // [DF_NR][256] indices
constexpr int DF_V_HT = DF_V_YI + DF_NR * 256;                // [DF_B][DF_H] fc outputs
constexpr int DF_V_L0 = df_al64(DF_V_HT + DF_B * DF_H);       // per-layer regions
constexpr int DF_L_QR = 0;                                    // [DF_B][DF_QD] q_proj (no bias)
constexpr int DF_L_KR = DF_L_QR + DF_B * DF_QD;               // [DF_B][DF_KVD]
constexpr int DF_L_VR = DF_L_KR + DF_B * DF_KVD;              // [DF_B][DF_KVD]
constexpr int DF_APB = df_al64(DF_B * 68);                    // (head, split) partial: DF_B x (64 o, m, l)
constexpr int DF_L_AP = DF_L_VR + DF_B * DF_KVD;
constexpr int DF_L_O = df_al64(DF_L_AP + DF_NQ * DF_SP * DF_APB);  // [DF_B][DF_QD] attention output
constexpr int DF_L_X2 = DF_L_O + DF_B * DF_QD;                // [DF_B][DF_H] after attention
constexpr int DF_L_ACT = DF_L_X2 + DF_B * DF_H;               // [DF_B][DF_FF]
constexpr int DF_L_X = DF_L_ACT + DF_B * DF_FF;               // [DF_B][DF_H] layer output
constexpr int DF_L_SZ = df_al64(DF_L_X + DF_B * DF_H);
constexpr int DF_LPI = DF_NR * 256;                           // stride of the three LM partial arrays
constexpr int DF_V_LP = DF_V_L0 + DF_NL * DF_L_SZ;  // LM partials [DF_B - 1][256]: value, index (+ DF_LPI), exp sum (+ 2 DF_LPI)
constexpr int DF_V_FLOATS = DF_V_LP + 3 * DF_LPI;
static_assert(DF_V_FLOATS <= 4194304, "demo.py allocates 4M floats of scratch");

struct MpkDflash {
  unsigned char const *fc;                                  // e4m3 K-major tiles [180][113][2 KB]
  unsigned char const *wq, *wk, *wv, *wo, *wg, *wu, *wd;    // [8][tiles][KT][2 KB]
  float const *sfc, *swq, *swk, *swv, *swo, *swg, *swu, *swd;  // per-row scales ([8][rows])
  float const *bq, *bk, *bv, *bo;                           // biases [8][rows]
  unsigned short const *nq, *nk, *nin, *npost;              // bf16 [8][64], [8][64], [8][2880], [8][2880]
  unsigned short const *nhid, *nfin;                        // bf16 [2880]
  unsigned char const *wlm, *slm;                           // LM head MXFP4 tiles [12568][23][1 KB], scales
  unsigned short const *embed;                              // target embedding [201088][2880] bf16
  float const *rope;                                        // [maxpos][64]: cos x 32, sin x 32
  unsigned short *kc, *vc;                                  // context K/V [8][maxpos][8][64] bf16
  float *vec;
  long long *draft_out;  // [0..DF_B-2] drafts, [8] pass start step, [DF_CONF_SLOT + j] draft j's probability (f32 bits)
  unsigned short const *apo[DF_NTL];
  float const *ws[DF_NTL];
  long long const *tokens;
  int const *prompt_len;
  int ws_stride, maxpos, nworkers, enabled;
};
__constant__ MpkDflash g_df;

// g_spec_state is written once per pass by prepare_next_batch; the LM-head task's dependency check ran an
// agent-scope acquire on this XCD after its release. Not volatile (volatile = system-scope loads on gfx940+).
struct DfSt {
  int st0, n, s, kind;
};
__device__ __forceinline__ DfSt df_state() {
  unsigned long long const *const p = (unsigned long long const *)(g_spec_state);
  unsigned long long const a = p[0];
  unsigned long long const b = p[1];
  return DfSt{(int)(unsigned)a, (int)(unsigned)(a >> 32), (int)(unsigned)b, (int)(unsigned)(b >> 32)};
}

__device__ __forceinline__ float df_bf(unsigned b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ unsigned df_tobf(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float df_rbf(float f) { return df_bf(df_tobf(f)); }
// df_rbf of two values with one gfx950 v_cvt_pk_bf16_f32 (round to nearest even, as df_tobf for finite values)
__device__ __forceinline__ void df_rbf2(float &a, float &b) {
  unsigned p;
  asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(p) : "v"(a), "v"(b));
  a = __uint_as_float(p << 16);
  b = __uint_as_float(p & 0xFFFF0000u);
}
__device__ __forceinline__ unsigned df_ld_sys_u32(void const *p) {
  return __hip_atomic_load((unsigned const *)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float df_ld_sys_f32(float const *p) { return __uint_as_float(df_ld_sys_u32(p)); }
__device__ __forceinline__ void df_st_sys_u32(void *p, unsigned v) {
  __hip_atomic_store((unsigned *)p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void df_st2(int off, unsigned v) { df_st_sys_u32(g_df.vec + off, v); }
__device__ __forceinline__ void df_st2f(int off, float v) { df_st2(off, __float_as_uint(v)); }
// bf16 bits b as element i of the bf16 array at float offset off of the scratch
__device__ __forceinline__ void df_st2h(int off, int i, unsigned b) {
  unsigned short *const p = reinterpret_cast<unsigned short *>(g_df.vec + off) + i;
  asm volatile("global_store_short %0, %1, off sc0 sc1" ::"v"(p), "v"(b) : "memory");
}

__device__ __forceinline__ float df_wave_sum(float v) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}
__device__ __forceinline__ float df_wave_max(float v) {
#pragma unroll
  for (int o = 32; o >= 1; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
  return v;
}
// Wave reductions without LDS round trips (gfx950): permlane32 / permlane16 swaps then DPP row_shl steps toward
// lane 0 of each row of 16 -- the same 32/16/8/4/2/1 association as the __shfl_xor butterfly above, so lane 0's
// value is bit-identical (and every lane of the butterfly holds that same value). Valid in lane 0 only; all 64
// lanes must be active. Lane 0's tree reads only in-row lanes, so bound_ctrl zeros never reach it (also for max).
__device__ __forceinline__ float df_wsum_l0(float v) {
  float p = v;
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
               : [s] "+v"(v), [p] "+v"(p));
  return v;
}
__device__ __forceinline__ float df_wmax_l0(float v) {
  float p = v;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s], %[p]\n"
               "v_max_f32 %[s], %[s], %[p]\n"
               "v_mov_b32_e32 %[p], %[s]\n"
               "s_nop 1\n"
               "v_permlane16_swap_b32 %[s], %[p]\n"
               "v_max_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_max_f32_dpp %[s], %[s], %[s] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_max_f32_dpp %[s], %[s], %[s] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_max_f32_dpp %[s], %[s], %[s] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_max_f32_dpp %[s], %[s], %[s] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1"
               : [s] "+v"(v), [p] "+v"(p));
  return v;
}
__device__ __forceinline__ void df_wsum4_l0(float &a, float &b, float &c, float &d) {
  float pa = a, pb = b, pc = c, pd = d;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s0], %[p0]\n"
               "v_permlane32_swap_b32 %[s1], %[p1]\n"
               "v_permlane32_swap_b32 %[s2], %[p2]\n"
               "v_permlane32_swap_b32 %[s3], %[p3]\n"
               "v_add_f32 %[s0], %[s0], %[p0]\n"
               "v_add_f32 %[s1], %[s1], %[p1]\n"
               "v_add_f32 %[s2], %[s2], %[p2]\n"
               "v_add_f32 %[s3], %[s3], %[p3]\n"
               "v_mov_b32_e32 %[p0], %[s0]\n"
               "v_mov_b32_e32 %[p1], %[s1]\n"
               "v_mov_b32_e32 %[p2], %[s2]\n"
               "v_mov_b32_e32 %[p3], %[s3]\n"
               "v_permlane16_swap_b32 %[s0], %[p0]\n"
               "v_permlane16_swap_b32 %[s1], %[p1]\n"
               "v_permlane16_swap_b32 %[s2], %[p2]\n"
               "v_permlane16_swap_b32 %[s3], %[p3]\n"
               "v_add_f32 %[s0], %[s0], %[p0]\n"
               "v_add_f32 %[s1], %[s1], %[p1]\n"
               "v_add_f32 %[s2], %[s2], %[p2]\n"
               "v_add_f32 %[s3], %[s3], %[p3]\n"
               "v_add_f32_dpp %[s0], %[s0], %[s0] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s1], %[s1], %[s1] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s2], %[s2], %[s2] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s3], %[s3], %[s3] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s0], %[s0], %[s0] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s1], %[s1], %[s1] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s2], %[s2], %[s2] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s3], %[s3], %[s3] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s0], %[s0], %[s0] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s1], %[s1], %[s1] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s2], %[s2], %[s2] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s3], %[s3], %[s3] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s0], %[s0], %[s0] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s1], %[s1], %[s1] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s2], %[s2], %[s2] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32_dpp %[s3], %[s3], %[s3] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1"
               : [s0] "+v"(a), [s1] "+v"(b), [s2] "+v"(c), [s3] "+v"(d), [p0] "+v"(pa), [p1] "+v"(pb),
                 [p2] "+v"(pc), [p3] "+v"(pd));
}
__device__ __forceinline__ void df_wmax4_l0(float &a, float &b, float &c, float &d) {
  float pa = a, pb = b, pc = c, pd = d;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s0], %[p0]\n"
               "v_permlane32_swap_b32 %[s1], %[p1]\n"
               "v_permlane32_swap_b32 %[s2], %[p2]\n"
               "v_permlane32_swap_b32 %[s3], %[p3]\n"
               "v_max_f32 %[s0], %[s0], %[p0]\n"
               "v_max_f32 %[s1], %[s1], %[p1]\n"
               "v_max_f32 %[s2], %[s2], %[p2]\n"
               "v_max_f32 %[s3], %[s3], %[p3]\n"
               "v_mov_b32_e32 %[p0], %[s0]\n"
               "v_mov_b32_e32 %[p1], %[s1]\n"
               "v_mov_b32_e32 %[p2], %[s2]\n"
               "v_mov_b32_e32 %[p3], %[s3]\n"
               "v_permlane16_swap_b32 %[s0], %[p0]\n"
               "v_permlane16_swap_b32 %[s1], %[p1]\n"
               "v_permlane16_swap_b32 %[s2], %[p2]\n"
               "v_permlane16_swap_b32 %[s3], %[p3]\n"
               "v_max_f32 %[s0], %[s0], %[p0]\n"
               "v_max_f32 %[s1], %[s1], %[p1]\n"
               "v_max_f32 %[s2], %[s2], %[p2]\n"
               "v_max_f32 %[s3], %[s3], %[p3]\n"
               "v_max_f32_dpp %[s0], %[s0], %[s0] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s1], %[s1], %[s1] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s2], %[s2], %[s2] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s3], %[s3], %[s3] row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s0], %[s0], %[s0] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s1], %[s1], %[s1] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s2], %[s2], %[s2] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s3], %[s3], %[s3] row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s0], %[s0], %[s0] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s1], %[s1], %[s1] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s2], %[s2], %[s2] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s3], %[s3], %[s3] row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s0], %[s0], %[s0] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s1], %[s1], %[s1] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s2], %[s2], %[s2] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "v_max_f32_dpp %[s3], %[s3], %[s3] row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1"
               : [s0] "+v"(a), [s1] "+v"(b), [s2] "+v"(c), [s3] "+v"(d), [p0] "+v"(pa), [p1] "+v"(pb),
                 [p2] "+v"(pc), [p3] "+v"(pd));
}
// Lane i ^ 32's value (an exact move): after the swap the lower lanes hold the partner's value in p, the upper
// lanes in s. All 64 lanes must be active.
__device__ __forceinline__ float df_xor32(float v) {
  float s = v, p = v;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s], %[p]"
               : [s] "+v"(s), [p] "+v"(p));
  return (threadIdx.x & 32) ? s : p;
}
// The lane-0 sum in every lane (uniform).
__device__ __forceinline__ float df_wave_sum_u(float v) {
  return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(df_wsum_l0(v))));
}
// N sums / maxima over the block at once; red holds 4 * N floats; the result in every thread.
template <int N>
__device__ __forceinline__ void df_block_sum_n(float (&v)[N], float *red) {
#pragma unroll
  for (int k = 0; k + 4 <= N; k += 4) df_wsum4_l0(v[k], v[k + 1], v[k + 2], v[k + 3]);
#pragma unroll
  for (int k = N & ~3; k < N; ++k) v[k] = df_wsum_l0(v[k]);
  __syncthreads();
  if ((threadIdx.x & 63) == 0)
#pragma unroll
    for (int k = 0; k < N; ++k) red[(threadIdx.x >> 6) * N + k] = v[k];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < N; ++k) v[k] = (red[k] + red[N + k]) + (red[2 * N + k] + red[3 * N + k]);
  __syncthreads();
}
template <int N>
__device__ __forceinline__ void df_block_max_n(float (&v)[N], float *red) {
#pragma unroll
  for (int k = 0; k + 4 <= N; k += 4) df_wmax4_l0(v[k], v[k + 1], v[k + 2], v[k + 3]);
#pragma unroll
  for (int k = N & ~3; k < N; ++k) v[k] = df_wmax_l0(v[k]);
  __syncthreads();
  if ((threadIdx.x & 63) == 0)
#pragma unroll
    for (int k = 0; k < N; ++k) red[(threadIdx.x >> 6) * N + k] = v[k];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < N; ++k) v[k] = fmaxf(fmaxf(red[k], red[N + k]), fmaxf(red[2 * N + k], red[3 * N + k]));
  __syncthreads();
}
// Block argmax: every thread holds (v, key); returns in all threads the key of the max value, ties to the lower
// key (and the max value in *vmax). red holds 8 floats.
__device__ __forceinline__ int df_block_argmax(float v, int key, float *red, float *vmax = nullptr) {
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
  if (vmax) *vmax = bv;
  return bk;
}

// All nworkers workers: per-die NC counter, the die's last arrival bumps the global counter, the global last
// arrival publishes `gen` into both AID flag replicas; every worker polls the replica of its own physical AID.
__device__ __forceinline__ void df_barrier(int gen, int die) {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    int const per_die = g_df.nworkers / MPK_NUM_XCDS;
    int *const xc = g_aid_nc_rep[die >> 2] + DF_NC_INTS + die * 64;
    int const old = __hip_atomic_fetch_add(xc, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (old % per_die == per_die - 1) {
      df_st_sys_u32(g_aid_flag_rep[0] + DF_FLAG_DIE + die * 64, (unsigned)gen);
      df_st_sys_u32(g_aid_flag_rep[1] + DF_FLAG_DIE + die * 64, (unsigned)gen);
    }
    unsigned xcc;
    MPK_GET_XCC_ID32(xcc);
    int const *const fl = g_aid_flag_rep[(xcc & 7u) >> 2] + DF_FLAG_DIE;
    for (;;) {
      int mn = 0x7fffffff;
#pragma unroll
      for (int d = 0; d < MPK_NUM_XCDS; ++d) mn = min(mn, (int)df_ld_sys_u32(fl + d * 64));
      if (mn >= gen) break;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
}

// N items over W workers, sizes differing by at most one.
__device__ __forceinline__ void df_bsplit(int N, int wid, int W, int &t0, int &t1) {
  int const base = N / W, rem = N % W;
  t0 = wid * base + min(wid, rem);
  t1 = t0 + base + (wid < rem ? 1 : 0);
}

// MPK_DF_HALVES (default; -DMPK_DF_NO_HALVES restores the single-copy layout, demo.py follows the same flag): each
// large weight is one halves_malloc buffer whose region d (HALF bytes, a 2 MiB multiple, physically on AID d) holds
// the tiles device d's workers compute, [layer][tile][TB bytes]: region 0 tiles [0, T0) of every layer, region 1
// tiles [T0, T). Workers 0 .. W/2 - 1 (dies 0-3, device 0) take region 0's tiles, the others region 1's: every weight
// byte comes from the reader's own AID and both AIDs' HBM stream at once.
#ifndef MPK_DF_NO_HALVES
#define MPK_DF_HALVES 1
constexpr int DF_NDEV = 2;
#else
constexpr int DF_NDEV = 1;
#endif
template <int NLY, int T, int T0, int TB>
struct DfSpl {
  static constexpr int T1 = T - T0;
  static constexpr size_t HALF = (((size_t)NLY * (size_t)(T0 > T1 ? T0 : T1) * TB + (2u << 20) - 1) >> 21) << 21;
  __device__ __forceinline__ static unsigned char const *at(unsigned char const *base, int l, int t) {
#ifdef MPK_DF_HALVES
    return t < T0 ? base + ((size_t)l * T0 + t) * TB : base + HALF + ((size_t)l * T1 + (t - T0)) * TB;
#else
    return base + ((size_t)l * T + t) * TB;
#endif
  }
};
using DfFc = DfSpl<1, DF_T_H, DF_T_H / 2, DF_KT_FEAT * DF_F8>;
using DfWq = DfSpl<DF_NL, DF_T_Q, DF_T_Q / 2, DF_KT_H * DF_F8>;
using DfWkv = DfSpl<DF_NL, DF_T_KV, DF_T_KV / 2, DF_KT_H * DF_F8>;
using DfWo = DfSpl<DF_NL, DF_T_H, DF_T_H / 2, DF_KT_Q * DF_F8>;
using DfWgu = DfSpl<DF_NL, DF_T_FF, DF_T_FF / 2, DF_KT_H * DF_F8>;
using DfWd = DfSpl<DF_NL, DF_T_H, DF_T_H / 2, DF_KT_FF * DF_F8>;
using DfWlm = DfSpl<1, DF_T_V, DF_T_V / 2, DF_KT_H * DF_F4>;
using DfSlm = DfSpl<1, DF_T_V, DF_T_V / 2, DF_SC4>;
struct DfDev {
  int dev, lw, wd;  // device, worker index on it, workers on it
};
__device__ __forceinline__ DfDev df_dev(int wid, int W) {
#ifdef MPK_DF_HALVES
  int const Wd = W / 2, d = wid >= Wd ? 1 : 0;
  return DfDev{d, wid - d * Wd, d ? W - Wd : Wd};
#else
  return DfDev{0, wid, W};
#endif
}
// [t0, t1) of n tiles: device d's workers split [d * n / 2, (d + 1) * n / 2) (halves) or all n (single copy).
__device__ __forceinline__ void df_dsplit(int n, int wid, int W, int &t0, int &t1) {
  DfDev const dd = df_dev(wid, W);
  int const nd = n / DF_NDEV;
  df_bsplit(nd, dd.lw, dd.wd, t0, t1);
  t0 += dd.dev * nd;
  t1 += dd.dev * nd;
}

// n floats (n % 4 == 0) of scratch into LDS. All threads; ends in a barrier.
__device__ __forceinline__ void df_load_rep(float *dst, float const *rv, int off, int n) {
  constexpr int U = 8;
  float4 const *const src = reinterpret_cast<float4 const *>(rv + off);
  float4 *const d4 = reinterpret_cast<float4 *>(dst);
  int const nc = n / 4;
  for (int c0 = threadIdx.x; c0 < nc; c0 += blockDim.x * U) {
    float4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int const c = c0 + u * blockDim.x;
      v[u] = c < nc ? src[c] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int const c = c0 + u * blockDim.x;
      if (c < nc) d4[c] = v[u];
    }
  }
  __syncthreads();
}

// One bf16 row of DF_H (read-only weights) into LDS as f32. All threads; ends in a barrier.
__device__ __forceinline__ void df_load_bf_row(float *dst, unsigned short const *row) {
  constexpr int NC = DF_H / 8;
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
      d[0] = df_bf(v[u].x & 0xFFFFu); d[1] = df_bf(v[u].x >> 16);
      d[2] = df_bf(v[u].y & 0xFFFFu); d[3] = df_bf(v[u].y >> 16);
      d[4] = df_bf(v[u].z & 0xFFFFu); d[5] = df_bf(v[u].z >> 16);
      d[6] = df_bf(v[u].w & 0xFFFFu); d[7] = df_bf(v[u].w >> 16);
    }
  }
  __syncthreads();
}

// Qwen3RMSNorm of nr f32 LDS rows of DF_H in place: normalize in f32, round to bf16, times the bf16 weight.
__device__ __forceinline__ void df_rms_rows(float *x, int nr, unsigned short const *w, float *red) {
  constexpr int NI = (DF_H + 255) / 256;  // 256 threads
  int const tid = threadIdx.x & 255;
  float g[NI];
#pragma unroll
  for (int k = 0; k < NI; ++k) {
    int const i = tid + k * 256;
    g[k] = i < DF_H ? df_bf(w[i]) : 0.f;
  }
  float ss[DF_B];
#pragma unroll
  for (int r = 0; r < DF_B; ++r) {
    ss[r] = 0.f;
    if (r < nr) {
#pragma unroll
      for (int k = 0; k < NI; ++k) {
        int const i = tid + k * 256;
        if (i < DF_H) ss[r] += x[r * DF_H + i] * x[r * DF_H + i];
      }
    }
  }
  df_block_sum_n<DF_B>(ss, red);
#pragma unroll
  for (int r = 0; r < DF_B; ++r) {
    if (r < nr) {
      float const inv = rsqrtf(ss[r] / (float)DF_H + 1e-5f);
#pragma unroll
      for (int k = 0; k < NI; ++k) {
        int const i = tid + k * 256;
        if (i < DF_H) x[r * DF_H + i] = df_rbf(g[k] * df_rbf(x[r * DF_H + i] * inv));
      }
    }
  }
  __syncthreads();
}

// q_norm / k_norm (64-dim RMSNorm, lane = dim) then YaRN RoPE (dim d pairs with d +- 32) at position pos.
// Wave-wide: all 64 lanes.
__device__ __forceinline__ float df_qk_norm_rope(float v, unsigned short const *w64, int pos, int d) {
  float const ss = df_wave_sum_u(v * v);
  float const inv = rsqrtf(ss / (float)DF_HD + 1e-5f);
  float const vn = df_rbf(df_bf(w64[d]) * df_rbf(v * inv));
  float const pr = df_xor32(vn);
  float const rot = d < 32 ? -pr : pr;
  float const *const t = g_df.rope + (size_t)pos * 64;
  float const c = t[d & 31], sn = t[32 + (d & 31)];
  return df_rbf(df_rbf(vn * c) + df_rbf(rot * sn));
}
// The same with the weight (as float) and the rope cos / sin already loaded.
__device__ __forceinline__ float df_qk_norm_rope_p(float v, float gw, float c, float sn, int d) {
  float const ss = df_wave_sum_u(v * v);
  float const inv = rsqrtf(ss / (float)DF_HD + 1e-5f);
  float const vn = df_rbf(gw * df_rbf(v * inv));
  float const pr = df_xor32(vn);
  float const rot = d < 32 ? -pr : pr;
  return df_rbf(df_rbf(vn * c) + df_rbf(rot * sn));
}

typedef int __attribute__((ext_vector_type(4))) df_i32x4;
typedef int __attribute__((ext_vector_type(8))) df_i32x8;
typedef float __attribute__((ext_vector_type(4))) df_f32x4;
typedef short __attribute__((ext_vector_type(2))) df_fp8x2;

__device__ __forceinline__ float df_dpp_max(float v, int ctrl_sel) {
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
// R rows of x (row stride xs floats, K valid values each) -> q8 [r][KP] e4m3 + qs [r][KP / 128] E8M0 (one scale
// per 128 values: a 128-value block is one 16-lane DPP row). All threads; ends in a barrier.
template <int K, int KP>
__device__ __forceinline__ void df_q8act_r(float const *__restrict__ x, int xs, int R, unsigned char *__restrict__ q8,
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
      amax = df_dpp_max(amax, 0);
      amax = df_dpp_max(amax, 1);
      amax = df_dpp_max(amax, 2);
      amax = df_dpp_max(amax, 3);
      unsigned se = 0;
      if (amax != 0.f) {
        unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
        se = min(255u, ((u >> 23) & 0xffu) + ((u & 0x7fffffu) ? 1u : 0u));
      }
      float const sf = se ? __uint_as_float(se << 23) : 1.0f;
      if (l < tot) {
        df_fp8x2 p0 = {0, 0}, p1 = {0, 0};
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.x, v0.y, sf, false);
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.z, v0.w, sf, true);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.x, v1.y, sf, false);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.z, v1.w, sf, true);
        *reinterpret_cast<df_fp8x2 *>(q8 + r * KP + k) = p0;
        *reinterpret_cast<df_fp8x2 *>(q8 + r * KP + k + 4) = p1;
        if ((l & 15) == 0) qs[r * (KP / 128) + (k >> 7)] = (unsigned char)se;
      }
    }
  }
  __syncthreads();
}

// As df_q8act_r, but the R rows come straight from global memory (row stride xs floats): no LDS staging, B items
// (8 values each) per thread in flight per batch. All threads; ends in a barrier.
// One batch of df_q8act_g (items b0 .. b0 + 256 B - 1): ld issues the loads, fin quantizes into LDS (no barrier),
// so the caller can issue other loads between them.
template <int K, int KP, int B>
struct DfQ8B {
  static_assert(K % 8 == 0 && KP % 128 == 0, "8-value lanes, 128-value blocks");
  static constexpr int NL = KP / 8;
  float4 v[B][2];
  __device__ __forceinline__ void ld(float const *__restrict__ x, int xs, int R, int b0) {
    int const tid = threadIdx.x & 255, tot = R * NL;
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      bool const ok = l < tot && k < K;
      float4 const *const src = reinterpret_cast<float4 const *>(x + (size_t)r * xs + k);
      v[j][0] = ok ? src[0] : make_float4(0.f, 0.f, 0.f, 0.f);
      v[j][1] = ok ? src[1] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  __device__ __forceinline__ void fin(int R, int b0, unsigned char *__restrict__ q8, unsigned char *__restrict__ qs) {
    int const tid = threadIdx.x & 255, tot = R * NL;
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      float4 const v0 = v[j][0], v1 = v[j][1];
      float amax = fmaxf(fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w))),
                         fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
      amax = df_dpp_max(amax, 0);
      amax = df_dpp_max(amax, 1);
      amax = df_dpp_max(amax, 2);
      amax = df_dpp_max(amax, 3);
      unsigned se = 0;
      if (amax != 0.f) {
        unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
        se = min(255u, ((u >> 23) & 0xffu) + ((u & 0x7fffffu) ? 1u : 0u));
      }
      float const sf = se ? __uint_as_float(se << 23) : 1.0f;
      if (l < tot) {
        df_fp8x2 p0 = {0, 0}, p1 = {0, 0};
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.x, v0.y, sf, false);
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.z, v0.w, sf, true);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.x, v1.y, sf, false);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.z, v1.w, sf, true);
        *reinterpret_cast<df_fp8x2 *>(q8 + r * KP + k) = p0;
        *reinterpret_cast<df_fp8x2 *>(q8 + r * KP + k + 4) = p1;
        if ((l & 15) == 0) qs[r * (KP / 128) + (k >> 7)] = (unsigned char)se;
      }
    }
  }
};
// As DfQ8B, but the R rows are bf16 in global memory (row stride xs elements): one 16-byte load per 8 values.
template <int K, int KP, int B>
struct DfQ8Bh {
  static_assert(K % 8 == 0 && KP % 128 == 0, "8-value lanes, 128-value blocks");
  static constexpr int NL = KP / 8;
  uint4 u[B];
  __device__ __forceinline__ void ld(unsigned short const *__restrict__ x, int xs, int R, int b0) {
    int const tid = threadIdx.x & 255, tot = R * NL;
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      bool const ok = l < tot && k < K;
      u[j] = ok ? *reinterpret_cast<uint4 const *>(x + (size_t)r * xs + k) : make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __device__ __forceinline__ void fin(int R, int b0, unsigned char *__restrict__ q8, unsigned char *__restrict__ qs) {
    int const tid = threadIdx.x & 255, tot = R * NL;
#pragma unroll
    for (int j = 0; j < B; ++j) {
      int const l = b0 + j * 256 + tid;
      int const r = l / NL, k = 8 * (l - r * NL);
      float4 const v0 = make_float4(df_bf(u[j].x & 0xFFFFu), df_bf(u[j].x >> 16), df_bf(u[j].y & 0xFFFFu),
                                    df_bf(u[j].y >> 16));
      float4 const v1 = make_float4(df_bf(u[j].z & 0xFFFFu), df_bf(u[j].z >> 16), df_bf(u[j].w & 0xFFFFu),
                                    df_bf(u[j].w >> 16));
      float amax = fmaxf(fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w))),
                         fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
      amax = df_dpp_max(amax, 0);
      amax = df_dpp_max(amax, 1);
      amax = df_dpp_max(amax, 2);
      amax = df_dpp_max(amax, 3);
      unsigned se = 0;
      if (amax != 0.f) {
        unsigned const uu = __float_as_uint(amax * (1.0f / 448.0f));
        se = min(255u, ((uu >> 23) & 0xffu) + ((uu & 0x7fffffu) ? 1u : 0u));
      }
      float const sf = se ? __uint_as_float(se << 23) : 1.0f;
      if (l < tot) {
        df_fp8x2 p0 = {0, 0}, p1 = {0, 0};
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.x, v0.y, sf, false);
        p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v0.z, v0.w, sf, true);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.x, v1.y, sf, false);
        p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v1.z, v1.w, sf, true);
        *reinterpret_cast<df_fp8x2 *>(q8 + r * KP + k) = p0;
        *reinterpret_cast<df_fp8x2 *>(q8 + r * KP + k + 4) = p1;
        if ((l & 15) == 0) qs[r * (KP / 128) + (k >> 7)] = (unsigned char)se;
      }
    }
  }
};
template <int K, int KP, int B>
__device__ __forceinline__ void df_q8act_g(float const *__restrict__ x, int xs, int R, unsigned char *__restrict__ q8,
                                           unsigned char *__restrict__ qs) {
#pragma unroll 1
  for (int b0 = 0; b0 < R * (KP / 8); b0 += 256 * B) {
    DfQ8B<K, KP, B> q;
    q.ld(x, xs, R, b0);
    q.fin(R, b0, q8, qs);
  }
  __syncthreads();
}

// R (<= DF_B) f32 rows of DF_H in global memory (row stride DF_H) -> Qwen3RMSNorm with the bf16 weight w -> e4m3
// q8 [r][DF_HP] + E8M0 qs [r][DF_KT_H] in LDS, in one pass through registers (no LDS staging of the f32 rows): lane
// item l holds values 8 (l % NLH) .. + 8 of row l / NLH, so a 16-lane DPP row is one 128-value scale block (as in
// df_q8act_r). Same per-element arithmetic as df_rms_rows + df_q8act_r; only the sum-of-squares order differs.
// All threads; ends in a barrier.
// ld issues the loads, fin (all threads, ends in a barrier) normalizes and quantizes: the caller may issue other
// loads between the two.
template <bool H>  // H: the rows are bf16 in global memory (the values every writer rounded to bf16 anyway)
struct DfRqT {
  static constexpr int NLH = DF_HP / 8;                // 368 lane items per row, 360 carry data
  static constexpr int NJ = (DF_B * NLH + 255) / 256;  // items per thread
  float4 v[H ? 1 : NJ][2];
  uint4 vh[H ? NJ : 1];
  uint4 gw[NJ];
  __device__ __forceinline__ float4 va(int j) const {
    if constexpr (H)
      return make_float4(df_bf(vh[j].x & 0xFFFFu), df_bf(vh[j].x >> 16), df_bf(vh[j].y & 0xFFFFu), df_bf(vh[j].y >> 16));
    else
      return v[j][0];
  }
  __device__ __forceinline__ float4 vb(int j) const {
    if constexpr (H)
      return make_float4(df_bf(vh[j].z & 0xFFFFu), df_bf(vh[j].z >> 16), df_bf(vh[j].w & 0xFFFFu), df_bf(vh[j].w >> 16));
    else
      return v[j][1];
  }
  __device__ __forceinline__ void ld(unsigned short const *__restrict__ x, int R, unsigned short const *__restrict__ w) {
    static_assert(H, "bf16 rows");
    int const tid = threadIdx.x & 255, tot = R * NLH;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int const l = j * 256 + tid, r = l / NLH, k = 8 * (l - r * NLH);
      bool const ok = l < tot && k < DF_H;
      vh[j] = ok ? *reinterpret_cast<uint4 const *>(x + r * DF_H + k) : make_uint4(0u, 0u, 0u, 0u);
      gw[j] = ok ? *reinterpret_cast<uint4 const *>(w + k) : make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __device__ __forceinline__ void ld(float const *__restrict__ x, int R, unsigned short const *__restrict__ w) {
    static_assert(!H, "f32 rows");
    int const tid = threadIdx.x & 255, tot = R * NLH;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      int const l = j * 256 + tid, r = l / NLH, k = 8 * (l - r * NLH);
      bool const ok = l < tot && k < DF_H;
      float4 const *const src = reinterpret_cast<float4 const *>(x + r * DF_H + k);
      v[j][0] = ok ? src[0] : make_float4(0.f, 0.f, 0.f, 0.f);
      v[j][1] = ok ? src[1] : make_float4(0.f, 0.f, 0.f, 0.f);
      gw[j] = ok ? *reinterpret_cast<uint4 const *>(w + k) : make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __device__ __forceinline__ void fin(int R, unsigned char *__restrict__ q8, unsigned char *__restrict__ qs,
                                      float *red, unsigned long long *fts = nullptr);
};
using DfRq = DfRqT<false>;
template <bool H>
__device__ __forceinline__ void DfRqT<H>::fin(int R, unsigned char *__restrict__ q8, unsigned char *__restrict__ qs,
                                              float *red, unsigned long long *fts) {
  int const tid = threadIdx.x & 255, tot = R * NLH;
  float ss[DF_B];
#pragma unroll
  for (int r = 0; r < DF_B; ++r) ss[r] = 0.f;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    int const r = (j * 256 + tid) / NLH;
    float4 const a = va(j), b = vb(j);
    float const s8 = ((a.x * a.x + a.y * a.y) + (a.z * a.z + a.w * a.w)) + ((b.x * b.x + b.y * b.y) + (b.z * b.z + b.w * b.w));
#pragma unroll
    for (int rr = 0; rr < DF_B; ++rr) ss[rr] += r == rr ? s8 : 0.f;
  }
  if (fts && tid == 0) {
    asm volatile("" ::"v"(ss[0]), "v"(ss[1]), "v"(ss[2]), "v"(ss[3]));
    fts[0] = __builtin_amdgcn_s_memrealtime();
  }
  df_block_sum_n<DF_B>(ss, red);
  if (fts && tid == 0) {
    asm volatile("" ::"v"(ss[0]), "v"(ss[1]), "v"(ss[2]), "v"(ss[3]));
    fts[1] = __builtin_amdgcn_s_memrealtime();
  }
  float inv[DF_B];
#pragma unroll
  for (int r = 0; r < DF_B; ++r) inv[r] = rsqrtf(ss[r] / (float)DF_H + 1e-5f);
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    int const l = j * 256 + tid, r = l / NLH, k = 8 * (l - r * NLH);
    float iv = inv[0];
#pragma unroll
    for (int rr = 1; rr < DF_B; ++rr) iv = r == rr ? inv[rr] : iv;
    float4 const xa = va(j), xb = vb(j);
    float const xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    unsigned const gu[4] = {gw[j].x, gw[j].y, gw[j].z, gw[j].w};
    float y[8];
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e += 2) {  // y = rbf(g * rbf(x * iv)), two values per bf16 conversion
      float t0 = xv[e] * iv, t1 = xv[e + 1] * iv;
      df_rbf2(t0, t1);
      float u0 = df_bf(gu[e >> 1] & 0xFFFFu) * t0, u1 = df_bf(gu[e >> 1] >> 16) * t1;
      df_rbf2(u0, u1);
      y[e] = u0;
      y[e + 1] = u1;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(y[e]));
    amax = df_dpp_max(amax, 0);
    amax = df_dpp_max(amax, 1);
    amax = df_dpp_max(amax, 2);
    amax = df_dpp_max(amax, 3);
    unsigned se = 0;
    if (amax != 0.f) {
      unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
      se = min(255u, ((u >> 23) & 0xffu) + ((u & 0x7fffffu) ? 1u : 0u));
    }
    float const sf = se ? __uint_as_float(se << 23) : 1.0f;
    if (l < tot) {
      df_fp8x2 p0 = {0, 0}, p1 = {0, 0};
      p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, y[0], y[1], sf, false);
      p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, y[2], y[3], sf, true);
      p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, y[4], y[5], sf, false);
      p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, y[6], y[7], sf, true);
      *reinterpret_cast<df_fp8x2 *>(q8 + r * DF_HP + k) = p0;
      *reinterpret_cast<df_fp8x2 *>(q8 + r * DF_HP + k + 4) = p1;
      if ((l & 15) == 0) qs[r * DF_KT_H + (k >> 7)] = (unsigned char)se;
    }
  }
  __syncthreads();
}
__device__ __forceinline__ void df_rmsq8_g(float const *__restrict__ x, int R, unsigned short const *__restrict__ w,
                                           unsigned char *__restrict__ q8, unsigned char *__restrict__ qs, float *red) {
  DfRq a;
  a.ld(x, R, w);
  a.fin(R, q8, qs, red);
}
// df_rmsq8_g on bf16 rows (row stride DF_H elements)
__device__ __forceinline__ void df_rmsq8_gh(unsigned short const *__restrict__ x, int R,
                                            unsigned short const *__restrict__ w, unsigned char *__restrict__ q8,
                                            unsigned char *__restrict__ qs, float *red) {
  DfRqT<true> a;
  a.ld(x, R, w);
  a.fin(R, q8, qs, red);
}

// df_rmsq8_g for the DF_B block rows taken from the bf16 embedding table (row tok[r]). All threads; ends in a barrier.
__device__ __forceinline__ void df_rmsq8_emb(long long const *tok, unsigned short const *__restrict__ w,
                                             unsigned char *__restrict__ q8, unsigned char *__restrict__ qs, float *red) {
  constexpr int NLH = DF_HP / 8;
  constexpr int NJ = (DF_B * NLH + 255) / 256;
  int const tid = threadIdx.x & 255, tot = DF_B * NLH;
  uint4 ev[NJ];
  uint4 gw[NJ];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    int const l = j * 256 + tid, r = l / NLH, k = 8 * (l - r * NLH);
    bool const ok = l < tot && k < DF_H;
    ev[j] = ok ? *reinterpret_cast<uint4 const *>(g_df.embed + (size_t)tok[r] * DF_H + k) : make_uint4(0u, 0u, 0u, 0u);
    gw[j] = ok ? *reinterpret_cast<uint4 const *>(w + k) : make_uint4(0u, 0u, 0u, 0u);
  }
  float ss[DF_B];
#pragma unroll
  for (int r = 0; r < DF_B; ++r) ss[r] = 0.f;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    int const r = (j * 256 + tid) / NLH;
    unsigned const eu[4] = {ev[j].x, ev[j].y, ev[j].z, ev[j].w};
    float s8 = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float const xe = df_bf((e & 1) ? (eu[e >> 1] >> 16) : (eu[e >> 1] & 0xFFFFu));
      s8 += xe * xe;
    }
#pragma unroll
    for (int rr = 0; rr < DF_B; ++rr) ss[rr] += r == rr ? s8 : 0.f;
  }
  df_block_sum_n<DF_B>(ss, red);
  float inv[DF_B];
#pragma unroll
  for (int r = 0; r < DF_B; ++r) inv[r] = rsqrtf(ss[r] / (float)DF_H + 1e-5f);
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    int const l = j * 256 + tid, r = l / NLH, k = 8 * (l - r * NLH);
    float iv = inv[0];
#pragma unroll
    for (int rr = 1; rr < DF_B; ++rr) iv = r == rr ? inv[rr] : iv;
    unsigned const eu[4] = {ev[j].x, ev[j].y, ev[j].z, ev[j].w};
    unsigned const gu[4] = {gw[j].x, gw[j].y, gw[j].z, gw[j].w};
    float y[8];
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float const xe = df_bf((e & 1) ? (eu[e >> 1] >> 16) : (eu[e >> 1] & 0xFFFFu));
      float const ge = df_bf((e & 1) ? (gu[e >> 1] >> 16) : (gu[e >> 1] & 0xFFFFu));
      y[e] = df_rbf(ge * df_rbf(xe * iv));
      amax = fmaxf(amax, fabsf(y[e]));
    }
    amax = df_dpp_max(amax, 0);
    amax = df_dpp_max(amax, 1);
    amax = df_dpp_max(amax, 2);
    amax = df_dpp_max(amax, 3);
    unsigned se = 0;
    if (amax != 0.f) {
      unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
      se = min(255u, ((u >> 23) & 0xffu) + ((u & 0x7fffffu) ? 1u : 0u));
    }
    float const sf = se ? __uint_as_float(se << 23) : 1.0f;
    if (l < tot) {
      df_fp8x2 p0 = {0, 0}, p1 = {0, 0};
      p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, y[0], y[1], sf, false);
      p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, y[2], y[3], sf, true);
      p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, y[4], y[5], sf, false);
      p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, y[6], y[7], sf, true);
      *reinterpret_cast<df_fp8x2 *>(q8 + r * DF_HP + k) = p0;
      *reinterpret_cast<df_fp8x2 *>(q8 + r * DF_HP + k + 4) = p1;
      if ((l & 15) == 0) qs[r * DF_KT_H + (k >> 7)] = (unsigned char)se;
    }
  }
  __syncthreads();
}

// e4m3-weight MFMA over NT 16-row tiles: wr[t] = the tile's first K tile (fragment kt at kt * 2048: half 0 then
// half 1, 1 KB each in lane order). The activation rows are columns: lane column n reads row min(n, R - 1) of q8
// (row stride qst bytes) and its scale from qs (row stride qsst). Weight scale 1.0 (the caller applies the
// per-row fp8 scale). NKT K tiles in groups of D, branch-free: the body computes group gi and loads gi + 1.
// Result acc[t][i] in lane l: row (l >> 4) * 4 + i, column l & 15.
// The first 4 K tiles of a wave's df_m8gemv share, loaded ahead of time (df_pre8) so a round's weights can be in
// flight while the activations are prepared.
struct DfPre8 {
  df_i32x4 lo[4][2], hi[4][2];
};
template <int NKT, int NT, int D, bool PRE = false>
__device__ __forceinline__ void df_m8gemv(unsigned char const *const (&wr)[2], unsigned char const *q8, int qst,
                                          unsigned char const *qs, int qsst, int R, df_f32x4 (&acc)[2],
                                          DfPre8 const *pre = nullptr) {
  static_assert(NKT >= D && NT <= 2, "at least one group, at most 2 tiles");
  constexpr int G = NKT / D, TL = NKT % D;
  int const lane = threadIdx.x & 63, g = lane >> 4, rs = min(lane & 15, R - 1);
  unsigned char const *const qb = q8 + rs * qst;
  unsigned char const *const qsb = qs + rs * qsst;
  int sa1;  // weight scale E8M0 127 = 1.0 in a VGPR (an SGPR scale operand reads as 0)
  asm volatile("v_mov_b32 %0, 127\n\ts_nop 2" : "=v"(sa1));
  df_i32x4 lo[D][NT], hi[D][NT];
  auto ld = [&](int kt, int d, int t) {
    lo[d][t] = *reinterpret_cast<df_i32x4 const *>(wr[t] + kt * 2048 + lane * 16);
    hi[d][t] = *reinterpret_cast<df_i32x4 const *>(wr[t] + kt * 2048 + 1024 + lane * 16);
  };
  auto mma = [&](int kt, int d) {
    df_i32x4 const blo = *reinterpret_cast<df_i32x4 const *>(qb + kt * 128 + g * 16);
    df_i32x4 const bhi = *reinterpret_cast<df_i32x4 const *>(qb + kt * 128 + 64 + g * 16);
    df_i32x8 const b = {blo[0], blo[1], blo[2], blo[3], bhi[0], bhi[1], bhi[2], bhi[3]};
    int sb;
    asm volatile("v_mov_b32 %0, %1\n\ts_nop 2" : "=v"(sb) : "v"((int)qsb[kt]));
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      df_i32x8 const av = {lo[d][t][0], lo[d][t][1], lo[d][t][2], lo[d][t][3],
                           hi[d][t][0], hi[d][t][1], hi[d][t][2], hi[d][t][3]};
      acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, b, acc[t], 0, 0, 0, sa1, 0, sb);
    }
  };
#pragma unroll
  for (int t = 0; t < NT; ++t) acc[t] = df_f32x4{0.f, 0.f, 0.f, 0.f};
  if constexpr (PRE) {
    static_assert(D == 4, "DfPre8 holds 4 K tiles");
#pragma unroll
    for (int d = 0; d < D; ++d)
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        lo[d][t] = pre->lo[d][t];
        hi[d][t] = pre->hi[d][t];
      }
  } else {
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
// Load ahead the first 4 K tiles of this wave's share of a df_round8<KT> round over NT tiles. Branch-free, so the
// compiler counts vmcnt for loads issued before it (the activations) instead of draining.
template <int KT, int NT>
__device__ __forceinline__ void df_pre8(unsigned char const *tb0, unsigned char const *tb1, DfPre8 &p) {
  constexpr int KQ0 = KT / 4, KR = KT % 4;
  static_assert(KQ0 >= 4, "4 K tiles per wave at least");
  int const wave = (threadIdx.x & 255) >> 6, lane = threadIdx.x & 63;
  int const ka = wave * KQ0 + min(wave, KR);
  unsigned char const *const w0 = tb0 + (size_t)ka * DF_F8;
  unsigned char const *const w1 = tb1 + (size_t)ka * DF_F8;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    p.lo[d][0] = *reinterpret_cast<df_i32x4 const *>(w0 + d * 2048 + lane * 16);
    p.hi[d][0] = *reinterpret_cast<df_i32x4 const *>(w0 + d * 2048 + 1024 + lane * 16);
  }
  if constexpr (NT > 1)
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      p.lo[d][1] = *reinterpret_cast<df_i32x4 const *>(w1 + d * 2048 + lane * 16);
      p.hi[d][1] = *reinterpret_cast<df_i32x4 const *>(w1 + d * 2048 + 1024 + lane * 16);
    }
}

// MXFP4-weight MFMA over NT 16-row tiles (K-major: wr[t] lane l's 16 bytes of K tile kt at (kt * 64 + l) * 16,
// sr[t] its scales at ((kt >> 2) * 64 + l) * 4 + (kt & 3)) against R e4m3 activation columns as in df_m8gemv.
template <int NKT, int NT, int D>
__device__ __forceinline__ void df_m4gemv(unsigned char const *const (&wr)[3], unsigned char const *const (&sr)[3],
                                          unsigned char const *q8, int qst, unsigned char const *qs, int qsst, int R,
                                          df_f32x4 (&acc)[3]) {
  static_assert(NKT >= D && NT <= 3, "at least one group, at most 3 tiles");
  constexpr int G = NKT / D, TL = NKT % D;
  int const lane = threadIdx.x & 63, g = lane >> 4, rs = min(lane & 15, R - 1);
  unsigned char const *const qb = q8 + rs * qst;
  unsigned char const *const qsb = qs + rs * qsst;
  df_i32x4 a[D][NT];
  int sa[D][NT];
  auto ld = [&](int kt, int d, int t) {
    a[d][t] = *reinterpret_cast<df_i32x4 const *>(wr[t] + (kt * 64 + lane) * 16);
    sa[d][t] = sr[t][((kt >> 2) * 64 + lane) * 4 + (kt & 3)];
  };
  auto mma = [&](int kt, int d) {
    df_i32x4 const blo = *reinterpret_cast<df_i32x4 const *>(qb + kt * 128 + g * 16);
    df_i32x4 const bhi = *reinterpret_cast<df_i32x4 const *>(qb + kt * 128 + 64 + g * 16);
    df_i32x8 const b = {blo[0], blo[1], blo[2], blo[3], bhi[0], bhi[1], bhi[2], bhi[3]};
    int sb;
    asm volatile("v_mov_b32 %0, %1\n\ts_nop 2" : "=v"(sb) : "v"((int)qsb[kt]));
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      df_i32x8 const av = {a[d][t][0], a[d][t][1], a[d][t][2], a[d][t][3], 0, 0, 0, 0};
      acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, b, acc[t], 4, 0, 0, sa[d][t], 0, sb);
    }
  };
#pragma unroll
  for (int t = 0; t < NT; ++t) acc[t] = df_f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int t = 0; t < NT; ++t) ld(d, d, t);
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

// One round of up to 2 tiles (tb0, tb1 = each tile's first fragment) of an e4m3 matrix with KT K tiles: wave w
// sums its K range (waves < KT % 4 take one more K tile); the partials go to s_red [4 waves][2][16][DF_RC].
// Ends in a barrier; the caller sums the 4 waves with df_sum4 and syncs again.
template <int KT, bool PRE = false>
__device__ __forceinline__ void df_round8(unsigned char const *tb0, unsigned char const *tb1, int nt,
                                          unsigned char const *q8, unsigned char const *qs, int R, float *s_red,
                                          DfPre8 const *pre = nullptr) {
  constexpr int KQ0 = KT / 4, KR = KT % 4;
  int const tid = threadIdx.x, wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
  int const ka = wave * KQ0 + min(wave, KR);
  unsigned char const *wr[2] = {tb0 + (size_t)ka * DF_F8, tb1 + (size_t)ka * DF_F8};
  df_f32x4 acc[2];
  unsigned char const *const qw = q8 + ka * 128;
  unsigned char const *const qsw = qs + ka;
  if (wave < KR) {
    if (nt > 1) df_m8gemv<KQ0 + 1, 2, 4, PRE>(wr, qw, KT * 128, qsw, KT, R, acc, pre);
    else df_m8gemv<KQ0 + 1, 1, 4, PRE>(wr, qw, KT * 128, qsw, KT, R, acc, pre);
  } else {
    if (nt > 1) df_m8gemv<KQ0, 2, 4, PRE>(wr, qw, KT * 128, qsw, KT, R, acc, pre);
    else df_m8gemv<KQ0, 1, 4, PRE>(wr, qw, KT * 128, qsw, KT, R, acc, pre);
  }
  if (col < DF_RC) {
#pragma unroll
    for (int j = 0; j < 2; ++j)
      if (j < nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) s_red[((wave * 2 + j) * 16 + gq * 4 + i) * DF_RC + col] = acc[j][i];
  }
  __syncthreads();
}
__device__ __forceinline__ float df_sum4(float const *s_red, int j, int row, int c) {
  return (s_red[((0 * 2 + j) * 16 + row) * DF_RC + c] + s_red[((1 * 2 + j) * 16 + row) * DF_RC + c]) +
         (s_red[((2 * 2 + j) * 16 + row) * DF_RC + c] + s_red[((3 * 2 + j) * 16 + row) * DF_RC + c]);
}

// Block q / k / v of draft layer l: tiles [t0, t1) of the concatenated (q 256, k 32, v 32) row tiles; outputs
// (scaled, no bias, bf16) to the layer's QR / KR / VR regions.
// Device dev's q/k/v tile u (0 .. DF_T_QKV / DF_NDEV - 1) -> T in the concatenated (q 256, k 32, v 32) tiles:
// device d owns q tiles [128 d, 128 d + 128), k and v tiles [16 d, 16 d + 16) (halves), or T = u (single copy).
__device__ __forceinline__ int df_qkv_T(int dev, int u) {
#ifdef MPK_DF_HALVES
  constexpr int Q2 = DF_T_Q / 2, K2 = DF_T_KV / 2;
  return u < Q2 ? dev * Q2 + u
                : (u < Q2 + K2 ? DF_T_Q + dev * K2 + (u - Q2) : DF_T_Q + DF_T_KV + dev * K2 + (u - Q2 - K2));
#else
  (void)dev;
  return u;
#endif
}
__device__ __forceinline__ unsigned char const *df_qkv_tp(int l, int T) {
  if (T < DF_T_Q) return DfWq::at(g_df.wq, l, T);
  if (T < DF_T_Q + DF_T_KV) return DfWkv::at(g_df.wk, l, T - DF_T_Q);
  return DfWkv::at(g_df.wv, l, T - DF_T_Q - DF_T_KV);
}
template <bool PRE>
__device__ __forceinline__ void df_qkv_tiles(int l, int dev, int t0, int t1, unsigned char const *q8,
                                             unsigned char const *qs, float *s_red, int L, DfPre8 const *pre) {
  auto tp = [&](int T) -> unsigned char const * { return df_qkv_tp(l, T); };
  int const tid = threadIdx.x;
  for (int t = t0; t < t1; t += 2) {
    int const nt = min(2, t1 - t);
    int const Ta = df_qkv_T(dev, t), Tb = df_qkv_T(dev, t + nt - 1);
    float sc_pf = 0.f;
    if (tid < nt * DF_TPT) {
      int const T = (tid >> (4 + DF_RCS)) ? Tb : Ta, o = T * 16 + ((tid >> DF_RCS) & 15);
      sc_pf = T < DF_T_Q ? g_df.swq[l * DF_QD + o]
                         : (T < DF_T_Q + DF_T_KV ? g_df.swk[l * DF_KVD + o - DF_QD]
                                                 : g_df.swv[l * DF_KVD + o - DF_QD - DF_KVD]);
    }
    if (PRE && t == t0) df_round8<DF_KT_H, true>(tp(Ta), tp(Tb), nt, q8, qs, DF_B, s_red, pre);
    else df_round8<DF_KT_H>(tp(Ta), tp(Tb), nt, q8, qs, DF_B, s_red);
    if (tid < nt * DF_TPT) {
      int const j = tid >> (4 + DF_RCS), row = (tid >> DF_RCS) & 15, c = tid & (DF_RC - 1);
      if (c < DF_B) {
        int const T = j ? Tb : Ta, o = T * 16 + row;
        float const v = df_sum4(s_red, j, row, c);
        if (T < DF_T_Q) {
          df_st2f(L + DF_L_QR + c * DF_QD + o, df_rbf(v * sc_pf));
        } else if (T < DF_T_Q + DF_T_KV) {
          int const ok = o - DF_QD;
          df_st2f(L + DF_L_KR + c * DF_KVD + ok, df_rbf(v * sc_pf));
        } else {
          int const ov = o - DF_QD - DF_KVD;
          df_st2f(L + DF_L_VR + c * DF_KVD + ov, df_rbf(v * sc_pf));
        }
      }
    }
    __syncthreads();
  }
}

#ifdef MPK_DF_TS
#define DF_ATS(k)                                                   \
  do {                                                              \
    if (ats && tid == 0) ats[k] = __builtin_amdgcn_s_memrealtime(); \
  } while (0)
#else
#define DF_ATS(k) \
  do {            \
  } while (0)
#endif
// Attention of q head h over context split sp (context positions [0, P) cut in DF_SP parts; the last split also
// takes the DF_B block keys), all DF_B block rows; the head's last split to arrive merges the partials into O.
__device__ __forceinline__ void df_attn(int l, int L, int wid, int P, float *red, float *lds,
                                        unsigned long long *ats = nullptr) {
  int const tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
  // Context splits per q head: one while the context and the block keys fit one pass of the 256 threads (no
  // partials, no merge; workers 0 .. 63, on the device holding the K/V cache), DF_SP otherwise.
  int const nsp = P + DF_B <= 256 ? 1 : DF_SP;
  if (wid >= DF_NQ * nsp) return;
  int const h = wid / nsp, sp = wid - h * nsp, g = h / DF_GQ;
  float const *const rv = g_df.vec;
  float *const qv = lds;                // [DF_B][64]
  float *const kb = qv + DF_B * DF_HD;  // block keys
  float *const vb = kb + DF_B * DF_HD;  // block values
  float *const sc = vb + DF_B * DF_HD;  // [DF_B][cs] scores
  int const len = (P + nsp - 1) / nsp;
  int const c0 = min(P, sp * len), c1 = min(P, c0 + len), ncx = c1 - c0;
  int const nb = sp == nsp - 1 ? DF_B : 0;
  int const nk = ncx + nb;
  int const cs = (nk + 3) & ~3;
  float *const pv = sc + DF_B * cs;     // [16][DF_B][64] partial V sums
  for (int b = wave; b < DF_B; b += 4) {  // one block row per wave and round
    int const d = lane;
    float const q = df_rbf(rv[L + DF_L_QR + b * DF_QD + h * DF_HD + d] + g_df.bq[l * DF_QD + h * DF_HD + d]);
    qv[b * DF_HD + d] = df_qk_norm_rope(q, g_df.nq + l * DF_HD, P + b, d);
    if (nb) {
      int const ko = g * DF_HD + d;
      float const k = df_rbf(rv[L + DF_L_KR + b * DF_KVD + ko] + g_df.bk[l * DF_KVD + ko]);
      kb[b * DF_HD + d] = df_qk_norm_rope(k, g_df.nk + l * DF_HD, P + b, d);
      vb[b * DF_HD + d] = df_rbf(rv[L + DF_L_VR + b * DF_KVD + ko] + g_df.bv[l * DF_KVD + ko]);
    }
  }
  __syncthreads();
  DF_ATS(0);
  size_t const rs = (size_t)DF_NKV * DF_HD;  // cache row stride (one position)
  unsigned short const *const kc = g_df.kc + (size_t)l * g_df.maxpos * rs + g * DF_HD;
  unsigned short const *const vc = g_df.vc + (size_t)l * g_df.maxpos * rs + g * DF_HD;
  float mx[DF_B], sm[DF_B];
#pragma unroll
  for (int b = 0; b < DF_B; ++b) mx[b] = -3.0e38f;
  for (int t = tid; t < nk; t += blockDim.x) {
    float d4[DF_B];
#pragma unroll
    for (int b = 0; b < DF_B; ++b) d4[b] = 0.f;
    if (t < ncx) {
      uint4 const *const kr = reinterpret_cast<uint4 const *>(kc + (size_t)(c0 + t) * rs);
      uint4 kv[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) kv[c] = kr[c];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        unsigned const w4[4] = {kv[c].x, kv[c].y, kv[c].z, kv[c].w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float const kf = df_bf((e & 1) ? (w4[e >> 1] >> 16) : (w4[e >> 1] & 0xFFFFu));
#pragma unroll
          for (int b = 0; b < DF_B; ++b) d4[b] += kf * qv[b * DF_HD + c * 8 + e];
        }
      }
    } else {
      float const *const kk = kb + (t - ncx) * DF_HD;
#pragma unroll 8
      for (int dd = 0; dd < DF_HD; ++dd)
#pragma unroll
        for (int b = 0; b < DF_B; ++b) d4[b] += kk[dd] * qv[b * DF_HD + dd];
    }
#pragma unroll
    for (int b = 0; b < DF_B; ++b) {
      float const x = d4[b] * 0.125f;
      sc[b * cs + t] = x;
      mx[b] = fmaxf(mx[b], x);
    }
  }
  df_block_max_n<DF_B>(mx, red);
  DF_ATS(1);
#pragma unroll
  for (int b = 0; b < DF_B; ++b) sm[b] = 0.f;
  for (int t = tid; t < nk; t += blockDim.x) {
#pragma unroll
    for (int b = 0; b < DF_B; ++b) {
      float const e = expf(sc[b * cs + t] - mx[b]);
      sc[b * cs + t] = e;
      sm[b] += e;
    }
  }
  df_block_sum_n<DF_B>(sm, red);
  DF_ATS(2);
  int const pg = tid >> 4, dc = tid & 15;  // 16 position groups x 16 lanes of 4 dims
  float a[DF_B][4];
#pragma unroll
  for (int b = 0; b < DF_B; ++b)
#pragma unroll
    for (int e = 0; e < 4; ++e) a[b][e] = 0.f;
  constexpr int PVJ = 16;  // positions pg + 16 j per batch of loads
  for (int j0 = 0; pg + 16 * j0 < nk; j0 += PVJ) {
    uint2 vq[PVJ];
#pragma unroll
    for (int j = 0; j < PVJ; ++j) {
      int const t = pg + 16 * (j0 + j);
      vq[j] = t < ncx ? *reinterpret_cast<uint2 const *>(vc + (size_t)(c0 + t) * rs + dc * 4) : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < PVJ; ++j) {
      int const t = pg + 16 * (j0 + j);
      if (t < nk) {
        float vf[4];
        if (t < ncx) {
          vf[0] = df_bf(vq[j].x & 0xFFFFu); vf[1] = df_bf(vq[j].x >> 16);
          vf[2] = df_bf(vq[j].y & 0xFFFFu); vf[3] = df_bf(vq[j].y >> 16);
        } else {
          float const *const vv = vb + (t - ncx) * DF_HD + dc * 4;
          vf[0] = vv[0]; vf[1] = vv[1]; vf[2] = vv[2]; vf[3] = vv[3];
        }
#pragma unroll
        for (int b = 0; b < DF_B; ++b) {
          float const p = sc[b * cs + t];
#pragma unroll
          for (int e = 0; e < 4; ++e) a[b][e] += p * vf[e];
        }
      }
    }
  }
#pragma unroll
  for (int b = 0; b < DF_B; ++b)
#pragma unroll
    for (int e = 0; e < 4; ++e) pv[(pg * DF_B + b) * DF_HD + dc * 4 + e] = a[b][e];
  __syncthreads();
  DF_ATS(3);
  if (nsp == 1) {  // the head's whole context: O = sum / l (what the merge of a single split computes)
    for (int x = tid; x < DF_B * DF_HD; x += 256) {
      int const b = x >> 6, d = x & 63;
      float sum = 0.f;
#pragma unroll
      for (int q = 0; q < 16; ++q) sum += pv[(q * DF_B + b) * DF_HD + d];
      df_st2h(L + DF_L_O, b * DF_QD + h * DF_HD + d, df_tobf(sum / sm[b]));
    }
    DF_ATS(4);
    return;
  }
  float *const part = g_df.vec + L + DF_L_AP + (h * DF_SP + sp) * DF_APB;
  for (int x = tid; x < DF_B * DF_HD; x += 256) {
    int const b = x >> 6, d = x & 63;
    float sum = 0.f;
#pragma unroll
    for (int q = 0; q < 16; ++q) sum += pv[(q * DF_B + b) * DF_HD + d];
    df_st_sys_u32(part + b * 68 + d, __float_as_uint(sum));
    if (d == 0) {
      df_st_sys_u32(part + b * 68 + 64, __float_as_uint(mx[b]));
      df_st_sys_u32(part + b * 68 + 65, __float_as_uint(sm[b]));
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    int const old = __hip_atomic_fetch_add(g_aid_nc_rep[0] + DF_NC_HEAD + h * 16, 1, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
    red[29] = __int_as_float(old % DF_SP == DF_SP - 1 ? 1 : 0);
  }
  __syncthreads();
  DF_ATS(4);
  if (__float_as_int(red[29]) != 1) return;  // not the head's last split
  for (int x = tid; x < DF_B * DF_HD; x += 256) {
    int const b = x >> 6, d = x & 63;
    float const *const ph = g_df.vec + L + DF_L_AP + h * DF_SP * DF_APB + b * 68;
    float m[DF_SP], ls[DF_SP], os[DF_SP];
    float M = -3.0e38f;
#pragma unroll
    for (int q = 0; q < DF_SP; ++q) {
      m[q] = df_ld_sys_f32(ph + q * DF_APB + 64);
      ls[q] = df_ld_sys_f32(ph + q * DF_APB + 65);
      os[q] = df_ld_sys_f32(ph + q * DF_APB + d);
      M = fmaxf(M, m[q]);
    }
    float Ls = 0.f, Os = 0.f;
#pragma unroll
    for (int q = 0; q < DF_SP; ++q) {
      float const f = expf(m[q] - M);
      Ls += ls[q] * f;
      Os += os[q] * f;
    }
    df_st2h(L + DF_L_O, b * DF_QD + h * DF_HD + d, df_tobf(Os / Ls));
  }
}

// Short context (P + DF_B <= 256): rows [rb, rb + RW) of q head h over the whole context and the DF_B block keys
// on one worker (wid = h * (DF_B / RW) + rb / RW): complete output rows, no partials or merge. Per row the same
// arithmetic as a single context split of df_attn.
template <int RW>
__device__ DF_SINL void df_attn_rows(int l, int L, int wid, int P, float *red_in, float *lds_in,
                                     unsigned long long *ats = nullptr) {
  constexpr int NRG = DF_B / RW;  // workers per q head
  static_assert(DF_B % RW == 0, "whole row groups");
  static_assert(RW <= 4, "one output row per 64 threads");
  static_assert(DF_NQ * NRG <= 248, "one worker per (q head, row group)");
  static_assert(DF_B <= 8, "block rows wave and wave + 4 in the prep");
  int const tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
  if (wid >= DF_NQ * NRG) return;
#if MPK_MAX_NUM_BATCHED_TOKENS > 4
  // own function: LDS re-derived from the extern symbol (a pointer parameter would make every LDS access FLAT); the
  // block-reduction scratch above this function's LDS (qv, kb, vb, sc, pv <= 6400 floats at RW 4, DF_B 8)
  extern __shared__ char _rnlm_smem[];
  float *const lds = reinterpret_cast<float *>(_rnlm_smem);
  float *const red = lds + 8192;
  static_assert(RW * DF_HD + 2 * DF_B * DF_HD + RW * 256 + 16 * RW * DF_HD <= 8192, "attention LDS below red");
  (void)red_in;
  (void)lds_in;
#else
  float *const lds = lds_in;
  float *const red = red_in;
#endif
  int const h = wid / NRG, rb = (wid - h * NRG) * RW, g = h / DF_GQ;
  float const *const rv = g_df.vec;
  float *const qv = lds;                // [RW][64]
  float *const kb = qv + RW * DF_HD;    // [DF_B][64] block keys
  float *const vb = kb + DF_B * DF_HD;  // block values
  float *const sc = vb + DF_B * DF_HD;  // [RW][cs] scores
  int const ncx = P, nk = P + DF_B;     // nk <= 256: one context position per thread, one PV batch
  int const cs = (nk + 3) & ~3;
  float *const pv = sc + RW * cs;  // [16][RW][64] partial V sums
  size_t const rs = (size_t)DF_NKV * DF_HD;  // cache row stride (one position)
  unsigned short const *const kc = g_df.kc + (size_t)l * g_df.maxpos * rs + g * DF_HD;
  unsigned short const *const vc = g_df.vc + (size_t)l * g_df.maxpos * rs + g * DF_HD;
  int const d = lane;
  // The prep's loads first (scratch q / k / v rows, biases, norm weights, rope), then this thread's K row and V
  // slices with clamped addresses: their round trips overlap the prep's arithmetic, and the prep waits only for its
  // own loads (vmcnt retires in issue order).
  bool const qw = wave < RW;
  int const bq = rb + (qw ? wave : 0), ko = g * DF_HD + d;
  float const q_raw = rv[L + DF_L_QR + bq * DF_QD + h * DF_HD + d], q_b = g_df.bq[l * DF_QD + h * DF_HD + d];
  float const k_raw = rv[L + DF_L_KR + wave * DF_KVD + ko], k_b = g_df.bk[l * DF_KVD + ko];
  float const v_raw = rv[L + DF_L_VR + wave * DF_KVD + ko], v_b = g_df.bv[l * DF_KVD + ko];
  constexpr bool KV2 = DF_B > 4;  // past block 4 wave w also prepares block row w + 4 (when < DF_B)
  bool const kw2 = KV2 && wave + 4 < DF_B;
  int const b2 = kw2 ? wave + 4 : wave;
  float k_raw2 = 0.f, v_raw2 = 0.f;
  if constexpr (KV2) {
    k_raw2 = rv[L + DF_L_KR + b2 * DF_KVD + ko];
    v_raw2 = rv[L + DF_L_VR + b2 * DF_KVD + ko];
  }
  float const gq = df_bf(g_df.nq[l * DF_HD + d]), gk = df_bf(g_df.nk[l * DF_HD + d]);
  float const *const tq = g_df.rope + (size_t)(P + bq) * 64;
  float const *const tk = g_df.rope + (size_t)(P + wave) * 64;
  float const cq = tq[d & 31], sq = tq[32 + (d & 31)], ck = tk[d & 31], sk = tk[32 + (d & 31)];
  float ck2 = 0.f, sk2 = 0.f;
  if constexpr (KV2) {
    float const *const tk2 = g_df.rope + (size_t)(P + b2) * 64;
    ck2 = tk2[d & 31];
    sk2 = tk2[32 + (d & 31)];
  }
  uint4 kv[8];
  {
    uint4 const *const kr = reinterpret_cast<uint4 const *>(kc + (size_t)(tid < ncx ? tid : 0) * rs);
#pragma unroll
    for (int c = 0; c < 8; ++c) kv[c] = kr[c];
  }
  int const pg = tid >> 4, dc = tid & 15;  // 16 position groups x 16 lanes of 4 dims
  constexpr int PVJ = 16;
  uint2 vq[PVJ];
#pragma unroll
  for (int j = 0; j < PVJ; ++j) {
    int const t = pg + 16 * j;
    vq[j] = *reinterpret_cast<uint2 const *>(vc + (size_t)(t < ncx ? t : 0) * rs + dc * 4);
  }
  {
    float const qn = df_qk_norm_rope_p(df_rbf(q_raw + q_b), gq, cq, sq, d);
    if (qw) qv[wave * DF_HD + d] = qn;
    kb[wave * DF_HD + d] = df_qk_norm_rope_p(df_rbf(k_raw + k_b), gk, ck, sk, d);
    vb[wave * DF_HD + d] = df_rbf(v_raw + v_b);
    if constexpr (KV2) {
      if (kw2) {  // wave-uniform
        kb[b2 * DF_HD + d] = df_qk_norm_rope_p(df_rbf(k_raw2 + k_b), gk, ck2, sk2, d);
        vb[b2 * DF_HD + d] = df_rbf(v_raw2 + v_b);
      }
    }
  }
  __syncthreads();
  DF_ATS(0);
  float mx[RW], sm[RW];
#pragma unroll
  for (int b = 0; b < RW; ++b) mx[b] = -3.0e38f;
  if (tid < nk) {
    int const t = tid;
    float d4[RW];
#pragma unroll
    for (int b = 0; b < RW; ++b) d4[b] = 0.f;
    if (t < ncx) {
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        unsigned const w4[4] = {kv[c].x, kv[c].y, kv[c].z, kv[c].w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float const kf = df_bf((e & 1) ? (w4[e >> 1] >> 16) : (w4[e >> 1] & 0xFFFFu));
#pragma unroll
          for (int b = 0; b < RW; ++b) d4[b] += kf * qv[b * DF_HD + c * 8 + e];
        }
      }
    } else {
      float const *const kk = kb + (t - ncx) * DF_HD;
#pragma unroll 8
      for (int dd = 0; dd < DF_HD; ++dd)
#pragma unroll
        for (int b = 0; b < RW; ++b) d4[b] += kk[dd] * qv[b * DF_HD + dd];
    }
#pragma unroll
    for (int b = 0; b < RW; ++b) {
      float const x = d4[b] * 0.125f;
      sc[b * cs + t] = x;
      mx[b] = fmaxf(mx[b], x);
    }
  }
  df_block_max_n<RW>(mx, red);
  DF_ATS(1);
#pragma unroll
  for (int b = 0; b < RW; ++b) sm[b] = 0.f;
  for (int t = tid; t < nk; t += blockDim.x) {
#pragma unroll
    for (int b = 0; b < RW; ++b) {
      float const e = expf(sc[b * cs + t] - mx[b]);
      sc[b * cs + t] = e;
      sm[b] += e;
    }
  }
  df_block_sum_n<RW>(sm, red);
  DF_ATS(2);
  float a[RW][4];
#pragma unroll
  for (int b = 0; b < RW; ++b)
#pragma unroll
    for (int e = 0; e < 4; ++e) a[b][e] = 0.f;
#pragma unroll
  for (int j = 0; j < PVJ; ++j) {
    int const t = pg + 16 * j;
    if (t < nk) {
      float vf[4];
      if (t < ncx) {
        vf[0] = df_bf(vq[j].x & 0xFFFFu); vf[1] = df_bf(vq[j].x >> 16);
        vf[2] = df_bf(vq[j].y & 0xFFFFu); vf[3] = df_bf(vq[j].y >> 16);
      } else {
        float const *const vv = vb + (t - ncx) * DF_HD + dc * 4;
        vf[0] = vv[0]; vf[1] = vv[1]; vf[2] = vv[2]; vf[3] = vv[3];
      }
#pragma unroll
      for (int b = 0; b < RW; ++b) {
        float const p = sc[b * cs + t];
#pragma unroll
        for (int e = 0; e < 4; ++e) a[b][e] += p * vf[e];
      }
    }
  }
#pragma unroll
  for (int b = 0; b < RW; ++b)
#pragma unroll
    for (int e = 0; e < 4; ++e) pv[(pg * RW + b) * DF_HD + dc * 4 + e] = a[b][e];
  __syncthreads();
  DF_ATS(3);
  if (tid < RW * DF_HD) {
    int const b = tid >> 6, dd = tid & 63;
    float sum = 0.f;
#pragma unroll
    for (int q = 0; q < 16; ++q) sum += pv[(q * RW + b) * DF_HD + dd];
    df_st2h(L + DF_L_O, (rb + b) * DF_QD + h * DF_HD + dd, df_tobf(sum / sm[b]));
  }
  DF_ATS(4);
}

// Workers without attention work touch layer l's second-round gate/up pairs (pair 2 j + 1 of each device's half:
// the second pair of the workers that have two), one dword per 128-byte line, so the lines are MALL-resident when
// gate/up streams them; the HBM is otherwise idle during the attention, and the MALL is memory-side, so a remote
// read warms the owning AID's MALL. Sized to finish inside the attention. k of nw warmers.
__device__ __forceinline__ void df_warm_gu2(int l, int k, int nw) {
#ifdef MPK_DF_HALVES
  int const wave = (threadIdx.x & 255) >> 6, lane = threadIdx.x & 63;
  constexpr int NP2 = DF_T_FF / 2 - 124;  // second pairs per device: workers 0 .. 115 of 124 take two pairs
  static_assert(NP2 == 116, "two pairs for 116 of the 124 workers of a device");
  constexpr int LPT = DF_KT_H * DF_F8 / 128;  // lines per tile
  constexpr long long TL = 2LL * NP2 * 2 * LPT;  // devices x pairs x (gate, up) x lines
  long long const nwv = (long long)nw * 4, gw = (long long)k * 4 + wave;
  long long const lo = TL * gw / nwv, hi = TL * (gw + 1) / nwv;
  auto at = [&](long long g) -> unsigned char const * {
    long long const c = g / LPT;
    int const line = (int)(g - c * LPT), gu = (int)(c & 1), dj = (int)(c >> 1);
    int const d = dj / NP2, j = dj - d * NP2;
    return DfWgu::at(gu ? g_df.wu : g_df.wg, l, d * (DF_T_FF / 2) + 2 * j + 1) + (size_t)line * 128;
  };
  for (long long g0 = lo + lane; g0 < hi; g0 += 8 * 64) {
    unsigned char const *p[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      long long const g = g0 + 64LL * j;
      p[j] = at(g < hi ? g : g0);
    }
    int d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile(
        "global_load_dword %0, %8, off\n\t"
        "global_load_dword %1, %9, off\n\t"
        "global_load_dword %2, %10, off\n\t"
        "global_load_dword %3, %11, off\n\t"
        "global_load_dword %4, %12, off\n\t"
        "global_load_dword %5, %13, off\n\t"
        "global_load_dword %6, %14, off\n\t"
        "global_load_dword %7, %15, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3), "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7])
        : "memory");
  }
#else
  (void)l; (void)k; (void)nw;
#endif
}

// Workers without a down tile touch the next layer's q/k/v tiles of their own device (local), one dword per line,
// so the next q/k/v stage's prefetch and GEMV read them from the MALL. k of nw warmers on device dev.
__device__ __forceinline__ void df_warm_qkv(int l1, int dev, int k, int nw) {
#ifdef MPK_DF_HALVES
  int const wave = (threadIdx.x & 255) >> 6, lane = threadIdx.x & 63;
  constexpr int NU = DF_T_QKV / DF_NDEV;
  constexpr int LPT = DF_KT_H * DF_F8 / 128;  // lines per tile
  constexpr long long TL = (long long)NU * LPT;
  long long const nwv = (long long)nw * 4, gw = (long long)k * 4 + wave;
  long long const lo = TL * gw / nwv, hi = TL * (gw + 1) / nwv;
  auto at = [&](long long g) -> unsigned char const * {
    int const u = (int)(g / LPT), line = (int)(g - (long long)u * LPT);
    return df_qkv_tp(l1, df_qkv_T(dev, u)) + (size_t)line * 128;
  };
  for (long long g0 = lo + lane; g0 < hi; g0 += 8 * 64) {
    unsigned char const *p[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      long long const g = g0 + 64LL * j;
      p[j] = at(g < hi ? g : g0);
    }
    int d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile(
        "global_load_dword %0, %8, off\n\t"
        "global_load_dword %1, %9, off\n\t"
        "global_load_dword %2, %10, off\n\t"
        "global_load_dword %3, %11, off\n\t"
        "global_load_dword %4, %12, off\n\t"
        "global_load_dword %5, %13, off\n\t"
        "global_load_dword %6, %14, off\n\t"
        "global_load_dword %7, %15, off\n\t"
        "s_waitcnt vmcnt(0)"
        : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3), "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
        : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]), "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7])
        : "memory");
  }
#else
  (void)l1; (void)dev; (void)k; (void)nw;
#endif
}

#ifdef MPK_DF_TS
#ifndef MPK_DF_TS_W
#define MPK_DF_TS_W 0
#endif
// Stage stamps of worker 0 (10 ns ticks) for two verify passes past prompt + 30: after each stage's own work and
// after each barrier.
__device__ int g_df_tsn;
#define DF_TSL(id)                                                         \
  do {                                                                     \
    if (wid == MPK_DF_TS_W && tid == 0 && dfn < 240) {                     \
      dfts[dfn] = __builtin_amdgcn_s_memrealtime();                        \
      dflb[dfn++] = (unsigned char)(id);                                   \
    }                                                                      \
  } while (0)
#else
#define DF_TSL(id) \
  do {             \
  } while (0)
#endif
#define DF_TS() DF_TSL(0)

// Stage 0, at the start of the LM-head task: target features of this pass's n slots into scratch.
__device__ __noinline__ void mpk_dflash_features(int wid, DfSt const st) {
  if (!g_df.enabled || st.st0 != 1) return;
  int const W = g_df.nworkers;
  int const total = st.n * DF_FEAT;
  int const per = (total + W - 1) / W;
  int const e0 = wid * per, e1 = min(total, e0 + per);
  for (int e = e0 + threadIdx.x; e < e1; e += blockDim.x) {
    int const r = e / DF_FEAT, i = e - r * DF_FEAT;
    int const Lt = i / DF_H, c = i - Lt * DF_H;
    int const ai = r * DF_HP + c;
    unsigned const wa = df_ld_sys_u32(g_df.apo[Lt] + (ai & ~1));
    float v = df_bf((ai & 1) ? (wa >> 16) : (wa & 0xFFFFu));
    float const *const wsr = g_df.ws[Lt] + (size_t)r * g_df.ws_stride + c;
#pragma unroll
    for (int k = 0; k < 4; ++k) v += df_ld_sys_f32(wsr + k * DF_HP);
    df_st2h(DF_V_FEAT, e, df_tobf(v));
  }
}

// wid: this worker 0..nworkers-1; die: its logical XCD.
__device__ __noinline__ void mpk_dflash_phase(int wid, int die, DfSt const st) {
  if (!g_df.enabled || st.st0 != 1) return;
  extern __shared__ char _rnlm_smem[];
  float *const lds = reinterpret_cast<float *>(_rnlm_smem);
  __shared__ float df_red[32];
  __shared__ long long df_y[DF_B];
  __shared__ long long df_tok[DF_B];
  int const tid = threadIdx.x, wave = tid >> 6, lane = tid & 63, col = lane & 15, gq = lane >> 4;
  int const W = g_df.nworkers;
  float const *const rv = g_df.vec;
  int const n = st.n, s = st.s, kind = st.kind;
  int const prompt_len = *g_df.prompt_len;
  int gen = (s + 1) * 64;
  bool const full = kind == 1 || s + n >= prompt_len;
#ifdef MPK_DF_TS
  unsigned long long dfts[240];
  unsigned char dflb[240];
  unsigned long long dfa[8];
  int dfn = 0;
#endif
  DF_TSL(1);
  df_barrier(++gen, die);  // features + target argmax partials
  DF_TSL(2);

  // ---- acceptance: R final rows (positions s .. s + R - 1); the next anchor y[R - 1] sits at P = s + R
  // g_spec_prev_in[1 .. DF_B - 1]: the drafts this verify pass checked (thread 0). Plain loads, like df_state():
  // written by prepare_next_batch before this task's agent-scope acquire (volatile = system scope, which costs a
  // device-1 worker ~10 us against the device-0 buffer).
  long long pin[DF_B];
  if (tid == 0) {
    long long const *const pvi = (long long const *)(g_spec_prev_in);
#pragma unroll
    for (int j = 1; j < DF_B; ++j) pin[j] = pvi[j];
  }
  {  // per row j: argmax over the W target partials (ties to the lower worker), the winner's index as payload
    float vv[DF_B];
    int ky[DF_B];
    unsigned ix[DF_B];
#pragma unroll
    for (int j = 0; j < DF_B; ++j) {
      bool const ok = j < n && tid < W;
      vv[j] = ok ? df_bf(__float_as_uint(rv[DF_V_YV + j * 256 + tid])) : -3.0e38f;
      ix[j] = ok ? __float_as_uint(rv[DF_V_YI + j * 256 + tid]) : 0u;
      ky[j] = tid < W ? tid : 0x7fffffff;
    }
#pragma unroll
    for (int o = 32; o >= 1; o >>= 1) {
#pragma unroll
      for (int j = 0; j < DF_B; ++j) {
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
    DF_TSL(80);
    float *const ar = lds;  // [4 waves][DF_B][3]
    if (lane == 0)
#pragma unroll
      for (int j = 0; j < DF_B; ++j) {
        ar[(wave * DF_B + j) * 3] = vv[j];
        ar[(wave * DF_B + j) * 3 + 1] = __int_as_float(ky[j]);
        ar[(wave * DF_B + j) * 3 + 2] = __uint_as_float(ix[j]);
      }
    __syncthreads();
    if (tid < n) {
      int const j = tid;
      float bv = ar[j * 3];
      int bk = __float_as_int(ar[j * 3 + 1]);
      unsigned bi = __float_as_uint(ar[j * 3 + 2]);
      for (int w4 = 1; w4 < 4; ++w4) {
        float const ov = ar[(w4 * DF_B + j) * 3];
        int const ok = __float_as_int(ar[(w4 * DF_B + j) * 3 + 1]);
        if (ov > bv || (ov == bv && ok < bk)) {
          bv = ov;
          bk = ok;
          bi = __float_as_uint(ar[(w4 * DF_B + j) * 3 + 2]);
        }
      }
      df_y[j] = (long long)bi;
    }
  }
  __syncthreads();
  DF_TSL(81);
  if (tid == 0) {
    int rr = n;
    if (kind == 1) {
      int a = 0;
#pragma unroll
      for (int j = 0; j < DF_B - 1; ++j)
        if (a == j && a < n - 1 && pin[j + 1] == df_y[j]) ++a;
      rr = a + 1;
    }
    df_red[31] = __int_as_float(rr);
    long long const y = df_y[rr - 1];
    df_tok[0] = y < 0 ? 0 : y;
    for (int b = 1; b < DF_B; ++b) df_tok[b] = DF_MASK;
  }
  __syncthreads();
  int const R = __float_as_int(df_red[31]);
  DF_TSL(82);
  int const P = s + R;
  if (R < 1 || P + DF_B > g_df.maxpos) return;  // same decision on every worker
  DF_TSL(3);

  // ---- stage 1: fc of the R final rows
  {
    // past block 4 the e4m3 rows start at the LDS base (xf is unused) so that 8 rows of DF_FEATP fit
    float *const xf = lds;                                                         // one feature row
    unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + (DF_B <= 4 ? DF_FEAT : 0));  // R x DF_FEATP
    unsigned char *const qs = q8 + DF_B * DF_FEATP;                                // R x DF_KT_FEAT
    float *const s_red = reinterpret_cast<float *>(qs + (DF_B <= 4 ? 512 : 1024));
    static_assert(DF_B * DF_KT_FEAT <= (DF_B <= 4 ? 512 : 1024), "fc scale rows fit their reserve");
    (void)xf;
    int t0, t1;
    df_dsplit(DF_T_H, wid, W, t0, t1);
    if (t0 < t1) {  // workers without a tile skip the stage
    {
      unsigned short const *const fb = reinterpret_cast<unsigned short const *>(rv + DF_V_FEAT);
      constexpr int FB = 16;
#pragma unroll 1
      for (int b0 = 0; b0 < R * (DF_FEATP / 8); b0 += 256 * FB) {
        DfQ8Bh<DF_FEAT, DF_FEATP, FB> q;
        q.ld(fb, DF_FEAT, R, b0);
        q.fin(R, b0, q8, qs);
      }
      __syncthreads();
    }
    DF_TSL(13);
    for (int t = t0; t < t1; t += 2) {
      int const nt = min(2, t1 - t);
      unsigned char const *const tb = DfFc::at(g_df.fc, 0, t);
      float const sc_pf =
          tid < nt * DF_TPT ? g_df.sfc[(t + (tid >> (4 + DF_RCS))) * 16 + ((tid >> DF_RCS) & 15)] : 0.f;
      df_round8<DF_KT_FEAT>(tb, DfFc::at(g_df.fc, 0, t + nt - 1), nt, q8, qs, R, s_red);
      if (tid < nt * DF_TPT) {
        int const j = tid >> (4 + DF_RCS), row = (tid >> DF_RCS) & 15, c = tid & (DF_RC - 1);
        if (c < R) {
          int const o = (t + j) * 16 + row;
          df_st2h(DF_V_HT, c * DF_H + o, df_tobf(df_sum4(s_red, j, row, c) * sc_pf));
        }
      }
      __syncthreads();
    }
    }
  }
  DF_TSL(4);
  df_barrier(++gen, die);
  DF_TSL(9);

  // ---- stage 2: context K/V of the R final rows for all 8 layers (one (layer, k|v, kv head) unit per worker,
  // its 4 waves the head's 4 row tiles over the full K); on the other workers (full passes) layer 0's block q/k/v
  {
    float *const xr = lds;                                                          // DF_B x DF_H
    unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + DF_B * DF_H);  // DF_B x DF_HP
    unsigned char *const qs = q8 + DF_B * DF_HP;                                    // DF_B x DF_KT_H
    float *const s_red = reinterpret_cast<float *>(qs + DF_QSP);
    constexpr int NCU = DF_NL * 2 * DF_NKV;
    constexpr int NCUD = NCU / DF_NDEV;
    DfDev const dd = df_dev(wid, W);
    if (dd.lw < NCUD) {
      (void)xr;
      df_rmsq8_gh(reinterpret_cast<unsigned short const *>(rv + DF_V_HT), R, g_df.nhid, q8, qs, df_red);
      DF_TSL(15);
#ifdef MPK_DF_HALVES
      int const l = dd.lw >> 3, kv = (dd.lw >> 2) & 1, g = dd.dev * 4 + (dd.lw & 3);
#else
      int const l = wid >> 4, kv = (wid >> 3) & 1, g = wid & 7;
#endif
      unsigned char const *const tb = DfWkv::at(kv ? g_df.wv : g_df.wk, l, g * 4 + wave);
      unsigned char const *wr[2] = {tb, tb};
      df_f32x4 acc[2];
      df_m8gemv<DF_KT_H, 1, 4>(wr, q8, DF_HP, qs, DF_KT_H, R, acc);
      if (col < DF_RC)
#pragma unroll
        for (int i = 0; i < 4; ++i) s_red[(wave * 16 + gq * 4 + i) * DF_RC + col] = acc[0][i];  // [dim][row]
      __syncthreads();
      for (int r = wave; r < R; r += 4) {  // one accepted row per wave and round
        int const d = lane, ro = l * DF_KVD + g * DF_HD + d;
        float v = df_rbf(df_rbf(s_red[d * DF_RC + r] * (kv ? g_df.swv : g_df.swk)[ro]) + (kv ? g_df.bv : g_df.bk)[ro]);
        if (!kv) v = df_qk_norm_rope(v, g_df.nk + l * DF_HD, s + r, d);
        float const v1 = __shfl_down(v, 1, 64);
        unsigned short *const crow =
            (kv ? g_df.vc : g_df.kc) + (((size_t)l * g_df.maxpos + s + r) * DF_NKV + g) * DF_HD;
        if ((d & 1) == 0) df_st_sys_u32(crow + d, df_tobf(v) | (df_tobf(v1) << 16));
      }
    } else if (full) {
      df_rmsq8_emb(df_tok, g_df.nin, q8, qs, df_red);
      int t0, t1;
      df_bsplit(DF_T_QKV / DF_NDEV, dd.lw - NCUD, dd.wd - NCUD, t0, t1);
      df_qkv_tiles<false>(0, dd.dev, t0, t1, q8, qs, s_red, DF_V_L0, nullptr);
    }
  }
  DF_TSL(5);
  if (!full) return;
  df_barrier(++gen, die);
  DF_TSL(9);

  for (int l = 0; l < DF_NL; ++l) {
    int const L = DF_V_L0 + l * DF_L_SZ;
    int const Lp = DF_V_L0 + (l - 1) * DF_L_SZ;
    if (l > 0) {  // ---- q / k / v of the block from the previous layer's output
      float *const xr = lds;
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + DF_B * DF_H);
      unsigned char *const qs = q8 + DF_B * DF_HP;
      float *const s_red = reinterpret_cast<float *>(qs + DF_QSP);
      (void)xr;
      int t0, t1;
      DfDev const dq = df_dev(wid, W);
      df_bsplit(DF_T_QKV / DF_NDEV, dq.lw, dq.wd, t0, t1);
      DfPre8 qpre;
      {  // activations first, then the first round's weights, then the prologue math
        DfRqT<true> act;
        act.ld(reinterpret_cast<unsigned short const *>(rv + Lp + DF_L_X), DF_B, g_df.nin + l * DF_H);
        int const nt = min(2, t1 - t0);  // >= 1 (fewer workers than tiles)
        df_pre8<DF_KT_H, 2>(df_qkv_tp(l, df_qkv_T(dq.dev, t0)), df_qkv_tp(l, df_qkv_T(dq.dev, t0 + nt - 1)), qpre);
#ifdef MPK_DF_TS
        unsigned long long fts[2];
        act.fin(DF_B, q8, qs, df_red, wid == MPK_DF_TS_W ? fts : nullptr);
        if (wid == MPK_DF_TS_W && tid == 0 && dfn + 2 < 240) {
          dfts[dfn] = fts[0]; dflb[dfn++] = 21;
          dfts[dfn] = fts[1]; dflb[dfn++] = 22;
        }
#else
        act.fin(DF_B, q8, qs, df_red);
#endif
      }
      DF_TSL(23);
      df_qkv_tiles<true>(l, dq.dev, t0, t1, q8, qs, s_red, L, &qpre);
      DF_TSL(20);
      df_barrier(++gen, die);
      DF_TSL(9);
    }
    // ---- attention
#ifndef MPK_DF_ATT_RW
#define MPK_DF_ATT_RW (DF_B <= 6 ? 2 : 4)  // rows per worker: 64 q heads x DF_B / RW row groups <= 248 workers
#endif
#ifdef MPK_DF_TS
    if (P + DF_B <= 256) df_attn_rows<MPK_DF_ATT_RW>(l, L, wid, P, df_red, lds, wid == MPK_DF_TS_W ? dfa : nullptr);
    else if (wid < DF_NQ * DF_SP) df_attn(l, L, wid, P, df_red, lds, wid == MPK_DF_TS_W ? dfa : nullptr);
    if (wid == MPK_DF_TS_W && tid == 0)
      for (int k = 0; k < 5 && dfn < 240; ++k) { dfts[dfn] = dfa[k]; dflb[dfn++] = (unsigned char)(31 + k); }
#else
    if (P + DF_B <= 256) df_attn_rows<MPK_DF_ATT_RW>(l, L, wid, P, df_red, lds);
    else if (wid < DF_NQ * DF_SP) df_attn(l, L, wid, P, df_red, lds);
#endif
#ifndef MPK_DF_NO_WARM
    {
      int const na = P + DF_B <= 256 ? DF_NQ * (DF_B / MPK_DF_ATT_RW) : DF_NQ * DF_SP;  // attention workers
      if (wid >= na) df_warm_gu2(l, wid - na, W - na);
    }
#endif
    DF_TSL(30);
    df_barrier(++gen, die);
    DF_TSL(9);
    // ---- o_proj + bias + residual
    {
      // past block 4 the e4m3 rows start at the LDS base (xo is unused): 8 f32 rows of DF_QD would leave no room
      float *const xo = lds;
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + (DF_B <= 4 ? DF_B * DF_QD : 0));
      unsigned char *const qs = q8 + DF_B * DF_QD;
      float *const s_red = reinterpret_cast<float *>(qs + DF_QSP);
      (void)xo;
      DF_TSL(41);
      int t0, t1;
      df_dsplit(DF_T_H, wid, W, t0, t1);
      if (t0 < t1) {  // workers without a tile skip the prologue
      DfPre8 opre;
      {
        constexpr int NIQ = DF_B * DF_QD / 8;  // 8-value items, in batches of 256 threads x 8
        unsigned short const *const xo16 = reinterpret_cast<unsigned short const *>(rv + L + DF_L_O);
        DfQ8Bh<DF_QD, DF_QD, 8> qa;
        qa.ld(xo16, DF_QD, DF_B, 0);
        int const tq = t0 < t1 ? t0 : t0 - 1;  // t0 >= 1 when empty
        df_pre8<DF_KT_Q, 1>(DfWo::at(g_df.wo, l, tq), DfWo::at(g_df.wo, l, tq), opre);
        qa.fin(DF_B, 0, q8, qs);
#pragma unroll 1
        for (int b0 = 256 * 8; b0 < NIQ; b0 += 256 * 8) {
          DfQ8Bh<DF_QD, DF_QD, 8> qb;
          qb.ld(xo16, DF_QD, DF_B, b0);
          qb.fin(DF_B, b0, q8, qs);
        }
        __syncthreads();
      }
      DF_TSL(42);
      for (int t = t0; t < t1; t += 2) {
        int const nt = min(2, t1 - t);
        unsigned char const *const tb = DfWo::at(g_df.wo, l, t);
        float sw_pf = 0.f, b_pf = 0.f, r_pf = 0.f;
        if (tid < nt * DF_TPT && (tid & (DF_RC - 1)) < DF_B) {
          int const o = (t + (tid >> (4 + DF_RCS))) * 16 + ((tid >> DF_RCS) & 15), c = tid & (DF_RC - 1);
          sw_pf = g_df.swo[l * DF_H + o];
          b_pf = g_df.bo[l * DF_H + o];
          r_pf = l == 0 ? df_bf(g_df.embed[(size_t)df_tok[c] * DF_H + o])
                        : df_bf(reinterpret_cast<unsigned short const *>(rv + Lp + DF_L_X)[c * DF_H + o]);
        }
        if (t == t0 && nt == 1) df_round8<DF_KT_Q, true>(tb, tb, 1, q8, qs, DF_B, s_red, &opre);
        else df_round8<DF_KT_Q>(tb, DfWo::at(g_df.wo, l, t + nt - 1), nt, q8, qs, DF_B, s_red);
        if (tid < nt * DF_TPT) {
          int const j = tid >> (4 + DF_RCS), row = (tid >> DF_RCS) & 15, c = tid & (DF_RC - 1);
          if (c < DF_B) {
            int const o = (t + j) * 16 + row;
            float const y = df_rbf(df_rbf(df_sum4(s_red, j, row, c) * sw_pf) + b_pf);
            float const res = r_pf;
            df_st2h(L + DF_L_X2, c * DF_H + o, df_tobf(res + y));
          }
        }
        __syncthreads();
      }
      }
    }
    DF_TSL(40);
    df_barrier(++gen, die);
    DF_TSL(9);
    // ---- gate / up: a worker's gate and up row tiles of the same 16 outputs in one round
    {
      float *const xr = lds;
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + DF_B * DF_H);
      unsigned char *const qs = q8 + DF_B * DF_HP;
      float *const s_red = reinterpret_cast<float *>(qs + DF_QSP);
      (void)xr;
      int p0, p1;
      df_dsplit(DF_T_FF, wid, W, p0, p1);
      DfPre8 gpre;
      {
        DfRqT<true> act;
        act.ld(reinterpret_cast<unsigned short const *>(rv + L + DF_L_X2), DF_B, g_df.npost + l * DF_H);
        df_pre8<DF_KT_H, 2>(DfWgu::at(g_df.wg, l, p0), DfWgu::at(g_df.wu, l, p0), gpre);  // >= 1 pair per worker
#ifdef MPK_DF_TS
        unsigned long long fts[2];
        act.fin(DF_B, q8, qs, df_red, wid == MPK_DF_TS_W ? fts : nullptr);
        if (wid == MPK_DF_TS_W && tid == 0 && dfn + 2 < 240) {
          dfts[dfn] = fts[0]; dflb[dfn++] = 51;
          dfts[dfn] = fts[1]; dflb[dfn++] = 52;
        }
#else
        act.fin(DF_B, q8, qs, df_red);
#endif
      }
      DF_TSL(53);
      for (int p = p0; p < p1; ++p) {
        unsigned char const *const tg = DfWgu::at(g_df.wg, l, p);
        unsigned char const *const tu = DfWgu::at(g_df.wu, l, p);
        float sg_pf = 0.f, su_pf = 0.f;
        if (tid < DF_TPT) {
          int const o = p * 16 + ((tid >> DF_RCS) & 15);
          sg_pf = g_df.swg[l * DF_FF + o];
          su_pf = g_df.swu[l * DF_FF + o];
        }
        if (p == p0) df_round8<DF_KT_H, true>(tg, tu, 2, q8, qs, DF_B, s_red, &gpre);
        else df_round8<DF_KT_H>(tg, tu, 2, q8, qs, DF_B, s_red);
        if (tid < DF_TPT) {
          int const row = (tid >> DF_RCS) & 15, c = tid & (DF_RC - 1);
          if (c < DF_B) {
            int const o = p * 16 + row;
            float const gg = df_rbf(df_sum4(s_red, 0, row, c) * sg_pf);
            float const uu = df_rbf(df_sum4(s_red, 1, row, c) * su_pf);
            float const sg = df_rbf(gg / (1.f + expf(-gg)));
            df_st2h(L + DF_L_ACT, c * DF_FF + o, df_tobf(sg * uu));
          }
        }
        __syncthreads();
        if (p == p0) DF_TSL(54);
      }
    }
    DF_TSL(50);
    df_barrier(++gen, die);
    DF_TSL(9);
    // ---- down + residual
    {
      float *const xa = lds;                                                         // one act row
      unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + DF_FF);      // DF_B x DF_FF
      unsigned char *const qs = q8 + DF_B * DF_FF;                                    // DF_B x DF_KT_FF
      float *const s_red = reinterpret_cast<float *>(qs + (DF_B <= 4 ? 256 : 512));
      static_assert(DF_B * DF_KT_FF <= (DF_B <= 4 ? 256 : 512), "down scale rows fit their reserve");
      (void)xa;
      int t0, t1;
      df_dsplit(DF_T_H, wid, W, t0, t1);
      if (t0 < t1) {  // workers without a tile skip the prologue
      DfPre8 dpre;
      {
        constexpr int NID = DF_B * DF_FF / 8;  // 8-value items
        static_assert(DF_B != 4 || NID == 256 * 8 + 256 * 7, "block 4: batches of 8 and 7 items per thread");
        unsigned short const *const xa16 = reinterpret_cast<unsigned short const *>(rv + L + DF_L_ACT);
        DfQ8Bh<DF_FF, DF_FF, 8> qa;
        qa.ld(xa16, DF_FF, DF_B, 0);
        int const tq = t0 < t1 ? t0 : t0 - 1;  // t0 >= 1 when empty
        df_pre8<DF_KT_FF, 1>(DfWd::at(g_df.wd, l, tq), DfWd::at(g_df.wd, l, tq), dpre);
        qa.fin(DF_B, 0, q8, qs);
        if constexpr (DF_B == 4) {
          DfQ8Bh<DF_FF, DF_FF, 7> qb;
          qb.ld(xa16, DF_FF, DF_B, 256 * 8);
          qb.fin(DF_B, 256 * 8, q8, qs);
        } else {
#pragma unroll 1
          for (int b0 = 256 * 8; b0 < NID; b0 += 256 * 8) {
            DfQ8Bh<DF_FF, DF_FF, 8> qb;
            qb.ld(xa16, DF_FF, DF_B, b0);
            qb.fin(DF_B, b0, q8, qs);
          }
        }
        __syncthreads();
      }
      DF_TSL(61);
      for (int t = t0; t < t1; t += 2) {
        int const nt = min(2, t1 - t);
        unsigned char const *const tb = DfWd::at(g_df.wd, l, t);
        float sd_pf = 0.f, x2_pf = 0.f;
        if (tid < nt * DF_TPT && (tid & (DF_RC - 1)) < DF_B) {
          int const o = (t + (tid >> (4 + DF_RCS))) * 16 + ((tid >> DF_RCS) & 15), c = tid & (DF_RC - 1);
          sd_pf = g_df.swd[l * DF_H + o];
          x2_pf = df_bf(reinterpret_cast<unsigned short const *>(rv + L + DF_L_X2)[c * DF_H + o]);
        }
        if (t == t0 && nt == 1) df_round8<DF_KT_FF, true>(tb, tb, 1, q8, qs, DF_B, s_red, &dpre);
        else df_round8<DF_KT_FF>(tb, DfWd::at(g_df.wd, l, t + nt - 1), nt, q8, qs, DF_B, s_red);
        if (tid < nt * DF_TPT) {
          int const j = tid >> (4 + DF_RCS), row = (tid >> DF_RCS) & 15, c = tid & (DF_RC - 1);
          if (c < DF_B) {
            int const o = (t + j) * 16 + row;
            float const y = df_rbf(df_sum4(s_red, j, row, c) * sd_pf);
            df_st2h(L + DF_L_X, c * DF_H + o, df_tobf(x2_pf + y));
          }
        }
        __syncthreads();
      }
      }
#if defined(MPK_DF_HALVES) && defined(MPK_DF_QKV_WARM)  // measured: the q/k/v stage gains less than down loses
      else if (l + 1 < DF_NL) {
        DfDev const dv = df_dev(wid, W);
        int const busy = DF_T_H / 2 < dv.wd ? DF_T_H / 2 : dv.wd;  // workers with a down tile on a device
        if (dv.lw >= busy) df_warm_qkv(l + 1, dv.dev, dv.lw - busy, dv.wd - busy);
      }
#endif
    }
    DF_TSL(60);
    df_barrier(++gen, die);
    DF_TSL(9);
  }

  // ---- LM head on block rows 1 .. DF_B - 1: per-worker argmax, the last arrival reduces and publishes
  {
    constexpr int RL = DF_B - 1;
    float *const xr = lds;
    unsigned char *const q8 = reinterpret_cast<unsigned char *>(lds + DF_B * DF_H);
    unsigned char *const qs = q8 + DF_B * DF_HP;
    float *const s_best = reinterpret_cast<float *>(qs + DF_QSP);  // [4 waves][16 cols][3]
    int const Lx = DF_V_L0 + (DF_NL - 1) * DF_L_SZ + DF_L_X;
    (void)xr;
    df_rmsq8_gh(reinterpret_cast<unsigned short const *>(rv + Lx) + DF_H, RL, g_df.nfin, q8, qs, df_red);
    DF_TSL(73);
    int t0, t1;
    df_dsplit(DF_T_V, wid, W, t0, t1);
    // per column (draft row): the argmax, and sum exp(v - best) over the same values for the draft's softmax
    float best = -3.0e38f, se = 0.f;
    int bidx = 0x7fffffff;
    for (int tb = t0 + wave; tb < t1; tb += 12) {  // a wave's tiles tb, tb + 4, tb + 8
      int const nv = tb + 8 < t1 ? 3 : (tb + 4 < t1 ? 2 : 1);
      unsigned char const *wr[3], *sr[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        int const tt = j < nv ? tb + 4 * j : tb;
        wr[j] = DfWlm::at(g_df.wlm, 0, tt);
        sr[j] = DfSlm::at(g_df.slm, 0, tt);
      }
      df_f32x4 acc[3];
      if (nv == 3) df_m4gemv<DF_KT_H, 3, 4>(wr, sr, q8, DF_HP, qs, DF_KT_H, RL, acc);
      else if (nv == 2) df_m4gemv<DF_KT_H, 2, 4>(wr, sr, q8, DF_HP, qs, DF_KT_H, RL, acc);
      else df_m4gemv<DF_KT_H, 1, 4>(wr, sr, q8, DF_HP, qs, DF_KT_H, RL, acc);
#pragma unroll
      for (int j = 0; j < 3; ++j)
        if (j < nv)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int const o = (tb + 4 * j) * 16 + gq * 4 + i;
            float const v = acc[j][i];
            float const nb = fmaxf(best, v);
            se = se * __expf(best - nb) + __expf(v - nb);
            if (v > best || (v == best && o < bidx)) {
              best = v;
              bidx = o;
            }
          }
    }
#pragma unroll
    for (int m = 16; m <= 32; m <<= 1) {  // lanes of the same column
      float const ob = __shfl_xor(best, m, 64);
      float const os = __shfl_xor(se, m, 64);
      int const oi = __shfl_xor(bidx, m, 64);
      float const nb = fmaxf(best, ob);
      se = se * __expf(best - nb) + os * __expf(ob - nb);
      if (ob > best || (ob == best && oi < bidx)) {
        best = ob;
        bidx = oi;
      }
    }
    if (lane < 16) {
      s_best[(wave * 16 + lane) * 3] = best;
      s_best[(wave * 16 + lane) * 3 + 1] = __int_as_float(bidx);
      s_best[(wave * 16 + lane) * 3 + 2] = se;
    }
    __syncthreads();
    if (tid < RL) {
      float bv = -3.0e38f, bs = 0.f;
      int bi = 0x7fffffff;
      for (int w4 = 0; w4 < 4; ++w4) {
        float const v = s_best[(w4 * 16 + tid) * 3];
        int const i = __float_as_int(s_best[(w4 * 16 + tid) * 3 + 1]);
        float const sv = s_best[(w4 * 16 + tid) * 3 + 2];
        float const nb = fmaxf(bv, v);
        bs = bs * __expf(bv - nb) + sv * __expf(v - nb);
        if (v > bv || (v == bv && i < bi)) {
          bv = v;
          bi = i;
        }
      }
      df_st2f(DF_V_LP + tid * 256 + wid, bv);
      df_st2(DF_V_LP + DF_LPI + tid * 256 + wid, (unsigned)bi);
      df_st2f(DF_V_LP + 2 * DF_LPI + tid * 256 + wid, bs);
    }
    DF_TSL(70);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      int const c = __hip_atomic_fetch_add(g_aid_nc_rep[0] + DF_NC_INTS + 9 * 64, 1, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
      df_red[30] = __int_as_float(c % W == W - 1 ? 1 : 0);
    }
    __syncthreads();
    if (__float_as_int(df_red[30]) == 1) {
      for (int r = 0; r < RL; ++r) {
        int const pi = tid < W ? (int)df_ld_sys_u32(g_df.vec + DF_V_LP + DF_LPI + r * 256 + tid) : 0x7fffffff;
        float const pvv = tid < W ? df_ld_sys_f32(g_df.vec + DF_V_LP + r * 256 + tid) : -3.0e38f;
        float const psv = tid < W ? df_ld_sys_f32(g_df.vec + DF_V_LP + 2 * DF_LPI + r * 256 + tid) : 0.f;
        float M;
        int const fi = df_block_argmax(pvv, pi, df_red, &M);
        // the draft's softmax probability: 1 / sum over workers of s_w exp(m_w - M)
        float sw[1] = {tid < W ? psv * __expf(pvv - M) : 0.f};
        df_block_sum_n<1>(sw, df_red);
        if (tid == 0) {
          __hip_atomic_store(&g_df.draft_out[r], (long long)fi, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          __hip_atomic_store(&g_df.draft_out[DF_CONF_SLOT + r], (long long)__float_as_uint(1.f / sw[0]),
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
      if (tid == 0) __hip_atomic_store(&g_df.draft_out[8], (long long)s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  DF_TSL(75);
#ifdef MPK_DF_TS
  if (wid == MPK_DF_TS_W && tid == 0 && kind == 1 && s >= prompt_len + 30 && atomicAdd(&g_df_tsn, 1) < 2) {
    printf("[DFTS] s=%d R=%d P=%d n=%d dt(10ns):", s, R, P, dfn);
    for (int q = 1; q < dfn; ++q) printf(" %d:%llu", (int)dflb[q], dfts[q] - dfts[q - 1]);
    printf(" total %llu\n", dfts[dfn - 1] - dfts[0]);
  }
#endif
}

}  // namespace kernel
