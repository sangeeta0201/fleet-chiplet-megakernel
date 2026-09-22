/* gfx1250 fused RMSNorm + MXFP4 gang linear + bias.
 *
 * Port of gang_rmsnorm_linear_mxfp4_bias_kernel from
 * tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh (2962 lines).
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A FRACTION OF THE SIZE OF THE gfx950 FILE
 * ---------------------------------------------------------------------------
 *
 * The mi300 file is large because it carries several hand-scheduled MFMA
 * pipelines (depth-4 N-parallel, depth-4 K-parallel, plus variants in the
 * mulsumadd-fused sibling), each spelling out its own register rotation and
 * prefetch schedule. Those depths exist to keep gfx950's AGPR-backed MFMA
 * pipe fed.
 *
 * gfx1250 has no AGPRs, and the measured behaviour on this target is that
 * deep rotations backfire: the compiler emits ~26 v_dual_mov_b32 per iteration
 * shuffling buffers, costing more VALU than the latency they hide (documented
 * in gang_moe_linear_mxfp4_mi450.cuh). So every pipeline collapses onto one
 * validated depth-2 helper, mx_k_loop_f4xf8 in mx_pipeline_mi450.cuh, and what
 * remains here is tile dispatch plus the two epilogues, which genuinely differ.
 *
 * This is a deliberate behavioural change from gfx950, not an oversight.
 *
 * ---------------------------------------------------------------------------
 * WHAT HAD TO BE RE-DERIVED, NOT RESCALED
 * ---------------------------------------------------------------------------
 *
 * 8 wave32s per 256-thread block, not 4 wave64s. Three consequences:
 *
 *  1. NUM_WAVES 4 -> 8.
 *  2. THE gfx950 BRANCH CONDITION SILENTLY REROUTES SHAPES. mi300 picks
 *     N-parallel when OUTPUT_PER_WG >= 64 and K-parallel otherwise, which at 4
 *     waves means "N-parallel iff there are at least as many 16-row tiles as
 *     waves". OUTPUT_PER_WG=64 -- the actual QKV shape -- is 4 tiles: exactly
 *     N-parallel at 4 waves, and one wave short of it at 8. Transcribing the
 *     condition would have dropped Fleet's main shape into a K-parallel path
 *     whose gfx950 form assumes a single output tile. Caught by a
 *     static_assert on the first instantiation, which is the only reason it is
 *     not a silent wrong answer.
 *
 *     So the split here is 2D rather than either/or: N_GROUPS waves each take
 *     a distinct output tile, and WAVES_PER_TILE waves cooperate on the K
 *     extent of that tile. gfx950's two modes are the two corners
 *     (WAVES_PER_TILE==1 is pure N-parallel; N_GROUPS==1 is pure K-parallel),
 *     and 64-wide QKV lands in the middle at 4x2, which neither corner covers.
 *  3. THE CROSS-WAVE REDUCTION BUFFER DOUBLES. Each K-cooperating wave holds a
 *     partial that must be summed through LDS. Sizing that from the literal 4
 *     would let waves 4..7 overwrite waves 0..3 -- silently, and with a
 *     plausible result, since the sum would simply be missing half its terms.
 *
 * The accumulator geometry is also different in kind, not just in size: gfx950
 * MFMA gives each lane 4 rows of one column (acc[i] = C[g*4+i][col]); gfx1250
 * WMMA gives each lane 8 CONSECUTIVE rows of one column
 * (acc[i] = C[8*(lane/16)+i][col]). The epilogues below use mx_acc_row() and
 * mx_acc_col() rather than open-coding either layout.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_pipeline_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_quant_mi450.cuh"
// gang_rmsnorm_detail::rmsnorm_inline_amd -- the wave-reduction prologue, which
// was ported in place rather than duplicated here, so this file depends on the
// mi300 header for it. Include it explicitly: without this the header only
// compiles when something upstream happens to have included it first, which is
// exactly what happened -- it built fine in its own test and then failed with
// "use of undeclared identifier 'gang_rmsnorm_detail'" the moment it was added
// to task_header.cuh, where the mi300 include necessarily comes later (it must
// follow the MPK_LINEAR_KERNEL #define that selects the WMMA GEMM path).
#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

namespace kernel {
namespace mi450 {

// Dynamic LDS the kernel requires, in bytes.
//
// OUTSIDE the arch guard on purpose: the host has to size the launch, and on
// the host pass __gfx1250__ is not defined. A device-only version compiles
// fine and then fails at the call site with "no member named
// mx_rnlm_smem_bytes", which is at least loud -- but callers would work around
// it by recomputing the expression by hand, and that is exactly the
// duplication this helper exists to prevent.
//
// Callers must use this rather than recomputing: the reduction buffer is
// indexed by absolute warp_id, so it is NUM_WAVES*16 floats even in
// configurations where only two waves per tile actually cooperate. Undersizing
// it runs the high-numbered waves off the end of the allocation, which under
// FFM reads back as zeros -- a plausible-looking wrong answer, not a fault.
//
// REDUCTION_SIZE % 128 == 0 (enforced in the kernel) makes REDUCTION_SIZE/32 a
// multiple of 4, so s_reduce lands 4-byte aligned without explicit padding.
template <int REDUCTION_SIZE>
__host__ __device__ constexpr int mx_rnlm_smem_bytes() {
  return REDUCTION_SIZE                          // s_tok_fp8
         + REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK // s_tok_scales
         + 8 * 16 * (int)sizeof(float);          // s_reduce (NUM_WAVES * 16)
}

} // namespace mi450

#if defined(MIRAGE_ARCH_GFX1250)

namespace mi450 {

__device__ __forceinline__ unsigned short mx_float_to_bf16(float f) {
  unsigned u;
  __builtin_memcpy(&u, &f, 4);
  // Round-to-nearest-even, matching _gang_float_to_bf16 on gfx950. Truncation
  // here would bias every output low and show up as a slow accuracy drift
  // rather than a test failure.
  unsigned rounded = u + 0x7FFF + ((u >> 16) & 1);
  return (unsigned short)(rounded >> 16);
}

} // namespace mi450

// Fused RMSNorm + MXFP4 Gang Linear + Bias.
//
// Template params match the mi300 kernel exactly so the dispatch layer can
// swap implementations without touching call sites:
//   BATCH_SIZE        - max batch size (usually 1 for decode)
//   OUTPUT_PER_WG     - output rows per workgroup (e.g. 64)
//   REDUCTION_SIZE    - reduction dimension, padded (e.g. 3072)
//   ACTUAL_HIDDEN_DIM - unpadded hidden size for the RMSNorm divisor (2880)
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __noinline__ void gang_rmsnorm_linear_mxfp4_bias_kernel_mi450(
    void const *norm_input_ptr,  // [batch, REDUCTION_SIZE] bf16
    void const *norm_weight_ptr, // [REDUCTION_SIZE] bf16
    void *norm_output_ptr,       // [batch, REDUCTION_SIZE] bf16 scratch
    void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP4
    void const *bias_ptr,        // [1, output_size_per_xcd] bf16 (partitioned)
    void *output_ptr,            // [batch, output_stride] bf16 (partitioned)
    int num_active_tokens,
    int n_wgs_per_xcd,
    int output_stride,
    int tile_idx) {
  using namespace mi450;

  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % MX_K_PER_TILE == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 WMMA");

  // ── Weight layout constants (byte-identical to gfx950) ──────────────────
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  constexpr int K_ITERS = REDUCTION_SIZE / MX_K_PER_TILE;

  // 8 wave32s, not 4 wave64s.
  constexpr int NUM_WAVES = 8;
  constexpr int N_TILES = OUTPUT_PER_WG / 16;

  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;
  // Cross-wave reduction scratch, only used when WAVES_PER_TILE > 1. Indexed
  // by ABSOLUTE warp_id (not rank within the tile group) so every wave has a
  // private 16-float slot: NUM_WAVES * 16 floats. Sized from NUM_WAVES (8),
  // never a literal 4 -- see consequence 3 in the header. Also see
  // mx_rnlm_smem_bytes() below, which is what callers must allocate.
  float *s_reduce = (float *)(s_tok_scales + NUM_BLOCKS_32);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 5; // wave 0..7  (was tid >> 6)
  int const lane_id = tid & 31; // lane 0..31 (was tid & 63)
  int const col = mx_operand_mn(lane_id);

  // ── Step 1: redundant RMSNorm ───────────────────────────────────────────
  // Every worker computes the same RMSNorm and writes norm_output_ptr. The
  // helper reduces through mirage::arch::wave_reduce_sum, which takes its
  // butterfly width from the arch traits, so it is already wave32-correct.
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  for (int b = 0; b < batch_count; b++) {
    unsigned short const *row_in =
        (unsigned short const *)norm_input_ptr + b * REDUCTION_SIZE;
    unsigned short *row_out =
        (unsigned short *)norm_output_ptr + b * REDUCTION_SIZE;
    gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, norm_weight_ptr, row_out);
  }

  // ── Tile dispatch ───────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  if (tok_idx >= batch_count) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 2: quantize the normalized row to FP8 in LDS ───────────────────
  unsigned short const *input_row =
      (unsigned short const *)norm_output_ptr + tok_idx * REDUCTION_SIZE;
  mx_quant_fp8<REDUCTION_SIZE>(input_row, s_tok_fp8, s_tok_scales);
  // mx_quant_fp8 is block-cooperative and the WMMA below reads every byte of
  // it from every wave, so this barrier is load-bearing.
  __syncthreads();

  mx_global_u8 *g_data = mx_to_global(wg_data);
  mx_global_u8 *g_scales = mx_to_global(wg_scales);

  // ── Step 3: WMMA FP4(weights) x FP8(tokens) ─────────────────────────────
  //
  // 2D wave split. N_GROUPS waves take distinct output tiles; within a group,
  // WAVES_PER_TILE waves split the K extent and their partials are summed
  // through LDS. See consequence 2 in the header for why neither of gfx950's
  // two corners covers OUTPUT_PER_WG=64 at 8 waves.
  constexpr int N_GROUPS = N_TILES < NUM_WAVES ? N_TILES : NUM_WAVES;
  constexpr int WAVES_PER_TILE = NUM_WAVES / N_GROUPS;
  constexpr int TILES_PER_GROUP = N_TILES / N_GROUPS;

  static_assert(NUM_WAVES % N_GROUPS == 0,
                "wave count must divide evenly into output-tile groups");
  static_assert(N_TILES % N_GROUPS == 0,
                "output tiles must divide evenly across tile groups");
  static_assert(K_ITERS % WAVES_PER_TILE == 0,
                "K tiles must divide evenly across the waves cooperating on "
                "one output tile");

  constexpr int K_PER_WAVE = REDUCTION_SIZE / WAVES_PER_TILE;

  // Which output-tile group this wave belongs to, and its rank within it.
  int const grp = warp_id / WAVES_PER_TILE;
  int const krank = warp_id % WAVES_PER_TILE;

  for (int tile_iter = 0; tile_iter < TILES_PER_GROUP; tile_iter++) {
    int wave_tile = grp + tile_iter * N_GROUPS;

    // The weight row THIS LANE SUPPLIES -- not the row its accumulator ends up
    // holding. The epilogue uses mx_acc_row() for that.
    int const w_row = wave_tile * 16 + col;

    mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    if constexpr (WAVES_PER_TILE == 1) {
      // Whole K extent: the validated depth-2 pipelined helper.
      acc = mx_k_loop_f4xf8<REDUCTION_SIZE>(
          g_data, g_scales, w_row, s_tok_fp8, s_tok_scales, acc);
    } else {
      // A K sub-range. Deliberately NOT mx_k_loop_f4xf8: that helper derives
      // its scale-block stride and tail handling from its template parameter,
      // which here would be the slice length while the row pitch is still
      // REDUCTION_SIZE. Passing the slice length would silently walk rows at
      // the wrong stride. An unpipelined loop over the slice is correct and
      // costs little, since WAVES_PER_TILE>1 shapes are the small ones.
      int const k_off = krank * K_PER_WAVE;
#pragma unroll 1
      for (int kt = k_off; kt < k_off + K_PER_WAVE; kt += MX_K_PER_TILE) {
        mx_i32x16_t a =
            mx_load_operand_fp4_global(g_data, w_row, kt, REDUCTION_SIZE);
        mx_i32x16_t b = mx_load_operand_fp8(s_tok_fp8, 0, kt, REDUCTION_SIZE);
        unsigned int sa =
            mx_pack_scales_global(g_scales + (size_t)w_row * NUM_BLOCKS_32 +
                                  kt / MX_K_PER_SCALE_BLOCK);
        unsigned int sb =
            mx_pack_scales(s_tok_scales + kt / MX_K_PER_SCALE_BLOCK);
        acc = _gang_wmma_f4xf8(a, b, acc, (int)sa, (int)sb);
      }
    }

    // Only column 0 is a real token for BATCH_SIZE=1; lanes 0 and 16 carry all
    // 16 output rows of the tile between them.
    if constexpr (WAVES_PER_TILE == 1) {
      if (col == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int out_n =
              wg_idx * OUTPUT_PER_WG + wave_tile * 16 + mx_acc_row(lane_id, i);
          unsigned bt = (unsigned)d_bias[out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          d_output[tok_idx * output_stride + out_n] =
              mx_float_to_bf16(acc[i] + bv);
        }
      }
    } else {
      // Sum this tile's K partials across its cooperating waves. Indexed by
      // absolute warp_id, so the buffer must be NUM_WAVES * 16 floats -- not
      // WAVES_PER_TILE * 16, and never the gfx950 literal.
      if (col == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          s_reduce[warp_id * 16 + mx_acc_row(lane_id, i)] = acc[i];
        }
      }
      // Every wave in the group must have landed its partial before any wave
      // reads them, and the next tile_iter overwrites these same slots.
      //
      // FFM CANNOT VERIFY THIS BARRIER. Deleting it was run as a mutant and
      // ALL FOUR test shapes still passed bit-exactly, including the pure
      // K-parallel corner where 8 waves write and read s_reduce with nothing
      // between them. FFM keeps the waves lockstep enough that the race never
      // resolves badly. Do not "simplify" this away on the strength of a green
      // suite -- the suite is structurally incapable of noticing.
      __syncthreads();

      // 16 outputs per tile, N_GROUPS tiles in flight: threads 0..N_GROUPS*16
      // each own one output.
      if (tid < N_GROUPS * 16) {
        int const t_grp = tid / 16;
        int const t_row = tid % 16;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WAVES_PER_TILE; w++) {
          v += s_reduce[(t_grp * WAVES_PER_TILE + w) * 16 + t_row];
        }
        int out_n =
            wg_idx * OUTPUT_PER_WG + (t_grp + tile_iter * N_GROUPS) * 16 + t_row;
        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);
        d_output[tok_idx * output_stride + out_n] = mx_float_to_bf16(v + bv);
      }
      __syncthreads();
    }
  }

  __syncthreads();
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace kernel
