#!/usr/bin/env python3
"""Give drive_phase7 an output that can actually disagree with itself.

Everything this harness printed was zero in every configuration, which quietly
disarmed the correctness gate: a barrier that released early read the wrong
layer's data and still produced zero, so comparing outputs proved nothing.
Three separate reasons, all fixed here.

1. No weights. The K-parallel arm takes its MFMA B operand from LDS at
   oproj_lds_w_off(), which the fused caller fills during Phase 6. This harness
   invokes Phase 7 on its own and never staged anything, so the GEMM multiplied
   by an empty tile. Fixed by copying the tile in from HBM before the call.

2. Underflowed scales. The whole weight buffer was memset to 0x11, including
   each group's E8M0 scale half, where 0x11 is 2^-110 -- every product
   underflows. Fixed by filling the two halves separately.

3. A flat row. With every weight byte equal, all 184 blocks produce identical
   output columns, and RMSNorm divides a flat row by its own RMS to get exactly
   1.0 everywhere regardless of magnitude -- erasing the layer dependence. Fixed
   by giving each column its own layer-dependent residual.

With all three, the two barriers under test become observable:

  * attn_slice_release -- out[col] sums the whole 4096-element reduction, i.e.
    all eight XCDs' attn_out slices, and each slice carries its own
    (layer, xcd) value. A slice read before its producer published shows up in
    attn_proj_out.
  * the hierarchical barrier -- RMSNorm reads the full 2880-element row, which
    spans all eight XCDs' columns. A column read before its XCD stored it shows
    up in rmsnorm_out.

Magnitudes are picked so staleness moves the stored bf16 rather than rounding
away: the reduction lands near 20 (not 4096), where bf16 resolves 0.125, and one
stale slice moves it by 1.0.

Run on mi355x-thor-2:  python3 patch_lds_stage.py
"""
import shutil
import sys

DRV = "/home/schowdha/fleet-chiplet-megakernel/drive_phase7.cu"
SUFFIX = ".pre-ldsstage"

# ?? 1. LDS geometry ????????????????????????????????????????????????????????
C_OLD = """#define WG_BYTES (WG_DATA_BYTES + WG_SCALE_BYTES)
"""
C_NEW = """#define WG_BYTES (WG_DATA_BYTES + WG_SCALE_BYTES)

// Byte offset of the scale half within the staged LDS tile. Must match the
// kernel's OPROJ_LDS_DATA_PAD exactly, or the MFMA reads its scales from the
// wrong place; both round WG_DATA_BYTES up to a whole 256-thread x 16-byte
// pass the same way.
#define LDS_W_DATA_PAD (((WG_DATA_BYTES / 16 + 255) / 256) * 256 * 16)
"""

# ?? 2. address the same dynamic LDS the callee does ????????????????????????
D_OLD = """  int const xcd = drv_get_xcd();
  int const tid = threadIdx.x;
"""
D_NEW = """  int const xcd = drv_get_xcd();
  int const tid = threadIdx.x;

  // The same dynamic LDS the callee declares, so the tile staged below is the
  // tile its MFMA reads. The offset comes from the kernel's own constexpr
  // rather than a copied literal.
  extern __shared__ char _lm_smem[];
  constexpr int LDS_W_OFF =
      kernel::oproj_lds_w_off(OPROJ_BATCH, OPROJ_REDUCTION);
"""

# ?? 3. weights that survive the multiply ???????????????????????????????????
W_OLD = """  HIP_OK(hipMemset(weight, 0x11, weight_bytes));
"""
W_NEW = """  // Two fills, because a group's data and scale halves are different formats.
  // e2m1 0x2 is 1.0, so 0x22 is a pair of ones. The scale is E8M0 biased at
  // 127, and 0x77 is 2^-8, chosen so a 4096-long reduction of ones lands near
  // 20 instead of 4096: bf16 resolves 0.125 there, and one XCD's slice going
  // stale moves the result by 1.0, so it survives the store instead of
  // rounding away.
  //
  // The single 0x11 fill this replaces left every scale at 2^-110, which
  // underflowed all 4096 products and made the harness compute zeros.
  for (int g = 0; g < NXCD * tiles; ++g) {
    char *wg = (char *)weight + (size_t)g * WG_BYTES;
    HIP_OK(hipMemset(wg, 0x22, WG_DATA_BYTES));
    HIP_OK(hipMemset(wg + WG_DATA_BYTES, 0x77, WG_SCALE_BYTES));
  }
"""

# ?? 4. a producer value that identifies its layer and XCD ??????????????????
P_OLD = """      // 256 threads x 1 dword = 512 bf16 = this XCD's slice.
      drv_st_wt_u32((unsigned int *)my_slice + tid,
                    (unsigned)(layer * 2654435761u + xcd));
"""
P_NEW = """      // 256 threads x 1 dword = 512 bf16 = this XCD's slice.
      //
      // A well-conditioned bf16 rather than a hash reinterpreted as two
      // arbitrary exponents. The value depends on both the layer and the XCD,
      // which is what lets a slice read too early -- still carrying the
      // previous layer's value -- change the O-proj result. Every term is a
      // negative power of two, so the stored bf16 is the intended value rather
      // than a rounding of it.
      float const fv = 1.0f + 0.5f * (float)(layer & 3) + 0.0625f * (float)xcd;
      unsigned int const bf = __float_as_uint(fv) >> 16;
      drv_st_wt_u32((unsigned int *)my_slice + tid, bf | (bf << 16));
"""

# ?? 5. stage the tile, and break up the flat row ???????????????????????????
S_OLD = """        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

    kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<
"""
S_NEW = """        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

    // ---- stand-in for Phase 6's buffer_load_lds weight DMA --------------
    //
    // Without this the MFMA's B operand is an unpopulated LDS tile and the
    // whole GEMM is zero, whatever the barriers do.
    //
    // Re-staged every layer rather than once before the loop: the kernel's own
    // scratch lives below oproj_lds_w_off() and should not reach this region,
    // but a harness meant to catch ordering bugs should not rest on that.
    {
      unsigned char const *w_src = my_weight + (size_t)local * WG_BYTES;
      unsigned char *w_dst = (unsigned char *)_lm_smem + LDS_W_OFF;
      for (int i = tid * 16; i < WG_DATA_BYTES; i += NTHREADS * 16) {
        __builtin_memcpy(w_dst + i, w_src + i, 16);
      }
      unsigned char const *s_src = w_src + WG_DATA_BYTES;
      unsigned char *s_dst = w_dst + LDS_W_DATA_PAD;
      for (int i = tid * 16; i < WG_SCALE_BYTES; i += NTHREADS * 16) {
        __builtin_memcpy(s_dst + i, s_src + i, 16);
      }
    }

    // ---- give RMSNorm a row that is not flat ----------------------------
    //
    // Every weight byte is identical, so all 16 of a block's output columns
    // come out equal, and so does every other block's. RMSNorm divides a flat
    // row by its own RMS and returns exactly 1.0 in every column whatever the
    // magnitude, which would erase the layer dependence the check needs.
    // A per-column residual makes the row non-flat and layer-dependent, so a
    // column read before its XCD stored it shows up in rmsnorm_out.
    //
    // Each block writes only the 16 entries it goes on to read itself, so the
    // __syncthreads below is the whole of the ordering required.
    if (tid < OPROJ_OUTPUT_PER_WG) {
      int const gcol = tile_idx * OPROJ_OUTPUT_PER_WG + tid;
      float const rv = 0.5f * (float)((gcol + layer) & 15);
      my_residual[local * OPROJ_OUTPUT_PER_WG + tid] =
          (unsigned short)(__float_as_uint(rv) >> 16);
    }
    __syncthreads();

    kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<
"""

EDITS = [("lds geometry", C_OLD, C_NEW),
         ("lds declaration", D_OLD, D_NEW),
         ("weight fill", W_OLD, W_NEW),
         ("producer value", P_OLD, P_NEW),
         ("stage + residual", S_OLD, S_NEW)]

with open(DRV, "r", encoding="utf-8", newline="") as fh:
    text = fh.read()

for name, old, _ in EDITS:
    n = text.count(old)
    if n != 1:
        sys.exit("anchor %r matched %d times, expected 1" % (name, n))

shutil.copyfile(DRV, DRV + SUFFIX)
for name, old, new in EDITS:
    text = text.replace(old, new)
    print("  ok  %s" % name)
with open(DRV, "w", encoding="utf-8", newline="") as fh:
    fh.write(text)

print("patched %s (backup at %s%s)" % (DRV, DRV, SUFFIX))
