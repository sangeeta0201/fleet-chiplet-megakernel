"""lmbf16_kernel_patch.py: MPK_LM_BF16 (BF16 LM head) in gang_rmsnorm_linear_mxfp4_bias_argmax_mi300.cuh.
ASCII-only anchors and text (the file's own comments use box-drawing characters). Idempotent."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
if "MPK_LM_BF16" in s:
    print("already patched")
    sys.exit(0)


def sub1(old, new):
    global s
    assert s.count(old) == 1, f"anchor count {s.count(old)}: {old[:60]!r}"
    s = s.replace(old, new)


GUARD = """#if defined(MPK_LM_BF16) && \\
    (defined(MPK_LM_HEAD_GROUP_PIPELINE) || defined(MPK_LM_PF0) || defined(MPK_LMT_FQ))
#error "MPK_LM_BF16 streams BF16 weight records into registers; it replaces the MXFP4 group pipeline, MPK_LM_PF0 and MPK_LMT_FQ"
#endif
"""
sub1("#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && !defined(MPK_LM_HEAD_KMAJOR)",
     GUARD + "#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && !defined(MPK_LM_HEAD_KMAJOR)")

sub1("#ifndef MPK_LMT_FQ\n    _gang_multirow_fp8_quant<",
     "#if !defined(MPK_LMT_FQ) && !defined(MPK_LM_BF16)\n    _gang_multirow_fp8_quant<")

BLOCK = r"""#ifdef MPK_LM_BF16
    // -- BF16 LM head -------------------------------------------------------
    // The host shuffles the BF16 weight into K-major records: record (wg, tile t,
    // step s) holds the 64 lanes' A fragments of rows wg * 64 + 16 t + (lane & 15),
    // K = 32 s + 8 (lane >> 4) .. +8, so a wave reads 1 KiB contiguous per step.
    // Wave t streams tile t through v_mfma_f32_16x16x32_bf16, whose D layout is
    // the scaled MFMA's (lane (g, col): rows 4 g + i of token col). The ring holds
    // LMB_D steps; its last LMB_D refills fetch the next group's first steps, so
    // LMB_D must divide the step count for the slots to line up across groups.
    {
      constexpr int LMB_K = REDUCTION_SIZE / 32;
      constexpr int LMB_D = 23;
      static_assert(LMB_K % LMB_D == 0, "MPK_LM_BF16: LMB_D must divide K / 32");
      static_assert(OUTPUT_PER_WG == 64 && NUM_WAVES == 4,
                    "MPK_LM_BF16: one 16-row tile per wave");
      typedef unsigned int lmb_u4 __attribute__((ext_vector_type(4)));
      typedef __bf16 lmb_bf8 __attribute__((ext_vector_type(8)));
      typedef float lmb_f4 __attribute__((ext_vector_type(4)));
      typedef __attribute__((address_space(1))) lmb_u4 const lmb_gu4;
      typedef __attribute__((address_space(1))) unsigned long long const lmb_gb8;
      constexpr size_t LMB_WG_BYTES = (size_t)OUTPUT_PER_WG * REDUCTION_SIZE * 2;
#if defined(MPK_LM_NORM_LDS) || (defined(MPK_LMT_NORM) && !defined(MPK_LMT_GLB))
      typedef __attribute__((address_space(3))) unsigned short const lmb_xs;
      typedef __attribute__((address_space(3))) lmb_u4 const volatile lmb_xu4;
#else
      // volatile on global memory would be a system-scope load: plain here
      typedef __attribute__((address_space(1))) unsigned short const lmb_xs;
      typedef __attribute__((address_space(1))) lmb_u4 const lmb_xu4;
#endif
#if defined(MPK_LM_NORM_LDS)
      lmb_xs *const lmb_x0 = (lmb_xs *)((uint8_t *)_rnlm_smem + LM_NORM_OFF);
#elif defined(MPK_LMT_NORM) && !defined(MPK_LMT_GLB)
      lmb_xs *const lmb_x0 = (lmb_xs *)lmt_rows;
#else
      lmb_xs *const lmb_x0 =
          (lmb_xs *)((unsigned short const *)norm_output_ptr + lmnorm_aid_off);
#endif
      lmb_xs *const lmb_x =
          lmb_x0 + (size_t)(tok_active ? tok_row : bblk * MFMA_N) * REDUCTION_SIZE + g * 8;
      auto lmb_rec = [&](int wgi) -> lmb_gu4 * {
        return (lmb_gu4 *)(W + (size_t)wgi * LMB_WG_BYTES +
                           (size_t)warp_id * (LMB_K * 1024)) +
               lane_id;
      };
      lmb_u4 lmb_w[LMB_D];
      int lmb_wg = worker_rank;
      {
        lmb_gu4 *const r0 = lmb_rec(lmb_wg < n_wgs_per_xcd ? lmb_wg : 0);
#pragma unroll
        for (int s = 0; s < LMB_D; ++s) lmb_w[s] = __builtin_nontemporal_load(r0 + s * 64);
      }
      for (; lmb_wg < n_wgs_per_xcd; lmb_wg += workers_per_xcd) {
        int const nwg = lmb_wg + workers_per_xcd;
        lmb_gu4 *const r = lmb_rec(lmb_wg);
        lmb_gu4 *const rn = lmb_rec(nwg < n_wgs_per_xcd ? nwg : lmb_wg);
        // This tile's 4 bias values, loaded before the refills so their wait
        // does not drain the ring (vmcnt retires in issue order).
        unsigned long long const lmb_bias =
            *(lmb_gb8 *)(d_bias + lmb_wg * OUTPUT_PER_WG + warp_id * 16 + g * 4);
        lmb_f4 acc = {0.f, 0.f, 0.f, 0.f};
        // The activation slice is re-read from LDS one step ahead, as volatile
        // loads: it is the same in every group, and hoisted out of the group loop
        // it pins 92 x 4 registers, which pushes the weight ring into AGPRs and
        // turns its waits into vmcnt(0).
        lmb_u4 lmb_xc = *(lmb_xu4 *)lmb_x;
#pragma unroll
        for (int s = 0; s < LMB_K; ++s) {
          lmb_u4 const w = lmb_w[s % LMB_D];
          if (s + LMB_D < LMB_K)
            lmb_w[s % LMB_D] = __builtin_nontemporal_load(r + (s + LMB_D) * 64);
          else
            lmb_w[s % LMB_D] = __builtin_nontemporal_load(rn + (s + LMB_D - LMB_K) * 64);
          lmb_u4 lmb_xn = lmb_xc;
          if (s + 1 < LMB_K) lmb_xn = *(lmb_xu4 *)(lmb_x + (s + 1) * 32);
          lmb_bf8 a, b;
          __builtin_memcpy(&a, &w, 16);
          __builtin_memcpy(&b, &lmb_xc, 16);
          acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0);
          lmb_xc = lmb_xn;
        }
        if (tok_active) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            int const rel_idx = warp_id * 16 + g * 4 + i;
            int const abs_idx = partition_start + lmb_wg * OUTPUT_PER_WG + rel_idx;
            unsigned const bt = (unsigned)((lmb_bias >> (16 * i)) & 0xffffu) << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            float const val = acc[i] + bv;
            if (val > thread_max) {
              thread_max = val;
              thread_max_idx = abs_idx;
            }
            if (logits_out_ptr != nullptr) {
              reinterpret_cast<float *>(
                  logits_out_ptr)[(long long)(step + tok_row) * output_stride + abs_idx] = val;
            }
          }
        }
      }
    }
#else
"""
sub1("    int group_iteration = 0;\n    for (int wg_idx = worker_rank; wg_idx < n_wgs_per_xcd;",
     BLOCK + "    int group_iteration = 0;\n    for (int wg_idx = worker_rank; wg_idx < n_wgs_per_xcd;")
sub1("    } // end tile loop\n", "    } // end tile loop\n#endif  // MPK_LM_BF16\n")
open(p, "w", encoding="utf-8").write(s)
print("patched kernel")
