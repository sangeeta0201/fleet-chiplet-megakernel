#!/usr/bin/env python3
"""Apply the MPK_OPROJ_KSPLIT_CEIL probe to the tree given as argv[1].

Idempotent: re-running is a no-op. Written as a patch script rather than a
file copy because the container tree carries uncommitted work the host tree
does not (MPK_W13_T0_COUNTED_HANDOFF, MPK_EP_WAIT_AT_USE), and overwriting
persistent_kernel.py / env_common.sh would silently revert it.
"""
import sys
import os

ROOT = sys.argv[1]

OPROJ = "include/mirage/persistent_kernel/tasks/mi300/gang_oproj_router_fused_mi300.cuh"
FULLL = "include/mirage/persistent_kernel/tasks/mi300/gang_mla_full_layer_fused_mi300.cuh"
PK = "python/mirage/mpk/persistent_kernel.py"
ENVC = "demo/glm5/env_common.sh"

EDITS = [
    (OPROJ,
     """      int *my_flag = &wuv_barrier[xcd_id * HIER_STRIDE];
      MPK_WS_WAIT_BEGIN(767, wuv_expected);""",
     """#if MPK_OPROJ_KSPLIT_CEIL >= 1
      // -- CEILING PROBE, WRONG OUTPUT BY CONSTRUCTION --------------------
      // Prices the o_proj K-split's first half: under a row-sharded o_proj
      // each XCD contracts only over the heads its OWN W_UV wrote, so this
      // W_UV -> o_proj rendezvous has nothing left to order and disappears.
      // Here the wait is simply deleted, so o_proj may read a peer XCD's
      // stale v slice. Timing only; never a correctness arm.
      (void)wuv_expected;
#else
      int *my_flag = &wuv_barrier[xcd_id * HIER_STRIDE];
      MPK_WS_WAIT_BEGIN(767, wuv_expected);"""),

    (OPROJ,
     """          if (heal) {
            st_flag_u32((void *)my_flag, (unsigned)wuv_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");""",
     """          if (heal) {
            st_flag_u32((void *)my_flag, (unsigned)wuv_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
#endif // MPK_OPROJ_KSPLIT_CEIL
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");"""),

    (FULLL,
     """    if (tid == 0) {
      int *const my_flag = &attn_release[xcd_id * HIER_STRIDE];
      // Watch this poll. a0 is the raw global arrival counter: it is""",
     """#if MPK_OPROJ_KSPLIT_CEIL >= 2
    // -- CEILING PROBE, WRONG OUTPUT BY CONSTRUCTION ----------------------
    // Prices the o_proj K-split's second half. Under a row-sharded o_proj the
    // merge -> W_UV -> o_proj chain is pair-local, so this GPU-wide Phase 8
    // rendezvous is replaced by a 2-XCD one; deleting it outright is the
    // upper bound of that replacement. Combined with level 1 this is the
    // ceiling on the WHOLE change: both rendezvous the K-split targets, gone.
    // o_proj then reads whatever v_out happens to hold. Timing only.
    (void)attn_release_expected;
#else
    if (tid == 0) {
      int *const my_flag = &attn_release[xcd_id * HIER_STRIDE];
      // Watch this poll. a0 is the raw global arrival counter: it is"""),

    (FULLL,
     """            st_flag_u32((void *)my_flag, (unsigned)attn_release_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    MPK_WS_PHASE(63, task_layer_idx, xcd_id);""",
     """            st_flag_u32((void *)my_flag, (unsigned)attn_release_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
#endif // MPK_OPROJ_KSPLIT_CEIL >= 2
    MPK_WS_PHASE(63, task_layer_idx, xcd_id);"""),

    (PK,
     """            flags = flags + ["-DMPK_WUV_SKIP_PEER_WAIT"]
""",
     """            flags = flags + ["-DMPK_WUV_SKIP_PEER_WAIT"]
        _ksplit_ceil = int(os.environ.get("MPK_OPROJ_KSPLIT_CEIL", "0"))
        if _ksplit_ceil:
            # CEILING PROBE for the o_proj N-split -> K-split rewrite.
            # WRONG OUTPUT by construction; never a correctness arm.
            #
            # The rewrite's whole claim is that a row-sharded (K-split) o_proj
            # makes the attention tail head-local and therefore deletes TWO
            # GPU-wide rendezvous: 767 (W_UV -> o_proj, oproj:685) and Phase 8
            # (attention -> o_proj, full_layer:1855). Neither has ever been
            # priced directly -- price_attention_shard.py leaves the row
            # UNPRICED because no ablation existed. This is that ablation:
            #
            #   1  delete the 767 local-flag wait      (W_UV -> o_proj)
            #   2  ...and the Phase 8 local-flag wait  (attention -> o_proj)
            #
            # Level 2 is a strict UPPER bound on the rewrite, which replaces
            # Phase 8 with a pair-local barrier rather than with nothing. If
            # level 2 does not clear the 0.26 ms wall noise floor, the rewrite
            # cannot, and the K-split is not worth building.
            flags = flags + ["-DMPK_OPROJ_KSPLIT_CEIL=%d" % _ksplit_ceil]
"""),

    (ENVC,
     """  MPK_WUV_SKIP_PEER_WAIT
""",
     """  MPK_WUV_SKIP_PEER_WAIT
  # o_proj N-split -> K-split ceiling probe. Deletes the 767 and Phase 8
  # local-flag waits. Compile-time, so an unforwarded rank builds a different
  # megakernel and the layer barriers deadlock rather than error.
  MPK_OPROJ_KSPLIT_CEIL
"""),
]

MARKER = "MPK_OPROJ_KSPLIT_CEIL"

rc = 0
for rel, old, new in EDITS:
    path = os.path.join(ROOT, rel)
    with open(path) as fh:
        src = fh.read()
    if new in src:
        print("skip (already applied): %s" % rel)
        continue
    n = src.count(old)
    if n != 1:
        print("FAIL: %s -- anchor matched %d times, expected 1" % (rel, n))
        rc = 1
        continue
    with open(path, "w") as fh:
        fh.write(src.replace(old, new, 1))
    print("patched: %s" % rel)

for rel in (OPROJ, FULLL, PK, ENVC):
    with open(os.path.join(ROOT, rel)) as fh:
        c = fh.read().count(MARKER)
    print("  %-70s %d occurrences" % (rel, c))

sys.exit(rc)
