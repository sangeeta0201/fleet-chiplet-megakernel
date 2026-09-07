#!/usr/bin/env python3
"""Make the LDS staging a dwordx4 copy instead of a byte loop.

The first version copied through `unsigned char *` with a 16-byte
__builtin_memcpy. Those pointers carry alignment 1, so the compiler could not
assume anything and emitted a byte-granular loop: 1417 us per layer, against
about 10 us for the kernel itself. That left correctness verified on a build 140
times slower than the one the latency numbers came from, where the race windows
the barriers guard are nothing like the same shape.

Same bytes, same order, so the output hash has to come out unchanged -- which is
the check that this rewrite did not alter what is being tested.

Run on mi355x-thor-2:  python3 patch_lds_fast.py
"""
import shutil
import sys

DRV = "/home/schowdha/fleet-chiplet-megakernel/drive_phase7.cu"
SUFFIX = ".pre-ldsfast"

OLD = """      unsigned char const *w_src = my_weight + (size_t)local * WG_BYTES;
      unsigned char *w_dst = (unsigned char *)_lm_smem + LDS_W_OFF;
      for (int i = tid * 16; i < WG_DATA_BYTES; i += NTHREADS * 16) {
        __builtin_memcpy(w_dst + i, w_src + i, 16);
      }
      unsigned char const *s_src = w_src + WG_DATA_BYTES;
      unsigned char *s_dst = w_dst + LDS_W_DATA_PAD;
      for (int i = tid * 16; i < WG_SCALE_BYTES; i += NTHREADS * 16) {
        __builtin_memcpy(s_dst + i, s_src + i, 16);
      }
"""
NEW = """      // 16 bytes per thread per pass, as one dwordx4. Every base is at least
      // 16-byte aligned -- hipMalloc returns 256, WG_BYTES is a multiple of 16,
      // and oproj_lds_w_off() rounds to 16 -- but saying so through uint4 is
      // what lets the compiler emit the wide access. Spelled as a byte memcpy
      // it has to assume alignment 1 and copies a byte at a time.
      uint4 const *w_src =
          (uint4 const *)(my_weight + (size_t)local * WG_BYTES);
      uint4 *w_dst = (uint4 *)((unsigned char *)_lm_smem + LDS_W_OFF);
      for (int i = tid; i < WG_DATA_BYTES / 16; i += NTHREADS) {
        w_dst[i] = w_src[i];
      }
      uint4 const *s_src = w_src + WG_DATA_BYTES / 16;
      uint4 *s_dst =
          (uint4 *)((unsigned char *)_lm_smem + LDS_W_OFF + LDS_W_DATA_PAD);
      for (int i = tid; i < WG_SCALE_BYTES / 16; i += NTHREADS) {
        s_dst[i] = s_src[i];
      }
"""

with open(DRV, "r", encoding="utf-8", newline="") as fh:
    text = fh.read()

n = text.count(OLD)
if n != 1:
    sys.exit("staging anchor matched %d times, expected 1" % n)

shutil.copyfile(DRV, DRV + SUFFIX)
with open(DRV, "w", encoding="utf-8", newline="") as fh:
    fh.write(text.replace(OLD, NEW))

print("patched %s (backup at %s%s)" % (DRV, DRV, SUFFIX))
