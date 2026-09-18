#pragma once
// NPS1 concat-Mod7 pack: keep hipMalloc (2 MiB PTEs, mixed map 0) but store
// bytes only in 4 KiB pages whose DF stack is on the caller's AID.
// Validated 2026-09-04 on 0000:66:00.0: concat_mod7(gpu_va+off) sign-flips
// pointer-chase (~242-267 ns local vs ~296-366 ns far). Mosaic bit-13 and
// (va>>12)%7 do NOT. Identity window is 4 KiB, not Mosaic's 8 KiB.
//
// Host writes a uint32 map at the replica base: map[logical_4k] = phys_off
// of that 4 KiB page from the replica base. Logical intra-page bits pass.

#include <cstdint>

inline __host__ __device__ int nps1_concat_mod7(uint64_t pa) {
  uint64_t hi = pa >> 16;
  uint64_t x = (hi & 0xF) ^ ((pa >> 12) & 0xF);
  return (int)(((hi << 4) | x) % 7);
}

inline __host__ __device__ int nps1_stack_aid(int st) {
  return (st == 0 || st == 1 || st == 4) ? 0 : 1;
}

enum { NPS1_PACK_HEADER = 16384 };

inline __device__ uint32_t nps1_map_voff(uint32_t const *map, uint32_t logical) {
  return map[logical >> 12] + (logical & 0xFFFu);
}
