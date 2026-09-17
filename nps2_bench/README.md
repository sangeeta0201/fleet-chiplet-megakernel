# SPX+NPS2 synchronization benchmarks

Standalone harnesses used to locate fleet's NPS2 slowdown. Each runs the same
binary in both memory partition modes so the comparison is like-for-like.

| file | what it measures |
|---|---|
| `fleetbar.cpp` | fleet's ACTUAL O-proj barrier + an arrival-skew knob |
| `fleetbar2.cpp` | release path A/B: shared-NC flags vs AID-replicated |
| `fleetbar4.cpp` | per-level AID isolation + store-drain sweep |
| `aid_rt.h` | AID memory classes + rendezvous primitives |
| `rt_selftest.cpp` | barrier gate for `aid_rt.h` |
| `mem_selftest.cpp` | per-consumer placement gate |
| `moe_layer.cpp` | MoE parity placement + data-directed dispatch prototype |
| `parity_map.cpp` | exhaustive coverage check for the parity dispatch map |

Build (needs the patched driver for the AID allocator):

```
hipcc -O2 --offload-arch=gfx950 -std=c++17 -DMIRAGE_BACKEND_USE_ROCM \
  -I../include/mirage/persistent_kernel -I/usr/include/libdrm \
  fleetbar2.cpp -ldrm -o fleetbar2
```

Findings and closed avenues: `NPS2_BARRIER_LOCALITY.md`.
