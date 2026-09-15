---
name: rocprof-compute-viewer
description: >-
  Capture and analyze ROCprof Compute Viewer-compatible rocprofv3 Advanced
  Thread Trace (ATT/SQTT) collateral, or add diagnostic SQTT stage markers, for
  Redline persistent decoders and other AMD GPU megakernels on Fleet. Use when
  an agent needs ISA latency and stall hotspots, wave or occupancy timelines,
  dispatch resource usage, ATT hardware activity counters, named sub-blocks in
  one fused kernel, an agent-friendly summary, or a complete
  ui_output_agent_*_dispatch_* directory to hand to ROCprof Compute Viewer.
---

# Capture ROCprof Compute Viewer Traces

Use the repository wrapper to collect one bounded ATT trace on a Fleet GPU,
retain the complete viewer payload remotely, and inspect the small agent
summary before proposing a megakernel optimization.

## Read the local contract

Before capturing, read:

- the repository `AGENTS.md`;
- `lib-rocket/bench/persistent_decoder/README.md` for the current wrapper;
- `optimize-megakernel` when the trace will drive a kernel change; and
- the module note at the top of
  `lib-rocket/bench/persistent_decoder/rocprof_compute_summary.py` before a
  broad or multi-SE GPT-OSS capture.

Use `fleetctl`; never use a GPU on the local agent node. Use a Fleet catalog
image by immutable digest. ROCm 7.13 or newer is required for the bundled trace
decoder; ROCm 7.14 is the validated configuration.

## Choose the least intrusive profiler

- Use `phase-profiling` first when the expensive fused stage is unknown.
- Use ATT for instruction-by-instruction latency, stalls, wave states,
  occupancy, and memory-operation-to-wait relationships on selected CUs/SIMDs.
- Use `pc-sampling` for lower-overhead breadth only after a short canary emits
  nonempty samples.
- Use ordinary PMC counters for aggregate device-wide questions.
- Do not start with SPM for this persistent megakernel; it is not a validated
  default and has shown uncontrolled runtime and output growth.

ATT already records instruction latency and stalls. Use `--att-activity` for
the viewer's standard pipeline-utilization summary. Use custom
`--att-perfcounters` only for a focused counter question; these modes are
mutually exclusive.

## Follow the profiling ladder

1. Measure at least five unprofiled production samples and record median TPOT.
2. Run the diagnostic DSO without rocprofv3. Reject quantitative comparisons
   if its resource use or TPOT differs materially from production.
3. Use `s_memrealtime` phase timing to rank stages and synchronization tails.
4. Capture one exact dispatch on one SE, one CU, and one SIMD with ATT activity
   and `--marker-hotspots`.
5. Inspect `worker_xcd_N` and `worker_rank_N`. If the needed logical role is
   absent, repeat another physical selection; physical placement is dynamic.
6. Expand to all four SIMDs or more SEs only to fill a specific coverage gap.
7. Run a separate custom-counter replay only after the instruction trace
   identifies a focused resource question. Useful starting counters for
   outstanding-memory work are `SQ_INST_LEVEL_LDS`, `SQ_INST_LEVEL_VMEM`,
   `SQ_INST_LEVEL_SMEM`, and `SQ_WAIT_ANY`.
8. Accept an optimization only when unprofiled TPOT and the target phase timing
   improve. ATT cycles are causal evidence, not production duration.

For every ATT run, record `profiled_tpot / matching_unprofiled_tpot` and all
completeness warnings. Trace backpressure can alter memory latency, polling,
barrier arrival order, and cross-wave overlap even when the payload is complete.

## Add named stages to a fused kernel

Use source markers when a single fused dispatch needs logical phase labels.
Markers are navigation and attribution aids, not the quantitative timing source.

For the gfx950 GPT-OSS decoder:

1. Build the ROCm SQTT marker pass from
   `ROCm/rocm-systems/projects/rocprof-trace-decoder/markers` against the same
   ROCm installation used for Redline. Retain `libsqttinstrumentpass.so` and
   the directory containing `markers.hpp`.
2. Configure with `REDLINE_PD_SQTT_MARKERS=ON`,
   `REDLINE_PD_SQTT_PASS_PLUGIN=<absolute-plugin-path>`, and
   `REDLINE_PD_SQTT_INCLUDE_DIR=<directory-containing-markers.hpp>`.
   Enable `REDLINE_PD_PHASE_TIMING` separately when phase timing is needed.
   The marker-enabled diagnostic target also emits optimized device line tables
   for instruction-to-source correlation; the production DSO remains unchanged.
3. Use balanced `sqtt_marker_enter("name")` and `sqtt_marker_exit("name")`
   scopes. Use `sqtt_marker_point("name")` for instants. Preserve the
   long-lived `worker_xcd_N` and `worker_rank_N` scopes around
   `worker_engine_body` so worker identity is visible and stage work nests
   beneath the correct logical rank.
4. Apply markers only to
   `redline_persistent_decoder_gpt_oss_gfx950_diagnostic`, never the production
   DSO.

The marker CU scope is a bitmask, while rocprofv3's target CU is an index. For
all four SIMDs of CU index 1, use:

```text
compile: SQTT_SCOPE_CU=0x2 SQTT_SCOPE_SIMD=0xf
capture: --target-cu 1 --simd-select 0xf
```

Keep `SQTT_SCOPE_WAVE=-1`, `SQTT_SCOPE_WG=-1`, and
`SQTT_MEM_BARRIER=fence` unless a narrower experiment requires otherwise. Do
not use `SQTT_SCOPE_CU=-1` for the current GPT-OSS diagnostic decoder; that
scope failed its native correctness preflight.

Before profiling, run the diagnostic DSO without rocprofv3 and require the
native numerical oracle to pass. Verify that marker names, `.sqtt_funcmap`, and
`s_ttracedata` occur only in the diagnostic code object. Keep phase-timing
exports and `s_memrealtime` instrumentation available when enabled.

A marker name present in `.sqtt_funcmap` but absent from one trace usually
means the selected wave did not execute that role or branch; it is not proof of
instrumentation failure.

## Understand ATT selection

| Wrapper option | Native rocprofv3 option | Meaning |
| --- | --- | --- |
| `--shader-engine-mask` | `--att-shader-engine-mask` | Bitmask of SEs producing trace data |
| `--target-cu` | `--att-target-cu` | One local CU index in every selected SE |
| `--simd-select` | `--att-simd-select` | Bitmask of the four SIMDs in each selected CU |
| `--consecutive-kernels N` | `--att-consecutive-kernels N` | Capture `N` total dispatches after the selected dispatch; later dispatches need not match the regex |
| `--kernel-iteration-range R` | `--kernel-iteration-range R` | Select one-based matching dispatch iterations |

`--target-cu` is one scalar CU index, not a list or mask. On a 32-SE,
8-CU-per-SE MI355X, `--shader-engine-mask 0xffffffff --target-cu 1` selects CU
index 1 in each SE, not all 256 CUs. Full physical coverage requires separate
replays for target CU indices 0 through 7, and those replays are not one
coherent timeline.

`--consecutive-kernels` does not make the application launch more quantums.
Verify dispatch order with a lightweight kernel trace, then prefer exact
iteration captures. Capture two quantums as separate `[1]` and `[2]` replays;
do not assume `[1-2]` provides complete wave data for both.

Define completeness explicitly:

- **Collateral:** viewer-ready, all referenced wave files present, no data
  loss, incomplete waves, or unclosed occupancy.
- **Temporal:** each required dispatch iteration has an exact replay.
- **Physical:** the union covers the requested SE/CU/SIMD selections.
- **Logical:** markers cover the worker ranks being investigated.
- **Counters:** activity and custom-counter questions use separate replays.

## Capture a bounded trace

Freeze and record the source revision and diff, image digest, ROCm version,
checkpoint, manifest, token file, executable and DSO hashes, and Fleet job ID.
Use one output epoch and one sample unless the experiment requires otherwise.

```bash
HIP_VISIBLE_DEVICES=0 \
REDLINE_PD_QUANTUM_PREFIX_LENGTH=1024 \
  lib-rocket/bench/persistent_decoder/rocprof_compute_viewer.sh \
    --output-directory "/run/${FLEET_IDENTITY}/results/${EXPERIMENT}/capture" \
    --mode att \
    --kernel-regex '.*worker_kernel.*' \
    --source-directory /workspace/redline/lib-rocket \
    --consecutive-kernels 1 \
    --kernel-iteration-range '[1]' \
    --shader-engine-mask 0x1 \
    --target-cu 1 \
    --simd-select 0x1 \
    --buffer-size 268435456 \
    --gpu-index 0 \
    -- \
    "${REPRO}" "${MODEL}" "${MANIFEST}" "${TOKENS}" \
      1 1 0 "${DIAGNOSTIC_DSO}"
```

`--source-directory` must name the same `lib-rocket` source root represented by
`.` after the target's `-ffile-prefix-map`. The wrapper exposes that tree from
its rocprofv3 work directory and requires a populated `snapshots.json`, all
referenced snapshot files, and nonempty source references in `code.json`. Use
this option whenever the viewer must link ISA rows to C++/HIP source.

Use task-specific variables and resolve every path before submitting. The
wrapper must print at least one `RCV_UI_DIR` and exits nonzero when required
collateral or a referenced wave file is missing. Preserve
`capture-command.txt`.

## Expand coverage and buffer size carefully

Start narrow. For one SE, one CU, and all four SIMDs:

```text
--shader-engine-mask 0x1
--target-cu 1
--simd-select 0xf
--buffer-size 2147483648
--kernel-iteration-range '[1]'
```

For a selective two-SE capture, set one bit per desired SE. SE 0 and SE 1 use
`0x3`; SE 0 and SE 16 use `0x00010001`:

```text
--shader-engine-mask 0x3
--target-cu 1
--simd-select 0xf
--buffer-size 2147483648
--kernel-iteration-range '[1]'
```

This still captures only CU index 1 in each selected SE. Keep marker compilation
matched with `SQTT_SCOPE_CU=0x2 SQTT_SCOPE_SIMD=0xf`.

If a complete logical-role map requires broader coverage:

1. Keep one exact iteration and shorten the workload first.
2. Retry the same selection with `--buffer-size 2147483648`.
3. If ROCm accepts it, canary `--buffer-size 4294967296`; this 4 GiB value was
   validated on ROCm 7.14 but exceeds public guidance and is not portable.
4. If 4 GiB is rejected, incomplete, or too expensive, split the SE mask—for
   example `0x0000ffff` and `0xffff0000`—then reduce batches further as needed.

Reject every replay containing `Data Lost`, incomplete waves, missing wave
files, or unclosed occupancy. Increasing buffer capacity can prevent
truncation, but it does not reduce trace traffic or guarantee unchanged
synchronization timing. Never merge timestamps from separate replays.

Use broad captures for coverage and role mapping, then confirm synchronization
hotspots with narrow, role-matched traces. The summary parser's module note
records the brief validated perturbation findings behind this rule.

## Keep large files on Fleet

Store captures under a durable path such as:

```text
/run/<fleet-identity>/results/<experiment>/
```

Do not use `/tmp`. Inventory the capture before transferring it:

```bash
find "${CAPTURE_ROOT}" -type f -printf '%s\t%p\n' | sort -n \
  > "${RESULTS}/file-sizes.tsv"
```

Pull only `agent-summary.json`, `agent-summary.md`, `capture-command.txt`,
`file-sizes.tsv`, and `run.log` by default. Leave raw `.att`, code objects,
per-counter JSON, wave JSON, and `wstates*.json` on Fleet unless the user needs
the viewer payload. For RCV, copy the complete
`ui_output_agent_<agent>_dispatch_<dispatch>` directory without filtering or
renaming it. Never commit profiler output, model data, or run logs.

## Read the agent summary

The wrapper runs the parser automatically. To reparse existing collateral:

```bash
python3 lib-rocket/bench/persistent_decoder/rocprof_compute_summary.py \
  "${ROCPROF_RESULT_ROOT}" \
  --require-viewer-ready \
  --marker-hotspots \
  --json "${RESULTS}/agent-summary.json" \
  --markdown "${RESULTS}/agent-summary.md"
```

Use `--marker-hotspots` only when instruction-by-stage attribution is needed;
it streams the large wave JSON files and can take minutes on a broad capture.
Interpret the report in this order:

1. Confirm viewer readiness and completeness.
2. Read VGPR, SGPR, LDS, block size, and occupancy limits.
3. Rank ISA rows by total stall and latency, not hit count alone.
4. Compare wait/barrier, VMEM, LDS, scalar, VALU, and matrix classes.
5. Rank marker regions by mean and P95 cycles; do not add overlapping inclusive
   totals.
6. Inspect the top producer-to-wait instruction relationships in the selected
   stage and logical role.
7. Use activity utilization as supporting evidence. Treat unavailable average
   occupancy as unknown.

The summary approximates RCV hotspot, resource, occupancy, activity, and marker
views. Use RCV for interactive timelines, marker flamegraphs, and hidden-latency
inspection.

## Handle failures conservatively

- **No UI directory:** Verify the kernel regex and confirm the dispatch ran.
- **Invalid buffer size:** Pass integer bytes; for 256 MiB use `268435456`.
- **Data loss or incomplete waves:** Shorten the exact capture, then use the
  2 GiB/4 GiB/batched-SE escalation above.
- **Missing `snapshots.json`:** It is optional when core metadata, code,
  occupancy, and every wave referenced by `filenames.json` are present.
- **Unclosed occupancy:** Report peak observed occupancy only.
- **Profiler hang or uncontrolled growth:** Cancel the Fleet job, inventory the
  output, and switch to a narrower ATT capture or another profiler.

Finish with the job ID and node, source and binary hashes, image and ROCm
version, exact command, remote `RCV_UI_DIR`, file inventory, completeness
status, headline findings, and separate correctness and unprofiled-performance
results.
