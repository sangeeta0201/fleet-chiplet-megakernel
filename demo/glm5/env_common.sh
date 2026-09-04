#!/bin/bash
# Shared environment for the GLM megakernel runs.
# Sourced by run_mp8_dp_ep_fused.sh -- not run directly.
#
# Deliberately parallel to demo/gpt_oss/env_common.sh; the only GLM-specific
# parts are MODEL_PATH and the GLM_* knob names in MPK_FORWARD_VARS.

FLEET_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# MIRAGE_HOME decides which tree the megakernel compiles against, and
# PYTHONPATH decides which one `import mirage` resolves to. A stale editable
# install (~/.local/.../__editable__.mirage_project*.pth) points at a
# *different* checkout -- /home/claudeuser/mirage, branch
# amd-multi-gpu-rocshmem, which has no GLM code -- so both must be pinned or
# the run silently exercises the wrong codebase.
export MIRAGE_HOME="$FLEET_HOME"
export PYTHONPATH="$FLEET_HOME/python:${PYTHONPATH:-}"

export MODEL_PATH="${MODEL_PATH:-zai-org/GLM-4.7-Flash}"

# rocSHMEM + the MPI it was built against. rocSHMEM's IPC backend only needs
# MPI for bootstrap (rank exchange), not for the data path.
#
# The path is probed rather than hardcoded: gpt_oss/env_common.sh names
# /home/claudeuser/rocshmem, which does not exist on this box -- the install
# is /home/claudeuser/rocshmem_install (the source tree is /home/claudeuser/
# rocSHMEM and has headers but no librocshmem.a). Getting this wrong fails
# late and unhelpfully, at codegen, on every rank at once.
if [ -z "${ROCSHMEM_INC_PATH:-}" ]; then
  for d in /home/claudeuser/rocshmem_install /home/claudeuser/rocshmem \
           /opt/rocshmem /usr/local/rocshmem; do
    if [ -f "$d/include/rocshmem/rocshmem.hpp" ] && \
       [ -f "$d/lib/librocshmem.a" ]; then
      export ROCSHMEM_INC_PATH="$d/include"
      export ROCSHMEM_LIB_PATH="${ROCSHMEM_LIB_PATH:-$d/lib}"
      break
    fi
  done
fi
if [ -z "${ROCSHMEM_INC_PATH:-}" ]; then
  echo "env_common.sh: no rocSHMEM install found; set ROCSHMEM_INC_PATH" >&2
fi
export ROCSHMEM_INC_PATH="${ROCSHMEM_INC_PATH:-}"
export ROCSHMEM_LIB_PATH="${ROCSHMEM_LIB_PATH:-}"
if [ -d /home/claudeuser/ompi/lib ]; then
  export MPI_INC_PATH="${MPI_INC_PATH:-/home/claudeuser/ompi/include}"
  export MPI_LIB_PATH="${MPI_LIB_PATH:-/home/claudeuser/ompi/lib}"
  export PATH="/home/claudeuser/ompi/bin:$PATH"
  export LD_LIBRARY_PATH="/home/claudeuser/ompi/lib:/home/claudeuser/ucx/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
else
  export MPI_INC_PATH="${MPI_INC_PATH:-/usr/lib/x86_64-linux-gnu/openmpi/include}"
  export MPI_LIB_PATH="${MPI_LIB_PATH:-/usr/lib/x86_64-linux-gnu/openmpi/lib}"
  export LD_LIBRARY_PATH="$MPI_LIB_PATH:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
fi

# Host thread oversubscription. torch and OpenMP both default to one thread per
# core -- 256 on this box -- and mpirun starts NP copies of that, so an 8-rank
# run puts ~2048 runnable threads on 256 cores. The megakernel is persistent and
# the host thread's only job between iterations is to bump the step counter, so
# losing that thread to the scheduler idles the whole GPU. Divide the cores.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  _mpk_cores=$(nproc 2>/dev/null || echo 8)
  _mpk_np="${NP:-1}"
  export OMP_NUM_THREADS=$(( _mpk_cores / _mpk_np ))
  [ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
  unset _mpk_cores _mpk_np
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# Every knob the GLM path reads, forwarded to all ranks by mpirun. Listing them
# unconditionally is deliberate: -x on an unset variable is a no-op, so a knob
# set in the caller's shell reaches every rank without editing this list.
MPK_FORWARD_VARS=(
  MIRAGE_HOME PYTHONPATH MODEL_PATH GLM_MODEL_PATH
  HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
  ROCSHMEM_INC_PATH ROCSHMEM_LIB_PATH MPI_INC_PATH MPI_LIB_PATH
  LD_LIBRARY_PATH PATH
  OMP_NUM_THREADS MKL_NUM_THREADS
  ROCSHMEM_MAX_NUM_CONTEXTS MASTER_PORT
  ATTN_DP MOE_EP EP_FOLD_RANK
  PRECOMPUTED_DISPATCH MPK_ML_REPLAY
  GLM_FUSE_FULL_LAYER GLM_FUSE_ATTN GLM_FUSE_OPROJ_ROUTER
  GLM_FUSE_MOE_SWIGLU GLM_FUSE_MOE_MULSUMADD
  GLM_MOE_MXFP4 GLM_MOE_MXFP8 GLM_FAKE_MXFP4_EXPERTS GLM_FAKE_MXFP4_ATTN
  GLM_MOE_WIDEN_MXFP8
  PYTHONFAULTHANDLER PYTHONUNBUFFERED
  GLM_DENSE_MXFP8 GLM_DENSE_MXFP8_OPW GLM_OPROJ_MXFP8 GLM_QB_MXFP8
  GLM_DENSE_MLP_MXFP8 GLM_DENSE_MLP_DOWN_ROWS GLM_LMHEAD_TP GLM_DENSE_MLP_TP
  GLM_OPROJ_MXFP4
  GLM_QKV_MXFP8_OPW GLM_MOE_W13_OPW GLM_MOE_W2_OPW
  GLM_OPROJ_GEMV_ROWS GLM_OPROJ_PREFETCH GLM_OPROJ_TP GLM_DENSE_OPROJ_TP
  GLM_PROLOGUE_PREFETCH GLM_OPROJ_RESTAGE
  GLM_UNABSORB_OPROJ GLM_WUV_GEMV_ROWS WUV_MFMA GLM_WUV_TP
  GLM_UNABSORB_QB GLM_WUK_GEMV_ROWS GLM_QB_OPW GLM_QB_TP
  GLM_MLA_NUM_KV_CHUNKS GLM_MLA_MERGE_DIM_SPLITS GLM_MLA_MERGE_WT
  # Compile-time and read per-rank via os.environ in persistent_kernel.py, so
  # without -x only rank 0 builds the pair-local decode barrier and the ranks
  # deadlock at Phase 6. Same bug the ceiling probes below had.
  GLM_MLA_PAIR_MERGE
  GANG_TILE_N GANG_WGM GANG_K_SPLITS
  MPK_SPAN_TIMING MPK_SUBPHASE_TIMING MPK_DEVICE_TIMING MPK_WORKER_STATE
  # Host-side getenv in launch_persistent_kernel: dispatch the 8-block
  # scheduler grid BEFORE the 240-block worker grid so the schedulers cannot
  # be starved of CUs by the workers. Default 1; =0 restores the old order.
  # Read per-rank, so every rank must see it or the arms are mixed.
  MPK_SCHED_LAUNCH_FIRST
  # pre/fused/post split of the iteration, stamped by worker 0. Compile-time,
  # so a mismatch between ranks is a different binary and the barriers
  # deadlock.
  MPK_ITER_SPLIT
  MPK_EP_SIG_DBG MPK_EP_FORCE_STAGED MPK_EP_ABLATE MPK_EP_WAIT_TIMEOUT
  MPK_EP_TMO_PRINT_LAYERS
  # Ceiling probes. All are WRONG OUTPUT by construction and all are
  # compile-time, so every rank has to see them or the ranks build different
  # megakernels and the layer barriers deadlock.
  MPK_MLA_SKIP_DECODE MPK_ATTN_HALFK MPK_W13_EARLY_REL MPK_QB_SKIP_PEER_WAIT
  MPK_WUV_SKIP_PEER_WAIT
  MPK_ABL_QKV MPK_ABL_QKV_PRO MPK_ABL_ML_BOUNDARY
  # Adjacent-phase overlap ceiling probe. =1 is a CORRECT-output control,
  # =2 is the wrong-output probe; decide on 2 vs 1.
  MPK_ABL_PIPE_W13W2
  # Shared-expert makespan pricing probe. CORRECT OUTPUT (rank 0 recomputes
  # its shared-expert W13 tiles and stores the same bits), additive, decide
  # 1 vs 0. Compile-time, so it must reach every rank or the barriers deadlock.
  MPK_SHARED_DUP
  # Register budget for the whole megakernel. 3 => 252 unified VGPRs => 2
  # waves/SIMD. MUST reach every rank: a mismatched register budget is a
  # different binary and the layer barriers deadlock.
  MPK_WORKER_WAVES_PER_EU
  # The other half of the occupancy gate: the per-block dynamic LDS request.
  # 155 of 160 KB/CU is what actually pins 1 block/CU. Compile-time, so a
  # mismatch between ranks is a different binary and the barriers deadlock.
  MPK_WORKER_LDS_KB
  # Bootstrap residency census. Host-pinned coherent slots the device stamps
  # and a detached host thread prints to stderr at t=8s and t=20s, with no HIP
  # call at read time -- the only channel that survives a wedged runtime AND
  # the watchdog's SIGKILL. Runtime-only, so it does not have to match across
  # ranks, but it is in the list so every rank reports.
  MPK_BOOT_PROBE
  # TaskDesc slots in execute_worker's LDS staging buffer. Compile-time, so it
  # must match across ranks; in the whitelist so every rank gets the same one.
  MPK_TASK_DESC_SLOTS
  MPK_TASK_DESC_PAD
  # MoE k-loop prefetch distance. The shipping loop has NO load/MFMA overlap
  # (ISA: global_load x8 then s_waitcnt vmcnt(1) in the same block), and 4
  # k-groups in flight is the measured knee. Compile-time, so a mismatch
  # between ranks is a different binary and the barriers deadlock.
  MPK_MOE_PF_GROUPS
  # addrspace(1) the MoE weight + scale loads AND hoist the B-operand ds_reads
  # above the MFMA group, so the per-MFMA lgkmcnt(0) stops draining the weight
  # prefetch. Compile-time; a rank that misses it is a different binary.
  MPK_MOE_WGLOBAL
  # Threads that addrspace(1) through the k-loop's pointer TYPE, which is what
  # makes MPK_MOE_WGLOBAL land at all -- the leaf cast alone is inert three
  # inlines below the loader lambdas. Compile-time.
  MPK_MOE_WGPTR
  MPK_MOE_SCBASE
  MPK_MOE_STREAM_NT
  MPK_ATTN_STREAM_NT
  # One-trip software pipeline on the MXFP8 GEMV weight stream (W_UK, W_UV,
  # o_proj), paid for with the `av` registers that loop spends batching LDS
  # reads. Compile-time; a rank that misses it is a different binary.
  MPK_ATTN_GEMV_PF
  # Double-buffered form of that same k-loop: swap two register buffers instead
  # of copying one into the other at the backedge, which is what forces the two
  # s_waitcnt vmcnt(0) per trip in the shipped ISA. Compile-time.
  MPK_MOE_PF_DBUF
  # ...and the per-GEMM split of the same two knobs. The +29 VGPR bill that
  # killed MPK_MOE_PF_DBUF as a single knob is entirely W2's, and the loop form
  # only wins at GROUPS >= 6 (standalone: ping-pong is 10.11 us/tile at 4,
  # against the copying form's 9.71, but 8.43 at 6 and 7.82 at 8). Compile-time.
  MPK_MOE_PF_DBUF_W13 MPK_MOE_PF_DBUF_W2 MPK_MLA_DECODE_DBLBUF
  MPK_MOE_PF_GROUPS_W13 MPK_MOE_PF_GROUPS_W2
  # Scheduling of the same k-loop's B-operand (LDS activation) reads against
  # the MFMA group. Both settings are closed no-gos on the image -- see the
  # define -- but it is compile-time, so forward it rather than let an
  # experiment build rank 0 differently from the other seven.
  MPK_MOE_BSCHED
  # K-major MoE weight layout. Both a -D and a change to how demo.py PACKS the
  # weight, so a rank that misses it reads a permuted buffer as if it were
  # row-major and produces silent garbage -- forward it or leave it unset.
  MPK_MOE_KMAJOR
  MPK_DENSE_KMAJOR
  # The same knob for the attention-half GEMM's k-loop. Compile-time.
  MPK_ATTN_PF_GROUPS MPK_ATTN_PF_DBUF
  # ...and the per-BRANCH split of the depth. The unified knob drives the
  # N-parallel loop (walks MFMA_ITERS) and the K-parallel loop (walks
  # MFMA_ITERS/4) with one request, so its 2/4/8 sweep reported their sum.
  # Compile-time; an unforwarded one builds a different megakernel per rank
  # and the layer barriers deadlock.
  MPK_ATTN_PF_GROUPS_N MPK_ATTN_PF_GROUPS_K
  # latent_to_cache's flattened index chase. Compile-time.
  MPK_KVUPD_FAST
  # Scratch backing store. Raising occupancy to 2 waves/SIMD doubles what ROCr
  # must reserve for spills (1192 B/thread x 256 CU x 4 SIMD x 2 waves x 64
  # lanes = 149 MB), which can cross the runtime's default single-dispatch
  # scratch limit. Runtime, not compile-time, but still per-rank.
  HSA_SCRATCH_SINGLE_LIMIT HSA_NO_SCRATCH_THREAD_LIMITER
  HSA_NO_SCRATCH_RECLAIM HSA_ENABLE_SCRATCH_ASYNC_RECLAIM HSA_ENABLE_SCRATCH_ALT
  # Correct-output capacity probe, compile-time all the same.
  MPK_MOE_SHADOW_KB
  # Next-layer qkv_a weight prefetch into the MoE worker hole. Real change,
  # correct output, compile-time -- and the publisher lives in the shared
  # multi-layer loop, so every rank must build it identically.
  MPK_QKVA_PF_KB MPK_QKVA_PF_AT
  # Correct-output pricing probe for item 2's dependency half.
  MPK_QKVA_REPS MPK_W13_REPS
  # Correct-output pricing probe, but compile-time all the same.
  MPK_NULL_PHASES MPK_NULL_TREE MPK_NULL_TILES
  # Real changes, not probes, but compile-time: every rank must build them or
  # the arrival counters (BAR_TREE) / the W2 tile space (W2_KSPLIT) disagree
  # with the host loop bound and the layer barriers wedge.
  MPK_BAR_TREE MPK_W2_KSPLIT MPK_MOE_LIVE_BOUND MPK_W2_STAGE_FULL MPK_VPROBE
  MPK_MOE_ACT_FP8
  MPK_QKV_EP_FOLD MPK_QKV_PRO_HOIST MPK_QUANT_V16 MPK_QKV_FOLD_ROWS
  MPK_BAR_SKEW MPK_EP_FOLD_WGS MPK_EP_POLL_BATCH MPK_ML_PTR_PREFETCH
  # Not a semantic change -- both settings are coherent -- but still
  # compile-time, and an A/B is only one variable if every rank agrees.
  MPK_BAR_POLL_NT MPK_PEER_POLL_NT MPK_BAR_FLAG_MAX
  # The drift-immune arm of the barrier self-heal. Compile-time, and a rank
  # that misses it heals on a different predicate than its peers.
  MPK_BAR_PEER_HEAL
  MPK_ML_BOUNDARY_PAD MPK_BAR_SKEW_DROP_NS MPK_WUV_IN_MERGE GLM_RESADD_UNROLL
  GLM_RESADD_BATCH GLM_RESADD_GLOBAL GLM_MLFL_LDS GLM_EP_ASSUME_DIRECT GLM_MERGE_GLOBAL
  GLM_CONST_BLOCKDIM
  MPK_PRINT_GEOMETRY MPK_LINE_TABLES
  MPK_PRINT_ALL_RANKS
  MPK_HOST_DBG_POLL
  # -DMPK_MAX_TOKENS_PER_REQUEST=1: pins prefill to one token per iteration so
  # a batch_size > 1 *build* can be gated with only one active row. Read via
  # os.environ in persistent_kernel.py, i.e. compile-time and per-rank -- an
  # unforwarded run silently keeps 2 tokens/iter (it did once, and the
  # "garbage at one row" conclusion drawn from it was wrong).
  CK_FMHA_1TOK
  # Speculative decode harness (prepare_next_batch). Compile-time and
  # per-rank: every rank runs its own prepare_next_batch, so a rank that did
  # not build it dispatches one row while its peers dispatch two and the DP
  # attention shapes disagree.
  MPK_SPEC_DECODE MPK_SPEC_ORACLE
  # Per-stage activation checksums (mpk_bsdbg.cuh). Compile-time, so every
  # rank must see it or the ranks build different megakernels.
  #
  # MPK_BSDBG_LAYER0 offsets the dump window by an absolute task_layer_idx --
  # task_layer_idx is run-monotonic ((pc_iter-1)*76 + ml), so 76 is iteration 1.
  # Same forwarding requirement as MPK_BS_DEBUG, and it was missing: a run that
  # sets it in the caller's shell only rebuilds rank 0, which is a different
  # binary from the other seven.
  MPK_BS_DEBUG MPK_BSDBG_LAYER0
  MAX_SAVE_TOKENS
  # Perplexity mode. Not just a host-side flag: it makes the LM head write
  # output row step+1 instead of row 0, which changes the EMITTED kernel
  # argument list, so a rank that misses it builds a different megakernel from
  # its peers and the layer barriers deadlock. It also resizes the sink, so an
  # unforwarded rank would allocate one row and be written n.
  PPL_MODE
  # Host-side corpus selection, but still forwarded: every rank loads the
  # corpus itself and scores its own copy, so a rank that missed one of these
  # would score DIFFERENT TEXT than its peers. The cross-rank agreement check
  # (P2) compares perplexities across ranks and would report a spread that
  # looks like an EP fold bug. Both have in-code defaults, so an unforwarded
  # override is silent: the caller's shell changes and the ranks do not.
  PPL_PREFIX PPL_SKIP_TOKENS
  MPK_NUM_WORKERS
  # Runtime (not compile-time): picks the single-grid persistent_kernel over
  # the two-stream worker/scheduler split. A rank that misses it runs a
  # different launch topology than its peers, which is worse than either arm.
  MPK_SPLIT_SCHED
)

mpk_x_args() {
  local v
  for v in "${MPK_FORWARD_VARS[@]}"; do printf -- '-x %s ' "$v"; done
}
