import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig

from ..core import *
from ..kernel import get_key_paths, KNGraph, TBGraph
from .speculative import (
    SpecDecodeConfig,
    PromptLookupConfig,
)
from typing import Optional

HARD_CODE = """
#include <Python.h>

// HIP/CUDA runtime abstraction
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) || defined(MIRAGE_BACKEND_USE_ROCM)
#include <hip/hip_runtime.h>
typedef hipStream_t cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer;
  std::vector<void*> meta_tensors;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiL", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id);

  Py_RETURN_NONE;
}

static PyObject *init_request_func(PyObject *self, PyObject *args) {
  Py_BEGIN_ALLOW_THREADS
  init_request_resources();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  PyObject *py_stream;
  cudaStream_t stream;
  if (!PyArg_ParseTuple(args, "O", &py_stream)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  launch_persistent_kernel(stream);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyObject *set_rope_tables_func(PyObject *self, PyObject *args) {
  PyObject *py_cos, *py_sin;
  if (!PyArg_ParseTuple(args, "OO", &py_cos, &py_sin)) {
    PyErr_SetString(PyExc_TypeError, "Expected (cos_ptr, sin_ptr)");
    return NULL;
  }
  void *cos_ptr = PyLong_AsVoidPtr(py_cos);
  void *sin_ptr = PyLong_AsVoidPtr(py_sin);
  set_rope_tables(cos_ptr, sin_ptr);
  Py_RETURN_NONE;
}

#ifdef MPK_SPEC_DECODE
static PyObject *set_spec_draft_tokens_func(PyObject *self, PyObject *args) {
  PyObject *py_ptr;
  if (!PyArg_ParseTuple(args, "O", &py_ptr)) {
    PyErr_SetString(PyExc_TypeError, "Expected (draft_tokens_ptr)");
    return NULL;
  }
  set_spec_draft_tokens(PyLong_AsVoidPtr(py_ptr));
  Py_RETURN_NONE;
}
#endif

static PyObject *read_shmem_alloc_func(PyObject *self, PyObject *args) {
  int index;
  PyObject *py_dst;
  unsigned long long nbytes;
  if (!PyArg_ParseTuple(args, "iOK", &index, &py_dst, &nbytes)) {
    PyErr_SetString(PyExc_TypeError, "Expected (index, dst_ptr, nbytes)");
    return NULL;
  }
  void *dst = PyLong_AsVoidPtr(py_dst);
  int rc = mpk_read_shmem_alloc(index, dst, (size_t)nbytes);
  return PyLong_FromLong((long)rc);
}

static PyObject *num_shmem_allocs_func(PyObject *self, PyObject *args) {
  return PyLong_FromLong((long)mpk_num_shmem_allocs());
}

static PyObject *shmem_alloc_size_func(PyObject *self, PyObject *args) {
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    PyErr_SetString(PyExc_TypeError, "Expected (index)");
    return NULL;
  }
  return PyLong_FromUnsignedLongLong(mpk_shmem_alloc_size(index));
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"init_request_func", init_request_func, METH_VARARGS, "initialize request resources"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {"set_rope_tables_func", set_rope_tables_func, METH_VARARGS, "set RoPE cos/sin tables"},
#ifdef MPK_SPEC_DECODE
  {"set_spec_draft_tokens_func", set_spec_draft_tokens_func, METH_VARARGS, "set the speculative draft-token buffer"},
#endif
  {"read_shmem_alloc_func", read_shmem_alloc_func, METH_VARARGS, "snapshot symmetric-heap alloc into device buffer"},
  {"num_shmem_allocs_func", num_shmem_allocs_func, METH_VARARGS, "number of recorded symmetric-heap allocs"},
  {"shmem_alloc_size_func", shmem_alloc_size_func, METH_VARARGS, "byte size of a recorded symmetric-heap alloc"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""

valid_persistent_kernel_modes = {"offline", "online", "online_notoken", "onepass", "online_multi_turn"}

def get_compile_command(
    mpk,
    target_cc,
    cc,
    file_name,
    py_include_dir,
    mirage_home_path,
    mirage_inc_path,
    mirage_deps_path,
    nvshmem_inc_path,
    nvshmem_lib_path,
    mpi_inc_path,
    mpi_lib_path,
    py_so_path,
    profiling,
    use_nvshmem,
    use_rocshmem=False,
    rocshmem_inc_path=None,
    rocshmem_lib_path=None,
    num_workers=None,
    num_local_schedulers=None,
    num_remote_schedulers=None,
    use_cutlass_kernel=True,
):
    max_worker_per_scheduler = 128
    if num_workers != None and num_local_schedulers != None and num_remote_schedulers != None:
        min_schedulers = 0
        if num_remote_schedulers == 0:
            min_schedulers = num_local_schedulers
        else:
            min_schedulers = min(num_local_schedulers, num_remote_schedulers)
        # This bounds two stack arrays in execute_scheduler -- my_workers[] and
        # worker_queue_next_free_task_pos[] -- and on AMD it is NOT a
        # divisibility question. Each scheduler collects the workers whose
        # HARDWARE-REPORTED XCD (worker_xcd_map[w], written by the worker
        # itself) matches its own, so the bound is "how many of the N worker
        # blocks can the dispatcher land on a single XCD", not "N / nsched".
        # Nothing makes that even. The old `(num_workers // min_schedulers) + 1`
        # gave exactly ONE slot of slack: at the shipping 240 workers / 8
        # schedulers it is 31 against a mean of 30, so a dispatch skew of +2 on
        # any one XCD overruns the array.
        #
        # That overrun IS the launch wedge (#135). The collection loop sits
        # ABOVE the `[SCHED_XCD]` printf, so a run that overflows dies before
        # printing any -- which is the exact observed signature: four
        # `launch_persistent_kernel ENTER` lines, ZERO `[SCHED_XCD]` lines, and
        # then either 100% GPU utilisation forever (the write landed on an
        # adjacent stack slot and corrupted my_num_workers / the queue state) or
        # HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on all ranks (it landed
        # past the scratch allocation). Both presentations, one bug. It also
        # explains the worker-count cliff: 248 workers is a mean of 31 against a
        # bound of 32 and hangs near-deterministically, and 464 is 58 against
        # 59. Healthy runs print workers_on_xcd=30 for all 8 schedulers on all 4
        # ranks -- never 31 -- i.e. the runs that survive are the ones the
        # dispatcher happened to split evenly.
        #
        # Fix: CLAMP the kernel-side collection loop (that is what makes the
        # store safe at any bound) and give the array a little real headroom on
        # top of the mean.
        #
        # MEASURED, and this is the part to not re-litigate: the overflow is a
        # genuine latent out-of-bounds write, but it is NOT what wedges the
        # launch. Over n=10 at bound 46 the box still produced 1 aperture
        # violation and 1 hang (8 OK / 2 BAD, against a 4-bad-in-11 baseline --
        # no significant change), the clamp's [SCHED_OVF] line never printed
        # once, and every healthy run reports workers_on_xcd=30 for all 8
        # schedulers on all 4 ranks. The dispatcher does not in fact skew here.
        # So the headroom is cheap insurance, not a cure; keep it small, since
        # both arrays are indexed dynamically in the scheduler's dispatch loop
        # and a larger scratch footprint is not free.
        _mean = (num_workers + min_schedulers - 1) // min_schedulers
        max_worker_per_scheduler = _mean + 4

    common_cmd = [
        cc,
        # "--default-stream per-thread" is used to create new stream for 
        # each host thread as default stream instead of using the same 
        # legacy stream for all host threads
        # This is important in multi-threaded environment.
        # "--default-stream",
        # "per-thread",
        file_name,
        "-O3",
        # Use following flags when debugging
        # "-O0",
        # "-g",
        # "-G",
        # "--ptxas-options=-v",
        # "-Xptxas=-v",
        "-lineinfo",
        f"-I{py_include_dir}",
        f"-I{mirage_inc_path}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
        f"-I{os.path.join(mirage_home_path, 'deps/json/include')}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
        f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
    ]

    flags = [
        "-shared",
        "-std=c++17",
        "-rdc=false" if not use_nvshmem else "-rdc=true",
        "-use_fast_math",
        "-lcuda",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        py_so_path,
    ]
    flags = flags + [f"-DMPK_TARGET_CC={target_cc}", "-DMIRAGE_BACKEND_USE_CUDA"]

    if mpk.mode == "offline":
        flags = flags + ["-DMODE_OFFLINE"]
    elif mpk.mode == "online":
        flags = flags + ["-DMODE_ONLINE"]
    elif mpk.mode == "online_notoken":
        flags = flags + ["-DMODE_ONLINE_NOTOKEN"]
    elif mpk.mode == "onepass":
        flags = flags + ["-DMODE_ONEPASS"]
    elif mpk.mode == "online_multi_turn":
        flags = flags + ["-DMODE_MULTI_TURN"]
    else:
        raise ValueError(f"Invalid persistent kernel mode: {mpk.mode}")

    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}"]
    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}"]
    flags = flags + [f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}"]
    flags = flags + [f"-DMPK_PAGE_SIZE={mpk.page_size}"]
    flags = flags + [f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}"]
    # Enable gang task support (adds extra code paths in persistent kernel)
    if int(os.environ.get("USE_GANG", "0")) == 1:
        flags = flags + ["-DMPK_ENABLE_GANG_TASKS"]
    if int(os.environ.get("USE_NT_WEIGHTS", "0")) == 1:
        flags = flags + ["-DMPK_NT_WEIGHT_LOADS"]
    if int(os.environ.get("W13_LDS_WEIGHTS", "0")) == 1:
        flags = flags + ["-DMPK_W13_LDS_WEIGHTS"]
    if int(os.environ.get("W13_LDS_PREFETCH", "1")) == 1:
        flags = flags + ["-DMPK_W13_LDS_PREFETCH"]
    # W_UV takes an FP8 activation through v_mfma_scale_f32_16x16x128_f8f6f4
    # instead of dequantizing the weight to bf16 for v_dot2c_f32_bf16. Needs
    # GLM_WUV_GEMV_ROWS=64, which demo/glm5/demo.py pins when this is set.
    #
    # Off by default: measured a 26% regression on the stage. See the note in
    # demo/glm5/demo.py -- W_UV's K=512 is exactly the MFMA kernel's minimum
    # pipeline depth, so the swap is all fill and no steady state.
    if int(os.environ.get("WUV_MFMA", "0")) == 1:
        flags = flags + ["-DMPK_WUV_MFMA"]
    if int(os.environ.get("W13_QFIRST", "0")) == 1:
        flags = flags + ["-DMPK_W13_QFIRST"]
    if int(os.environ.get("W13_ILV", "0")) == 1:
        flags = flags + ["-DMPK_W13_ILV"]
    if int(os.environ.get("W13_BIASPRE", "0")) == 1:
        flags = flags + ["-DMPK_W13_BIASPRE"]
    if int(os.environ.get("QKV_RMSPRE", "0")) == 1:
        flags = flags + ["-DMPK_QKV_RMSPRE"]
    if int(os.environ.get("TK_DEFER", "0")) == 1:
        # Hoist TopK's three st_wt stores out of the k=4 loop. The loop
        # alternates them with `asm volatile` blanking blocks, and LLVM's
        # waitcnt pass cannot see into inline asm, so each iteration drains
        # write-through stores to HBM. CORRECT output: nothing in the loop
        # reads what it stores, and the `output` store it removed was dead
        # (renormalize is literally true, so the renorm pass overwrote it).
        flags = flags + ["-DMPK_TK_DEFER"]
    if int(os.environ.get("TK_NOELOAD", "0")) == 1:
        # TopK's completer read the routing_ready epoch back with an uncached
        # ld_nt_s32 before publishing epoch+1. The consumer never reads it --
        # it derives layer_counter + 1 -- so the producer takes the same value
        # as an argument. CORRECT output: same value, one fewer HBM round trip
        # on the serial path 239 workers wait behind.
        flags = flags + ["-DMPK_TK_NOELOAD"]
    if int(os.environ.get("EP9_DIRECT", "0")) == 1:
        # Phase 9a's release flags removed: the 8 folding workgroups poll the
        # GPU-wide arrival counter (ep_moe_done >= 8 * (layer_idx + 1))
        # instead of a flag the closer publishes after observing it. Drops the
        # closer's 8 write-through stores and their drain from the span
        # between the last W2 tile and the first byte of folding. CORRECT
        # output: same predicate, same instant; ordering was never carried by
        # the flag (it is each worker's own vmcnt drain before its arrival
        # atomic, plus the folder's buffer_inv).
        #
        # MEASURED NULL: 2.219 vs 2.138 baseline, +2.25 us/layer. The 8 `nt`
        # spin loops contend with the 8 XCD leaders' RMWs on that same line;
        # the separate release lines were what kept poll traffic off the
        # critical atomic. See the long note at MPK_EP9_DIRECT.
        flags = flags + ["-DMPK_EP9_DIRECT=1"]
    # Use when debugging
    # flags = flags + [f"-DMPK_ENABLE_VERBOSE"]
    if int(os.environ.get("PRECOMPUTED_DISPATCH", "1")) == 1:
        flags = flags + ["-DMPK_PRECOMPUTED_DISPATCH"]
    if int(os.environ.get("TRACE_MOE", "0")) == 1:
        flags = flags + ["-DMPK_TRACE_MOE_DISPATCH"]
    if int(os.environ.get("EMBED_DEBUG", "0")) == 1:
        flags = flags + ["-DEMBED_DEBUG"]
    if int(os.environ.get("MPK_DEBUG_LAYERS", "0")) == 1:
        flags = flags + ["-DMPK_DEBUG_RMSNORM", "-DMPK_DEBUG_MOE_MUL_SUM"]
    # Enable debug output for HIP builds
    # if target_cc == 94:
    #    flags = flags + ["-DMPK_ENABLE_VERBOSE"]

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        nvshmem_flags = ["-DUSE_NVSHMEM", "-ccbin=mpic++", "-lnvshmem_host", "-lnvshmem_device", "-lmpi"]
        common_cmd = common_cmd + nvshmem_cmd
        flags = flags + nvshmem_flags

    if target_cc == 90:
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_HOPPER",
            "-DNDEBUG",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 100:
        specific_cmd = [
            "-arch=sm_100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_BLACKWELL",
        ]
    elif target_cc in (94, 95):
        # MI300/MI350 ROCm: use HIP. specific_cmd set below with ROCm-specific args.
        specific_cmd = []
    else:
        specific_cmd = [
            "-arch=native",
        ]

    if target_cc in (94, 95):
        # ROCm/MI300/MI350 compile path: hipcc, ROCm includes/libs
        rocm_home = os.environ.get("ROCM_PATH", "/opt/rocm")
        rocm_include = os.path.join(rocm_home, "include")
        rocm_lib = os.path.join(rocm_home, "lib")
        rocm_lib64 = os.path.join(rocm_home, "lib64") if os.path.exists(os.path.join(rocm_home, "lib64")) else rocm_lib
        rocblas_inc = os.path.join(mirage_deps_path, "rocblas", "include")
        # HIP compatibility headers directory (must come before CUTLASS includes)
        hip_compat_inc = os.path.join(mirage_inc_path, 'mirage/hip_compat')
        common_cmd = [
            cc,
            "-x", "hip",
            file_name,
            "-O2",  # -O3 causes LLVM AMDGPU register allocator to hang on large fused kernels
            # NOTE: do NOT add --save-temps here. Under multi-GPU SPMD both ranks
            # share a cwd and compile an input named test.cu, so --save-temps
            # dumps colliding intermediate object files (test-hip-amdgcn-*.o),
            # corrupting the device link (undefined __hip_gpubin_handle).
            # Omit -lineinfo for ROCm: hipcc forwards it to ld.lld which treats it as -l lineinfo
            # MPK_LINE_TABLES=1 adds -gline-tables-only so `llvm-objdump -d -l`
            # maps an ISA address back to file:line. It changes nothing about
            # codegen at -O2 but it does grow the image, so it is opt-in and
            # must never be on for a run whose latency is being quoted.
            *(["-gline-tables-only"]
              if os.environ.get("MPK_LINE_TABLES", "0") == "1" else []),
            f"-I{py_include_dir}",
            f"-I{mirage_inc_path}",
            f"-I{hip_compat_inc}",  # HIP compatibility headers (before CUTLASS, includes cuda/std/ compatibility)
            f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
            f"-I{rocblas_inc}",  # ROCm-compatible CUTLASS headers (before main CUTLASS)
            f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",  # Main CUTLASS (fallback)
            f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
            # CK submodule for PagedKV FMHA (must come before system CK)
            f"-I{os.path.join(mirage_deps_path, 'composable_kernel/include')}",  # CK-tile headers
            f"-I{rocm_include}",
            f"-I{os.path.join(mirage_home_path, 'deps/json/include')}",
            f"-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
            f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
            f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
        ]
        flags = [
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-D__HIP_PLATFORM_AMD__=1",
            "-DMIRAGE_AMD_MI300",
            "-DMIRAGE_BACKEND_USE_ROCM",
            f"-DMPK_TARGET_CC={target_cc}",
            f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}",
            f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}",
            f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}",
            f"-DMPK_PAGE_SIZE={mpk.page_size}",
            f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}",
            "-o",
            py_so_path,
        ]
        if mpk.mode == "offline":
            flags = flags + ["-DMODE_OFFLINE"]
        elif mpk.mode == "online":
            flags = flags + ["-DMODE_ONLINE"]
        elif mpk.mode == "online_notoken":
            flags = flags + ["-DMODE_ONLINE_NOTOKEN"]
        elif mpk.mode == "onepass":
            flags = flags + ["-DMODE_ONEPASS"]
        elif mpk.mode == "online_multi_turn":
            flags = flags + ["-DMODE_MULTI_TURN"]
        if profiling:
            flags = flags + ["-DMPK_ENABLE_PROFILING"]
            flags = flags + ["-DMPK_ENABLE_TIMING"]
            profiling_iters = int(os.environ.get("MPK_PROFILING_ITERS", "1"))
            flags = flags + [f"-DMPK_PROFILING_NUM_ITERS={profiling_iters}"]
        else:
            flags = flags + ["-DMPK_PROFILING_NUM_ITERS=0"]
        if int(os.environ.get("MPK_ITER_SPLIT", "0")) == 1:
            # Worker 0 stamps BEGIN_TASK_GRAPH / first fused task / last fused
            # task, so the pre-loop (embed + 3 dense layers) and post-loop
            # (final norm + LM head + argmax) segments are measured directly
            # instead of inferred from the --max-layers intercept.
            flags = flags + ["-DMPK_ITER_SPLIT"]
        if int(os.environ.get("MPK_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_TIMING"]
        if int(os.environ.get("MPK_DEVICE_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_DEVICE_TASK_TIMING"]
        if int(os.environ.get("MPK_DEVICE_ACCUM", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_DEVICE_TASK_ACCUM"]
        if int(os.environ.get("MPK_SUBPHASE_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_SUBPHASE_TIMING"]
        # Ablation for the o_proj weight prefetch in the fused MLA layer.
        # It is a pure L2 warm-up -- the LDS copy is never read -- so turning
        # it off is safe by construction and changes only bandwidth timing.
        # Worth an ablation because the working set scales with the model:
        # PF_WG_BYTES = OPROJ_ROWS_PER_WG * OPROJ_REDUCTION_SIZE * 33/32, which
        # is 160 KB at GLM-4.7-Flash but 528 KB at GLM-5, i.e. 15.3 MB per XCD
        # against 4 MB of L2. See gang_mla_full_layer_fused_mi300.cuh.
        if int(os.environ.get("GLM_OPROJ_PREFETCH", "1")) == 0:
            flags = flags + ["-DMPK_GLM_OPROJ_PREFETCH_OFF"]
        # Ablation for the LDS prologue + hoisted A-tile prefetch in
        # gang_rmsnorm_linear_mxfp8_bias_kernel (the qkv_a and q_b GEMMs).
        # Unlike the o_proj knob above this one does change what is computed --
        # the row and the norm weight are read once into LDS instead of twice
        # from global -- so the ablation is also the correctness fallback.
        if int(os.environ.get("GLM_PROLOGUE_PREFETCH", "1")) == 0:
            flags = flags + ["-DMPK_GLM_PROLOGUE_PREFETCH_OFF"]
        # Pair-local decode -> merge rendezvous in the fused MLA attention
        # task: re-cut the decode work map so XCD pair (2k, 2k+1) owns every
        # kv chunk of q_group k, which is what the merge map already assumes,
        # and shrink Phase 6's barrier from 8 XCDs to 2. This is the gpt-oss
        # CROC chunk-barrier idiom, adapted -- GLM's merge is 128 tiles, far
        # too big for gpt-oss's "last arriver runs it inline", so the barrier
        # is narrowed rather than deleted. Needs 8 % NUM_Q_GROUPS == 0 and
        # (8 / NUM_Q_GROUPS) | NUM_KV_CHUNKS; the kernel falls back to the
        # global barrier when either fails. See
        # gang_mla_attn_fused_mi300.cuh.
        if int(os.environ.get("GLM_MLA_PAIR_MERGE", "0")) == 1:
            flags = flags + ["-DMPK_GLM_MLA_PAIR_MERGE"]
        # Unroll factor for qkv_a's residual-resolve + EP-fold prologue loop
        # (_rnlm8_resadd_norm_rcp in gang_rmsnorm_linear_mxfp8_bias_mi300.cuh).
        # That loop is 46.2% of the qkv_a tile -- 6119 of 13246 ns, measured
        # with MPK_SUBPHASE_TIMING against the call site's own tile count
        # (8b01e7e) -- and at REDUCTION_SIZE 6144 it runs ITERS=6 trips that
        # `unroll 1` serializes, so it eats six memory latencies with only ~9
        # loads in flight while sitting at 30% of the per-CU L2 share.
        # N divides the number of serialized latencies by N at ~18 VGPRs per
        # extra trip in flight; registers are free at 320 of 512.
        #
        # Default is now 6 (== full at ITERS=6): -0.072 ms on min-of-115 over
        # n=3 paired reps, -0.070 on the device clock, correctness 4/4 with all
        # 8 ranks identical. That is under the 0.26 ms wall noise floor and is
        # only believable because the two arms rank-separate 3v3; see the long
        # note at the loop itself. Set GLM_RESADD_UNROLL=1 to get the old code
        # path back for an A/B.
        _ru = int(os.environ.get("GLM_RESADD_UNROLL", "6"))
        flags = flags + ["-DMPK_RESADD_UNROLL=%d" % _ru]
        # Issue every trip's global loads before consuming any of them, rather
        # than trusting the unroll to interleave them. The unroll alone left
        # the loop at ~251 cycles/load (~1.5 in flight) because the per-trip
        # ds_write to s_x raises lgkmcnt and fences each trip's fetch group
        # off from the next; splitting the loop hoists all 48 EP loads above
        # the first ds_write. Costs ~96-108 VGPRs on top of 284+36 of 512 --
        # inside the budget, but READ THE VGPR COUNT off the built image
        # before believing any A/B, because a spill to scratch is strictly
        # worse. Measured: no spill (284 VGPR / 36 AGPR / spill 0, unchanged),
        # resolve loop 5646 -> 4009 ns (-29.0%), tile 12750 -> 11101 (-12.9%),
        # wall 10.377 -> 10.261 (-0.116, n=3 paired on min-of-115, device
        # clock -0.159), correctness 4/4 with all 8 ranks identical. Default 1;
        # set GLM_RESADD_BATCH=0 for the old code path.
        _rb = int(os.environ.get("GLM_RESADD_BATCH", "1"))
        flags = flags + ["-DMPK_RESADD_BATCH=%d" % _rb]
        # Address the batched resolve loads through addrspace(1). Read off the
        # built image, the batch already carries all 30 loads to a single
        # `s_waitcnt vmcnt(0)` -- but 24 of the 30 are `flat_load_dwordx2`,
        # because only the norm-weight load carries an addrspace cast. flat has
        # no SGPR-base form on gfx9, so each of those 24 pays a 64-bit
        # v_add_co/v_addc_co pair, and flat also raises lgkmcnt, which is why
        # the drain reads `vmcnt(0) lgkmcnt(0)`. This is 3c1fa83's edit applied
        # to the resolve loop; it is NOT the MoE k-loop case that regressed,
        # because that one changed the s_waitcnt count and this region has
        # exactly one drain.
        _rg = int(os.environ.get("GLM_RESADD_GLOBAL", "1"))
        flags = flags + ["-DMPK_RESADD_GLOBAL=%d" % _rg]
        # The fused layer's 29-entry input_ptrs / 11-entry output_ptrs tables
        # live in LDS -- task_descs is a reinterpret_cast of a __shared__ char
        # array -- but the cast drops the address space, so they arrive as
        # generic pointers and every subscript compiles to flat_load against
        # the LDS aperture. A flat access to LDS raises vmcnt as well as
        # lgkmcnt, so the pointer table pollutes every vector-memory wait in
        # the phase. addrspace(3) turns them into ds_read_b64. NOT addrspace(1)
        # -- that faults, the data is not global. See MPK_MLFL_LDS in
        # gang_mla_full_layer_fused_mi300.cuh.
        _mg = int(os.environ.get("GLM_MLFL_LDS", "1"))
        flags = flags + ["-DMPK_MLFL_LDS=%d" % _mg]
        # MEASURED NEUTRAL, default 0. The EP fold's staged `else` branch --
        # rocshmem_putmem_signal_wg -- inlines into the fused layer twice per
        # instantiation at 53 flat ops each and is never executed here, but
        # eliding it (=2) takes the fused layer from 454 flat ops to 142 and
        # moves the wall 10.086 -> 10.104: nothing. =1 additionally plants a
        # __builtin_trap and costs +0.067. The flat-op census is not the metric;
        # only flat that EXECUTES is. See MPK_EP_ASSUME_DIRECT in mpk_comm.cuh.
        _ad = int(os.environ.get("GLM_EP_ASSUME_DIRECT", "0"))
        flags = flags + ["-DMPK_EP_ASSUME_DIRECT=%d" % _ad]
        # addrspace(1) on the split-KV merge's lse_acc / o_acc reads. Located by
        # line-table census, not by guessing: merge_splitkv.cuh:268 and :273 are
        # 32 + 32 flat_load_dword in the fused MLA layer and 16 + 16 more in
        # worker_kernel / persistent_kernel -- the largest EXECUTED flat cluster
        # left after MPK_EP_ASSUME_DIRECT retired the dead putmem one.
        # MEASURED NEUTRAL: -0.019 ms, n=6 arm vs n=8 control, interleaved
        # ranges. Default 1 because it is the correct address space, not because
        # it paid. See MPK_MERGE_GLOBAL in merge_splitkv.cuh for the samples.
        _mrg = int(os.environ.get("GLM_MERGE_GLOBAL", "1"))
        flags = flags + ["-DMPK_MERGE_GLOBAL=%d" % _mrg]
        # Replace blockDim.x with the compile-time NUM_THREADS in every task
        # kernel's grid-stride loop. blockDim.x is not a register on gfx9 -- it
        # is a dispatch-packet read (global_load_ushort + s_waitcnt vmcnt(0), a
        # full VMEM drain) that the compiler re-emits on every backedge. Every
        # task kernel is launched at WORKER_NUM_THREADS == 256, so the value is
        # a constant. See MPK_NT in tasks/common/worker_config.h.
        _cbd = int(os.environ.get("GLM_CONST_BLOCKDIM", "1"))
        flags = flags + ["-DMPK_CONST_BLOCKDIM=%d" % _cbd]
        # Hoist the un-absorbed W_UV GEMV out of the MoE half and run it in the
        # attention half, straight after the split-KV merge, behind a PAIR-LOCAL
        # barrier instead of a GPU-wide one. The layer's existing Phase 8
        # attention -> o_proj rendezvous then covers W_UV -> o_proj as well, so
        # the MoE half's own W_UV barrier (767) is deleted outright: ten
        # per-layer rendezvous become nine, and the one that goes is 8.28
        # us/layer of *uniform* within-rank spin (ab10e49).
        #
        # Legal because the merge -> W_UV dependency is pair-local. W_UV on XCD
        # x covers heads [8x, 8x+8), which is q_group x/2, and the merge map
        # already gives q_group k to XCDs 2k and 2k+1 (see the PAIR_MERGE note
        # in gang_mla_attn_fused_mi300.cuh). W_UV -> o_proj stays GPU-wide --
        # o_proj contracts over all 64 heads -- which is exactly what Phase 8
        # already is. Compile-time and in MPK_FORWARD_VARS: it moves a barrier's
        # writer set, so a rank that misses it deadlocks.
        if int(os.environ.get("MPK_WUV_IN_MERGE", "0")) == 1:
            flags = flags + ["-DMPK_WUV_IN_MERGE"]
        # Ablation for skipping the o_proj GEMV's activation re-stage on the
        # second grid-stride pass. m_tiles is 1 there, so the two passes stage
        # the same 64 KB row; =1 restores the redundant copy.
        if int(os.environ.get("GLM_OPROJ_RESTAGE", "0")) == 1:
            flags = flags + ["-DMPK_GLM_OPROJ_RESTAGE"]
        if int(os.environ.get("MPK_MOE_SUBPHASE", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_MOE_SUBPHASE"]
        if int(os.environ.get("MPK_FUSED_PHASE_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_FUSED_PHASE_TIMING"]
        if int(os.environ.get("MPK_SPAN_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_SPAN_TIMING"]
        if int(os.environ.get("USE_GANG", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_GANG_TASKS"]
        if int(os.environ.get("USE_NT_WEIGHTS", "0")) == 1:
            flags = flags + ["-DMPK_NT_WEIGHT_LOADS"]
        if int(os.environ.get("W13_LDS_WEIGHTS", "0")) == 1:
            flags = flags + ["-DMPK_W13_LDS_WEIGHTS"]
        if int(os.environ.get("W13_LDS_PREFETCH", "1")) == 1:
            flags = flags + ["-DMPK_W13_LDS_PREFETCH"]
        if int(os.environ.get("WUV_MFMA", "0")) == 1:
            flags = flags + ["-DMPK_WUV_MFMA"]
        if int(os.environ.get("W13_QFIRST", "0")) == 1:
            flags = flags + ["-DMPK_W13_QFIRST"]
        if int(os.environ.get("W13_ILV", "0")) == 1:
            flags = flags + ["-DMPK_W13_ILV"]
        if int(os.environ.get("W13_BIASPRE", "0")) == 1:
            flags = flags + ["-DMPK_W13_BIASPRE"]
        if int(os.environ.get("QKV_RMSPRE", "0")) == 1:
            # QKV pass 1's loads issued ABOVE the weight buffer_load_lds block.
            # MEASURED NULL: p1 1.76 -> 1.53 and drain 0.70 -> 0.50, but pre
            # 1.44 -> 2.02 -- issue backpressure just relocates the stall into
            # the issue block. 2.150/2.123 vs 2.116 baseline. Kept because the
            # negative result is what rules the approach out.
            flags = flags + ["-DMPK_QKV_RMSPRE"]
        if int(os.environ.get("TK_DEFER", "0")) == 1:
            flags = flags + ["-DMPK_TK_DEFER"]
        if int(os.environ.get("TK_NOELOAD", "0")) == 1:
            flags = flags + ["-DMPK_TK_NOELOAD"]
        if int(os.environ.get("EP9_DIRECT", "0")) == 1:
            flags = flags + ["-DMPK_EP9_DIRECT=1"]
        if int(os.environ.get("MPK_GAP_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_GAP_TIMING"]
            flags = flags + ["-DMPK_ENABLE_DEVICE_TASK_ACCUM"]
        if int(os.environ.get("MPK_HBM_LATENCY", "0")) == 1:
            flags = flags + ["-DMPK_HBM_LATENCY"]
        if int(os.environ.get("MPK_MOE_SINGLE_EXPERT", "0")) == 1:
            flags = flags + ["-DMPK_MOE_SINGLE_EXPERT"]
        if int(os.environ.get("MPK_FUSED_TAIL_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_FUSED_TAIL_TIMING"]
        if int(os.environ.get("MPK_K2944_DEBUG", "0")) == 1:
            flags = flags + ["-DMPK_K2944_DEBUG"]
        if int(os.environ.get("MPK_EP9_ONLY", "0")) == 1:
            # Phase 9 breakdown only: keeps the accumulate-and-dump-once EP9
            # report and compiles out the per-layer FUSED_PHASE printf, which
            # on its own takes the iteration from 2.5 ms to ~440 ms.
            flags = flags + ["-DMPK_EP9_ONLY", "-DMPK_ENABLE_DEVICE_TASK_TIMING"]
        if int(os.environ.get("MPK_EP_WAIT_AT_USE", "0")) == 1:
            # Moves Phase 9d's cross-GPU peer wait to its point of use in the
            # next layer's QKV prologue. Both placements are CORRECT -- the
            # gather buffers are per-layer, so deferring the wait exposes no
            # WAR hazard -- so unlike MPK_EP_ABLATE this is a scheduling knob
            # and its output must still pass the correctness suite.
            flags = flags + ["-DMPK_EP_WAIT_AT_USE=1"]
        if int(os.environ.get("MPK_MOE_NOPAD", "0")) == 1:
            # Drops the 240-tile W13 padding when W13 already fits in one
            # round, so the freed slots carry real W2 tiles that can overlap
            # their weight prefetch with the W13 phase. EP-only in practice
            # (1-GPU W13 overflows the round and keeps the padding). CORRECT
            # output: it reorders which worker runs which tile, not what any
            # tile computes.
            flags = flags + ["-DMPK_MOE_NOPAD"]
        _w13_early = int(os.environ.get("MPK_W13_EARLY_REL", "0"))
        if _w13_early:
            # Fire the W13->W2 release at _w13_early/16 of the W13 arrivals.
            # WRONG OUTPUT by construction -- prices the ceiling of any
            # dependency-narrowing scheme (MoK-style indexed counters, W2
            # split-K with a half-width required_count) before building one.
            # If 8/16 does not move the token, the W2 wait is W13's DURATION,
            # not its arrival count, and narrowing the count cannot help.
            flags = flags + [f"-DMPK_W13_EARLY_REL={_w13_early}"]
        if int(os.environ.get("MPK_MLA_SKIP_DECODE", "0")) == 1:
            # Delete the Phase 5 decode loop, keeping every barrier and every
            # other phase. WRONG OUTPUT by construction. The companion probe to
            # MPK_W13_EARLY_REL, for the other half of the layer: it prices the
            # 1.88 ms/iter that gang_mla_full_layer_fused_mi300.cuh:1391
            # attributes to 232 workers waiting on the 16 that run the decode.
            flags = flags + ["-DMPK_MLA_SKIP_DECODE"]
        if int(os.environ.get("MPK_QB_SKIP_PEER_WAIT", "0")) == 1:
            # Delete the q_b head-shard CROSS-RANK gather (7 peer stores + the
            # poll of all 7 peers), keeping the local barrier, the flag release
            # and every other phase. WRONG OUTPUT by construction: the query
            # row keeps the peers' previous-layer heads.
            #
            # Prices the ceiling on head-sharding attention end-to-end. That
            # rewrite would DELETE this rendezvous rather than narrow one, so
            # the usual "skew relocates" verdict does not apply on its face --
            # but the deciding number is what survives at the WALL after S22
            # and S28 re-absorb the freed skew, not what leaves S19->S20.
            flags = flags + ["-DMPK_QB_SKIP_PEER_WAIT"]
        _wuv_abl = int(os.environ.get("MPK_WUV_SKIP_PEER_WAIT", "0"))
        if _wuv_abl == 2:
            # Level 2: keep the rendezvous (the leader's peer signal and the
            # poll of all peers) and delete ONLY the per-tile data push into
            # the peers' copies of mla_v_out. Bisects "the collective hangs"
            # into "the handshake hangs" vs "the data stores land somewhere
            # they should not". WRONG OUTPUT by construction.
            flags = flags + ["-DMPK_WUV_ABL=2"]
        if _wuv_abl == 3:
            # Level 3: full collective, but the leader's peer poll is bounded
            # and dumps (expected, remaining mask, all four slot values) once
            # before giving up. Diagnostic only -- it releases on a poll that
            # never satisfied, so the output is wrong past that layer.
            flags = flags + ["-DMPK_WUV_ABL=3"]
        if _wuv_abl == 1:
            # The W_UV twin of the knob above: delete the W_UV row-shard
            # CROSS-RANK all-gather (the per-tile peer stores and the leader's
            # poll of all peers) while keeping the rank shard itself -- the
            # sliced weight, the biased row base, the local barrier and the
            # flag release. WRONG OUTPUT by construction: 3/4 of the v row
            # keeps whatever the peers left there last layer.
            #
            # This is the bisect for "does GLM_WUV_TP=1 hang in the geometry
            # or in the collective". It is not a perf probe; the additive
            # rule in glm-additive-probes-overprice-deletions applies.
            flags = flags + ["-DMPK_WUV_SKIP_PEER_WAIT"]
        if int(os.environ.get("MPK_ATTN_HALFK", "0")) == 1:
            # Halve the K-loop of every MXFP8 attention/dense GEMM (qkv_a, q_b,
            # o_proj, W_UV, W_UK) at an unchanged tile map and WG stride.
            # WRONG OUTPUT by construction. Upper bound on the MXFP4 lever for
            # those weights, which are 41% of all bytes moved per token.
            flags = flags + ["-DMPK_ATTN_HALFK"]
        if int(os.environ.get("MPK_ABL_QKV", "0")) == 1:
            # Delete the Phase 1 qkv_a tile loop, keeping every barrier and
            # every other phase. WRONG OUTPUT by construction.
            flags = flags + ["-DMPK_ABL_QKV"]
        if int(os.environ.get("MPK_ABL_QKV_PRO", "0")) == 1:
            # Keep qkv_a's GEMM, run 1 of 6 passes of its resadd+RMSNorm
            # prologue. WRONG OUTPUT by construction. Splits SP4[0] into
            # prologue and GEMM; MPK_ABL_QKV is the sum of the two.
            flags = flags + ["-DMPK_ABL_QKV_PRO"]
        if int(os.environ.get("MPK_QKV_EP_FOLD", "0")) == 1:
            # Hoist the EP reduction out of qkv_a's tile: six workgroups per
            # XCD sum the 8 peer slots once, an XCD-local release publishes
            # the resolved row, and the 24 tiles read one plane instead of
            # eight. CORRECT OUTPUT -- same addresses, same summation order,
            # same bf16 rounding. Removes 20.7 -> 2.6 MB/layer/rank.
            flags = flags + ["-DMPK_QKV_EP_FOLD"]
        if int(os.environ.get("MPK_QKV_PRO_HOIST", "0")) == 1:
            # Hoist the WHOLE qkv_a prologue, not just the fold: the same six
            # workgroups per XCD fold, block-reduce the sum of squares across
            # the six slices, normalize and quantize, and publish E4M3 + one
            # E8M0 per 128 into the tail of rmsnorm_out. The 24 tiles copy 6192
            # bytes into LDS instead of staging 24 KB of bf16 and re-deriving
            # the norm 24 times. 48fea7f measured that prologue at 40.4% of a
            # 17.20 us tile against a K-loop already at 90% of its byte roof.
            # Supersedes MPK_QKV_EP_FOLD, which this disables. Compile-time and
            # in MPK_FORWARD_VARS: it changes the XCD-local release's writer
            # set, so a rank that misses it deadlocks.
            flags = flags + ["-DMPK_QKV_PRO_HOIST"]
        if int(os.environ.get("MPK_QKV_FOLD_ROWS", "0")) == 1:
            # Put the batch row on the MFMA's output columns instead of on
            # qkv_a's tile index. The 16x16x128 scaled MFMA computes 16 output
            # columns and at bs=1 the epilogue reads one; the B-operand gather
            # addresses by k-block only, so every lane already feeds the same
            # token. Folded, lane column `col` feeds row `col`, the tile space
            # drops from BATCH_SIZE * n_wgs to n_wgs -- one grid-stride round
            # instead of two at bs=2 -- and the weight slab is fetched once per
            # tile instead of once per (tile, row). No-op at BATCH_SIZE 1.
            # Compile-time and in MPK_FORWARD_VARS: it changes the tile space,
            # and a rank that misses it disagrees on the phase's work.
            flags = flags + ["-DMPK_QKV_FOLD_ROWS"]
        if int(os.environ.get("MPK_BAR_TREE", "0")) == 1:
            # Two-level arrival for every GPU-wide Mechanism-C rendezvous:
            # 29 atomics on the XCD's own line, then 8 on the global one,
            # instead of 232 on a single line. Correct output -- the barrier
            # semantics are unchanged, only who counts. Priced by the null
            # probe below at 3.77 -> 2.11 us per rendezvous.
            flags = flags + ["-DMPK_BAR_TREE=1"]
        _bfm = os.environ.get("MPK_BAR_FLAG_MAX")
        if _bfm is not None:
            # Publish every Mechanism-C release flag with an atomicMax instead
            # of a plain write-through store, so a late lower epoch is dropped
            # (mpk_atoms.cuh:444, st_flag_u32). Header default is 1. It closes
            # a real backwards-stepping-flag race, but it was never priced at
            # the wall -- 30 sites, 222 flat_atomic_smax in the image -- so
            # plumb it here to A/B it against 0 in one batch.
            assert _bfm in ("0", "1"), "MPK_BAR_FLAG_MAX is 0 or 1"
            flags = flags + [f"-DMPK_BAR_FLAG_MAX={_bfm}"]
        _bph = os.environ.get("MPK_BAR_PEER_HEAL")
        if _bph is not None:
            # The drift-immune arm of every Mechanism-C self-heal
            # (mpk_atoms.cuh, hier_barrier_should_heal). The counter-quota
            # test it ORs with is permanently false past ~epoch 155 because
            # EP_TAIL_ONLY advances the epoch without arriving, one round of
            # drift per generated token; the peer scan asks instead whether
            # some OTHER XCD's flag already reached this round, which proves
            # the election fired and cannot release early. Header default 1.
            assert _bph in ("0", "1"), "MPK_BAR_PEER_HEAL is 0 or 1"
            flags = flags + [f"-DMPK_BAR_PEER_HEAL={_bph}"]
        _bpn = os.environ.get("MPK_BAR_POLL_NT")
        if _bpn is not None:
            assert _bpn in ("0", "1"), "MPK_BAR_POLL_NT is 0 or 1"
            # The `nt` hint on the intra-GPU barrier flag poll. 1 restores the
            # historical form; 0 (the header default) drops it. Coherence is
            # unchanged either way -- `sc0 sc1` still bypasses L1 and the
            # per-XCD L2 -- so this is a replacement-policy change on the most
            # read-shared line in the kernel, worth -0.36 us per rendezvous in
            # tests/standalone/test_barrier_release.hip.
            flags = flags + [f"-DMPK_BAR_POLL_NT={_bpn}"]
        _ppn = os.environ.get("MPK_PEER_POLL_NT")
        if _ppn is not None:
            assert _ppn in ("0", "1"), "MPK_PEER_POLL_NT is 0 or 1"
            # The same `nt` drop on the CROSS-RANK peer poll (ld_sys_u64 and
            # ld_sys_u64_x8): the EP signal wait and the q_b gather. 1 restores
            # the historical form, 0 (the header default) drops it. `sc0 sc1`
            # is untouched, so the stale-L2 livelock this poll was hardened
            # against cannot come back.
            flags = flags + [f"-DMPK_PEER_POLL_NT={_ppn}"]
        if int(os.environ.get("MPK_EP_POLL_BATCH", "0")) == 1:
            # One s_waitcnt for all eight EP signal lines instead of one per
            # peer. The seven-line pass was seven serialized uncached round
            # trips, measured at 9.65 us on the rank that waits least.
            flags = flags + ["-DMPK_EP_POLL_BATCH=1"]
        _ep_fold_wgs = int(os.environ.get("MPK_EP_FOLD_WGS", "0"))
        if _ep_fold_wgs > 0:
            # Folding work-groups per XCD. Default 1 (eight total) is the
            # shape this branch has always had.
            flags = flags + ["-DMPK_EP_FOLD_WGS=%d" % _ep_fold_wgs]
        if int(os.environ.get("MPK_ML_PTR_PREFETCH", "0")) == 1:
            # Hold the next layer's 34+13 TaskDesc pointers in registers
            # across the layer instead of loading them at the layer boundary.
            # Stage stamp 12 priced that copy at 15.33 us/layer -- two
            # dependent cold round trips, because the loads feed shared
            # memory and the layer evicts the table from L2 in between.
            flags = flags + ["-DMPK_ML_PTR_PREFETCH=1"]
        _abl_mlb = int(os.environ.get("MPK_ABL_ML_BOUNDARY", "0"))
        if _abl_mlb > 0:
            # WRONG OUTPUT ceiling probe. Deletes the multi-layer loop's
            # per-layer boundary cost -- the pointer-refresh __syncthreads,
            # and at level 2 the inter-layer threadfence_gpu too -- and
            # implies MPK_ML_PTR_PREFETCH so the 34+13 pointer entries come
            # from registers rather than two dependent cold round trips.
            # The loads still happen, one layer early, so the 65 MB/layer
            # weight stream is unchanged and the win cannot be a cache
            # artifact. See the long note in mpk_atoms.cuh.
            assert _abl_mlb in (1, 2), "MPK_ABL_ML_BOUNDARY is 1 or 2"
            flags = flags + ["-DMPK_ABL_ML_BOUNDARY=%d" % _abl_mlb]
        if int(os.environ.get("GLM_OPROJ_MXFP4", "1")) == 1:
            # o_proj's GEMV reads E2M1 nibbles instead of E4M3 bytes. Must
            # agree with the host packer -- demo.py reads the SAME env var to
            # decide whether to hand pack_dense_mxfp8 a quantize_mxfp4 output,
            # and a mismatch mis-addresses every weight row rather than
            # failing to build. o_proj is 100.7 MB/layer/GPU of the ~166 MB of
            # attention weight and is the one stage measured byte-bound (76%
            # of HBM peak), so it is the member of the MXFP4-attention set
            # where halving bytes should convert to time.
            flags = flags + ["-DMPK_OPROJ_MXFP4=1"]
        _ml_pad = int(os.environ.get("MPK_ML_BOUNDARY_PAD", "0"))
        if _ml_pad > 0:
            # Nanoseconds of uniform delay injected into the multi-layer
            # loop's per-layer boundary, paid by every thread of every worker.
            # CORRECT OUTPUT -- an additive pricing probe, so its wall number
            # is valid and gateable. Measures the slope of wall against
            # boundary time, which upper-bounds what deleting the real
            # 15.33 us of boundary bookkeeping could ever buy.
            assert 0 < _ml_pad <= 100000, "MPK_ML_BOUNDARY_PAD is 1..100000 ns"
            flags = flags + [f"-DMPK_ML_BOUNDARY_PAD={_ml_pad}"]
        _shadow_kb = int(os.environ.get("MPK_MOE_SHADOW_KB", "0"))
        if _shadow_kb > 0:
            # Item-2 capacity probe. CORRECT OUTPUT: the W13-idle workers
            # pull a qkv_a-sized dose of cold o_proj weight during the W13
            # phase and discard it behind an untakeable branch. Answers
            # "is there a free worker-group-shaped hole in the MoE phase"
            # without touching the router's inputs.
            assert 0 < _shadow_kb <= 4096, "MPK_MOE_SHADOW_KB is 1..4096"
            flags = flags + ["-DMPK_MOE_SHADOW_KB=%d" % _shadow_kb]
        _qkva_pf_kb = int(os.environ.get("MPK_QKVA_PF_KB", "0"))
        if _qkva_pf_kb > 0:
            # The real lever the shadow probe stood in for. Each W13-idle
            # worker pulls this many KB of the NEXT layer's qkv_a weight into
            # cache during this layer's W13 phase. CORRECT OUTPUT: it reads a
            # constant nobody writes and stores nothing.
            assert 0 < _qkva_pf_kb <= 4096, "MPK_QKVA_PF_KB is 1..4096"
            flags = flags + ["-DMPK_QKVA_PF_KB=%d" % _qkva_pf_kb]
            _qkva_pf_at = int(os.environ.get("MPK_QKVA_PF_AT", "2"))
            assert _qkva_pf_at in (2, 13), "MPK_QKVA_PF_AT is 13 (W13 hole) or 2 (W2 hole)"
            flags = flags + ["-DMPK_QKVA_PF_AT=%d" % _qkva_pf_at]
        _qkva_reps = int(os.environ.get("MPK_QKVA_REPS", "1"))
        if _qkva_reps != 1:
            # Item-2 dependency-half pricing probe. CORRECT OUTPUT: the qkv_a
            # tile loop's body is idempotent on the unfolded path, so running
            # it N times writes the same bytes. Prices what an extra
            # un-hidden qkv_a pass costs, which is exactly the second GEMM the
            # RMSNorm-linearity split would add. See mpk_atoms.cuh.
            assert 1 <= _qkva_reps <= 4, "MPK_QKVA_REPS is 1..4"
            flags = flags + ["-DMPK_QKVA_REPS=%d" % _qkva_reps]
        _w13_reps = int(os.environ.get("MPK_W13_REPS", "1"))
        if _w13_reps != 1:
            # Marginal-cost probe for the MoE W13 phase. CORRECT OUTPUT --
            # W13's epilogue is pure stores into the swiglu scratch. See
            # mpk_atoms.cuh.
            assert 1 <= _w13_reps <= 4, "MPK_W13_REPS is 1..4"
            flags = flags + ["-DMPK_W13_REPS=%d" % _w13_reps]
        _bs_debug = int(os.environ.get("MPK_BS_DEBUG", "0"))
        if _bs_debug != 0:
            # Per-stage activation checksums for the first N fused layers, one
            # printf per (layer, stage) for the whole run. CORRECTNESS tool --
            # it does not change any value, but do not quote a latency with it
            # on. See mpk_bsdbg.cuh.
            assert 1 <= _bs_debug <= 16, "MPK_BS_DEBUG is 1..16 (layers)"
            flags = flags + ["-DMPK_BS_DEBUG=%d" % _bs_debug]
            # Which forward pass to dump, as the task_layer_idx of its layer 0.
            # task_layer_idx is run-monotonic, so iteration k of a 76-layer
            # multi-layer scan starts at 76 * k. Default 0 is iteration 0, the
            # original behaviour.
            _bs_l0 = int(os.environ.get("MPK_BSDBG_LAYER0", "0"))
            assert 0 <= _bs_l0 <= 100000, "MPK_BSDBG_LAYER0 is 0..100000"
            flags = flags + ["-DMPK_BSDBG_LAYER0=%d" % _bs_l0]
        _bar_skew = int(os.environ.get("MPK_BAR_SKEW", "0"))
        if _bar_skew >= 1:
            # Per-rendezvous first-arriver-to-last-arriver spread. O(1) per
            # barrier epoch, so unlike MPK_SUBPHASE_TIMING its cost does not
            # scale with tile count.
            flags = flags + ["-DMPK_BAR_SKEW=%d" % _bar_skew]
            _drop_ns = int(os.environ.get("MPK_BAR_SKEW_DROP_NS", "0"))
            if _drop_ns > 0:
                # Stale-reference drop threshold for the stage stamps. The
                # 10 ms default lets one sample per decode iteration -- the
                # one whose reference is the previous iteration's last
                # barrier -- survive with ~10000x the weight of a real
                # 136 us layer. Set 1000000 for a clean per-layer mean.
                flags = flags + ["-DMPK_BAR_SKEW_DROP_NS=%dull" % _drop_ns]
        if int(os.environ.get("MPK_QUANT_V16", "0")) == 1:
            # 16-byte loads in the shared RMSNorm+quant prologue. The scalar
            # form only vectorized to dwordx2 because the addrspace(1) cast
            # hides the real 16-byte alignment. Bit-identical arithmetic.
            flags = flags + ["-DMPK_QUANT_V16=1"]
        _null_phases = int(os.environ.get("MPK_NULL_PHASES", "0"))
        if _null_phases:
            # Insert N extra GPU-wide rendezvous at the head of every layer,
            # doing no work between them. CORRECT OUTPUT -- this is a pricing
            # probe, not a ceiling probe, so its wall number is valid and it
            # can be gated by the correctness suite like any real change.
            assert 0 <= _null_phases <= 4, "MPK_NULL_PHASES is 0..4"
            flags = flags + [f"-DMPK_NULL_PHASES={_null_phases}"]
            if int(os.environ.get("MPK_NULL_TREE", "0")) == 1:
                # Same null rendezvous, two-level arrival: 29 atomics on the
                # XCD's own line, then 8 on the global one, instead of 232 on
                # one line. The controlled A/B for "is the barrier cost just
                # serialized atomics on a single line?".
                flags = flags + ["-DMPK_NULL_TREE=1"]
            _null_tiles = int(os.environ.get("MPK_NULL_TILES", "0"))
            if _null_tiles:
                # Put an empty grid-stride tile loop in front of each null
                # rendezvous, so the probe prices a whole ROUND instead of
                # just its barrier. 24 matches qkv_a's tiles/XCD.
                assert 0 < _null_tiles <= 256, "MPK_NULL_TILES is 1..256"
                flags = flags + [f"-DMPK_NULL_TILES={_null_tiles}"]
        _perf_iter = int(os.environ.get("MPK_PERFETTO", "0"))
        if _perf_iter:
            # Capture raw per-worker phase spans for ONE decode iteration and
            # dump them as [PERF] CSV lines; perf_to_perfetto.py turns those
            # into a Perfetto trace. Correct output -- this only stores
            # timestamps the fused task already takes. Implies device timing,
            # which is where those timestamps come from, and EP9_ONLY to
            # suppress the per-layer printf storm that would otherwise
            # dominate the very timeline being captured.
            flags = flags + [
                "-DMPK_PERFETTO",
                f"-DMPK_PERFETTO_ITER={_perf_iter}",
                "-DMPK_ENABLE_DEVICE_TASK_TIMING",
                "-DMPK_EP9_ONLY",
            ]
        for _nq in ("MPK_W13_NOQUANT", "MPK_W2_NOQUANT"):
            if int(os.environ.get(_nq, "0")) == 1:
                # Skip the per-tile FP8 token quant. WRONG OUTPUT by
                # construction -- prices the ceiling of hoisting the quant,
                # which is redundant across every tile of an expert (all 92 W13
                # / 45 W2 workgroups quantize the same vector). The quant also
                # hides the weight-prefetch latency, so the measured delta is
                # the NET win a real hoist could deliver, not the gross 1.51 us.
                flags = flags + [f"-D{_nq}"]
        if int(os.environ.get("MPK_W2_HALFK", "0")) == 1:
            # Halves W2's MFMA iteration count to price a K-split of W2 against
            # the W13 -> barrier -> W2 chain that sets Phase 8's length. See the
            # W2_MFMA_ITERS definition in gang_moe_fused_mxfp4_mi300.cuh.
            # WRONG OUTPUT: latency attribution only.
            flags = flags + ["-DMPK_W2_HALFK"]
        if int(os.environ.get("MPK_W2_SPLITK", "0")) == 1:
            # Splits W2's K in two, doubling the W2 tile count so each worker
            # loads half the weight bytes. Under EP a rank owns 2 experts, so
            # W2 is 92 tiles on 240 workers (38% occupancy); the split takes it
            # to 184. Both halves atomicAdd into moe_workspace_f32, so no new
            # combine is needed. CORRECT output: it redistributes the reduction
            # rather than dropping it (unlike MPK_W2_HALFK above).
            flags = flags + ["-DMPK_W2_SPLITK"]
        _moe_lb = os.environ.get("MPK_MOE_LIVE_BOUND")
        if _moe_lb is not None:
            # Default is 1 in the header; only forward an explicit override so
            # the A/B is one -D and every rank builds the same thing.
            assert _moe_lb in ("0", "1"), "MPK_MOE_LIVE_BOUND is 0 or 1"
            flags = flags + [f"-DMPK_MOE_LIVE_BOUND={_moe_lb}"]
        _vprobe = os.environ.get("MPK_VPROBE")
        if _vprobe is not None and _vprobe != "0":
            # Measurement only: prints the activated-expert UNION size U from
            # one worker per rank every <stride> routing epochs, so the MoE
            # row fold's ceiling V = 2*TOPK - (U - s) can be priced instead of
            # assumed. Does not change any tile's work.
            assert _vprobe.isdigit(), "MPK_VPROBE is a non-negative integer"
            flags = flags + [f"-DMPK_VPROBE={_vprobe}"]
        _pipe = int(os.environ.get("MPK_ABL_PIPE_W13W2", "0"))
        if _pipe:
            # Adjacent-phase overlap ceiling probe. 1 = control, CORRECT
            # output; 2 = probe, WRONG OUTPUT (the moved W2 tiles run above
            # the W13 -> W2 rendezvous). Decide on 2 vs 1, not 2 vs default.
            # Long note at the define in mpk_atoms.cuh.
            assert _pipe in (1, 2), "MPK_ABL_PIPE_W13W2 is 0, 1 or 2"
            flags = flags + ["-DMPK_ABL_PIPE_W13W2=%d" % _pipe]
        _shdup = os.environ.get("MPK_SHARED_DUP")
        if _shdup is not None and _shdup != "0":
            # Shared-expert makespan pricing probe. Runs rank 0's shared-expert
            # W13 tiles 1 + MPK_SHARED_DUP times, storing the same bits every
            # time: CORRECT OUTPUT, additive, one variable. The value is the
            # number of EXTRA copies, so 1 doubles the excess and 2 triples it
            # -- two magnitudes are what separate a LINEAR peer-idle->wall
            # response from a slack THRESHOLD. W13 only -- W2's atomicAdd
            # epilogue would double-count. Needs MPK_MOE_LIVE_BOUND=1 (the
            # default) so the widened W13 tile space is actually walked. Long
            # note at the define in mpk_atoms.cuh. Compile-time, so every rank
            # must see it or the ranks build different megakernels.
            assert _shdup in ("1", "2", "3"), (
                "MPK_SHARED_DUP is 0 (off) or the number of EXTRA shared-expert "
                "W13 copies, 1..3"
            )
            assert os.environ.get("MPK_MOE_LIVE_BOUND", "1") == "1", (
                "MPK_SHARED_DUP needs the live clamp: with MPK_MOE_LIVE_BOUND=0 "
                "the W13 bound is the full static tile space, the duplicate "
                "tiles are already inside it, and the probe silently does "
                "nothing"
            )
            flags = flags + ["-DMPK_SHARED_DUP=%s" % _shdup]
        _wpe = os.environ.get("MPK_WORKER_WAVES_PER_EU")
        if _wpe is not None:
            # The megakernel's register budget -- MIN_WARPS_PER_EXECUTION_UNIT
            # on both persistent_kernel and worker_kernel. 3 is what puts the
            # image at 252 unified VGPRs (granule 256) and therefore 2
            # waves/SIMD; 2 is WORSE than the default because of the
            # independent-maxima sum rule. Long note at the define in
            # persistent_kernel.cuh. Compile-time, so every rank must agree.
            assert _wpe in ("1", "2", "3", "4"), "MPK_WORKER_WAVES_PER_EU 1..4"
            flags = flags + [f"-DMPK_WORKER_WAVES_PER_EU={_wpe}"]

        _lds = os.environ.get("MPK_WORKER_LDS_KB")
        if _lds is not None:
            # The per-block dynamic LDS request, in KB. The default 155 of 160
            # KB/CU is what pins the megakernel to 1 block/CU regardless of
            # registers; 78 is the largest value that still leaves room for a
            # second block. Long note at the define in runtime_header.h.
            # Compile-time, so every rank must agree.
            assert 8 <= int(_lds) <= 155, "MPK_WORKER_LDS_KB 8..155"
            flags = flags + [f"-DMPK_WORKER_LDS_KB={_lds}"]

        _tds = os.environ.get("MPK_TASK_DESC_SLOTS")
        if _tds is not None:
            # How many TaskDesc slots execute_worker's LDS staging buffer
            # holds. The shipping image hardcodes 16 while every consumer
            # clamps to TASK_DESCS_BUFFER_LENGTH (7 at sizeof(TaskDesc) ~424),
            # so 9 slots are allocated and never read -- 3816 B of static LDS
            # that is THE reason 2 * (static + dyn) overran 160 KB/CU and left
            # a 464-worker grid at 1 block/CU. Dropping to 7 cuts
            # group_segment_fixed_size 7328 -> 3512 B and opens that gate, but
            # it also slides the dynamic-LDS base, which every task carves at
            # fixed offsets, so it is swept rather than assumed free.
            # Compile-time, so every rank must agree.
            assert 4 <= int(_tds) <= 16, "MPK_TASK_DESC_SLOTS 4..16"
            flags = flags + [f"-DMPK_TASK_DESC_SLOTS={_tds}"]

        _tdp = os.environ.get("MPK_TASK_DESC_PAD")
        if _tdp is not None:
            # Dead LDS bytes that hold the dynamic-segment base where the task
            # swizzles expect it. Shrinking MPK_TASK_DESC_SLOTS alone costs
            # +1.49 ms at the wall with registers held identical, because the
            # dynamic base slides and every _fused_smem carve changes bank
            # phase. Long note at the array in persistent_kernel.cuh.
            # Compile-time, so every rank must agree.
            assert 0 <= int(_tdp) <= 8192, "MPK_TASK_DESC_PAD 0..8192"
            flags = flags + [f"-DMPK_TASK_DESC_PAD={_tdp}"]

        _pf = os.environ.get("MPK_MOE_PF_GROUPS")
        if _pf is not None:
            # MoE k-loop prefetch distance: how many k-groups are in flight
            # while that many are being consumed. 0 keeps the shipping loop,
            # which the ISA shows issues its loads and then waits on them
            # immediately -- no load/MFMA overlap at all. 4 is the measured
            # knee (11.80 us/tile against 16.22, standalone); 16 is worse.
            # Long note at the define in gang_moe_linear_mxfp8_mi300.cuh.
            # Compile-time, so every rank must agree.
            assert _pf in ("0", "2", "4", "8", "16"), \
                "MPK_MOE_PF_GROUPS is 0, 2, 4, 8 or 16"
            flags = flags + [f"-DMPK_MOE_PF_GROUPS={_pf}"]

        # Per-kernel prefetch depth. The C++ has carried
        # MPK_MOE_PF_GROUPS_W13 / _W2 since the two kernels were shown to want
        # opposite depths, but neither was ever reachable from a run: the only
        # env knob was the unified one above, whose membership assert excludes
        # every odd divisor, so W13's 3 / 6 / 12 / 24 could not be selected at
        # all. That is task #134's blocker, not a preference.
        #
        # Why re-taking this sweep is worth a build. The recorded numbers say
        # W13 depth 6 is -0.167 ms and depth 8 -0.256 ms, reproducible to
        # 0.001 ms across separate batches -- and they are all RETRACTED,
        # because depth 8 answered "The capital of France is" with 135 of 136
        # tokens of one bigram. Root cause was found (task #133) and FIXED:
        # batching the token-scale loads made s_tok_scales[k] provably
        # wave-uniform, LLVM scalarised it, and v_mfma_scale_* silently
        # computes garbage from an SGPR scale. The fix is MPK_MFMA_VSCALE,
        # default 1, an asm launder costing zero instructions. So the axis is
        # real, worth ~0.25 ms, and every number on it was measured on a build
        # that no longer exists.
        #
        # tests/standalone/test_moe_kloop_width.hip is the oracle and it now
        # reports MATCH -- bit-identical accumulators, not merely close -- at
        # ship / deep 4,6,8,12,16 / dbuf 4,6,8. Bit-exactness is the right
        # gate: the MFMA chain is `acc = mfma(A[j], B[j], acc)` in ascending j
        # at every width, so a correct deep loop cannot differ.
        #
        # THAT ORACLE IS NECESSARY AND NOT SUFFICIENT. It calls the k-loop on
        # its own buffers; the scalarisation it is standing in for is a
        # whole-function property of how the megakernel's caller feeds
        # s_tok_scales. A width set here needs an IN-SITU gate too, per
        # [[glm-wrong-output-probes-upstream-of-router-are-invalid]] -- the MoE
        # is downstream of its own router but UPSTREAM of the next layer's, so
        # a wrong W13 rewrites every later layer's expert set and the resulting
        # wall number is not a timing of this model.
        #
        # USE THE CHECKSUM, NOT GENERATED TEXT. Text cannot settle a width at
        # any affordable sample count: correctness_gate.py's own header
        # measures ~50% token disagreement between two runs of the SAME build,
        # because the EP fold and the atomic accumulations retire in arrival
        # order and one flipped argmax cascades. Depth 6 spent seven samples
        # producing an unresolvable 3/7-vs-0/4 before this was believed.
        #
        # What settles it, in ONE run per arm:
        #
        #   MPK_BS_DEBUG=2 MPK_BSDBG_LAYER0=0 MAX_NEW_TOKENS=4
        #
        # dumps a double-precision sum, absmax and first two elements of every
        # stage buffer on all four ranks for the first two fused layers, and
        # the deterministic prefix of a decode is long enough for it. Compare
        # arm against control record by record; they must be IDENTICAL, and for
        # depth 6 all 68 were. Stage 5 is W13's input and stage 6 its output,
        # so an identical 5 with a differing 6 localises the fault to this
        # k-loop, and an identical layer-1 stage 0 proves the whole MoE half
        # matched, W2's atomic fold included. Confirm from each run's own log
        # that the builds really differed -- grep the hipcc line for the -D.
        #
        # Both must divide their kernel's trip count or _gang_moe_pf_groups
        # falls back to the shipping loop and the arm silently measures the
        # control. At GLM-5 those are 48 for W13 (6144/128) and 16 for W2
        # (2048/128 at K_SPLITS=1).
        for _var, _trip in (("MPK_MOE_PF_GROUPS_W13", 48),
                            ("MPK_MOE_PF_GROUPS_W2", 16)):
            _v = os.environ.get(_var)
            if _v is None:
                continue
            # Compile-time, so every rank must agree.
            assert _v.isdigit() and (int(_v) == 0 or 2 <= int(_v) <= 24), \
                f"{_var} is 0 (ship loop) or 2..24"
            assert int(_v) == 0 or _trip % int(_v) == 0, \
                (f"{_var}={_v} does not divide the GLM-5 trip count {_trip}; "
                 "_gang_moe_pf_groups would fall back to the shipping loop "
                 "and the arm would measure the control")
            # NBLK degeneracy. The deep loop peels its last block, so the
            # pipelined body runs NBLK-1 = trip/depth - 1 times. At depth 8 on
            # W2 that is ONE trip, LLVM peels it, and the software pipeline
            # stops existing: measured on the image, the W2 kernel goes from
            # two 4-MFMA loops to ZERO MFMA loops with all 32 MFMAs inlined
            # straight-line and AGPR shuffles 16 -> 64. The wall says +0.238 ms
            # against its own control (9.327 n=3 vs 9.089), with worker_kernel
            # unchanged at 284 VGPR / 36 AGPR / 0 spills -- so this is the loop
            # structure collapsing, NOT the register cliff the older W2 note
            # blamed. Require at least two pipelined trips.
            assert int(_v) == 0 or _trip // int(_v) - 1 >= 2, \
                (f"{_var}={_v} leaves {_trip // int(_v) - 1} pipelined trip(s) "
                 f"of the deep loop (trip count {_trip}, last block is "
                 "peeled); LLVM peels a 1-trip loop and the k-loop degenerates "
                 "to straight-line code with no software pipeline at all")
            flags = flags + [f"-D{_var}={_v}"]

        _wg = os.environ.get("MPK_MOE_WGLOBAL")
        if _wg is not None:
            # Address the MoE weight and its E8M0 scales through addrspace(1)
            # AND hoist the whole batch of B-operand ds_reads above the MFMA
            # group. Both halves together: a flat_load increments lgkmcnt as
            # well as vmcnt, so today's three per-MFMA `s_waitcnt lgkmcnt(0)`
            # each drain the weight prefetch. Casting alone was measured
            # +0.87 ms (task #77) because it left those three waits in place
            # and added vmcnt waits on top. Long note at the define in
            # gang_moe_linear_mxfp8_mi300.cuh.
            # Compile-time, so every rank must agree.
            assert _wg in ("0", "1"), "MPK_MOE_WGLOBAL is 0 or 1"
            flags = flags + [f"-DMPK_MOE_WGLOBAL={_wg}"]

        _wgp = os.environ.get("MPK_MOE_WGPTR")
        if _wgp is not None:
            # Makes MPK_MOE_WGLOBAL actually land. The leaf-level cast in
            # _gang_ld_g is inert three inlines below a lambda, so the two MoE
            # k-loops are the ONLY all-flat, all-vmcnt(0) MAC loops left in the
            # megakernel while every other one is global with a partial vmcnt.
            # This threads addrspace(1) through the pointer TYPE instead. Long
            # note at the define in gang_moe_linear_mxfp8_mi300.cuh.
            # Compile-time, so every rank must agree.
            assert _wgp in ("0", "1"), "MPK_MOE_WGPTR is 0 or 1"
            flags = flags + [f"-DMPK_MOE_WGPTR={_wgp}"]

        _scb = os.environ.get("MPK_MOE_SCBASE")
        if _scb is not None:
            # Load the GROUPS-wide E8M0 scale batch off ONE base pointer with
            # immediate offsets instead of GROUPS independent 64-bit address
            # chains. Register-reducing; see load_ws_batch's header in
            # gang_moe_linear_mxfp8_mi300.cuh.
            # Compile-time, so every rank must agree.
            assert _scb in ("0", "1"), "MPK_MOE_SCBASE is 0 or 1"
            flags = flags + [f"-DMPK_MOE_SCBASE={_scb}"]

        _snt = os.environ.get("MPK_MOE_STREAM_NT")
        if _snt is not None:
            # Mark the MoE k-loop weight + E8M0 scale stream non-temporal. It is
            # ~57 MB/layer/rank with zero reuse at every level, yet it carries no
            # cache hint at all, while the W2 prefetch three lines away is
            # already `sc0 sc1 nt`. `nt` is a replacement-policy hint only -- no
            # coherence change. ISA-gated: nt goes 0->32 (W13) and 8->72 (W2)
            # with load/s_waitcnt/mfma counts UNCHANGED.
            # Compile-time, so every rank must agree.
            assert _snt in ("0", "1"), "MPK_MOE_STREAM_NT is 0 or 1"
            flags = flags + [f"-DMPK_MOE_STREAM_NT={_snt}"]

        _bsc = os.environ.get("MPK_MOE_BSCHED")
        if _bsc is not None:
            # Pin the GROUPS-wide batch of B-operand ds_reads above the MFMA
            # group in the MoE k-loop. The vmem side of that loop is already
            # healthy (8 loads carried, zero vmcnt(0)); what is exposed is LDS,
            # because the allocator recycles v[12:19] across three MFMAs and
            # pays two `s_waitcnt lgkmcnt(0)` per trip for the WAR. The sibling
            # dense kernel fixes the identical defect with the identical
            # intrinsic and it is the only sched_barrier in tasks/mi300. Emits
            # no instruction, so the only risk is allocation -- ISA-gate
            # lgkmcnt(0) DOWN, mfma UNCHANGED, vgpr NOT UP, and run
            # tests/standalone/test_mxfp8_moe.hip, which is what catches the
            # recorded silent miscompile class. Long note at the define in
            # gang_moe_linear_mxfp8_mi300.cuh.
            # Compile-time, so every rank must agree.
            assert _bsc in ("0", "1", "2"), "MPK_MOE_BSCHED is 0, 1 or 2"
            flags = flags + [f"-DMPK_MOE_BSCHED={_bsc}"]

        _ant = os.environ.get("MPK_ATTN_STREAM_NT")
        if _ant is not None:
            # The other half of the layer. Same argument as MPK_MOE_STREAM_NT:
            # at bs=1 decode every attention/dense weight byte is read exactly
            # once per token, and the #97 census found ZERO cache hints in
            # gang_mla_full_layer_fused_kernel. Covers _rnlm8_load_w /
            # _rnlm8_load_sc (qkv_a, q_b, the dense MLP, the LM head) and the
            # two GEMV k-loops (W_UK, W_UV, o_proj) -- weight and E8M0 scale
            # only, never the staged activation, which IS re-read.
            # `nt` is replacement policy, not coherence. Compile-time, so
            # every rank must agree.
            # 0 = off (shipping). 1 = weight + E8M0 scale. 2 = weight only.
            # The scale load has intra-loop line reuse -- 32 consecutive k map
            # to one scale byte and 64 scale bytes share a 64 B line -- so `nt`
            # on it evicts a line that IS re-read. Arm 2 separates the two.
            assert _ant in ("0", "1", "2"), "MPK_ATTN_STREAM_NT is 0, 1 or 2"
            flags = flags + [f"-DMPK_ATTN_STREAM_NT={_ant}"]

        _gpf = os.environ.get("MPK_ATTN_GEMV_PF")
        if _gpf is not None:
            # Software-pipeline the weight+scale stream of the MXFP8 GEMV
            # k-loop (W_UK, W_UV, o_proj) one trip deep, paying for the second
            # buffer with the `av` registers the loop currently spends
            # batching LDS reads. See the note in the kernel body: that loop
            # is latency-bound with carry 0, so depth at the top of the trip
            # is the payoff variable.
            assert _gpf in ("0", "1"), "MPK_ATTN_GEMV_PF is 0 or 1"
            flags = flags + [f"-DMPK_ATTN_GEMV_PF={_gpf}"]

        _pfd = os.environ.get("MPK_MOE_PF_DBUF")
        if _pfd is not None:
            # Double-buffered form of the same k-loop. The deep loop's
            # `A[j] = N[j]` backedge copy is a USE of every load destination,
            # so the shipped ISA drains to vmcnt(0) twice per trip and the
            # outstanding count returns to zero every k-block -- the unroll=1
            # point (3446 GB/s) on the curve in
            # tests/standalone/test_waves_per_simd_payoff.hip. 1 swaps two
            # buffers instead of copying. Needs an even NBLK >= 4, which both
            # GLM-5 MoE shapes have at GROUPS=4; anything else silently keeps
            # the copying form. Compile-time, so every rank must agree.
            assert _pfd in ("0", "1"), "MPK_MOE_PF_DBUF is 0 or 1"
            flags = flags + [f"-DMPK_MOE_PF_DBUF={_pfd}"]

        # W13 and W2 select the loop form and the prefetch width separately.
        # The +29 unified VGPR bill that killed MPK_MOE_PF_DBUF as a single
        # knob is entirely W2's: at GROUPS=4 the image reads 335 for
        # W13-dbuf/W2-ship and 364 for W13-ship/W2-dbuf, against a 335
        # shipping baseline. Long note at the defines in
        # gang_moe_linear_mxfp8_mi300.cuh. All compile-time, so every rank
        # must agree.
        for _v in (
            "MPK_MOE_PF_DBUF_W13",
            "MPK_MOE_PF_DBUF_W2",
            # Same shape of fix one phase over: the MLA decode's KV prefetch is
            # a single register buffer, so the tile t+2 issue is a WAR hazard on
            # the tile t+1 drain and cannot be hoisted above it. Alternating two
            # buffers lets the loads live across the backedge. Long note at
            # kv_pre_odd in gang_mla_decode_mi300.cuh. Compile-time, so every
            # rank must agree.
            "MPK_MLA_DECODE_DBLBUF",
        ):
            _x = os.environ.get(_v)
            if _x is not None:
                assert _x in ("0", "1"), f"{_v} is 0 or 1"
                flags = flags + [f"-D{_v}={_x}"]
        for _v in (
            "MPK_MOE_PF_GROUPS_W13",
            "MPK_MOE_PF_GROUPS_W2",
        ):
            # Range is 0..16, not 0..8. The single knob MPK_MOE_PF_GROUPS
            # asserts membership in ("0","2","4","8","16") -- powers of two --
            # and W13's trip count is 48, so 3/6/12/24 were unreachable through
            # it and were never in any sweep. 6 is worth -0.16 ms at the wall
            # (task #129), which is how the hole was found. VGPR bill for
            # W13-only, /tmp/vgpr.sh on the real image: 325 at 4/6/8, 327 at
            # 12, 344 at 16, and 512 with 180 spills at 24 -- so 24 is the
            # cliff and the ceiling here is set to 16. The historical "depth 8
            # costs 325 -> 362" was the UNIFIED knob, i.e. it was W2's bill.
            _x = os.environ.get(_v)
            if _x is not None:
                assert _x.isdigit() and 0 <= int(_x) <= 16, f"{_v} is 0..16"
                flags = flags + [f"-D{_v}={_x}"]

        _km = os.environ.get("MPK_MOE_KMAJOR")
        if _km is not None:
            # Permutes the data half (1) and optionally the E8M0 scales (2) of
            # the packed MoE weight so a wave's k-group is contiguous instead
            # of sixteen pieces W_ROW_BYTES apart -- 8 L2 requests per
            # global_load_dwordx4 instead of 16. Same bytes, same MFMA.
            # demo/glm5/demo.py's packer reads the SAME env var and has to
            # agree, so this is one -D and every rank must see it. Long note at
            # the define in gang_moe_linear_mxfp8_mi300.cuh.
            assert _km in ("0", "1", "2"), "MPK_MOE_KMAJOR is 0, 1 or 2"
            flags = flags + [f"-DMPK_MOE_KMAJOR={_km}"]

        _dkm = os.environ.get("MPK_DENSE_KMAJOR")
        if _dkm is not None:
            # The same permutation on the attention-half GEMM, at a 64-byte
            # granule because an FP8 k-tile is 128 and one instruction reaches
            # half of it. Applies to the OUTPUT_PER_WG == 16 K-parallel call
            # sites only -- qkv_a and the un-absorbed q_b; the LM head and the
            # dense MLP share the packer and the kernel but not this layout.
            # Long note at the define in
            # gang_rmsnorm_linear_mxfp8_bias_mi300.cuh.
            assert _dkm in ("0", "1", "2"), "MPK_DENSE_KMAJOR is 0, 1 or 2"
            flags = flags + [f"-DMPK_DENSE_KMAJOR={_dkm}"]

        _apf = os.environ.get("MPK_ATTN_PF_GROUPS")
        if _apf is not None:
            # Same knob for the attention-half GEMM (qkv_a, q_b, W_UK, W_UV,
            # o_proj). 0 restores the rotating depth-4 loop, whose ISA shows a
            # full `s_waitcnt vmcnt(0)` plus 28 v_mov at every loop back-edge.
            # Long note at _rnlm8_kloop_deep in
            # gang_rmsnorm_linear_mxfp8_bias_mi300.cuh. Compile-time, so every
            # rank must agree.
            assert _apf in ("0", "2", "4", "8", "16"), \
                "MPK_ATTN_PF_GROUPS is 0, 2, 4, 8 or 16"
            flags = flags + [f"-DMPK_ATTN_PF_GROUPS={_apf}"]

        for _v in ("MPK_ATTN_PF_GROUPS_N", "MPK_ATTN_PF_GROUPS_K"):
            # The unified knob above drives TWO branches with different trip
            # counts -- N-parallel walks MFMA_ITERS (48 qkv_a, 16 q_b, 4
            # W_UK/W_UV), K-parallel walks MFMA_ITERS/4 (12 for qkv_a) -- so
            # one request lands on different depths in each and its 2/4/8
            # sweep could only report their sum. That is the same trap the MoE
            # twin fell into: split per GEMM, W13 wants depth 8 (-0.256 ms) and
            # W2 wants 4 (+0.476 at 8). Range is 0..16 and NOT restricted to
            # powers of two: 3/6/12 are divisors of 48 that the unified knob's
            # membership assert cannot express, and 6 is where the MoE hole was
            # found. Compile-time, so every rank must agree.
            _x = os.environ.get(_v)
            if _x is not None:
                assert _x.isdigit() and 0 <= int(_x) <= 16, f"{_v} is 0..16"
                flags = flags + [f"-D{_v}={_x}"]

        _apfd = os.environ.get("MPK_ATTN_PF_DBUF")
        if _apfd is not None:
            # The attention-half twin of MPK_MOE_PF_DBUF: swap two register
            # buffers at the backedge instead of copying one into the other,
            # which is what forces the s_waitcnt vmcnt(0) the deep loop was
            # written to avoid. 0 = copying everywhere, 1 = dbuf everywhere,
            # 2 = dbuf only where the block count is odd, which at GLM-5's
            # shapes is exactly the K-parallel qkv_a site (NBLK=3) and nothing
            # else. Compile-time, so every rank must agree.
            assert _apfd in ("0", "1", "2"), "MPK_ATTN_PF_DBUF is 0, 1 or 2"
            flags = flags + [f"-DMPK_ATTN_PF_DBUF={_apfd}"]

        _kvf = os.environ.get("MPK_KVUPD_FAST")
        if _kvf is not None:
            # latent_to_cache's index chase: 1 issues all five req-indexed
            # index loads above the early-return compare, drops the LDS page
            # table, and holds the latent row in registers across the norm
            # reduction. 0 restores the four-round-trip original.
            assert _kvf in ("0", "1"), "MPK_KVUPD_FAST is 0 or 1"
            flags = flags + [f"-DMPK_KVUPD_FAST={_kvf}"]

        _w2_sf = os.environ.get("MPK_W2_STAGE_FULL")
        if _w2_sf is not None:
            # Restores W2's pre-fdca420 full-width activation staging so the
            # staging A/B is one -D in one build. Compile-time, so every rank
            # must see it.
            assert _w2_sf in ("0", "1"), "MPK_W2_STAGE_FULL is 0 or 1"
            flags = flags + [f"-DMPK_W2_STAGE_FULL={_w2_sf}"]

        _sk = os.environ.get("MPK_MOE_SHARED_KSHARD")
        if _sk is not None:
            # Shard the shared expert's intermediate across the EP ranks so
            # EP_SHARED_PE stops carrying it alone. Long note at the define in
            # mpk_atoms.cuh. Must equal the world size: every rank takes one
            # slice and the EP collective sums the partials, so a value that
            # disagrees with the rank count drops or double-counts a slice.
            # 4 is the cap: W2's depth-4 pipeline needs
            # MFMA_ITERS/slices >= 4 and GLM-5's W2 has MFMA_ITERS = 16.
            assert _sk in ("0", "2", "4"), "MPK_MOE_SHARED_KSHARD is 0, 2, or 4"
            flags = flags + [f"-DMPK_MOE_SHARED_KSHARD={_sk}"]
        _moe_afp8 = os.environ.get("MPK_MOE_ACT_FP8")
        if _moe_afp8 is not None:
            # W13 emits the SwiGLU result as MXFP8 (E4M3 + one E8M0 per 32) and
            # W2 stages those bytes instead of re-quantizing the same bf16
            # vector once per tile. Default is 1 in the header; forward only an
            # explicit override, and only as one -D, because producer and
            # consumer layouts have to agree across every rank.
            assert _moe_afp8 in ("0", "1"), "MPK_MOE_ACT_FP8 is 0 or 1"
            flags = flags + [f"-DMPK_MOE_ACT_FP8={_moe_afp8}"]
        _w2_ks = int(os.environ.get("MPK_W2_KSPLIT", "1"))
        if _w2_ks != 1:
            # The GLM fused-layer W2 (gang_moe_linear_mxfp8_mi300.cuh), not the
            # gpt-oss MPK_W2_SPLITK above -- different kernel, different file.
            # The host multiplies moe_w2_tiles_per_xcd by the same number, so a
            # rank that misses the flag builds a tile space that disagrees with
            # its own loop bound. Compile-time, hence in MPK_FORWARD_VARS.
            assert _w2_ks >= 1, "MPK_W2_KSPLIT is at least 1"
            flags = flags + [f"-DMPK_W2_KSPLIT={_w2_ks}"]
        if int(os.environ.get("MPK_EP_SKEW_PROBE", "0")) == 1:
            # Measures how much earlier a column slice's W2 tiles finish than
            # the last W2 tile on the GPU -- i.e. the headroom a per-slice
            # "send as you go" release could claim over the current GPU-wide
            # Phase 9 barrier. One timestamp per W2 tile, so cheap, but it
            # still perturbs; do not quote latency from a probe run.
            flags = flags + ["-DMPK_EP_SKEW_PROBE"]
        # Inline-EP combine ablation. 1 = keep every barrier but drop the
        # cross-GPU put/wait; 2 = drop Phase 9 entirely but keep the 64/64
        # expert slicing. BOTH PRODUCE WRONG OUTPUT -- they exist to price the
        # collective's parts against the DP baseline, nothing else.
        _ep_ablate = int(os.environ.get("MPK_EP_ABLATE", "0"))
        if _ep_ablate:
            flags = flags + [f"-DMPK_EP_ABLATE={_ep_ablate}"]
        # Bound the Phase 9 peer wait so a hung run still reaches kernel exit
        # and flushes its device printf buffer. See MPK_EP_WAIT_TIMEOUT in
        # gang_full_layer_fused_mi300.cuh. ALSO PRODUCES WRONG OUTPUT when it
        # fires -- falling through consumes peer slots that were never written.
        _ep_wait_tmo = int(os.environ.get("MPK_EP_WAIT_TIMEOUT", "0"))
        if _ep_wait_tmo:
            flags = flags + [f"-DMPK_EP_WAIT_TIMEOUT={_ep_wait_tmo}"]
        # How many layers of [EPTMO]/[EPFOLD]/[EPREL] to print, on the
        # run-monotonic layer counter. Default 2; set it to the fused layer
        # count to cover the whole first decode iteration.
        _ep_tmo_layers = int(os.environ.get("MPK_EP_TMO_PRINT_LAYERS", "0"))
        if _ep_tmo_layers:
            flags = flags + [f"-DMPK_EP_TMO_PRINT_LAYERS={_ep_tmo_layers}"]
        if int(os.environ.get("MPK_EP_SIG_DBG", "0")) == 1:
            # EP collective diagnostics: prints, once per rank, which of the
            # two Phase 9 publication paths is actually live. Cheap enough to
            # leave reachable, and the one thing worth checking first whenever
            # an EP latency number looks unexplained -- see the [EPPATH] probe
            # in gang_full_layer_fused_mi300.cuh.
            flags = flags + ["-DMPK_EP_SIG_DBG"]
        if int(os.environ.get("MPK_EP_FORCE_STAGED", "0")) == 1:
            # Force the staged rocSHMEM publication even where every peer is
            # directly mapped. Both paths write the same value to the same
            # symmetric address, so this bisects "the direct peer store is not
            # becoming visible" against everything else in the layer. Slower by
            # construction -- a diagnostic, not a configuration.
            flags = flags + ["-DMPK_EP_FORCE_STAGED"]
        # The precomputed worker-dispatch template is baked from single-GPU task
        # timing. Under multi-GPU (rocSHMEM) the cross-GPU put+signal waits
        # perturb that timing and the fixed template deadlocks: a worker parks
        # on a task it must itself produce, the ranks drift apart, and the run
        # hangs. Force the dynamic scheduler-dispatch path for multi-GPU;
        # single-GPU keeps the precomputed fast path. An explicit
        # PRECOMPUTED_DISPATCH=0 still disables it everywhere.
        precomputed_default = "0" if use_rocshmem else "1"
        if int(os.environ.get("PRECOMPUTED_DISPATCH", precomputed_default)) == 1:
            flags = flags + ["-DMPK_PRECOMPUTED_DISPATCH"]
            flags = flags + ["-DMPK_FUSED_LAYER_BATCHING"]
        if int(os.environ.get("MPK_NIL_TRIPWIRE", "0")) == 1:
            # Breadcrumbs in pinned host memory + SIGABRT dump, to attribute
            # the nil-address memory fault. Off by default: the per-layer
            # writes cost a little and only matter while chasing that bug.
            flags = flags + ["-DMPK_NIL_TRIPWIRE"]
        if int(os.environ.get("MPK_WORKER_STATE", "0")) == 1:
            # Per-phase worker-state breadcrumbs: which phase/barrier each
            # worker is in, dumped on a hang. This is how the fused-layer
            # deadlocks were attributed, so keep it reachable -- but off by
            # default. The stores go to pinned *host* memory over PCIe from
            # ~30 sites in the two hottest task headers, several inside spin
            # loops; measured 2.386 -> 2.321 ms/iter when compiled out.
            flags = flags + ["-DMPK_WORKER_STATE"]
        if int(os.environ.get("TRACE_MOE", "0")) == 1:
            flags = flags + ["-DMPK_TRACE_MOE_DISPATCH"]
        if int(os.environ.get("EMBED_DEBUG", "0")) == 1:
            flags = flags + ["-DEMBED_DEBUG"]
        if int(os.environ.get("MPK_DEBUG_LAYERS", "0")) == 1:
            flags = flags + ["-DMPK_DEBUG_RMSNORM", "-DMPK_DEBUG_MOE_MUL_SUM"]
        if int(os.environ.get("CK_FMHA_1TOK", "0")) == 1:
            # Force seqlen_q=1: uses merge path only (faster decode, slower prefill)
            flags = flags + ["-DMPK_MAX_TOKENS_PER_REQUEST=1"]
        if int(os.environ.get("MPK_SPEC_DECODE", "0")) == 1:
            # Speculative decode: a decode iteration dispatches the verify row
            # plus one draft row and prepare_next_batch keeps the draft only
            # if the verify row predicted it. Needs a batch_size >= 2 build
            # (--max-num-batched-tokens 2); demo.py asserts that.
            #
            # Compile-time, so every rank must see it -- the qo_indptr the
            # ranks agree on comes from each rank's own prepare_next_batch.
            flags = flags + ["-DMPK_SPEC_DECODE"]
            if int(os.environ.get("MPK_SPEC_ORACLE", "0")) == 1:
                # Draft = whatever the host pre-filled at tokens[step+1]
                # (--spec-oracle-tokens). The acceptance == 1.0 arm.
                flags = flags + ["-DMPK_SPEC_ORACLE"]
        amdgpu_target = os.environ.get("AMDGPU_TARGETS", "gfx950")
        if use_rocshmem:
            # rocSHMEM's IPC backend requires an xnack-off code object on gfx950.
            if ":" not in amdgpu_target:
                amdgpu_target = f"{amdgpu_target}:xnack-"
        specific_cmd = [
            f"--offload-arch={amdgpu_target}",
            f"-L{rocm_lib}",
            f"-L{rocm_lib64}",
            "-lamdhip64",
            "-lrocblas",
            "-lhipblas",
        ]
        if use_rocshmem:
            # Device-initiated one-sided communication (rocSHMEM), the AMD
            # analog of NVSHMEM. -fgpu-rdc + --hip-link are the HIP equivalents
            # of nvcc's -rdc=true device link, required so the megakernel can
            # call rocSHMEM device functions across translation units.
            rocshmem_lib_archive = os.path.join(rocshmem_lib_path, "librocshmem.a")
            common_cmd = common_cmd + [
                f"-I{rocshmem_inc_path}",
                f"-I{mpi_inc_path}",
            ]
            flags = flags + [
                "-DUSE_ROCSHMEM",
                "-fgpu-rdc",
                "--hip-link",
            ]
            specific_cmd = specific_cmd + [
                # Reset the input language ("-x hip" is still active from
                # common_cmd) so the driver treats the rocSHMEM archive as a
                # link input rather than compiling it as HIP source.
                "-x", "none",
                rocshmem_lib_archive,
                f"-L{mpi_lib_path}",
                "-lmpi",
                "-lhsa-runtime64",
            ]
        return common_cmd + specific_cmd + flags

    if profiling:
        flags = flags + ["-DMPK_ENABLE_PROFILING"]

    return common_cmd + specific_cmd + flags


class PersistentKernel:
    def __init__(
        self,
        mode: str,
        world_size: int,
        mpi_rank: int,
        num_workers: int,
        num_local_schedulers: int,
        num_remote_schedulers: int,
        max_seq_length: int,
        max_num_batched_requests: int,
        max_num_batched_tokens: int,
        max_num_pages: int,
        page_size: int,
        meta_tensors: dict,
        profiler_tensor: torch.Tensor,
        trace_name: str,
        spec_decode_config: SpecDecodeConfig,
        use_cutlass_kernel: bool,
        eos_token_id: int64 = -1,
    ):
        self.__finalized__ = False
        self._is_compiled = False
        self._dummy_counter = 0
        self._dummy_tensor_refs = []  # prevent GC of dummy tensors (pointer reuse)
        if mode not in valid_persistent_kernel_modes:
            raise ValueError(f"Invalid persistent kernel mode: {mode}")
        self.mode = mode
        self.world_size = world_size
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        self.max_seq_length = max_seq_length
        self.max_num_batched_requests = max_num_batched_requests
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.eos_token_id = eos_token_id
        self.kn_graph = KNGraph(CyKNGraph(disable_fingerprint=True))
        self.meta_tensors = meta_tensors
        self.profiler_tensor = profiler_tensor
        self.trace_name = trace_name
        # Multi-GPU comm backend: NVSHMEM on CUDA, rocSHMEM on ROCm.
        _is_rocm = bool(getattr(torch.version, "hip", None))
        self.use_nvshmem = (world_size > 1) and not _is_rocm
        self.use_rocshmem = (world_size > 1) and _is_rocm
        self.spec_decode_config = spec_decode_config
        self._spec_decode_handlers = {
            "promptlookup": self.prompt_lookup_spec_handler,
        }
        self._spec_verify_handlers = {
            "promptlookup": self.prompt_lookup_verify_handler,
        }
        # determine total number of requests for offline serving
        self.total_num_requests = meta_tensors["tokens"].shape[0]
        assert self.max_seq_length == meta_tensors["tokens"].shape[1]
        self.is_rocm = bool(getattr(torch.version, "hip", None))
        # Force CUTLASS off on ROCm - CUTLASS requires CUDA tensor cores
        if self.is_rocm:
            self.use_cutlass_kernel = False
        else:
            self.use_cutlass_kernel = use_cutlass_kernel
        if self.is_rocm:
            # Detect AMD GPU generation from offload-arch target
            amdgpu_target = os.environ.get("AMDGPU_TARGETS", "gfx950")
            if "gfx950" in amdgpu_target:
                self.target_cc = 95  # MI350 (gfx950): 160KB LDS
            else:
                self.target_cc = 94  # MI300 (gfx942): 64KB LDS
        else:
            self.target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        # Check tensor shapes
        qo_indptr_buffer = self.meta_tensors["qo_indptr_buffer"]
        # Asserts "==" below is not guaranteed by vllm, because the shape is changed depending on real situation. But the mem space won't change.
        assert qo_indptr_buffer.shape[0] <= self.max_num_batched_requests+1, f"qo_indptr_buffer.shape: {qo_indptr_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        paged_kv_indptr_buffer = self.meta_tensors["paged_kv_indptr_buffer"]
        assert paged_kv_indptr_buffer.shape[0] <= self.max_num_batched_requests+1, f"paged_kv_indptr_buffer.shape: {paged_kv_indptr_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        paged_kv_indices_buffer = self.meta_tensors["paged_kv_indices_buffer"]
        # assert paged_kv_indices_buffer.shape == (self.max_num_pages,), f"paged_kv_indices_buffer.shape: {paged_kv_indices_buffer.shape}, max_num_pages: {self.max_num_pages}"
        # TODO: This is because the paged_kv_indices_buffer can be limited by max len on vllm side
        assert paged_kv_indices_buffer.shape[0] <= self.max_num_pages, f"paged_kv_indices_buffer.shape: {paged_kv_indices_buffer.shape}, max_num_pages: {self.max_num_pages}"
        paged_kv_last_page_len_buffer = self.meta_tensors["paged_kv_last_page_len_buffer"]
        assert paged_kv_last_page_len_buffer.shape[0] <= self.max_num_batched_requests, f"paged_kv_last_page_len_buffer.shape: {paged_kv_last_page_len_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        
        # check type of meta_tensors
        assert self.meta_tensors["tokens"].dtype == torch.int64, f"tokens.dtype: {self.meta_tensors['tokens'].dtype}"
        assert self.meta_tensors["input_tokens"].dtype == torch.int64, f"input_tokens.dtype: {self.meta_tensors['input_tokens'].dtype}"
        assert self.meta_tensors["output_tokens"].dtype == torch.int64, f"output_tokens.dtype: {self.meta_tensors['output_tokens'].dtype}"
        assert self.meta_tensors["num_new_tokens"].dtype == torch.int32, f"num_new_tokens.dtype: {self.meta_tensors['num_new_tokens'].dtype}"
        assert self.meta_tensors["prompt_lengths"].dtype == torch.int32, f"prompt_lengths.dtype: {self.meta_tensors['prompt_lengths'].dtype}"
        assert qo_indptr_buffer.dtype == torch.int32, f"qo_indptr_buffer.dtype: {qo_indptr_buffer.dtype}"
        assert paged_kv_indptr_buffer.dtype == torch.int32, f"paged_kv_indptr_buffer.dtype: {paged_kv_indptr_buffer.dtype}"
        assert paged_kv_indices_buffer.dtype == torch.int32, f"paged_kv_indices_buffer.dtype: {paged_kv_indices_buffer.dtype}"
        assert paged_kv_last_page_len_buffer.dtype == torch.int32, f"paged_kv_last_page_len_buffer.dtype: {paged_kv_last_page_len_buffer.dtype}"

    def _raise_unsupported_target_cc(self, layer: str, supported_tasks: list) -> None:
        """Raise NotImplementedError for ROCm/AMD or other unsupported target_cc."""
        hip = getattr(torch.version, "hip", None)
        if hip:
            raise NotImplementedError(
                f"MPK {layer} is not supported on ROCm/AMD GPUs. "
                "The persistent kernel layer compiles CUDA code (nvcc) and currently supports "
                "NVIDIA sm_80, sm_90, sm_100 only. Use a non-MPK mirage path or an NVIDIA GPU."
            )
        raise NotImplementedError(
            f"MPK {layer} does not support target_cc={self.target_cc}. "
            f"Supported: sm_80 (80), sm_90 (90), sm_100 (100). "
            f"Tasks for this layer: {supported_tasks}."
        )

    def attach_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        dims = tuple([d for d in torch_tensor.shape])
        strides = tuple([s for s in torch_tensor.stride()])
        # Assert a row-major layout
        for d in range(len(dims) - 1):
            assert strides[d] == strides[d + 1] * dims[d + 1]
        dtype = convert_torch_type_to_dtype(torch_tensor.dtype)
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        self.kn_graph.attach_torch_tensor(t, torch_tensor, name)
        return t

    def new_tensor(
        self,
        dims: tuple,
        strides: tuple = None,
        dtype: dtype = bfloat16,
        name: str = None,
        io_category: str = "cuda_tensor",
    ) -> DTensor:
        # Assert a row-major layout
        # if strides is not None:
        #     for d in range(len(dims) - 1):
        #         assert strides[d] == strides[d + 1] * dims[d + 1]
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        if io_category == "cuda_tensor":
            self.kn_graph.attach_cuda_tensor(t, name)
        elif io_category == "nvshmem_tensor":
            self.kn_graph.attach_nvshmem_tensor(t, name)
        else:
            raise RuntimeError(f"Invalid io_category: {io_category}")
        return t

    def fuse_tensors(
        self, inputs: list[DTensor], fused_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support fusing the 0-th dimension
        assert fused_dim == 0
        t = self.kn_graph.fuse_tensors(inputs, fused_dim, num_groups, name)
        return t

    def shuffle_tensors(
        self, inputs: list[DTensor], shuffled_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support shuffling the 0-th dimension
        assert shuffled_dim == 0
        t = self.kn_graph.shuffle_tensors(inputs, shuffled_dim, num_groups, name)
        return t

    def embed_layer(
        self,
        input: DTensor, # [batch_size, num_spec_tokens]
        weight: DTensor, # [vocab_size, hidden_size]
        output: DTensor, # [batch_size, hidden_size]
        grid_dim: tuple,
        block_dim: tuple,
        input_source: int = 0, # 0: all_tokens, 1: input_token
    ):
        # TODO: Support batch size > 1
        # tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # tb_graph.new_input(input, (-1, -1, -1), -1, True)
        # tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        # tb_graph.new_input(output, (-1, -1, -1), -1, True)
        # self.kn_graph.customized([input, weight, output], tb_graph)
        # self.kn_graph.register_task(tb_graph, "embedding")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(weight, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "embedding" if self.target_cc == 90 else "embedding", [input_source])

    def rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        actual_hidden_dim: int = 0,
    ):
        # Currently assume that the input/output are 2D tensors
        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(output, (0, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        # 94 (MI300): use base "rmsnorm"; 90/100 use hopper
        # actual_hidden_dim: if > 0, divide by this instead of HIDDEN_DIM in RMS computation
        # (used when padding hidden dim to avoid bf16 rounding errors from scale factor)
        params = [actual_hidden_dim] if actual_hidden_dim > 0 else []
        self.kn_graph.register_task(
            tb_graph,
            "rmsnorm_hopper" if (self.target_cc == 90 or self.target_cc == 100) else "rmsnorm",
            params,
        )

    def rmsnorm_linear_layer(
        self,
        input: DTensor,
        weight_norm: DTensor,
        weight_linear: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/weight_linear/output are 2D tensors
        assert input.num_dims == 2
        assert weight_linear.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight_norm, (-1, -1, -1), 0, True)
        tb_graph.new_input(weight_linear, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "rmsnorm_linear")

    def attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "attention", params)

    def single_batch_extend_attention_layer(
        self,
        input: DTensor, # [6, 6144]
        k_cache: DTensor, 
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple, # (6, 8, 1)
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim # 32
        rotary_embed = 0
        output_stride = output.dim(1)

        extend_num = input.dim(0) - 1
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: extend_num
        # params[5]: output_stride
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, extend_num, output_stride]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "single_batch_extend_attention", params)

    def paged_attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # If q_norm/k_norm are None, create dummy tensors (kernel still expects 8 inputs)
        if q_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            q_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_q_norm_pa_{self._dummy_counter}")
            self._dummy_counter += 1
        if k_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            k_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_k_norm_pa_{self._dummy_counter}")
            self._dummy_counter += 1

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_hopper", params)
        elif self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_sm100", params)
        else:
            self.kn_graph.register_task(tb_graph, "paged_attention", params)

    
    def paged_attention_split_kv_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        lse: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        assert output.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv * head_dim / world_size, num_kv_heads)
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        # params[6]: num_kv_chunks
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size, num_kv_chunks]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse, (-1, 2, 1), -1, True)
        tb_graph.new_input(output, (-1, 2, 1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                lse,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_sm100", params)
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_mi300", params)
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_hopper", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")

    def paged_attention_split_kv_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)
        assert output_tmp.num_dims == 3  # (num_tokens, num_chunks, hidden_size / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_qo_heads_per_kv = num_q_heads / grid_dim[1]
        num_kv_heads = grid_dim[1]
        # params[0]: num_qo_heads_per_kv
        # params[1]: head_dim
        # params[2]: max_seq_len
        # params[3]: page_size
        # params[4]: num_kv_heads
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length, self.page_size, num_kv_heads]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(lse, (-1, 2, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, 2, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                lse,
                output_tmp,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_merge_sm100", params)
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_merge_mi300", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")

    def kv_cache_update_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        q_workspace: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4
        assert q_workspace.num_dims == 2  # (num_tokens, q_workspace_stride)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = q_workspace.dim(1) // head_dim
        q_workspace_stride = q_workspace.dim(1)
        rotary_embed = 1 if cos_pos_embed is not None else 0
        qk_norm = 1 if q_norm is not None else 0

        # If q_norm/k_norm are None, create dummy tensors (kernel still expects 8 inputs)
        if q_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            q_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_q_norm_{self._dummy_counter}")
            self._dummy_counter += 1
        if k_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            k_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_k_norm_{self._dummy_counter}")
            self._dummy_counter += 1

        # params: num_q_heads, num_kv_heads, qk_norm, rotary_embed, max_seq_len, page_size, q_workspace_stride
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
                  self.max_seq_length, self.page_size, q_workspace_stride]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [input, k_cache, v_cache, q_norm, k_norm,
             cos_pos_embed, sin_pos_embed, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "kv_cache_update_mi300", params)

    def mla_kv_cache_update_layer(
        self,
        q_absorbed: DTensor,
        kv_latent: DTensor,
        kv_cache: DTensor,
        kv_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        q_workspace: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        kv_offset: int = 0,
    ):
        """Phase A of absorbed MLA (GLM-5): latent cache append + partial RoPE.

        The GQA kv_cache_update_layer splits QKV out of one fused tensor into
        separate K and V caches. Under absorption there is one latent row per
        token, kv_row = [c_kv | k_rope], and V is the leading kv_lora_rank dims
        of that same row, so the two paths cannot share a task.

        q_absorbed comes from q_b_proj with W_UK folded in, so each head is
        already [q_nope @ W_UK | q_rope] and only needs RoPE on the trailing
        slice before it lands in q_workspace. kv_latent is the raw
        kv_a_proj_with_mqa output; kv_a_layernorm is applied here, on the write
        side of the cache, because kv_b_proj (hence W_UK/W_UV) is linear in the
        normalised latent.

        ``kv_offset`` is the column the latent starts at within its row, for
        callers that fuse kv_a_proj into a wider GEMM.
        """
        assert q_absorbed.num_dims == 2   # (num_tokens, num_q_heads * qk_dim)
        assert kv_latent.num_dims == 2    # (num_tokens, >= kv_offset + qk_dim)
        assert kv_cache.num_dims == 4     # (num_pages, page_size, 1, qk_dim)
        assert q_workspace.num_dims == 2  # (num_tokens, q_workspace_stride)
        assert kv_cache.dim(2) == 1, "MLA keeps a single shared latent head"

        qk_dim = kv_cache.dim(3)
        qk_rope_head_dim = cos_pos_embed.dim(cos_pos_embed.num_dims - 1)
        kv_lora_rank = qk_dim - qk_rope_head_dim
        num_qo_heads = q_workspace.dim(1) // qk_dim
        q_workspace_stride = q_workspace.dim(1)
        assert kv_latent.dim(1) >= kv_offset + qk_dim
        assert kv_norm.dim(0) == kv_lora_rank

        # params: num_qo_heads, kv_lora_rank, qk_rope_head_dim, max_seq_len,
        #         page_size, q_workspace_stride, [kv_offset]
        params = [num_qo_heads, kv_lora_rank, qk_rope_head_dim,
                  self.max_seq_length, self.page_size, q_workspace_stride]
        if kv_offset:
            params.append(kv_offset)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_absorbed, (-1, 1, -1), -1, True)
        tb_graph.new_input(kv_latent, (-1, 1, -1), -1, True)
        tb_graph.new_input(kv_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(kv_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [q_absorbed, kv_latent, kv_cache, kv_norm,
             cos_pos_embed, sin_pos_embed, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "mla_kv_cache_update_mi300", params)

    def paged_attention_ck_fmha_layer(
        self,
        q_workspace: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        o_acc: DTensor,
        lse_acc: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
        sinks: DTensor = None,
        sliding_window: int = 0,
    ):
        assert q_workspace.num_dims == 2  # (num_tokens, q_workspace_stride)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4
        assert o_acc.num_dims == 2   # (num_tokens, kv_heads*chunks*qo_per_kv*head_dim)
        assert lse_acc.num_dims == 2 # (num_tokens, kv_heads*chunks*qo_per_kv)

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        max_num_requests = attention_params[2]
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        # params: num_q_heads, num_kv_heads, head_dim, page_size, max_seq_len,
        #         num_kv_chunks, q_workspace_stride, kv_cache_stride,
        #         max_num_requests, sliding_window
        params = [num_q_heads, num_kv_heads, head_dim, self.page_size,
                  self.max_seq_length, num_kv_chunks, q_workspace_stride,
                  kv_cache_stride, max_num_requests, sliding_window]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == max_num_requests
        assert grid_dim[1] == num_kv_heads
        assert grid_dim[2] == num_kv_chunks
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        # Optional sinks input (GPT-OSS per-head attention sinks).
        # When provided, sink correction is fused into the attention epilogue,
        # eliminating the standalone attention_sink_layer task.
        if sinks is not None:
            assert sinks.num_dims == 1  # (num_q_heads,)
            tb_graph.new_input(sinks, (-1, -1, -1), -1, True)
        # o_acc/lse_acc: 2D flat, no partitioning — kernel offsets by kv_head internally
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)
        if sinks is not None:
            self.kn_graph.customized(
                [q_workspace, k_cache, v_cache, sinks, o_acc, lse_acc],
                tb_graph,
            )
        else:
            self.kn_graph.customized(
                [q_workspace, k_cache, v_cache, o_acc, lse_acc],
                tb_graph,
            )
        self.kn_graph.register_task(tb_graph, "paged_attention_ck_fmha_split_kv_mi300", params)

    def paged_attention_ck_fmha_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
        sinks: DTensor = None,
        dim_splits: int = 1,
        write_through: bool = False,
    ):
        # dim_splits > 1 splits each kv head's merge across that many tasks,
        # one per contiguous head_dim slice. grid.y must then be
        # num_kv_heads * dim_splits, and the kernel reads bid.y as
        # (kv_head * dim_splits + slice). Only worth it when head_dim is wide
        # relative to num_kv_heads, i.e. absorbed MLA.
        # lse: 2D (num_tokens, num_kv_heads * chunks * qo_per_kv)
        # output_tmp: 2D (num_tokens, num_kv_heads * chunks * qo_per_kv * head_dim)
        # output: 2D (num_tokens, num_q_heads * head_dim)
        # sinks (optional): 1D (num_q_heads,) — applies sigmoid(LSE - sink) per
        # q-head to the merged output (GPT-OSS attention sinks).
        assert lse.num_dims == 2
        assert output_tmp.num_dims == 2
        assert output.num_dims == 2
        if sinks is not None:
            assert sinks.num_dims == 1

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_kv_chunks = attention_params[2]
        num_kv_heads = attention_params[3]
        num_qo_heads_per_kv = num_q_heads // num_kv_heads
        assert dim_splits >= 1 and head_dim % dim_splits == 0
        assert grid_dim[1] == num_kv_heads * dim_splits, (
            f"grid.y must be num_kv_heads * dim_splits "
            f"({num_kv_heads} * {dim_splits}), got {grid_dim[1]}")
        # params: num_qo_heads_per_kv, head_dim, max_seq_len, page_size,
        #         num_kv_heads, num_kv_chunks, dim_splits, write_through
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length,
                  self.page_size, num_kv_heads, num_kv_chunks, dim_splits,
                  1 if write_through else 0]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # No partitioning — merge kernel handles all offsets internally
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, -1, -1), -1, True)
        if sinks is not None:
            tb_graph.new_input(sinks, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        if sinks is not None:
            self.kn_graph.customized(
                [lse, output_tmp, sinks, output],
                tb_graph,
            )
        else:
            self.kn_graph.customized(
                [lse, output_tmp, output],
                tb_graph,
            )
        self.kn_graph.register_task(tb_graph, "paged_attention_ck_fmha_merge_mi300", params)

    def attention_sink_layer(
        self,
        attn_out: DTensor,
        lse_acc: DTensor,
        sinks: DTensor,
        num_q_heads: int,
        head_dim: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Post-attention sink correction.
        Multiplies attention output by sigmoid(LSE - sink) per head.
        attn_out is modified in-place (same tensor for input and output).
        """
        assert attn_out.num_dims == 2  # (max_tokens, num_q_heads * head_dim)
        assert lse_acc.num_dims == 2   # (max_tokens, num_q_heads)
        assert sinks.num_dims == 1     # (num_q_heads,)

        params = [num_q_heads, head_dim]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(attn_out, (-1, 1, -1), -1, True)  # in-place: both input and output
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)
        tb_graph.new_input(sinks, (-1, -1, -1), -1, True)
        tb_graph.new_input(attn_out, (-1, 1, -1), -1, True)  # output = same tensor
        self.kn_graph.customized(
            [attn_out, lse_acc, sinks, attn_out],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "attention_sink_mi300", params)

    def gang_paged_attention_split_kv_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        lse: DTensor,
        output: DTensor,
        q_workspace: DTensor,
        attention_params: tuple,
        block_dim: tuple,
    ):
        """Gang CK FMHA attention: 8 tasks (1 per XCD), broadcast to workers.
        Fuses KV cache update + CK FMHA attention into one gang task.
        Each worker decodes tile_idx → (request_id, kv_head).
        """
        assert input.num_dims == 2
        assert k_cache.num_dims == 4
        assert v_cache.num_dims == 4
        assert self.target_cc in (94, 95), "Gang attention only supported on MI300X"

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        q_workspace_stride = attention_params[2]

        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2
            assert sin_pos_embed.num_dims == 2
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1
            assert k_norm.num_dims == 1
            qk_norm = 1

        # Total work items = max_requests * num_kv_heads
        total_work_items = self.max_num_batched_requests * num_kv_heads
        import math
        total_work_items_per_xcd = math.ceil(total_work_items / 8)

        # params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
        #          max_seq_len, page_size, num_kv_chunks,
        #          total_work_items_per_xcd, total_work_items,
        #          q_workspace_stride]
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
                  self.max_seq_length, self.page_size, num_kv_chunks,
                  total_work_items_per_xcd, total_work_items,
                  q_workspace_stride]

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs: qkv, k_cache, v_cache, q_norm, k_norm, cos, sin
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        # 3 outputs: lse, output, q_workspace
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, k_cache, v_cache, q_norm, k_norm,
             cos_pos_embed, sin_pos_embed, lse, output, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "gang_attn_split_kv_mi300", params)

    def gang_mla_decode_layer(
        self,
        q_workspace: DTensor,
        kv_cache: DTensor,
        lse: DTensor,
        output: DTensor,
        mla_params: tuple,
        block_dim: tuple,
        q_workspace_slots: int = None,
    ):
        """Gang absorbed MLA decode (GLM-5): 8 tasks (1 per XCD).

        MLA has a single shared latent KV head, so unlike the GQA gang
        attention there is no kv_head == xcd_id mapping. Each worker decodes
        tile_idx -> (request_id, q_head_group, kv_chunk), where a q_head_group
        is one MFMA M tile of 16 query heads.

        q_workspace holds the absorbed queries [tokens, num_q_heads * (
        kv_lora_rank + qk_rope_head_dim)]; kv_cache holds one latent row
        [c_kv | k_rope] per token. The KV cache append is done by the
        preceding task, so this is attention only.

        With num_kv_chunks > 1 the output is float o_acc + natural-log LSE in
        the layout paged_attention_ck_fmha_merge_layer already consumes, with
        the q_head_group standing in for kv_head.
        """
        assert q_workspace.num_dims == 2
        assert self.target_cc in (94, 95), "Gang MLA decode is MI300/MI350 only"

        num_q_heads = mla_params[0]
        kv_lora_rank = mla_params[1]
        qk_rope_head_dim = mla_params[2]
        qk_head_dim = mla_params[3]
        num_kv_chunks = mla_params[4]

        assert num_q_heads % 16 == 0, "num_q_heads must be a multiple of the MFMA M tile"
        assert kv_lora_rank % 64 == 0
        assert (kv_lora_rank + qk_rope_head_dim) % 64 == 0
        assert kv_cache.dim(kv_cache.num_dims - 1) == kv_lora_rank + qk_rope_head_dim

        q_workspace_stride = num_q_heads * (kv_lora_rank + qk_rope_head_dim)
        # q_workspace may be declared narrower than the kernel indexes. The
        # producing q_b GEMM partitions its output row across the XCDs, so the
        # declared width is what decides where its head slots land, and the
        # caller narrows it to drop padding slots the model never fills. This
        # task maps q_workspace replicated -- (-1, -1, -1) below -- so the
        # declared width reaches nothing but this check: the kernel indexes
        # head h at h * qk_head_dim off the base pointer, and always reads
        # num_q_heads of them because a q group is one 16-row MFMA tile.
        # q_workspace_slots is therefore an assertion that the caller has
        # backed the declared row with a num_q_heads-wide allocation whose
        # tail is zero, not a shape the kernel adapts to.
        q_workspace_slots = q_workspace_slots or num_q_heads
        assert q_workspace_slots <= num_q_heads
        assert q_workspace.dim(1) == q_workspace_slots * (
            kv_lora_rank + qk_rope_head_dim)

        num_q_groups = num_q_heads // 16
        # The tile decomposition is (request, query row, q group, kv chunk) --
        # see the comment above gang_mla_decode_task_impl. The token dimension
        # sits INSIDE the request one because a request's rows share its page
        # list, so this product needs the batch_size factor the fused caller
        # already carries (gang_mla_attn_fused_layer's mla_total_work_items).
        # Without it the kernel is dispatched exactly one token's worth of
        # tiles, every one of them decodes token 0, and row 1 of the output is
        # never written -- which is precisely how the dense prologue produced
        # attn_out row 1 == 0.000000 at BATCH_SIZE 2 while the MoE layers,
        # which take the fused path, were correct. At batch_size 1 this is the
        # old product unchanged.
        total_work_items = (self.max_num_batched_requests
                            * self.max_num_batched_tokens
                            * num_q_groups * num_kv_chunks)
        import math
        total_work_items_per_xcd = math.ceil(total_work_items / 8)

        # params: [num_q_heads, kv_lora_rank, qk_rope_head_dim, qk_head_dim,
        #          max_seq_len, page_size, num_kv_chunks,
        #          total_work_items_per_xcd, total_work_items,
        #          q_workspace_stride, batch_size]
        #
        # batch_size is a template argument, not just a bound: the kernel
        # recovers token_idx from tile_idx by dividing by it, so a dispatch
        # that carries the factor while the kernel is instantiated at 1 would
        # decode token 0 four times over and address the wrong request.
        params = [num_q_heads, kv_lora_rank, qk_rope_head_dim, qk_head_dim,
                  self.max_seq_length, self.page_size, num_kv_chunks,
                  total_work_items_per_xcd, total_work_items,
                  q_workspace_stride, self.max_num_batched_tokens]

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 2 inputs: q_workspace, kv_cache
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache, (-1, -1, -1), -1, True)
        # 2 outputs: lse, output
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([q_workspace, kv_cache, lse, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "gang_mla_decode_mi300", params)

    def gang_mla_attn_fused_layer(
        self,
        # qkv_a
        x: DTensor,
        pre_norm_weight: DTensor,
        pre_norm_scratch: DTensor,
        qkv_mxfp8_weight: DTensor,
        qkv_bias: DTensor,
        # q_b + latent append
        q_a_norm_weight: DTensor,
        q_a_norm_scratch: DTensor,
        qb_mxfp8_weight: DTensor,
        qb_bias: DTensor,
        kv_norm_weight: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        kv_cache: DTensor,
        attn_counters: DTensor,
        moe_workspace_f32: DTensor,
        # outputs
        qkv_a_out: DTensor,
        q_workspace: DTensor,
        lse: DTensor,
        o_acc: DTensor,
        attn_out: DTensor,
        x_out: DTensor,
        # parameters
        qkv_output_per_wg: int,
        qkv_actual_hidden_dim: int,
        qb_output_per_wg: int,
        qb_reduction_size: int,
        qb_actual_hidden_dim: int,
        kv_offset: int,
        mla_params: tuple,
        q_workspace_slots: int = None,
        merge_dim_splits: int = 1,
        merge_write_through: bool = False,
        block_dim: tuple = (256, 1, 1),
    ):
        """The attention half of a GLM decoder layer in one gang dispatch.

        input_layernorm + [q_a_proj | kv_a_proj_with_mqa], q_a_layernorm +
        absorbed q_b_proj + latent cache append, absorbed MLA decode, and the
        split-KV merge, with three in-kernel barriers where the task graph
        used to put four events. Every sub-kernel keeps its standalone
        semantics; see the kernel header for the barrier layout and for which
        producers have to write through.

        ``qkv_a_out`` and ``q_workspace`` are declared unpartitioned even
        though each XCD only writes its own columns, because the phases after
        them read across XCDs. The kernel reconstructs the column slice.

        ``mla_params`` is (num_q_heads, kv_lora_rank, qk_rope_head_dim,
        qk_head_dim, num_kv_chunks), the same tuple gang_mla_decode_layer
        takes. num_kv_chunks must be > 1 -- with a single chunk the decode
        writes attn_out directly and there is no merge phase to fuse.

        Phase 1 also resolves the residual stream: ``x`` is the *previous*
        layer's pre-MoE residual and ``moe_workspace_f32`` its MoE's f32
        accumulator, and ``x_out`` receives ``bf16(workspace + x)``, which the
        o_proj that follows this task takes as its own residual. That add used
        to be moe_residual_add_f32, a grid_dim (1,1,1) task run once per layer;
        folding it into the prologue that was going to read the row anyway is
        what gpt-oss does in gang_resaddf32_rmsnorm_linear_mxfp4_bias. The task
        also zeroes ``moe_workspace_f32`` for this layer's own MoE, behind the
        qkv barrier where every reader is provably done -- so the caller must
        hand the *first* layer a workspace that is already zero, and must still
        run a moe_residual_add_f32 after the *last* layer to resolve its MoE
        and re-zero the buffer for the next token.

        Inputs (15): x, pre_norm_weight, pre_norm_scratch, qkv_mxfp8_weight,
                     qkv_bias, q_a_norm_weight, q_a_norm_scratch,
                     qb_mxfp8_weight, qb_bias, kv_norm_weight, cos, sin,
                     kv_cache, attn_counters, moe_workspace_f32.
        Outputs (6): qkv_a_out, q_workspace, lse, o_acc, attn_out, x_out.
        """
        assert self.target_cc == 95, "MXFP8 MFMA is gfx950-only"
        assert x.num_dims == 2
        assert qkv_mxfp8_weight.num_dims == 2
        assert qb_mxfp8_weight.num_dims == 2
        assert qkv_a_out.num_dims == 2
        assert q_workspace.num_dims == 2
        assert attn_out.num_dims == 2
        assert kv_cache.num_dims == 4  # (num_pages, page_size, 1, qk_dim)
        assert kv_cache.dim(2) == 1, "MLA keeps a single shared latent head"
        assert attn_counters.num_dims == 1

        batch_size = self.max_num_batched_tokens
        # Both GEMMs now take their input row stride explicitly (qkv_a's
        # happens to equal its reduction; q_b's is KV_INPUT_STRIDE), so the
        # narrowed reduction no longer stands in for one.

        num_q_heads = mla_params[0]
        kv_lora_rank = mla_params[1]
        qk_rope_head_dim = mla_params[2]
        qk_head_dim = mla_params[3]
        num_kv_chunks = mla_params[4]
        qk_dim = kv_lora_rank + qk_rope_head_dim
        assert kv_cache.dim(3) == qk_dim
        assert kv_norm_weight.dim(0) == kv_lora_rank
        assert cos_pos_embed.dim(cos_pos_embed.num_dims - 1) == qk_rope_head_dim
        assert num_kv_chunks > 1, (
            "one chunk means no merge phase; use the unfused path")

        # ── qkv_a tiling, from gang_rmsnorm_linear_mxfp8_bias_layer ──
        qkv_reduction = x.dim(1)
        # K must clear the depth-4 pipeline's tail: only slot 3 is guarded.
        assert qkv_reduction % 512 == 0, qkv_reduction
        qkv_n_wgs = qkv_mxfp8_weight.dim(0)
        assert qkv_n_wgs % 8 == 0
        qkv_n_wgs_per_xcd = qkv_n_wgs // 8
        qkv_output_stride = qkv_a_out.dim(1)
        assert qkv_n_wgs * qkv_output_per_wg == qkv_output_stride, (
            f"packed qkv_a weight covers {qkv_n_wgs * qkv_output_per_wg} "
            f"columns, qkv_a_out has {qkv_output_stride}")
        assert qkv_bias.dim(1) == qkv_output_stride

        # ── q_b tiling, from gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_layer ──
        assert qb_reduction_size % 512 == 0, qb_reduction_size
        assert qb_actual_hidden_dim <= qb_reduction_size <= qkv_output_stride
        assert kv_offset + qk_dim <= qkv_output_stride
        # See the same assert in gang_mla_full_layer_fused_layer: the rope slice
        # only has to fit inside one workgroup, not fill it. This path is always
        # absorbed, so qk_dim is 576 and only 64 divides it anyway.
        assert qb_output_per_wg >= qk_rope_head_dim and \
            qk_dim % qb_output_per_wg == 0, (
                "output_per_wg must be at least qk_rope_head_dim and divide "
                "the head span")
        qb_n_wgs = qb_mxfp8_weight.dim(0)
        assert qb_n_wgs % 8 == 0
        qb_n_wgs_per_xcd = qb_n_wgs // 8
        qb_output_stride = q_workspace.dim(1)
        assert qb_n_wgs * qb_output_per_wg == qb_output_stride
        assert qb_mxfp8_weight.dim(1) == qb_output_per_wg * (
            qb_reduction_size + qb_reduction_size // 32)
        assert qb_bias.dim(1) == qb_output_stride
        assert (qb_n_wgs_per_xcd * qb_output_per_wg) % qk_dim == 0, (
            "per-XCD chunk must hold whole heads")

        # ── decode + merge tiling ──
        assert num_q_heads % 16 == 0
        num_q_groups = num_q_heads // 16
        q_workspace_stride = num_q_heads * qk_dim
        # Same contract as gang_mla_decode_layer: the declared row may be
        # narrower than the kernel indexes, so long as the allocation behind
        # it is num_q_heads wide with a zero tail.
        q_workspace_slots = q_workspace_slots or num_q_heads
        assert q_workspace_slots <= num_q_heads
        assert qb_output_stride == q_workspace_slots * qk_dim
        # The sub-kernels take a single request index, and a gang task has no
        # request dimension to hang one on -- grid.x is the XCD. The registrar
        # therefore passes a literal 0 (task_metadata.request_id aliases
        # n_tile_start for this task type and is not readable).
        assert self.max_num_batched_requests == 1, (
            "gang_mla_attn_fused_layer is single-request; the standalone "
            "decode + merge path handles batched requests")
        # batch_size, not just the request count: a work item is
        # (query row, q group, kv chunk). The rows of one request share its
        # page list, so the decode's token dimension lives inside the request
        # one -- see the decomposition comment in gang_mla_decode_mi300.cuh.
        # At batch_size 1 this is exactly the old product.
        mla_total_work_items = (
            self.max_num_batched_requests * batch_size
            * num_q_groups * num_kv_chunks)
        import math
        mla_tiles_per_xcd = math.ceil(mla_total_work_items / 8)
        assert merge_dim_splits >= 1 and kv_lora_rank % merge_dim_splits == 0
        merge_total = (self.max_num_batched_requests * num_q_groups
                       * merge_dim_splits)
        merge_tiles_per_xcd = math.ceil(merge_total / 8)

        # The dispatch width is the widest phase, clamped to the resident
        # workers -- the phases grid-stride past that, and the in-kernel
        # barriers are sized in dispatched workers, so a tile parked behind a
        # busy worker would deadlock them. q_b is +1 for the latent tile,
        # which owns a slot of its own on XCD 0.
        tiles_per_xcd = min(max(batch_size * qkv_n_wgs_per_xcd,
                                batch_size * qb_n_wgs_per_xcd + 1,
                                mla_tiles_per_xcd, merge_tiles_per_xcd),
                            self.num_workers // 8)
        # 29 cache-line-strided int32 slots: qkv_a->q_b at [0..8],
        # q_b->decode at [10..18], decode->merge at [20..28]. All monotonic,
        # so nothing is reset and one buffer serves every layer.
        assert attn_counters.dim(0) >= 29 * 16

        # The residual resolve reads and writes exactly the reduction width,
        # so the qkv_a row cannot be padded on this path.
        assert qkv_actual_hidden_dim == qkv_reduction, (
            f"the fused residual resolve needs an unpadded qkv_a reduction; "
            f"got actual_hidden_dim={qkv_actual_hidden_dim} against "
            f"reduction={qkv_reduction}")
        assert moe_workspace_f32.num_dims == 2
        assert (moe_workspace_f32.dim(0), moe_workspace_f32.dim(1)) == (
            batch_size, qkv_reduction)
        assert x_out.num_dims == 2
        assert (x_out.dim(0), x_out.dim(1)) == (batch_size, qkv_reduction)
        # One workgroup per XCD zeroes its own eighth of the row with
        # dwordx4 write-throughs.
        assert (batch_size * qkv_reduction) % 32 == 0

        params = [batch_size, qkv_output_per_wg, qkv_actual_hidden_dim,
                  qkv_n_wgs_per_xcd, qkv_output_stride, qb_output_per_wg,
                  qb_reduction_size, qb_actual_hidden_dim, qb_n_wgs_per_xcd,
                  qb_output_stride, kv_lora_rank, qk_rope_head_dim, kv_offset,
                  self.max_seq_length, self.page_size, num_q_heads,
                  qk_head_dim, num_kv_chunks, q_workspace_stride,
                  mla_total_work_items, mla_tiles_per_xcd, merge_dim_splits,
                  1 if merge_write_through else 0, merge_tiles_per_xcd,
                  tiles_per_xcd]

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), 1, True)
        tb_graph.new_input(pre_norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(pre_norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(qkv_mxfp8_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(qkv_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(q_a_norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(q_a_norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(qb_mxfp8_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(qb_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(kv_norm_weight, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(attn_counters, (-1, -1, -1), 0, True)
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), 1, True)
        tb_graph.new_input(qkv_a_out, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)
        tb_graph.new_input(attn_out, (-1, -1, -1), -1, True)
        tb_graph.new_input(x_out, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [x, pre_norm_weight, pre_norm_scratch, qkv_mxfp8_weight, qkv_bias,
             q_a_norm_weight, q_a_norm_scratch, qb_mxfp8_weight, qb_bias,
             kv_norm_weight, cos_pos_embed, sin_pos_embed, kv_cache,
             attn_counters, moe_workspace_f32,
             qkv_a_out, q_workspace, lse, o_acc, attn_out, x_out],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_mla_attn_fused_mi300", params)

    def gang_mla_full_layer_fused_layer(
        self,
        # ── attention half ──
        x: DTensor,
        pre_norm_weight: DTensor,
        pre_norm_scratch: DTensor,
        qkv_mxfp8_weight: DTensor,
        qkv_bias: DTensor,
        q_a_norm_weight: DTensor,
        q_a_norm_scratch: DTensor,
        qb_mxfp8_weight: DTensor,
        qb_bias: DTensor,
        kv_norm_weight: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        kv_cache: DTensor,
        moe_workspace_f32: DTensor,
        # ── the one counter buffer both halves share ──
        counters: DTensor,
        # ── MoE half ──
        oproj_mxfp8_weight: DTensor,
        residual: DTensor,
        post_norm_weight: DTensor,
        post_norm_output: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        moe_gate_up_weight: DTensor,
        moe_down_weight: DTensor,
        moe_w13_bias: DTensor,
        moe_w2_bias: DTensor,
        moe_swiglu_out: DTensor,
        # ── outputs ──
        qkv_a_out: DTensor,
        q_workspace: DTensor,
        lse: DTensor,
        o_acc: DTensor,
        attn_out: DTensor,
        x_out: DTensor,
        hidden: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        # ── attention parameters ──
        qkv_output_per_wg: int,
        qkv_actual_hidden_dim: int,
        qb_output_per_wg: int,
        qb_reduction_size: int,
        qb_actual_hidden_dim: int,
        kv_offset: int,
        mla_params: tuple,
        q_workspace_slots: int = None,
        merge_dim_splits: int = 1,
        # ── MoE parameters ──
        oproj_rows_per_wg: int = 16,
        oproj_reduction_size: int = 0,
        actual_hidden_dim: int = 0,
        num_experts_per_tok: int = 4,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        moe_w13_output_per_wg: int = 64,
        moe_w2_output_per_wg: int = 64,
        # -- un-absorbed kv_b_v --
        # Passing wuv_mxfp8_weight moves W_UV back out of o_proj: the layer
        # gains a block-diagonal GEMV
        #   v[h * v_head + j] = sum_c attn_out[h * kv_lora + c] * W_UV[h][c][j]
        # writing v_out, and o_proj reduces over that narrower row instead of
        # over the latent. At GLM-5 that is 6144 x 32768 -> 6144 x 16384 plus
        # a 64 x 512 x 256 stack, i.e. 201 MB -> 109 MB of MXFP8 per layer per
        # rank, against one extra cross-XCD barrier.
        #
        # The weight is packed like every other GEMV weight in this file:
        # [num_q_heads * v_head_dim, kv_lora_rank] rows, pack_dense_mxfp8 at
        # wuv_rows_per_wg. oproj_reduction_size must then be
        # num_q_heads * v_head_dim, and oproj_mxfp8_weight packed at it.
        wuv_mxfp8_weight: DTensor = None,
        v_out: DTensor = None,
        wuv_rows_per_wg: int = 0,
        wuv_v_head_dim: int = 0,
        # -- un-absorbed kv_b_k --
        # The Q-side mirror. Passing wuk_mxfp8_weight moves W_UK back out of
        # q_b: q_b's per-head output narrows from kv_lora_rank + qk_rope to
        # qk_nope_head_dim + qk_rope and lands in q_nope, then a
        # block-diagonal GEMV
        #   q[h * qk_dim + c] = sum_j q_nope[h * (nope + rope) + j] * W_UK[h][j][c]
        # writes the query row's latent columns. q_b's rope workgroup still
        # rotates straight into the query row, so q_nope's own rope columns
        # are written and never read. 77.9 MB -> 34.6 + 6.5 per layer.
        #
        # The weight is packed [num_q_heads * kv_lora_rank, qk_nope_head_dim]
        # at wuk_rows_per_wg, and q_workspace stays the 576-wide query row --
        # it is q_nope that q_b's output stride now describes.
        wuk_mxfp8_weight: DTensor = None,
        q_nope: DTensor = None,
        wuk_rows_per_wg: int = 0,
        qk_nope_head_dim: int = 0,
        # -- router tile width --
        # Experts per router call. 1 is one worker per expert, which at
        # GLM-5's 256 experts is 32 tiles per XCD against 29 workers: two
        # rounds for a mean of 1.10, and the second round re-pays the o_proj
        # barrier spin, the redundant RMSNorm, two block reductions and the
        # arrival atomic to add one dot product. 2 makes it 16 tiles and one
        # round; the row is read once and both gate rows ride the same
        # prefetch across the barrier.
        router_experts_per_tile: int = 1,
        # -- the router fold --
        # logit_e = irms * sum_i h_i * gamma_i * W[e,i], and irms is a positive
        # scalar over the whole row, so both the gate dot and the RMSNorm's
        # sum-of-squares are contractions over exactly the hidden partition
        # the sharded o_proj already imposes. Passing these moves both out of
        # Phase 3 and into the o_proj epilogue, which reduces them across the
        # ranks on the all-gather rendezvous that already exists -- so the
        # router's own work shrinks to a scale by irms and the normed write,
        # and the 3.1 MB of gate weight a rank read per layer becomes 393 KB.
        #
        # Needs the column shard, so EP only, and both or neither.
        #   router_weight_t: [hidden // world_size, num_experts] bf16, the gate
        #     weight transposed and sliced to this rank's o_proj columns.
        #     Transposed because a tile owns four hidden columns and all the
        #     experts, so the row-major form would read num_experts lines at a
        #     hidden-sized stride to use 8 bytes of each.
        #   router_partials: symmetric f32 [8 + 2 * world_size, num_experts+1].
        #     Lines [0..7] are this rank's per-XCD accumulators; the rest are
        #     the per-rank lines the peers push, double-buffered by parity.
        #     Element num_experts of a line is the sum-of-squares.
        router_weight_t: DTensor = None,
        router_partials: DTensor = None,
        # -- expert parallelism --
        # Symmetric-heap tensors (io_category="nvshmem_tensor"). Passing them
        # turns on the head-of-layer fold: this rank's f32 MoE partial plus
        # (on ep_fold_rank only) the residual is rounded to bf16 into slot
        # my_pe of ep_gather, exchanged with every peer, and Phase 1 sums all
        # world_size slots instead of reading (moe_workspace_f32, residual).
        #
        # ep_gather must be a DISTINCT buffer per fused layer -- after the peer
        # wait a peer may fold layer L+1 while this rank is still reading
        # layer L's row. ep_signal can be one buffer for the whole model: its
        # thresholds ride the monotonic layer counter.
        ep_gather: DTensor = None,
        ep_signal: DTensor = None,
        ep_fold_rank: int = 0,
        # The tail variant: run the entry barrier, the fold, the exchange and
        # the peer wait, then return before Phase 1. demo.py emits exactly one
        # of these, immediately after the last real layer, because the fold
        # sits at the HEAD of a layer and the last layer has no successor.
        ep_tail_only: bool = False,
        block_dim: tuple = (256, 1, 1),
    ):
        """A whole GLM decoder layer in one gang dispatch.

        The union of ``gang_mla_attn_fused_layer`` and
        ``gang_oproj_router_fused_layer``, with an in-kernel cross-XCD barrier
        between them where the task graph used to put an event. Six dispatched
        tasks per layer become one.

        Three things differ from calling the two halves back to back, and all
        three are forced rather than optional:

        * ``merge_write_through`` is not a parameter. The split-KV merge writes
          only this XCD's tiles of ``attn_out`` and o_proj reduces over the
          whole row, so with an in-kernel barrier in place of an event the
          store has to go past the producing XCD's L2. The registrar asserts
          it.
        * The three counter buffers the halves used become one ``counters``
          tensor of at least ``71 * 16`` int32. Not for tidiness -- the merged
          input list is 27 slots against a ``MAX_INPUTS_PER_TASK`` of 28, and
          three separate buffers would not fit. See the kernel header for the
          slot map.
        * ``tiles_per_xcd`` is the max over every phase in the *layer*, and
          both halves are given that one number. They each decode
          ``xcd_id = tile_idx / tiles_per_xcd``, so a disagreement across the
          Phase 8 barrier would put the two halves on different XCDs.

        ``attn_out`` is declared once, as an output; the MoE half reads it from
        there. ``moe_workspace_f32`` is declared twice, as the input the
        residual resolve reads (the previous layer's accumulator) and as the
        output W2 accumulates into -- the same buffer in both roles, which is
        what the two-task form did across a layer boundary.

        Inputs (27), outputs (11): see the registrar's comment for the map.
        """
        assert x.num_dims == 2
        assert qkv_mxfp8_weight.num_dims == 2
        assert qb_mxfp8_weight.num_dims == 2
        assert qkv_a_out.num_dims == 2
        assert q_workspace.num_dims == 2
        assert attn_out.num_dims == 2
        assert kv_cache.num_dims == 4  # (num_pages, page_size, 1, qk_dim)
        assert kv_cache.dim(2) == 1, "MLA keeps a single shared latent head"
        assert counters.num_dims == 1
        assert oproj_mxfp8_weight.num_dims == 2
        assert residual.num_dims == 2
        assert hidden.num_dims == 2
        assert router_weight.num_dims == 2
        assert router_bias.num_dims == 1
        assert routing_indices.num_dims == 2
        assert active_expert_ids.num_dims == 1

        ep_inline = ep_gather is not None
        if ep_inline:
            assert ep_signal is not None, \
                "the EP fold needs both ep_gather and ep_signal"
            assert self.world_size > 1, "the EP fold needs world_size > 1"
            assert ep_gather.num_dims == 3   # (world_size, batch, hidden)
            assert ep_gather.dim(0) == self.world_size
            assert 0 <= ep_fold_rank < self.world_size
            # One 64-byte line per PE, so a peer's signal store never shares a
            # line with another's. int32 units because mi.uint64 has no
            # get_datatype_size() entry; the kernel reinterprets the pointer
            # as uint64*.
            assert ep_signal.dim(0) >= self.world_size * 16
        else:
            assert ep_signal is None and not ep_tail_only, \
                "ep_signal / ep_tail_only only mean anything with ep_gather"

        batch_size = self.max_num_batched_tokens
        # q_b, W_UK and W_UV all take an explicit input row stride now; see
        # INPUT_ROW_STRIDE in gang_rmsnorm_linear_mxfp8_bias / gang_gemv_mxfp8
        # / gang_linear_mxfp8.
        assert self.max_num_batched_requests == 1, (
            "the fused layer is single-request; the registrar passes a "
            "literal request index of 0")

        # ══ attention geometry, from gang_mla_attn_fused_layer ══
        num_q_heads = mla_params[0]
        kv_lora_rank = mla_params[1]
        qk_rope_head_dim = mla_params[2]
        qk_head_dim = mla_params[3]
        num_kv_chunks = mla_params[4]
        qk_dim = kv_lora_rank + qk_rope_head_dim
        assert kv_cache.dim(3) == qk_dim
        assert kv_norm_weight.dim(0) == kv_lora_rank
        assert cos_pos_embed.dim(cos_pos_embed.num_dims - 1) == qk_rope_head_dim
        assert num_kv_chunks > 1, (
            "one chunk means no merge phase; the fused layer has no such "
            "variant")

        qkv_reduction = x.dim(1)
        assert qkv_reduction % 512 == 0, qkv_reduction
        qkv_n_wgs = qkv_mxfp8_weight.dim(0)
        assert qkv_n_wgs % 8 == 0
        qkv_n_wgs_per_xcd = qkv_n_wgs // 8
        qkv_output_stride = qkv_a_out.dim(1)
        assert qkv_n_wgs * qkv_output_per_wg == qkv_output_stride, (
            f"packed qkv_a weight covers {qkv_n_wgs * qkv_output_per_wg} "
            f"columns, qkv_a_out has {qkv_output_stride}")
        assert qkv_bias.dim(1) == qkv_output_stride

        assert qb_reduction_size % 512 == 0, qb_reduction_size
        assert qb_actual_hidden_dim <= qb_reduction_size <= qkv_output_stride
        assert kv_offset + qk_dim <= qkv_output_stride
        # q_b's per-head output span, and the row it writes: the query row
        # itself when W_UK is absorbed, the [nope | rope] scratch when it is
        # not. Everything about q_b's shape is stated against these two rather
        # than against qk_dim / q_workspace, since the pairs stop coinciding.
        unabsorb_k = wuk_mxfp8_weight is not None
        qb_head_span = (qk_nope_head_dim + qk_rope_head_dim) if unabsorb_k \
            else qk_dim
        qb_out_tensor = q_nope if unabsorb_k else q_workspace
        # The rope slice has to fit inside one workgroup so the rotation is not
        # split across two workers, and it is the tail of a head, so a wider
        # workgroup is fine as long as heads still divide evenly. The kernel
        # offsets its rope pointer by (output_per_wg - qk_rope_head_dim).
        #
        # Un-absorbed, that floor is gone: the kernel defers the rotation past
        # Phase 3b's XCD-local W_UK release, which is the barrier the split
        # slice needed, and runs it on the last W_UK tile of each head. The
        # switch is compile-time off exactly this comparison, so the two sides
        # cannot disagree.
        assert (unabsorb_k or qb_output_per_wg >= qk_rope_head_dim) and \
            qb_head_span % qb_output_per_wg == 0, (
                "output_per_wg must be at least qk_rope_head_dim (absorbed "
                "only) and divide the head span")
        qb_n_wgs = qb_mxfp8_weight.dim(0)
        assert qb_n_wgs % 8 == 0
        qb_n_wgs_per_xcd = qb_n_wgs // 8
        qb_output_stride = qb_out_tensor.dim(1)
        # Either the whole nope row, or this rank's 1/world-th of it under the
        # head-sharded q_b. The kernel reads the shard off exactly this ratio
        # -- there is no flag -- so nothing between the two is legal, and
        # qb_output_stride stays the FULL row either way: it is the scratch's
        # declared width, which every rank allocates whole and only writes its
        # own head slice of.
        qb_cols = qb_n_wgs * qb_output_per_wg
        assert qb_cols == qb_output_stride or (
            ep_inline and qb_cols * self.world_size == qb_output_stride), (
                f"packed q_b covers {qb_cols} columns; the nope row is "
                f"{qb_output_stride} and world is "
                f"{self.world_size if ep_inline else 1}")
        assert qb_mxfp8_weight.dim(1) == qb_output_per_wg * (
            qb_reduction_size + qb_reduction_size // 32)
        # The sharded form may keep the bias at the full row -- it is indexed
        # by a column offset that never leaves this rank's slice, so a whole
        # row is in bounds and one fewer tensor changes shape.
        assert qb_bias.dim(1) in (qb_cols, qb_output_stride)
        assert (qb_n_wgs_per_xcd * qb_output_per_wg) % qb_head_span == 0, (
            "per-XCD chunk must hold whole heads")

        assert num_q_heads % 16 == 0
        num_q_groups = num_q_heads // 16
        q_workspace_stride = num_q_heads * qk_dim
        q_workspace_slots = q_workspace_slots or num_q_heads
        assert q_workspace_slots <= num_q_heads
        assert qb_output_stride == q_workspace_slots * qb_head_span
        # batch_size, not just the request count: a work item is
        # (query row, q group, kv chunk). The rows of one request share its
        # page list, so the decode's token dimension lives inside the request
        # one -- see the decomposition comment in gang_mla_decode_mi300.cuh.
        # At batch_size 1 this is exactly the old product.
        mla_total_work_items = (
            self.max_num_batched_requests * batch_size
            * num_q_groups * num_kv_chunks)
        import math
        mla_tiles_per_xcd = math.ceil(mla_total_work_items / 8)
        assert merge_dim_splits >= 1 and kv_lora_rank % merge_dim_splits == 0
        merge_total = (self.max_num_batched_requests * num_q_groups
                       * merge_dim_splits)
        merge_tiles_per_xcd = math.ceil(merge_total / 8)

        assert qkv_actual_hidden_dim == qkv_reduction, (
            f"the fused residual resolve needs an unpadded qkv_a reduction; "
            f"got actual_hidden_dim={qkv_actual_hidden_dim} against "
            f"reduction={qkv_reduction}")
        assert moe_workspace_f32.num_dims == 2
        assert (moe_workspace_f32.dim(0), moe_workspace_f32.dim(1)) == (
            batch_size, qkv_reduction)
        assert x_out.num_dims == 2
        assert (x_out.dim(0), x_out.dim(1)) == (batch_size, qkv_reduction)
        assert (batch_size * qkv_reduction) % 32 == 0

        # ══ MoE geometry, from gang_oproj_router_fused_layer ══
        assert oproj_rows_per_wg >= 4 and \
            (oproj_rows_per_wg & (oproj_rows_per_wg - 1)) == 0
        assert 0 < oproj_reduction_size <= attn_out.dim(1)
        assert oproj_reduction_size % 32 == 0
        n_wgs = oproj_mxfp8_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        oproj_tiles_per_xcd = n_wgs // 8
        # MXFP8 stores K data bytes a row, MXFP4 K/2; the scale half is K/32
        # either way. Accept both and let the width say which, so the check
        # still catches a genuinely wrong shape without needing the flag
        # threaded down here.
        assert oproj_mxfp8_weight.dim(1) in (
            oproj_rows_per_wg * (oproj_reduction_size +
                                 oproj_reduction_size // 32),
            oproj_rows_per_wg * (oproj_reduction_size // 2 +
                                 oproj_reduction_size // 32)), (
            oproj_mxfp8_weight.dim(1), oproj_rows_per_wg, oproj_reduction_size)
        unabsorb_v = wuv_mxfp8_weight is not None
        if unabsorb_v:
            assert v_out is not None and wuv_rows_per_wg > 0 \
                and wuv_v_head_dim > 0, \
                "un-absorbing kv_b_v needs v_out, wuv_rows_per_wg and " \
                "wuv_v_head_dim as well as the weight"
            assert oproj_reduction_size == num_q_heads * wuv_v_head_dim, (
                f"un-absorbed o_proj reduces over num_q_heads * v_head_dim = "
                f"{num_q_heads * wuv_v_head_dim}, got {oproj_reduction_size}")
            assert wuv_v_head_dim % wuv_rows_per_wg == 0
            wuv_n_wgs = wuv_mxfp8_weight.dim(0)
            assert wuv_n_wgs % 8 == 0
            wuv_tiles_per_xcd = wuv_n_wgs // 8
            # Either the whole V row, or this rank's 1/world-th of it under
            # output-wise sharded W_UV followed by the in-kernel all-gather.
            # The kernel detects the shard off exactly this ratio -- there is
            # no flag -- and oproj_reduction_size stays the FULL row either
            # way: it is o_proj's K and the all-gather's target width. Same
            # relaxation the o_proj weight already has in task_register.cc.
            wuv_covered = wuv_n_wgs * wuv_rows_per_wg
            assert wuv_covered > 0 and oproj_reduction_size % wuv_covered == 0, (
                f"packed W_UV covers {wuv_covered} columns, which neither is "
                f"nor evenly divides the V row {oproj_reduction_size}")
            assert wuv_mxfp8_weight.dim(1) == wuv_rows_per_wg * (
                kv_lora_rank + kv_lora_rank // 32)
            assert v_out.num_dims == 2
            assert (v_out.dim(0), v_out.dim(1)) == (
                batch_size, oproj_reduction_size)
        else:
            assert v_out is None and wuv_rows_per_wg == 0 \
                and wuv_v_head_dim == 0
            assert oproj_reduction_size == num_q_heads * kv_lora_rank, (
                f"absorbed o_proj reduces over num_q_heads * kv_lora_rank = "
                f"{num_q_heads * kv_lora_rank}, got {oproj_reduction_size}")
            wuv_tiles_per_xcd = 0
        if unabsorb_k:
            assert q_nope is not None and wuk_rows_per_wg > 0 \
                and qk_nope_head_dim > 0, \
                "un-absorbing kv_b_k needs q_nope, wuk_rows_per_wg and " \
                "qk_nope_head_dim as well as the weight"
            assert q_workspace.dim(1) == num_q_heads * qk_dim, (
                "un-absorbed, q_b writes q_nope and Phase 3b writes the whole "
                "query row, so q_workspace cannot be a slot subset")
            assert q_nope.num_dims == 2
            assert (q_nope.dim(0), q_nope.dim(1)) == (
                batch_size, num_q_heads * qb_head_span)
            assert kv_lora_rank % wuk_rows_per_wg == 0
            wuk_n_wgs = wuk_mxfp8_weight.dim(0)
            assert wuk_n_wgs % 8 == 0
            wuk_tiles_per_xcd = wuk_n_wgs // 8
            # Whole query row, or this rank's 1/world-th of it under the head
            # shard. W_UK follows q_b head for head; the XCD-local-barrier
            # assert below re-checks that they agree.
            wuk_rows = wuk_n_wgs * wuk_rows_per_wg
            assert wuk_rows == num_q_heads * kv_lora_rank or (
                ep_inline
                and wuk_rows * self.world_size == num_q_heads * kv_lora_rank), (
                    f"packed W_UK covers {wuk_rows} rows, the query row's "
                    f"latent columns are {num_q_heads * kv_lora_rank} and "
                    f"world is {self.world_size if ep_inline else 1}")
            assert wuk_mxfp8_weight.dim(1) == wuk_rows_per_wg * (
                qk_nope_head_dim + qk_nope_head_dim // 32)
            # Phase 3b's barrier is XCD-local, which is only legal because the
            # producer and the consumer sit on the same heads.
            assert wuk_tiles_per_xcd // (kv_lora_rank // wuk_rows_per_wg) == \
                (qb_n_wgs_per_xcd * qb_output_per_wg) // qb_head_span, \
                "q_b and W_UK disagree on how many heads live on an XCD"
        else:
            assert q_nope is None and wuk_rows_per_wg == 0 \
                and qk_nope_head_dim == 0
            wuk_tiles_per_xcd = 0
        hidden_size = hidden.dim(1)
        # Either the whole row, or this rank's 1/world-th of it under
        # output-wise sharded o_proj. The kernel reads the shard off exactly
        # this ratio -- there is no flag -- so nothing between the two is
        # legal, and hidden_size stays the FULL row either way: it is the
        # RMSNorm's and the router's K, and the all-gather's target width.
        oproj_cols = n_wgs * oproj_rows_per_wg
        assert oproj_cols == hidden_size or (
            ep_inline and oproj_cols * self.world_size == hidden_size), (
                f"packed weight covers {oproj_cols} columns; hidden has "
                f"{hidden_size} and world is "
                f"{self.world_size if ep_inline else 1}")
        assert hidden_size == qkv_reduction, (
            "o_proj's N is the next layer's qkv_a K; they are one row")

        num_experts = router_weight.dim(0)
        assert num_experts % 8 == 0
        # router_tile_n is the TILE count per XCD, not the expert count: each
        # tile carries router_experts_per_tile experts off one pass over the
        # row. total_router_tiles is what the gang counter's TopK tail elects
        # on, so it follows.
        assert router_experts_per_tile >= 1
        assert num_experts % (8 * router_experts_per_tile) == 0, (
            f"{num_experts} experts do not split into 8 XCDs of "
            f"{router_experts_per_tile}-expert tiles")
        router_tile_n = num_experts // 8 // router_experts_per_tile
        total_router_tiles = router_tile_n * 8
        assert router_bias.dim(0) == num_experts
        # -- the router fold --
        router_fold = router_weight_t is not None
        assert router_fold == (router_partials is not None), \
            "the router fold needs both router_weight_t and router_partials"
        if router_fold:
            assert ep_inline, \
                "the router fold reduces over the o_proj column shard, " \
                "which only exists under EP"
            # The same slice the o_proj shard takes; see OPROJ_TP_COLS.
            oproj_tp_cols = hidden_size // self.world_size
            assert (router_weight_t.num_dims == 2
                    and router_weight_t.dim(0) == oproj_tp_cols
                    and router_weight_t.dim(1) == num_experts), (
                f"router_weight_t must be [{oproj_tp_cols}, {num_experts}], "
                f"got [{router_weight_t.dim(0)}, {router_weight_t.dim(1)}]")
            assert (router_partials.num_dims == 2
                    and router_partials.dim(0) == 8 + 2 * self.world_size
                    and router_partials.dim(1) == num_experts + 1), (
                f"router_partials must be "
                f"[{8 + 2 * self.world_size}, {num_experts + 1}], got "
                f"[{router_partials.dim(0)}, {router_partials.dim(1)}]")
        num_shared_experts = routing_indices.dim(0) - num_experts
        assert num_shared_experts in (0, 1)
        assert topk_weight.dim(1) == num_experts_per_tok + num_shared_experts

        assert moe_gate_up_weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_down_weight.num_dims == 3
        assert moe_w13_bias.num_dims == 2
        assert moe_w2_bias.num_dims == 2
        assert moe_swiglu_out.num_dims == 3      # [batch, topk_total, inter]
        # The WEIGHT tensors hold this rank's slice; the router, the routing
        # table and the activated list stay replicated over all num_experts,
        # which is what the kernel is templated on.
        moe_num_local_experts = moe_gate_up_weight.dim(0)
        moe_num_experts = num_experts + num_shared_experts
        ep_ws = self.world_size if ep_inline else 1
        assert num_experts % ep_ws == 0, (
            f"ep_slice needs {num_experts} routed experts to divide by "
            f"world_size {ep_ws}")
        assert moe_down_weight.dim(0) == moe_num_local_experts
        assert moe_num_local_experts == \
            num_experts // ep_ws + num_shared_experts, (
                f"expert weights hold {moe_num_local_experts} experts; this "
                f"rank's ep_slice is {num_experts // ep_ws} routed + "
                f"{num_shared_experts} shared")
        moe_w13_width = moe_gate_up_weight.dim(1) * moe_w13_output_per_wg
        assert moe_w13_bias.dim(1) == moe_w13_width
        assert moe_down_weight.dim(1) * moe_w2_output_per_wg == hidden_size
        assert moe_w2_bias.dim(1) == hidden_size
        moe_intermediate = moe_w13_width // 2
        assert moe_swiglu_out.dim(2) == moe_intermediate
        assert moe_swiglu_out.dim(1) == num_experts_per_tok + num_shared_experts
        assert hidden_size % 512 == 0, \
            f"MXFP8 W13 K={hidden_size} not divisible by 512"
        assert moe_intermediate % 512 == 0, \
            f"MXFP8 W2 K={moe_intermediate} not divisible by 512"

        def _moe_wg_bytes(opw, k, fp4):
            return opw * ((k // 2 if fp4 else k) + k // 32)

        moe_fp4 = moe_gate_up_weight.dim(2) == _moe_wg_bytes(
            moe_w13_output_per_wg, hidden_size, True)
        assert moe_gate_up_weight.dim(2) == _moe_wg_bytes(
            moe_w13_output_per_wg, hidden_size, moe_fp4), (
            f"W13 workgroup stride {moe_gate_up_weight.dim(2)} is neither the "
            f"MXFP8 nor the MXFP4 packing of {moe_w13_output_per_wg} rows of "
            f"K={hidden_size}")
        assert moe_down_weight.dim(2) == _moe_wg_bytes(
            moe_w2_output_per_wg, moe_intermediate, moe_fp4), (
            f"W2 workgroup stride {moe_down_weight.dim(2)} disagrees with "
            f"W13 on the element width (W13 is "
            f"{'MXFP4' if moe_fp4 else 'MXFP8'})")

        moe_topk_total = moe_swiglu_out.dim(1)
        moe_max_activated = min(moe_topk_total * batch_size, moe_num_experts)
        moe_w13_tiles_per_xcd = (
            moe_max_activated * batch_size * moe_gate_up_weight.dim(1) + 7) // 8
        # MPK_W2_KSPLIT widens W2's per-expert tile space by the split factor
        # (each tile covers 1/N of the reduction), so the dispatch loop bound
        # has to widen with it or the tiles carrying the tail of K are never
        # issued and W2's output is short by that fraction. The device reads
        # the same env var as a -D; they must agree.
        moe_w2_tiles_per_xcd = (
            moe_max_activated * batch_size * moe_down_weight.dim(1)
            * int(os.environ.get("MPK_W2_KSPLIT", "1")) + 7) // 8

        oproj_topk_tiles_per_xcd = max(oproj_tiles_per_xcd, router_tile_n)

        # ══ the one dispatch width ══
        # Every phase of the layer, both halves. Whichever is widest sets the
        # worker count, and every barrier in the task is sized against it --
        # but only up to the resident workers, because the barriers count
        # *dispatched* workers and a tile parked behind a busy one deadlocks
        # them. Past that point the phases grid-stride.
        #
        # MEASURED at the current geometry with MPK_PRINT_GEOMETRY=1 (the
        # numbers this comment used to carry -- 48 o_proj, 33 q_b, 32 router --
        # predate the head shard and the router EPT=2 and are wrong):
        #
        #   dispatch width 29 tiles/XCD
        #   qkv_a 24  q_b+kvupd 17  mla_decode 8  merge 16
        #   o_proj 24  router 16  W_UV 16  W_UK 4      -- all ONE round
        #   moe_W13 72 (3 rounds)   moe_W2 108 (4 rounds)
        #
        # So the attention half is not round-quantized at all; it is simply
        # occupancy-starved, 4/29 to 24/29 busy. The only multi-round phases in
        # the layer are the two MoE loops, and their bounds are the REPLICATED
        # worst case (moe_max_activated = topk+shared), not what a rank owns --
        # under EP=8 a rank owns about one activated expert, so most of those
        # 3+4 rounds decode a tile that is not theirs and return false.
        tiles_per_xcd = min(max(batch_size * qkv_n_wgs_per_xcd,
                                batch_size * qb_n_wgs_per_xcd + 1,
                                mla_tiles_per_xcd, merge_tiles_per_xcd,
                                oproj_topk_tiles_per_xcd,
                                wuv_tiles_per_xcd, wuk_tiles_per_xcd,
                                moe_w13_tiles_per_xcd, moe_w2_tiles_per_xcd),
                            self.num_workers // 8)
        total_barrier_arrivals = min(oproj_topk_tiles_per_xcd,
                                     tiles_per_xcd) * 8

        # Host-side only, costs a run nothing, and every tile-geometry decision
        # on this branch has turned on the same arithmetic: a phase's makespan
        # is ceil(tiles / workers_per_xcd) ROUNDS, so a phase that wants 33
        # tiles against 29 workers costs two full rounds with 25 of 29 workers
        # idle through the second. "util" below is the fraction of the last
        # round that is doing work -- the smaller it is, the more of the phase
        # is round-quantization waste rather than work. The MoE rows are the
        # padded worst case (moe_max_activated), not the owned count.
        if os.environ.get("MPK_PRINT_GEOMETRY", "0") == "1":
            _w = max(1, tiles_per_xcd)
            print(f"[GEOM] dispatch width = {_w} tiles/XCD", flush=True)
            for _nm, _t in (("qkv_a", batch_size * qkv_n_wgs_per_xcd),
                            ("q_b+kvupd", batch_size * qb_n_wgs_per_xcd + 1),
                            ("mla_decode", mla_tiles_per_xcd),
                            ("merge", merge_tiles_per_xcd),
                            ("o_proj", oproj_tiles_per_xcd),
                            ("router", router_tile_n),
                            ("W_UV", wuv_tiles_per_xcd),
                            ("W_UK", wuk_tiles_per_xcd),
                            ("moe_W13*", moe_w13_tiles_per_xcd),
                            ("moe_W2*", moe_w2_tiles_per_xcd)):
                if _t <= 0:
                    print(f"[GEOM]   {_nm:<11} -- not live", flush=True)
                    continue
                _r = -(-_t // _w)
                _last = _t - (_r - 1) * _w
                print(f"[GEOM]   {_nm:<11} tiles/XCD={_t:<5} rounds={_r}"
                      f"  last round {_last}/{_w} busy"
                      f"  ({100.0 * _t / (_r * _w):.0f}% of the rounds it pays)",
                      flush=True)
            # The two moe_* rows above are STATIC and are ~3x the live count.
            # They use moe_max_activated = topk + shared = every slot landing on
            # one rank; at EP=4 the busiest rank actually owns E[max] = 3.0
            # activated experts, so the live loop bounds -- computed on device
            # at gang_oproj_router_fused_mi300.cuh:~1414 -- are ~24 W13 and ~18
            # W2 tiles/XCD, i.e. ONE grid-stride round each on 30 workers with
            # 6 and 12 workers idle. Three separate levers have now been derived
            # from the static numbers and all three were nulls (the last was
            # MPK_W13_PRESTAGE, which hoists a per-tile prologue out of a loop
            # that runs exactly once). Read the device-side *_live computation,
            # not this table, before pricing anything in the MoE.
            if moe_num_experts > 0:
                _e_live = 3.0  # E[max owned] at EP=4; see the header comment
                for _nm, _tpe in (("moe_W13*", moe_gate_up_weight.dim(1)),
                                  ("moe_W2*", moe_down_weight.dim(1))):
                    _lt = _e_live * batch_size * _tpe / 8.0
                    print(f"[GEOM]   {_nm:<11} LIVE ~{_lt:.0f} tiles/XCD"
                          f"  ({-(-int(_lt) // _w)} round(s)) -- the row above "
                          f"is the static worst case", flush=True)

        # 80 cache-line-strided int32 slots, or 96 under EP. See the kernel
        # header for the map; every barrier is monotonic, so nothing is reset
        # and one buffer serves every layer of every iteration.
        #
        # 80, not 71: FULL_LAYER_ENTRY_SLOT is 71 and the layer-entry barrier
        # spans nine lines from there (eight per-XCD release flags plus the
        # arrival counter at [79]). Under EP add the fold counter at [88].
        # Undersizing this does not fail loudly, it corrupts whatever torch
        # allocated next -- demo.py already asks for 96 either way.
        # Un-absorbed kv_b_v adds Phase 8b's barrier at [96 .. 104].
        counter_slots = (96 if ep_inline else 80)
        if unabsorb_v:
            counter_slots = 106
        # Phase 3b's XCD-local barrier adds [106 .. 113]; it sits past W_UV's
        # region whether or not that one is live, so the map never renumbers.
        if unabsorb_k:
            counter_slots = 114
        assert counters.dim(0) >= counter_slots * 16, (
            f"the fused layer needs {counter_slots * 16} int32 of counters, "
            f"got {counters.dim(0)}")

        scaling_milli = int(round(routed_scaling_factor * 1000.0))
        assert abs(scaling_milli / 1000.0 - routed_scaling_factor) < 1e-9

        params = [
            # attention half (25)
            batch_size, qkv_output_per_wg, qkv_actual_hidden_dim,
            qkv_n_wgs_per_xcd, qkv_output_stride, qb_output_per_wg,
            qb_reduction_size, qb_actual_hidden_dim, qb_n_wgs_per_xcd,
            qb_output_stride, kv_lora_rank, qk_rope_head_dim, kv_offset,
            self.max_seq_length, self.page_size, num_q_heads, qk_head_dim,
            num_kv_chunks, q_workspace_stride, mla_total_work_items,
            mla_tiles_per_xcd, merge_dim_splits,
            1,  # merge_write_through, forced -- see the docstring
            merge_tiles_per_xcd, tiles_per_xcd,
            # MoE half (17)
            hidden_size, oproj_rows_per_wg, oproj_tiles_per_xcd,
            total_barrier_arrivals, actual_hidden_dim, num_experts,
            num_experts_per_tok, router_tile_n, total_router_tiles,
            oproj_reduction_size, scaling_milli, 1 if norm_topk_prob else 0,
            moe_intermediate, moe_w13_output_per_wg, moe_w2_output_per_wg,
            moe_w13_tiles_per_xcd, moe_w2_tiles_per_xcd,
            # expert parallelism (4)
            self.world_size if ep_inline else 1,
            self.mpi_rank if ep_inline else 0,
            ep_fold_rank if ep_inline else 0,
            1 if ep_tail_only else 0,
            # un-absorbed kv_b_v (3)
            wuv_rows_per_wg, wuv_v_head_dim, wuv_tiles_per_xcd,
            # un-absorbed kv_b_k (3)
            qk_nope_head_dim, wuk_rows_per_wg, wuk_tiles_per_xcd,
            # router tile width (1) -- appended last so nothing renumbers
            router_experts_per_tile,
            # the router fold (1)
            1 if router_fold else 0,
        ]
        assert len(params) == 54

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Partition maps are each half's, unchanged: the weights split by
        # column across the XCDs, the shared rows do not.
        tb_graph.new_input(x, (-1, -1, -1), 1, True)
        tb_graph.new_input(pre_norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(pre_norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(qkv_mxfp8_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(qkv_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(q_a_norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(q_a_norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(qb_mxfp8_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(qb_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(kv_norm_weight, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), 1, True)
        tb_graph.new_input(counters, (-1, -1, -1), 0, True)
        tb_graph.new_input(oproj_mxfp8_weight, (0, -1, -1), 1, True)
        # Unpartitioned; see the note on the same input in
        # gang_oproj_router_fused_layer. Was (1, -1, -1), same arithmetic.
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(post_norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(post_norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(router_bias, (-1, -1, -1), 0, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(moe_gate_up_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_down_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_w13_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_w2_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_swiglu_out, (-1, 2, -1), -1, True)
        if ep_inline:
            tb_graph.new_input(ep_gather, (-1, -1, -1), -1, True)   # [27]
            tb_graph.new_input(ep_signal, (-1, -1, -1), -1, True)   # [28]
        if unabsorb_v:
            # Last input, so that the EP pair keeps 27/28 either way.
            tb_graph.new_input(wuv_mxfp8_weight, (0, -1, -1), 1, True)
        if unabsorb_k:
            # And after that one, in a fixed W_UV-then-W_UK order so either
            # can be on alone. Dim-0 partitioned like every other GEMV weight:
            # the kernel gets this XCD's slice and indexes it locally.
            tb_graph.new_input(wuk_mxfp8_weight, (0, -1, -1), 1, True)
        if router_fold:
            # Unpartitioned, both of them. router_weight_t is already sliced
            # to this RANK's hidden columns and every XCD reads the rows its
            # own o_proj tiles own, which is a slice of dim 0 the partition
            # map cannot express (it is by tile, not by XCD-equal-chunk);
            # router_partials is indexed by XCD and by PE explicitly.
            tb_graph.new_input(router_weight_t, (-1, -1, -1), -1, True)
            tb_graph.new_input(router_partials, (-1, -1, -1), -1, True)
        tb_graph.new_input(qkv_a_out, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)
        tb_graph.new_input(attn_out, (-1, -1, -1), -1, True)
        tb_graph.new_input(x_out, (-1, -1, -1), -1, True)
        tb_graph.new_input(hidden, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), -1, True)
        if unabsorb_v:
            tb_graph.new_input(v_out, (-1, -1, -1), -1, True)   # output [11]
        if unabsorb_k:
            tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)  # output [12]
        self.kn_graph.customized(
            [x, pre_norm_weight, pre_norm_scratch, qkv_mxfp8_weight, qkv_bias,
             q_a_norm_weight, q_a_norm_scratch, qb_mxfp8_weight, qb_bias,
             kv_norm_weight, cos_pos_embed, sin_pos_embed, kv_cache,
             moe_workspace_f32, counters,
             oproj_mxfp8_weight, residual, post_norm_weight, post_norm_output,
             router_weight, router_bias, logits_scratch,
             moe_gate_up_weight, moe_down_weight, moe_w13_bias, moe_w2_bias,
             moe_swiglu_out]
            + ([ep_gather, ep_signal] if ep_inline else [])
            + ([wuv_mxfp8_weight] if unabsorb_v else [])
            + ([wuk_mxfp8_weight] if unabsorb_k else [])
            + ([router_weight_t, router_partials] if router_fold else [])
            + [qkv_a_out, q_workspace, lse, o_acc, attn_out, x_out,
             hidden, topk_weight, routing_indices, active_expert_ids,
             moe_workspace_f32]
            + ([v_out] if unabsorb_v else [])
            + ([q_nope] if unabsorb_k else []),
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_mla_full_layer_fused_mi300", params)

    def gang_paged_attention_split_kv_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        block_dim: tuple,
    ):
        """Gang merge split-KV: 8 tasks (1 per XCD), broadcast to workers."""
        assert lse.num_dims == 3
        assert output_tmp.num_dims == 3
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang attention only supported on MI300X"

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_kv_heads = attention_params[2]
        num_qo_heads_per_kv = num_q_heads // num_kv_heads

        total_work_items = self.max_num_batched_requests * num_kv_heads
        import math
        total_work_items_per_xcd = math.ceil(total_work_items / 8)

        # params: [num_qo_heads_per_kv, head_dim, max_seq_len, page_size,
        #          num_kv_heads, total_work_items_per_xcd, total_work_items]
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length,
                  self.page_size, num_kv_heads, total_work_items_per_xcd,
                  total_work_items]

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # No bid.y partitioning — full tensors
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([lse, output_tmp, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "gang_attn_merge_mi300", params)

    # MoE Layers
    def tensor_init_layer(
        self,
        input: DTensor,
        dummy_input: DTensor,
        dummy_output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that output
        assert input.num_dims == 2  # (batch_size, output_size)
        assert dummy_input.num_dims == 2 # (batch_size, hidden_size)
        assert dummy_output.num_dims == 2 # (batch_size, output_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_output, (0, -1, -1), -1, True)
        self.kn_graph.customized([input, dummy_input, dummy_output], tb_graph)

        self.kn_graph.register_task(tb_graph, "tensor_init")
    
    def moe_topk_softmax_routing_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, num_experts)
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (num_experts + 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_masks, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, moe_topk_weight, moe_routing_indices, moe_masks], tb_graph)

        if self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_topk_softmax_mi300")
        else:
            self.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100")

    def moe_topk_sigmoid_bias_routing_layer(
        self,
        input: DTensor,
        bias: DTensor,
        output: tuple[DTensor, DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
    ):
        # `noaux_tc` router (GLM-5 / DeepSeek-V3): sigmoid scoring, selection on
        # score + e_score_correction_bias, weights taken from the unbiased score.
        assert input.num_dims == 2  # (batch_size, num_experts)
        assert bias.num_dims == 1  # (num_experts,)
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (num_experts + 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_masks, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, bias, moe_topk_weight, moe_routing_indices, moe_masks], tb_graph)

        assert self.target_cc in (94, 95), "sigmoid+bias router is MI300/MI350 only"
        # routed_scaling_factor travels as an int (register_task takes ints only);
        # 1/1000 units is exact for the values these configs use (GLM-5: 2.5).
        scaling_milli = int(round(routed_scaling_factor * 1000.0))
        assert abs(scaling_milli / 1000.0 - routed_scaling_factor) < 1e-9, (
            f"routed_scaling_factor {routed_scaling_factor} not representable in 1/1000 units"
        )
        self.kn_graph.register_task(
            tb_graph,
            "moe_topk_sigmoid_bias_mi300",
            [scaling_milli, 1 if norm_topk_prob else 0],
        )


    def moe_w13_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 3  # (num_experts, 2*intermediate_size, hidden_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert bias.num_dims == 2  # (num_experts, output_stride)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, 2*intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm100")
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_mi300")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm90")
        else:
            assert False
            
    def moe_silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3 # (batch_size, num_expert_per_tok, 2 * intermediate_size)
        assert output.num_dims == 3 # (batch_size, num_expert_per_tok, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_silu_mul")

    def moe_swigluoai_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # SwigluOAI activation for GPT-OSS
        # Input: (batch_size, num_expert_per_tok, 2 * intermediate_size) - interleaved gate/up
        # Output: (batch_size, num_expert_per_tok, intermediate_size)
        assert input.num_dims == 3
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_swigluoai")

    def moe_w2_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_expert_per_tok, intermediate_size)
        assert weight.num_dims == 3  # (num_experts, hidden_size, intermediate_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert bias.num_dims == 2  # (num_experts, output_stride)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm100")
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_mi300")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm90")
        else:
            assert False
        
    def moe_w13_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W13 linear with MXFP4 weights + bias (native FP4 format, no dequant).
        Weight shape: [num_experts, expert_wgs, wg_bytes] as uint8.
        Bias shape: [num_experts, 2*intermediate_size] as bfloat16.
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [num_experts, expert_wgs, wg_bytes] uint8
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3    # [num_experts, expert_wgs, output_per_wg]
        assert output.num_dims == 3  # [batch, topk, 2*intermediate]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w13_linear_mxfp4_mi300", [output_per_wg])

    def moe_w2_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W2 linear with MXFP4 weights + bias (native FP4 format, no dequant).
        Weight shape: [num_experts, expert_wgs, wg_bytes] as uint8.
        Bias shape: [num_experts, hidden_size] as bfloat16.
        """
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [num_experts, expert_wgs, wg_bytes] uint8
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3    # [num_experts, expert_wgs, output_per_wg]
        assert output.num_dims == 3  # [batch, topk, hidden]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w2_linear_mxfp4_mi300", [output_per_wg])

    def moe_w13_linear_mxfp4_ck_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W13 linear with MXFP4 weights + bias (MFMA-based, replaces scalar GEMV).
        Same weight format as moe_w13_linear_mxfp4_layer.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 3
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w13_linear_mxfp4_ck_mi300", [output_per_wg])

    def moe_w2_linear_mxfp4_ck_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W2 linear with MXFP4 weights + bias (MFMA-based, replaces scalar GEMV).
        Same weight format as moe_w2_linear_mxfp4_layer.
        """
        assert input.num_dims == 3
        assert weight.num_dims == 3
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w2_linear_mxfp4_ck_mi300", [output_per_wg])

    def gang_moe_w13_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        block_dim: tuple = (256, 1, 1),
        fuse_swiglu: bool = False,
    ):
        """Gang MoE W13 linear: 8 tasks (1/XCD), workers cooperate per expert.
        All workers on an XCD process the same expert's GEMM tiles concurrently,
        eliminating L2 thrashing from concurrent expert weight loads.

        fuse_swiglu folds the SiLU-mul into the epilogue, so `output` is the
        half-width [batch, topk, intermediate] activation and no separate
        moe_silu_mul task is needed. It requires `weight` to carry gate and up
        rows *pairwise* interleaved (row 2j = gate_j, row 2j+1 = up_j) -- the
        epilogue only sees a gate/up pair in registers when they are adjacent
        columns. Bias must be interleaved to match.
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [num_experts, 2*intermediate, hidden_size]
        assert moe_routing_indices.num_dims == 2  # [num_experts, batch_size]
        assert moe_mask.num_dims == 1  # [num_experts + 1]
        assert bias.num_dims == 2  # [num_experts, output_stride]
        assert output.num_dims == 3  # [batch, topk, 2*intermediate]
        assert self.target_cc in (94, 95), "Gang MoE linear only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        output_size = weight.dim(1)
        if fuse_swiglu:
            assert output.dim(2) * 2 == output_size, (
                f"fuse_swiglu output must be half the GEMM width: "
                f"{output.dim(2)} vs {output_size}")
        else:
            assert output.dim(2) == output_size
        reduction_size = weight.dim(2)
        tile_n = 64
        assert output_size % tile_n == 0, f"output_size {output_size} not divisible by {tile_n}"
        assert reduction_size % 256 == 0, f"W13 K={reduction_size} not divisible by 256"
        n_tiles = output_size // tile_n
        m_tiles = max(1, batch_size // 16)
        tiles_per_expert = m_tiles * n_tiles
        total_tiles_all = num_experts * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t (bs={batch_size})"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_linear_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, 1 if fuse_swiglu else 0],
        )

    def gang_moe_w2_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        block_dim: tuple = (256, 1, 1),
        routing_weight: DTensor = None,
    ):
        """Gang MoE W2 linear: 8 tasks (1/XCD), workers cooperate per expert.
        W2 processes tokens one at a time (per-token topk_slot offsets differ).
        tiles_per_expert = n_tiles * batch_size.

        Passing `routing_weight` ([batch, topk] f32) fuses the topk weighting
        and the cross-expert sum into the epilogue: it scales by the routing
        weight and float32-atomicAdds into `output`, which is then the
        [batch, hidden] f32 workspace that moe_residual_add_f32 consumes,
        rather than a [batch, topk, hidden] slab.
        """
        fuse_mulsumadd = routing_weight is not None
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [num_experts, hidden_size, intermediate]
        assert moe_routing_indices.num_dims == 2  # [num_experts, batch_size]
        assert moe_mask.num_dims == 1  # [num_experts + 1]
        assert bias.num_dims == 2  # [num_experts, output_stride]
        if fuse_mulsumadd:
            assert output.num_dims == 2       # [batch, hidden_size] f32
            assert routing_weight.num_dims == 2  # [batch, topk] f32
            assert routing_weight.dim(1) == input.dim(1)
        else:
            assert output.num_dims == 3  # [batch, topk, hidden_size]
        assert self.target_cc in (94, 95), "Gang MoE linear only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        output_size = weight.dim(1)
        reduction_size = weight.dim(2)
        tile_n = 64
        assert output_size % tile_n == 0, f"output_size {output_size} not divisible by {tile_n}"
        assert reduction_size % 128 == 0, f"W2 K={reduction_size} not divisible by 128"
        n_tiles = output_size // tile_n
        # W2: one token at a time, so tiles_per_expert = n_tiles * batch_size
        tiles_per_expert = n_tiles * batch_size
        total_tiles_all = num_experts * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t (bs={batch_size})"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tensors = [input, weight, moe_routing_indices, moe_mask, bias]
        if fuse_mulsumadd:
            tb_graph.new_input(routing_weight, (-1, -1, -1), -1, True)
            tensors.append(routing_weight)
            tb_graph.new_input(output, (-1, 1, -1), -1, True)
        else:
            tb_graph.new_input(output, (-1, 2, -1), -1, True)
        tensors.append(output)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w2_linear_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd,
             1 if fuse_mulsumadd else 0],
        )

    def gang_moe_w13_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W13 MXFP4 linear: 8 tasks (1/XCD), MFMA-based MXFP4 dequant.
        Weight format: [E, expert_wgs, wg_bytes] (MXFP4 packed per workgroup).
        Bias format: [E, output_stride] (2D flat).
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, output_stride]
        assert output.num_dims == 3  # [batch, topk, output_size]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)
        output_size = expert_wgs * output_per_wg

        # tiles_per_expert = batch_size * expert_wgs (one tile per token per wg)
        tiles_per_expert = batch_size * expert_wgs
        # Spread all expert tiles across 8 XCDs (flat round-robin).
        # Old: each expert assigned to 1 XCD → only top-k XCDs active.
        # New: all tiles pooled and distributed → all 8 XCDs active.
        num_topk = output.dim(1)  # topk dimension from output shape
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_linear_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def gang_moe_w13_swiglu_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W13 MXFP4 with SwiGLU fused into epilogue.
        Same MFMA as W13 but applies SwiGLU(gate+bias, up+bias) in the
        epilogue and writes half-sized output.
        Input: [batch, hidden] (2D, same as W13).
        Weight: [E, expert_wgs, wg_bytes] (interleaved gate/up, MXFP4).
        Bias: [E, 2*intermediate] (interleaved gate/up bias).
        Output: [batch, topk, intermediate] (activated, half of W13 output).
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, 2*intermediate]
        assert output.num_dims == 3  # [batch, topk, intermediate]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)

        tiles_per_expert = batch_size * expert_wgs
        num_topk = output.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_swiglu_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def gang_moe_w2_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W2 MXFP4 linear: 8 tasks (1/XCD), MFMA-based MXFP4 dequant.
        Weight format: [E, expert_wgs, wg_bytes] (MXFP4 packed per workgroup).
        Bias format: [E, output_stride] (2D flat).
        """
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, output_stride]
        assert output.num_dims == 3  # [batch, topk, hidden_size]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)

        tiles_per_expert = batch_size * expert_wgs
        num_topk = input.dim(1)  # topk dimension from input shape
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w2_linear_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def gang_moe_w13_linear_mxfp8_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
        fuse_swiglu: bool = False,
    ):
        """Gang MoE W13 MXFP8 linear (gfx950): FP8 weight x FP8 activation MFMA.
        Weight format: [E, expert_wgs, wg_bytes] (MXFP8 packed per workgroup).
        Bias format: [E, output_stride] (2D flat).

        Same fuse_swiglu contract as gang_moe_w13_linear_layer: `output` is the
        half-width [batch, topk, intermediate] activation and the weight rows
        must be pairwise interleaved (row 2j = gate_j, row 2j+1 = up_j).
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, output_stride]
        assert output.num_dims == 3  # [batch, topk, output_size (/2 if fused)]
        assert self.target_cc == 95, "Gang MoE MXFP8 requires gfx950 (MI350)"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)
        output_size = expert_wgs * output_per_wg
        assert bias.dim(1) == output_size, (
            f"bias width {bias.dim(1)} does not match the packed weight's "
            f"{expert_wgs} x {output_per_wg} = {output_size} rows")
        if fuse_swiglu:
            assert output.dim(2) * 2 == output_size, (
                f"fuse_swiglu output must be half the GEMM width: "
                f"{output.dim(2)} vs {output_size}")
        else:
            assert output.dim(2) == output_size
        # Depth-4 MFMA pipeline: only the last slot carries a tail guard.
        assert input.dim(1) % 512 == 0, \
            f"MXFP8 W13 K={input.dim(1)} not divisible by 512"

        tiles_per_expert = batch_size * expert_wgs
        num_topk = output.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_linear_mxfp8_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg,
             1 if fuse_swiglu else 0],
        )

    def gang_moe_w2_linear_mxfp8_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
        routing_weight: DTensor = None,
    ):
        """Gang MoE W2 MXFP8 linear (gfx950): FP8 weight x FP8 activation MFMA.
        Weight format: [E, expert_wgs, wg_bytes] (MXFP8 packed per workgroup).
        Bias format: [E, output_stride] (2D flat).

        Same fuse contract as gang_moe_w2_linear_layer: passing `routing_weight`
        ([batch, topk] f32) folds the topk weighting and the cross-expert sum
        into the epilogue, making `output` the [batch, hidden] f32 workspace.
        """
        fuse_mulsumadd = routing_weight is not None
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, output_stride]
        if fuse_mulsumadd:
            assert output.num_dims == 2          # [batch, hidden_size] f32
            assert routing_weight.num_dims == 2  # [batch, topk] f32
            assert routing_weight.dim(1) == input.dim(1)
        else:
            assert output.num_dims == 3  # [batch, topk, hidden_size]
        assert self.target_cc == 95, "Gang MoE MXFP8 requires gfx950 (MI350)"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)
        output_size = expert_wgs * output_per_wg
        assert bias.dim(1) == output_size, (
            f"bias width {bias.dim(1)} does not match the packed weight's "
            f"{expert_wgs} x {output_per_wg} = {output_size} rows")
        assert output.dim(output.num_dims - 1) == output_size
        assert input.dim(2) % 512 == 0, \
            f"MXFP8 W2 K={input.dim(2)} not divisible by 512"

        tiles_per_expert = batch_size * expert_wgs
        num_topk = input.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tensors = [input, weight, moe_routing_indices, moe_mask, bias]
        if fuse_mulsumadd:
            tb_graph.new_input(routing_weight, (-1, -1, -1), -1, True)
            tensors.append(routing_weight)
            tb_graph.new_input(output, (-1, 1, -1), -1, True)
        else:
            tb_graph.new_input(output, (-1, 2, -1), -1, True)
        tensors.append(output)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w2_linear_mxfp8_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg,
             1 if fuse_mulsumadd else 0],
        )

    def gang_moe_fused_mxfp4_layer(
        self,
        input: DTensor,
        gate_up_weight: DTensor,
        down_weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        routing_weight: DTensor,
        swiglu_out: DTensor,
        workspace_f32: DTensor,
        barrier: DTensor,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
        num_experts_global: int = None,
    ):
        """Fused W13+SwiGLU+W2 MoE gang kernel with per-expert pipelining.
        Single gang task replaces separate W13 and W2 tasks. Phase-ordered
        tile encoding (all W13 before all W2) with in-kernel atomicAdd
        barrier per expert. Supports different OPW for W13 and W2.

        W2 epilogue does atomicAdd(workspace_f32, (result+bias)*routing_weight)
        instead of writing bf16 to mlp_out. This eliminates the standalone
        MulSumAdd task (~10.2us/layer savings).

        8 inputs: input, gate_up_weight, down_weight, routing, mask, w13_bias, w2_bias, routing_weight
        3 outputs: swiglu_out (intermediate), workspace_f32 (f32 accumulator), barrier
        """
        assert input.num_dims == 2           # [batch, hidden_size]
        assert gate_up_weight.num_dims == 3  # [E, W13_WGS, wg_bytes]
        assert down_weight.num_dims == 3     # [E, W2_WGS, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert w13_bias.num_dims == 2        # [E, 2*intermediate]
        assert w2_bias.num_dims == 2         # [E, hidden]
        assert routing_weight.num_dims == 2  # [batch, topk] f32
        assert swiglu_out.num_dims == 3      # [batch, topk, intermediate]
        assert workspace_f32.num_dims == 2   # [batch, hidden] f32
        assert barrier.num_dims == 1         # [2*E]
        assert self.target_cc in (94, 95), "Fused MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        # Expert-parallel: gate_up_weight is sliced to this rank's local experts.
        # The routing/mask/tile space stays GLOBAL, so tile/pad/dispatch math uses
        # num_experts_global; only the weight storage is local (num_local_experts).
        num_local_experts = gate_up_weight.dim(0)
        if num_experts_global is None:
            num_experts_global = num_local_experts
        num_experts = num_experts_global
        # Only shard the expert range when weights are actually sliced per rank
        # (expert-parallel). When replicated (num_local_experts == global), every
        # rank owns the full [0, num_experts) range.
        if num_local_experts < num_experts_global:
            expert_base = self.mpi_rank * num_local_experts
        else:
            expert_base = 0
        w13_wgs = gate_up_weight.dim(1)  # 2*intermediate/W13_OPW
        w2_wgs = down_weight.dim(1)      # hidden/W2_OPW

        # Combined tile count: W13 tiles + W2 tiles (phase-ordered in kernel)
        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        tiles_per_expert = w13_tiles + w2_tiles

        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        # Pad W13 tile space to next multiple of 240 (30 workers/XCD × 8 XCDs)
        # so every worker's first tile is W13, eliminating compute imbalance.
        # Must match PAD_MULTIPLE in gang_moe_fused_mxfp4_mi300.cuh.
        PAD_MULTIPLE = 240
        total_w13_real = max_activated * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        total_w2 = max_activated * w2_tiles
        total_tiles_all = total_w13_padded + total_w2
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 8 inputs
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(routing_weight, (-1, -1, -1), -1, True)
        # 3 outputs
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), -1, True)
        tb_graph.new_input(barrier, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, gate_up_weight, down_weight,
             moe_routing_indices, moe_mask, w13_bias, w2_bias,
             routing_weight,
             swiglu_out, workspace_f32, barrier], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_fused_mxfp4_mi300",
            [tiles_per_expert, w13_output_per_wg, total_tiles_per_xcd, w2_output_per_wg,
             expert_base, num_local_experts],
        )

    def gang_moe_swiglu_w2_mxfp4_layer(
        self,
        w13_output: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang fused SwiGLU+W2 MXFP4: reads interleaved gate/up from W13,
        applies SwiGLU during FP8 quantization, feeds into W2 MFMA.
        No cross-WG barrier needed (croc-style activation fusion).
        Input: [batch, topk, 2*intermediate] (interleaved gate/up from W13).
        Weight: [E, expert_wgs, wg_bytes] (W2 down weights, MXFP4 packed).
        Bias: [E, output_stride] (W2 bias).
        Output: [batch, topk, hidden_size] BF16.
        """
        assert w13_output.num_dims == 3   # [batch, topk, 2*intermediate]
        assert weight.num_dims == 3       # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2         # [E, output_stride]
        assert output.num_dims == 3       # [batch, topk, hidden_size]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)

        tiles_per_expert = batch_size * expert_wgs
        num_topk = w13_output.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(w13_output, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [w13_output, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_swiglu_w2_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def moe_mul_sum_add_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_experts_per_tok, hidden_size)
        assert weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 2, -1), -1, True)
        tb_graph.new_input(weight, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)

        if self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_mi300")
        else:
            self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100")

    def splitk_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), 1, True)
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        if self.target_cc in (94, 95):
            # MI300X: workspace output partitioned by grid_dim.x (N) and grid_dim.y (K-splits)
            tb_graph.new_input(output, (1, 0, -1), -1, True)
        else:
            tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "splitk_linear_sm100")
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "splitk_linear_mi300")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "splitk_linear_swapAB_hopper")
        else:
            assert False

    def gang_ksplit_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        k_splits: int = 8,
        block_dim: tuple = (256, 1, 1),
    ):
        """Cross-XCD K-split linear with residual (SKXCCM-style).
        Phase 1: 8 gang tasks, each XCD handles K/8 for ALL N-tiles.
        Phase 2: 8 gang tasks, each XCD finalizes its N-partition."""
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        reduction_size = weight.dim(1) if weight.num_dims == 2 else input.dim(1)
        assert reduction_size % k_splits == 0
        n_tiles = output_size // tile_n
        n_cols_per_xcd = output_size // 8

        # Phase 1: K-split GEMM — each XCD reads full input+weight, writes to workspace
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)     # full input
        tb_graph.new_input(weight, (-1, -1, -1), 1, True)    # full weight
        tb_graph.new_input(workspace, (-1, -1, -1), -1, True) # full workspace (atomic target)
        self.kn_graph.customized([input, weight, workspace], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_ksplit_gemm_mi300",
            [output_stride, tile_n, n_tiles, k_splits]
        )

        # Phase 2: Finalize — each XCD reads its workspace partition + residual
        finalize_tiles = max(1, (batch_size * n_cols_per_xcd + 511) // 512)
        tb_graph2 = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph2.new_input(workspace, (-1, -1, -1), 1, True)  # full workspace
        tb_graph2.new_input(residual, (1, -1, -1), 1, True)    # XCD's residual partition
        tb_graph2.new_input(output, (1, -1, -1), -1, True)     # XCD's output partition
        self.kn_graph.customized([workspace, residual, output], tb_graph2)
        self.kn_graph.register_task(
            tb_graph2, "gang_ksplit_finalize_mi300",
            [output_stride, n_cols_per_xcd, finalize_tiles]
        )

    def gang_splitk_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,  # [batch, hidden_size] float32
        output: DTensor,
        tile_n: int,
        output_stride: int,
        k_splits: int = 4,
        block_dim: tuple = (256, 1, 1),
        reduction_size: int = None,
    ):
        """Gang split-K linear with residual: splits K within XCD for better utilization.
        8 tasks (1 per XCD), each with n_tiles × k_splits total tiles.
        Uses XCD-local atomics for merge (cheaper than GPU-scope).

        reduction_size stops the reduction short of the input tensor's width,
        for a weight whose trailing columns are all zero (the de-padded
        absorbed o_proj). Defaults to the full width."""
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        if reduction_size is None:
            reduction_size = (weight.dim(1) if weight.num_dims == 2
                              else input.dim(1))
        assert 0 < reduction_size <= input.dim(1)
        assert reduction_size % k_splits == 0, f"K={reduction_size} not divisible by k_splits={k_splits}"
        total_tiles = n_tiles_per_xcd * k_splits
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)        # input_ptrs[0]
        tb_graph.new_input(weight, (0, -1, -1), 1, True)        # input_ptrs[1]
        tb_graph.new_input(residual, (1, -1, -1), 1, True)      # input_ptrs[2]
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)     # input_ptrs[3] float32 workspace
        tb_graph.new_input(output, (1, -1, -1), -1, True)       # output_ptrs[0]
        self.kn_graph.customized([input, weight, residual, workspace, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_splitk_linear_res_mi300",
            [output_stride, tile_n, n_tiles_per_xcd, k_splits,
             reduction_size if reduction_size != input.dim(1) else 0]
        )

    def gang_rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        chain_after: DTensor = None,
        block_dim: tuple = (128, 1, 1),
    ):
        """Gang RMSNorm: 8 tasks (1 per XCD), each computes same RMSNorm.
        Enables XCD-local event counting to avoid cross-XCD barrier.

        ``chain_after`` is a dependency edge, not data: the task graph is a
        linear chain and register_mugraph asserts that consecutive ops share a
        tensor, so an op whose only real input was produced several ops back
        has no way to be scheduled. Naming the immediate predecessor's output
        here supplies the edge. The kernel never reads it. GLM's MTP draft
        layer needs this: its hnorm reads the LM head's residual output, but
        argmax and the token embedding sit in between."""
        assert self.target_cc in (94, 95), "Gang RMSNorm only supported on MI300X"
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        io = [input, weight]
        if chain_after is not None:
            tb_graph.new_input(chain_after, (-1, -1, -1), 1, True)
            io.append(chain_after)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        io.append(output)
        self.kn_graph.customized(io, tb_graph)
        self.kn_graph.register_task(tb_graph, "gang_rmsnorm_mi300", [])

    def gang_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with HipKittens Algorithm 1 windowed traversal.
        8 tasks (1 per XCD), each broadcast to m_tiles * n_tiles_per_xcd workers.

        Args:
            m_tiles: number of M-tiles to split batch across (1=no M-split)
            wgm: window height W for Algorithm 1 (0 = full M-major, default)
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0, f"Output size {output_size} must be divisible by 8"
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0, f"Chunk {chunk_n} must be divisible by tile_n {tile_n}"
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0, f"batch {batch_size} must be divisible by m_tiles {m_tiles}"
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (rows) by bid.x → each XCD gets chunk of weight rows
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # output: partition dim 1 (columns) by bid.x → each XCD writes to its column range
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
        #          n_tiles_per_xcd, wgm]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )

    def gang_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        reduction_size: int = 0,
        gemv: bool = False,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with residual + HipKittens Algorithm 1 windowed traversal.

        `reduction_size` defaults to the full input row. Pass a smaller value
        (and a correspondingly narrow weight) when the input carries a padded
        tail the GEMM should skip -- the task chain matches producers to
        consumers by tensor guid, so the input has to stay the full tensor the
        previous op wrote even when only a prefix of it is meaningful.

        `gemv` swaps the CK MFMA tile for the narrow-tile GEMV, which lifts the
        tile_n >= 64 floor the 16x64x256 MFMA imposes. Set it when the output
        is only hidden_size wide, so tile_n can drop to 8 and the op gets 256
        tiles instead of 32 -- see gang_gemv_mi300.cuh. At batch 1 the MFMA was
        discarding 15 of its 16 rows anyway.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (rows) by bid.x
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # residual: partition dim 1 (columns) by bid.x
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        # output: partition dim 1 (columns) by bid.x
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        assert reduction_size == 0 or reduction_size == weight.dim(1), (
            "reduction_size must match the weight's reduction extent")
        if gemv:
            assert tile_n >= 4 and (tile_n & (tile_n - 1)) == 0, (
                f"gemv tile_n must be a power of two >= 4, got {tile_n}")
            assert m_per_tile == 1, (
                "the gemv reads the input at row stride reduction_size, which "
                "only matches the tensor when there is one row per tile")
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
        #          n_tiles_per_xcd, wgm, reduction_size, gemv_rows]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_res_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm, reduction_size, tile_n if gemv else 0, 0]
        )

    def gang_gemv_mxfp8_with_residual_layer(
        self,
        input: DTensor,
        mxfp8_weight: DTensor,
        residual: DTensor,
        output: DTensor,
        rows_per_wg: int,
        output_stride: int,
        reduction_size: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """gang_linear_with_residual_layer(gemv=True) with an MXFP8 weight.

        Same narrow-tile GEMV, same tile addressing; the weight is E4M3 plus
        one E8M0 per 32 K instead of bf16, so it arrives workgroup-packed as
        [n_wgs, rows_per_wg * (K + K/32)] bytes and ``rows_per_wg`` plays both
        roles the bf16 path splits between ``tile_n`` and the weight's row
        count. The packing erases the logical output width, so n_tiles comes
        from the workgroup count rather than from N.

        ``reduction_size`` is mandatory here rather than defaulting to the
        input row: a packed weight cannot report its own K, and the caller
        that wants this kernel (GLM's absorbed o_proj) is narrowing past a
        padded tail anyway.
        """
        assert input.num_dims == 2
        assert mxfp8_weight.num_dims == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc == 95, "MXFP8 dequant is gfx950-only"
        batch_size = self.max_num_batched_tokens
        assert rows_per_wg >= 4 and (rows_per_wg & (rows_per_wg - 1)) == 0, (
            f"rows_per_wg must be a power of two >= 4, got {rows_per_wg}")
        assert reduction_size > 0 and reduction_size <= input.dim(1)
        assert reduction_size % 32 == 0
        n_wgs = mxfp8_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_tiles_per_xcd = n_wgs // 8
        assert mxfp8_weight.dim(1) == rows_per_wg * (
            reduction_size + reduction_size // 32), (
            f"weight row {mxfp8_weight.dim(1)} is not {rows_per_wg} rows "
            f"packed at K={reduction_size}")
        assert n_wgs * rows_per_wg == output.dim(1), (
            f"packed weight covers {n_wgs * rows_per_wg} columns, output has "
            f"{output.dim(1)}")
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        # The registrar hands the kernel input.dim(1) as INPUT_ROW_STRIDE, so
        # a reduction narrower than the row no longer implies one row.
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (workgroups) by bid.x
        tb_graph.new_input(mxfp8_weight, (0, -1, -1), 1, True)
        # residual and output: partition dim 1 (columns) by bid.x
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, mxfp8_weight, residual, output],
                                 tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile,
        #          total_tiles_per_xcd, n_tiles_per_xcd, wgm, reduction_size,
        #          gemv_rows, mxfp8]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_res_mi300",
            [output_stride, rows_per_wg, m_tiles, m_per_tile,
             total_tiles_per_xcd, n_tiles_per_xcd, wgm, reduction_size,
             rows_per_wg, 1]
        )


    def gang_linear_bias_layer(
        self,
        input: DTensor,
        weight: DTensor,
        bias: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with fused bias_add in epilogue.
        3 inputs (activation, weight, bias), 1 output."""
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)  # bias: partition dim 1 (columns) by bid.x
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_linear_bias_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )

    def gang_rmsnorm_linear_bias_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        linear_weight: DTensor,
        bias: DTensor,
        output: DTensor,
        actual_hidden_dim: int,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        norm_span: int = 0,
        reduction_size: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + Gang Linear + Bias.

        Eliminates the dispatch barrier between rmsnorm and a downstream
        gang_linear_bias by having every gang-linear worker compute the
        RMSNorm prologue locally before its MFMA. All workers write the
        same normalized values to ``norm_output`` (idempotent), then read
        from it for their linear tile.

        Inputs: norm_input, norm_weight, norm_output (writable scratch),
                linear_weight, bias.
        Output: linear output.

        ``actual_hidden_dim`` is the unpadded hidden size used for the RMS
        denominator (e.g. 2880 for GPT-OSS, with norm_input padded to 3072).

        ``norm_span`` (default: the whole row) is how far the sum of squares
        runs. It only differs from the row width when the row holds something
        besides the normed vector and its zero padding -- GLM's q_a_layernorm
        reads the fused ``[q_a | kv_latent]`` projection and must leave the
        latent columns out of the denominator.

        ``reduction_size`` (default: the whole row) narrows *both* the norm and
        the GEMM to a leading prefix of the row. Where ``norm_span`` still
        normalises and multiplies the full width and only shortens the RMS
        denominator, this drops the tail from the GEMM entirely -- worth it
        when the tail is zero padding, since the weight columns against it are
        dead traffic. Requires ``m_per_tile == 1``: the kernel takes the
        reduction extent as the row stride too, so a narrowed value only
        addresses correctly when there is a single row.
        """
        assert norm_input.num_dims == 2
        assert linear_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        output_size = linear_weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # forloop_dim must reference an existing dim for each tensor
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)  # 1D tensor
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(linear_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, linear_weight, bias, output],
            tb_graph,
        )
        params = [output_stride, tile_n, m_tiles, m_per_tile,
                  total_tiles_per_xcd, n_tiles_per_xcd, wgm, actual_hidden_dim]
        if reduction_size and not norm_span:
            norm_span = reduction_size
        if norm_span:
            assert actual_hidden_dim <= norm_span <= norm_input.dim(1)
            params.append(norm_span)
        if reduction_size:
            assert norm_span <= reduction_size <= norm_input.dim(1)
            assert reduction_size == linear_weight.dim(1), (
                "reduction_size must match the linear weight's reduction extent")
            assert m_per_tile == 1, (
                "a narrowed reduction doubles as the row stride; needs one row")
            params.append(reduction_size)
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_bias_mi300", params
        )

    def gang_rmsnorm_linear_bias_mla_kvupd_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        linear_weight: DTensor,
        bias: DTensor,
        kv_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        kv_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        tile_n: int,
        output_stride: int,
        norm_span: int,
        reduction_size: int,
        kv_offset: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """gang_rmsnorm_linear_bias_layer with mla_kv_cache_update_layer folded in.

        Replaces the (q_a_layernorm + absorbed q_b_proj) -> MLA_KV_CACHE_UPDATE
        pair with a single task. The old update was one workgroup that all 240
        workers waited behind, for ~37 KB of traffic; here its Q half becomes
        an in-place rotation by whichever worker already owns the rope columns,
        and its latent half rides on one worker concurrently with everyone
        else's MFMA. See the kernel header for why the tiling makes that work.

        ``norm_input`` doubles as the latent source, read at ``kv_offset``:
        GLM fuses q_a_proj and kv_a_proj_with_mqa into one GEMM, so the latent
        is the tail of the row this task is already norming. ``q_workspace``
        replaces the old ``q_absorbed`` buffer, which stops existing.

        ``norm_span`` and ``reduction_size`` are required, unlike in the plain
        variant -- the fused projection is the only caller and always narrows.
        """
        assert norm_input.num_dims == 2
        assert linear_weight.num_dims == 2
        assert q_workspace.num_dims == 2
        assert kv_cache.num_dims == 4  # (num_pages, page_size, 1, qk_dim)
        assert kv_cache.dim(2) == 1, "MLA keeps a single shared latent head"
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        qk_dim = kv_cache.dim(3)
        qk_rope_head_dim = cos_pos_embed.dim(cos_pos_embed.num_dims - 1)
        kv_lora_rank = qk_dim - qk_rope_head_dim
        assert kv_norm.dim(0) == kv_lora_rank
        assert norm_input.dim(1) >= kv_offset + qk_dim
        # The rope slice of a head has to be exactly one tile, or the in-place
        # rotation would need a cross-workgroup exchange.
        assert tile_n == qk_rope_head_dim and qk_dim % tile_n == 0, (
            "tile_n must equal qk_rope_head_dim and divide the head width")

        batch_size = self.max_num_batched_tokens
        output_size = linear_weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        # Each XCD's column chunk must hold whole heads, so that "is this a
        # rope tile" is a question about n_tile alone.
        assert chunk_n % qk_dim == 0, "per-XCD chunk must hold whole heads"
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        assert m_per_tile == 1, (
            "a narrowed reduction doubles as the row stride; needs one row")
        # +1: tile 0 is the latent cache update, which owns a dispatch slot so
        # that it runs beside the GEMM instead of behind one worker's share of
        # it. Only XCD 0's is live; the other seven return immediately.
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles + 1
        assert actual_hidden_dim <= norm_span <= reduction_size
        assert reduction_size <= norm_input.dim(1)
        assert reduction_size == linear_weight.dim(1), (
            "reduction_size must match the linear weight's reduction extent")

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(linear_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(kv_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_workspace, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, linear_weight, bias,
             kv_norm, cos_pos_embed, sin_pos_embed, kv_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_bias_mla_kvupd_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm, actual_hidden_dim, norm_span,
             reduction_size, kv_lora_rank, qk_rope_head_dim, kv_offset,
             self.max_seq_length, self.page_size]
        )

    def gang_rmsnorm_linear_bias_topk_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        linear_weight: DTensor,
        bias: DTensor,
        logits_scratch: DTensor,
        gang_counter: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        actual_hidden_dim: int,
        tile_n: int,
        output_stride: int,
        num_experts_per_tok: int = 4,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + Gang Linear + Bias + TopK Softmax.

        Eliminates the scheduler gaps between the router gang linear and the
        TopK softmax routing task. The last gang worker across all 8 XCDs
        detects completion via an atomic counter and computes TopK inline.

        Inputs (7): norm_input, norm_weight, norm_output (scratch),
                    linear_weight, bias, logits_scratch (scratch),
                    gang_counter (scratch).
        Outputs (3): topk_weight, routing_indices, active_expert_ids.
        """
        assert norm_input.num_dims == 2
        assert linear_weight.num_dims == 2
        assert logits_scratch.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        num_experts = output_stride  # router output width = num_experts
        output_size = linear_weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        total_gang_tiles = total_tiles_per_xcd * 8  # 8 XCDs on MI300X
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(linear_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(gang_counter, (-1, -1, -1), 0, True)
        # 3 outputs
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, linear_weight, bias,
             logits_scratch, gang_counter,
             topk_weight, routing_indices, active_expert_ids],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_bias_topk_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm, actual_hidden_dim, num_experts,
             num_experts_per_tok, total_gang_tiles]
        )

    def gang_rmsnorm_linear_bias_topk_sigmoid_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        linear_weight: DTensor,
        bias: DTensor,
        logits_scratch: DTensor,
        gang_counter: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        actual_hidden_dim: int,
        tile_n: int,
        output_stride: int,
        num_experts_per_tok: int = 4,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + Gang Linear + `noaux_tc` sigmoid/bias TopK.

        The GLM / DeepSeek counterpart of gang_rmsnorm_linear_bias_topk_layer.
        It collapses the three single-task ops that used to open every MoE
        layer -- rmsnorm, the router GEMV, and the routing task -- into one
        gang task spread over num_experts workers.

        Two things differ from the softmax variant, both because of how
        `noaux_tc` uses its bias. ``bias`` is ``e_score_correction_bias``: it
        steers selection via sigmoid(logit) + bias but never reaches the
        emitted weight, so it stays out of the GEMV and is declared
        unpartitioned -- the TopK tail reads all num_experts entries.
        ``routing_indices`` / ``active_expert_ids`` are sized for the *total*
        expert count, one row past num_experts when a shared expert rides along
        as an extra routing slot; the kernel infers it from that gap.

        Inputs (7): norm_input, norm_weight, norm_output (scratch),
                    linear_weight, bias, logits_scratch (scratch),
                    gang_counter (scratch).
        Outputs (3): topk_weight, routing_indices, active_expert_ids.
        """
        assert norm_input.num_dims == 2
        assert linear_weight.num_dims == 2
        assert logits_scratch.num_dims == 2
        assert bias.num_dims == 1  # (num_experts,)
        assert routing_indices.num_dims == 2  # (num_total_experts, batch)
        assert active_expert_ids.num_dims == 1  # (num_total_experts + 1,)
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        num_experts = output_stride  # router output width = num_experts
        assert bias.dim(0) == num_experts
        num_shared_experts = routing_indices.dim(0) - num_experts
        assert num_shared_experts in (0, 1)
        assert topk_weight.dim(1) == num_experts_per_tok + num_shared_experts
        output_size = linear_weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        total_gang_tiles = total_tiles_per_xcd * 8  # 8 XCDs on MI300X
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(linear_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (-1, -1, -1), 0, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(gang_counter, (-1, -1, -1), 0, True)
        # 3 outputs
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, linear_weight, bias,
             logits_scratch, gang_counter,
             topk_weight, routing_indices, active_expert_ids],
            tb_graph,
        )
        # routed_scaling_factor travels as an int (register_task takes ints
        # only); 1/1000 units is exact for the values these configs use.
        scaling_milli = int(round(routed_scaling_factor * 1000.0))
        assert abs(scaling_milli / 1000.0 - routed_scaling_factor) < 1e-9, (
            f"routed_scaling_factor {routed_scaling_factor} not representable "
            "in 1/1000 units"
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_bias_topk_sigmoid_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm, actual_hidden_dim, num_experts,
             num_experts_per_tok, total_gang_tiles,
             scaling_milli, 1 if norm_topk_prob else 0]
        )

    def gang_rmsnorm_linear_mxfp4_bias_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + MXFP4 Gang Linear + Bias.

        Same as gang_rmsnorm_linear_bias_layer but uses MXFP4 weights with
        hardware FP4xFP8 MFMA instead of BF16 CK GEMM. Weight is packed in
        workgroup layout: [n_wgs, wg_bytes].

        ``actual_hidden_dim`` is the unpadded hidden size used for the RMS
        denominator (e.g. 2880 for GPT-OSS, with norm_input padded to 3072).
        """
        assert norm_input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, mxfp4_weight, bias, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp4_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim]
        )

    def gang_rmsnorm_linear_mxfp8_bias_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        mxfp8_weight: DTensor,
        bias: DTensor,
        output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        norm_output: DTensor = None,
        resadd_workspace_f32: DTensor = None,
        resadd_x_out: DTensor = None,
        # Expert parallelism. > 1 re-reads ``norm_input`` as a symmetric
        # gather buffer of this many [batch, K] bf16 planes and makes the
        # residual fold the cross-rank sum. Used by the LM head, which under
        # EP is the consumer of the tail layer's exchange.
        ep_peer_slots: int = 0,
        block_dim: tuple = (256, 1, 1),
        # Perplexity only. Makes the kernel write ``output`` row step+1 rather
        # than row 0, so a prefill-only pass keeps one logit row per scored
        # position. ``output`` must then be [max_seq_length, output_stride].
        # Costs one extra HBM row per step, so decode leaves it False and the
        # emitted argument list is unchanged.
        ppl_sink: bool = False,
    ):
        """Fused RMSNorm + MXFP8 Gang Linear + Bias.

        Same as gang_rmsnorm_linear_mxfp4_bias_layer but with 8-bit weights:
        E4M3 values plus one E8M0 exponent per 32 contiguous K elements, so
        1.03 bytes per weight against bf16's 2. Weight is packed in workgroup
        layout: [n_wgs, wg_bytes], wg_bytes = OPW*K + OPW*(K/32).

        ``actual_hidden_dim`` is the unpadded hidden size used for the RMS
        denominator.

        ``norm_output`` is bf16 scratch of the same shape as ``norm_input``;
        every worker writes the same normalized row to it (idempotent) and
        then reads its own tile back. It is a required buffer, not an output,
        and defaults to ``norm_input`` only where the caller is content to
        normalize in place -- which the GLM demo is not, so it always passes
        one.

        The two ``resadd_*`` arguments are all-or-nothing and turn on the
        residual fold. Under it ``norm_input`` is reinterpreted: it is the
        unresolved residual stream rather than a finished row, and the prologue
        resolves ``resadd_workspace_f32 + norm_input`` -- an f32 MoE accumulator
        plus that residual -- reduces the result in the same pass, and publishes
        the bf16 row to ``resadd_x_out`` for this layer's o_proj to add back.
        That replaces a standalone moe_residual_add_f32 task per layer, which is
        what gpt-oss's gang_resaddf32_rmsnorm_linear_mxfp4_bias does.

        Two obligations come with it. The caller must hand the *first* folding
        layer a workspace that is already zero, and something must re-zero the
        workspace between the fold and the next accumulate -- the o_proj task
        does this up front, behind its own W13->W2 barrier. After the last
        folding layer the caller still needs a standalone moe_residual_add_f32
        to resolve that layer's MoE and re-zero for the next token.
        """
        ep = ep_peer_slots > 1
        assert not ep or norm_input.num_dims == 3, \
            "under EP norm_input is the gather buffer (slots, batch, K)"
        assert ep or norm_input.num_dims == 2
        assert mxfp8_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc == 95, "MXFP8 MFMA is gfx950-only"
        assert norm_output is not None, "MXFP8 rmsnorm+linear needs scratch"
        resadd = [resadd_workspace_f32, resadd_x_out]
        fuse_resadd = any(t is not None for t in resadd)
        assert not fuse_resadd or all(t is not None for t in resadd), (
            "the residual fold needs both resadd_workspace_f32 and resadd_x_out")
        assert not ep or fuse_resadd, \
            "the EP sum rides inside the residual fold"
        assert not ep or ep_peer_slots == self.world_size, \
            f"ep_peer_slots {ep_peer_slots} != world_size {self.world_size}"
        assert not ep or norm_input.dim(0) == ep_peer_slots
        batch_size = self.max_num_batched_tokens
        # K must clear the depth-4 pipeline's tail: only slot 3 is guarded.
        K = norm_input.dim(2) if ep else norm_input.dim(1)
        assert K % 512 == 0, f"reduction {K} must be a multiple of 512"
        n_wgs = mxfp8_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        # The packed weight erases the logical N, so cross-check it against
        # the bias, which is the only tensor that still carries it.
        assert bias.dim(1) == n_wgs * output_per_wg, (
            f"bias width {bias.dim(1)} != n_wgs {n_wgs} x opw {output_per_wg}")
        if fuse_resadd:
            assert actual_hidden_dim == K, (
                "the residual fold reads workspace and residual rows of exactly "
                f"the reduction width: {actual_hidden_dim} != {K}")
            for name, t in zip(("resadd_workspace_f32", "resadd_x_out"),
                               resadd):
                assert t.num_dims == 2 and t.dim(0) == batch_size and \
                    t.dim(1) == K, f"{name} must be ({batch_size}, {K})"
            assert (batch_size * K) % 32 == 0, (
                "the fold walks the row as 256 threads x float4")
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp8_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        io = [norm_input, norm_weight, norm_output, mxfp8_weight, bias]
        if fuse_resadd:
            tb_graph.new_input(resadd_workspace_f32, (-1, -1, -1), 1, True)
            io.append(resadd_workspace_f32)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        io.append(output)
        if fuse_resadd:
            tb_graph.new_input(resadd_x_out, (-1, -1, -1), -1, True)
            io.append(resadd_x_out)
        self.kn_graph.customized(io, tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp8_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim, ep_peer_slots,
             int(ppl_sink)]
        )

    def gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp8_weight: DTensor,
        bias: DTensor,
        kv_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        kv_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        reduction_size: int,
        kv_offset: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """gang_rmsnorm_linear_bias_mla_kvupd_layer with an MXFP8 weight.

        Same fusion, same tile-0-owns-the-latent-row dispatch; the GEMM half
        is the MXFP8 kernel instead of the bf16 one, so the weight arrives
        workgroup-packed as [n_wgs, wg_bytes] and ``output_per_wg`` replaces
        ``tile_n``. There is no separate ``norm_span``: the MXFP8 kernel norms
        exactly the ``reduction_size`` columns it multiplies, which is what
        the bf16 caller asked for anyway.
        """
        assert norm_input.num_dims == 2
        assert mxfp8_weight.num_dims == 2
        assert q_workspace.num_dims == 2
        assert kv_cache.num_dims == 4  # (num_pages, page_size, 1, qk_dim)
        assert kv_cache.dim(2) == 1, "MLA keeps a single shared latent head"
        assert self.target_cc == 95, "MXFP8 MFMA is gfx950-only"

        qk_dim = kv_cache.dim(3)
        qk_rope_head_dim = cos_pos_embed.dim(cos_pos_embed.num_dims - 1)
        kv_lora_rank = qk_dim - qk_rope_head_dim
        assert kv_norm.dim(0) == kv_lora_rank
        assert norm_input.dim(1) >= kv_offset + qk_dim
        # The rope slice of a head has to be exactly one workgroup, or the
        # in-place rotation would straddle two of them.
        assert output_per_wg == qk_rope_head_dim and qk_dim % output_per_wg == 0, (
            "output_per_wg must equal qk_rope_head_dim and divide the head")

        batch_size = self.max_num_batched_tokens
        # The GEMM takes norm_input's real row width (kv_input_stride) as its
        # input row stride now, so the narrowed reduction no longer stands in
        # for one and batch_size is free.
        # K must clear the depth-4 pipeline's tail: only slot 3 is guarded.
        assert reduction_size % 512 == 0, reduction_size
        assert actual_hidden_dim <= reduction_size <= norm_input.dim(1)
        n_wgs = mxfp8_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        # The packed weight erases both N and K, so recover K from the row
        # width and cross-check N against the bias -- the only tensor that
        # still carries it.
        assert mxfp8_weight.dim(1) == output_per_wg * (
            reduction_size + reduction_size // 32), (
            f"wg_bytes {mxfp8_weight.dim(1)} does not match opw "
            f"{output_per_wg} x K {reduction_size}")
        assert bias.dim(1) == n_wgs * output_per_wg, (
            f"bias width {bias.dim(1)} != n_wgs {n_wgs} x opw {output_per_wg}")
        # Each XCD's column chunk must hold whole heads, so that "does this
        # worker own a rope slice" is a question about wg_idx alone.
        assert (n_wgs_per_xcd * output_per_wg) % qk_dim == 0, (
            "per-XCD chunk must hold whole heads")
        # +1: tile 0 is the latent cache update, which owns a dispatch slot so
        # that it runs beside the GEMM instead of behind one worker's share of
        # it. Only XCD 0's is live; the other seven return immediately.
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd + 1

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp8_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(kv_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_workspace, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, mxfp8_weight, bias,
             kv_norm, cos_pos_embed, sin_pos_embed, kv_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim, reduction_size,
             kv_lora_rank, qk_rope_head_dim, kv_offset,
             self.max_seq_length, self.page_size]
        )

    def gang_rmsnorm_linear_mxfp4_bias_argmax_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        argmax_part_value: DTensor,
        argmax_part_index: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
        ppl_logits: DTensor = None,
    ):
        """Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax (norm-once).

        CROC-style: each worker enters once, does RMSNorm+FP8 quant once,
        then loops internally over all its assigned WGs. Argmax accumulated
        in registers across ALL tiles. No logits written to HBM.

        total_tiles_per_xcd = workers_per_xcd (each worker enters once).
        Output: one (bf16 max, int64 abs_idx) per worker.
        Follow with argmax_reduce_layer(CHUNK_SIZE=0) for final token.

        ppl_logits: optional [max_seq_length, output_stride] float32 sink.
        When supplied the kernel ALSO writes the full logit row for the
        position being scored, which perplexity needs and argmax discards.
        Costs a 393KB HBM write per step, so leave it None for serving.
        """
        assert norm_input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        workers_per_xcd = self.num_workers // 8
        total_tiles_per_xcd = workers_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(argmax_part_value, (1, -1, -1), -1, True)
        tb_graph.new_input(argmax_part_index, (1, -1, -1), -1, True)
        tensors = [norm_input, norm_weight, norm_output, mxfp4_weight, bias,
                   argmax_part_value, argmax_part_index]
        if ppl_logits is not None:
            # input_map (-1,-1,-1), NOT (1,-1,-1) like the argmax partials.
            # A non-negative map partitions that dim across grid_dim.x, and the
            # runtime pre-shifts each block's base pointer by
            # bid.x * vocab/8. The kernel indexes logits by the *absolute*
            # vocab column (the same abs_idx it uses for argmax), so a
            # pre-shifted base would double-count the partition offset --
            # blocks 0..3 would write every other 25152-column stripe and
            # blocks 4..7 would run off the end of the row into the next one.
            tb_graph.new_input(ppl_logits, (-1, -1, -1), -1, True)
            tensors.append(ppl_logits)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp4_bias_argmax_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             workers_per_xcd, actual_hidden_dim]
        )
        # Absolute index — CHUNK_SIZE=0 in argmax_reduce skips chunk math
        self.argmax_partial_output_size = 0

    def gang_mulsumradd_rmsnorm_linear_mxfp4_bias_layer(
        self,
        mlp_out: DTensor,
        routing_weight: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        qkv_output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        num_topk: int = 4,
        input_stride: int = -1,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + Bias.

        Merges the MulSumAdd from the previous layer's MoE block into the
        QKV kernel's prologue, eliminating one task dispatch.

        7 inputs (mlp_out, routing_weight, residual, norm_weight, norm_scratch,
        mxfp4_weight, bias) + 2 outputs (x_output, qkv_output).
        """
        assert mlp_out.num_dims == 3   # (batch, topk, hidden)
        assert routing_weight.num_dims == 2  # (batch, topk)
        assert residual.num_dims == 2  # (batch, hidden)
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        if input_stride < 0:
            input_stride = residual.dim(1)

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(mlp_out, (-1, -1, -1), 1, True)
        tb_graph.new_input(routing_weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 2 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(qkv_output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, qkv_output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_mulsumradd_rmsnorm_linear_mxfp4_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim, num_topk, input_stride]
        )

    def gang_rmsnorm_linear_mxfp4_bias_kvupd_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + MXFP4 Gang Linear + KV Cache Update (layer 0).

        Combines QKV MFMA with KV cache update: the epilogue applies RoPE and
        writes Q to q_workspace, K/V to paged caches directly.
        5 inputs + 3 outputs.
        """
        assert norm_input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 5 inputs
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 3 outputs (un-partitioned caches + workspace)
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, mxfp4_weight, bias,
             k_cache, v_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp4_bias_kvupd_mi300",
            [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride]
        )

    def gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_layer(
        self,
        mlp_out: DTensor,
        routing_weight: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        num_topk: int,
        input_stride: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).

        7 inputs + 4 outputs.
        """
        assert mlp_out.num_dims == 3
        assert routing_weight.num_dims == 2
        assert residual.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        if input_stride < 0:
            input_stride = residual.dim(1)

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(mlp_out, (-1, -1, -1), 1, True)
        tb_graph.new_input(routing_weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 4 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, k_cache, v_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_mi300",
            [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
             actual_hidden_dim, num_topk, input_stride,
             head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride]
        )

    def gang_resaddf32_rmsnorm_linear_mxfp4_bias_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        qkv_output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + Bias.

        Reads from f32 workspace (pre-accumulated by W2 atomicAdd) instead of
        doing MulSumAdd from 4 expert bf16 slots. Zeros workspace after read.

        6 inputs (workspace_f32, residual, norm_weight, norm_scratch,
        mxfp4_weight, bias) + 2 outputs (x_output, qkv_output).
        """
        assert workspace_f32.num_dims == 2   # (batch, hidden) f32
        assert residual.num_dims == 2        # (batch, hidden) bf16
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 6 inputs
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 2 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(qkv_output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, qkv_output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_resaddf32_rmsnorm_linear_mxfp4_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim]
        )

    def gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).

        6 inputs + 4 outputs.
        """
        assert workspace_f32.num_dims == 2
        assert residual.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 6 inputs
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 4 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, k_cache, v_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_mi300",
            [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride]
        )

    def gang_qkv_attn_fused_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        sinks: DTensor,
        barrier: DTensor,
        lse_acc: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        o_acc: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        num_kv_chunks: int,
        num_kv_heads: int,
        sliding_window: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused QKV + Attention gang task (layers 1+).

        Phase 1: ResAddF32+RMSNorm+QKV+KVUpdate (all workers, gang tiles)
        Phase 2: CK FMHA attention (1 worker per XCD, after hierarchical barrier)

        9 inputs + 5 outputs.
        """
        assert workspace_f32.num_dims == 2
        assert residual.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_qkv_tiles_per_xcd = batch_size * n_wgs_per_xcd

        has_sinks = 1 if sinks is not None else 0
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 9 inputs: slot[6] is sinks (or barrier as placeholder when no sinks)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)   # [0]
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)        # [1]
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)     # [2]
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)    # [3]
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)     # [4]
        tb_graph.new_input(bias, (1, -1, -1), 1, True)             # [5]
        sinks_or_placeholder = sinks if sinks is not None else barrier
        tb_graph.new_input(sinks_or_placeholder, (-1, -1, -1), -1, True)  # [6]
        tb_graph.new_input(barrier, (-1, -1, -1), -1, True)        # [7]
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)        # [8]
        # 5 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)       # [0]
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)        # [1]
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)        # [2]
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)    # [3]
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)          # [4]
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, sinks_or_placeholder, barrier, lse_acc,
             x_output, k_cache, v_cache, q_workspace, o_acc],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_qkv_attn_fused_mi300",
            [output_per_wg, n_wgs_per_xcd, total_qkv_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride,
             self.max_seq_length, num_kv_chunks, q_workspace_stride,
             kv_cache_stride, num_kv_heads, sliding_window, has_sinks]
        )

    def moe_residual_add_f32_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """MoE residual add from f32 workspace (last layer).

        output[b,h] = bf16(workspace_f32[b,h] + residual_bf16[b,h])
        workspace_f32[b,h] = 0 (zero for next iteration)
        """
        assert workspace_f32.num_dims == 2  # (batch, hidden) f32
        assert residual.num_dims == 2       # (batch, hidden) bf16
        assert output.num_dims == 2         # (batch, hidden) bf16
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(workspace_f32, (0, 1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [workspace_f32, residual, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "moe_residual_add_f32_mi300",
            [residual.dim(1)]  # output_stride = hidden_size
        )

    def gang_linear_mxfp4_res_bias_layer(
        self,
        input: DTensor,
        mxfp4_weight: DTensor,
        residual: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """MXFP4 Gang Linear with Residual + Bias.

        Replaces gang_splitk_linear_res_bias_layer when weights are MXFP4.
        No split-K needed since MXFP4 tiles are fast enough.

        4 inputs (input, mxfp4_weight, residual, bias), 1 output.
        """
        assert input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, mxfp4_weight, residual, bias, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_linear_mxfp4_res_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd]
        )

    def gang_linear_mxfp4_res_bias_rmsnorm_topk_layer(
        self,
        # O-PROJ inputs
        input: DTensor,
        mxfp4_weight: DTensor,
        residual: DTensor,
        oproj_bias: DTensor,
        # TopK inputs
        norm_weight: DTensor,
        norm_output: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        counters: DTensor,
        # Outputs
        output: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        # Parameters
        output_per_wg: int,
        output_stride: int,
        actual_hidden_dim: int,
        num_experts: int,
        topk_k: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused O-PROJ + RMSNorm + Router Linear + TopK Softmax.

        Combines gang_linear_mxfp4_res_bias and gang_rmsnorm_linear_bias_topk
        into a single gang task, eliminating one event barrier per layer.

        10 inputs, 4 outputs.
        """
        assert input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens

        # O-PROJ tiling
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        oproj_tiles_per_xcd = batch_size * n_wgs_per_xcd

        # TopK tiling (one expert per worker)
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8  # chunk_N per XCD
        topk_tiles_per_xcd = router_tile_n  # 1 tile per expert
        total_topk_tiles = topk_tiles_per_xcd * 8

        # Gang dispatch uses max of both tile counts
        total_tiles_per_xcd = max(oproj_tiles_per_xcd, topk_tiles_per_xcd)
        # All dispatched workers enter the oproj barrier, not just oproj workers
        total_oproj_tiles = total_tiles_per_xcd * 8

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 10 inputs
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(counters, (-1, -1, -1), 0, True)
        # 4 outputs
        # output is replicated so RMSNorm can read full hidden dim;
        # O-PROJ epilogue uses xcd_output_col_offset for correct writes
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, mxfp4_weight, residual, oproj_bias,
             norm_weight, norm_output, router_weight, router_bias,
             logits_scratch, counters,
             output, topk_weight, routing_indices, active_expert_ids],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
             actual_hidden_dim, num_experts, topk_k, router_tile_n,
             total_topk_tiles, total_tiles_per_xcd]
        )

    def gang_oproj_router_fused_layer(
        self,
        # o_proj
        input: DTensor,
        oproj_mxfp8_weight: DTensor,
        residual: DTensor,
        # router
        norm_weight: DTensor,
        norm_output: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        router_counter: DTensor,
        oproj_counters: DTensor,
        # MoE
        moe_gate_up_weight: DTensor,
        moe_down_weight: DTensor,
        moe_w13_bias: DTensor,
        moe_w2_bias: DTensor,
        moe_swiglu_out: DTensor,
        # outputs
        hidden: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        moe_workspace_f32: DTensor,
        # parameters
        rows_per_wg: int,
        reduction_size: int,
        actual_hidden_dim: int,
        num_experts_per_tok: int = 4,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        moe_w13_output_per_wg: int = 64,
        moe_w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """The whole tail of a GLM MoE layer in one gang dispatch.

        o_proj + post-attention RMSNorm + sigmoid/bias router + TopK + MoE
        W13/SwiGLU + MoE W2/MulSumAdd, with in-kernel barriers where the task
        graph used to put five events. Every sub-kernel keeps its standalone
        semantics; see the kernel header for the barrier layout and for which
        producers have to write through.

        ``hidden`` is declared unpartitioned even though the o_proj half only
        writes this XCD's columns, because the router half norms the whole row.
        The kernel reconstructs the column slice itself. ``residual`` stays
        partitioned -- nothing after Phase 1 reads it.

        Inputs (15): input, oproj_mxfp8_weight, residual, norm_weight,
                     norm_output, router_weight, router_bias, logits_scratch,
                     router_counter, oproj_counters, moe_gate_up_weight,
                     moe_down_weight, moe_w13_bias, moe_w2_bias,
                     moe_swiglu_out.
        Outputs (5): hidden, topk_weight, routing_indices, active_expert_ids,
                     moe_workspace_f32.
        """
        assert self.target_cc == 95, "MXFP8 dequant is gfx950-only"
        assert input.num_dims == 2
        assert oproj_mxfp8_weight.num_dims == 2
        assert residual.num_dims == 2
        assert hidden.num_dims == 2
        assert router_weight.num_dims == 2
        assert router_bias.num_dims == 1
        assert routing_indices.num_dims == 2
        assert active_expert_ids.num_dims == 1
        batch_size = self.max_num_batched_tokens
        assert batch_size == 1, (
            "the fused o_proj GEMV reads its input at row stride "
            "reduction_size, which only matches the tensor at one row")

        # o_proj tiling, from gang_gemv_mxfp8_with_residual_layer.
        assert rows_per_wg >= 4 and (rows_per_wg & (rows_per_wg - 1)) == 0
        assert reduction_size > 0 and reduction_size <= input.dim(1)
        assert reduction_size % 32 == 0
        n_wgs = oproj_mxfp8_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        oproj_tiles_per_xcd = n_wgs // 8
        assert oproj_mxfp8_weight.dim(1) == rows_per_wg * (
            reduction_size + reduction_size // 32)
        hidden_size = hidden.dim(1)
        assert n_wgs * rows_per_wg == hidden_size, (
            f"packed weight covers {n_wgs * rows_per_wg} columns, hidden has "
            f"{hidden_size}")

        # Router tiling, from gang_rmsnorm_linear_bias_topk_sigmoid_layer.
        num_experts = router_weight.dim(0)
        assert num_experts % 8 == 0
        router_tile_n = num_experts // 8
        total_router_tiles = router_tile_n * 8
        assert router_bias.dim(0) == num_experts
        num_shared_experts = routing_indices.dim(0) - num_experts
        assert num_shared_experts in (0, 1)
        assert topk_weight.dim(1) == num_experts_per_tok + num_shared_experts

        # MoE tiling, from gang_moe_w{13,2}_linear_mxfp8_layer.
        assert moe_gate_up_weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_down_weight.num_dims == 3
        assert moe_w13_bias.num_dims == 2
        assert moe_w2_bias.num_dims == 2
        assert moe_swiglu_out.num_dims == 3      # [batch, topk_total, inter]
        assert moe_workspace_f32.num_dims == 2
        moe_num_experts = moe_gate_up_weight.dim(0)
        assert moe_down_weight.dim(0) == moe_num_experts
        assert moe_num_experts == num_experts + num_shared_experts
        moe_w13_width = moe_gate_up_weight.dim(1) * moe_w13_output_per_wg
        assert moe_w13_bias.dim(1) == moe_w13_width
        assert moe_down_weight.dim(1) * moe_w2_output_per_wg == hidden_size
        assert moe_w2_bias.dim(1) == hidden_size
        moe_intermediate = moe_w13_width // 2
        assert moe_swiglu_out.dim(2) == moe_intermediate
        assert moe_swiglu_out.dim(1) == num_experts_per_tok + num_shared_experts
        # Depth-4 MFMA pipeline: only the last slot carries a tail guard.
        assert hidden_size % 512 == 0, \
            f"MXFP8 W13 K={hidden_size} not divisible by 512"
        assert moe_intermediate % 512 == 0, \
            f"MXFP8 W2 K={moe_intermediate} not divisible by 512"

        # MXFP8 or MXFP4 on the expert side. The kernel picks the width off
        # the packed workgroup stride -- OPW*(K + K/32) against
        # OPW*(K/2 + K/32) -- because packing erases everything else, so the
        # only thing to establish here is that the two stacks agree.
        def _moe_wg_bytes(opw, k, fp4):
            return opw * ((k // 2 if fp4 else k) + k // 32)

        moe_fp4 = moe_gate_up_weight.dim(2) == _moe_wg_bytes(
            moe_w13_output_per_wg, hidden_size, True)
        assert moe_gate_up_weight.dim(2) == _moe_wg_bytes(
            moe_w13_output_per_wg, hidden_size, moe_fp4), (
            f"W13 workgroup stride {moe_gate_up_weight.dim(2)} is neither the "
            f"MXFP8 nor the MXFP4 packing of {moe_w13_output_per_wg} rows of "
            f"K={hidden_size}")
        assert moe_down_weight.dim(2) == _moe_wg_bytes(
            moe_w2_output_per_wg, moe_intermediate, moe_fp4), (
            f"W2 workgroup stride {moe_down_weight.dim(2)} disagrees with "
            f"W13 on the element width (W13 is "
            f"{'MXFP4' if moe_fp4 else 'MXFP8'})")

        moe_topk_total = moe_swiglu_out.dim(1)
        moe_max_activated = min(moe_topk_total * batch_size, moe_num_experts)
        moe_w13_tiles_per_xcd = (
            moe_max_activated * batch_size * moe_gate_up_weight.dim(1) + 7) // 8
        moe_w2_tiles_per_xcd = (
            moe_max_activated * batch_size * moe_down_weight.dim(1) + 7) // 8

        # The o_proj barrier counts only the workers that run o_proj or the
        # router; the MoE-only workers wait on the routing-ready epoch instead.
        oproj_topk_tiles_per_xcd = max(oproj_tiles_per_xcd, router_tile_n)

        # The dispatch width, clamped to the resident workers. Every phase in
        # the task grid-strides by it, so a phase with more tiles than workers
        # takes more rounds instead of asking for a worker that does not
        # exist. The clamp is not a tuning choice: the in-kernel barriers are
        # sized in *dispatched* workers, and a tile parked behind a busy
        # worker would deadlock them. GLM-5 needs it -- 108 W2 tiles per XCD
        # against 30 workers -- and GLM-4.7-Flash never reaches it.
        tiles_per_xcd = min(max(oproj_topk_tiles_per_xcd,
                                moe_w13_tiles_per_xcd, moe_w2_tiles_per_xcd),
                            self.num_workers // 8)
        # ...and the o_proj barrier's arrival count is the participating set
        # after the clamp, not before it.
        total_barrier_arrivals = min(oproj_topk_tiles_per_xcd,
                                     tiles_per_xcd) * 8
        # 29 cache-line-strided int32 slots: the o_proj barrier's 8 release
        # flags plus its counter at [0..8], the routing-ready epoch and its 8
        # flags at [10..18], and the W13->W2 barrier at [20..28]. All monotonic,
        # so nothing is ever reset and one buffer serves every layer.
        assert oproj_counters.dim(0) >= 29 * 16

        scaling_milli = int(round(routed_scaling_factor * 1000.0))
        assert abs(scaling_milli / 1000.0 - routed_scaling_factor) < 1e-9

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(oproj_mxfp8_weight, (0, -1, -1), 1, True)
        # Unpartitioned: the kernel now derives the residual's XCD column slice
        # from the same base as the output's, so that the two cannot drift and
        # so the base can become rank-dependent under sharded o_proj, which a
        # partition map has no axis for. See gang_oproj_router_fused_mi300.cuh's
        # oproj_col_base. Was (1, -1, -1); the arithmetic is identical.
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(router_bias, (-1, -1, -1), 0, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(router_counter, (-1, -1, -1), 0, True)
        tb_graph.new_input(oproj_counters, (-1, -1, -1), 0, True)
        tb_graph.new_input(moe_gate_up_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_down_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_w13_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_w2_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_swiglu_out, (-1, 2, -1), -1, True)
        tb_graph.new_input(hidden, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, oproj_mxfp8_weight, residual, norm_weight, norm_output,
             router_weight, router_bias, logits_scratch, router_counter,
             oproj_counters, moe_gate_up_weight, moe_down_weight,
             moe_w13_bias, moe_w2_bias, moe_swiglu_out,
             hidden, topk_weight, routing_indices, active_expert_ids,
             moe_workspace_f32],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_oproj_router_fused_mi300",
            [hidden_size, rows_per_wg, oproj_tiles_per_xcd, tiles_per_xcd,
             total_barrier_arrivals, actual_hidden_dim, num_experts,
             num_experts_per_tok, router_tile_n, total_router_tiles,
             reduction_size, scaling_milli, 1 if norm_topk_prob else 0,
             batch_size, moe_intermediate, moe_w13_output_per_wg,
             moe_w2_output_per_wg, moe_w13_tiles_per_xcd,
             moe_w2_tiles_per_xcd]
        )

    def gang_oproj_topk_moe_fused_layer(
        self,
        # O-PROJ inputs
        input: DTensor,
        oproj_weight: DTensor,
        residual: DTensor,
        oproj_bias: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        counters: DTensor,
        # MoE inputs
        gate_up_weight: DTensor,
        down_weight: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        moe_barrier: DTensor,
        swiglu_out: DTensor,
        # Outputs
        oproj_output: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        routing_weight_moe: DTensor,
        workspace_f32: DTensor,
        # Parameters
        output_per_wg: int,
        output_stride: int,
        actual_hidden_dim: int,
        num_experts: int,
        topk_k: int,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused O-PROJ+TopK+MoE: combines task 213 and 187 into one gang task.
        Eliminates one inter-task event barrier per layer.

        16 inputs, 6 outputs.
        """
        assert input.num_dims == 2
        assert oproj_weight.num_dims == 2
        assert gate_up_weight.num_dims == 3
        assert down_weight.num_dims == 3
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens

        # O-PROJ tiling (same as task 213)
        n_wgs = oproj_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        oproj_tiles_per_xcd = batch_size * n_wgs_per_xcd

        # TopK tiling (same as task 213)
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8
        topk_tiles_per_xcd = router_tile_n
        total_topk_tiles = topk_tiles_per_xcd * 8
        total_oproj_tiles = max(oproj_tiles_per_xcd, topk_tiles_per_xcd) * 8

        # MoE dimensions from tensors
        # hidden_size for MoE = norm_output dimension (PADDED_HIDDEN_SIZE),
        # NOT input (attn_out) dimension which is num_heads * head_dim.
        hidden_size = norm_output.dim(1)
        intermediate_size = swiglu_out.dim(2)

        # MoE tiling (same as task 187)
        moe_num_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)
        w2_wgs = down_weight.dim(1)
        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, moe_num_experts)
        PAD_MULTIPLE = 240

        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        # Slot-parallel EP builds the tile space over the experts this rank
        # OWNS, so the loop bound must shrink to match; the device-side decode
        # in gang_moe_fused_mxfp4_mi300.cuh does the same arithmetic. Sizing
        # this for the full activated list would not be wrong -- the extra
        # trips decode past total_tiles and return -- but it costs a poll per
        # worker per layer for nothing.
        _ep_ws = 1
        _ep_me = 0
        _owned = ((max_activated - _ep_me + _ep_ws - 1) // _ep_ws
                  if _ep_ws > 1 else max_activated)
        total_w13_real = _owned * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        # MPK_W2_SPLITK doubles the W2 tile space on the device (each tile
        # covers half of K), so the host loop bound has to double too --
        # otherwise the un-dispatched half never arrives and the W13->W2
        # per-expert barrier hangs.
        _w2_splitk = 2 if int(os.environ.get("MPK_W2_SPLITK", "0")) == 1 else 1
        total_w2 = _owned * w2_tiles * _w2_splitk
        total_tiles_all = total_w13_padded + total_w2
        moe_total_tiles_per_xcd = (total_tiles_all + 7) // 8

        # Match standalone MoE worker count (30 = 240 workers / 8 XCDs)
        workers_per_xcd = self.num_workers // 8  # 30

        reduction_size = input.dim(1)

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 16 inputs
        tb_graph.new_input(input, (-1, -1, -1), 1, True)            # [0] attn_out
        tb_graph.new_input(oproj_weight, (0, -1, -1), 1, True)      # [1] O-proj weight
        tb_graph.new_input(residual, (1, -1, -1), 1, True)          # [2] residual
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)        # [3] O-proj bias
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)      # [4] RMSNorm weight
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)      # [5] norm output
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)     # [6] router weight
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)       # [7] router bias
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)    # [8] logits scratch
        tb_graph.new_input(counters, (-1, -1, -1), 0, True)         # [9] hier barrier
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)    # [10] W13 weight
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)       # [11] W2 weight
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)        # [12] W13 bias
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)         # [13] W2 bias
        tb_graph.new_input(moe_barrier, (-1, -1, -1), -1, True)     # [14] MoE barrier
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)       # [15] SwiGLU scratch
        # 6 outputs
        tb_graph.new_input(oproj_output, (-1, -1, -1), -1, True)    # [0] O-proj output
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)      # [1] topk weight
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)  # [2] routing indices
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)  # [3] expert mask
        tb_graph.new_input(routing_weight_moe, (-1, -1, -1), -1, True)  # [4] routing weight (MoE)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), -1, True)   # [5] MoE accumulator

        self.kn_graph.customized(
            [input, oproj_weight, residual, oproj_bias,
             norm_weight, norm_output, router_weight, router_bias,
             logits_scratch, counters,
             gate_up_weight, down_weight, w13_bias, w2_bias,
             moe_barrier, swiglu_out,
             oproj_output, topk_weight, routing_indices,
             active_expert_ids, routing_weight_moe, workspace_f32],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_oproj_topk_moe_fused_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
             actual_hidden_dim, num_experts, topk_k, router_tile_n,
             total_topk_tiles, oproj_tiles_per_xcd,
             intermediate_size, hidden_size,
             w13_output_per_wg, w2_output_per_wg,
             moe_total_tiles_per_xcd, workers_per_xcd]
        )

    def gang_full_layer_fused_layer(
        self,
        # QKV+Attn inputs (from type 214)
        workspace_f32: DTensor,
        residual: DTensor,
        norm_weight_pre: DTensor,
        norm_scratch_pre: DTensor,
        qkv_weight: DTensor,
        qkv_bias: DTensor,
        sinks: DTensor,
        qkv_barrier: DTensor,
        lse_acc: DTensor,
        # O-proj+TopK inputs (from type 215)
        oproj_weight: DTensor,
        oproj_bias: DTensor,
        norm_weight_post: DTensor,
        norm_scratch_post: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        oproj_counters: DTensor,
        # MoE inputs
        gate_up_weight: DTensor,
        down_weight: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        moe_barrier: DTensor,
        swiglu_out: DTensor,
        o_acc_f32: DTensor,
        # Outputs (QKV+Attn)
        x_output: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        o_acc: DTensor,
        # Outputs (O-proj+TopK+MoE)
        attn_proj_out: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        routing_weight_moe: DTensor,
        moe_workspace_f32: DTensor,
        # Parameters
        actual_hidden_dim: int,
        qkv_output_per_wg: int,
        oproj_output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        num_kv_chunks: int,
        num_kv_heads: int,
        num_experts: int,
        topk_k: int,
        sliding_window: int = 0,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        expert_base: int = 0,
        num_local_experts: int = None,
        # Inline expert-parallel combine (Phase 9)
        ep_gather: DTensor = None,
        ep_signal: DTensor = None,
        ep_combined: DTensor = None,
        ep_fold_rank: int = 0,
        # The EP reduce, dissolved into this layer's QKV prologue. Pass the
        # PREVIOUS layer's ep_gather and the prologue sums its per-rank slots
        # in the pass it already makes over that vector, instead of reading a
        # combined residual that a separate reduce + exit barrier had to
        # produce. None on layer 0 (its residual is the embedding) and on every
        # non-EP config.
        ep_prev_gather: DTensor = None,
        # True only on the layer whose consumer is a separate task -- the last
        # one, read by the tail. That layer still materializes the sum into
        # ep_combined and still pays the exit barrier; the other 35 do not.
        ep_write_combined: bool = False,
        # Slot-parallel expert split. When ep_slot_ws > 1 the rank owns the
        # activated-list slots congruent to ep_slot_me, and gate_up/down must
        # be the FULL replicated weights (expert_base=0,
        # num_local_experts=num_experts).
        ep_slot_ws: int = 1,
        ep_slot_me: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Full-layer fused gang task: QKV+Attn+O-proj+TopK+MoE.
        Combines task 214 and 215 into one gang task per layer.
        24 inputs, 11 outputs (26/12 with the inline EP combine).

        Under expert parallelism the caller passes sliced gate_up/down weights
        and biases plus the ownership window [expert_base, expert_base +
        num_local_experts). Routing, mask and barrier stay replicated global
        and keyed by global expert id; only weight storage is local. Defaults
        reproduce the single-GPU identity mapping.

        Passing ep_gather/ep_signal/ep_combined additionally fuses the MoE
        cross-rank combine into the task, replacing the three dispatched tasks
        (moe_residual_add_f32 -> identity -> allreduce) that used to follow it.
        ep_gather and ep_signal must be symmetric-heap tensors
        (io_category="nvshmem_tensor"); ep_combined carries the layer output
        and is read by the next layer as its residual. Omitting them keeps the
        single-GPU / replicated-MoE behaviour exactly as before.
        """
        assert residual.num_dims == 2
        assert qkv_weight.num_dims == 2
        assert oproj_weight.num_dims == 2
        assert gate_up_weight.num_dims == 3
        assert down_weight.num_dims == 3
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        ep_inline = ep_gather is not None
        if ep_inline:
            assert ep_signal is not None and ep_combined is not None, \
                "inline EP combine needs ep_gather, ep_signal and ep_combined"
            assert self.world_size > 1, \
                "inline EP combine requires world_size > 1"
            assert ep_gather.num_dims == 3  # (world_size, batch, hidden)
            assert ep_gather.dim(0) == self.world_size
            assert ep_combined.num_dims == 2
            if ep_prev_gather is not None:
                assert ep_prev_gather.num_dims == 3
                assert ep_prev_gather.dim(0) == self.world_size
            # One 64-byte line per PE so a peer's SIGNAL_ADD never shares a
            # line with another's (FULL_LAYER_EP_SIGNAL_STRIDE in the kernel).
            # 64 bytes (one cache line) of signal space per PE. Declared in
            # int32 units because mi.uint64 has no get_datatype_size() entry;
            # the kernel reinterprets the pointer as uint64*.
            assert ep_signal.dim(0) >= self.world_size * 16
        else:
            assert ep_prev_gather is None, \
                "ep_prev_gather only means anything with the inline EP combine"

        batch_size = self.max_num_batched_tokens

        # QKV tiling (from type 214)
        qkv_n_wgs = qkv_weight.dim(0)
        assert qkv_n_wgs % 8 == 0
        qkv_n_wgs_per_xcd = qkv_n_wgs // 8
        total_qkv_tiles_per_xcd = batch_size * qkv_n_wgs_per_xcd

        has_sinks = 1 if sinks is not None else 0
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        # O-PROJ tiling (from type 215)
        oproj_n_wgs = oproj_weight.dim(0)
        assert oproj_n_wgs % 8 == 0
        oproj_n_wgs_per_xcd = oproj_n_wgs // 8
        oproj_tiles_per_xcd = batch_size * oproj_n_wgs_per_xcd
        oproj_output_stride = norm_scratch_post.dim(1)

        # TopK tiling
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8
        total_topk_tiles = router_tile_n * 8
        total_oproj_tiles = max(oproj_tiles_per_xcd, router_tile_n) * 8

        # MoE tiling (from type 187/215)
        # gate_up_weight.dim(0) is the LOCAL expert count under expert
        # parallelism, but the tile space must cover every GLOBALLY activated
        # expert: tiles are decoded against the replicated global mask and
        # non-owned experts early-return inside the kernel. Sizing the tile
        # space with the local count would drop the tiles of experts owned by
        # higher ranks.
        if num_local_experts is None:
            num_local_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)
        w2_wgs = down_weight.dim(1)
        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        PAD_MULTIPLE = 240

        intermediate_size = swiglu_out.dim(2)

        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        # Slot-parallel EP builds the tile space over the experts this rank
        # OWNS, so the loop bound must shrink to match; the device-side decode
        # in gang_moe_fused_mxfp4_mi300.cuh does the same arithmetic. Sizing
        # this for the full activated list would not be wrong -- the extra
        # trips decode past total_tiles and return -- but it costs a poll per
        # worker per layer for nothing.
        _ep_ws = ep_slot_ws
        _ep_me = ep_slot_me
        _owned = ((max_activated - _ep_me + _ep_ws - 1) // _ep_ws
                  if _ep_ws > 1 else max_activated)
        total_w13_real = _owned * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        # MPK_W2_SPLITK doubles the W2 tile space on the device (each tile
        # covers half of K), so the host loop bound has to double too --
        # otherwise the un-dispatched half never arrives and the W13->W2
        # per-expert barrier hangs.
        _w2_splitk = 2 if int(os.environ.get("MPK_W2_SPLITK", "0")) == 1 else 1
        total_w2 = _owned * w2_tiles * _w2_splitk
        total_tiles_all = total_w13_padded + total_w2
        moe_total_tiles_per_xcd = (total_tiles_all + 7) // 8

        workers_per_xcd = self.num_workers // 8  # 30

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 24 inputs
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)         # [0]
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)              # [1]
        tb_graph.new_input(norm_weight_pre, (-1, -1, -1), 0, True)       # [2]
        tb_graph.new_input(norm_scratch_pre, (-1, -1, -1), 1, True)      # [3]
        tb_graph.new_input(qkv_weight, (0, -1, -1), 1, True)            # [4]
        tb_graph.new_input(qkv_bias, (1, -1, -1), 1, True)              # [5]
        sinks_or_placeholder = sinks if sinks is not None else qkv_barrier
        tb_graph.new_input(sinks_or_placeholder, (-1, -1, -1), -1, True) # [6]
        tb_graph.new_input(qkv_barrier, (-1, -1, -1), -1, True)         # [7]
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)             # [8]
        tb_graph.new_input(oproj_weight, (0, -1, -1), 1, True)          # [9]
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)            # [10]
        tb_graph.new_input(norm_weight_post, (-1, -1, -1), 0, True)     # [11]
        tb_graph.new_input(norm_scratch_post, (-1, -1, -1), 1, True)    # [12]
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)         # [13]
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)           # [14]
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)        # [15]
        tb_graph.new_input(oproj_counters, (-1, -1, -1), 0, True)       # [16]
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)        # [17]
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)           # [18]
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)            # [19]
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)             # [20]
        tb_graph.new_input(moe_barrier, (-1, -1, -1), -1, True)         # [21]
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)           # [22]
        tb_graph.new_input(o_acc_f32, (-1, -1, -1), -1, True)           # [23]
        # Inline-EP inputs. The runtime splits this flat operator list at the
        # registered num_inputs, so these must sit BETWEEN the inputs and the
        # outputs -- appending them at the end would reclassify x_output and
        # k_cache as inputs and shift every output index the kernel hard-codes.
        if ep_inline:
            tb_graph.new_input(ep_gather, (-1, -1, -1), -1, True)       # [24]
            tb_graph.new_input(ep_signal, (-1, -1, -1), -1, True)       # [25]
            if ep_prev_gather is not None:
                tb_graph.new_input(ep_prev_gather, (-1, -1, -1), -1, True)  # [26]
        # 11 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)            # [0]
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)             # [1]
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)             # [2]
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)         # [3]
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)               # [4]
        tb_graph.new_input(attn_proj_out, (-1, -1, -1), -1, True)       # [5]
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)          # [6]
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)     # [7]
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)   # [8]
        tb_graph.new_input(routing_weight_moe, (-1, -1, -1), -1, True)  # [9]
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), -1, True)   # [10]

        # 12th output, EP only.
        if ep_inline:
            tb_graph.new_input(ep_combined, (-1, -1, -1), -1, True)     # [11]

        # Must mirror the tb_graph operator order exactly.
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight_pre, norm_scratch_pre,
             qkv_weight, qkv_bias, sinks_or_placeholder, qkv_barrier, lse_acc,
             oproj_weight, oproj_bias, norm_weight_post, norm_scratch_post,
             router_weight, router_bias, logits_scratch, oproj_counters,
             gate_up_weight, down_weight, w13_bias, w2_bias,
             moe_barrier, swiglu_out, o_acc_f32]
            + ([ep_gather, ep_signal] if ep_inline else [])
            + ([ep_prev_gather] if ep_inline and ep_prev_gather is not None
               else [])
            + [x_output, k_cache, v_cache, q_workspace, o_acc,
               attn_proj_out, topk_weight, routing_indices,
               active_expert_ids, routing_weight_moe, moe_workspace_f32]
            + ([ep_combined] if ep_inline else []),
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_full_layer_fused_mi300",
            [qkv_output_per_wg, qkv_n_wgs_per_xcd, total_qkv_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride,
             self.max_seq_length, num_kv_chunks, q_workspace_stride,
             kv_cache_stride, num_kv_heads, sliding_window, has_sinks,
             oproj_output_per_wg, oproj_output_stride, total_oproj_tiles,
             num_experts, topk_k, router_tile_n, total_topk_tiles,
             oproj_tiles_per_xcd, moe_total_tiles_per_xcd,
             w13_output_per_wg, w2_output_per_wg,
             intermediate_size, workers_per_xcd,
             expert_base, num_local_experts,
             self.world_size if ep_inline else 1,
             self.mpi_rank if ep_inline else 0,
             ep_fold_rank,
             ep_slot_ws, ep_slot_me,
             (self.world_size
              if (ep_inline and ep_prev_gather is not None) else 1),
             1 if (ep_inline and ep_write_combined) else 0]
        )

    def gang_full_layer_with_lmhead_fused_layer(
        self,
        # QKV+Attn inputs (same as type 216)
        workspace_f32: DTensor,
        residual: DTensor,
        norm_weight_pre: DTensor,
        norm_scratch_pre: DTensor,
        qkv_weight: DTensor,
        qkv_bias: DTensor,
        sinks: DTensor,
        qkv_barrier: DTensor,
        lse_acc: DTensor,
        # O-proj+TopK inputs
        oproj_weight: DTensor,
        oproj_bias: DTensor,
        norm_weight_post: DTensor,
        norm_scratch_post: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        oproj_counters: DTensor,
        # MoE inputs
        gate_up_weight: DTensor,
        down_weight: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        moe_barrier: DTensor,
        swiglu_out: DTensor,
        o_acc_f32: DTensor,
        # LM head inputs (4 extra)
        lm_norm_weight: DTensor,
        lm_norm_scratch: DTensor,
        lm_mxfp4_weight: DTensor,
        lm_bias: DTensor,
        # Outputs (QKV+Attn)
        x_output: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        o_acc: DTensor,
        # Outputs (O-proj+TopK+MoE)
        attn_proj_out: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        routing_weight_moe: DTensor,
        moe_workspace_f32: DTensor,
        # LM head outputs (2 extra)
        lm_logits: DTensor,
        argmax_output: DTensor,
        # Parameters
        actual_hidden_dim: int,
        qkv_output_per_wg: int,
        oproj_output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        num_kv_chunks: int,
        num_kv_heads: int,
        num_experts: int,
        topk_k: int,
        lm_output_per_wg: int,
        lm_output_stride: int,
        sliding_window: int = 0,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Full-layer + LM head + argmax fused gang task (type 217).
        28 inputs, 13 outputs, 33 params.
        """
        assert residual.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens

        # QKV tiling
        qkv_n_wgs = qkv_weight.dim(0)
        assert qkv_n_wgs % 8 == 0
        qkv_n_wgs_per_xcd = qkv_n_wgs // 8
        total_qkv_tiles_per_xcd = batch_size * qkv_n_wgs_per_xcd

        has_sinks = 1 if sinks is not None else 0
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        # O-PROJ tiling
        oproj_n_wgs = oproj_weight.dim(0)
        assert oproj_n_wgs % 8 == 0
        oproj_tiles_per_xcd = batch_size * (oproj_n_wgs // 8)
        oproj_output_stride = norm_scratch_post.dim(1)

        # TopK tiling
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8
        total_topk_tiles = router_tile_n * 8
        total_oproj_tiles = max(oproj_tiles_per_xcd, router_tile_n) * 8

        # MoE tiling
        moe_num_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)
        w2_wgs = down_weight.dim(1)
        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, moe_num_experts)
        PAD_MULTIPLE = 240

        intermediate_size = swiglu_out.dim(2)

        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        # Slot-parallel EP builds the tile space over the experts this rank
        # OWNS, so the loop bound must shrink to match; the device-side decode
        # in gang_moe_fused_mxfp4_mi300.cuh does the same arithmetic. Sizing
        # this for the full activated list would not be wrong -- the extra
        # trips decode past total_tiles and return -- but it costs a poll per
        # worker per layer for nothing.
        _ep_ws = 1
        _ep_me = 0
        _owned = ((max_activated - _ep_me + _ep_ws - 1) // _ep_ws
                  if _ep_ws > 1 else max_activated)
        total_w13_real = _owned * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        # MPK_W2_SPLITK doubles the W2 tile space on the device (each tile
        # covers half of K), so the host loop bound has to double too --
        # otherwise the un-dispatched half never arrives and the W13->W2
        # per-expert barrier hangs.
        _w2_splitk = 2 if int(os.environ.get("MPK_W2_SPLITK", "0")) == 1 else 1
        total_w2 = _owned * w2_tiles * _w2_splitk
        total_tiles_all = total_w13_padded + total_w2
        moe_total_tiles_per_xcd = (total_tiles_all + 7) // 8

        workers_per_xcd = self.num_workers // 8  # 30

        # LM head tiling
        lm_n_wgs = lm_mxfp4_weight.dim(0)
        assert lm_n_wgs % 8 == 0
        lm_n_wgs_per_xcd = lm_n_wgs // 8

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 24 base inputs (same as type 216)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)         # [0]
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)              # [1]
        tb_graph.new_input(norm_weight_pre, (-1, -1, -1), 0, True)       # [2]
        tb_graph.new_input(norm_scratch_pre, (-1, -1, -1), 1, True)      # [3]
        tb_graph.new_input(qkv_weight, (0, -1, -1), 1, True)            # [4]
        tb_graph.new_input(qkv_bias, (1, -1, -1), 1, True)              # [5]
        sinks_or_placeholder = sinks if sinks is not None else qkv_barrier
        tb_graph.new_input(sinks_or_placeholder, (-1, -1, -1), -1, True) # [6]
        tb_graph.new_input(qkv_barrier, (-1, -1, -1), -1, True)         # [7]
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)             # [8]
        tb_graph.new_input(oproj_weight, (0, -1, -1), 1, True)          # [9]
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)            # [10]
        tb_graph.new_input(norm_weight_post, (-1, -1, -1), 0, True)     # [11]
        tb_graph.new_input(norm_scratch_post, (-1, -1, -1), 1, True)    # [12]
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)         # [13]
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)           # [14]
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)        # [15]
        tb_graph.new_input(oproj_counters, (-1, -1, -1), 0, True)       # [16]
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)        # [17]
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)           # [18]
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)            # [19]
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)             # [20]
        tb_graph.new_input(moe_barrier, (-1, -1, -1), -1, True)         # [21]
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)           # [22]
        tb_graph.new_input(o_acc_f32, (-1, -1, -1), -1, True)           # [23]
        # 4 extra LM head inputs
        tb_graph.new_input(lm_norm_weight, (-1, -1, -1), 0, True)       # [24]
        tb_graph.new_input(lm_norm_scratch, (-1, -1, -1), 1, True)      # [25]
        tb_graph.new_input(lm_mxfp4_weight, (0, -1, -1), 1, True)       # [26]
        tb_graph.new_input(lm_bias, (1, -1, -1), 1, True)               # [27]
        # 11 base outputs (same as type 216)
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)            # [0]
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)             # [1]
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)             # [2]
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)         # [3]
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)               # [4]
        tb_graph.new_input(attn_proj_out, (-1, -1, -1), -1, True)       # [5]
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)          # [6]
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)     # [7]
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)   # [8]
        tb_graph.new_input(routing_weight_moe, (-1, -1, -1), -1, True)  # [9]
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), -1, True)   # [10]
        # 2 extra LM head outputs
        tb_graph.new_input(lm_logits, (1, -1, -1), -1, True)            # [11]
        tb_graph.new_input(argmax_output, (-1, -1, -1), -1, True)       # [12]

        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight_pre, norm_scratch_pre,
             qkv_weight, qkv_bias, sinks_or_placeholder, qkv_barrier, lse_acc,
             oproj_weight, oproj_bias, norm_weight_post, norm_scratch_post,
             router_weight, router_bias, logits_scratch, oproj_counters,
             gate_up_weight, down_weight, w13_bias, w2_bias,
             moe_barrier, swiglu_out, o_acc_f32,
             lm_norm_weight, lm_norm_scratch, lm_mxfp4_weight, lm_bias,
             x_output, k_cache, v_cache, q_workspace, o_acc,
             attn_proj_out, topk_weight, routing_indices,
             active_expert_ids, routing_weight_moe, moe_workspace_f32,
             lm_logits, argmax_output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_full_layer_with_lmhead_fused_mi300",
            [qkv_output_per_wg, qkv_n_wgs_per_xcd, total_qkv_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride,
             self.max_seq_length, num_kv_chunks, q_workspace_stride,
             kv_cache_stride, num_kv_heads, sliding_window, has_sinks,
             oproj_output_per_wg, oproj_output_stride, total_oproj_tiles,
             num_experts, topk_k, router_tile_n, total_topk_tiles,
             oproj_tiles_per_xcd, moe_total_tiles_per_xcd,
             w13_output_per_wg, w2_output_per_wg,
             intermediate_size, workers_per_xcd,
             lm_output_per_wg, lm_n_wgs_per_xcd,
             lm_output_stride, actual_hidden_dim]
        )

    def gang_splitk_linear_res_bias_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        bias: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        k_splits: int = 4,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang split-K linear with residual + fused bias_add in epilogue.
        5 inputs (input, weight, residual, workspace, bias), 1 output."""
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        reduction_size = weight.dim(1) if weight.num_dims == 2 else input.dim(1)
        assert reduction_size % k_splits == 0
        total_tiles = n_tiles_per_xcd * k_splits
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)  # bias: partition dim 1 (columns) by bid.x
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, workspace, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_splitk_linear_res_bias_mi300",
            [output_stride, tile_n, n_tiles_per_xcd, k_splits]
        )

    def linear_silu_layer(
        self, input, weight, output, output_stride,
        tile_n=64, m_tiles=1, wgm=0, block_dim=(256, 1, 1)):
        """CU-task linear with fused SiLU+mul. Same kernel as gang version."""
        assert input.num_dims == 2 and weight.num_dims == 2 and output.num_dims == 2
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        gate_up_size = weight.dim(0)
        n_weight_tiles = gate_up_size // tile_n
        n_output_tiles = n_weight_tiles // 2
        m_per_tile = batch_size // m_tiles if m_tiles > 0 else batch_size
        total_tiles = n_output_tiles * m_tiles
        grid_dim = (total_tiles, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_silu_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles, n_output_tiles, wgm])

    def gang_linear_silu_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with fused SiLU+mul.
        Weight is interleaved gate+up from shuffle_tensors(num_groups=G).
        Output is [bs, inter_size] (half of gate_up_size).

        Weight layout per XCD chunk: [gate_0(128), up_0(128), gate_1(128), up_1(128), ...]
        With tile_n=64: each 128-row block = 2 tiles, each group = 4 tiles.
        n_tiles_per_xcd counts OUTPUT tiles (= half of weight tiles per XCD).
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear SiLU only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        gate_up_size = weight.dim(0)
        assert gate_up_size % 8 == 0
        chunk_n_gateup = gate_up_size // 8  # weight rows per XCD
        # Each group = 4 tiles (2 gate + 2 up). Output tiles = half of weight tiles.
        n_weight_tiles = chunk_n_gateup // tile_n
        n_tiles_per_xcd = n_weight_tiles // 2  # output tiles (one per gate+up pair)
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (rows) by bid.x → each XCD gets its interleaved chunk
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # output: partition dim 1 (columns) by bid.x → each XCD writes to its column range
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
        #          n_tiles_per_xcd, wgm]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_silu_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )

    def splitk_reduce_layer(
        self,
        workspace: DTensor,
        residual: DTensor,
        output: DTensor,
        k_splits: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert workspace.num_dims == 2  # (k_splits * batch_size, hidden_size)
        assert residual.num_dims == 2   # (batch_size, hidden_size)
        assert output.num_dims == 2     # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)  # full K-splits visible
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([workspace, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "splitk_reduce_mi300", [k_splits])

    def splitk_linear_res_atomic_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        done_counter: DTensor,
        output: DTensor,
        k_splits: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Single-task split-K with float32 atomicAdd.
        grid_dim = (N_blocks, K_splits, 1)
        """
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input [batch, K]: grid_dim.y partitions dim 1 (K)
        tb_graph.new_input(input, (-1, 1, -1), 1, True)
        # weight [N, K]: grid_dim.x partitions dim 0 (N), grid_dim.y partitions dim 1 (K)
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        # residual [batch, hidden]: grid_dim.x partitions dim 1 (N portion)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        # workspace [batch, hidden] float32: grid_dim.x partitions dim 1, shared across K-splits
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)
        # done_counter [n_blocks] int32: grid_dim.x partitions dim 0
        tb_graph.new_input(done_counter, (0, -1, -1), 1, True)
        # output [batch, hidden] bf16: grid_dim.x partitions dim 1
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, residual, workspace, done_counter, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "splitk_linear_res_atomic_mi300", [k_splits])

    def linear_m_parallel_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Linear with M-dimension parallelism for L2 cache sharing.
        grid_dim = (N_tiles, M_tiles, 1): partitions both output columns and input rows.
        Workers on same XCD with different M-tiles share weight columns in L2.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input[batch, K]: dim 0 (batch) partitioned by grid_dim.y (M_tiles)
        tb_graph.new_input(input, (-1, 0, -1), 1, True)
        # weight[N, K]: dim 0 (N) partitioned by grid_dim.x (N_tiles)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # output[batch, N]: dim 0 (batch) by grid_dim.y, dim 1 (N) by grid_dim.x
        tb_graph.new_input(output, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear")

    def linear_with_residual_m_parallel_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Linear + residual with M-dimension parallelism.
        grid_dim = (N_tiles, M_tiles, 1).
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 0, -1), 1, True)      # batch by grid_dim.y
        tb_graph.new_input(weight, (0, -1, -1), 1, True)      # N by grid_dim.x
        tb_graph.new_input(residual, (1, 0, -1), 1, True)     # batch by grid_dim.y, N by grid_dim.x
        tb_graph.new_input(output, (1, 0, -1), -1, True)      # batch by grid_dim.y, N by grid_dim.x
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_with_residual")

    def splitk_linear_res_atomic_m_parallel_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        done_counter: DTensor,
        output: DTensor,
        k_splits: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Split-K atomic linear + residual with M-dimension parallelism.
        grid_dim = (N_tiles, K_splits, M_tiles).
        M-tiling via grid_dim.z enables L2 weight sharing between M-tiles.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input[batch, K]: dim 0 (batch) by grid_dim.z, dim 1 (K) by grid_dim.y
        tb_graph.new_input(input, (-1, 1, 0), 1, True)
        # weight[N, K]: dim 0 (N) by grid_dim.x, dim 1 (K) by grid_dim.y
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        # residual[batch, N]: dim 0 (batch) by grid_dim.z, dim 1 (N) by grid_dim.x
        tb_graph.new_input(residual, (1, -1, 0), 1, True)
        # workspace[batch, N] fp32: dim 0 (batch) by grid_dim.z, dim 1 (N) by grid_dim.x
        tb_graph.new_input(workspace, (1, -1, 0), 1, True)
        # done_counter[N_blocks * M_tiles]: dim 0 by grid_dim.x, dim 1(?) by grid_dim.z
        # Actually done_counter is indexed by bid.x only in the kernel.
        # With M-tiles, each M-tile needs its own counter set.
        # Use (0, -1, 2) to partition dim 0 by grid_dim.x and dim 2 by grid_dim.z
        # But done_counter is 2D: (n_blocks, m_tiles) or (n_blocks * m_tiles, 1)
        # Simplest: make done_counter (n_blocks * m_tiles, 1), partition by (0, -1, -1)
        # and let the kernel compute the right index.
        # done_counter[n_blocks, m_tiles]: dim 0 by grid_dim.x, dim 1 by grid_dim.z
        tb_graph.new_input(done_counter, (0, -1, 1), 1, True)
        # output[batch, N]: dim 0 by grid_dim.z, dim 1 by grid_dim.x
        tb_graph.new_input(output, (1, -1, 0), -1, True)
        self.kn_graph.customized(
            [input, weight, residual, workspace, done_counter, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "splitk_linear_res_atomic_mi300", [k_splits])

    def linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "linear_sm100")
        elif self.target_cc == 90:
            if weight.dim(0) // grid_dim[0] <= 64:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
        elif self.target_cc == 80 or self.target_cc in (94, 95):
            # 94: MI300/ROCm – use sm_80-style "linear" (base PTX path)
            self.kn_graph.register_task(tb_graph, "linear")
        else:
            self._raise_unsupported_target_cc("linear_layer", ["linear_sm100", "linear_swapAB_hopper", "linear"])

    def linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "linear_with_residual_sm100")
        elif self.target_cc == 90:
            if weight.dim(0) // grid_dim[0] <= 64:
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_with_residual_hopper")
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper")
        elif self.target_cc == 80 or self.target_cc in (94, 95):
            # 94: MI300/ROCm – use sm_80-style "linear_with_residual"
            self.kn_graph.register_task(tb_graph, "linear_with_residual")
        else:
            self._raise_unsupported_target_cc(
                "linear_with_residual_layer",
                ["linear_with_residual_sm100", "linear_swapAB_with_residual_hopper", "linear_with_residual"],
            )

    def allreduce_layer(
        self,
        input: DTensor,
        buffer: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size)
        assert buffer.num_dims == 3  # (world_size, batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        # params[0]: num_gpus
        # params[1]: my_gpu_id
        params = [self.world_size, self.mpi_rank]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), -1, True)
        tb_graph.new_input(buffer, (2, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, buffer, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "allreduce", params)

    def bias_add_layer(
        self,
        input: DTensor,
        bias: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Element-wise bias add: output = input + bias (broadcast across batch)."""
        assert input.num_dims == 2  # (batch_size, size)
        assert bias.num_dims == 2   # (1, size) - pre-unsqueezed
        assert output.num_dims == 2 # (batch_size, size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(bias, (-1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, bias, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "bias_add_mi300")

    def silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2 # (batch_size, 2 * intermediate_size)
        assert output.num_dims == 2 # (batch_size, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul" if self.target_cc == 90 else "silu_mul")

    def identity_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        dependent_tensor: DTensor = None,
    ):
        # TODO: Add support from kn_graph
        last_dim = 0
        assert input.num_dims == output.num_dims
        for i in range(input.num_dims):
            assert input.dim(i) == output.dim(i)
            last_dim = i
        assert last_dim == 1 or last_dim == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (last_dim, -1, -1), 1, True)
        tb_graph.new_input(output, (last_dim, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "identity")

    def silu_mul_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, 2*intermediate_size)
        assert weight.num_dims == 2  # (hidden_size, intermediate_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul_linear_with_residual")

    def argmax_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "argmax")

    def argmax_partial_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert len(output) == 2
        output_value, output_index = output
        assert output_value.num_dims == 2  # (batch_size, num_tasks)
        assert output_index.num_dims == 2  # (batch_size, num_tasks)
        num_tasks = grid_dim[0]
        self.argmax_partial_output_size = input.dim(1) // num_tasks
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, 0, -1), -1, True)
        tb_graph.new_input(output_value, (1, 0, -1), -1, True)
        tb_graph.new_input(output_index, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, output_value, output_index], tb_graph)
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "argmax_partial_sm100", [num_tasks])
        else:
            self.kn_graph.register_task(tb_graph, "argmax_partial", [num_tasks])

    def argmax_reduce_layer(
        self,
        input: tuple[DTensor, DTensor],
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert len(input) == 2
        input_value, input_index = input
        assert input_value.num_dims == 2  # (batch_size, num_tasks)
        assert input_index.num_dims == 2  # (batch_size, num_tasks)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_value, (1, 0, -1), -1, True)
        tb_graph.new_input(input_index, (1, 0, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True) #TODO: Make sure the output map is expected
        self.kn_graph.customized([input_value, input_index, output], tb_graph)
        if self.target_cc == 100:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce_sm100", [self.argmax_partial_output_size]
            )
        else:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce", [self.argmax_partial_output_size]
            )

    def argmax_reduce_xrank_layer(
        self,
        input: tuple[DTensor, DTensor],
        xrank: DTensor,
        output: DTensor,
        vocab_shard: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """argmax_reduce over a vocab-SHARDED logit slice, plus a peer exchange.

        The sharded LM head leaves rank p holding logits for global vocab rows
        [p*vocab_shard, (p+1)*vocab_shard). This finishes the local reduce the
        same way argmax_reduce_layer does, rebases the winning index into the
        global vocab, and then exchanges one 64-bit (value, index) word with
        every peer so all ranks emit the same token.

        ``xrank`` is a symmetric-heap u64 buffer of at least
        world_size * ARGMAX_XRANK_SLOT_U64 (= 8) words; the kernel writes this
        rank's slot on every peer and spins on the others'.
        """
        assert len(input) == 2
        input_value, input_index = input
        assert input_value.num_dims == 2  # (batch_size, num_tasks)
        assert input_index.num_dims == 2  # (batch_size, num_tasks)
        assert output.num_dims == 2  # (batch_size, 1)
        assert self.world_size > 1, "xrank argmax needs more than one rank"
        num_parts = input_value.dim(1)
        assert vocab_shard % num_parts == 0
        chunk_size = vocab_shard // num_parts
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_value, (1, 0, -1), -1, True)
        tb_graph.new_input(input_index, (1, 0, -1), -1, True)
        tb_graph.new_input(xrank, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [input_value, input_index, xrank, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph,
            "argmax_reduce_xrank",
            [chunk_size, self.world_size, self.mpi_rank, vocab_shard],
        )

    def xrank_sum_add_layer(
        self,
        partial: DTensor,
        residual: DTensor,
        xbuf: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        dep: DTensor = None,
    ):
        """output = residual + sum over ranks of each rank's `partial`.

        For the intermediate-sharded dense MLP: gate_up is column-parallel and
        down_proj is K-parallel over the matching slice, so every rank produces
        a full-width hidden vector that is only a PARTIAL sum. This exchanges
        the four partials and adds the residual exactly once.

        ``xbuf`` is a symmetric-heap buffer of at least
        world_size * batch * hidden bf16 payload words followed by
        world_size * XRANK_SUM_SLOT_U64 (= 8) u64 epoch words. The caller must
        pass the down_proj GEMV a ZERO residual -- a folded residual would be
        summed world_size times here.

        ``dep`` is a dependency-only trailing input, never read by the kernel.
        Pass it when the producer writes a column SLICE of ``partial`` -- a
        distinct DTensor over the same storage -- so the linear task chain's
        shared-guid edge test can still see the edge.
        """
        assert partial.num_dims == 2  # (batch_size, hidden)
        assert residual.num_dims == 2
        assert output.num_dims == 2
        assert self.world_size > 1, "xrank_sum_add needs more than one rank"
        hidden = partial.dim(1)
        assert residual.dim(1) == hidden and output.dim(1) == hidden
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(residual, (-1, -1, -1), -1, True)
        tb_graph.new_input(xbuf, (-1, -1, -1), -1, True)
        ins = [partial, residual, xbuf]
        if dep is not None:
            tb_graph.new_input(dep, (-1, -1, -1), -1, True)
            ins.append(dep)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized(ins + [output], tb_graph)
        self.kn_graph.register_task(
            tb_graph,
            "xrank_sum_add",
            [hidden, self.world_size, self.mpi_rank],
        )

    def sampling_sm100_layer(
        self,
        logits: DTensor,      # [batch_size, vocab_size]
        output: DTensor,      # [batch_size, 1]
        grid_dim: tuple,
        block_dim: tuple,
        seed: int = 42,
    ):
        """Sampling from logits using Gumbel-Max trick for stochastic token generation."""
        assert logits.num_dims == 2      # (batch_size, vocab_size)
        assert output.num_dims == 2      # (batch_size, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized([logits, output], tb_graph)

        # Register task with seed parameter
        self.kn_graph.register_task(tb_graph, "sampling_sm100", [seed])

    def find_ngram_partial_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, seq_len)
        assert output.num_dims == 2  # (batch_size, num_tasks)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_partial", [ngram_size])
        
    def find_ngram_global_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3, spec_length: int = 5):
        # Currently assume that input/output
        assert len(input) == 2
        partial_results, tokens = input
        assert partial_results.num_dims == 2  # (batch_size, num_tasks)
        assert tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_results, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([partial_results, tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_global", [ngram_size, spec_length])

    def prompt_lookup_spec_handler(
        self, 
        spec_decode_config: PromptLookupConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        partial_ngram_output = self.new_tensor(
            dims=(tokens.dim(0), 96),
            dtype=int64,
            name="partial_ngram_output",
            io_category="cuda_tensor",
        )
        self.find_ngram_partial_layer(
            input=tokens, 
            output=partial_ngram_output, 
            grid_dim=grid_dim, 
            block_dim=block_dim, 
            ngram_size=spec_decode_config.ngram_size
        )
        spec_tokens = self.new_tensor(
            dims=(tokens.dim(0), spec_decode_config.spec_length + 1),
            dtype=int64,
            name="spec_tokens",
            io_category="cuda_tensor",
        )   
        self.find_ngram_global_layer(
            input=(partial_ngram_output, tokens), 
            output=spec_tokens, 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1), 
            ngram_size=spec_decode_config.ngram_size,
            spec_length=spec_decode_config.spec_length
        )
        return spec_tokens
    
    def draft_forward_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        method = spec_decode_config.method
        handler = self._spec_decode_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, tokens, grid_dim, block_dim)
    
    def target_verify_greedy_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple):
        # Currently assume that input/output
        # This tensor is not realy used
        assert len(input) == 2
        spec_tokens, target_tokens = input
        assert spec_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert target_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(spec_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([spec_tokens, target_tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "target_verify_greedy")
        
    def prompt_lookup_verify_handler(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        # This tensor is not realy used
        verify_out = self.new_tensor(
            dims=(1, 1),
            dtype=int64,
            name="verify_out",
            io_category="cuda_tensor",
        )
        self.target_verify_greedy_layer(
            input=(spec_tokens, target_output), output=verify_out, grid_dim=grid_dim, block_dim=block_dim
        )
        return verify_out
    
    def verify_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int] = (1, 1, 1),
        block_dim: tuple[int, int, int] = (128, 1, 1),
    ):
        method = spec_decode_config.method
        handler = self._spec_verify_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, spec_tokens, target_output, grid_dim, block_dim)

    def compile(
        self,
        **kwargs,
    ):
        assert not self._is_compiled
        
        output_dir = kwargs.get("output_dir", None)

        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        # Each rank bakes its own (process-local) torch device pointers into the
        # generated test.cu, so multi-GPU SPMD runs MUST NOT share an output
        # directory; otherwise the ranks race on the same file/.so and one rank
        # compiles the other rank's pointers (-> cudaMemcpy DtoD "invalid
        # argument"). Give each rank its own directory when world_size > 1.
        if self.world_size > 1:
            base_output_dir = f"./permanent_output_dir_rank{self.mpi_rank}/"
        else:
            base_output_dir = "./permanent_output_dir/"
        if self.mode == "online_notoken" or self.mode == "online" or self.mode == "multi_turn":
            # We will init for multiple times so the output directory should be permanent
            tempdir = base_output_dir
        else:
            tempdir_obj = tempfile.TemporaryDirectory()
            #tempdir = tempdir_obj.name
            tempdir = base_output_dir
        os.makedirs(tempdir, exist_ok=True)
        results = self.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)

        cuda_code_path = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")
        # check json file
        json_file_path = os.path.join(tempdir, "task_graph.json")
        # build if files are not exist
            
        with open(json_file_path, "w") as f:
            f.write(results["json_file"])

        # Event fusion DISABLED to match NVIDIA implementation
        # Original event fusion code (AMD only) reduced events but didn't improve performance
        # if self.target_cc in (94, 95):  # AMD MI300
        #     from .event_fusion import fuse_events
        #     import json
        #     with open(json_file_path, "r") as f:
        #         task_graph = json.load(f)
        #     original_events = len(task_graph['all_events'])
        #     fused_graph = fuse_events(task_graph)
        #     fused_events = len(fused_graph['all_events'])
        #     print(f"Event fusion: {original_events} -> {fused_events} events ({(1-fused_events/original_events)*100:.1f}% reduction)")
        #     with open(json_file_path, "w") as f:
        #         json.dump(fused_graph, f)

        with open(cuda_code_path, "w") as f:
            f.write(results["cuda_code"] + HARD_CODE)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{self.mpi_rank}.cu"))
            shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{self.mpi_rank}.json"))

        if self.target_cc in (94, 95):
            rocm_home = os.environ.get("ROCM_PATH", "/opt/rocm")
            cc = shutil.which("hipcc") or os.path.join(rocm_home, "bin", "hipcc")
            if not cc or not os.path.isfile(cc):
                raise RuntimeError(
                    "hipcc not found. For MI300/ROCm builds set ROCM_PATH or ensure hipcc is on PATH."
                )
        else:
            cc = shutil.which("nvcc")
            if cc is None:
                raise RuntimeError(
                    "nvcc not found. Please make sure you have installed CUDA."
                )
        # This function was renamed and made public in Python 3.10
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

        # find mirage home
        if "MIRAGE_HOME" in os.environ:
            MIRAGE_HOME_PATH = os.environ.get("MIRAGE_HOME")
        else:
            raise RuntimeError(
                "MIRAGE_HOME unspecified; Please set MIRAGE_HOME to be the root of the Mirage folder"
            )

        NVSHMEM_INC_PATH = None
        NVSHMEM_LIB_PATH = None
        ROCSHMEM_INC_PATH = None
        ROCSHMEM_LIB_PATH = None
        MPI_INC_PATH = None
        MPI_LIB_PATH = None
        if self.use_nvshmem:
            # find nvshmem include folder and library folder
            if "NVSHMEM_INC_PATH" in os.environ:
                NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}"
                    )
            else:
                NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Cannot find nvshmem.h, please set environment variable NVSHMEM_INC_PATH"
                    )
            # find nvshmem shared library
            if "NVSHMEM_LIB_PATH" in os.environ:
                NVSHMEM_LIB_PATH = os.environ.get("NVSHMEM_LIB_PATH")
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_LIB_PATH is set but cannot find libnvshmem.a at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/"
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Cannot find libnvshmem.a, please set environment variable NVSHMEM_LIB_PATH"
                    )
            # find mpi include foler
            if "MPI_INC_PATH" in os.environ:
                MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_INC_PATH is set but cannot find mpi.h at {header_file_path}"
                    )
            else:
                MPI_INC_PATH = "/usr/include/"
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Cannot find mpi.h, please set environment variable MPI_INC_PATH"
                    )
            # find mpi shared library
            if "MPI_LIB_PATH" in os.environ:
                MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_LIB_PATH is set but cannot find libmpi.so at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/"
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Cannot find libmpi.so, please set environment variable MPI_LIB_PATH"
                    )

        if self.use_rocshmem:
            # find rocSHMEM include folder
            if "ROCSHMEM_INC_PATH" in os.environ:
                ROCSHMEM_INC_PATH = os.environ.get("ROCSHMEM_INC_PATH")
            else:
                ROCSHMEM_INC_PATH = os.path.join(
                    os.path.expanduser("~"), "rocshmem", "include"
                )
            header_file_path = os.path.join(
                ROCSHMEM_INC_PATH, "rocshmem", "rocshmem.hpp"
            )
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    f"Cannot find rocshmem/rocshmem.hpp at {header_file_path}, "
                    "please set environment variable ROCSHMEM_INC_PATH"
                )
            # find rocSHMEM static library (librocshmem.a)
            if "ROCSHMEM_LIB_PATH" in os.environ:
                ROCSHMEM_LIB_PATH = os.environ.get("ROCSHMEM_LIB_PATH")
            else:
                ROCSHMEM_LIB_PATH = os.path.join(
                    os.path.expanduser("~"), "rocshmem", "lib"
                )
            lib_file_path = os.path.join(ROCSHMEM_LIB_PATH, "librocshmem.a")
            if not os.path.exists(lib_file_path):
                raise RuntimeError(
                    f"Cannot find librocshmem.a at {lib_file_path}, "
                    "please set environment variable ROCSHMEM_LIB_PATH"
                )
            # find mpi include folder (rocSHMEM bootstraps through MPI)
            if "MPI_INC_PATH" in os.environ:
                MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
            else:
                MPI_INC_PATH = "/usr/lib/x86_64-linux-gnu/openmpi/include"
            header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
            if not os.path.exists(header_file_path):
                raise RuntimeError(
                    f"Cannot find mpi.h at {header_file_path}, "
                    "please set environment variable MPI_INC_PATH"
                )
            # find mpi shared library
            if "MPI_LIB_PATH" in os.environ:
                MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
            else:
                MPI_LIB_PATH = "/usr/lib/x86_64-linux-gnu/openmpi/lib"
            lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
            if not os.path.exists(lib_file_path):
                raise RuntimeError(
                    f"Cannot find libmpi.so at {lib_file_path}, "
                    "please set environment variable MPI_LIB_PATH"
                )

        cc_cmd = get_compile_command(
            mpk=self,
            target_cc=self.target_cc,
            cc=cc,
            file_name=cuda_code_path,
            py_include_dir=py_include_dir,
            mirage_home_path=MIRAGE_HOME_PATH,
            mirage_inc_path=INCLUDE_PATH,
            mirage_deps_path=DEPS_PATH,
            nvshmem_inc_path=NVSHMEM_INC_PATH,
            nvshmem_lib_path=NVSHMEM_LIB_PATH,
            mpi_inc_path=MPI_INC_PATH,
            mpi_lib_path=MPI_LIB_PATH,
            py_so_path=so_path,
            profiling=True if self.profiler_tensor is not None else False,
            use_nvshmem=self.use_nvshmem,
            use_rocshmem=self.use_rocshmem,
            rocshmem_inc_path=ROCSHMEM_INC_PATH,
            rocshmem_lib_path=ROCSHMEM_LIB_PATH,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_local_schedulers, 
            num_remote_schedulers=self.num_remote_schedulers,
            use_cutlass_kernel=self.use_cutlass_kernel,
        )
        print("Compiling megakernel using the following command line:")
        print(cc_cmd)
        subprocess.check_call(cc_cmd)

        import importlib.util

        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.init_request_func = getattr(mod, "init_request_func")
        self.finalize_func = getattr(mod, "finalize_func")
        self._set_rope_tables_func = getattr(mod, "set_rope_tables_func", None)
        self._set_spec_draft_tokens_func = getattr(
            mod, "set_spec_draft_tokens_func", None
        )
        self._read_shmem_alloc_func = getattr(mod, "read_shmem_alloc_func", None)
        self._num_shmem_allocs_func = getattr(mod, "num_shmem_allocs_func", None)
        self._shmem_alloc_size_func = getattr(mod, "shmem_alloc_size_func", None)
        print("Finished megakernel compilation...")

        #meta_tensors_ptr = [tensor.data_ptr() for tensor in self.meta_tensors]
        meta_tensors = list()
        meta_tensors.append(self.meta_tensors["step"])
        meta_tensors.append(self.meta_tensors["tokens"])
        meta_tensors.append(self.meta_tensors["input_tokens"])
        meta_tensors.append(self.meta_tensors["output_tokens"])
        meta_tensors.append(self.meta_tensors["num_new_tokens"])
        meta_tensors.append(self.meta_tensors["prompt_lengths"])
        meta_tensors.append(self.meta_tensors["qo_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indices_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_last_page_len_buffer"])
        meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        profiler_buffer_ptr = (
            self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        )
        self.eos_token_id = kwargs.get("eos_token_id", self.eos_token_id)
        self.init_func(
            meta_tensors_ptr,
            profiler_buffer_ptr,
            self.mpi_rank,
            self.num_workers,
            self.num_local_schedulers,
            self.num_remote_schedulers,
            self.max_seq_length,
            self.total_num_requests,
            self.eos_token_id,
        )

        self._is_compiled = True

        # self.call_func = getattr(mod, "call_func")

    def set_rope_tables(self, cos_tensor: "torch.Tensor", sin_tensor: "torch.Tensor"):
        """Set RoPE cos/sin tables in RuntimeConfig (call after compile)."""
        assert self._is_compiled, "Must call compile() before set_rope_tables()"
        assert self._set_rope_tables_func is not None
        self._set_rope_tables_func(cos_tensor.data_ptr(), sin_tensor.data_ptr())

    def set_spec_draft_tokens(self, tensor: "torch.Tensor"):
        """Point RuntimeConfig at the draft head's output (call after compile).

        Kept out of meta_tensors deliberately -- that list is asserted at size
        10 and is shared with every other model in the repo. Only exists when
        the megakernel was built with MPK_SPEC_DECODE."""
        assert self._is_compiled, "Must call compile() before set_spec_draft_tokens()"
        assert self._set_spec_draft_tokens_func is not None, (
            "megakernel was not built with MPK_SPEC_DECODE"
        )
        self._set_spec_draft_tokens_func(tensor.data_ptr())

    def num_shmem_allocs(self) -> int:
        """Number of recorded symmetric-heap (nvshmem/rocshmem) allocations."""
        assert self._num_shmem_allocs_func is not None
        return int(self._num_shmem_allocs_func())

    def shmem_alloc_size(self, index: int) -> int:
        """Byte size of the index-th symmetric-heap allocation (0 if invalid)."""
        assert self._shmem_alloc_size_func is not None
        return int(self._shmem_alloc_size_func(index))

    def read_shmem_alloc(self, index: int, dst_tensor: "torch.Tensor") -> int:
        """Snapshot the index-th symmetric-heap allocation into dst_tensor
        (a CUDA tensor). Returns 0 on success, -1 on bad index/size."""
        assert self._read_shmem_alloc_func is not None
        nbytes = dst_tensor.numel() * dst_tensor.element_size()
        return int(
            self._read_shmem_alloc_func(index, dst_tensor.data_ptr(), nbytes)
        )

    def __call__(self, **kwargs):
        stream = kwargs.get("default_stream", None)
        if stream is None:
           stream = torch.cuda.current_stream()
        # Convert torch.cuda.Stream to raw pointer (integer) for the C launcher
        stream_ptr = 0
        if hasattr(stream, "cuda_stream"):
            try:
                stream_ptr = int(stream.cuda_stream)
            except Exception:
                try:
                    stream_ptr = int(stream.cuda_stream.value)
                except Exception as e:
                    raise ValueError(f"Invalid stream object: {stream} is of type {type(stream)}: {e}")
        elif isinstance(stream, int):
            stream_ptr = stream
        else:
            raise ValueError("Invalid stream object")
        self.launch_func(stream_ptr)
        if self.profiler_tensor is not None:
            from .profiler_persistent import export_to_perfetto_trace
            
            if self.trace_name:
                trace_name = self.trace_name + ".perfetto-trace"
            else:
                trace_name = f"mirage_{self.mpi_rank}.perfetto-trace"

            export_to_perfetto_trace(
                self.profiler_tensor, trace_name
            )
            # Also save raw profiler tensor for programmatic analysis
            raw_path = trace_name.replace(".perfetto-trace", ".pt")
            torch.save(self.profiler_tensor.cpu(), raw_path)
            print(f"Saved raw profiler tensor to {raw_path}")

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func()
        self.__finalized__ = True
