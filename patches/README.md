# Out-of-tree patches

Fixes to dependencies that this branch's performance depends on. Each one has
to be applied and the dependency rebuilt, or the megakernel silently falls back
to a slower path -- these are not optional cleanups.

## rocshmem-ipc-shmem-ptr.patch

Applies to rocSHMEM (`/home/claudeuser/rocshmem_src`, upstream ee5363b).

`IPCContext::shmem_ptr` returned `nullptr` unconditionally -- `shmem_ptr` was
only ever implemented for the reverse-offload backend, and the IPC backend got
a stub. Since `nullptr` is also the legitimate "no peer mapping" answer, the
gap is invisible at the call site: the megakernel's EP combine has a
direct-store fast path gated on `mpk_shmem_peer_ptr() != nullptr`, and it had
never once executed. Every EP number on this branch before ae6e4b7 was
measuring the staged `putmem_signal` fallback.

The fix is the arithmetic `IPCContext::putmem` already does three lines below:
`ipc_bases[pe] + (dest - ipc_bases[my_pe])`.

    cd /home/claudeuser/rocshmem_src
    git apply /path/to/fleet/patches/rocshmem-ipc-shmem-ptr.patch
    cd build && make -j32 && make install

Verify with `tests/standalone/test_rocshmem_ptr.hip`, which checks the
end-to-end property rather than just non-null: each rank stores a tag through
the returned pointer and the peer reads it back. A non-null pointer that does
not carry data would be worse than the stub.

### Ordering requirement this exposed

rocSHMEM never calls `hipSetDevice` itself, and `rocshmem_init()` publishes the
default device context via `hipMemcpyToSymbol`, which writes only to the
CURRENT device. Select the device BEFORE `rocshmem_init()` or rank 1's
`ROCSHMEM_CTX_DEFAULT` stays null and every device-side rocSHMEM call
dereferences it.

The megakernel already gets this right (`cudaSetDevice(my_rank)` precedes
`rocshmem_init()` in `persistent_kernel.cuh`). It only surfaced here because
the old stub never dereferenced `this`, so a null default context was
harmless -- it just returned null, like it always did. Anything else linking
rocSHMEM needs the same ordering.
