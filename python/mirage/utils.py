import os
import torch

# This function returns the shared memory limit (in bytes)
# for the given GPU hardware architecture
def get_shared_memory_capacity(target_cc):
    if target_cc == 80:
        # A100 GPUs
        return 163 * 1024
    elif target_cc == 86:
        # A5000 GPUs
        return 99 * 1024
    elif target_cc == 89:
        # A6000 GPUs
        return 99 * 1024
    elif target_cc == 90:
        # H100 GPUs
        return 223 * 1024
    elif target_cc == 100:
        # B200 GPUs
        return 227 * 1024
    else:
        assert False, "Unsupported compute capacity: {}".format(target_cc)


def get_scheduler(sm_cnt, worker):
    scheduler = sm_cnt - worker
    assert scheduler > 0, "worker count is not compatible with sm count on"
    "the GPU"
    return sm_cnt - worker

# MAX_NUM_WORKERS must match C++ runtime_header.h (used in MPK device asserts).
# Increased to 304 to support full CU utilization on AMD MI300X (304 CUs)
MAX_NUM_WORKERS = 304


# This method auto probe GPUs and return the worker and scheduler count for
# them.
def get_configurations_from_gpu(rank):
    # Reference: https://github.com/mirage-project/mirage/issues/354
    props = torch.cuda.get_device_properties(rank)
    sm_cnt = props.multi_processor_count
    print("sm_cnt: ", sm_cnt)

    # Check if this is an AMD GPU (ROCm/HIP)
    is_amd = hasattr(torch.version, "hip") and torch.version.hip is not None

    worker = 0
    if is_amd:
        # AMD MI300X configuration (split_worker_scheduler mode)
        # XCD-aligned scheduling: 1 scheduler per XCD (8 total for MI300X).
        # Each scheduler handles all workers on its XCD via stride-based mapping.
        # scheduler_kernel launches num_schedulers blocks with 1 warp (32 threads) each.
        if sm_cnt >= 300:
            num_xcds = 8
            worker = sm_cnt - num_xcds  # 296 workers, use all CUs
            scheduler = num_xcds
            worker = min(worker, MAX_NUM_WORKERS)
            print(f"AMD config: workers={worker}, schedulers={scheduler}, "
                  f"physical_blocks={worker + scheduler}, CUs={sm_cnt}")
            return worker, scheduler
        elif sm_cnt >= 200:
            # MI350: 256 CUs, 8 XCDs (32 CUs/XCD)
            num_xcds = 8
            worker = 240  # Same as MI300X for compatibility
            # 240 workers + 8 schedulers = 248 blocks on 256 CUs, i.e. 31 of
            # the 32 CUs on every XCD. The megakernel needs every block
            # co-resident, so this margin is load-bearing.
            #
            # At NP=8 it is not enough. The last worker on an XCD (xcd_rank 29)
            # never becomes resident at all -- worker-state slot still at
            # MPK_WS_UNWRITTEN, worker_xcd_ready_count stuck at 239 of 240, and
            # all eight schedulers parked in the bootstrap wait having never
            # dispatched a task. An earlier reading of this as "stops executing
            # mid-spin" was wrong: the block never ran. Workers and schedulers
            # are separate kernels on separate streams and are round-robined
            # onto XCDs independently, so nothing puts the 8 scheduler blocks
            # one per XCD, and an XCD drawing two of them needs 33 slots for 32.
            #
            # MPK_NUM_WORKERS buys the headroom. demo/glm5/run_mp8_dp_ep_fused.sh
            # sets 232 (29/XCD), which is what makes the full 78-layer NP=8 run
            # complete; see the measurements there. The default stays 240 so
            # gpt-oss and single-GPU runs keep the config their recorded
            # latencies were measured against. Rounded down to a multiple of 8
            # so every XCD still gets the same number.
            _w_env = os.environ.get("MPK_NUM_WORKERS")
            if _w_env:
                worker = max(num_xcds, (int(_w_env) // num_xcds) * num_xcds)
            scheduler = num_xcds
            worker = min(worker, MAX_NUM_WORKERS)
            print(f"AMD MI350 config: workers={worker}, schedulers={scheduler}, "
                  f"physical_blocks={worker + scheduler}, CUs={sm_cnt}")
            return worker, scheduler
        elif sm_cnt >= 120:
            worker = 96   # 96 + 96 = 192 -> scaled for 120+ CUs
        elif sm_cnt >= 60:
            worker = 48
        else:
            worker = 24
    else:
        # NVIDIA GPU configuration (unchanged)
        if sm_cnt >= 160:
            worker = 144  # Blackwell B200
        elif sm_cnt >= 132:
            worker = 128  # Hopper H100
        elif sm_cnt >= 108:
            worker = 96   # Ampere A100
        elif sm_cnt >= 68:
            worker = 64
        elif sm_cnt >= 40:
            worker = 30
        else:
            worker = 20

    # Cap workers at MAX_NUM_WORKERS
    worker = min(worker, MAX_NUM_WORKERS)
    scheduler = get_scheduler(sm_cnt, worker)

    if is_amd:
        # In split mode, each scheduler block has 1 warp (32 threads).
        # Physical blocks = workers + schedulers = sm_cnt (all CUs used).
        print(f"AMD config: workers={worker}, schedulers={scheduler}, "
              f"physical_blocks={worker + scheduler}, CUs={sm_cnt}")

    return worker, scheduler
