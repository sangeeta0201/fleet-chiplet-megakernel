#!/bin/bash
# Isolated experiment tree, so two concurrent experiments cannot poison each
# other's runs.
#
# WHY THIS EXISTS. The megakernel is JIT-compiled, so a .cuh edit is live on the
# next run. CLAUDE.md's rule -- "Do not edit any .cuh or .py while a batch is in
# flight; the JIT picks the edit up mid-batch and poisons every remaining run" --
# is violated automatically the moment two agents share one tree. They also share
# one permanent_output_dir, so arm A's code object can serve arm B's run.
#
# WHY /mnt/nvme1 AND NOT A git worktree. Two reasons. `/` is at 99% (a full root
# has previously made every run fail in ways that looked like kernel bugs) while
# /mnt/nvme1 has ~1.8 TB, and `/mnt` is bind-mounted into fleet_v2 at the same
# path so the container sees the tree unchanged. A git worktree would check out a
# clean branch and DROP the uncommitted work in flight, which is exactly the
# state an experiment needs to fork from -- so this copies instead.
#
# The tree is ~1.1 GB excluding build dirs, so a copy is cheap.
#
#   ./new_exp_tree.sh coll_fuse
#   docker exec -u claudeuser fleet_v2 bash -lc \
#     'cd /mnt/nvme1/exp/coll_fuse/demo/glm5 && HIP_VISIBLE_DEVICES=0,1,2,3 ./run_latency_1k1k.sh'
#
# Then apply only what won back to the original tree, one lever at a time.
set -eu

NAME="${1:?usage: new_exp_tree.sh <name>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DST="/mnt/nvme1/exp/${NAME}"

if [ -e "$DST" ]; then
  echo "[exp] $DST already exists; pick another name or rm -rf it" >&2
  exit 1
fi

# Guard the failure mode this script exists to avoid: never stage an experiment
# onto the nearly-full root.
avail_mb=$(df -Pm /mnt/nvme1 | awk 'NR==2 {print $4}')
if [ "$avail_mb" -lt 8192 ]; then
  echo "[exp] /mnt/nvme1 has only ${avail_mb} MiB free; refusing" >&2
  exit 1
fi

mkdir -p /mnt/nvme1/exp
echo "[exp] copying $SRC -> $DST"
# Exclude build outputs: permanent_output_dir* is per-tree JIT state that MUST
# NOT be inherited (a stale code object is how an A/B silently measures the same
# binary twice), and .git/build/deps are large and regenerable.
tar -C "$SRC" \
    --exclude='./permanent_output_dir*' \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./deps' \
    -cf - . | (mkdir -p "$DST" && tar -C "$DST" -xf -)

# build/ and deps/ are needed to run but are identical across experiments, so
# link rather than duplicate ~800 MB per tree.
for shared in build deps; do
  [ -d "$SRC/$shared" ] && ln -s "$SRC/$shared" "$DST/$shared"
done

echo "[exp] ready: $DST"
echo "[exp] run it with an explicit device set, e.g.:"
echo "  docker exec -u claudeuser fleet_v2 bash -lc \\"
echo "    'cd $DST/demo/glm5 && HIP_VISIBLE_DEVICES=0,1,2,3 STALL_SECS=900 ./run_latency_1k1k.sh'"
echo "[exp] NOTE: absolutes on devices 0-3 are NOT comparable to the 14.0 ms"
echo "      canonical baseline (devices 4-7, see run_mp8_dp_ep_fused.sh's"
echo "      socket handling). Run BOTH arms on the same devices and quote the delta."
