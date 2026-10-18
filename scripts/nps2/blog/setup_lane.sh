#!/bin/bash
# setup_lane.sh NAME [COMMIT]: a worktree of this repository at $HOST_HOME/fleet-lanes/NAME for the blog runs, reusing
# the built dependencies of existing trees instead of rebuilding them:
#   deps/{composable_kernel,cutlass,json} and build/  from $HOST_HOME/fleet-lanes/$DEPS_TREE   (default repro2)
#   deps/{rocblas,z3}                                 from $HOST_HOME/fleet-lanes/$DEPS_TREE2  (default cpx)
#   python/mirage/core.cpython-312-x86_64-linux-gnu.so from this checkout
# The links are relative, so they also resolve inside the container. Without such trees, clone and build instead
# (README_GPT_OSS_120B_MI355X.md, step 0: submodules + `python setup.py build_ext --inplace` in the container).
# Use a separate worktree for each GPU that runs at the same time: builds go to <tree>/demo/gpt_oss.
set -eu
NAME=${1:?usage: setup_lane.sh NAME [COMMIT]}
COMMIT=${2:-HEAD}
HOST_HOME=${HOST_HOME:-$HOME}
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
SRCNAME=$(basename "$SRC")
DEPS_TREE=${DEPS_TREE:-repro2}
DEPS_TREE2=${DEPS_TREE2:-cpx}
T=$HOST_HOME/fleet-lanes/$NAME
[ -d "$T" ] || git -C "$SRC" worktree add --detach "$T" "$COMMIT" > /dev/null
cd "$T"
for d in composable_kernel cutlass json; do rm -rf deps/$d; ln -s ../../$DEPS_TREE/deps/$d deps/$d; done
for d in rocblas z3; do rm -rf deps/$d; ln -s ../../$DEPS_TREE2/deps/$d deps/$d; done
rm -rf build
ln -s ../$DEPS_TREE/build build
cp "$SRC/python/mirage/core.cpython-312-x86_64-linux-gnu.so" python/mirage/
sed -e "s|FLEET_LANE=$SRCNAME\$|FLEET_LANE=$NAME|" -e "s|/fleet-lanes/$SRCNAME/|/fleet-lanes/$NAME/|" \
    -e "s|/fleet-lanes/$SRCNAME\$|/fleet-lanes/$NAME|" -e "s|lane_$SRCNAME\$|lane_$NAME|" "$SRC/LANE.env" > LANE.env
echo "worktree $T at $(git log -1 --format='%h %s' | cut -c1-80)"
grep -E 'FLEET_LANE|FLEET_TREE|FLEET_LOG_DIR|MIRAGE_OUTPUT_DIR' LANE.env
for d in deps/composable_kernel deps/cutlass deps/json deps/rocblas deps/z3 build; do
  [ -e "$d/" ] || echo "MISSING: $d -> $(readlink $d)"
done

