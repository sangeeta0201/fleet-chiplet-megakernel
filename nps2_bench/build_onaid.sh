#!/bin/bash
# Build the AID-boundary isolation benchmark.
#
# aid_local.h's namespace sits behind
#   #if defined(MIRAGE_BACKEND_USE_ROCM) && (__HIP_PLATFORM_AMD__ || MIRAGE_AMD_MI300)
# so without -DMIRAGE_BACKEND_USE_ROCM the header is found but contributes
# nothing and the build fails with "use of undeclared identifier 'mirage'".
set -e
cd "$(dirname "$0")"
hipcc -O3 --offload-arch=gfx950 \
      -DMIRAGE_BACKEND_USE_ROCM \
      -I. -I../include/mirage/persistent_kernel \
      onaid.cpp -o onaid 2> >(grep -v "nodiscard\|hipMemset\|hipMalloc\|hipDeviceSync\|^ *[0-9]* |\|^ *\^\|^In file included\|warning: ignoring" >&2)
echo "built: $(pwd)/onaid"
