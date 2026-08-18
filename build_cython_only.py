#!/usr/bin/env python3
"""Rebuild python/mirage/core*.so from the already-built libmirage_runtime.a.

The root setup.py does this too, but it first re-runs cargo and re-configures
cmake with CC/CXX forced to gcc -- and this tree is configured with ROCm's
clang++, so that reconfigure would either fail or trigger a full rebuild of
the runtime for nothing. `cd build && make -j32` already refreshed the static
lib; the only thing left is to relink the Cython extension against it.

python/cython_setup.py is not usable here: it hardcodes the CUDA backend.
This mirrors setup.py's config_cython() for the ROCm branch, nothing more.

    python3 build_cython_only.py
"""
import os
import sys
from os import path

import z3
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

MIRAGE = path.dirname(path.abspath(__file__))
ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
z3_path = path.dirname(z3.__file__)

ext = Extension(
    "mirage.core",
    ["python/mirage/_cython/core.pyx"],
    include_dirs=[
        path.join(MIRAGE, "include"),
        path.join(MIRAGE, "deps", "json", "include"),
        path.join(MIRAGE, "deps", "rocblas", "include"),
        path.join(MIRAGE, "build", "abstract_subexpr", "release"),
        path.join(MIRAGE, "build", "formal_verifier", "release"),
        path.join(z3_path, "include"),
        path.join(ROCM, "include"),
    ],
    library_dirs=[
        path.join(MIRAGE, "build"),
        path.join(z3_path, "lib"),
        path.join(MIRAGE, "build", "abstract_subexpr", "release"),
        path.join(MIRAGE, "build", "formal_verifier", "release"),
        path.join(ROCM, "lib"),
        path.join(ROCM, "lib64"),
        path.join(ROCM, "llvm", "lib"),
    ],
    libraries=["mirage_runtime", "z3", "rt", "abstract_subexpr",
               "formal_verifier", "omp", "amdhip64", "rocblas", "hipblas"],
    define_macros=[("MIRAGE_BACKEND_USE_ROCM", None),
                   ("MIRAGE_FINGERPRINT_USE_ROCM", None)],
    extra_compile_args=["-std=c++17", "-fopenmp", "-D__HIP_PLATFORM_AMD__=1"],
    extra_link_args=[
        "-fPIC", "-fopenmp", "-lrt",
        f"-L{path.join(ROCM, 'llvm', 'lib')}", "-lomp",
        f"-Wl,-rpath,{path.join(ROCM, 'llvm', 'lib')}",
        f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'abstract_subexpr', 'release')}",
        f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'formal_verifier', 'release')}",
    ],
    language="c++",
)

os.chdir(MIRAGE)
sys.argv = [sys.argv[0], "build_ext", "--inplace"]
# package_dir is what tells --inplace to land the .so in python/mirage/ rather
# than in a mirage/ directory at the repo root, which does not exist.
setup(name="mirage-core-only", packages=["mirage"], package_dir={"": "python"},
      ext_modules=cythonize([ext], compiler_directives={"language_level": 3}))
