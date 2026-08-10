#!/usr/bin/env python
# Minimal setup.py -- metadata lives in pyproject.toml.
#
# Two build systems share this file, each owning what it is good at:
#   * setuptools builds the Cython cores as ordinary Python extensions, so the
#     wheel gets a correct ABI tag and no .so is ever written into the source
#     tree by an install.
#   * cmake builds the CUDA executables, which are plain binaries copied into
#     susan/bin and shipped as package data.
# When the CUDA modules are disabled, cmake is not invoked at all: the install
# then needs only a C compiler and Cython, not the CUDA toolkit nor cmake.

import os
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel
from Cython.Build import cythonize


# The .pyx sources move arrays across the boundary as typed memoryviews (the
# PEP 3118 buffer protocol) and never touch the NumPy C API, so they need no
# NumPy headers and the compiled modules work with NumPy 1.x and 2.x alike.
EXT_MODULES = cythonize(
    [
        Extension("susan.utils._functions_core", ["susan/utils/_functions_core.pyx"]),
        Extension("susan.data._particles_core",  ["susan/data/_particles_core.pyx"]),
        Extension("susan.data._ptclsgeom_core",  ["susan/data/_ptclsgeom_core.pyx"]),
    ],
)


def _want_cuda():
    """Whether to build the CUDA executables.

    Set SUSAN_NO_CUDA=1 (or SUSAN_BUILD_CUDA=0) to install only the Python
    package and its Cython cores, for machines that read and analyse SUSAN
    files but never run the GPU modules.
    """
    if os.environ.get("SUSAN_NO_CUDA", "").lower() in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("SUSAN_BUILD_CUDA", "").lower() in ("0", "false", "no", "off"):
        return False
    return True


def _cmake_build():
    """Build the CUDA executables into susan/bin, unless they are disabled."""
    if not _want_cuda():
        print("SUSAN: CUDA modules disabled, building the Python package only.")
        return
    root_dir  = os.path.abspath(os.path.dirname(__file__))
    build_dir = os.path.join(root_dir, "bin")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(["cmake", root_dir], cwd=build_dir)
    subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)


class CMakeBuild(build_py):
    def run(self):
        _cmake_build()
        super().run()

class CMakeBuildEditable(editable_wheel):
    def run(self):
        _cmake_build()
        super().run()


setup(
    packages=find_packages(include=["susan*"]),
    ext_modules=EXT_MODULES,
    cmdclass={
        "build_py":       CMakeBuild,
        "editable_wheel": CMakeBuildEditable,
    },
)
