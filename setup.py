#!/usr/bin/env python
# Minimal setup.py — metadata lives in pyproject.toml.
# This file exists solely to plug in the cmake build step.

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel


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


def _cmake_build(root_dir):
    build_dir = os.path.join(root_dir, "bin")
    os.makedirs(build_dir, exist_ok=True)
    with_cuda = _want_cuda()
    if not with_cuda:
        print("SUSAN: CUDA modules disabled, building the Python package only.")
    subprocess.check_call(
        [
            "cmake",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DSUSAN_BUILD_CUDA={'ON' if with_cuda else 'OFF'}",
            root_dir,
        ],
        cwd=build_dir,
    )
    subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)


class CMakeBuild(build_py):
    def run(self):
        _cmake_build(os.path.abspath(os.path.dirname(__file__)))
        super().run()

class CMakeBuildEditable(editable_wheel):
    def run(self):
        _cmake_build(os.path.abspath(os.path.dirname(__file__)))
        super().run()


setup(
    packages=find_packages(include=["susan*"]),
    cmdclass={
        "build_py":      CMakeBuild,
        "editable_wheel": CMakeBuildEditable,
    },
)
