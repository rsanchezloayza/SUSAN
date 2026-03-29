#!/usr/bin/env python
# Minimal setup.py — metadata lives in pyproject.toml.
# This file exists solely to plug in the cmake build step.

import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel


def _cmake_build(root_dir):
    build_dir = os.path.join(root_dir, "bin")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(["cmake", root_dir], cwd=build_dir)
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
