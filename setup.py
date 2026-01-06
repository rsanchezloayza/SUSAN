#!/usr/bin/env python

import os
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py

class CMakeBuild(build_py):
    def run(self):
        root_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(root_dir, "bin")
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(["cmake", root_dir],cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)
        super().run()

setup(name='susan',
	version='0.5',
	description='SUSAN framework for CryoET subtomogram averaging',
	author='Ricardo Miguel Sánchez Loayza',
	author_email='ricardo.sanchez@embl.de',
	url='https://github.com/rkms86/SUSAN',
	packages=['susan','susan.data','susan.io','susan.utils','susan.project','susan.utils.txt_parser'],
	package_dir={'susan': 'susan'},
	package_data={'susan': ['bin/susan*']},
	install_requires=['numpy>=1.20.0','scipy>=1.6.0','numba'],
	zip_safe=False,
	cmdclass={"build_py": CMakeBuild,},

)


