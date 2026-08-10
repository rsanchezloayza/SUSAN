<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="doc/source/_static/logo_dark.svg">
    <img alt="SUSAN logo" src="doc/source/_static/logo_light.svg" width="400">
  </picture>
</p>

# SubStack Analysis: High performance Subtomogram Averaging

> [!NOTE]
> The documentation is being actively updated in preparation for the publication of the paper. Please check the latest version at [https://rsanchezloayza.github.io/SUSAN](https://rsanchezloayza.github.io/SUSAN).

## Contents
- [Description](#description)
- [Building and setup instructions](#building-and-setup-instructions)
  - [Dependencies](#dependencies)
  - [Setup and compilation](#initial-setup-and-compilation)
  - [`Python` setup](#python-setup)
  - [`Matlab` setup](#matlab-setup)
- [Installing in a `conda` environment (`Python`)](#installing-susan-in-a-conda-environment-for-python)
- [Tutorial](#tutorial)

## Description
`SUSAN` is a low-level/mid-level framework for fast Subtomogram Averaging (StA) for CryoET. It uses *susbtacks* instead of *subtomograms* that are cropped *on-the-fly* from the aligned stacks to reduce the computational complexity and to increase the overall performace of the StA pipeline.

`SUSAN` was designed to be modular, flexible and fast. It is conformed by two layers:
- **Low-level layer**: Set of executables that perform the demanding computations. They were written in `C++` using a minimal set of dependencies: `Eigen`, as a header-only mathemetical engine, `CUDA` for GPU acceleration, and `PThreads` for lightweight multi-threading. Optionally, they can be built with [`MPI`](https://en.wikipedia.org/wiki/Message_Passing_Interface) support to run in multi-node environments.
- **Mid-level layer**: Set of wrappers to the previous layer that simplify its use and provides a set of non time-critical operations. It is used to create the workflows or pipelines as scripts with wrappers for `Matlab` and for `Python`. The `Matlab` one was designed to complement [DYNAMO](https://wiki.dynamo.biozentrum.unibas.ch/w/index.php/Main_Page), while the `Python` one is provided to enable integration to other pipelines based on this language.

### About
I started the development of `SUSAN` at the [Independent Research Group (Sofja Kovaleskaja) of Dr. Misha Kudryashev](https://www.biophys.mpg.de/2149775/members) at the Department of Structural Biology [Max Planck Institute of Biophysics (MPIBP)](https://www.biophys.mpg.de/en) in Frankfurt am Main, Germany. Currently, I am an ARISE fellow at the [Kreshuk group](https://www.embl.org/groups/kreshuk/members/) at the [European Molecular Biology Laboratory (EMBL)](https://www.embl.org/) in Heidelberg, Germany. Dr. Misha Kudryashev has a new [group](https://www.mdc-berlin.de/kudryashev) at the [Max Delbrück Center of Molecular Medicine (MDCMM)](https://www.mdc-berlin.de/) in Berlin, Germany.

`SUSAN` is an Open Source project ([AGPLv3.0](LICENSE))

## Building and setup instructions
### Dependencies
- `CUDA`.
- `gcc`.
- `cmake`.
- `git`.
#### Optional
- `OpenMPI`, or equivalent, for multi-node support
- `Matlab`

### Initial setup and compilation
We assume that `SUSAN` will be installed in the `LOCAL_SUSAN_PATH` folder (`LOCAL_SUSAN_PATH` can be `/home/user/Software/`, for example)
1. Install the desired dependencies.
2. Clone `SUSAN` to `LOCAL_SUSAN_PATH`:
   ```
   cd LOCAL_SUSAN_PATH
   git clone https://github.com/rsanchezloayza/SUSAN
   ```
3. Compile `SUSAN`:
   ```
   mkdir bin
   cd bin
   cmake ../
   make -j
   ```
   **Note:** The `cmake` procedure detects the availabilty of `OpenMPI` and `Matlab` and compiles their functionalities accordingly.
   
   **HINT:** You can use ``` cmake ../ -DCMAKE_CUDA_COMPILER=$(which nvcc) ``` in order to avoid problems with CMake and Cuda.

### `Python` setup
#### Dependencies
Besides the standard libraries, the `SUSAN` module for `Python` has only two dependencies: [`NumPy`](https://numpy.org/) and [`Numba`](https://numba.pydata.org/). Either `NumPy` 1.x or 2.x works, and a build made against one runs against the other. Install them if needed:
- Using [`conda`](https://conda.io) (or equivalent):
  ```
  conda install numpy numba
  ```
- Using [`pip`](https://pypi.org/)
  ```
  pip install numpy numba
  ```

#### Option 1: Using `susan` without installation
The module contains a few `Cython` cores that must be compiled in place first.
In `LOCAL_SUSAN_PATH`:
```
python setup.py build_ext --inplace
```
This only needs `Cython` and a C compiler: no `CUDA`, no `cmake`, and no
`NumPy` headers. Then `LOCAL_SUSAN_PATH` must be added to path and the `susan`
module can be imported. On the `Python` command line, or on a `Python` script:
```
import sys
sys.path.insert(1,'LOCAL_SUSAN_PATH')
import susan
```

#### Option 2: Install `susan` in the current `Python` environment
After executing the optional steps for the `Python` module, it can be installed using [`pip`](https://pypi.org/) and the provided [setup.py](setup.py) file. In the  `LOCAL_SUSAN_PATH` execute:
```
pip install .
```
After this step, the module `susan` should be available to be imported.

#### Option 3: Install `susan` without the CUDA modules
On a machine that only reads SUSAN files and analyses results (a laptop, a
cluster login node), the GPU executables are not needed and the CUDA toolkit
may not even be available. Set `SUSAN_NO_CUDA=1` at install time:
```
SUSAN_NO_CUDA=1 pip install .
SUSAN_NO_CUDA=1 pip install git+https://github.com/rsanchezloayza/SUSAN.git
```
This still compiles the small Cython cores (only a C compiler is needed) so
file I/O and geometry keep their normal speed, but it skips `susan_aligner`,
`susan_reconstruct` and the rest. `cmake` is not invoked at all in this mode,
so neither `cmake` nor the `CUDA` toolkit has to be present. `import susan` then emits a
warning that the executables are missing; everything except `susan.modules`
works. Use `susan.has_binaries()` to check at runtime.

### `Matlab` setup
`LOCAL_SUSAN_PATH` must be added to path. On the `Matlab` command line, or on a `Matlab` script:
```
addpath LOCAL_SUSAN_PATH
```

## Installing `SUSAN` in a `conda` environment (for `Python`)
`SUSAN` can be built and installed inside a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):

-  **(Optional)** Create a new `conda` environment and activate it:
   ```
   conda create -n susan_env
   conda activate susan_env
   ```
1. Install the basic packages needed for building and using `SUSAN`
   ```
   conda install -c conda-forge git cmake make cudatoolkit-dev=11 gxx=10 numpy numba
   ```
-  **(Optional)** Install `openmpi`:
   ```
   conda install -c conda-forge openmpi
   ```
-  **(Optional)** Install packages to run the examples and tutorials:
   ```
   conda install -c conda-forge jupyter scipy matplotlib scikit-image
   ```
2. Go to the directory `LOCAL_SUSAN_PATH`, where `SUSAN` will be compiled. For example, `LOCAL_SUSAN_PATH` can be `~/Software/`:
   ```
   cd LOCAL_SUSAN_PATH
   ```
3. Clone `SUSAN`, compile it and install it:
   ```
   git clone https://github.com/rsanchezloayza/SUSAN
   mkdir SUSAN/bin
   cd SUSAN/bin
   cmake ../
   make -j
   pip install ../
   ```
After these step the module `susan` should be available on the current environment.

## Tutorial
A tutorial is available for `Python` and `Matlab` for the `mixedCTEM` dataset from the [EMPIAR-10064](https://www.ebi.ac.uk/empiar/EMPIAR-10064/). It is assumed that the `wget` and `gunzip` commands and the `IMOD` framework are installed in the system.

### Preparing the data
1. Download the dataset (uses `wget`):
   ```
   cd LOCAL_SUSAN_PATH/tutorials/empiar_10064/data
   ./download_data.sh
   ```
2. Create the aligned stacks (uses `IMOD`):
   ```
   ./create_binned_aligned_stacks.sh
   ```
3. Uncompress the initial reference (uses `gunzip`):
   ```
   cd LOCAL_SUSAN_PATH/tutorials/empiar_10064
   gunzip emd_3420_b4.mrc.gz
   ```

### Running the Tutorial
Depending on the system setup:
- For `Matlab` use [workflow.m](tutorials/empiar_10064/workflow.m).
- For `Python` use [workflow.ipynb](tutorials/empiar_10064/workflow.ipynb) (as a [Jupyter Notebook](https://jupyter.org/install)).






