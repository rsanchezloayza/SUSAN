Installing SUSAN
================

SUSAN's Python layer installs with ``pip``; the pip build step compiles the
C++/CUDA engine automatically via CMake. The only mandatory system requirement
is an NVIDIA GPU with a supported CUDA toolkit (≥ 10, ≤ 13). All other
requirements depend on the chosen installation path.

.. list-table::
   :header-rows: 1
   :widths: 35 25 30 10

   * - Method
     - Best for
     - Requires
     - Difficulty
   * - :ref:`install-quick`
     - Fresh system, all tools
     - conda only
     - Easiest
   * - :ref:`install-pip`
     - Existing Python env
     - System cmake/CUDA/gcc
     - Easy
   * - :ref:`install-conda-system`
     - New conda env, system tools
     - System cmake/CUDA/gcc
     - Easy
   * - :ref:`install-conda-devel`
     - New conda env, self-contained
     - conda only
     - Moderate
   * - :ref:`install-hpc`
     - SLURM clusters
     - Cluster modules
     - Moderate
   * - :ref:`install-source`
     - Custom builds, MATLAB
     - System cmake/CUDA/gcc
     - Advanced


.. _install-quick:

Quick install
-------------

The fastest path to a fully featured installation on a fresh system. No
system compilers or CUDA installation required; everything is managed by
conda.

.. code-block:: bash

   conda env create --file envs/environment_full.yml
   conda activate susan-full
   conda env update --file envs/environment_susan_devel_cuda12.yml

This creates an environment with Jupyter, Spyder, PyTorch, and all analysis
tools, then compiles and installs SUSAN using conda-managed cmake, g++, and
the CUDA 12 toolkit.


Prerequisites
-------------

The following tools must be on ``PATH`` before installing (except for the
self-contained conda paths, which manage them via conda):

* **CUDA toolkit** ≥ 10 and ≤ 13 (provides ``nvcc`` and ``cuFFT``).
* **CMake** ≥ 3.14.
* **gcc / g++** compatible with the installed CUDA version.
* **git** (for cloning or pip-from-git installs).
* **OpenMPI** (optional): detected automatically by CMake; enables multi-node
  execution if present.
* **MATLAB** (optional): detected automatically by CMake; enables the MATLAB
  interface if present.

.. note::

   Eigen3 and LodePNG are fetched automatically by CMake during the build if
   not already present on the system; no manual installation is needed.


.. _install-pip:

pip into an existing environment
---------------------------------

If cmake, gcc, and the CUDA toolkit are already on ``PATH``, SUSAN installs
directly into any active Python environment.

**Core install:**

.. code-block:: bash

   pip install "git+https://github.com/rsanchezloayza/SUSAN"

**Optional extras:**

* PyTorch (required for the Noise2Noise denoiser):

  .. code-block:: bash

     pip install "SUSAN[ml] @ git+https://github.com/rsanchezloayza/SUSAN"

* Analysis tools (scikit-image, scikit-learn, numba, bm4d):

  .. code-block:: bash

     pip install "SUSAN[analysis] @ git+https://github.com/rsanchezloayza/SUSAN"

* Both:

  .. code-block:: bash

     pip install "SUSAN[full] @ git+https://github.com/rsanchezloayza/SUSAN"


.. _install-conda-system:

conda environment with system build tools
------------------------------------------

Use this path when cmake, gcc, and the CUDA toolkit are already available
system-wide (most Linux workstations with an NVIDIA driver installed).

**Step 1 — create and activate the working environment:**

.. tab-set::

   .. tab-item:: Jupyter / VS Code

      .. code-block:: bash

         conda env create --file envs/environment_jupyter.yml
         conda activate susan-jupyter

   .. tab-item:: Spyder

      .. code-block:: bash

         conda env create --file envs/environment_spyder.yml
         conda activate susan-spyder

   .. tab-item:: Jupyter + Spyder

      .. code-block:: bash

         conda env create --file envs/environment_full.yml
         conda activate susan-full

**Step 2 — install SUSAN:**

.. code-block:: bash

   conda env update --file envs/environment_susan.yml

.. note::

   The install overlay (``environment_susan.yml``) carries only NumPy, SciPy,
   and the pip install step. All other packages (PyTorch, scikit-image, etc.)
   are already provided by the context environment from Step 1.

.. note::

   The conda environment files use ``nodefaults`` to avoid the Anaconda
   default channel, which has commercial licensing restrictions. Packages are
   resolved exclusively from ``pytorch`` and ``conda-forge``.


.. _install-conda-devel:

Self-contained conda environment (CUDA 12)
-------------------------------------------

Use this path when cmake, gcc, or the CUDA toolkit are not available
system-wide (e.g., a fresh workstation without a system CUDA installation).

**Step 1 — create and activate the working environment:**

.. tab-set::

   .. tab-item:: Jupyter / VS Code

      .. code-block:: bash

         conda env create --file envs/environment_jupyter.yml
         conda activate susan-jupyter

   .. tab-item:: Spyder

      .. code-block:: bash

         conda env create --file envs/environment_spyder.yml
         conda activate susan-spyder

   .. tab-item:: Jupyter + Spyder

      .. code-block:: bash

         conda env create --file envs/environment_full.yml
         conda activate susan-full

**Step 2 — install SUSAN with conda-managed build tools:**

.. code-block:: bash

   conda env update --file envs/environment_susan_devel_cuda12.yml

This installs cmake, make, g++, and the CUDA 12 toolkit from ``conda-forge``
before invoking the pip build step. No system compilers or CUDA installation
are required.


.. _install-hpc:

HPC / SLURM clusters
---------------------

On cluster systems, compilers and CUDA are provided by the module system.

**Step 1 — create and activate the working environment:**

.. code-block:: bash

   conda env create --file envs/environment_hpc.yml
   conda activate susan-hpc

**Step 2 — load cluster modules and install SUSAN:**

.. code-block:: bash

   module load cuda cmake gcc        # exact names vary by cluster
   module load openmpi               # optional, for multi-node support
   pip install "git+https://github.com/rsanchezloayza/SUSAN"

.. note::

   If the PyTorch bundled by the ``pytorch`` conda channel does not match the
   cluster's CUDA version, install it separately after SUSAN:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cu<VER>

   Replace ``<VER>`` with the numeric CUDA version (e.g., ``cu121`` for
   CUDA 12.1).


.. _install-source:

Building from source (custom configurations and MATLAB)
--------------------------------------------------------

Manual compilation is needed when you require a custom CMake configuration,
want to build without pip, or need the MATLAB interface.

**Clone and compile:**

.. code-block:: bash

   git clone https://github.com/rsanchezloayza/SUSAN
   cd SUSAN
   mkdir bin && cd bin
   cmake ../
   make -j

.. hint::

   If CMake cannot locate ``nvcc`` automatically, pass it explicitly:

   .. code-block:: bash

      cmake ../ -DCMAKE_CUDA_COMPILER=$(which nvcc)

CMake automatically detects OpenMPI and MATLAB and compiles their respective
targets if found.

**Install the Python package:**

.. tab-set::

   .. tab-item:: Standard

      .. code-block:: bash

         pip install .

   .. tab-item:: Editable (development)

      .. code-block:: bash

         pip install -e .

**MATLAB interface:**

No extra steps are required. If MATLAB is found by CMake, the MEX files are
compiled and placed in the ``+SUSAN/`` directory automatically. Add the
repository root to the MATLAB path:

.. code-block:: matlab

   addpath /path/to/SUSAN
