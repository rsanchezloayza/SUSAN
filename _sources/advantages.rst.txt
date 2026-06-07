Why SUSAN?
==========

This page covers the practical advantages that follow from SUSAN's
projection-domain design, how SUSAN compares to other available tools, and
what resolutions have been achieved on benchmark datasets. The mathematical
foundations are described in :doc:`cryoet`.


Native projection-domain design
--------------------------------

Most high-resolution StA pipelines were originally designed around
reconstructed subtomograms and have since been extended with per-projection
refinement capabilities. SUSAN takes a different approach: every component,
from CTF estimation and alignment to CTF refinement and averaging, was
implemented from scratch to operate natively on 2D tilt-series projections.
No subtomogram or tomogram is constructed at any point in the refinement loop.

This architectural commitment has direct consequences for both performance and
accuracy. Because substacks are compact 2D representations, the entire
processing pipeline is structured around them: substacks are extracted
on-the-fly from preloaded tilt stacks in RAM, transferred directly to GPU
memory, and processed without materialising any 3D volume. The same native
design also ensures that the alignment metric (pCC and cFSC) is not adapted
from a different context but derived directly from the projection-domain
formulation of the StA problem.


Performance and scalability
----------------------------

**Substack-native design eliminates redundant I/O.**
Because SUSAN extracts substacks on-the-fly from the original tilt series, it
never writes subtomograms or full tomograms to disk during the refinement loop.
This removes the dominant I/O and storage bottleneck of conventional pipelines
and can reduce storage requirements by one to two orders of magnitude on large
*in situ* datasets.

**Coalesced heterogeneous pipeline.**
SUSAN maintains a streaming pipeline in which disk I/O, host-to-device
transfer, and GPU computation proceed concurrently. A dedicated loader thread
preloads the next tilt stack into RAM while a thread pool processes substacks
from the current stack; within each node, one thread crops and uploads
substacks to the GPU while a second runs the GPU kernels. This design hides
latency at every level and keeps GPUs near-continuously utilised. It is
possible because substacks are small enough to remain memory-resident
throughout processing, which is not feasible for large 3D subtomograms.

**Reduced algorithmic complexity.**
The projected cross-correlation (pCC) reduces alignment complexity from
:math:`\mathcal{O}(N^3 \log N)` to
:math:`\mathcal{O}(N_\text{proj} N^2 \log N + L^3)`. In practice this yields
runtime reductions of one to two orders of magnitude compared with 3D
cross-correlation at equivalent accuracy.

**GPU-accelerated and distributed.**
All performance-critical kernels run on NVIDIA GPUs via CUDA. Intra-node
parallelism uses PThreads; cross-node distribution uses OpenMPI. Adding GPUs
or nodes yields near-linear scaling.

**Accessible on standard hardware.**
Because substacks are small, SUSAN processes large particle sets without
requiring the large GPU memory budgets that high-resolution 3D correlation
demands. The same pipeline that runs on a workstation with one or two consumer
GPUs scales transparently to a multi-node HPC cluster. The resulting reduction
in computational cost lowers the hardware barrier to high-resolution StA and
makes it feasible to reinvest the same resources into larger datasets,
higher-resolution targets, or broader multi-reference analyses.


Accuracy
---------

**Exact 3D cross-correlation.**
The pCC algorithm computes the 3D cross-correlation exactly from the
per-projection 2D cross-correlations, with no approximation introduced to the
alignment metric. This is a direct consequence of the commutativity of
cross-correlation with the projection and reconstruction operators; see
:doc:`cryoet` for the derivation.

**Per-projection CTF correction.**
Because substacks are extracted directly from the raw tilt series, the CTF is
applied and refined per projection. This is the natural domain for
phase-contrast correction and correctly handles depth-dependent defocus
variation across the specimen without requiring a separately reconstructed
CTF-corrected tomogram.

**Spectral robustness via cFSC.**
The cumulative Fourier Shell Correlation (cFSC) replaces the raw pCC metric
with a shell-wise normalised similarity score that is invariant to any
isotropic spectral envelope, including CTF modulation and radiation-damage
amplitude decay. Alignment is therefore driven by the signal-to-noise ratio
per Fourier shell rather than by the overall spectral amplitude, making it
robust to the spectral coloring common in CryoET data.

**Robustness to crowded environments.**
Cross-correlation is a linear operator, so the contribution of any
overlapping neighbouring particle is additive and separable from the target
particle's signal. In a dataset where neighbouring particles adopt many
orientations, their aggregate contribution to the cross-correlation volume is
diffuse and does not bias the alignment peak. This makes SUSAN naturally
robust to the dense packing common in *in situ* CryoET data, without
requiring explicit masking of the local environment during alignment.

**Benchmark results.**

.. list-table::
   :header-rows: 1
   :widths: 38 17 15 30

   * - Dataset
     - EMPIAR
     - Resolution
     - Notes
   * - Apoferritin
     - 11273
     - ~2.0 Å
     - Purified complex, octahedral symmetry
   * - HIV-Gag
     - 10164
     - ~3.0 Å
     - Viral assembly, C6 symmetry
   * - 70S ribosome (*M. pneumoniae*, *in situ*)
     - 10499
     - ~3.7 Å
     - Crowded *in situ* environment, matches state-of-the-art

Detailed benchmarks and comparisons are available in the associated
publications (see :doc:`overview` for references).


Modularity and flexibility
---------------------------

**Agent-based MACE framework.**
SUSAN decomposes subtomogram averaging into independent agents: volume
reconstruction, 3D alignment, 2D projection refinement, CTF refinement, and a
volume-prior agent (currently Noise2Noise). Agents communicate only through a
shared consensus volume and can be swapped, tuned, or skipped without touching
the others. Classical expectation-maximisation, plug-and-play priors, and
regularisation-by-denoising are all special cases of this unified framework.

**MAP-like regularisation priors.**
Configurable Gaussian priors on translational offsets, orientational angles,
and defocus values constrain the alignment and CTF refinement search,
stabilising convergence and preventing noise-driven parameter drift. These
priors are user-specified per run and require no modification to the
underlying agents.

**Integration-friendly design.**
SUSAN is built as a composable low-level framework rather than a monolithic
application. Individual agents, including reconstruction, alignment, CTF
refinement, and the denoising prior, can be driven independently from external
Python code without requiring a dedicated runtime environment, making it
straightforward to embed SUSAN within larger automated pipelines. The package
installs without conflicts in existing CryoET environments and complements
upstream preprocessing tools (motion correction, tomogram reconstruction) and
downstream analysis packages. Template matching directly against the tilt
series, where the same pCC components are repurposed for particle localisation,
is a concrete example of this flexibility; worked examples are included in the
tutorials.

**Noise2Noise denoiser.**
A lightweight self-supervised denoiser operating on half-maps is integrated as
the volume-prior agent within the MACE framework. It stabilises high-resolution
reconstructions within the iterative loop without requiring external training
data.

**Multi-reference alignment and classification.**
SUSAN natively supports multiple reference volumes within a single run.
Particles are assigned to references on a per-iteration basis, enabling
multi-reference alignment (MRA) and classification (MRC) without reprocessing
the tilt series.

**Multiresolution processing.**
Particle metadata is decoupled from tomogram geometry. The same particle set
can be processed at different binning levels by pointing to a different
tomogram metadata file; no data re-extraction is needed.

**Scriptable Python interface.**
Pipeline coordination, metadata management, and the MACE loop are implemented
in Python. A complete multi-resolution StA project can be expressed as a
short, reproducible Python script. The package installs with ``pip`` and does
not conflict with other CryoET software in the same environment.

**Minimal external dependencies.**
The Python layer requires only NumPy (≥ 1.20) and SciPy (≥ 1.6); PyTorch is
an optional dependency needed only for the Noise2Noise denoiser. Building from
source additionally requires Cython (≥ 3.0) and the standard C++/CUDA
compilers; these are build-time tools and are not needed after installation.
The compiled binaries require only the CUDA toolkit (which provides cuFFT)
and, optionally, OpenMPI for multi-node execution.


Similar works
--------------

Several other frameworks support high-resolution subtomogram averaging or
related CryoET tasks. The descriptions below focus on the relationship to
SUSAN; each tool represents a distinct set of design choices suited to
different workflows and use cases.

Projection-domain methods
~~~~~~~~~~~~~~~~~~~~~~~~~

These tools, like SUSAN, perform some or all of their computation directly on
tilt-series projections rather than on reconstructed subtomograms.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Software
     - Relationship to SUSAN
   * - `RELION <https://relion.readthedocs.io>`_
     - A leading general-purpose cryo-EM package. RELION 5 introduced a
       pseudo-subtomogram formulation that performs alignment directly on
       tilt-series projections using a per-projection Gaussian log-likelihood
       adapted from its single-particle engine. SUSAN's pCC derives the exact
       3D cross-correlation from projections via a different factorisation
       based on the linearity of the cross-correlation operator. The two
       methods represent distinct mathematical routes to projection-domain
       alignment, built on different software foundations.
   * - `M / WarpTools <https://www.warpem.com>`_
     - M performs multi-particle refinement by back-projecting directly from
       tilt-series frames, avoiding explicit tomogram reconstruction during
       refinement. M uses gradient-based optimisation rather than exhaustive
       correlation search. Both M and SUSAN work in the projection domain and
       avoid subtomogram storage; they differ in alignment algorithm (gradient
       descent vs. pCC) and framework origin.
   * - `nextPYP <https://nextpyp.app>`_  / BISECT / CSPT
     - An integrated end-to-end CryoET platform from Bartesaghi's group,
       covering preprocessing through final map. Its alignment engine
       (BISECT/CSPT) works with per-particle tilt projections and refines
       orientations by projection matching adapted from single-particle
       reconstruction software, aggregating per-projection scores by uniform
       averaging. nextPYP is designed as a complete turnkey system with
       emphasis on automation and scalability.
   * - `EMAN2 / EMAN3 <https://blake.bcm.edu/emanwiki/EMAN2>`_
     - A long-established cryo-EM package with comprehensive CryoET support.
       EMAN's subtilt refinement mode refines particle orientations using
       individual tilt projections, quality-weighting each tilt's contribution,
       by adapting EMAN's single-particle projection-matching framework to the
       tomographic context.

Subtomogram-based methods
~~~~~~~~~~~~~~~~~~~~~~~~~

These tools perform alignment on reconstructed subtomograms extracted from
tomograms. They remain the most widely used approach and provide a rich set
of features for geometry handling, classification, and workflow integration.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Software
     - Relationship to SUSAN
   * - `STOPGAP <https://github.com/wan-lab-vanderbilt/STOPGAP>`_
     - A widely used open-source MATLAB package for subtomogram alignment and
       classification, particularly popular for *in situ* data. STOPGAP
       operates on reconstructed subtomograms and provides fine-grained control
       over each step of the workflow.
   * - `GapStop <https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm>`_
     - A GPU-accelerated Python reimplementation of STOPGAP's template-matching
       routines (GAPSTOP_TM), focused on particle localisation from tomograms.
       GapStop targets template matching rather than the full
       alignment-and-averaging pipeline, and complements StA tools rather than
       replacing them.
   * - `Dynamo <https://dynamo-em.org>`_
     - A MATLAB-based StA framework with flexible geometry tools and an
       interactive working environment. Widely used for *in situ* data with
       irregular or complex particle arrangements. SUSAN's Python interface
       follows a similar scripting philosophy while replacing 3D operations
       with the pCC algorithm.
   * - `emClarity <https://github.com/bHimes/emClarity>`_
     - A high-resolution StA pipeline with per-particle CTF and geometry
       corrections applied during subtomogram reconstruction. emClarity
       achieves high resolutions on challenging datasets; storage requirements
       scale with particle count and box size.
