Basic Concepts
==============

This page introduces the key ideas behind SUSAN and the abstractions the
framework uses: how data is organised, what the processing modules do, and
how they compose into a typical workflow.  Mathematical derivations are in
:doc:`cryoet`; practical advantages are in :doc:`advantages`.


From specimen to structure
--------------------------

A CryoET experiment records a *tilt series*: a set of 2D images of the same
specimen area acquired at different angles. A *tomogram* is the 3D
reconstruction of that volume. Individual copies of the target complex
(*particles*) are identified in the tomogram and cropped out as *subtomograms*,
which are then aligned and averaged to produce a high-resolution density map.

SUSAN skips the subtomogram step. Instead of reconstructing 3D boxes, it
works directly with *substacks*: for each particle, the 2D image patches
cropped from the tilt stacks together with all associated tilt geometry and
CTF parameters. Substacks are assembled on-the-fly during processing; no
subtomogram is ever written to disk. The alignment and averaging are performed
in the projection domain, which is faster, more memory-efficient, and avoids 
the errors introduced by tomogram reconstruction. See :doc:`cryoet` for an
illustration and the mathematical basis.


Key terms
---------

**Particle**
   One identified copy of the target complex in a tilt series, described by
   its position, orientation, and CTF parameters.

**Substack**
   The complete data unit for one particle: the 2D image patches cropped
   from the tilt stacks, together with the per-projection tilt geometry,
   CTF parameters, and the particle's 3D alignment. Everything needed to
   align and reconstruct a particle is carried in its substack.

**Reference**
   The current best-guess 3D density map of the target complex, used as the
   alignment target in each iteration.

**Iteration**
   One cycle of alignment followed by reconstruction; the reference improves
   with each cycle until it converges.

**Resolution**
   Assessed by Fourier Shell Correlation (FSC) between two independent
   half-dataset reconstructions; the 0.143 threshold is reported in
   Ångströms.


Data organisation
-----------------

SUSAN organises all project data through three file types. The Python API
provides a class for each; the compiled binaries read them directly.

Tomogram information — ``.tomostxt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One ``.tomostxt`` file describes a complete set of tomograms **at one
binning level**.  A multiresolution project keeps one file per level.
Each entry stores the paths to the tilt-series stacks, the per-projection
tilt geometry (rotation matrices and translations), per-projection CTF
parameters (defocus, astigmatism, voltage, spherical aberration, amplitude
contrast), and tomogram geometry (pixel size, dimensions).

The file is plain text and can be inspected or edited manually. In Python,
use :class:`susan.data.Tomograms`.

Reference information — ``.refstxt``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A lightweight description of the reference volumes used in one alignment
iteration: paths to the maps, their masks, and the half-maps used for
resolution-dependent weighting and Noise2Noise training. A ``.refstxt`` can
hold one reference (single-reference StA) or several (multi-reference
alignment and classification, MRA/MRC), with each particle assigned to one
reference per iteration. SUSAN creates a new ``.refstxt`` at the end of
each iteration pointing to the freshly reconstructed maps. In Python, use
:class:`susan.data.Reference`.

Particle information — ``.ptclsraw``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A compact binary file holding all metadata for every particle in the
project. Binary storage is used to keep file sizes small on large datasets
and to preserve full floating-point precision for rotation matrices, defocus
values, and sub-pixel translations, none of which round-trip safely through
plain text. Each record contains:

* **identifiers** — tomogram ID, particle ID, reference (class) ID;
* **3D alignment** — per-reference rotation and translation
  (:math:`\mathcal{T}_i^v`);
* **per-projection 2D refinement** — in-plane shifts and rotations;
* **per-projection CTF** — individual defocus and astigmatism values.

Because coordinates are stored in Ångströms and geometry lives in
``.tomostxt``, the same ``.ptclsraw`` file works unchanged at any binning
level; multiresolution processing requires no re-extraction. In Python,
use :class:`susan.data.Particles`.

All three Python classes expose their data as NumPy arrays, making it
straightforward to inspect parameter distributions, compute summary
statistics, and filter or select particle subsets.


Processing modules
------------------

All computationally intensive work is performed by compiled C++/CUDA
binaries invoked through Python wrapper classes in :mod:`susan.modules`.

:class:`~susan.modules.Aligner`
   The alignment engine. Extracts substacks on-the-fly, computes the
   projected cross-correlation (pCC) for each candidate rotation, and
   writes updated 3D orientations and 2D in-plane refinements back to the
   ``.ptclsraw`` file. Supports full 3D and 2D-only search modes,
   multi-reference alignment, and CTF-weighted scoring. Configurable
   Gaussian priors on translational offsets and orientational angles
   regularise the search, preventing noise-driven parameter drift.

:class:`~susan.modules.Averager`
   The reconstruction engine. Reads the current particle orientations and
   accumulates substack projections into 3D Fourier-space numerator and
   denominator volumes weighted by the per-projection CTF; dividing the two
   yields a CTF-deconvolved map via Wiener inversion (Direct Fourier
   Reconstruction). In multi-reference projects all classes are
   reconstructed in a single pass through the data. The resulting half-maps
   are used for FSC estimation and as input to Noise2Noise.

:class:`~susan.modules.CtfEstimator`
   Estimates the CTF from the raw tilt series. Uses a CryoET-specific
   algorithm that handles the strong defocus gradients present in tilted
   projections of thick *in situ* specimens, where different depths in the
   specimen are at different defocus values. Results are saved as a
   per-tomogram ``defocus.txt`` file and can be loaded back to update the
   ``.tomostxt`` file.

:class:`~susan.modules.CtfRefiner`
   Refines per-particle defocus values using the current high-SNR average
   as a reference, improving the high-frequency content of subsequent
   reconstructions. Configurable Gaussian priors on translational offsets
   and defocus values regularise the refinement, preventing noise-driven
   drift of the per-particle CTF parameters.

:class:`~susan.modules.SubtomoRec`
   Reconstructs explicit subtomograms from the tilt series using the
   current particle parameters. Useful for visualisation, downstream
   analyses, and interfacing with tools that expect subtomogram files.

:class:`~susan.modules.CropProjection`
   A utility that crops individual projections for diagnostics or external
   tools.

The high-level project class :class:`susan.project.SubtomoAvg` wraps all of these
modules and the iterative loop into a single interface.


The iterative refinement loop (MACE)
-------------------------------------

SUSAN solves the subtomogram averaging problem by *alternation*: at each
iteration, the reference, particle orientations, and CTF parameters are each
updated in turn while the others are held fixed. This is implemented as the
**MACE** (Multi-Agent Consensus Equilibrium) framework, in which each update
step is an independent *agent* that can be configured, replaced, or skipped
without affecting the others. Classical expectation-maximisation, plug-and-play
denoising priors (Noise2Noise), and CTF refinement are all special cases of
this unified loop.


Typical workflow
----------------

A high-resolution StA project with SUSAN follows these steps:

1. **Set up metadata.**
   Create a :class:`~susan.data.Tomograms` object pointing to the tilt
   series, populate tilt angles and microscope parameters, and save it as
   ``.tomostxt``. Create a :class:`~susan.data.Particles` object with
   initial particle coordinates and save it as ``.ptclsraw``.

2. **Estimate initial CTF.**
   Run :class:`~susan.modules.CtfEstimator` to fill in per-projection
   defocus values. These become the starting point for all subsequent
   refinement.

3. **Prepare the initial reference.**
   Either import an existing density map or generate a random-phase initial
   model. Create a ``.refstxt`` pointing to the map and its mask.

4. **Iterative refinement (MACE loop).**
   Each iteration runs the following agents in sequence:

   a. :class:`~susan.modules.Aligner` and/or :class:`~susan.modules.CtfRefiner`:
      update 3D orientations, 2D in-plane refinements, and CTF parameters,
      depending on which agents are active in the current iteration.
   b. :class:`~susan.modules.Averager`: reconstruct updated half-maps and
      full map.
   c. *(optional)* Noise2Noise: denoise the map using the half-maps as a
      self-supervised training signal. It can be replaced by any other
      volume prior, or skipped entirely.

5. **Post-processing.**
   FSC curves, map sharpening, and masking can be applied to the final
   half-maps with standard tools.

At the end of each iteration, SUSAN writes the reconstructed full map and
half-maps, and an updated ``.ptclsraw`` with refined particle parameters.
Metadata is checkpointed to disk so a workflow can be interrupted and
resumed at any iteration.


Quick example
-------------

The following snippet sets up a minimal single-reference StA project,
one tomogram, estimates CTF, and runs five 3D refinement iterations:

.. code-block:: python

   import susan
   import numpy as np

   # --- Build tomogram metadata ---
   tomos = susan.data.Tomograms(n_tomo=1, n_proj=61)
   tomos.tomo_id[0]   = 1
   tomos.set_stack(0,  'tilt_series.mrc')
   tomos.set_angles(0, 'tilt_series.tlt')
   tomos.pix_size[0]  = 2.62              # Å/pixel
   tomos.tomo_size[0] = (3710, 3710, 880) # voxels
   tomos.save('tomos_raw.tomostxt')

   # --- Estimate CTF ---
   ctf_grid = susan.data.Particles.grid_2d(tomos, step_pixels=256)  # half-box overlap with box_size=512
   ctf_grid.save('ctf_grid.ptclsraw')

   ctf_est = susan.modules.CtfEstimator()
   ctf_est.list_gpus_ids = [0, 1]
   tomos = ctf_est.estimate('data/', 'tomos_raw.tomostxt', 'ctf_grid.ptclsraw',
                             box_size=512, tomos_out='tomos.tomostxt')

   # --- Import picked particles and create the initial reference ---
   coords   = np.loadtxt('picked_coords.txt')   # Nx3 array, XYZ positions in voxels at the same binning level as the tomogram
   tomo_ids = np.ones(len(coords))
   ptcls = susan.data.Particles.import_data(tomograms=tomos,
                                             position=coords,
                                             tomos_id=tomo_ids,
                                             randomize_angles=True)
   ptcls.save('particles.ptclsraw')

   refs = susan.data.Reference(n_refs=1)
   refs.ref[0] = 'initial_ref.mrc'
   refs.msk[0] = 'mask.mrc'
   refs.save('project.refstxt')

   # --- Configure and run the StA loop ---
   # Simplest MACE loop: Aligner then Averager each iteration,
   # equivalent to classical StA with per-projection CTF correction.
   mngr = susan.project.SubtomoAvg('my_project', 128)
   mngr.initial_reference = 'project.refstxt'
   mngr.initial_particles = 'particles.ptclsraw'
   mngr.tomogram_file     = 'tomos.tomostxt'
   mngr.list_gpus_ids     = [0, 1]

   mngr.aligner.bandpass.lowpass = 30
   mngr.aligner.set_angular_search(360, 18, 360, 18)
   mngr.aligner.set_offset_search(6)
   # Multi-resolution angular refinement, equivalent to Dynamo's
   # high_convergence: each level halves the step over ±factor·step range.
   mngr.aligner.refine.levels = 4
   mngr.aligner.refine.factor = 2

   for i in range(1, 6):  # iterations are 1-indexed; range(5) would start at 0 and fail
       mngr.run_iteration(i)

Each iteration's results are stored in ``my_project/ite_XXXX/`` and contain
the reconstructed map, half-maps, and updated ``.ptclsraw`` file.

.. seealso::

   :doc:`cryoet` — mathematical foundations of the pCC and cFSC algorithms.

   :doc:`advantages` — practical advantages and benchmark results.

   :doc:`examples` — annotated, runnable examples for common use cases.

   :doc:`tutorials` — step-by-step tutorials on published benchmark datasets.
