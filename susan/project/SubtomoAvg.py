###########################################################################
# This file is part of the Substack Analysis (SUSAN) framework.
# Copyright (c) 2018-2021 Ricardo Miguel Sanchez Loayza.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
###########################################################################

from __future__ import annotations

import numpy as _np
import warnings as _warnings

from scipy.spatial      import KDTree as _KDTree

import susan.data    as _ssa_data
import susan.io      as _ssa_io
import susan.utils   as _ssa_utils
import susan.modules as _ssa_modules
import susan.solvers as _ssa_solvers

from susan.io.mrc import read     as _mrc_read
from susan.io.mrc import write    as _mrc_write
from susan.io.mrc import get_info as _mrc_get_info

import susan.utils.datatypes  as _dt
import susan.utils.txt_parser as _prsr

from os      import remove as _rm
from os      import mkdir  as _mkdir
from os.path import exists as _file_exists


###########################################################################
# Internal helpers
###########################################################################

def _ml_class_weights(cc, beta, floor=1e-3):
    """Soft ML classification weights from per-class CC (``n_refs x n_ptcl``).

    Responsibilities are a softmax at inverse-temperature *beta*; each particle
    is then scaled by a smooth evidence gate (a logistic on its log-sum-exp
    over classes) so low-signal particles fade toward *floor* rather than being
    cut.  Returns the reweighted CC, same shape as *cc*.
    """
    if beta <= 0:
        raise ValueError('reweight_classification (beta) must be > 0')
    a     = beta * cc
    m     = a.max(axis=0)
    e     = _np.exp(a - m)
    Z     = e.sum(axis=0)
    gamma = e / Z
    q     = (m + _np.log(Z)) / beta               # per-particle evidence
    scale = _np.median(_np.abs(q - _np.median(q)))
    if scale > 0:
        x = _np.clip((q - _np.quantile(q, 0.15)) / scale, -30.0, 30.0)
        s = floor + (1.0 - floor) / (1.0 + _np.exp(-x))
    else:
        s = _np.ones_like(q)
    return gamma * s[_np.newaxis, :]


class _IterationFiles:
    """File-path bundle for a single iteration."""

    def __init__(self):
        self.ptcl_rslt = ''
        self.ptcl_rec  = ''
        self.reference = ''
        self.ite_dir   = ''
        self.ite       = 0

    def check(self):
        if not _file_exists(self.ptcl_rslt):
            raise NameError('File ' + self.ptcl_rslt + ' does not exist')
        if not _file_exists(self.reference):
            raise NameError('File ' + self.reference + ' does not exist')


###########################################################################
# SubtomoAvgBase — project infrastructure and query interface
###########################################################################

class SubtomoAvgBase:
    """Project infrastructure: file paths and read-only query methods.

    Can be instantiated without *box_size* to inspect an existing project.

    Parameters
    ----------
    prj_name : str
        Project directory.  Created when *box_size* is supplied.
    box_size : int, optional
        Subvolume box size in pixels.  Reads ``info.prjtxt`` when omitted.
    """

    def __init__(self, prj_name, box_size=None):
        if box_size is None:
            fp = open(prj_name + '/info.prjtxt', 'r')
            args = _prsr.parse_args(fp)
            fp.close()
            self.prj_name = args['name']
            self.box_size = int(args['box_size'])
            # Optional fields: present in files written by SubtomoAvg,
            # absent in files written by the legacy STA class.
            tomofile  = args.get('tomogram_file')
            self.tomogram_file     = ''
            self._use_2d_stats  = int(args.get('use_2d_stats',0))
            self._use_ctf_stats = int(args.get('use_ctf_stats',0))
            self.initial_reference = args.get('initial_reference','')
            self.initial_particles = args.get('initial_particles','')
            if tomofile is not None:
                self.tomogram_file = tomofile
                self.pix_size = float(_ssa_data.Tomograms(tomofile).pix_size[0])
        else:
            if not _file_exists(prj_name):
                _mkdir(prj_name)
            self.prj_name          = prj_name
            self.box_size          = box_size
            self.tomogram_file     = ''
            self.initial_reference = ''
            self.initial_particles = ''
            self._use_2d_stats     = 0
            self._use_ctf_stats    = 0

    @property
    def use_2d_stats(self) -> int:
        return self._use_2d_stats

    @property
    def use_ctf_stats(self) -> int:
        return self._use_ctf_stats

    # ------------------------------------------------------------------
    # Resolution conversions
    # ------------------------------------------------------------------

    def A2fpix(self, angstroms) -> float:
        """Convert a resolution in angstroms to Fourier pixels.

        Uses :attr:`box_size` and :attr:`pix_size`.  A Fourier pixel *k*
        corresponds to a resolution of ``box_size * pix_size / k`` angstroms,
        so ``fpix = box_size * pix_size / angstroms``.

        Parameters
        ----------
        angstroms : float
            Resolution in angstroms.

        Returns
        -------
        float
            The corresponding radius in Fourier pixels.
        """
        return self.box_size * self.pix_size / angstroms

    def fpix2A(self, fpix) -> float:
        """Convert a radius in Fourier pixels to a resolution in angstroms.

        Uses :attr:`box_size` and :attr:`pix_size`.  Inverse of
        :meth:`A2fpix`: ``angstroms = box_size * pix_size / fpix``.

        Parameters
        ----------
        fpix : float
            Radius in Fourier pixels.

        Returns
        -------
        float
            The corresponding resolution in angstroms.
        """
        return self.box_size * self.pix_size / fpix

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def iteration_dir(self, ite) -> str:
        """Return the directory path for iteration *ite*.

        Parameters
        ----------
        ite : int

        Returns
        -------
        str
            ``<prj_name>/ite_NNNN/``
        """
        return self.prj_name + '/ite_%04d/' % ite

    def iteration_files(self, ite) -> _IterationFiles:
        """Return the standard file-path bundle for iteration *ite*.

        Parameters
        ----------
        ite : int
            Use ``0`` for the initial state.

        Returns
        -------
        _IterationFiles
        """
        rslt = _IterationFiles()
        if ite < 1:
            rslt.ptcl_rslt = self.initial_particles
            rslt.reference = self.initial_reference
        else:
            base = self.iteration_dir(ite)
            rslt.ptcl_rslt = base + 'particles.ptclsraw'
            rslt.ptcl_rec  = base + 'particles_rec.ptclsraw'
            rslt.reference = base + 'reference.refstxt'
            rslt.ite_dir   = base
            rslt.ite       = ite
        return rslt

    def path_map(self, ite, ref=1) -> str:
        """Path to the full reference map for iteration *ite*.

        Parameters
        ----------
        ite : int
        ref : int, optional
            1-based class index (default 1).

        Returns
        -------
        str
        """
        if ite == 0:
            info = _ssa_data.Reference(self.initial_reference)
            return info.ref[ref - 1]
        return self.iteration_dir(ite) + 'map_class%03d.mrc' % ref

    def path_halfmap(self, ite, ref=1) -> tuple[str, str]:
        """Paths to the two half-maps for iteration *ite*.

        Returns
        -------
        tuple of str
            ``(half1_path, half2_path)``
        """
        if ite == 0:
            info = _ssa_data.Reference(self.initial_reference)
            return (info.h1[ref - 1], info.h2[ref - 1])
        d = self.iteration_dir(ite)
        return (d + 'map_class%03d_half1.mrc' % ref,
                d + 'map_class%03d_half2.mrc' % ref)

    def path_mask(self, ite, ref=1) -> str:
        """Path to the soft mask for iteration *ite*.

        Returns
        -------
        str
        """
        if ite == 0:
            info = _ssa_data.Reference(self.initial_reference)
            return info.msk[ref - 1]
        info = _ssa_data.Reference(self.iteration_dir(ite) + 'reference.refstxt')
        return info.msk[ref - 1]

    def path_refstxt(self, ite) -> str:
        """Path to the ``.refstxt`` file for iteration *ite*.

        Returns
        -------
        str
        """
        return self.iteration_files(ite).reference

    def path_ptcls(self, ite) -> str:
        """Path to the ``.ptclsraw`` file for iteration *ite*.

        Returns
        -------
        str
        """
        return self.iteration_files(ite).ptcl_rslt

    def path_map_rec(self, ite) -> str:
        """Base path prefix used by the averager when reconstructing iteration *ite*.

        The averager appends ``_classNNN.mrc``, ``_classNNN_half1.mrc``, etc.
        to this prefix.

        Returns
        -------
        str
            ``<prj_name>/ite_NNNN/map``
        """
        return self.iteration_dir(ite) + 'map'

    # ------------------------------------------------------------------
    # Convenience loaders
    # ------------------------------------------------------------------

    def get_map(self, ite, ref=1) -> _np.ndarray:
        """Load and return the reference map for iteration *ite*.

        Returns
        -------
        numpy.ndarray
        """
        v, _ = _mrc_read(self.path_map(ite, ref))
        return v

    def get_mask(self, ite, ref=1) -> _np.ndarray:
        """Load and return the soft mask for iteration *ite*.

        Returns
        -------
        numpy.ndarray
        """
        v, _ = _mrc_read(self.path_mask(ite, ref))
        return v

    def get_ptcls(self, ite) -> _ssa_data.Particles:
        """Load and return the particles for iteration *ite*.

        Returns
        -------
        :class:`~susan.data.Particles`
        """
        return _ssa_data.Particles(self.path_ptcls(ite))

    def get_cc(self, ite, ref=1) -> _np.ndarray:
        """Per-particle CC scores for iteration *ite*, reference *ref*.

        Returns
        -------
        numpy.ndarray, shape (N,)
        """
        return self.get_ptcls(ite).ali_cc[ref - 1]

    def get_fsc(self, ite, ref=1) -> _np.ndarray:
        """Compute the FSC curve for iteration *ite*, reference *ref*.

        Returns
        -------
        numpy.ndarray
        """
        i = ref - 1
        refs = _ssa_data.Reference(self.path_refstxt(ite))
        return _ssa_utils.fsc_get(refs.h1[i], refs.h2[i], refs.msk[i])

    def map_change(self, ite, ref=1) -> float:
        """L2 norm of the voxel-wise difference between iterations *ite* and *ite-1*.

        Useful as a convergence monitor: a decreasing value indicates the
        reference is stabilising.

        Parameters
        ----------
        ite : int
            Iteration number (≥ 1).
        ref : int, optional
            1-based reference index.  Default: ``1``.

        Returns
        -------
        float
        """
        return float(_np.linalg.norm(self.get_map(ite, ref) - self.get_map(ite - 1, ref)))


###########################################################################
# SubtomoAvgMonitor — query-only view of an existing project
###########################################################################

class SubtomoAvgMonitor(SubtomoAvgBase):
    """Read-only monitor for an existing subtomogram averaging project.

    Inherits all path helpers and query methods from
    :class:`SubtomoAvgBase` but has no processing modules and cannot run
    iterations.  Useful for inspection, visualisation, and scripting on
    top of a finished or in-progress project.

    Parameters
    ----------
    prj_name : str
        Path to an existing project directory.
    """

    def __init__(self, prj_name):
        super().__init__(prj_name, box_size=None)


###########################################################################
# SubtomoAvgCore — overridable pipeline on top of the infrastructure
###########################################################################

class SubtomoAvgCore(SubtomoAvgBase):
    """Pipeline layer: overridable steps between project setup and output.

    Concrete subclasses implement :meth:`run_estimation`,
    :meth:`select_particles`, :meth:`run_reconstruction`, and
    :meth:`run_postprocessing`.  :meth:`run_iteration` orchestrates the
    full sequence and can also be overridden.

    Users who want to customise just one step should subclass
    :class:`SubtomoAvg` and override the relevant method rather than this
    class.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_iteration(self, ite) -> tuple[_IterationFiles, _IterationFiles]:
        """Create the iteration directory and validate previous outputs.

        Parameters
        ----------
        ite : int
            Iteration number (≥ 1).

        Returns
        -------
        tuple
            ``(cur, prv)`` — :class:`_IterationFiles` for this and the
            previous iteration.

        Raises
        ------
        NameError
            If the previous iteration's files are missing.
        """
        base = self.iteration_dir(ite)
        if not _file_exists(base):
            _mkdir(base)
        cur = self.iteration_files(ite)
        prv = self.iteration_files(ite - 1)
        prv.check()
        return cur, prv

    # ------------------------------------------------------------------
    # Pipeline steps (override in subclasses)
    # ------------------------------------------------------------------

    def run_estimation(self, cur, prv):
        """Run alignment or CTF refinement.

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        raise NotImplementedError

    def select_particles(self, cur, prv):
        """Select particles and write ``cur.ptcl_rec``.

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        raise NotImplementedError

    def run_reconstruction(self, cur, prv):
        """Reconstruct maps and update ``cur.reference``.

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        raise NotImplementedError

    def run_postprocessing(self, cur, prv) -> float | _np.ndarray:
        """Compute resolution estimates and apply post-reconstruction filtering.

        Parameters
        ----------
        cur, prv : _IterationFiles

        Returns
        -------
        float or numpy.ndarray
            Estimated resolution in Fourier pixels.
        """
        raise NotImplementedError

    def run_iteration(self, ite) -> float | _np.ndarray:
        """Run a complete STA iteration.

        Parameters
        ----------
        ite : int
            Iteration number (≥ 1).

        Returns
        -------
        float or numpy.ndarray
            Estimated resolution in Fourier pixels.
        """
        start_time = _ssa_utils.time_now()
        print('============================')
        print('Project: %s (Iteration %d)' % (self.prj_name, ite))
        cur, prv = self.setup_iteration(ite)
        self.run_estimation(cur, prv)
        self.select_particles(cur, prv)
        self.run_reconstruction(cur, prv)
        rslt = self.run_postprocessing(cur, prv)
        elapsed = _ssa_utils.time_now() - start_time
        print('Iteration %d Finished [Elapsed time: %.1f seconds (%s)]'
              % (ite, elapsed.total_seconds(), str(elapsed)))
        return rslt


###########################################################################
# SubtomoAvg — concrete user-facing implementation
###########################################################################

class SubtomoAvg(SubtomoAvgCore):
    """Subtomogram averaging project manager.

    Manages an STA project stored on disk.  Provide *box_size* to create
    (or reuse) a project directory; omit it to open an existing project.

    The main entry point for automated workflows is :meth:`run_iteration`.
    Individual pipeline steps (:meth:`run_estimation`,
    :meth:`select_particles`, :meth:`run_reconstruction`,
    :meth:`run_postprocessing`) can also be called directly.

    .. rubric:: Project files

    .. attribute:: prj_name
       :type: str

    .. attribute:: box_size
       :type: int

    .. attribute:: tomogram_file
       :type: str

    .. attribute:: initial_reference
       :type: str

    .. attribute:: initial_particles
       :type: str

    .. rubric:: GPU & processing

    .. attribute:: list_gpus_ids
       :type: list of int

       Default: ``[0]``.

    .. rubric:: Iteration control

    .. attribute:: iteration_type
       :type: int or str

       ==================  ===========================
       Value               Step
       ==================  ===========================
       ``3`` / ``'3D'``    3-D angular + offset search
       ``2`` / ``'2D'``    2-D in-plane alignment
       ``'ctf'``           CTF refinement
       ==================  ===========================

       Default: ``3``.

    .. attribute:: cc_threshold
       :type: float

       Fraction of top-scoring particles kept per half-set.  Default: ``0.8``.

    .. attribute:: fsc_threshold
       :type: float

       FSC threshold for resolution reporting.  Default: ``0.143``.

    .. rubric:: Modules

    .. attribute:: aligner
       :type: :class:`~susan.modules.Aligner`

    .. attribute:: averager
       :type: :class:`~susan.modules.Averager`

    .. attribute:: ctf_refiner
       :type: :class:`~susan.modules.CtfRefiner`

    .. rubric:: Advanced

    .. attribute:: mpi
       :type: :class:`~susan.utils.datatypes.mpi_params`

    .. attribute:: verbosity
       :type: int

       Default: ``1``.

    .. attribute:: solver_2d
       :type: :class:`~susan.solvers.Solver2D` or None

       Refinement of the 2-D iterations from the alignment statistics: the
       aligner writes ``output_type='alignment_stats'`` (recommended: offset
       step 0.5 px; the cone and in-plane search of the 2-D iterations
       define the sequential angular search of the orientation correction,
       e.g. ``set_angular_search(2, 1, 1, 0.5)``; with :attr:`aligner`
       ``halfsets_independ`` the statistics are per half-set), then
       :meth:`~susan.solvers.Solver2D.solve` gives the particles of the next
       iteration; the statistics also carry the uncertainty of what it applied
       (:attr:`solver_highres`).  ``None``: plain 2-D alignment (no fitting;
       overfits at low SNR).  Default: ``Solver2D()``.

    .. attribute:: solver_ctf
       :type: :class:`~susan.solvers.SolverCTF` or None

       Defocus correction of the CTF iterations from pooled statistics: the
       :attr:`ctf_refiner` writes ``output_type='refinement_stats'`` (no
       per-particle search; its :attr:`~susan.modules.CtfRefiner.defocus_angstroms`
       span is the range of the per-particle curves; with :attr:`ctf_refiner`
       ``halfsets_independ`` the statistics are per half-set), then
       :meth:`~susan.solvers.SolverCTF.solve` gives the particles of the next
       iteration: per-tilt offset, global tilt geometry and per-particle depth
       error, and, when the refiner has an offset search, the significant
       per-tilt translation bias.  Astigmatism and phase shift are kept.
       ``None``: the free per-particle refinement.
       Default: ``None``.

    .. attribute:: solver_highres
       :type: :class:`~susan.solvers.SolverHighRes` or None

       Particles of the reconstruction, in every iteration type, from the
       latest statistics (written to ``ite_xxxx/particles_rec.ptclsraw``,
       which never feeds the alignment): per-tilt weights, ``def_Bfct`` from
       the shift uncertainty (plus the per-tilt B-factor of the CTF
       statistics with ``SolverHighRes.b_tilt``), ``def_mres`` from the
       angular uncertainty (with the particle radius
       from the reference mask), and the :attr:`cc_threshold` selection.  The
       null of :meth:`update_null` is the baseline of the 2-D weights when
       there is one, otherwise the second CC peak.  ``None``: the
       :attr:`cc_threshold` selection only.  Default: ``SolverHighRes()``.

    .. attribute:: use_2d_stats
       :type: int

       Iteration whose 2-D statistics (``particles.stats.npz``) feed
       :attr:`solver_highres`, ``0`` none.  Set to the iteration by every 2-D
       iteration run with :attr:`solver_2d` (before :attr:`solver_highres`, so
       it uses its own); kept in ``info.prjtxt``.  After an aggressive 3-D
       refinement the statistics no longer describe the particles: set it to
       ``0``.  A missing file or a different number of particles skips that
       part with a warning.

    .. attribute:: use_ctf_stats
       :type: int

       Same for the CTF statistics (``particles.ctfstats.npz``), set by every
       CTF iteration run with :attr:`solver_ctf`.

    .. attribute:: max_tilt_reconstruction
       :type: None, float, or array-like of length 2

       * ``None`` or negative scalar — disabled (default: ``-1``).
       * Scalar — ``tilt_deg_max`` passed to ``enable_by_tilt``.
       * Two-element sequence ``[min, max]`` — passed to
         ``enable_by_tilt_range``.

    .. attribute:: use_nominal
       :type: bool

       If ``True``, the tilt comparison in :attr:`max_tilt_reconstruction`
       uses ``Tomograms.nominal_tilt_angles`` instead of deriving the tilt
       from ``proj_eZYZ``.  Default: ``False``.

    .. attribute:: discard_oversampled_views
       :type: None, int, or dict

       Flatten preferential orientation before reconstruction by keeping only
       the best particles per equal-area view bin (via
       :meth:`Particles.Geom.discard_oversampled_views`).

       * ``None`` or non-positive scalar — disabled (default: ``None``).
       * Positive integer — used as ``k_per_bin`` with default
         ``bin_size_deg=5.0``.
       * Dict — passed verbatim as keyword arguments (e.g.
         ``{'bin_size_deg': 4.0, 'k_per_bin': 2}``).

    .. attribute:: smooth_ctf
       :type: bool

       .. warning:: **Experimental.**

       Enable spatial smoothing of per-particle CTF defocus deltas (the
       change versus the previous iteration), per tilt.  The smoother is
       selected by :attr:`type_ctf_smoothing`.  Default: ``False``.

    .. attribute:: type_ctf_smoothing
       :type: str

       .. warning:: **Experimental.**

       Smoother used when :attr:`smooth_ctf` is enabled: ``'gaussian'``
       (local kNN average) or ``'tps'`` (regularised thin-plate spline warp).
       Default: ``'gaussian'``.

    .. attribute:: ctf_tps_lambda
       :type: float

       .. warning:: **Experimental.**

       Stiffness of the ``'tps'`` defocus warp (dimensionless; coordinates are
       normalised per tomogram).  ``0`` interpolates every delta, large values
       converge to the affine fit.
       Default: ``1.0``.

    .. attribute:: reweight_classification
       :type: bool or float

       .. warning:: **Experimental.**

       Multi-reference CC reweighting.  ``False`` keeps the raw CC; ``True``
       normalises each particle's per-class CC to sum to one; a positive number
       is an inverse-temperature ``beta`` selecting soft ML weights (softmax
       responsibilities scaled by a per-particle evidence gate, so noise fades
       toward a small floor).  Pairs with ``aligner.ignore_classes = True``,
       ``averager.ignore_classes = True`` and ``averager.weighting_type =
       '3DCC'``.  Default: ``False``.

    .. attribute:: cross_halfmaps
       :type: bool

       If ``True``, the half-map paths in the reference are swapped when
       writing ``cur.reference`` after each reconstruction: half1 particles
       will be aligned against the half2 map and vice versa in the next
       iteration.  Requires :attr:`aligner.halfsets_independ` to be
       ``True`` to take effect.  Default: ``False``.

    .. attribute:: save_raw_map
       :type: bool

       If ``True``, the unfiltered map produced by the averager is saved
       alongside the final map as ``map_classNNN.raw.mrc`` before any
       filter is applied.  Default: ``False``.

    .. attribute:: map_filter
       :type: callable or None

       Optional post-reconstruction filter that does not use the FSC.
       Signature: ``filter(vol) -> vol``.  Setting this clears
       :attr:`map_filter_fsc`.  Default: ``None``.

    .. attribute:: map_filter_fsc
       :type: callable or None

       Optional post-reconstruction filter that uses the FSC (e.g. FOM,
       spectral Wiener).  Signature: ``filter(vol, fsc) -> vol``, where
       *fsc* is the 1-D FSC array for that reference.  Setting this clears
       :attr:`map_filter`.  Default: ``None``.

    .. attribute:: rho_v
       :type: float

       MACE consensus weight for the volume.  Must be in ``(0, 1]``.

       * ``1.0`` (default) — classical mode: if a filter is set it is
         applied directly to ``V_data`` with no consensus.
       * ``< 1.0`` — MACE mode: ``V_cons = ρ·V_data + (1−ρ)·V_prior``,
         where ``V_prior`` is the filter output.  The residual
         ``U_V = U_V + V_data − V_cons`` is saved as
         ``map_classNNN.residual.mrc`` in the iteration directory and
         loaded from the previous iteration to form the denoiser input
         ``V_data + U_V``.  Setting ``rho_v < 1`` also forces
         :attr:`save_raw_map` behaviour (``V_data`` is always saved as
         ``map_classNNN.raw.mrc``).
    """

    # ------------------------------------------------------------------
    # Persistent-field properties
    # (assignments write back to info.prjtxt automatically)
    # ------------------------------------------------------------------

    def _save_prjtxt(self):
        fp = open(self.prj_name + '/info.prjtxt', 'w')
        _prsr.write(fp, 'name',               self.prj_name)
        _prsr.write(fp, 'box_size',           str(self.box_size))
        _prsr.write(fp, 'tomogram_file',      self._tomogram_file)
        _prsr.write(fp, 'initial_reference',  self._initial_reference)
        _prsr.write(fp, 'initial_particles',  self._initial_particles)
        _prsr.write(fp, 'use_2d_stats',       str(self._use_2d_stats))
        _prsr.write(fp, 'use_ctf_stats',      str(self._use_ctf_stats))
        fp.close()

    @property
    def tomogram_file(self) -> str:
        return self._tomogram_file

    @tomogram_file.setter
    def tomogram_file(self, value):
        self._tomogram_file = value
        if value:
            self.pix_size = float(_ssa_data.Tomograms(value).pix_size[0])
        self._save_prjtxt()

    @property
    def initial_reference(self) -> str:
        return self._initial_reference

    @initial_reference.setter
    def initial_reference(self, value):
        self._initial_reference = value
        self._save_prjtxt()

    @property
    def use_2d_stats(self) -> int:
        return self._use_2d_stats

    @use_2d_stats.setter
    def use_2d_stats(self, value):
        self._use_2d_stats = int(value)
        self._save_prjtxt()

    @property
    def use_ctf_stats(self) -> int:
        return self._use_ctf_stats

    @use_ctf_stats.setter
    def use_ctf_stats(self, value):
        self._use_ctf_stats = int(value)
        self._save_prjtxt()

    @property
    def initial_particles(self) -> str:
        return self._initial_particles

    @initial_particles.setter
    def initial_particles(self, value):
        self._initial_particles = value
        self._save_prjtxt()

    # ------------------------------------------------------------------

    def __init__(self, prj_name, box_size=None):
        # Initialise backing attrs before super().__init__ so that the
        # property setters (which call _save_prjtxt) work from the start.
        self._tomogram_file     = ''
        self._initial_reference = ''
        self._initial_particles = ''
        self._use_2d_stats      = 0
        self._use_ctf_stats     = 0
        self.pix_size           = None
        super().__init__(prj_name, box_size)

        self.list_gpus_ids   = [0]
        self.iteration_type  = 3

        self.cc_threshold  = 0.8
        self.fsc_threshold = 0.143

        self.max_tilt_reconstruction   = -1
        self.use_nominal               = False
        self.discard_oversampled_views = None
        self.solver_2d                 = _ssa_solvers.Solver2D()
        self.solver_ctf                = None
        self.solver_highres            = _ssa_solvers.SolverHighRes()
        self.smooth_ctf                = False
        self.type_ctf_smoothing        = 'gaussian'
        self.ctf_tps_lambda            = 1.0
        self.reweight_classification   = False

        self.cross_halfmaps  = False
        self.save_raw_map    = False
        self._map_filter     = None
        self._map_filter_fsc = None
        self._rho_v          = 1.0

        self.mpi       = _dt.mpi_params('srun -n %d ', 1)
        self.verbosity = 1

        self.aligner     = _ssa_modules.Aligner()
        self.averager    = _ssa_modules.Averager()
        self.ctf_refiner = _ssa_modules.CtfRefiner()

        self.aligner.ctf_correction    = 'on_reference'
        self.aligner.cc_type           = 'cfsc'
        self.aligner.halfsets_independ = False

        self.averager.ctf_correction    = 'wiener'
        self.averager.rec_halfsets      = True
        self.averager.normalize_type    = 'zero_mean'
        self.averager.bandpass.highpass = 0
        self.averager.bandpass.lowpass  = -1

    @property
    def map_filter(self):
        return self._map_filter

    @map_filter.setter
    def map_filter(self, value):
        self._map_filter     = value
        self._map_filter_fsc = None

    @property
    def map_filter_fsc(self):
        return self._map_filter_fsc

    @map_filter_fsc.setter
    def map_filter_fsc(self, value):
        self._map_filter_fsc = value
        self._map_filter     = None

    @property
    def rho_v(self):
        return self._rho_v

    @rho_v.setter
    def rho_v(self, value):
        if not (0.0 < value <= 1.0):
            raise ValueError('rho_v must be in (0, 1]  (got %g)' % value)
        self._rho_v = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_iteration_type(self):
        v = self.iteration_type
        if v in (3, '3', '3D', '3d'):
            return 3
        elif v in (2, '2', '2D', '2d'):
            return 2
        elif v in ('ctf', 'CTF', 'Ctf'):
            return 'ctf'
        else:
            raise ValueError('Invalid iteration_type (accepted: 3, 2, "ctf")')

    def _run_alignment(self, cur, prv, ite_type):
        self.aligner.list_gpus_ids   = self.list_gpus_ids
        self.aligner.dimensionality  = ite_type
        self.aligner.verbosity       = self.verbosity
        self.aligner.output_type     = 'alignment_stats' if (ite_type == 2 and self.solver_2d is not None) else 'alignment'

        print('  [%dD Alignment] Start:' % ite_type)
        t0 = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.aligner.mpi.cmd = self.mpi.cmd
            self.aligner.mpi.arg = self.mpi.arg
            self.aligner.align_mpi(cur.ptcl_rslt, prv.reference,
                                   self.tomogram_file, prv.ptcl_rslt, self.box_size)
        else:
            self.aligner.align(cur.ptcl_rslt, prv.reference,
                               self.tomogram_file, prv.ptcl_rslt, self.box_size)
        elapsed = _ssa_utils.time_now() - t0
        print('  [%dD Alignment] Finished. Elapsed time: %.1f seconds (%s).'
              % (ite_type, elapsed.total_seconds(), str(elapsed)))

    def _run_ctf_refinement(self, cur, prv):
        self.ctf_refiner.list_gpus_ids   = self.list_gpus_ids
        self.ctf_refiner.verbosity       = self.verbosity
        self.ctf_refiner.output_type     = 'refinement_stats' if self.solver_ctf is not None else 'refinement'

        print('  [CTF Refinement] Start:')
        t0 = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.ctf_refiner.mpi.cmd = self.mpi.cmd
            self.ctf_refiner.mpi.arg = self.mpi.arg
            self.ctf_refiner.refine_mpi(cur.ptcl_rslt, prv.reference,
                                        self.tomogram_file, prv.ptcl_rslt, self.box_size)
        else:
            self.ctf_refiner.refine(cur.ptcl_rslt, prv.reference,
                                    self.tomogram_file, prv.ptcl_rslt, self.box_size)
        elapsed = _ssa_utils.time_now() - t0
        print('  [CTF Refinement] Finished. Elapsed time: %.1f seconds (%s).'
              % (elapsed.total_seconds(), str(elapsed)))

    def _smooth_ctf_parameters(self, ptcls_in, cur, prv):
        """Smooth the per-particle CTF defocus deltas in place (:attr:`smooth_ctf`); saves ``cur.ptcl_rslt``."""

        def _smooth_deltas(points, deltas, sigma, k):
            tree = _KDTree(points)
            out = _np.zeros_like(deltas)
            for i, pt in enumerate(points):
                dists, idx = tree.query(pt, k=k)
                w = _np.exp(-dists ** 2 / (2 * sigma ** 2))
                w /= w.sum()
                out[i] = (deltas[idx] * w[:, _np.newaxis]).sum(axis=0)
            return out

        def _tps_fit(pt0, deltas, lam):
            # Regularised thin-plate spline warp of a 2-D shift field.
            # lam is a dimensionless stiffness: 0 -> exact interpolation,
            # large -> pure affine.  Coordinates are centred and scaled by
            # their median radius so lam is comparable across tomograms.
            m = pt0.shape[0]
            c = pt0 - pt0.mean(0)
            L = _np.median(_np.linalg.norm(c, axis=1))
            if L <= 0:
                return deltas
            c  = (c / L).astype(_np.float64)
            r2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            K  = _np.where(r2 > 0, 0.5 * r2 * _np.log(_np.maximum(r2, 1e-12)), 0.0)
            P  = _np.hstack([_np.ones((m, 1)), c])
            A  = _np.zeros((m + 3, m + 3), dtype=_np.float64)
            A[:m, :m] = K + lam * _np.eye(m)
            A[:m, m:] = P
            A[m:, :m] = P.T
            rhs = _np.zeros((m + 3, 2), dtype=_np.float64)
            rhs[:m] = deltas
            sol = _np.linalg.lstsq(A, rhs, rcond=None)[0]
            return (K @ sol[:m] + P @ sol[m:]).astype(deltas.dtype)

        method = self.type_ctf_smoothing.lower()
        print('    Smoothing CTF defocus deltas (%s).' % method)
        ptcls_old = _ssa_data.Particles(prv.ptcl_rslt)
        delta_U   = ptcls_in.def_U - ptcls_old.def_U
        delta_V   = ptcls_in.def_V - ptcls_old.def_V
        pt    = ptcls_in.pos()
        tomos = _ssa_data.Tomograms(self.tomogram_file)
        min_n = 4 if method == 'tps' else 2
        for tcix in range(tomos.n_tomos):
            idx = ptcls_in.tomo_id == tomos.tomo_id[tcix]
            n_t = idx.sum()
            if n_t < min_n:
                continue
            k_eff = min(7, n_t)
            for i in range(tomos.num_proj[tcix]):
                pt0 = (tomos.project(tcix, pt[idx], i, shift=False))[:, :2]
                d   = _np.stack([delta_U[idx, i], delta_V[idx, i]], axis=1)
                if method == 'tps':
                    s = _tps_fit(pt0, d, self.ctf_tps_lambda)
                else:
                    sigma = _np.median(_np.linalg.norm(pt0, axis=1)) * 0.25
                    s = _smooth_deltas(pt0, d, sigma=sigma, k=k_eff)
                ptcls_in.def_U[idx, i] = ptcls_old.def_U[idx, i] + s[:, 0]
                ptcls_in.def_V[idx, i] = ptcls_old.def_V[idx, i] + s[:, 1]

        ptcls_in.save(cur.ptcl_rslt)

    def _solve_2d(self, ptcls_aligned, cur, prv):
        """2-D solution of the aligned particles from the statistics (:attr:`solver_2d`); saves ``cur.ptcl_rslt``."""
        print('    2D solver (per tomogram%s):' % (' and half-set' if self.aligner.halfsets_independ else ''))
        P = self.solver_2d.solve(prv.ptcl_rslt, _ssa_io.ali_stats.path(cur.ptcl_rslt), self.tomogram_file)
        P.ali_cc[:] = ptcls_aligned.ali_cc
        P.prj_cc[:] = ptcls_aligned.prj_cc
        P.save(cur.ptcl_rslt)
        self.use_2d_stats = cur.ite
        return P

    def _solve_ctf(self, cur, prv):
        """Defocus correction of the input particles from the statistics (:attr:`solver_ctf`); saves ``cur.ptcl_rslt``."""
        print('    CTF solver (per tomogram%s):' % (' and half-set' if self.ctf_refiner.halfsets_independ else ''))
        offset_search = max(self.ctf_refiner.offset.span[:2]) > 0
        P = self.solver_ctf.solve(prv.ptcl_rslt, _ssa_io.ctf_stats.path(cur.ptcl_rslt), self.tomogram_file,
                                  offset_search=offset_search)
        P.save(cur.ptcl_rslt)
        self.use_ctf_stats = cur.ite
        return P

    def _stats_file(self, ite, path, name):
        """File of the statistics of iteration ite, None when disabled (0) or missing (a warning)."""
        if ite < 1:
            return None
        f = path(self.iteration_files(ite).ptcl_rslt)
        if not _file_exists(f):
            _warnings.warn('The %s statistics of iteration %d (%s) do not exist: not used.' % (name, ite, f),
                           RuntimeWarning, stacklevel=3)
            return None
        return f

    def _mask_radius(self, prv):
        """Particle radius (Angstroms) from the reference mask: 90th percentile of the distance from the box centre
        of the voxels above 0.5; None without a mask."""
        refs = _ssa_data.Reference(prv.reference)
        r = 0.0
        for msk in refs.msk:
            if not msk or not _file_exists(msk):
                continue
            vol, apix = _mrc_read(msk)
            c = _np.argwhere(vol > 0.5) - _np.array(vol.shape) // 2
            if len(c):
                r = max(r, float(_np.percentile(_np.linalg.norm(c, axis=1), 90)) * float(apix[0]))
        return r if r > 0 else None

    def _solve_highres(self, ptcls, cur, prv):
        """Particles of the reconstruction from the statistics of :attr:`use_2d_stats` and :attr:`use_ctf_stats`."""
        f2d = self._stats_file(self.use_2d_stats, _ssa_io.ali_stats.path, '2D')
        fct = self._stats_file(self.use_ctf_stats, _ssa_io.ctf_stats.path, 'CTF')
        null = self._null_baseline(ptcls) if f2d is not None else None
        print('    Reconstruction particles from the statistics: 2D %s, CTF %s.' % (
            ('iteration %d (%s baseline)' % (self.use_2d_stats, 'null' if null is not None else 'second peak')) if f2d else 'none',
            ('iteration %d' % self.use_ctf_stats) if fct else 'none'))
        return self.solver_highres.solve(ptcls, f2d, fct, self.tomogram_file, cc_threshold=self.cc_threshold,
                                         null_baseline=null, radius=self._mask_radius(prv))

    # ------------------------------------------------------------------
    # Null of the 2-D statistics
    # ------------------------------------------------------------------

    def _null_dir(self):
        return self.prj_name + '/null/'

    def _null_setup(self):
        """The settings a null depends on (written to and compared with ``null/null.json``)."""
        a = self.aligner
        return dict(box_size=int(self.box_size), tomogram_file=str(self.tomogram_file),
                    highpass=float(a.bandpass.highpass), lowpass=float(a.bandpass.lowpass), rolloff=float(a.bandpass.rolloff),
                    cc_type=str(a.cc_type), ctf_correction=str(a.ctf_correction), extra_padding=int(a.extra_padding),
                    offset_span=[float(v) for v in a.offset.span], offset_step=float(a.offset.step),
                    offset_type=str(a.offset.kind), halfsets_independ=bool(a.halfsets_independ))

    def update_null(self, iter_base, kind='rot90'):
        """Null of the 2-D statistics, kept in ``{project}/null/`` and used by the 2-D iterations until updated.

        Null references are made from the maps of iteration ``iter_base`` (:func:`susan.utils.null_map`), following
        the half-map policy of :attr:`aligner` (with ``halfsets_independ`` each half map gives its half's null), and
        the particles of ``iter_base`` are aligned to them in 2-D with the current :attr:`aligner` settings. The CC
        baseline (median null peak per tomogram, half-set and tilt) and a descriptor of the setup
        (``null.json``) are kept; the alignment outputs are removed. Before use the descriptor is compared with the
        setup of the iteration (a warning when they differ). Without a null the weights use the second CC peak,
        which matches it on EMPIAR-10064 (same reconstruction).

        Parameters
        ----------
        iter_base : int
            Iteration whose maps and particles are used.
        kind : str
            ``'rot90'`` (the map rotated 90 degrees about y; default) or ``'phase_random'``.
        """
        import json as _json
        base = self.iteration_files(iter_base)
        d = self._null_dir()
        if not _file_exists(d):
            _mkdir(d)
        refs = _ssa_data.Reference(base.reference)
        halves = bool(self.aligner.halfsets_independ)
        for i in range(refs.n_refs):
            msk, _ = _mrc_read(refs.msk[i])
            srcs = (('h1', refs.h1), ('h2', refs.h2)) if halves else (('ref', refs.ref),)
            for name, lst in srcs:
                vol, apix = _mrc_read(lst[i])
                out = d + 'null_class%03d_%s.mrc' % (i + 1, name)
                _mrc_write(_ssa_utils.null_map(vol, msk, kind, seed=i).astype(_np.float32), out, apix)
                lst[i] = out
        refs.save(d + 'null.refstxt')
        a = self.aligner
        prev = (a.output_type, a.dimensionality, a.cone, a.inplane)
        a.output_type, a.dimensionality = 'alignment_stats', 2
        a.cone, a.inplane = type(a.cone)(0, 1), type(a.inplane)(0, 1)
        a.list_gpus_ids, a.verbosity = self.list_gpus_ids, self.verbosity
        out = d + 'null_ali.ptclsraw'
        print('  [Null] Aligning the particles of iteration %d to the %s null:' % (iter_base, kind))
        try:
            if self.mpi.arg > 1:
                a.mpi.cmd, a.mpi.arg = self.mpi.cmd, self.mpi.arg
                a.align_mpi(out, d + 'null.refstxt', self.tomogram_file, base.ptcl_rslt, self.box_size)
            else:
                a.align(out, d + 'null.refstxt', self.tomogram_file, base.ptcl_rslt, self.box_size)
        finally:
            a.output_type, a.dimensionality, a.cone, a.inplane = prev
        S = _ssa_io.ali_stats.read(out)
        P = _ssa_data.Particles(base.ptcl_rslt)
        cc = S['projections/stats'][..., _ssa_io.ali_stats.FIELDS.index('cc')]
        tid, hid = P.tomo_id.flatten().astype(int), P.half_id.flatten().astype(int)
        tids = _np.unique(tid)
        K = cc.shape[1]
        b = _np.full((len(tids), 3, K), _np.nan)
        for it, t in enumerate(tids):
            for h in (0, 1, 2):
                r = (tid == t) & (hid == h)
                for k in range(K):
                    v = cc[r, k]
                    v = v[_np.isfinite(v) & (v > 0)]
                    if len(v) >= 10:
                        b[it, h, k] = _np.median(v)
        _np.savez(d + 'null_baseline.npz', tomo_id=tids, baseline=b)
        with open(d + 'null.json', 'w') as fp:
            _json.dump(dict(iter_base=int(iter_base), kind=kind, setup=self._null_setup()), fp, indent=1)
        for f in (out, _ssa_io.ali_stats.path(out)):
            if _file_exists(f):
                _rm(f)
        print('  [Null] Baseline kept in %s (median %.4g).' % (d, _np.nanmedian(b)))

    def _null_baseline(self, ptcls):
        """(n_ptcl, K) null baseline of the particles from ``null/``, None when there is no null."""
        import json as _json
        d = self._null_dir()
        if not (_file_exists(d + 'null.json') and _file_exists(d + 'null_baseline.npz')):
            return None
        with open(d + 'null.json') as fp:
            desc = _json.load(fp)
        cur = self._null_setup()
        diff = []
        for k, v in desc['setup'].items():
            c = cur.get(k)
            if isinstance(v, float) and isinstance(c, float):
                if abs(v - c) > 0.05 * max(abs(v), abs(c), 1e-12):
                    diff.append('%s %g -> %g' % (k, v, c))
            elif v != c:
                diff.append('%s %s -> %s' % (k, v, c))
        if diff:
            _warnings.warn('The null of iteration %d was made with another setup (%s): its baseline may not apply; '
                           'call update_null.' % (desc['iter_base'], '; '.join(diff)), RuntimeWarning, stacklevel=3)
        z = _np.load(d + 'null_baseline.npz')
        tids, b = list(z['tomo_id']), z['baseline']
        n, K = ptcls.prj_w.shape
        out = _np.full((n, K), _np.nan)
        tid, hid = ptcls.tomo_id.flatten().astype(int), ptcls.half_id.flatten().astype(int)
        for i in range(n):
            if tid[i] in tids:
                kk = min(K, b.shape[2])
                out[i, :kk] = b[tids.index(tid[i]), min(max(hid[i], 0), 2), :kk]
        return out

    def _apply_cc_threshold(self, ptcls_in):
        """Zero the half-set label of low-CC particles; return selected subset."""
        for i in range(ptcls_in.n_refs):
            idx = (ptcls_in.ref_cix == i).flatten()
            if _np.any(idx):
                hid = ptcls_in.half_id[idx].flatten()
                ccc = ptcls_in.ali_cc[i, idx].flatten()

                n_rf = hid.shape[0]
                n_h1 = (hid == 1).sum()
                n_h2 = (hid == 2).sum()

                if n_h1 > 0:
                    th1 = _np.quantile(ccc[hid == 1], 1 - self.cc_threshold)
                    hid[(hid == 1) & (ccc < th1)] = 0
                if n_h2 > 0:
                    th2 = _np.quantile(ccc[hid == 2], 1 - self.cc_threshold)
                    hid[(hid == 2) & (ccc < th2)] = 0

                ptcls_in.half_id[idx] = hid

                print('    Class %2d: %7d particles [%7d].' % (i + 1, n_rf, (hid > 0).sum()))
                print('      Half 1: %7d particles [%7d].'  % (n_h1, (hid == 1).sum()))
                print('      Half 2: %7d particles [%7d].'  % (n_h2, (hid == 2).sum()))
            else:
                print('    Class %2d: %7d particles.' % (i + 1, 0))
                print('      Half 1: %7d particles.'  % 0)
                print('      Half 2: %7d particles.'  % 0)

        return ptcls_in[(ptcls_in.half_id > 0).flatten()]

    def _apply_tilt_limit(self, ptcls_out):
        """Zero projections outside the allowed tilt range."""
        tomos = _ssa_data.Tomograms(filename=self.tomogram_file)
        prj_w = _np.copy(ptcls_out.prj_w)
        v = self.max_tilt_reconstruction
        src = 'nominal tilt' if self.use_nominal else 'proj_eZYZ'
        if isinstance(v, (list, tuple, _np.ndarray)):
            v = _np.asarray(v, dtype=_np.float32)
            print('    Restricting reconstruction to tilt range [%.2f, %.2f] degrees (%s).'
                  % (v.min(), v.max(), src))
            _ssa_data.Particles.Geom.enable_by_tilt_range(
                ptcls_out, tomos, tilt_deg_min=v.min(), tilt_deg_max=v.max(),
                use_nominal=self.use_nominal)
        else:
            print('    Restricting reconstruction to %.2f maximum tilt (%s).' % (v, src))
            _ssa_data.Particles.Geom.enable_by_tilt(
                ptcls_out, tomos, tilt_deg_max=v, use_nominal=self.use_nominal)
        ptcls_out.prj_w = ptcls_out.prj_w * prj_w

    def _apply_discard_oversampled_views(self, ptcls_out):
        """Subsample particles to flatten preferential orientation."""
        v = self.discard_oversampled_views
        if isinstance(v, dict):
            kwargs = dict(v)
        else:
            kwargs = {'k_per_bin': int(v)}
        kwargs.setdefault('bin_size_deg', 5.0)
        kwargs.setdefault('k_per_bin', 1)
        n_before = ptcls_out.n_ptcl
        ptcls_out = _ssa_data.Particles.Geom.discard_oversampled_views(
            ptcls_out, **kwargs)
        print('    Flattening orientation: kept %d / %d particles '
              '(bin_size_deg=%.2f, k_per_bin=%d).' % (
                  ptcls_out.n_ptcl, n_before,
                  kwargs['bin_size_deg'], kwargs['k_per_bin']))
        return ptcls_out

    # ------------------------------------------------------------------
    # Iteration orchestration
    # ------------------------------------------------------------------

    def run_iteration(self, ite) -> float | _np.ndarray:
        """Run a complete STA iteration, or skip the seed iteration.

        For ``ite >= 1`` this defers to
        :meth:`SubtomoAvgCore.run_iteration`.  Iteration ``0`` is the project
        seed and cannot be processed: a warning is issued, the iteration is
        skipped, and the configured starting lowpass is returned instead —
        the lowpass of :attr:`aligner` for a 3-D/2-D iteration or
        :attr:`ctf_refiner` for a CTF iteration.  For a multi-reference
        project (initial reference holding more than one map) a
        :class:`numpy.ndarray` of length ``n_refs`` filled with that value is
        returned, matching the per-reference shape of a real iteration's
        result; otherwise a scalar ``float``.

        Parameters
        ----------
        ite : int

        Returns
        -------
        float or numpy.ndarray
        """
        if ite < 1:
            _warnings.warn(
                'run_iteration(%d): iteration 0 is the project seed and cannot '
                'be run; returning the configured starting lowpass instead.'
                % ite, stacklevel=2)
            print('============================')
            print('Project: %s (Iteration %d) Skipped.' % (self.prj_name, ite))
            if self._validate_iteration_type() == 'ctf':
                lowpass = self.ctf_refiner.bandpass.lowpass
            else:
                lowpass = self.aligner.bandpass.lowpass
            n_refs = _ssa_data.Reference(self.path_refstxt(0)).n_refs
            if n_refs > 1:
                return _np.full(n_refs, lowpass, dtype=_np.float32)
            return lowpass

        return super().run_iteration(ite)

    def execute_iteration(self, ite) -> float | _np.ndarray:
        """Alias of :meth:`run_iteration`, for backward compatibility with
        :class:`~susan.project.STA.STA`."""
        return self.run_iteration(ite)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def run_estimation(self, cur, prv):
        """Run alignment or CTF refinement (dispatches on :attr:`iteration_type`).

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        ite_type = self._validate_iteration_type()
        if ite_type == 'ctf':
            self._run_ctf_refinement(cur, prv)
        else:
            self._run_alignment(cur, prv, ite_type)

    def select_particles(self, cur, prv):
        """Classify, regularise, threshold, and write ``cur.ptcl_rec``.

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        print('  [Aligned particles] Processing:')
        ptcls_in = _ssa_data.Particles(cur.ptcl_rslt)
        ite_type = self._validate_iteration_type()
        use_solver = ite_type == 2 and self.solver_2d is not None

        if use_solver:
            ptcls_in = self._solve_2d(ptcls_in, cur, prv)
        if ite_type == 'ctf' and self.solver_ctf is not None:
            ptcls_in = self._solve_ctf(cur, prv)
        if ite_type == 'ctf' and self.smooth_ctf:
            self._smooth_ctf_parameters(ptcls_in, cur, prv)

        if ptcls_in.n_refs > 1:
            ptcls_in.ref_cix = _np.argmax(ptcls_in.ali_cc, axis=0).astype(_np.uint32)
            if type(self.reweight_classification) is bool:
                if self.reweight_classification:
                    total = ptcls_in.ali_cc.sum(axis=0)
                    total[total == 0] = 1
                    ptcls_in.ali_cc = ptcls_in.ali_cc / total
            elif isinstance(self.reweight_classification, (int, float)):
                ptcls_in.ali_cc = _ml_class_weights(
                    ptcls_in.ali_cc.astype(_np.float64),
                    float(self.reweight_classification)).astype(ptcls_in.ali_cc.dtype)
            ptcls_in.save(cur.ptcl_rslt)

        if self.solver_highres is not None:
            ptcls_out = self._solve_highres(ptcls_in, cur, prv)
        else:
            ptcls_out = self._apply_cc_threshold(ptcls_in)

        v = self.discard_oversampled_views
        flatten_active = (v is not None) and (
            isinstance(v, dict) or v >= 1)
        if flatten_active:
            ptcls_out = self._apply_discard_oversampled_views(ptcls_out)

        v = self.max_tilt_reconstruction
        tilt_active = (v is not None) and (
            isinstance(v, (list, tuple, _np.ndarray)) or v >= 0)
        if tilt_active:
            self._apply_tilt_limit(ptcls_out)

        ptcls_out.save(cur.ptcl_rec)
        print('  [Aligned particles] Done.')

    def run_reconstruction(self, cur, prv):
        """Reconstruct reference maps and update ``cur.reference``.

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        self.averager.list_gpus_ids   = self.list_gpus_ids
        self.averager.verbosity       = self.verbosity

        print('  [Reconstruct Maps] Start:')
        t0 = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.averager.mpi.cmd = self.mpi.cmd
            self.averager.mpi.arg = self.mpi.arg
            self.averager.reconstruct_mpi(
                cur.ite_dir + 'map', self.tomogram_file, cur.ptcl_rec, self.box_size)
        else:
            self.averager.reconstruct(
                cur.ite_dir + 'map', self.tomogram_file, cur.ptcl_rec, self.box_size)
        elapsed = _ssa_utils.time_now() - t0
        print('  [Reconstruct Maps] Finished. Elapsed time: %.1f seconds (%s).'
              % (elapsed.total_seconds(), str(elapsed)))

        refs = _ssa_data.Reference(prv.reference)
        for i in range(refs.n_refs):
            refs.ref[i] = cur.ite_dir + 'map_class%03d.mrc'       % (i + 1)
            h1 = cur.ite_dir + 'map_class%03d_half1.mrc' % (i + 1)
            h2 = cur.ite_dir + 'map_class%03d_half2.mrc' % (i + 1)
            refs.h1[i]  = h2 if self.cross_halfmaps else h1
            refs.h2[i]  = h1 if self.cross_halfmaps else h2
        refs.save(cur.reference)

    def run_postprocessing(self, cur, prv) -> float | _np.ndarray:
        """Compute FSC-based resolution estimates and apply post-reconstruction
        filtering (classical or MACE consensus).

        Parameters
        ----------
        cur, prv : _IterationFiles

        Returns
        -------
        float or numpy.ndarray
            Estimated resolution in Fourier pixels.
        """
        refs = _ssa_data.Reference(cur.reference)
        if refs.n_refs == 1:
            print('  [FSC Calculation] Start (1 reference):')
        else:
            print('  [FSC Calculation] Start (%d references):' % refs.n_refs)

        active_filter = self.map_filter or self.map_filter_fsc
        mace_active   = active_filter is not None and self.rho_v < 1.0

        rslt = _np.zeros(refs.n_refs)
        for i in range(refs.n_refs):
            fsc = _ssa_utils.fsc_get(refs.h1[i], refs.h2[i], refs.msk[i])
            _, pix_size, _ = _mrc_get_info(refs.ref[i])
            fsc_rslt = _ssa_utils.fsc_analyse(fsc, pix_size, self.fsc_threshold)
            print('    - Reference %2d: %7.3f angstroms [%d fourier pixels]'
                  % (i + 1, fsc_rslt.res, fsc_rslt.fpix))
            rslt[i] = fsc_rslt.fpix

            if active_filter is None and not self.save_raw_map:
                continue

            map_file  = refs.ref[i]
            vol, apix = _mrc_read(map_file)
            raw_file  = map_file.replace('.mrc', '.raw.mrc')

            if mace_active:
                # Always save V_data when MACE is on
                _mrc_write(vol, raw_file, apix)
                # Load U_V from previous iteration (zeros if it doesn't exist)
                prv_res = (prv.ite_dir + 'map_class%03d.residual.mrc' % (i + 1)
                           if prv.ite_dir else '')
                U_V = (_mrc_read(prv_res)[0] if prv_res and _file_exists(prv_res)
                       else _np.zeros_like(vol))
                # Prior agent: filter receives V_data + U_V
                V_noisy = vol + U_V
                V_prior = (self.map_filter(V_noisy) if self.map_filter is not None
                           else self.map_filter_fsc(V_noisy, fsc))
                # Consensus and residual update
                V_cons  = self.rho_v * vol + (1.0 - self.rho_v) * V_prior
                U_new   = (U_V + vol - V_cons).astype(_np.float32)
                _mrc_write(U_new,                  map_file.replace('.mrc', '.residual.mrc'), apix)
                _mrc_write(V_cons.astype(_np.float32), map_file, apix)
            else:
                # Classical: optional raw save, then apply filter directly to V_data
                if self.save_raw_map:
                    _mrc_write(vol, raw_file, apix)
                if active_filter is not None:
                    V_out = (self.map_filter(vol) if self.map_filter is not None
                             else self.map_filter_fsc(vol, fsc))
                    _mrc_write(V_out.astype(_np.float32), map_file, apix)

        return rslt[0] if refs.n_refs == 1 else rslt
