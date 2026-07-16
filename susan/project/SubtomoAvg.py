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

import numpy as _np
import warnings as _warnings

from scipy.spatial import KDTree as _KDTree

import susan.data    as _ssa_data
import susan.utils   as _ssa_utils
import susan.modules as _ssa_modules

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

class _IterationFiles:
    """File-path bundle for a single iteration."""

    def __init__(self):
        self.ptcl_rslt = ''
        self.ptcl_temp = ''
        self.reference = ''
        self.ite_dir   = ''

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
            self.prj_name = _prsr.read(fp, 'name')
            self.box_size = int(_prsr.read(fp, 'box_size'))
            # Optional fields — present in files written by SubtomoAvg,
            # absent in files written by the legacy STA class.
            tomofile  = _prsr.read(fp, 'tomogram_file')
            initref   = _prsr.read(fp, 'initial_reference')
            initptcls = _prsr.read(fp, 'initial_particles')
            fp.close()
            self.tomogram_file     = ''
            self.initial_reference = initref   if initref   is not None else ''
            self.initial_particles = initptcls if initptcls is not None else ''
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

    # ------------------------------------------------------------------
    # Resolution conversions
    # ------------------------------------------------------------------

    def A2fpix(self, angstroms):
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

    def fpix2A(self, fpix):
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

    def iteration_dir(self, ite):
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

    def iteration_files(self, ite):
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
            rslt.ptcl_temp = base + 'temp.ptclsraw'
            rslt.reference = base + 'reference.refstxt'
            rslt.ite_dir   = base
        return rslt

    def path_map(self, ite, ref=1):
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

    def path_halfmap(self, ite, ref=1):
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

    def path_mask(self, ite, ref=1):
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

    def path_refstxt(self, ite):
        """Path to the ``.refstxt`` file for iteration *ite*.

        Returns
        -------
        str
        """
        return self.iteration_files(ite).reference

    def path_ptcls(self, ite):
        """Path to the ``.ptclsraw`` file for iteration *ite*.

        Returns
        -------
        str
        """
        return self.iteration_files(ite).ptcl_rslt

    def path_map_rec(self, ite):
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

    def get_map(self, ite, ref=1):
        """Load and return the reference map for iteration *ite*.

        Returns
        -------
        numpy.ndarray
        """
        v, _ = _mrc_read(self.path_map(ite, ref))
        return v

    def get_mask(self, ite, ref=1):
        """Load and return the soft mask for iteration *ite*.

        Returns
        -------
        numpy.ndarray
        """
        v, _ = _mrc_read(self.path_mask(ite, ref))
        return v

    def get_ptcls(self, ite):
        """Load and return the particles for iteration *ite*.

        Returns
        -------
        :class:`~susan.data.Particles`
        """
        return _ssa_data.Particles(self.path_ptcls(ite))

    def get_cc(self, ite, ref=1):
        """Per-particle CC scores for iteration *ite*, reference *ref*.

        Returns
        -------
        numpy.ndarray, shape (N,)
        """
        return self.get_ptcls(ite).ali_cc[ref - 1]

    def get_fsc(self, ite, ref=1):
        """Compute the FSC curve for iteration *ite*, reference *ref*.

        Returns
        -------
        numpy.ndarray
        """
        i = ref - 1
        refs = _ssa_data.Reference(self.path_refstxt(ite))
        return _ssa_utils.fsc_get(refs.h1[i], refs.h2[i], refs.msk[i])

    def map_change(self, ite, ref=1):
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

    def setup_iteration(self, ite):
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
        """Select particles and write ``cur.ptcl_temp``.

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

    def run_postprocessing(self, cur, prv):
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

    def run_iteration(self, ite):
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

    .. attribute:: max_2d_delta_angstroms
       :type: float

       Maximum per-iteration 2-D shift magnitude (Å).  ``0`` disables.
       Default: ``0``.

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

    .. attribute:: type_2d_shift_fitting
       :type: str

       .. warning:: **Experimental.**

       Post-alignment 2-D shift regularisation: ``'none'``, ``'affine'``,
       ``'gaussian'``, or ``'tps'``.  Default: ``'none'``.

       ``'tps'`` fits a regularised thin-plate spline to the per-tilt 2-D
       shift field.  It is the middle ground between ``'affine'`` (globally
       rigid) and ``'gaussian'`` (purely local): a globally smooth warp whose
       stiffness is set by :attr:`tps_lambda`.

    .. attribute:: tps_lambda
       :type: float

       .. warning:: **Experimental.**

       Stiffness of the ``'tps'`` shift warp (dimensionless; coordinates are
       normalised per tomogram so the value is comparable across datasets).
       ``0`` interpolates every shift (overfits noise), large values converge
       to the pure affine fit.  Default: ``1.0``.

    .. attribute:: fitting_from_origin
       :type: bool

       .. warning:: **Experimental.**

       Controls what the ``'affine'``/``'gaussian'``/``'tps'`` shift
       regularisers fit.  If ``True`` (default) they fit the absolute current
       ``prj_t`` (measured from origin).  If ``False`` they fit only the
       incremental ``prj_t`` (the delta versus the previous iteration) and
       add the smoothed delta back onto the previous shifts.  Incremental
       deltas are smaller and noisier, so :attr:`tps_lambda` typically needs
       to be larger in this mode.  Default: ``True``.

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

       Stiffness of the ``'tps'`` defocus warp, analogous to
       :attr:`tps_lambda` but applied independently to the CTF deltas.
       Default: ``1.0``.

    .. attribute:: reweight_classification
       :type: bool or float

       .. warning:: **Experimental.**

       Multi-reference CC reweighting.  Default: ``False``.

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
        fp.close()

    @property
    def tomogram_file(self):
        return self._tomogram_file

    @tomogram_file.setter
    def tomogram_file(self, value):
        self._tomogram_file = value
        if value:
            self.pix_size = float(_ssa_data.Tomograms(value).pix_size[0])
        self._save_prjtxt()

    @property
    def initial_reference(self):
        return self._initial_reference

    @initial_reference.setter
    def initial_reference(self, value):
        self._initial_reference = value
        self._save_prjtxt()

    @property
    def initial_particles(self):
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
        self.pix_size           = None
        super().__init__(prj_name, box_size)

        self.list_gpus_ids   = [0]
        self.iteration_type  = 3

        self.cc_threshold  = 0.8
        self.fsc_threshold = 0.143

        self.max_2d_delta_angstroms    = 0
        self.max_tilt_reconstruction   = -1
        self.use_nominal               = False
        self.discard_oversampled_views = None
        self.type_2d_shift_fitting     = 'none'
        self.tps_lambda                = 1.0
        self.fitting_from_origin       = True
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
        self.aligner.expfilt_gain      = 0.0
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

    def _regularize_2d_parameters(self, ptcls_in, cur, prv):
        """Apply 2-D shift regularisation and/or CTF smoothing in-place.

        Saves the modified particles to ``cur.ptcl_rslt``.
        """

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

        # When fitting_from_origin is False, the regularisers act on the
        # incremental shift (delta versus the previous iteration) and the
        # smoothed delta is added back onto the previous prj_t.
        prj_base = None
        if (not self.fitting_from_origin
                and self.type_2d_shift_fitting.lower() in ('affine', 'gaussian', 'tps')):
            prj_base = _ssa_data.Particles(prv.ptcl_rslt).prj_t

        if self.type_2d_shift_fitting == 'none':
            # max_2d_delta_angstroms only applies to 2D alignment, not CTF refinement
            if self.max_2d_delta_angstroms > 0 and self._validate_iteration_type() == 2:
                if self.aligner.allow_drift:
                    print('    Limiting 2D drift to %.2f Å.' % self.max_2d_delta_angstroms)
                    ptcls_old  = _ssa_data.Particles(prv.ptcl_rslt)
                    delta      = ptcls_in.prj_t - ptcls_old.prj_t
                    norm       = _np.linalg.norm(delta, axis=2)
                    scale      = self.max_2d_delta_angstroms / _np.maximum(norm, self.max_2d_delta_angstroms)
                    scale[norm < self.max_2d_delta_angstroms] = 1
                    ptcls_in.prj_t[:] = ptcls_old.prj_t + scale[:, :, _np.newaxis] * delta
                else:
                    print('    Limiting 2D shift to %.2f Å.' % self.max_2d_delta_angstroms)
                    norm  = _np.linalg.norm(ptcls_in.prj_t, axis=2)
                    scale = self.max_2d_delta_angstroms / _np.maximum(norm, self.max_2d_delta_angstroms)
                    scale[norm < self.max_2d_delta_angstroms] = 1
                    ptcls_in.prj_t[:] = scale[:, :, _np.newaxis] * ptcls_in.prj_t

        elif self.type_2d_shift_fitting.lower() == 'affine':
            R     = _np.eye(3, dtype=_np.float32)
            n     = ptcls_in.n_ptcl
            pt    = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                if idx.sum() < 4:
                    continue
                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix, i]).astype(_np.float32))
                    base    = 0.0 if prj_base is None else prj_base[idx, i]
                    pt0     = (pt[idx] @ R.T)[:, :2]
                    pt1     = pt0 + (ptcls_in.prj_t[idx, i] - base)
                    pt0_aug = _np.hstack([pt0, _np.ones((pt0.shape[0], 1))])
                    xform, _, _, _ = _np.linalg.lstsq(pt0_aug, pt1, rcond=None)
                    ptcls_in.prj_t[idx, i] = base + (pt0_aug @ xform - pt0)

        elif self.type_2d_shift_fitting.lower() == 'gaussian':
            R     = _np.eye(3, dtype=_np.float32)
            n     = ptcls_in.n_ptcl
            pt    = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                n_t = idx.sum()
                if n_t < 2:
                    continue
                k_eff = min(7, n_t)
                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix, i]).astype(_np.float32))
                    base  = 0.0 if prj_base is None else prj_base[idx, i]
                    pt0   = (pt[idx] @ R.T)[:, :2]
                    sigma = _np.median(_np.linalg.norm(pt0, axis=1)) * 0.25
                    ptcls_in.prj_t[idx, i] = base + _smooth_deltas(
                        pt0, ptcls_in.prj_t[idx, i] - base, sigma=sigma, k=k_eff)

        elif self.type_2d_shift_fitting.lower() == 'tps':
            R     = _np.eye(3, dtype=_np.float32)
            n     = ptcls_in.n_ptcl
            pt    = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                if idx.sum() < 4:
                    continue
                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix, i]).astype(_np.float32))
                    base = 0.0 if prj_base is None else prj_base[idx, i]
                    pt0  = (pt[idx] @ R.T)[:, :2]
                    ptcls_in.prj_t[idx, i] = base + _tps_fit(
                        pt0, ptcls_in.prj_t[idx, i] - base, self.tps_lambda)

        if self.smooth_ctf and self._validate_iteration_type() == 'ctf':
            method = self.type_ctf_smoothing.lower()
            print('    Smoothing CTF defocus deltas (%s).' % method)
            ptcls_old = _ssa_data.Particles(prv.ptcl_rslt)
            delta_U   = ptcls_in.def_U - ptcls_old.def_U
            delta_V   = ptcls_in.def_V - ptcls_old.def_V
            R     = _np.eye(3, dtype=_np.float32)
            n     = ptcls_in.n_ptcl
            pt    = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            min_n = 4 if method == 'tps' else 2
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                n_t = idx.sum()
                if n_t < min_n:
                    continue
                k_eff = min(7, n_t)
                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix, i]).astype(_np.float32))
                    pt0 = (pt[idx] @ R.T)[:, :2]
                    d   = _np.stack([delta_U[idx, i], delta_V[idx, i]], axis=1)
                    if method == 'tps':
                        s = _tps_fit(pt0, d, self.ctf_tps_lambda)
                    else:
                        sigma = _np.median(_np.linalg.norm(pt0, axis=1)) * 0.25
                        s = _smooth_deltas(pt0, d, sigma=sigma, k=k_eff)
                    ptcls_in.def_U[idx, i] = ptcls_old.def_U[idx, i] + s[:, 0]
                    ptcls_in.def_V[idx, i] = ptcls_old.def_V[idx, i] + s[:, 1]

        ptcls_in.save(cur.ptcl_rslt)

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

    def run_iteration(self, ite):
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

    def execute_iteration(self, ite):
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
        """Classify, regularise, threshold, and write ``cur.ptcl_temp``.

        Parameters
        ----------
        cur, prv : _IterationFiles
        """
        print('  [Aligned particles] Processing:')
        ptcls_in = _ssa_data.Particles(cur.ptcl_rslt)

        should_fix_2d = (
            (self.max_2d_delta_angstroms > 0)
            or (self.type_2d_shift_fitting != 'none')
            or self.smooth_ctf
        )
        if self._validate_iteration_type() in (2, 'ctf') and should_fix_2d:
            self._regularize_2d_parameters(ptcls_in, cur, prv)

        if ptcls_in.n_refs > 1:
            ptcls_in.ref_cix = _np.argmax(ptcls_in.ali_cc, axis=0).astype(_np.uint32)
            if type(self.reweight_classification) is bool:
                if self.reweight_classification:
                    total = ptcls_in.ali_cc.sum(axis=0)
                    total[total == 0] = 1
                    ptcls_in.ali_cc = ptcls_in.ali_cc / total
            elif isinstance(self.reweight_classification, (int, float)):
                ptcls_in.ali_cc = _np.power(ptcls_in.ali_cc, self.reweight_classification)
                total = ptcls_in.ali_cc.sum(axis=0)
                total[total == 0] = 1
                ptcls_in.ali_cc = ptcls_in.ali_cc / total
            ptcls_in.save(cur.ptcl_rslt)

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

        ptcls_out.save(cur.ptcl_temp)
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
                cur.ite_dir + 'map', self.tomogram_file, cur.ptcl_temp, self.box_size)
        else:
            self.averager.reconstruct(
                cur.ite_dir + 'map', self.tomogram_file, cur.ptcl_temp, self.box_size)
        elapsed = _ssa_utils.time_now() - t0
        print('  [Reconstruct Maps] Finished. Elapsed time: %.1f seconds (%s).'
              % (elapsed.total_seconds(), str(elapsed)))

        _rm(cur.ptcl_temp)

        refs = _ssa_data.Reference(prv.reference)
        for i in range(refs.n_refs):
            refs.ref[i] = cur.ite_dir + 'map_class%03d.mrc'       % (i + 1)
            h1 = cur.ite_dir + 'map_class%03d_half1.mrc' % (i + 1)
            h2 = cur.ite_dir + 'map_class%03d_half2.mrc' % (i + 1)
            refs.h1[i]  = h2 if self.cross_halfmaps else h1
            refs.h2[i]  = h1 if self.cross_halfmaps else h2
        refs.save(cur.reference)

    def run_postprocessing(self, cur, prv):
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
