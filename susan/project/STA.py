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
from susan.io.mrc import get_info as _mrc_get_info

import susan.utils.datatypes  as _dt
import susan.utils.txt_parser as _prsr

from os      import remove as _rm
from os      import mkdir  as _mkdir
from os.path import exists as _file_exists

class _iteration_files:
    def __init__(self):
        self.ptcl_rslt = ''
        self.ptcl_temp = ''
        self.reference = ''
        self.ite_dir   = ''
        
    def check(self):
        if not _file_exists(self.ptcl_rslt):
            raise NameError('File '+ self.ptcl_rslt + ' does not exist')
        if not _file_exists(self.reference):
            raise NameError('File '+ self.reference + ' does not exist')

class STA:
    """Subtomogram averaging project manager.

    Manages an STA project stored on disk.  If *box_size* is provided the
    project directory *prj_name* is created (or reused) and a metadata file
    is written.  If *box_size* is omitted the constructor reads the existing
    project.

    The main entry point for automated workflows is :meth:`execute_iteration`.
    Lower-level helpers (:meth:`exec_estimation`, :meth:`exec_particle_selection`,
    :meth:`exec_averaging`, :meth:`exec_postprocessing`) can also be called
    individually for custom pipelines.

    .. rubric:: Project files

    .. attribute:: prj_name
       :type: str

       Project directory path.  Set by the constructor.

    .. attribute:: box_size
       :type: int

       Subvolume box size in pixels.  Set by the constructor.

    .. attribute:: tomogram_file
       :type: str

       Path to the ``.tomostxt`` file used throughout the project.

    .. attribute:: initial_reference
       :type: str

       Path to the initial ``.refstxt`` file (iteration 0 reference).

    .. attribute:: initial_particles
       :type: str

       Path to the initial ``.ptclsraw`` file (iteration 0 particles).

    .. rubric:: GPU & processing

    .. attribute:: list_gpus_ids
       :type: list of int

       GPU device IDs forwarded to :attr:`aligner`, :attr:`averager`, and
       :attr:`ctf_refiner` at execution time.  Default: ``[0]``.

    .. rubric:: Iteration control

    .. attribute:: iteration_type
       :type: int or str

       Type of processing step executed by :meth:`execute_iteration`:

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

       Fraction of top-scoring particles kept for reconstruction (by
       cross-correlation score within each half-set).  Must be in
       ``(0, 1]``.  Default: ``0.8``.

    .. attribute:: fsc_threshold
       :type: float

       FSC threshold used for resolution estimation in
       :meth:`exec_postprocessing`.  Default: ``0.143``.

    .. rubric:: Modules

    .. attribute:: aligner
       :type: :class:`~susan.modules.Aligner`

       Aligner instance used for 3-D/2-D alignment steps.

    .. attribute:: averager
       :type: :class:`~susan.modules.Averager`

       Averager instance used for map reconstruction.

    .. attribute:: ctf_refiner
       :type: :class:`~susan.modules.CtfRefiner`

       CtfRefiner instance used for CTF refinement steps.

    .. rubric:: Advanced

    .. attribute:: mpi
       :type: :class:`~susan.utils.datatypes.mpi_params`

       MPI launcher forwarded to child modules when ``mpi.arg > 1``.
       Default: ``mpi_params('srun -n %d ', 1)``.

    .. attribute:: verbosity
       :type: int

       Verbosity level forwarded to child modules.  Default: ``1``.

    .. attribute:: max_2d_delta_angstroms
       :type: float

       Maximum 2-D shift magnitude (Å) allowed per iteration.  ``0``
       disables the limit.  Default: ``0``.

    .. attribute:: max_tilt_reconstruction
       :type: None, float, or array-like of length 2

       Tilt-angle limit applied during reconstruction.  Three forms:

       * ``None`` or a negative scalar — disabled (default: ``-1``).
       * Scalar (``int`` or ``float``) — passes ``tilt_deg_max`` to
         :meth:`~susan.data.Particles.Geom.enable_by_tilt`; projections
         whose absolute tilt exceeds this value are zeroed.
       * Two-element sequence ``[min, max]`` — passes both signed bounds
         to :meth:`~susan.data.Particles.Geom.enable_by_tilt_range`,
         allowing asymmetric tilt ranges.

    .. attribute:: type_2d_shift_fitting
       :type: str

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       Post-alignment 2-D shift regularisation: ``'none'``,
       ``'affine'``, ``'gaussian'``, or ``'tps'``.  Default: ``'none'``.

       ``'tps'`` fits a regularised thin-plate spline to the per-tilt 2-D
       shift field: a globally smooth warp sitting between ``'affine'``
       (globally rigid) and ``'gaussian'`` (purely local), with stiffness
       controlled by :attr:`tps_lambda`.

    .. attribute:: tps_lambda
       :type: float

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       Stiffness of the ``'tps'`` shift warp (dimensionless; coordinates are
       normalised per tomogram so the value is comparable across datasets).
       ``0`` interpolates every shift (overfits noise), large values converge
       to the pure affine fit.  Default: ``1.0``.

    .. attribute:: fitting_from_origin
       :type: bool

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       Controls what the ``'affine'``/``'gaussian'``/``'tps'`` shift
       regularisers fit.  If ``True`` (default) they fit the absolute current
       ``prj_t`` (measured from origin).  If ``False`` they fit only the
       incremental ``prj_t`` (the delta versus the previous iteration) and
       add the smoothed delta back onto the previous shifts.  Incremental
       deltas are smaller and noisier, so :attr:`tps_lambda` typically needs
       to be larger in this mode.  Default: ``True``.

    .. attribute:: smooth_ctf
       :type: bool

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       If ``True``, apply spatial smoothing to the per-particle CTF defocus
       deltas (difference from previous iteration) after CTF refinement.  The
       smoother is selected by :attr:`type_ctf_smoothing`.  Default: ``False``.

    .. attribute:: type_ctf_smoothing
       :type: str

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       Smoother used when :attr:`smooth_ctf` is enabled: ``'gaussian'``
       (local kNN average, same kernel as ``type_2d_shift_fitting =
       'gaussian'``) or ``'tps'`` (regularised thin-plate spline warp).
       Default: ``'gaussian'``.

    .. attribute:: ctf_tps_lambda
       :type: float

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       Stiffness of the ``'tps'`` defocus warp, analogous to
       :attr:`tps_lambda` but applied independently to the CTF deltas.
       Default: ``1.0``.

    .. attribute:: reweight_classification
       :type: bool or float

       .. warning:: **Experimental.** This feature may change or be removed
          in a future release.

       Multi-reference classification weight strategy.  ``False`` keeps
       raw CC scores; ``True`` normalises them to sum to 1; a float *p*
       raises them to the power *p* before normalising.
       Default: ``False``.
    """

    def __init__(self, prj_name, box_size=None):
        """Load an existing project or create a new one.

        Parameters
        ----------
        prj_name : str
            Path to the project directory.  Created if it does not exist
            (only when *box_size* is provided).
        box_size : int, optional
            Subvolume box size in pixels.  When given, initialises a new
            project.  When omitted, reads ``prj_name/info.prjtxt``.
        """
        if box_size is None:
            fp = open(prj_name+"/info.prjtxt","r")
            args = _prsr.parse_args(fp)
            fp.close()
            self.prj_name = args['name']
            self.box_size = int(args['box_size'])
        else:
            if not _file_exists(prj_name):
                _mkdir(prj_name)
            fp = open(prj_name+"/info.prjtxt","w")
            _prsr.write(fp,'name',prj_name)
            _prsr.write(fp,'box_size',str(box_size))
            fp.close()
            self.prj_name = prj_name
            self.box_size = box_size
        
        self.tomogram_file     = ''
        self.initial_reference = ''
        self.initial_particles = ''
        
        self.list_gpus_ids     = [0]
        self.iteration_type    = 3
        
        self.cc_threshold      = 0.8
        self.fsc_threshold     = 0.143
        
        self.max_2d_delta_angstroms  = 0
        self.max_tilt_reconstruction = -1
        self.type_2d_shift_fitting   = 'none' # affine / gaussian / tps
        self.tps_lambda              = 1.0
        self.fitting_from_origin     = True
        self.smooth_ctf              = False
        self.type_ctf_smoothing      = 'gaussian' # gaussian / tps
        self.ctf_tps_lambda          = 1.0
        self.reweight_classification = False
        
        self.mpi               = _dt.mpi_params('srun -n %d ',1)
        self.verbosity         = 1
        
        self.aligner           = _ssa_modules.Aligner()
        self.averager          = _ssa_modules.Averager()
        self.ctf_refiner       = _ssa_modules.CtfRefiner()
        
        self.aligner.ctf_correction    = 'on_reference'
        self.aligner.cc_type           = 'cfsc'
        self.aligner.halfsets_independ = False
        
        self.averager.ctf_correction    = 'wiener'
        self.averager.rec_halfsets      = True
        self.averager.bandpass.highpass = 0
        self.averager.bandpass.lowpass  = -1
    
    def get_iteration_dir(self, ite):
        """Return the directory path for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number (1-based).

        Returns
        -------
        str
            Path of the form ``<prj_name>/ite_NNNN/``.
        """
        return self.prj_name + '/ite_%04d/' % ite
    
    def get_iteration_files(self, ite):
        """Return the standard file paths for iteration *ite*.

        For ``ite < 1`` the initial files (:attr:`initial_particles` and
        :attr:`initial_reference`) are returned.

        Parameters
        ----------
        ite : int
            Iteration number.  Use ``0`` for the initial state.

        Returns
        -------
        _iteration_files
            Object with attributes ``ptcl_rslt``, ``ptcl_temp``,
            ``reference``, and ``ite_dir``.
        """
        rslt = _iteration_files()
        if ite < 1:
            rslt.ptcl_rslt = self.initial_particles
            rslt.reference = self.initial_reference
        else:
            base_dir = self.get_iteration_dir(ite)
            rslt.ptcl_rslt = base_dir + 'particles.ptclsraw'
            rslt.ptcl_temp = base_dir + 'temp.ptclsraw'
            rslt.reference = base_dir + 'reference.refstxt'
            rslt.ite_dir   = base_dir
        return rslt

    def get_names_map(self, ite, ref=1):
        """Return the path to the full reference map for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.  ``0`` returns the path from the initial
            ``.refstxt``.
        ref : int
            1-based reference (class) index.  Default: ``1``.

        Returns
        -------
        str
            Path to the MRC map file.
        """
        if ite == 0:
            refs_info = _ssa_data.Reference(self.initial_reference)
            map_name = refs_info.ref[ref-1]
        else:
            ite_dir = self.get_iteration_dir(ite)
            map_name = ite_dir + 'map_class%03d.mrc' % ref
        return map_name

    def get_names_mask(self, ite, ref=1):
        """Return the path to the soft mask for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.  ``0`` reads from the initial ``.refstxt``.
        ref : int
            1-based reference index.  Default: ``1``.

        Returns
        -------
        str
            Path to the mask MRC file.
        """
        if ite == 0:
            refs_info = _ssa_data.Reference(self.initial_reference)
            mask_name = refs_info.msk[ref-1]
        else:
            refs_info = _ssa_data.Reference(self.get_iteration_dir(ite)+'reference.refstxt')
            mask_name = refs_info.msk[ref-1]
        return mask_name

    def get_names_halfmaps(self, ite, ref=1):
        """Return the paths to the two half-maps for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.  ``0`` reads from the initial ``.refstxt``.
        ref : int
            1-based reference index.  Default: ``1``.

        Returns
        -------
        tuple of str
            ``(half1_path, half2_path)``.
        """
        if ite == 0:
            refs_info = _ssa_data.Reference(self.initial_reference)
            h1_name = refs_info.h1[ref-1]
            h2_name = refs_info.h2[ref-1]
        else:
            ite_dir = self.get_iteration_dir(ite)
            h1_name = ite_dir + 'map_class%03d_half1.mrc' % ref
            h2_name = ite_dir + 'map_class%03d_half2.mrc' % ref
        return (h1_name,h2_name)

    def get_name_refstxt(self, ite):
        """Return the path to the ``.refstxt`` file for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.  ``0`` returns :attr:`initial_reference`.

        Returns
        -------
        str
            Path to the ``.refstxt`` file.
        """
        files = self.get_iteration_files(ite)
        return files.reference

    def get_name_ptcls(self, ite):
        """Return the path to the ``.ptclsraw`` file for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.  ``0`` returns :attr:`initial_particles`.

        Returns
        -------
        str
            Path to the ``.ptclsraw`` file.
        """
        files = self.get_iteration_files(ite)
        return files.ptcl_rslt

    def get_map(self, ite, ref=1):
        """Load and return the reference map for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.
        ref : int
            1-based reference index.  Default: ``1``.

        Returns
        -------
        numpy.ndarray
            3-D map array.
        """
        v,_ = _mrc_read(self.get_names_map(ite,ref))
        return v

    def get_ptcls(self, ite):
        """Load and return the particles for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.

        Returns
        -------
        :class:`~susan.data.Particles`
            Particle container with aligned positions and scores.
        """
        files = self.get_iteration_files(ite)
        return _ssa_data.Particles(files.ptcl_rslt)

    def get_cc(self, ite, ref=1):
        """Return the per-particle cross-correlation scores for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.
        ref : int
            1-based reference index.  Default: ``1``.

        Returns
        -------
        numpy.ndarray, shape (N,)
            CC scores for all particles assigned to reference *ref*.
        """
        ptcls = self.get_ptcls(ite)
        return ptcls.ali_cc[ref-1]

    def get_fsc(self, ite, ref=1):
        """Compute and return the FSC curve for iteration *ite*.

        Parameters
        ----------
        ite : int
            Iteration number.
        ref : int
            1-based reference index.  Default: ``1``.

        Returns
        -------
        numpy.ndarray
            1-D FSC array indexed by Fourier shell.
        """
        i = ref-1
        refs = _ssa_data.Reference(self.get_name_refstxt(ite))
        return _ssa_utils.fsc_get(refs.h1[i],refs.h2[i],refs.msk[i])

    def setup_iteration(self, ite):
        """Prepare the directory and file-path objects for iteration *ite*.

        Creates the iteration directory if needed and validates that the
        previous iteration's output files exist.

        Parameters
        ----------
        ite : int
            Iteration number (must be ≥ 1).

        Returns
        -------
        tuple
            ``(cur, prv)`` — file-path objects for the current and previous
            iterations respectively.

        Raises
        ------
        NameError
            If the previous iteration's particle or reference files are
            missing.
        """
        base_dir = self.get_iteration_dir(ite)
        if not _file_exists(base_dir):
            _mkdir(base_dir)
        cur = self.get_iteration_files(ite)
        prv = self.get_iteration_files(ite-1)
        prv.check()
        return (cur,prv)
    
    def _validate_ite_type(self):
        if self.iteration_type in (3,'3','3D','3d'):
            return 3
        elif self.iteration_type in (2,'2','2D','2d'):
            return 2
        elif self.iteration_type in ('ctf','CTF','Ctf'):
            return 'ctf'
        else:
            raise ValueError('Invalid Iteration Type (accepted value: 3, 2, "ctf")')
    
    def _exec_alignment(self,cur,prv,ite_type):
        self.aligner.list_gpus_ids     = self.list_gpus_ids
        self.aligner.dimensionality    = ite_type
        self.aligner.verbosity         = self.verbosity
        
        print( '  [%dD Alignment] Start:' % ite_type )
        
        start_time = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.aligner.mpi.cmd = self.mpi.cmd
            self.aligner.mpi.arg = self.mpi.arg
            self.aligner.align_mpi(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        else:
            self.aligner.align(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        elapsed = _ssa_utils.time_now()-start_time
        
        print( '  [%dD Alignment] Finished. Elapsed time: %.1f seconds (%s).' % (ite_type,elapsed.total_seconds(),str(elapsed)) )

    def _exec_ctf_refinement(self,cur,prv):
        self.ctf_refiner.list_gpus_ids     = self.list_gpus_ids
        self.ctf_refiner.verbosity         = self.verbosity
        
        print( '  [CTF Refinement] Start:' )
        
        start_time = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.ctf_refiner.mpi.cmd = self.mpi.cmd
            self.ctf_refiner.mpi.arg = self.mpi.arg
            self.ctf_refiner.refine_mpi(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        else:
            self.ctf_refiner.refine(cur.ptcl_rslt,prv.reference,self.tomogram_file,prv.ptcl_rslt,self.box_size)
        elapsed = _ssa_utils.time_now()-start_time
        
        print( '  [CTF Refinement] Finished. Elapsed time: %.1f seconds (%s).' % (elapsed.total_seconds(),str(elapsed)) )

    def exec_estimation(self, cur, prv):
        """Run the alignment or CTF refinement step.

        Dispatches to :meth:`~susan.modules.Aligner.align` (or
        :meth:`~susan.modules.Aligner.align_mpi`) for 3-D/2-D iteration
        types, or to :meth:`~susan.modules.CtfRefiner.refine` for CTF
        iterations.  The type is determined by :attr:`iteration_type`.

        Parameters
        ----------
        cur : _iteration_files
            File paths for the current iteration (output).
        prv : _iteration_files
            File paths for the previous iteration (input).
        """
        ite_type  = self._validate_ite_type()
        
        if ite_type == 'ctf':
            self._exec_ctf_refinement(cur,prv)
        else:
            self._exec_alignment(cur,prv,ite_type)
    
    def _apply_2D_fixes(self,ptcls_in,cur,prv):

        def smooth_deltas(points, deltas, sigma, k):
            tree = _KDTree(points)
            smoothed_deltas = _np.zeros_like(deltas)
            for i, point in enumerate(points):
                distances, indices = tree.query(point, k=k)
                weights = _np.exp(-distances**2 / (2 * sigma**2))
                weights /= weights.sum()
                smoothed_deltas[i] = (deltas[indices] * weights[:,_np.newaxis]).sum(axis=0)
            return smoothed_deltas

        def tps_fit(pt0, deltas, lam):
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
            if self.max_2d_delta_angstroms > 0:
                if self.aligner.allow_drift:
                    print('    Limiting 2D drift to %.2f Å.' % self.max_2d_delta_angstroms )
                    ptcls_old  = _ssa_data.Particles(prv.ptcl_rslt)
                    delta_angs = ptcls_in.prj_t - ptcls_old.prj_t
                    norm_angs  = _np.linalg.norm( delta_angs, axis=2 )
                    scale_lim  = self.max_2d_delta_angstroms/_np.maximum(norm_angs,self.max_2d_delta_angstroms)
                    scale_lim[ norm_angs<self.max_2d_delta_angstroms ] = 1
                    scale_lim = scale_lim[:,:,_np.newaxis]
                    delta_angs = scale_lim*delta_angs
                    ptcls_in.prj_t[:] = ptcls_old.prj_t + delta_angs
                else:
                    print('    Limiting 2D shift to %.2f Å.' % self.max_2d_delta_angstroms )
                    norm_angs  = _np.linalg.norm( ptcls_in.prj_t, axis=2 )
                    scale_lim  = self.max_2d_delta_angstroms/_np.maximum(norm_angs,self.max_2d_delta_angstroms)
                    scale_lim[ norm_angs<self.max_2d_delta_angstroms ] = 1
                    scale_lim = scale_lim[:,:,_np.newaxis]
                    ptcls_in.prj_t[:] = scale_lim*ptcls_in.prj_t

        elif self.type_2d_shift_fitting.lower() == 'affine':
            R    = _np.eye(3, dtype=_np.float32)
            n    = ptcls_in.n_ptcl
            pt   = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                if idx.sum() < 4:
                    continue

                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix,i]).astype(_np.float32))
                    base = 0.0 if prj_base is None else prj_base[idx,i]
                    pt0 = pt[idx]@R.T
                    pt0 = pt0[:,:2]
                    pt1 = pt0 + (ptcls_in.prj_t[idx,i] - base)
                    pt0_aug = _np.hstack([pt0, _np.ones((pt0.shape[0],1))])
                    xform,_,_,_ = _np.linalg.lstsq(pt0_aug,pt1, rcond=None)
                    pt2 = pt0_aug @ xform
                    ptcls_in.prj_t[idx,i] = base + (pt2-pt0)

        elif self.type_2d_shift_fitting.lower() == 'gaussian':
            R    = _np.eye(3, dtype=_np.float32)
            n    = ptcls_in.n_ptcl
            pt   = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                n_ptcl_tomo = idx.sum()
                if n_ptcl_tomo < 2:
                    continue
                k_eff   = min(7, n_ptcl_tomo)

                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix,i]).astype(_np.float32))
                    base = 0.0 if prj_base is None else prj_base[idx,i]
                    pt0 = pt[idx]@R.T
                    pt0 = pt0[:,:2]
                    sigma = _np.median(_np.linalg.norm(pt0, axis=1)) * 0.25
                    ptcls_in.prj_t[idx,i] = base + smooth_deltas(pt0, ptcls_in.prj_t[idx,i] - base, sigma=sigma, k=k_eff)

        elif self.type_2d_shift_fitting.lower() == 'tps':
            R    = _np.eye(3, dtype=_np.float32)
            n    = ptcls_in.n_ptcl
            pt   = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                if idx.sum() < 4:
                    continue

                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix,i]).astype(_np.float32))
                    base = 0.0 if prj_base is None else prj_base[idx,i]
                    pt0 = pt[idx]@R.T
                    pt0 = pt0[:,:2]
                    ptcls_in.prj_t[idx,i] = base + tps_fit(pt0, ptcls_in.prj_t[idx,i] - base, self.tps_lambda)

        if self.smooth_ctf and self._validate_ite_type() == 'ctf':
            method = self.type_ctf_smoothing.lower()
            print('    Smoothing CTF defocus deltas (%s).' % method)
            ptcls_old = _ssa_data.Particles(prv.ptcl_rslt)
            delta_U   = ptcls_in.def_U - ptcls_old.def_U
            delta_V   = ptcls_in.def_V - ptcls_old.def_V
            R    = _np.eye(3, dtype=_np.float32)
            n    = ptcls_in.n_ptcl
            pt   = ptcls_in.position + ptcls_in.ali_t[ptcls_in.ref_cix, _np.arange(n)]
            tomos = _ssa_data.Tomograms(self.tomogram_file)
            min_n = 4 if method == 'tps' else 2
            for tcix in range(tomos.n_tomos):
                idx = ptcls_in.tomo_cix == tcix
                n_ptcl_tomo = idx.sum()
                if n_ptcl_tomo < min_n:
                    continue
                k_eff = min(7, n_ptcl_tomo)

                for i in range(tomos.num_proj[tcix]):
                    _ssa_utils.euZYZ_rotm(R, _np.deg2rad(tomos.proj_eZYZ[tcix,i]).astype(_np.float32))
                    pt0 = (pt[idx]@R.T)[:,:2]
                    d   = _np.stack([delta_U[idx,i], delta_V[idx,i]], axis=1)
                    if method == 'tps':
                        s = tps_fit(pt0, d, self.ctf_tps_lambda)
                    else:
                        sigma = _np.median(_np.linalg.norm(pt0, axis=1)) * 0.25
                        s = smooth_deltas(pt0, d, sigma=sigma, k=k_eff)
                    ptcls_in.def_U[idx,i] = ptcls_old.def_U[idx,i] + s[:,0]
                    ptcls_in.def_V[idx,i] = ptcls_old.def_V[idx,i] + s[:,1]

        ptcls_in.save(cur.ptcl_rslt)
    
    def _select_particles_reconstruction(self,ptcls_in):
        for i in range(ptcls_in.n_refs):
            idx = (ptcls_in.ref_cix == i).flatten()
            if _np.any( idx ):
                hid = ptcls_in.half_id[idx].flatten()
                ccc = ptcls_in.ali_cc [i,idx].flatten()

                n_rf = hid.shape[0]
                n_h1 = (hid==1).sum()
                n_h2 = (hid==2).sum()
                
                if n_h1 > 0:
                    th1 = _np.quantile(ccc[ hid==1 ], 1-self.cc_threshold)
                    hid[ (hid==1) & (ccc<th1) ] = 0

                if n_h2 > 0:
                    th2 = _np.quantile(ccc[ hid==2 ], 1-self.cc_threshold)
                    hid[ (hid==2) & (ccc<th2) ] = 0
            
                ptcls_in.half_id[idx] = hid
            
                print('    Class %2d: %7d particles [%7d].' % (i+1,n_rf,(hid >0).sum()) )
                print('      Half 1: %7d particles [%7d].'  % (    n_h1,(hid==1).sum()) )
                print('      Half 2: %7d particles [%7d].'  % (    n_h2,(hid==2).sum()) )
            else:
                print('    Class %2d: %7d particles.' % (i+1,0) )
                print('      Half 1: %7d particles.'  % ( 0 ) )
                print('      Half 2: %7d particles.'  % ( 0 ) )
            
        return ptcls_in[ (ptcls_in.half_id>0).flatten() ]
        
    def _limit_tilt_range_reconstruction(self,ptcls_out):
        tomos = _ssa_data.Tomograms(filename=self.tomogram_file)
        prj_w = _np.copy(ptcls_out.prj_w)
        v = self.max_tilt_reconstruction
        if isinstance(v, (list, tuple, _np.ndarray)):
            v = _np.asarray(v, dtype=_np.float32)
            print('    Restricting reconstruction to tilt range [%.2f, %.2f] degrees.' % (v.min(), v.max()) )
            _ssa_data.Particles.Geom.enable_by_tilt_range(ptcls_out,tomos,tilt_deg_min=v.min(),tilt_deg_max=v.max())
        else:
            print('    Restricting reconstruction to %.2f maximum tilt.' % v )
            _ssa_data.Particles.Geom.enable_by_tilt(ptcls_out,tomos,tilt_deg_max=v)
        ptcls_out.prj_w = ptcls_out.prj_w*prj_w
    
    def exec_particle_selection(self, cur, prv):
        """Select particles and prepare the input for reconstruction.

        Performs multi-reference classification (if ``n_refs > 1``), applies
        optional 2-D shift corrections (:attr:`max_2d_delta_angstroms`,
        :attr:`type_2d_shift_fitting`) for 2-D alignment and CTF iterations, filters
        particles by CC score (:attr:`cc_threshold`), and optionally limits
        the tilt range (:attr:`max_tilt_reconstruction`).  The selected
        particles are saved to ``cur.ptcl_temp``.

        Parameters
        ----------
        cur : _iteration_files
            File paths for the current iteration.
        prv : _iteration_files
            File paths for the previous iteration.
        """
        print('  [Aligned particles] Processing:')
        ptcls_in = _ssa_data.Particles(cur.ptcl_rslt)
        
        # Limit 2D shifts:
        should_fix_2D = (self.max_2d_delta_angstroms > 0) or self.type_2d_shift_fitting != 'none' or self.smooth_ctf
        if (self._validate_ite_type() in (2,'ctf')) and should_fix_2D:
            self._apply_2D_fixes(ptcls_in,cur,prv)
        
        # Classify
        if ptcls_in.n_refs > 1 :
            ptcls_in.ref_cix = _np.argmax(ptcls_in.ali_cc,axis=0).astype(_np.uint32)
            if type(self.reweight_classification) is bool:
                if self.reweight_classification:
                    total = ptcls_in.ali_cc.sum(axis=0)
                    total[total==0] = 1
                    ptcls_in.ali_cc = ptcls_in.ali_cc/total
            elif isinstance(self.reweight_classification, (int, float)):
                ptcls_in.ali_cc = _np.power(ptcls_in.ali_cc,self.reweight_classification)
                total = ptcls_in.ali_cc.sum(axis=0)
                total[total==0] = 1
                ptcls_in.ali_cc = ptcls_in.ali_cc/total
            ptcls_in.save(cur.ptcl_rslt)
        
        # Select particles for reconstruction
        ptcls_out = self._select_particles_reconstruction(ptcls_in)
        
        # Limit tilt range
        v = self.max_tilt_reconstruction
        _tilt_active = (v is not None) and (isinstance(v, (list, tuple, _np.ndarray)) or v >= 0)
        if _tilt_active:
            self._limit_tilt_range_reconstruction(ptcls_out)
        
        ptcls_out.save(cur.ptcl_temp)
        print('  [Aligned particles] Done.')
        
    def exec_averaging(self, cur, prv):
        """Reconstruct the reference maps and update the ``.refstxt`` file.

        Calls :meth:`~susan.modules.Averager.reconstruct` (or MPI variant),
        then updates ``cur.reference`` with the new map paths.

        Parameters
        ----------
        cur : _iteration_files
            File paths for the current iteration.
        prv : _iteration_files
            File paths for the previous iteration (provides the mask paths).
        """
        self.averager.list_gpus_ids     = self.list_gpus_ids
        self.averager.verbosity         = self.verbosity
        
        print( '  [Reconstruct Maps] Start:' )
        start_time = _ssa_utils.time_now()
        if self.mpi.arg > 1:
            self.averager.mpi.cmd = self.mpi.cmd
            self.averager.mpi.arg = self.mpi.arg
            self.averager.reconstruct_mpi(cur.ite_dir+'map',self.tomogram_file,cur.ptcl_temp,self.box_size)
        else:
            self.averager.reconstruct(cur.ite_dir+'map',self.tomogram_file,cur.ptcl_temp,self.box_size)
        elapsed = _ssa_utils.time_now()-start_time
        
        print( '  [Reconstruct Maps] Finished. Elapsed time: %.1f seconds (%s).' % (elapsed.total_seconds(),str(elapsed)) )
        
        _rm(cur.ptcl_temp)
        
        refs = _ssa_data.Reference(prv.reference)
        for i in range(refs.n_refs):
            refs.ref[i] = '%s/map_class%03d.mrc'       % (cur.ite_dir,i+1)
            refs.h1[i]  = '%s/map_class%03d_half1.mrc' % (cur.ite_dir,i+1)
            refs.h2[i]  = '%s/map_class%03d_half2.mrc' % (cur.ite_dir,i+1)
        refs.save(cur.reference)
    
    def exec_postprocessing(self, cur):
        """Compute FSC-based resolution estimates for all references.

        Parameters
        ----------
        cur : _iteration_files
            File paths for the current iteration.

        Returns
        -------
        float or numpy.ndarray
            Estimated resolution in Fourier pixels at the
            :attr:`fsc_threshold` level.  A scalar for single-reference
            projects; a 1-D array for multi-reference projects.
        """
        refs = _ssa_data.Reference(cur.reference)
        if refs.n_refs == 1:
            print( '  [FSC Calculation] Start (1 reference):' )
        else:
            print( '  [FSC Calculation] Start (%d references):' % refs.n_refs )
        
        rslt = _np.zeros( (refs.n_refs) )
        for i in range(refs.n_refs):
            fsc = _ssa_utils.fsc_get(refs.h1[i],refs.h2[i],refs.msk[i])
            _,pix_size,_ = _mrc_get_info(refs.ref[i])
            fsc_rslt = _ssa_utils.fsc_analyse(fsc,pix_size,self.fsc_threshold)
            print('    - Reference %2d: %7.3f angstroms [%d fourier pixels]' % (i+1,fsc_rslt.res,fsc_rslt.fpix))
            rslt[i] = fsc_rslt.fpix
        
        if refs.n_refs == 1:
            return rslt[0]
        else:
            return rslt
    
    def execute_iteration(self, ite):
        """Run a complete STA iteration.

        Executes :meth:`setup_iteration`, :meth:`exec_estimation`,
        :meth:`exec_particle_selection`, :meth:`exec_averaging`, and
        :meth:`exec_postprocessing` in sequence.

        Parameters
        ----------
        ite : int
            Iteration number (must be ≥ 1).  If the iteration directory
            already exists its results are overwritten.

        Returns
        -------
        float or numpy.ndarray
            Estimated resolution in Fourier pixels (see
            :meth:`exec_postprocessing`).

        Notes
        -----
        Iteration ``0`` is the project seed and cannot be processed.  In
        that case a warning is issued, the iteration is skipped, and the
        configured starting lowpass is returned instead — the lowpass of
        :attr:`aligner` for a 3-D/2-D iteration or :attr:`ctf_refiner` for a
        CTF iteration.  For a multi-reference project a
        :class:`numpy.ndarray` of length ``n_refs`` filled with that value
        is returned; otherwise a scalar.
        """
        if ite < 1:
            _warnings.warn(
                'execute_iteration(%d): iteration 0 is the project seed and '
                'cannot be run; returning the configured starting lowpass '
                'instead.' % ite, stacklevel=2)
            print('============================')
            print('Project: %s (Iteration %d) Skipped.'%(self.prj_name,ite))
            if self._validate_ite_type() == 'ctf':
                lowpass = self.ctf_refiner.bandpass.lowpass
            else:
                lowpass = self.aligner.bandpass.lowpass
            n_refs = _ssa_data.Reference(self.get_name_refstxt(0)).n_refs
            if n_refs > 1:
                return _np.full(n_refs, lowpass, dtype=_np.float32)
            return lowpass

        start_time = _ssa_utils.time_now()
        print('============================')
        print('Project: %s (Iteration %d)'%(self.prj_name,ite))
        cur,prv = self.setup_iteration(ite)
        self.exec_estimation(cur,prv)
        self.exec_particle_selection(cur,prv)
        self.exec_averaging(cur,prv)
        rslt = self.exec_postprocessing(cur)
        elapsed = _ssa_utils.time_now()-start_time
        print('Iteration %d Finished [Elapsed time: %.1f seconds (%s)]'%(ite,elapsed.total_seconds(),str(elapsed)))
        return rslt


Manager = STA  # Alias for back-compatibility

