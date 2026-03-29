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

import os as _os
import susan.utils.datatypes as _dt

def _get_gpu_str(list_gpus_ids):
    gpu_str = ','.join( str(num) for num in  list_gpus_ids )
    return gpu_str

###############################################################################

class Aligner:
    """Particle alignment engine for 3-D and 2-D subtomogram averaging.

    Wraps the ``susan_aligner`` binary.  Configure the attributes, then call
    :meth:`align` (single-node) or :meth:`align_mpi` (multi-node MPI).

    .. rubric:: Attributes

    Attributes
    ----------
    list_gpus_ids : list of int
        GPU device IDs to use.  Default: ``[0]``.
    threads_per_gpu : int
        *Deprecated.*  Must remain ``1``; use additional entries in
        ``list_gpus_ids`` to increase parallelism instead.
    bandpass : :class:`~susan.utils.datatypes.bandpass`
        Frequency bandpass applied to both particle and reference.
        Default: ``bandpass(0, -1, 2)`` (full range, 2-pixel rolloff).
    dimensionality : int
        Alignment search space: ``3`` for full 3-D, ``2`` for in-plane only.
        Default: ``3``.
    extra_padding : int
        Extra zero-padding (pixels) added on each side before FFT.
        Default: ``0``.
    allow_drift : bool
        If ``True``, 2-D shifts accumulate across iterations (drift mode).
        Default: ``True``.
    halfsets_independ : bool
        Process the two half-sets with independent references.
        Default: ``False``.
    ignore_classes : bool
        Ignore reference-class assignments; align against all references.
        Default: ``False``.
    cone : :class:`~susan.utils.datatypes.search_params`
        Out-of-plane (cone) angular search range and step in degrees.
        Default: ``search_params(0, 1)`` (no search).
    inplane : :class:`~susan.utils.datatypes.search_params`
        In-plane angular search range and step in degrees.
        Default: ``search_params(0, 1)`` (no search).
    refine : :class:`~susan.utils.datatypes.refine_params`
        Multi-level angular refinement policy.
        Default: ``refine_params(0, 1)`` (no refinement).
    offset : :class:`~susan.utils.datatypes.offset_params`
        Translational search range, step, and shape.
        Default: ``offset_params([4, 4, 4], 1, 'ellipsoid')``.
    offset_space : str
        Coordinate frame for the offset search: ``'reference'`` or
        ``'tomogram'``.  Default: ``'reference'``.
    padding_type : str
        Fill value for the padded region: ``'zero'`` or ``'noise'``.
        Default: ``'zero'``.
    normalize_type : str
        Per-substack normalisation applied before correlation.  One of
        ``'none'``, ``'zero_mean'``, ``'zero_mean_one_std'``,
        ``'zero_mean_proj_weight'``, ``'poisson_raw'``,
        ``'poisson_normal'``.  Default: ``'zero_mean_one_std'``.
    ctf_correction : str
        CTF correction strategy.  One of ``'none'``, ``'phase_flip'``,
        ``'on_reference'``, ``'on_substack'``, ``'wiener_ssnr'``.
        ``'cfsc'`` is a deprecated alias for ``'wiener_ssnr'``.
        Default: ``'on_reference'``.
    cc_type : str
        Cross-correlation variant used for scoring: ``'basic'`` or
        ``'cfsc'``.  Default: ``'basic'``.
    cc_stats_type : str
        Post-CC statistics normalisation: ``'none'``, ``'probability'``, or
        ``'sigma'``.  Default: ``'none'``.
    pseudo_symmetry : str
        Symmetry group applied to the angular search grid.  Default:
        ``'c1'``.

        .. list-table::
           :header-rows: 1
           :widths: 30 70

           * - Value
             - Description
           * - ``'c1'`` / ``'none'``
             - No symmetry (identity).
           * - ``'cN'`` / ``'CN'``
             - Cyclic *N*-fold (e.g. ``'c4'``).
           * - ``'dN'`` / ``'DN'``
             - Dihedral *N*-fold (e.g. ``'d2'``).
           * - ``'cbo'`` / ``'CBO'``
             - Cuboctahedral (order 24).
           * - ``'ico'`` / ``'ICO'`` / ``'i2'`` / ``'I2'``
             - Icosahedral, I2 convention (order 60; RELION default).
           * - ``'i1'`` / ``'I1'``
             - Icosahedral, I1 convention.
           * - ``'i3'`` / ``'I3'``
             - Icosahedral, I3 convention.
           * - ``'i4'`` / ``'I4'``
             - Icosahedral, I4 convention.
           * - ``'cone_flip'`` / ``'y_180'``
             - 180° rotation about the Y axis.
    ssnr : :class:`~susan.utils.datatypes.ssnr`
        Ad-hoc SSNR model used for CTF weighting.
        Default: ``ssnr(0, 0.001)``.
    mpi : :class:`~susan.utils.datatypes.mpi_params`
        MPI launcher configuration used by :meth:`align_mpi`.
        Default: ``mpi_params('srun -n %d ', 1)``.
    verbosity : int
        Verbosity level passed to the binary (0 = silent).  Default: ``0``.
    """

    def __init__(self):
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.bandpass          = _dt.bandpass(0,-1,2)
        self.dimensionality    = 3
        self.extra_padding     = 0
        self.allow_drift       = True
        self.halfsets_independ = False
        self.ignore_classes    = False
        self.cone              = _dt.search_params(0,1)
        self.inplane           = _dt.search_params(0,1)
        self.refine            = _dt.refine_params(0,1)
        self.offset            = _dt.offset_params([4,4,4],1,'ellipsoid')
        self.offset_space      = 'reference'
        self.padding_type      = 'zero'
        self.normalize_type    = 'zero_mean_one_std'
        self.ctf_correction    = 'on_reference'
        self.cc_type           = 'basic'
        self.cc_stats_type     = 'none'
        self.pseudo_symmetry   = 'c1'
        self.ssnr              = _dt.ssnr(0,0.001)
        self.mpi               = _dt.mpi_params('srun -n %d ',1)
        self.verbosity         = 0
        self.tm_type           = "none"
        self.tm_prefix         = "template_matching"
        self.tm_sigma          = 0
        self.dilate            = 0
        self.expfilt_gain      = 1
        
    def set_angular_search(self, c_r=0, c_s=1, i_r=0, i_s=1):
        """Set the cone and in-plane angular search parameters.

        Convenience wrapper that writes to :attr:`cone` and :attr:`inplane`
        in one call.

        Parameters
        ----------
        c_r : float
            Cone (out-of-plane) search range in degrees.  ``0`` disables
            the cone search.  Default: ``0``.
        c_s : float
            Cone search angular step in degrees.  Default: ``1``.
        i_r : float
            In-plane search range in degrees.  ``0`` disables the in-plane
            search.  Default: ``0``.
        i_s : float
            In-plane search angular step in degrees.  Default: ``1``.
        """
        self.cone.span    = c_r
        self.cone.step    = c_s
        self.inplane.span = i_r
        self.inplane.step = i_s
        
    def set_offset_search(self, off_range, off_step=1, off_type='ellipsoid'):
        """Set the translational offset search parameters.

        Parameters
        ----------
        off_range : float or sequence of float
            Search range in pixels.

            * **scalar** — same range applied to X, Y, and Z.
            * **2-element sequence** — ``[XY, Z]``: same range for X and Y,
              separate range for Z.
            * **3-element sequence** — ``[X, Y, Z]``: independent range per
              axis.
        off_step : float
            Search step size in pixels.  Default: ``1``.
        off_type : str
            Shape of the search volume.  For 3-D alignment: ``'ellipsoid'``
            (default), ``'cylinder'``, or ``'cuboid'``.  For 2-D alignment:
            ``'circle'`` (``'ellipsoid'``) or ``'rectangle'`` (``'cuboid'``).

        Raises
        ------
        ValueError
            If *off_type* is not valid for the current :attr:`dimensionality`,
            or if *off_range* has more than 3 elements.
        """
        if (self.dimensionality == 3) and (not off_type in ['ellipsoid', 'cylinder', 'cuboid']):
            raise ValueError('Invalid offset type. Only "ellipsoid", "cylinder" or "cuboid" are valid for the 3D alignment')
        if (self.dimensionality == 2) and (not off_type in ['ellipsoid', 'cuboid', 'circle', 'rectangle']):
            raise ValueError('Invalid offset type. Only "circle" ("ellipsoid") and "rectangle" ("cuboid") are valid for the 2D alignment')
        
        if isinstance(off_range,int) or isinstance(off_range,float):
            self.offset.span = (off_range,off_range,off_range)
        elif len(off_range) == 3:
            self.offset.span = off_range
        elif len(off_range) == 2:
            self.offset.span = (off_range[0],off_range[0],off_range[1])
        else:
            raise ValueError('Offset range can have up to 3 elements.')
        self.offset.step = off_step
        self.offset.kind = off_type
        
    def _validate(self):
        if not self.dimensionality in [2,3]:
            raise ValueError('Invalid dimensionality type. Only 2 or 3 are valid')
        
        if (self.dimensionality == 3) and (not self.offset.kind in ['ellipsoid', 'cylinder', 'cuboid']):
            raise ValueError('Invalid offset type. Only "ellipsoid", "cylinder" or "cuboid" are valid for the 3D alignment')
        if (self.dimensionality == 2) and (not self.offset.kind in ['ellipsoid', 'cuboid', 'circle', 'rectangle']):
            raise ValueError('Invalid offset type. Only "circle" ("ellipsoid") and "rectangle" ("cuboid") are valid for the 2D alignment')

        if not self.offset_space in ['reference','tomogram']:
            raise ValueError('Invalid offset space. Only "reference" or "tomogram" are valid')

        if not self.padding_type in ['zero','noise']:
            raise ValueError('Invalid padding type. Only "zero" or "noise" are valid')
        
        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight','poisson_raw','poisson_normal']:
            raise ValueError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std", "zero_mean_proj_weight", "poisson_raw" or "poisson_normal" are valid')
        
        if not self.ctf_correction in ['none','on_reference','on_substack','wiener_ssnr','cfsc']:
            raise ValueError('Invalid ctf correction type. Only "none", "on_reference", "on_substack", "wiener_ssnr" or "cfsc" are valid')
        
        if not self.cc_stats_type in ['none','probability','sigma']:
            raise ValueError('Invalid cc statistic method. Only "none", "probability" or "sigma" are valid')
        
        if not self.offset.step > 0 or not self.cone.step > 0 or not self.inplane.step > 0:
            raise ValueError('The steps values must be larger than 0')

        if (self.offset.span[0] > 0 and self.offset.span[0] < self.offset.step) or (self.offset.span[1] > 0 and self.offset.span[1] < self.offset.step) or (self.offset.span[2] > 0 and self.offset.span[2] < self.offset.step):
            raise ValueError('Offset: Step cannot be larger than Range/Span')

        if self.cone.span == 0:
            self.cone.step = 1
        else:
            if self.cone.span < self.cone.step:
                raise ValueError('Cone: Step cannot be larger than Range/Span')

        if self.inplane.span == 0:
            self.inplane.step = 1
        else:
            if self.inplane.span < self.inplane.step:
                raise ValueError('Inplane: Step cannot be larger than Range/Span')

    def get_args(self, ptcls_out, refs_file, tomos_file, ptcls_in, box_size):
        """Build the command-line argument string for ``susan_aligner``.

        Parameters
        ----------
        ptcls_out : str
            Path for the output ``.ptclsraw`` file with updated alignments.
        refs_file : str
            Path to the input ``.refstxt`` references file.
        tomos_file : str
            Path to the input ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Returns
        -------
        str
            Space-separated argument string ready to be appended to the
            ``susan_aligner`` command.
        """
        self._validate()
        if self.bandpass.lowpass <= 0:
            self.bandpass.lowpass = (box_size/2) - 1
        n_threads = len(self.list_gpus_ids)*self.threads_per_gpu
        gpu_str   = _get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_file '      + tomos_file
        args = args + ' -ptcls_in '        + ptcls_in
        args = args + ' -ptcls_out '       + ptcls_out
        args = args + ' -refs_file '       + refs_file
        args = args + ' -n_threads %d'     % n_threads
        args = args + ' -gpu_list '        + gpu_str
        args = args + ' -box_size %d'      % box_size
        args = args + ' -pad_size %d'      % self.extra_padding
        args = args + ' -pad_type '        + self.padding_type
        args = args + ' -cc_type '         + self.cc_type
        args = args + ' -cc_stats '        + self.cc_stats_type
        args = args + ' -norm_type '       + self.normalize_type
        args = args + ' -ctf_type '        + self.ctf_correction
        args = args + ' -ssnr_param %f,%f' % (self.ssnr.F,self.ssnr.S)
        args = args + ' -bandpass %f,%f'   % (self.bandpass.highpass,self.bandpass.lowpass)
        args = args + ' -rolloff_f %f'     % self.bandpass.rolloff
        args = args + ' -p_symmetry '      + self.pseudo_symmetry
        args = args + ' -ali_halves %d'    % self.halfsets_independ
        args = args + ' -ignore_ref %d'    % self.ignore_classes
        args = args + ' -allow_drift %d'   % self.allow_drift
        args = args + ' -cone %f,%f'       % (self.cone.span,self.cone.step)
        args = args + ' -inplane %f,%f'    % (self.inplane.span,self.inplane.step)
        args = args + ' -refine %d,%d'     % (self.refine.factor,self.refine.levels)
        args = args + ' -off_type '        + self.offset.kind
        args = args + ' -off_params %f,%f,%f,%f' % (self.offset.span[0],self.offset.span[1],self.offset.span[2],self.offset.step)
        args = args + ' -off_space '       + self.offset_space
        args = args + ' -type %d'          % self.dimensionality
        args = args + ' -dilate %d'        % self.dilate
        args = args + ' -verbosity %d'     % self.verbosity
        args = args + ' -tm_type '         + self.tm_type
        args = args + ' -tm_prefix '       + self.tm_prefix
        args = args + ' -tm_sigma %f'      % self.tm_sigma
        args = args + ' -expfilt_gain %f'  % self.expfilt_gain
        return args
    
    def align(self, ptcls_out, refs_file, tomos_file, ptcls_in, box_size):
        """Execute the alignment on a single node.

        Parameters
        ----------
        ptcls_out : str
            Path for the output ``.ptclsraw`` file.
        refs_file : str
            Path to the ``.refstxt`` references file.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the ``susan_aligner`` binary returns a non-zero exit code.
        """
        cmd = 'susan_aligner ' + self.get_args(ptcls_out, refs_file, tomos_file, ptcls_in, box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the alignment: ' + cmd)

    def align_mpi(self, ptcls_out, refs_file, tomos_file, ptcls_in, box_size):
        """Execute the alignment using MPI across multiple nodes.

        The MPI command is taken from :attr:`mpi`.

        Parameters
        ----------
        ptcls_out : str
            Path for the output ``.ptclsraw`` file.
        refs_file : str
            Path to the ``.refstxt`` references file.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the MPI binary returns a non-zero exit code.
        """
        cmd = self.mpi.gen_cmd() + ' ' + _os.path.dirname(_os.path.abspath(__file__)) + '/bin/susan_aligner_mpi ' + self.get_args(ptcls_out, refs_file, tomos_file, ptcls_in, box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the alignment: ' + cmd)

###############################################################################

class Averager:
    """Map reconstruction (averaging) engine for subtomogram averaging.

    Wraps the ``susan_reconstruct`` binary.  Configure the attributes, then
    call :meth:`reconstruct` (single-node) or :meth:`reconstruct_mpi`
    (multi-node MPI).

    .. rubric:: Attributes

    Attributes
    ----------
    list_gpus_ids : list of int
        GPU device IDs to use.  Default: ``[0]``.
    threads_per_gpu : int
        *Deprecated.*  Must remain ``1``; use additional entries in
        ``list_gpus_ids`` to increase parallelism instead.
    bandpass : :class:`~susan.utils.datatypes.bandpass`
        Frequency bandpass applied during back-projection.
        Default: ``bandpass(0, -1, 2)`` (full range, 2-pixel rolloff).
    extra_padding : int
        Extra zero-padding (pixels) added on each side before FFT.
        Default: ``0``.
    rec_halfsets : bool
        If ``True``, reconstruct separate half-maps (needed for FSC).
        Default: ``False``.
    padding_type : str
        Fill value for the padded region: ``'zero'`` or ``'noise'``.
        Default: ``'zero'``.
    normalize_type : str
        Per-substack normalisation.  One of ``'none'``, ``'zero_mean'``,
        ``'zero_mean_one_std'``, ``'zero_mean_proj_weight'``.
        Default: ``'zero_mean_one_std'``.
    weighting_type : str
        Particle weighting scheme.  One of ``'none'``, ``'particle'``,
        ``'projection'``, ``'3DCC'``, ``'2DCC'``.  Default: ``'none'``.
    ctf_correction : str
        CTF correction applied during back-projection.  One of ``'none'``,
        ``'phase_flip'``, ``'wiener'``, ``'wiener_ssnr'``;
        ``'wiener_atan'`` and ``'wiener_lgstc'`` are *experimental*.
        Default: ``'wiener'``.
    gridding_type : str
        Fourier-space gridding method: ``'linear'`` or ``'kb'``
        (Kaiser–Bessel).  Default: ``'linear'``.
    symmetry : str
        Point-group symmetry applied to the reconstructed map.  Default:
        ``'c1'``.

        .. list-table::
           :header-rows: 1
           :widths: 30 70

           * - Value
             - Description
           * - ``'c1'`` / ``'none'``
             - No symmetry (identity).
           * - ``'cN'`` / ``'CN'``
             - Cyclic *N*-fold (e.g. ``'c4'``).
           * - ``'dN'`` / ``'DN'``
             - Dihedral *N*-fold (e.g. ``'d2'``).
           * - ``'cbo'`` / ``'CBO'``
             - Cuboctahedral (order 24).
           * - ``'ico'`` / ``'ICO'`` / ``'i2'`` / ``'I2'``
             - Icosahedral, I2 convention (order 60; RELION default).
           * - ``'i1'`` / ``'I1'``
             - Icosahedral, I1 convention.
           * - ``'i3'`` / ``'I3'``
             - Icosahedral, I3 convention.
           * - ``'i4'`` / ``'I4'``
             - Icosahedral, I4 convention.
           * - ``'cone_flip'`` / ``'y_180'``
             - 180° rotation about the Y axis.
    ssnr : :class:`~susan.utils.datatypes.ssnr`
        Ad-hoc SSNR model for Wiener filter denominator.
        Default: ``ssnr(1, 0.01)``.
    inversion : :class:`~susan.utils.datatypes.inversion_params`
        Parameters for iterative sampling-function inversion.
        Default: ``inversion_params(10, 0.75)``.
    mpi : :class:`~susan.utils.datatypes.mpi_params`
        MPI launcher configuration used by :meth:`reconstruct_mpi`.
        Default: ``mpi_params('srun -n %d ', 1)``.
    verbosity : int
        Verbosity level passed to the binary.  Default: ``1``.
    normalize_output : bool
        Normalise the output map to unit standard deviation.  Default: ``True``.
    ignore_classes : bool
        Ignore reference-class assignments; reconstruct all particles.
        Default: ``False``.
    boost_lowfreq : :class:`~susan.utils.datatypes.boost_lowfreq_params`
        *Experimental.*  Optional low-frequency boost before reconstruction.
        Default: ``boost_lowfreq_params(0, 0, 0)`` (disabled).
    """

    def __init__(self):
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.bandpass          = _dt.bandpass(0,-1,2)
        self.extra_padding     = 0
        self.rec_halfsets      = False
        self.padding_type      = 'zero'
        self.normalize_type    = 'zero_mean_one_std'
        self.weighting_type    = 'none'
        self.ctf_correction    = 'wiener'
        self.gridding_type     = 'linear'
        self.symmetry          = 'c1'
        self.ssnr              = _dt.ssnr(1,0.01)
        self.inversion         = _dt.inversion_params(10,0.75)
        self.mpi               = _dt.mpi_params('srun -n %d ',1)
        self.verbosity         = 1
        self.normalize_output  = True
        self.ignore_classes    = False
        self.boost_lowfreq     = _dt.boost_lowfreq_params(0,0,0)
        
    def _validate(self):
        if not self.padding_type in ['zero','noise']:
            raise ValueError('Invalid padding type. Only "zero" or "noise" are valid')

        if not self.gridding_type in ['linear','kb']:
            raise ValueError('Invalid gridding type. Only "linear" or "kb" are valid')

        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight']:
            raise ValueError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std" or "zero_mean_proj_weight" are valid')

        if not self.weighting_type in ['none','particle','projection','3DCC','2DCC']:
            raise ValueError('Invalid weighting type. Only "none", "particle", "projection", "3DCC" or "2DCC" are valid')

        if not self.ctf_correction in ['none','phase_flip','wiener','wiener_ssnr','wiener_atan','wiener_lgstc']:
            raise ValueError('Invalid ctf correction type. Only "none", "phase_flip", "wiener", "wiener_ssnr", "wiener_atan" or "wiener_lgstc" are valid')
            
    def get_args(self, out_pfx, tomos_file, ptcls_in, box_size):
        """Build the command-line argument string for ``susan_reconstruct``.

        Parameters
        ----------
        out_pfx : str
            Output path prefix; maps are written as ``<out_pfx>_class001.mrc``
            etc.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Returns
        -------
        str
            Space-separated argument string ready to be appended to the
            ``susan_reconstruct`` command.
        """
        self._validate()
        if self.bandpass.lowpass <= 0:
            self.bandpass.lowpass = (box_size/2) - 1
        n_threads = len(self.list_gpus_ids)*self.threads_per_gpu
        gpu_str   = _get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_file '      + tomos_file
        args = args + ' -out_prefix '      + out_pfx
        args = args + ' -ptcls_file '      + ptcls_in
        args = args + ' -n_threads %d'     % n_threads
        args = args + ' -gpu_list '        + gpu_str
        args = args + ' -box_size %d'      % box_size
        args = args + ' -pad_size %d'      % self.extra_padding
        args = args + ' -pad_type '        + self.padding_type
        args = args + ' -norm_type '       + self.normalize_type
        args = args + ' -ctf_type '        + self.ctf_correction
        args = args + ' -wgt_type '        + self.weighting_type
        args = args + ' -grid_type '       + self.gridding_type
        args = args + ' -ssnr_param %f,%f' % (self.ssnr.F,self.ssnr.S)
        args = args + ' -w_inv_iter %d'    % self.inversion.ite
        args = args + ' -w_inv_gstd %f'    % self.inversion.std
        args = args + ' -bandpass %f,%f'   % (self.bandpass.highpass,self.bandpass.lowpass)
        args = args + ' -rolloff_f %f'     % self.bandpass.rolloff
        args = args + ' -symmetry '        + self.symmetry
        args = args + ' -rec_halves %d'    % self.rec_halfsets
        args = args + ' -ignore_ref %d'    % self.ignore_classes
        args = args + ' -verbosity %d'     % self.verbosity
        args = args + ' -norm_output %d'   % self.normalize_output
        args = args + ' -boost_lowfq %f,%f,%f' % (self.boost_lowfreq.scale,self.boost_lowfreq.value,self.boost_lowfreq.decay)
        return args
    
    def reconstruct(self, out_pfx, tomos_file, ptcls_in, box_size):
        """Execute the reconstruction on a single node.

        Parameters
        ----------
        out_pfx : str
            Output path prefix for the reconstructed maps.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the ``susan_reconstruct`` binary returns a non-zero exit code.
        """
        cmd = 'susan_reconstruct ' + self.get_args(out_pfx,tomos_file,ptcls_in,box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the reconstruction: ' + cmd)

    def reconstruct_mpi(self, out_pfx, tomos_file, ptcls_in, box_size):
        """Execute the reconstruction using MPI across multiple nodes.

        The MPI command is taken from :attr:`mpi`.

        Parameters
        ----------
        out_pfx : str
            Output path prefix for the reconstructed maps.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the MPI binary returns a non-zero exit code.
        """
        cmd = self.mpi.gen_cmd() + ' ' + _os.path.dirname(_os.path.abspath(__file__)) + '/bin/susan_reconstruct_mpi' + self.get_args(out_pfx,tomos_file,ptcls_in,box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the reconstruction: ' + cmd)

###############################################################################

class SubtomoRec:
    """Subtomogram reconstruction engine.

    Wraps the ``susan_rec_subtomos`` binary.  Each selected particle is
    reconstructed as an individual volume and written to *out_dir*.  Configure
    the attributes, then call :meth:`reconstruct`.

    .. rubric:: Attributes

    Attributes
    ----------
    list_gpus_ids : list of int
        GPU device IDs to use.  Default: ``[0]``.
    threads_per_gpu : int
        *Deprecated.*  Must remain ``1``; use additional entries in
        ``list_gpus_ids`` to increase parallelism instead.
    bandpass : :class:`~susan.utils.datatypes.bandpass`
        Frequency bandpass applied during back-projection.
        Default: ``bandpass(0, -1, 2)`` (full range, 2-pixel rolloff).
    extra_padding : int
        Extra zero-padding (pixels) added on each side before FFT.
        Default: ``0``.
    padding_type : str
        Fill value for the padded region: ``'zero'`` or ``'noise'``.
        Default: ``'zero'``.
    normalize_type : str
        Per-substack normalisation.  One of ``'none'``, ``'zero_mean'``,
        ``'zero_mean_one_std'``, ``'zero_mean_proj_weight'``.
        Default: ``'none'``.
    ctf_correction : str
        CTF correction applied during back-projection.  One of ``'none'``,
        ``'phase_flip'``, ``'wiener'``, ``'wiener_ssnr'``, ``'pre_wiener'``.
        Default: ``'phase_flip'``.
    format : str
        Output file format: ``'mrc'`` or ``'em'``.  Default: ``'mrc'``.
    ssnr : :class:`~susan.utils.datatypes.ssnr`
        Ad-hoc SSNR model for Wiener filter denominator.
        Default: ``ssnr(1, 0.01)``.
    inversion : :class:`~susan.utils.datatypes.inversion_params`
        Parameters for iterative sampling-function inversion.
        Default: ``inversion_params(0, 0.75)`` (inversion disabled).
    use_align : bool
        Apply stored 3-D alignment offsets during reconstruction.
        Default: ``False``.
    relion_ctf : bool
        Use RELION-style CTF convention (flipped sign).  Default: ``False``.
    invert_contrast : bool
        Invert the sign of the output volume.  Default: ``False``.
    verbosity : int
        Verbosity level passed to the binary.  Default: ``0``.
    normalize_output : bool
        Normalise the output volume to unit standard deviation.
        Default: ``False``.
    boost_lowfreq : :class:`~susan.utils.datatypes.boost_lowfreq_params`
        Optional low-frequency boost before reconstruction.
        Default: ``boost_lowfreq_params(0, 3, 10)``.
    """

    def __init__(self):
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.bandpass          = _dt.bandpass(0,-1,2)
        self.extra_padding     = 0
        self.padding_type      = 'zero'
        self.normalize_type    = 'none'
        self.ctf_correction    = 'phase_flip'
        self.format            = 'mrc'
        self.ssnr              = _dt.ssnr(1,0.01)
        self.inversion         = _dt.inversion_params(0,0.75)
        self.use_align         = False
        self.relion_ctf        = False
        self.invert_contrast   = False
        self.verbosity         = 0
        self.normalize_output  = False
        self.boost_lowfreq     = _dt.boost_lowfreq_params(0,3,10)

    def _validate(self):
        if not self.padding_type in ['zero','noise']:
            raise ValueError('Invalid padding type. Only "zero" or "noise" are valid')

        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight']:
            raise ValueError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std" or "zero_mean_proj_weight" are valid')

        if not self.ctf_correction in ['none','phase_flip','wiener','wiener_ssnr', 'pre_wiener']:
            raise ValueError('Invalid ctf correction type. Only "none", "phase_flip", "wiener", "pre_wiener" or "wiener_ssnr" are valid')

        if not self.format in ['mrc','em']:
            raise ValueError('Invalid output format. Only "mrc" or "em" are valid')

    def get_args(self, out_dir, tomos_file, ptcls_in, box_size):
        """Build the command-line argument string for ``susan_rec_subtomos``.

        Parameters
        ----------
        out_dir : str
            Output directory where individual subtomogram files are written.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Returns
        -------
        str
            Space-separated argument string ready to be appended to the
            ``susan_rec_subtomos`` command.
        """
        self._validate()
        if self.bandpass.lowpass <= 0:
            self.bandpass.lowpass = box_size/2-1
        n_threads = len(self.list_gpus_ids)*self.threads_per_gpu
        gpu_str   = _get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_file '      + tomos_file
        args = args + ' -out_dir '         + out_dir
        args = args + ' -ptcls_file '      + ptcls_in
        args = args + ' -n_threads %d'     % n_threads
        args = args + ' -gpu_list '        + gpu_str
        args = args + ' -box_size %d'      % box_size
        args = args + ' -pad_size %d'      % self.extra_padding
        args = args + ' -pad_type '        + self.padding_type
        args = args + ' -norm_type '       + self.normalize_type
        args = args + ' -ctf_type '        + self.ctf_correction
        args = args + ' -format '          + self.format
        args = args + ' -bandpass %f,%f'   % (self.bandpass.highpass,self.bandpass.lowpass)
        args = args + ' -rolloff_f %f'     % self.bandpass.rolloff
        args = args + ' -ssnr_param %f,%f' % (self.ssnr.F,self.ssnr.S)
        args = args + ' -w_inv_iter %d'    % self.inversion.ite
        args = args + ' -w_inv_gstd %f'    % self.inversion.std
        args = args + ' -use_align %d'     % self.use_align
        args = args + ' -relion_ctf %d'    % self.relion_ctf
        args = args + ' -invert %d'        % self.invert_contrast
        args = args + ' -norm_output %d'   % self.normalize_output
        args = args + ' -boost_lowfq %f,%f,%f' % (self.boost_lowfreq.scale,self.boost_lowfreq.value,self.boost_lowfreq.decay)
        return args
    
    def reconstruct(self, out_pfx, tomos_file, ptcls_in, box_size):
        """Execute the subtomogram reconstruction on a single node.

        Parameters
        ----------
        out_pfx : str
            Output directory for the reconstructed subtomograms.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the ``susan_rec_subtomos`` binary returns a non-zero exit code.
        """
        cmd = 'susan_rec_subtomos ' + self.get_args(out_pfx,tomos_file,ptcls_in,box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the subtomogram reconstruction: ' + cmd)

###############################################################################

class CropProjection:
    """Projection-cropping engine for 2-D subtomogram alignment.

    Wraps the ``susan_crop_projections`` binary.  Crops 2-D projection patches
    around each particle position into *out_dir* for subsequent 2-D alignment.
    Configure the attributes, then call :meth:`extract`.

    .. rubric:: Attributes

    Attributes
    ----------
    num_threads : int
        Number of CPU threads to use.  Default: ``1``.
    normalize_type : str
        Per-patch normalisation.  One of ``'none'``, ``'zero_mean'``,
        ``'zero_mean_one_std'``, ``'zero_mean_proj_weight'``.
        Default: ``'zero_mean_one_std'``.
    invert_contrast : bool
        Invert the sign of the cropped projections.  Default: ``False``.
    """

    def __init__(self):
        self.num_threads       = 1
        self.normalize_type    = 'zero_mean_one_std'
        self.invert_contrast   = False

    def _validate(self):
        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight']:
            raise ValueError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std" or "zero_mean_proj_weight" are valid')

    def get_args(self, out_dir, tomos_file, ptcls_in, box_size):
        """Build the command-line argument string for ``susan_crop_projections``.

        Parameters
        ----------
        out_dir : str
            Output directory where cropped projection patches are written.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Patch size in pixels.

        Returns
        -------
        str
            Space-separated argument string ready to be appended to the
            ``susan_crop_projections`` command.
        """
        self._validate()
        args =        ' -tomos_file '      + tomos_file
        args = args + ' -out_dir '         + out_dir
        args = args + ' -ptcls_file '      + ptcls_in
        args = args + ' -box_size %d'      % box_size
        args = args + ' -n_threads %d'     % self.num_threads
        args = args + ' -norm_type '       + self.normalize_type
        args = args + ' -invert %d'        % self.invert_contrast
        return args

    def extract(self, out_pfx, tomos_file, ptcls_in, box_size):
        """Crop projection patches for all particles.

        Parameters
        ----------
        out_pfx : str
            Output directory for the cropped patches.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Patch size in pixels.

        Raises
        ------
        RuntimeError
            If the ``susan_crop_projections`` binary returns a non-zero exit
            code.
        """
        cmd = 'susan_crop_projections ' + self.get_args(out_pfx,tomos_file,ptcls_in,box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error cropping projections: ' + cmd)

###############################################################################

class CtfEstimator:
    """CTF estimation engine.

    Wraps the ``susan_estimate_ctf`` binary.  Estimates per-tilt defocus
    and astigmatism from the tilt-series stacks.  Configure the attributes,
    then call :meth:`estimate`.

    .. rubric:: Attributes

    Attributes
    ----------
    list_gpus_ids : list of int
        GPU device IDs to use.  Default: ``[0]``.
    threads_per_gpu : int
        *Deprecated.*  Must remain ``1``; use additional entries in
        ``list_gpus_ids`` to increase parallelism instead.
    binning : int
        Binning factor applied to the patches before estimation.  ``0`` means
        no binning.  Default: ``0``.
    resolution_angs : :class:`~susan.utils.datatypes.range_params`
        Resolution range (min, max) in Ångströms used for CTF fitting.
        Default: ``range_params(40, 7)``.
    defocus_angstroms : :class:`~susan.utils.datatypes.range_params`
        Defocus search range (min, max) in Ångströms.
        Default: ``range_params(10000, 90000)``.
    tilt_search : float
        Tilt-specific defocus search range in Ångströms.  Default: ``3000``.
    refine_defocus : :class:`~susan.utils.datatypes.search_params`
        Fine defocus refinement range and step in Ångströms.
        Default: ``search_params(2000, 100)``.
    max_bfactor : float
        Maximum B-factor used in the amplitude spectrum weighting.
        Default: ``600``.
    resolution_thres : float
        CTF quality threshold; fits below this score are discarded.
        Default: ``0.75``.
    verbose : int
        Verbosity level passed to the binary.  Default: ``0``.
    """

    def __init__(self):
        self.list_gpus_ids     = [0]
        self.threads_per_gpu   = 1
        self.binning           = 0
        self.resolution_angs   = _dt.range_params(40,7)
        self.defocus_angstroms = _dt.range_params(10000,90000)
        self.tilt_search       = 3000
        self.refine_defocus    = _dt.search_params(2000,100)
        self.max_bfactor       = 600
        self.resolution_thres  = 0.75
        #self.mpi               = _dt.mpi_params('srun -n %d ',1)
        self.verbose           = 0
        #self.verbosity         = 0
        
    def _validate(self):
        if not self.refine_defocus.step > 0:
            raise ValueError('The steps values must be larger than 0')
        
        if self.refine_defocus.span < self.refine_defocus.step:
            raise ValueError('Refine Defocus: Step cannot be larger than Range/Span')

        #if self.resolution_angs.max_val < self.resolution_angs.min_val:
        #    raise ValueError('Resolution (angstroms): min is larger than max')

        if self.defocus_angstroms.max_val < self.defocus_angstroms.min_val:
            raise ValueError('Defocus (angstroms): min is larger than max')
            
    def get_args(self, out_dir, tomos_file, ptcls_in, box_size):
        """Build the command-line argument string for ``susan_estimate_ctf``.

        Parameters
        ----------
        out_dir : str
            Output directory where per-tilt CTF results are written.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file (used to select
            regions of interest).
        box_size : int
            Patch size in pixels used for CTF estimation.

        Returns
        -------
        str
            Space-separated argument string ready to be appended to the
            ``susan_estimate_ctf`` command.
        """
        self._validate()
        if out_dir[-1] == '/':
            out_dir = out_dir[:-1]
        n_threads = len(self.list_gpus_ids)*self.threads_per_gpu
        gpu_str   = _get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_in '        + tomos_file
        args = args + ' -data_out '        + out_dir
        args = args + ' -ptcls_file '      + ptcls_in
        args = args + ' -n_threads %d'     % n_threads
        args = args + ' -gpu_list '        + gpu_str
        args = args + ' -box_size %d'      % box_size
        args = args + ' -res_range %f,%f'  % (self.resolution_angs.min_val,self.resolution_angs.max_val)
        args = args + ' -res_thres %f'     % self.resolution_thres
        args = args + ' -def_range %f,%f'  % (self.defocus_angstroms.min_val,self.defocus_angstroms.max_val)
        args = args + ' -tilt_search %f'   % self.tilt_search
        args = args + ' -refine_def %f,%f' % (self.refine_defocus.span,self.refine_defocus.step)
        args = args + ' -binning %d'       % self.binning
        args = args + ' -bfactor_max %f'   % self.max_bfactor
        args = args + ' -verbose %d'       % self.verbose
        #args = args + ' -verbosity %d'     % self.verbosity
        return args
    
    def estimate(self, out_dir, tomos_file, ptcls_in, box_size):
        """Execute the CTF estimation.

        Parameters
        ----------
        out_dir : str
            Output directory for CTF results.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Patch size in pixels.

        Raises
        ------
        RuntimeError
            If the ``susan_estimate_ctf`` binary returns a non-zero exit code.
        """
        cmd = 'susan_estimate_ctf ' + self.get_args(out_dir,tomos_file,ptcls_in,box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the CTF estimation: ' + cmd)

###############################################################################

class CtfRefiner:
    """Per-particle CTF refinement engine.

    Wraps the ``susan_ctf_refiner`` binary.  Jointly refines defocus, tilt
    angles, and in-plane shifts for each particle.  Configure the attributes,
    then call :meth:`refine` (single-node) or :meth:`refine_mpi` (multi-node
    MPI).

    .. rubric:: Attributes

    Attributes
    ----------
    list_gpus_ids : list of int
        GPU device IDs to use.  Default: ``[0]``.
    threads_per_gpu : int
        *Deprecated.*  Must remain ``1``; use additional entries in
        ``list_gpus_ids`` to increase parallelism instead.
    bandpass : :class:`~susan.utils.datatypes.bandpass`
        Frequency bandpass applied during refinement.
        Default: ``bandpass(0, -1, 2)`` (full range, 2-pixel rolloff).
    extra_padding : int
        Extra zero-padding (pixels) added on each side before FFT.
        Default: ``0``.
    padding_type : str
        Fill value for the padded region: ``'zero'`` or ``'noise'``.
        Default: ``'zero'``.
    normalize_type : str
        Per-substack normalisation.  One of ``'none'``, ``'zero_mean'``,
        ``'zero_mean_one_std'``, ``'zero_mean_proj_weight'``,
        ``'poisson_raw'``, ``'poisson_normal'``.
        Default: ``'zero_mean_one_std'``.
    halfsets_independ : bool
        Process the two half-sets with independent references.
        Default: ``False``.
    estimate_dose_wgt : bool
        Estimate per-projection dose-weighting factors.  Default: ``False``.
    refine_astigmatism : bool
        Refine per-particle astigmatism (``def_U``, ``def_V``, ``def_ang``).
        Default: ``False``.
    defocus_angstroms : :class:`~susan.utils.datatypes.search_params`
        Defocus search range and step in Ångströms.
        Default: ``search_params(1000, 100)``.
    angles : :class:`~susan.utils.datatypes.search_params`
        Tilt-angle search range and step in degrees.
        Default: ``search_params(2, 1)``.
    offset : :class:`~susan.utils.datatypes.offset_params`
        In-plane translational search range and step.
        Default: ``offset_params([4, 4, 4], 1, 'circle')``.
    ssnr : :class:`~susan.utils.datatypes.ssnr`
        Ad-hoc SSNR model for CTF weighting.
        Default: ``ssnr(0, 0.001)``.
    mpi : :class:`~susan.utils.datatypes.mpi_params`
        MPI launcher configuration used by :meth:`refine_mpi`.
        Default: ``mpi_params('srun -n %d ', 1)``.
    verbosity : int
        Verbosity level passed to the binary.  Default: ``0``.
    """

    def __init__(self):
        self.list_gpus_ids      = [0]
        self.threads_per_gpu    = 1
        self.bandpass           = _dt.bandpass(0,-1,2)
        self.extra_padding      = 0
        self.padding_type       = 'zero'
        self.normalize_type     = 'zero_mean_one_std'
        self.halfsets_independ  = False
        self.estimate_dose_wgt  = False
        self.refine_astigmatism = False
        self.defocus_angstroms  = _dt.search_params(1000,100)
        self.angles             = _dt.search_params(2,1)
        self.offset             = _dt.offset_params([4,4,4],1,'circle')
        self.ssnr               = _dt.ssnr(0,0.001)
        self.mpi                = _dt.mpi_params('srun -n %d ',1)
        self.verbosity          = 0

    def set_offset_search(self, off_range):
        """Set the in-plane offset search range.

        Parameters
        ----------
        off_range : float or sequence of float
            Search range in pixels.

            * **scalar** — same range applied to X, Y, and Z.
            * **2-element sequence** — ``[XY, Z]``.
            * **3-element sequence** — ``[X, Y, Z]``.

        Raises
        ------
        ValueError
            If *off_range* has more than 3 elements.
        """
        if isinstance(off_range,int) or isinstance(off_range,float):
            self.offset.span = (off_range,off_range,off_range)
        elif len(off_range) == 3:
            self.offset.span = off_range
        elif len(off_range) == 2:
            self.offset.span = (off_range[0],off_range[0],off_range[1])
        else:
            raise ValueError('Offset range can have up to 3 elements.')
        
    def _validate(self):
        if not self.padding_type in ['zero','noise']:
            raise ValueError('Invalid padding type. Only "zero" or "noise" are valid')
        
        if not self.normalize_type in ['none','zero_mean','zero_mean_one_std','zero_mean_proj_weight','poisson_raw','poisson_normal']:
            raise ValueError('Invalid normalization type. Only "none", "zero_mean", "zero_mean_one_std", "zero_mean_proj_weight", "poisson_raw" or "poisson_normal" are valid')
        
        if not self.defocus_angstroms.step > 0 or not self.angles.step > 0:
            raise ValueError('The steps values must be larger than 0')
        
        if self.defocus_angstroms.span > 0 and self.defocus_angstroms.span < self.defocus_angstroms.step:
            raise ValueError('Defocus (Angstroms): Step cannot be larger than Range/Span')

        if self.angles.span > 0 and self.angles.span < self.angles.step:
            raise ValueError('Angles (degrees): Step cannot be larger than Range/Span')

    def get_args(self, ptcls_out, refs_file, tomos_file, ptcls_in, box_size):
        """Build the command-line argument string for ``susan_ctf_refiner``.

        Parameters
        ----------
        ptcls_out : str
            Path for the output ``.ptclsraw`` file with refined CTF parameters.
        refs_file : str
            Path to the ``.refstxt`` references file.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Returns
        -------
        str
            Space-separated argument string ready to be appended to the
            ``susan_ctf_refiner`` command.
        """
        self._validate()
        if self.bandpass.lowpass <= 0:
            self.bandpass.lowpass = (box_size/2) - 1
        n_threads = len(self.list_gpus_ids)*self.threads_per_gpu
        gpu_str   = _get_gpu_str(self.list_gpus_ids)
        args =        ' -tomos_file '      + tomos_file
        args = args + ' -ptcls_in '        + ptcls_in
        args = args + ' -ptcls_out '       + ptcls_out
        args = args + ' -refs_file '       + refs_file
        args = args + ' -n_threads %d'     % n_threads
        args = args + ' -gpu_list '        + gpu_str
        args = args + ' -box_size %d'      % box_size
        args = args + ' -pad_size %d'      % self.extra_padding
        args = args + ' -pad_type '        + self.padding_type
        args = args + ' -norm_type '       + self.normalize_type
        args = args + ' -ssnr_param %f,%f' % (self.ssnr.F,self.ssnr.S)
        args = args + ' -bandpass %f,%f'   % (self.bandpass.highpass,self.bandpass.lowpass)
        args = args + ' -rolloff_f %f'     % self.bandpass.rolloff
        args = args + ' -def_search %f,%f' % (self.defocus_angstroms.span,self.defocus_angstroms.step)
        args = args + ' -ang_search %f,%f' % (self.angles.span,self.angles.step)
        args = args + ' -use_halves %d'    % self.halfsets_independ
        args = args + ' -est_dose %d'      % self.estimate_dose_wgt
        args = args + ' -verbosity %d'     % self.verbosity
        args = args + ' -astigmatism %d'   % self.refine_astigmatism
        args = args + ' -off_params %f,%f,%f,%f' % (self.offset.span[0],self.offset.span[1],self.offset.span[2],self.offset.step)
        return args
    
    def refine(self, ptcls_out, refs_file, tomos_file, ptcls_in, box_size):
        """Execute the CTF refinement on a single node.

        Parameters
        ----------
        ptcls_out : str
            Path for the output ``.ptclsraw`` file.
        refs_file : str
            Path to the ``.refstxt`` references file.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the ``susan_ctf_refiner`` binary returns a non-zero exit code.
        """
        cmd = 'susan_ctf_refiner ' + self.get_args(ptcls_out, refs_file, tomos_file, ptcls_in, box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the refinement: ' + cmd)

    def refine_mpi(self, ptcls_out, refs_file, tomos_file, ptcls_in, box_size):
        """Execute the CTF refinement using MPI across multiple nodes.

        The MPI command is taken from :attr:`mpi`.

        Parameters
        ----------
        ptcls_out : str
            Path for the output ``.ptclsraw`` file.
        refs_file : str
            Path to the ``.refstxt`` references file.
        tomos_file : str
            Path to the ``.tomostxt`` tomograms file.
        ptcls_in : str
            Path to the input ``.ptclsraw`` particles file.
        box_size : int
            Subvolume box size in pixels.

        Raises
        ------
        RuntimeError
            If the MPI binary returns a non-zero exit code.
        """
        cmd = self.mpi.gen_cmd() + ' ' + _os.path.dirname(_os.path.abspath(__file__)) + '/bin/susan_ctf_refiner_mpi ' + self.get_args(ptcls_out, refs_file, tomos_file, ptcls_in, box_size)
        rslt = _os.system(cmd)
        if not rslt == 0:
            raise RuntimeError('Error executing the refinement: ' + cmd)

