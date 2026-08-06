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

import csv as _csv
import warnings as _warnings

import numpy as _np
from susan.data._particles_core import _load_all, _save_all, _update_new_defocus
from susan.data  import Tomograms       as _tomodef
from susan.utils import is_extension    as _is_ext
from susan.utils import force_extension as _force_ext
from susan.utils import euZYZ_rotm      as _euZYZ_rotm
from ._PtclsGeom import PtclsGeom       as _Geom
from ._PtclsMRA  import PtclsMRA        as _MRA

class Particles:
    """Per-particle metadata container for SUSAN subtomogram averaging.

    Stores coordinates, 3-D alignment, per-projection 2-D alignment,
    CTF defocus, and half-set assignments for a set of particles.

    Two auxiliary modules are accessible as class attributes:

    * ``Particles.Geom`` (:class:`~susan.data._PtclsGeom.PtclsGeom`) —
      geometry operations: in-place rotation/translation, symmetry
      expansion, tilt-angle filtering, and distance-based deduplication.
    * ``Particles.MRA`` (:class:`~susan.data._PtclsMRA.PtclsMRA`) —
      multi-reference alignment helpers: duplicating and selecting
      reference slots.

    File format: binary ``.ptclsraw`` (magic ``SsaPtcl1`` + uint32 header
    + one packed float32 record per particle).

    .. rubric:: Identification & bookkeeping

    .. attribute:: ptcl_id
       :type: ndarray, uint32, shape (M,)

       User-assigned particle identifier.

    .. attribute:: tomo_id
       :type: ndarray, uint32, shape (M,)

       Tomogram ID the particle belongs to (matches ``Tomograms.tomo_id``).

    .. attribute:: tomo_cix
       :type: ndarray, uint32, shape (M,)

       Contiguous index into the Tomograms array.

       .. deprecated::
          Not read by SUSAN anymore.  The index is derived from
          :attr:`tomo_id` through :meth:`Tomograms.get_cix`, so it cannot go
          stale when a particle set is paired with a different
          ``Tomograms``.  The field is still stored in the ``.ptclsraw``
          file for older SUSAN versions; refresh it with
          :meth:`update_tomo_cix` before saving.

    .. attribute:: ref_cix
       :type: ndarray, uint32, shape (M,)

       Current reference-class assignment (0-based index).

    .. attribute:: half_id
       :type: ndarray, uint32, shape (M,)

       Half-set label: 1 or 2.  0 means unassigned.

    .. attribute:: extra_1, extra_2
       :type: ndarray, float32, shape (M,)

       User-defined auxiliary fields.

    .. rubric:: Position & 3-D alignment

    .. attribute:: position
       :type: ndarray, float32, shape (M, 3)

       Particle centre in Ångströms (X, Y, Z).

    .. attribute:: ali_eu
       :type: ndarray, float32, shape (R, M, 3)

       3-D alignment: ZYZ Euler angles in radians, one set per reference.

    .. attribute:: ali_t
       :type: ndarray, float32, shape (R, M, 3)

       3-D alignment: translations (X, Y, Z) in Ångströms.

    .. attribute:: ali_cc
       :type: ndarray, float32, shape (R, M)

       3-D cross-correlation score per reference.

    .. attribute:: ali_w
       :type: ndarray, float32, shape (R, M)

       3-D alignment weight per reference.

    .. rubric:: Per-projection 2-D alignment

    .. attribute:: prj_eu
       :type: ndarray, float32, shape (M, P, 3)

       Per-projection 2-D alignment: ZYZ Euler angles in radians.

    .. attribute:: prj_t
       :type: ndarray, float32, shape (M, P, 2)

       Per-projection 2-D alignment: shifts (X, Y) in Ångströms.

    .. attribute:: prj_cc
       :type: ndarray, float32, shape (M, P)

       Per-projection cross-correlation score.

    .. attribute:: prj_w
       :type: ndarray, float32, shape (M, P)

       Per-projection weight (0 = excluded).

    .. rubric:: CTF

    .. attribute:: def_U, def_V
       :type: ndarray, float32, shape (M, P)

       Per-projection defocus major/minor axis in Ångströms.

    .. attribute:: def_ang
       :type: ndarray, float32, shape (M, P)

       Defocus astigmatism angle in degrees.

    .. attribute:: def_phas
       :type: ndarray, float32, shape (M, P)

       Phase shift in degrees.

    .. attribute:: def_Bfct
       :type: ndarray, float32, shape (M, P)

       Per-projection B-factor in Å², applied as :math:`e^{-s^2 B/4}`.

       It is part of the **CTF model**: the reconstruction multiplies it into
       the CTF, so it enters the Wiener numerator once and the denominator
       squared, and the division deconvolves it.  Use this field for an
       envelope that the reconstruction should *undo*.  It is never estimated
       automatically (``estimate_ctf`` only resets it to 0), so it is free for
       the user to set.

       See :doc:`/cryoet` for how it enters the reconstruction, and contrast
       with :attr:`def_ExFl`, which has the same form but the opposite effect.

    .. attribute:: def_ExFl
       :type: ndarray, float32, shape (M, P)

       Per-projection exposure filter (dose) in Å², applied as
       :math:`e^{-s^2 D/4}`.

       Unlike :attr:`def_Bfct`, it enters the Wiener numerator **only**, so it
       is *not* compensated: it survives into the reconstructed map and
       permanently attenuates that projection.  Use this field for an envelope
       that the reconstruction should *keep*, such as dose weighting.

       .. warning::

          This field is an **output of the aligner**, not a user input.  Every
          alignment overwrites it with ``expfilt_gain * dose``, where the dose
          is estimated from the width of the cross-correlation peak.  Setting
          ``expfilt_gain = 0`` writes zeros rather than preserving the field, so
          a hand-set value cannot survive an alignment; it is only meaningful
          between an alignment and a reconstruction.  For a user-owned envelope,
          use :attr:`def_Bfct` instead.

          A value of ``9999`` is the sentinel written when the CC peak width
          cannot be measured.  It zeroes the projection's contribution to the
          numerator while that projection still carries its full weight in the
          denominator.

       Only the ``'wiener'``, ``'pre_wiener'`` and ``'wiener_ssnr'``
       reconstruction policies apply it; ``'none'`` and ``'phase_flip'`` ignore
       it.

    .. attribute:: def_mres
       :type: ndarray, float32, shape (M, P)

       Maximum resolution for CTF fitting in Ångströms.

    .. attribute:: def_scor
       :type: ndarray, float32, shape (M, P)

       CTF fit score.
    """

    Geom = _Geom
    MRA  = _MRA

    def __init__(self, filename=None, n_ptcl=0, n_proj=0, n_refs=0):
        """Load from file or allocate an empty container.

        Parameters
        ----------
        filename : str, optional
            Path to a ``.ptclsraw`` file to load.
        n_ptcl : int
            Number of particles to allocate (used when filename is None).
        n_proj : int
            Maximum number of projections per particle.
        n_refs : int
            Number of reference slots (multi-reference alignment).
        """
        if isinstance(filename, str):
            self._load(filename)
        else:
            if n_proj > 0 and n_refs > 0:
                self._alloc(n_ptcl,n_proj,n_refs)
            else:
                raise ValueError('Invalid input')

    def get_n_ptcl(self) -> int: return self.ptcl_id.shape[0]
    def get_n_refs(self) -> int: return self.ali_eu.shape[0]
    def get_n_proj(self) -> int: return self.prj_eu.shape[1]
    
    n_ptcl = property(get_n_ptcl)
    n_refs = property(get_n_refs)
    n_proj = property(get_n_proj)
    
    @staticmethod
    def _check_filename(filename):
        if not _is_ext(filename,'ptclsraw'):
            raise ValueError( 'Wrong file extension, do you mean ' + _force_ext(filename,'ptclsraw') + '?')
    
    def _alloc(self,n_ptcl,n_proj,n_refs):
        self.ptcl_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.tomo_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.tomo_cix = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.position = _np.zeros((n_ptcl,3),dtype=_np.float32) # in Angstroms
        self.ref_cix  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.half_id  = _np.zeros( n_ptcl   ,dtype=_np.uint32 )
        self.extra_1  = _np.zeros( n_ptcl   ,dtype=_np.float32)
        self.extra_2  = _np.zeros( n_ptcl   ,dtype=_np.float32)
        
        # 3D alignment
        self.ali_eu   = _np.zeros((n_refs,n_ptcl,3),dtype=_np.float32) # in Radians
        self.ali_t    = _np.zeros((n_refs,n_ptcl,3),dtype=_np.float32) # in Angstroms
        self.ali_cc   = _np.zeros((n_refs,n_ptcl  ),dtype=_np.float32)
        self.ali_w    = _np.zeros((n_refs,n_ptcl  ),dtype=_np.float32)
        
        # 2D alignment
        self.prj_eu   = _np.zeros((n_ptcl,n_proj,3),dtype=_np.float32) # in Radians
        self.prj_t    = _np.zeros((n_ptcl,n_proj,2),dtype=_np.float32) # in Angstroms
        self.prj_cc   = _np.zeros((n_ptcl,n_proj  ),dtype=_np.float32)
        self.prj_w    = _np.zeros((n_ptcl,n_proj  ),dtype=_np.float32)
        
        # Defocus
        self.def_U    = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # U (angstroms)
        self.def_V    = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # V (angstroms)
        self.def_ang  = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # angles (sexagesimal)
        self.def_phas = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # phase shift (sexagesimal?)
        self.def_Bfct = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Bfactor
        self.def_ExFl = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Exposure filter
        self.def_mres = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # Max. resolution (angstroms)
        self.def_scor = _np.zeros((n_ptcl,n_proj),dtype=_np.float32) # score

    def _load_header(self,fp):
        buffer = fp.read( 8 + 4*3 )
        if not _np.array_equal( buffer[:8], b'SsaPtcl1' ):
            raise ValueError("Invalid File signature")
        return _np.frombuffer(buffer[8:],_np.uint32)
    
    def sort(self) -> None:
        """Sort particles in-place by (tomo_id, ptcl_id)."""
        idx = _np.lexsort((self.ptcl_id,self.tomo_id))
        self.ptcl_id  = self.ptcl_id [idx]
        self.tomo_id  = self.tomo_id [idx]
        self.tomo_cix = self.tomo_cix[idx]
        self.position = self.position[idx,:]
        self.ref_cix  = self.ref_cix [idx]
        self.half_id  = self.half_id [idx]
        self.extra_1  = self.extra_1 [idx]
        self.extra_2  = self.extra_2 [idx]
        # 3D alignment
        self.ali_eu   = self.ali_eu[:,idx,:]
        self.ali_t    = self.ali_t [:,idx,:]
        self.ali_cc   = self.ali_cc[:,idx]
        self.ali_w    = self.ali_w [:,idx]
        # 2D alignment
        self.prj_eu   = self.prj_eu[idx,:,:]
        self.prj_t    = self.prj_t [idx,:,:]
        self.prj_cc   = self.prj_cc[idx,:]
        self.prj_w    = self.prj_w [idx,:]
        # Defocus
        self.def_U    = self.def_U   [idx,:]
        self.def_V    = self.def_V   [idx,:]
        self.def_ang  = self.def_ang [idx,:]
        self.def_phas = self.def_phas[idx,:]
        self.def_Bfct = self.def_Bfct[idx,:]
        self.def_ExFl = self.def_ExFl[idx,:]
        self.def_mres = self.def_mres[idx,:]
        self.def_scor = self.def_scor[idx,:]

    def _load(self,filename):
        Particles._check_filename(filename)
        with open(filename,"rb") as fp:
            n_ptcl, n_proj, n_refs = self._load_header(fp)
        self._alloc(n_ptcl,n_proj,n_refs)
        _load_all(filename.encode(),
                  n_ptcl, n_refs, n_proj,
                  self.ptcl_id, self.tomo_id, self.tomo_cix, self.position,
                  self.ref_cix, self.half_id, self.extra_1,  self.extra_2,
                  self.ali_eu,  self.ali_t,   self.ali_cc,   self.ali_w,
                  self.prj_eu,  self.prj_t,   self.prj_cc,   self.prj_w,
                  self.def_U,   self.def_V,   self.def_ang,  self.def_phas,
                  self.def_Bfct,self.def_ExFl,self.def_mres, self.def_scor)

    def save(self, filename) -> None:
        """Save to a ``.ptclsraw`` binary file.

        Parameters
        ----------
        filename : str
            Output path; must have a ``.ptclsraw`` extension.
        """
        Particles._check_filename(filename)
        def _c32(a, dtype=_np.float32): return _np.ascontiguousarray(a, dtype=dtype)
        _save_all(filename.encode(),
                  self.n_ptcl, self.n_refs, self.n_proj,
                  _c32(self.ptcl_id,  _np.uint32), _c32(self.tomo_id,  _np.uint32),
                  _c32(self.tomo_cix, _np.uint32), _c32(self.position),
                  _c32(self.ref_cix,  _np.uint32), _c32(self.half_id,  _np.uint32),
                  _c32(self.extra_1), _c32(self.extra_2),
                  _c32(self.ali_eu),  _c32(self.ali_t),   _c32(self.ali_cc),  _c32(self.ali_w),
                  _c32(self.prj_eu),  _c32(self.prj_t),   _c32(self.prj_cc),  _c32(self.prj_w),
                  _c32(self.def_U),   _c32(self.def_V),   _c32(self.def_ang),  _c32(self.def_phas),
                  _c32(self.def_Bfct),_c32(self.def_ExFl),_c32(self.def_mres), _c32(self.def_scor))
    
    def __getitem__(self, idx) -> Particles:
        """Select particles by index, boolean mask, or slice.

        Parameters
        ----------
        idx : int, array-like, or slice
            Integer indices, a boolean mask of length ``n_ptcl``, or a
            standard Python slice.

        Returns
        -------
        Particles
            New container holding only the selected particles.
        """
        if isinstance(idx,slice):
            idx = _np.arange(*idx.indices(self.n_ptcl))
        return self.select(idx)
    
    def update_tomo_cix(self, tomograms) -> None:
        """Refresh the deprecated ``tomo_cix`` field from ``tomo_id``.

        ``tomo_cix`` is no longer read by SUSAN: the index of a particle's
        tomogram is derived from ``tomo_id`` through
        :meth:`Tomograms.get_cix` wherever it is needed.  The field is still
        written to the ``.ptclsraw`` file so that older SUSAN versions, which
        do read it, keep working.  Call this whenever a particle set is
        created or paired with a different ``Tomograms``, so the stored value
        does not go stale.

        Parameters
        ----------
        tomograms : Tomograms
            Tomograms this particle set refers to.  Every ``tomo_id`` must be
            present in it.

        Raises
        ------
        ValueError
            If a ``tomo_id`` is not in *tomograms*.
        """
        self.tomo_cix[:] = tomograms.get_cix(self.tomo_id).astype(_np.uint32)

    def select(self, idx) -> Particles:
        """Return a new Particles containing only the selected entries.

        Parameters
        ----------
        idx : array-like of int or bool
            Integer indices or boolean mask selecting which particles to keep.

        Returns
        -------
        Particles
        """
        idx = _np.atleast_1d(_np.array(idx))
        if idx.ndim >= 2:
            idx = idx[:,0]
        number_of_particles = int(idx.sum()) if idx.dtype == _np.bool_ else idx.shape[0]
        ptcls_out = Particles(n_ptcl=number_of_particles,n_proj=self.n_proj,n_refs=self.n_refs)
        if number_of_particles > 0:
            ptcls_out.ptcl_id  = self.ptcl_id [idx]
            ptcls_out.tomo_id  = self.tomo_id [idx]
            ptcls_out.tomo_cix = self.tomo_cix[idx]
            ptcls_out.position = self.position[idx,:]
            ptcls_out.ref_cix  = self.ref_cix [idx]
            ptcls_out.half_id  = self.half_id [idx]
            ptcls_out.extra_1  = self.extra_1 [idx]
            ptcls_out.extra_2  = self.extra_2 [idx]
            # 3D alignment
            ptcls_out.ali_eu   = self.ali_eu[:,idx,:]
            ptcls_out.ali_t    = self.ali_t [:,idx,:]
            ptcls_out.ali_cc   = self.ali_cc[:,idx]
            ptcls_out.ali_w    = self.ali_w [:,idx]
            # 2D alignment
            ptcls_out.prj_eu   = self.prj_eu[idx,:,:]
            ptcls_out.prj_t    = self.prj_t [idx,:,:]
            ptcls_out.prj_cc   = self.prj_cc[idx,:]
            ptcls_out.prj_w    = self.prj_w [idx,:]
            # Defocus
            ptcls_out.def_U    = self.def_U   [idx,:]
            ptcls_out.def_V    = self.def_V   [idx,:]
            ptcls_out.def_ang  = self.def_ang [idx,:]
            ptcls_out.def_phas = self.def_phas[idx,:]
            ptcls_out.def_Bfct = self.def_Bfct[idx,:]
            ptcls_out.def_ExFl = self.def_ExFl[idx,:]
            ptcls_out.def_mres = self.def_mres[idx,:]
            ptcls_out.def_scor = self.def_scor[idx,:]
        if number_of_particles > 1:
            ptcls_out.sort()
        return ptcls_out

    def copy(self) -> Particles:
        """Return a deep copy of this Particles object.

        All numpy arrays are copied (no shared memory with the original).
        Particle order is preserved.

        Returns
        -------
        Particles
        """
        ptcls_out = Particles(n_ptcl=self.n_ptcl, n_proj=self.n_proj, n_refs=self.n_refs)
        ptcls_out.ptcl_id  = self.ptcl_id .copy()
        ptcls_out.tomo_id  = self.tomo_id .copy()
        ptcls_out.tomo_cix = self.tomo_cix.copy()
        ptcls_out.position = self.position.copy()
        ptcls_out.ref_cix  = self.ref_cix .copy()
        ptcls_out.half_id  = self.half_id .copy()
        ptcls_out.extra_1  = self.extra_1 .copy()
        ptcls_out.extra_2  = self.extra_2 .copy()
        # 3D alignment
        ptcls_out.ali_eu   = self.ali_eu.copy()
        ptcls_out.ali_t    = self.ali_t .copy()
        ptcls_out.ali_cc   = self.ali_cc.copy()
        ptcls_out.ali_w    = self.ali_w .copy()
        # 2D alignment
        ptcls_out.prj_eu   = self.prj_eu.copy()
        ptcls_out.prj_t    = self.prj_t .copy()
        ptcls_out.prj_cc   = self.prj_cc.copy()
        ptcls_out.prj_w    = self.prj_w .copy()
        # Defocus
        ptcls_out.def_U    = self.def_U   .copy()
        ptcls_out.def_V    = self.def_V   .copy()
        ptcls_out.def_ang  = self.def_ang .copy()
        ptcls_out.def_phas = self.def_phas.copy()
        ptcls_out.def_Bfct = self.def_Bfct.copy()
        ptcls_out.def_ExFl = self.def_ExFl.copy()
        ptcls_out.def_mres = self.def_mres.copy()
        ptcls_out.def_scor = self.def_scor.copy()
        return ptcls_out

    def append_ptcls(self, ptcls) -> None:
        """Append another Particles object to this one in-place.

        Both objects must have the same ``n_proj`` and ``n_refs``.
        The combined list is sorted by (tomo_id, ptcl_id) after appending.

        Parameters
        ----------
        ptcls : Particles
        """
        self.ptcl_id  = _np.concatenate( (self.ptcl_id ,ptcls.ptcl_id ),axis=0 )
        self.tomo_id  = _np.concatenate( (self.tomo_id ,ptcls.tomo_id ),axis=0 )
        self.tomo_cix = _np.concatenate( (self.tomo_cix,ptcls.tomo_cix),axis=0 )
        self.position = _np.concatenate( (self.position,ptcls.position),axis=0 )
        self.ref_cix  = _np.concatenate( (self.ref_cix ,ptcls.ref_cix ),axis=0 )
        self.half_id  = _np.concatenate( (self.half_id ,ptcls.half_id ),axis=0 )
        self.extra_1  = _np.concatenate( (self.extra_1 ,ptcls.extra_1 ),axis=0 )
        self.extra_2  = _np.concatenate( (self.extra_2 ,ptcls.extra_2 ),axis=0 )
        # 3D alignment
        self.ali_eu   = _np.concatenate( (self.ali_eu,ptcls.ali_eu),axis=1 )
        self.ali_t    = _np.concatenate( (self.ali_t ,ptcls.ali_t ),axis=1 )
        self.ali_cc   = _np.concatenate( (self.ali_cc,ptcls.ali_cc),axis=1 )
        self.ali_w    = _np.concatenate( (self.ali_w ,ptcls.ali_w ),axis=1 )
        # 2D alignment
        self.prj_eu   = _np.concatenate( (self.prj_eu,ptcls.prj_eu),axis=0 )
        self.prj_t    = _np.concatenate( (self.prj_t ,ptcls.prj_t ),axis=0 )
        self.prj_cc   = _np.concatenate( (self.prj_cc,ptcls.prj_cc),axis=0 )
        self.prj_w    = _np.concatenate( (self.prj_w ,ptcls.prj_w ),axis=0 )
        # Defocus
        self.def_U    = _np.concatenate( (self.def_U   ,ptcls.def_U   ),axis=0 )
        self.def_V    = _np.concatenate( (self.def_V   ,ptcls.def_V   ),axis=0 )
        self.def_ang  = _np.concatenate( (self.def_ang ,ptcls.def_ang ),axis=0 )
        self.def_phas = _np.concatenate( (self.def_phas,ptcls.def_phas),axis=0 )
        self.def_Bfct = _np.concatenate( (self.def_Bfct,ptcls.def_Bfct),axis=0 )
        self.def_ExFl = _np.concatenate( (self.def_ExFl,ptcls.def_ExFl),axis=0 )
        self.def_mres = _np.concatenate( (self.def_mres,ptcls.def_mres),axis=0 )
        self.def_scor = _np.concatenate( (self.def_scor,ptcls.def_scor),axis=0 )
        # Sort
        self.sort()

    def set_weights(self, in_wgt) -> None:
        """Set per-projection weights, preserving already-excluded projections.

        Multiplies ``in_wgt`` by the existing ``prj_w > 0`` mask so that
        projections already set to zero remain excluded.

        Parameters
        ----------
        in_wgt : array-like, shape (P,) or (M, P)
            New weight values.
        """
        self.prj_w[:,:] = (self.prj_w > 0) * in_wgt
        
    def halfsets_by_Y(self) -> None:
        """Assign half-sets by splitting each tomogram at its Y-median.

        Particles above the median Y coordinate get half_id=2; those at
        or below get half_id=1.  Applied per tomogram independently.
        """
        tomo_ids = _np.unique( self.tomo_id )
        for tid in tomo_ids:
            idx = self.tomo_id == tid
            self.half_id[idx] = 1
            th = _np.quantile( self.position[idx,1].flatten() ,0.5)
            self.half_id[idx] = self.half_id[idx] + (self.position[idx,1]>th)

    def halfsets_even_odd(self):
        """Assign half-sets by even/odd index (half_id alternates 1, 2, 1, 2…)."""
        self.half_id[0::2] = 1
        self.half_id[1::2] = 2

    def halfsets_randomize(self):
        """Assign half-sets randomly (each particle independently drawn from {1, 2})."""
        self.half_id[:] = _np.random.randint(1,3,self.half_id.shape)
        
    def update_position(self, ref_id=0):
        """Absorb the 3-D alignment translation into the particle position.

        Adds ``ali_t[ref_id]`` to ``position`` and resets ``ali_t[ref_id]``
        to zero.  Useful after alignment to make coordinates absolute again.

        Parameters
        ----------
        ref_id : int, optional
            Reference index whose translation to absorb. Default 0.
        """
        self.position = self.position + self.ali_t[ref_id]
        self.ali_t[ref_id,:,:] = 0

    _update_new_defocus = staticmethod(_update_new_defocus)

    def update_defocus(self, tomos_info, ref_id=0, z_sign=None):
        """Recompute per-projection defocus values from the current 3-D positions.

        For each particle, projects the 3-D position (position + ali_t[ref_id])
        onto each tilt projection's Z-axis and corrects the tomogram-level
        defocus by the resulting depth offset.  Also copies projection weights,
        astigmatism, B-factor, and CTF scores from the Tomograms metadata.

        The depth-to-defocus-offset mapping depends on the defocus sign
        convention: with the underfocus (positive) convention, ``δ_z`` and
        ``dZ`` have opposite signs; with the overfocus (negative) convention,
        they have the same sign.  The effective Z coefficient is therefore
        multiplied by ``sign(base_defocus)`` so the addition stays consistent
        with whatever convention the tomogram's defocus values were stored in.

        Parameters
        ----------
        tomos_info : Tomograms
            Tomogram metadata providing tilt geometries and base defocus values.
        ref_id : int, optional
            Reference index whose translation is included in the position. Default 0.
        z_sign : float, optional
            Override the Z-axis handedness (+1 or −1).  If None (default),
            ``tomos_info.handedness`` is used per tomogram.  This value is
            still combined with the base-defocus sign internally.
        """
        
        # Calculate tilt rotation matrix
        R_arr = _np.zeros((tomos_info.n_tomos,tomos_info.n_projs,3,3),dtype=_np.float32)
        for t in range(tomos_info.n_tomos):
            for p in range(tomos_info.num_proj[t]):
                _euZYZ_rotm(R_arr[t,p],_np.deg2rad(tomos_info.proj_eZYZ[t,p]))
        
        # Update defocus
        cix = tomos_info.get_cix(self.tomo_id)
        for k in range(self.n_ptcl):
            tid = cix[k]

            self.prj_w   [k,:] = tomos_info.proj_wgt[tid,:]
            self.def_ang [k,:] = tomos_info.def_ang [tid,:]
            self.def_phas[k,:] = tomos_info.def_phas[tid,:]
            self.def_Bfct[k,:] = tomos_info.def_Bfct[tid,:]
            self.def_ExFl[k,:] = tomos_info.def_ExFl[tid,:]
            self.def_mres[k,:] = tomos_info.def_mres[tid,:]
            self.def_scor[k,:] = tomos_info.def_scor[tid,:]
            
            # Note: Numba makes it ~23.3 times faster
            pos = self.position[k] + self.ali_t[ref_id,k]
            z_sign_k = z_sign if z_sign is not None else tomos_info.handedness[tid]
            np = int(tomos_info.num_proj[tid])
            base_def = tomos_info.def_U[tid,:np]
            nz = base_def[base_def != 0]
            def_sign = _np.sign(nz[0]) if nz.size > 0 else 1.0
            Particles._update_new_defocus(
                self.def_U[k],
                self.def_V[k],
                R_arr[tid],
                pos,
                np,
                z_sign_k * def_sign,
                tomos_info.def_U[tid],
                tomos_info.def_V[tid]
            )

    def x(self, ref_idx=0) -> _np.ndarray:
        """Absolute X coordinate: position[:,0] + ali_t[ref_idx,:,0] (Ångströms)."""
        return self.position[:,0] + self.ali_t[ref_idx,:,0]

    def y(self, ref_idx=0) -> _np.ndarray:
        """Absolute Y coordinate: position[:,1] + ali_t[ref_idx,:,1] (Ångströms)."""
        return self.position[:,1] + self.ali_t[ref_idx,:,1]

    def z(self, ref_idx=0) -> _np.ndarray:
        """Absolute Z coordinate: position[:,2] + ali_t[ref_idx,:,2] (Ångströms)."""
        return self.position[:,2] + self.ali_t[ref_idx,:,2]

    def pos(self, ref_idx=0) -> _np.ndarray:
        """Absolute (X, Y, Z) positions: position + ali_t[ref_idx], shape (M, 3), Ångströms."""
        return self.position + self.ali_t[ref_idx]

    @staticmethod
    def _get_tomo_limit_angstroms(tomo_size,tomo_apix,border):
        return tomo_apix*( (_np.int32(tomo_size)-_np.int32(border)).clip(0) )/2

    @staticmethod
    def _validate_tomogram(tomograms):
        if not isinstance(tomograms,_tomodef.Tomograms):
            raise ValueError('Tomograms must be a Tomograms object.')
        apix = _np.unique( tomograms.pix_size )
        if apix.size != 1:
            raise ValueError('Tomograms must have the same pixel size.')
        return apix[0]

    @staticmethod
    def _get_grid_step(s_ang,s_pix,apix):
        if   s_ang is     None and s_pix is     None:
            raise ValueError('Set the steps either in angstroms or pixels')
        elif s_ang is not None and s_pix is not None:
            raise ValueError('Set step_angstroms or step_pixels, not both.')
        elif s_ang is not None and s_pix is     None:
            return s_ang
        else:
            return s_pix*apix

    @staticmethod
    def _get_border_pixels(pix):
        p = _np.array(pix,_np.int32)
        if p.size == 1:
            return _np.array((pix,pix,pix),_np.int32)
        elif p.size == 3:
            return p
        else:
            raise ValueError('skip_border_pixels must be either a scalar or a 3-element vector')

    @staticmethod
    def grid_2d(tomograms, step_angstroms=None, step_pixels=None,
                skip_border_pixels=0, angle_deg_Y=0) -> Particles:
        """Create a 2-D regular grid of particles at Z=0 across all tomograms.

        Positions are placed on an XY grid centred at the tomogram origin.
        Defocus values are initialised from the Tomograms metadata.

        Parameters
        ----------
        tomograms : Tomograms
            All tomograms must share the same pixel size.
        step_angstroms : float, optional
            Grid spacing in Ångströms. Mutually exclusive with step_pixels.
        step_pixels : float, optional
            Grid spacing in pixels. Mutually exclusive with step_angstroms.
        skip_border_pixels : int or array-like of int (3,), optional
            Pixels to exclude near each tomogram edge. Default 0.
        angle_deg_Y : float, optional
            Rotate the grid plane by this angle around Y (degrees). Default 0.

        Returns
        -------
        Particles
        """
        apix = Particles._validate_tomogram(tomograms)
        step = Particles._get_grid_step(step_angstroms,step_pixels,apix)
        brdr = Particles._get_border_pixels(skip_border_pixels)

        R = _np.eye(3, dtype=_np.float32)
        _euZYZ_rotm(R,_np.deg2rad(_np.array((0,angle_deg_Y,0), dtype=_np.float32)))
        
        pts = _np.zeros((0,3),dtype=_np.float32)
        tid = _np.zeros((0),dtype=_np.uint32)
        for i in range( tomograms.n_tomos ):
            tomo_range = Particles._get_tomo_limit_angstroms(tomograms.tomo_size[i],tomograms.pix_size[i],brdr)
            t_x = _np.arange(0,tomo_range[0],step,dtype=_np.float32)
            t_x = _np.concatenate( (-t_x[::-1],t_x[1:]) )
            t_y = _np.arange(0,tomo_range[1],step,dtype=_np.float32)
            t_y = _np.concatenate( (-t_y[::-1],t_y[1:]) )
            x,y,z = _np.float32(_np.meshgrid(t_x,t_y,(0)))
            pos = _np.stack( (x.flatten(),y.flatten(),z.flatten()), ).transpose()
            pos = pos@R
            pts = _np.concatenate( (pts,pos) )
            tid = _np.concatenate( (tid,_np.repeat(tomograms.tomo_id[i],pos.shape[0])) )
        
        ptcls = Particles(n_ptcl=pts.shape[0],n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = _np.arange(1,pts.shape[0]+1,dtype=_np.uint32)
        ptcls.position[:,:] = pts
        ptcls.tomo_id [:]   = tid
        ptcls.update_tomo_cix(tomograms)
        ptcls.ali_w[:] 	    = 1
        ptcls.update_defocus(tomograms)
        return ptcls
    
    @staticmethod
    def grid_3d(tomograms, step_angstroms=None, step_pixels=None,
                skip_border_pixels=0) -> Particles:
        """Create a 3-D regular grid of particles across all tomograms.

        Positions fill the full XYZ volume of each tomogram.  Half-sets are
        assigned by even/odd index; defocus values are initialised from the
        Tomograms metadata.

        Parameters
        ----------
        tomograms : Tomograms
            All tomograms must share the same pixel size.
        step_angstroms : float, optional
            Grid spacing in Ångströms. Mutually exclusive with step_pixels.
        step_pixels : float, optional
            Grid spacing in pixels. Mutually exclusive with step_angstroms.
        skip_border_pixels : int or array-like of int (3,), optional
            Pixels to exclude near each tomogram edge. Default 0.

        Returns
        -------
        Particles
        """
        apix = Particles._validate_tomogram(tomograms)
        step = Particles._get_grid_step(step_angstroms,step_pixels,apix)
        brdr = Particles._get_border_pixels(skip_border_pixels)
        
        pts = _np.zeros((0,3),dtype=_np.float32)
        tid = _np.zeros((0),dtype=_np.uint32)
        for i in range( tomograms.n_tomos ):
            tomo_range = Particles._get_tomo_limit_angstroms(tomograms.tomo_size[i],tomograms.pix_size[i],brdr)
            t_x = _np.arange(0,tomo_range[0],step,dtype=_np.float32)
            t_x = _np.concatenate( (-t_x[::-1],t_x[1:]) )
            t_y = _np.arange(0,tomo_range[1],step,dtype=_np.float32)
            t_y = _np.concatenate( (-t_y[::-1],t_y[1:]) )
            t_z = _np.arange(0,tomo_range[2],step,dtype=_np.float32)
            t_z = _np.concatenate( (-t_z[::-1],t_z[1:]) )
            x,y,z = _np.float32(_np.meshgrid(t_x,t_y,t_z))
            pos = _np.stack( (x.flatten(),y.flatten(),z.flatten()), ).transpose()
            pts = _np.concatenate( (pts,pos) )
            tid = _np.concatenate( (tid,_np.repeat(tomograms.tomo_id[i],pos.shape[0])) )
        
        ptcls = Particles(n_ptcl=pts.shape[0],n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = _np.arange(1,pts.shape[0]+1,dtype=_np.uint32)
        ptcls.position[:,:] = pts
        ptcls.tomo_id [:]   = tid
        ptcls.update_tomo_cix(tomograms)
        ptcls.ali_w[:]      = 1
        ptcls.halfsets_even_odd()
        ptcls.update_defocus(tomograms)
        return ptcls

    @staticmethod
    def _validate_import_args(position,ptcls_id,tomos_id):
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError('Position must be a N-by-3 2D matrix')
        N = position.shape[0]
        if ptcls_id.shape[0] != N:
            raise ValueError('Number of entries in ptcls_id do not match position')
        if tomos_id.shape[0] != N:
            raise ValueError('Number of entries in tomos_id do not match position')
        return N
        
    @staticmethod
    def _calc_position(p_out,p_in,tomograms,tomos_cix,apix):
        for i in range(p_in.shape[0]):
            pos = p_in[i,:] - tomograms.tomo_size[tomos_cix[i]]/2
            p_out[i,:] = apix*pos
            
    @staticmethod
    def import_data(tomograms, position, tomos_id, ptcls_id=None,
                    randomize_angles=False) -> Particles:
        """Create a Particles object from external coordinate data.

        Converts pixel-space coordinates (relative to tomogram corner) to
        Ångström-space coordinates centred at the tomogram origin, and
        initialises defocus from the Tomograms metadata.

        Parameters
        ----------
        tomograms : Tomograms
            All tomograms must share the same pixel size.
        position : ndarray, shape (N, 3)
            Particle positions in pixels, relative to the tomogram corner.
        tomos_id : array-like, shape (N,)
            Tomogram ID for each particle (must match ``Tomograms.tomo_id``).
        ptcls_id : array-like of int, shape (N,), optional
            Particle identifiers.  Defaults to 0, 1, …, N−1.
        randomize_angles : bool, optional
            If True, initialise ``ali_eu`` with random ZYZ Euler angles.
            Default False.

        Returns
        -------
        Particles
        """
        if ptcls_id is None:
            ptcls_id = _np.arange(tomos_id.shape[0])
        apix = Particles._validate_tomogram(tomograms)
        N = Particles._validate_import_args(position,ptcls_id,tomos_id)
        ptcls = Particles(n_ptcl=N,n_proj=tomograms.n_projs,n_refs=1)
        ptcls.ptcl_id[:]    = ptcls_id
        ptcls.tomo_id [:]   = tomos_id
        ptcls.ali_w[:]      = 1
        ptcls.update_tomo_cix(tomograms)
        Particles._calc_position(ptcls.position,position,tomograms,tomograms.get_cix(tomos_id),apix)
        ptcls.halfsets_even_odd()
        ptcls.sort()
        ptcls.update_defocus(tomograms)
        if randomize_angles:
            n_refs, n_ptcl = ptcls.ali_eu.shape[:2]
            ptcls.ali_eu[:,:,0] = _np.random.uniform(0, 2*_np.pi, (n_refs,n_ptcl))
            ptcls.ali_eu[:,:,1] = _np.arccos(_np.random.uniform(-1, 1, (n_refs,n_ptcl)))
            ptcls.ali_eu[:,:,2] = _np.random.uniform(0, 2*_np.pi, (n_refs,n_ptcl))
        return ptcls
    
    def export_positions(self, tomograms, ref_cix=0, tomo_id=None) -> _np.ndarray:
        """Convert particle positions back to pixel coordinates relative to the tomogram corner.

        Inverse of the conversion done by ``import_data``.  Useful for
        exporting coordinates to IMOD or other tools that expect pixel-space
        positions.

        Parameters
        ----------
        tomograms : Tomograms
            Provides pixel size and tomogram dimensions.
        ref_cix : int, optional
            Reference index whose translation is included. Default 0.
        tomo_id : int, optional
            Restrict the output to the particles of this tomogram, in their
            stored order.  All the tomograms must share the same pixel size
            when converting the whole set; selecting one tomogram uses its own
            pixel size, so a mixed-binning set can still be exported one
            tomogram at a time.

        Returns
        -------
        ndarray, float32, shape (M, 3)
            Positions in pixels, relative to the tomogram corner.  ``M`` is the
            number of particles of *tomo_id* when it is given.
        """
        if tomo_id is None:
            apix = Particles._validate_tomogram(tomograms)
            mask = _np.ones(self.n_ptcl,bool)
        else:
            if not isinstance(tomograms,_tomodef.Tomograms):
                raise ValueError('Tomograms must be a Tomograms object.')
            tomo_id = _np.uint32(tomo_id)
            mask = (self.tomo_id == tomo_id)
            if not mask.any():
                raise ValueError('tomo_id %d not found in the particles.'%tomo_id)
            apix = float(tomograms.pix_size[tomograms.get_cix(tomo_id)])

        cix = tomograms.get_cix(self.tomo_id[mask])
        pos = (self.position[mask,:] + self.ali_t[ref_cix,mask,:])/apix \
            + _np.float32(tomograms.tomo_size[cix,:])/2
        return pos.astype(_np.float32)

    _ARTIAX_FIXED_COLUMNS = ('pos_x','pos_y','pos_z','shift_x','shift_y','shift_z',
                             'phi','the','psi','cc','half_id','class','tomo_id')

    @staticmethod
    def _validate_artiax_extra_name(name,other):
        if not isinstance(name,str) or len(name.strip()) == 0:
            raise ValueError('The name of an extra column must be a non-empty string.')
        name = name.strip()
        if any( c in name for c in '\t\r\n' ):
            raise ValueError('The name of an extra column cannot contain tabs or '
                             'line breaks: ' + repr(name))
        if name in Particles._ARTIAX_FIXED_COLUMNS or name == other:
            raise ValueError('The extra column cannot be named ' + repr(name)
                             + ': the name is already in use.')
        return name

    def export_artiax(self, tomograms, filename, tomo_id=None, ref_cix=0,
                      save_extra1=False, extra1_name='extra_1',
                      save_extra2=False, extra2_name='extra_2') -> None:
        """Export one tomogram's particles as an ArtiaX *Generic Particle List*.

        The file is the tab-separated ``.tsv`` list read by the ArtiaX plugin
        for ChimeraX (also loadable as a Dynamo table, which uses the same
        angular convention).  One file describes one tomogram, because the
        positions are voxel coordinates of that tomogram.

        Angles are converted from SUSAN's ZYZ (``R = Rz(α)·Ry(β)·Rz(γ)``,
        radians) to ArtiaX's ZXZ (``M = Rz(psi)·Rx(the)·Rz(phi)``, degrees).
        Using ``Ry(β) = Rz(90°)·Rx(β)·Rz(-90°)`` the conversion is exact::

            phi = degrees(γ) - 90     the = degrees(β)     psi = degrees(α) + 90

        No transpose is involved: ArtiaX's placement matrix is SUSAN's ``R``.
        (Cross-check: SUSAN's RELION writer emits ``rot,tilt,psi = -γ,-β,-α``
        and ArtiaX's RELION reader builds ``Rz(-psi)·Ry(-tilt)·Rz(-rot)``,
        which is exactly ``R`` again.)

        Columns written: the nine required ones (``pos_x pos_y pos_z shift_x
        shift_y shift_z phi the psi``) plus ``cc`` (``ali_cc``), ``half_id``,
        ``class`` (``ref_cix + 1``, 1-based as in Dynamo/RELION) and
        ``tomo_id``, which ArtiaX keeps as per-particle metadata and can colour
        or filter by.  :meth:`import_artiax` reads them back.

        :attr:`extra_1` and :attr:`extra_2` are written on request, under a name
        of your choosing (segment IDs, for instance).  ArtiaX parses every
        column as a float, so there is no integer format to choose: the values
        are written in the shortest representation that reads back as the same
        float32, which keeps whole numbers free of decimals and fractions
        exact.

        The alignment shift is folded into ``pos_*`` (as in
        :meth:`export_positions`) and the ``shift_*`` columns are zero, so the
        result does not depend on the *translation* pixel size set in ArtiaX.
        Set the *origin* pixel size in ArtiaX to the tomogram's pixel size,
        printed by this method.

        Parameters
        ----------
        tomograms : Tomograms or str
            Tomogram metadata, or the path to a ``.tomostxt`` file.  Provides
            the pixel size and the tomogram dimensions.
        filename : str
            Output path; must have a ``.tsv`` extension.
        tomo_id : int, optional
            Tomogram to export.  May be omitted when the particles belong to a
            single tomogram; otherwise the available IDs are reported.
        ref_cix : int, optional
            Reference index whose alignment is exported. Default 0.
        save_extra1, save_extra2 : bool, optional
            Write :attr:`extra_1` / :attr:`extra_2` as an additional column.
            Default False.
        extra1_name, extra2_name : str, optional
            Column names for those fields.  Defaults ``'extra_1'`` and
            ``'extra_2'``, which :meth:`import_artiax` reads without further
            configuration; pass any other name back to it explicitly.

        Examples
        --------
        >>> ptcls.export_artiax('tomos.tomostxt','tomo003.tsv',tomo_id=3)
        >>> ptcls.export_artiax(tomos,'tomo003.tsv',save_extra1=True,
        ...                     extra1_name='segment_id')
        """
        if isinstance(tomograms,str):
            tomograms = _tomodef.Tomograms(tomograms)
        if not isinstance(tomograms,_tomodef.Tomograms):
            raise ValueError('Tomograms must be a Tomograms object or the path '
                             'to a .tomostxt file.')
        if not _is_ext(filename,'tsv'):
            raise ValueError('Wrong file extension, do you mean '
                             + _force_ext(filename,'tsv') + '?')

        extra_cols = []
        if save_extra1:
            extra1_name = Particles._validate_artiax_extra_name(extra1_name,None)
            extra_cols.append((extra1_name,self.extra_1))
        if save_extra2:
            extra2_name = Particles._validate_artiax_extra_name(extra2_name,
                                        extra1_name if save_extra1 else None)
            extra_cols.append((extra2_name,self.extra_2))

        tids = _np.unique(self.tomo_id)
        if tomo_id is None:
            if tids.size != 1:
                raise ValueError('The particles span several tomograms ('
                                 + ','.join(str(t) for t in tids)
                                 + '); set tomo_id to select one.')
            tomo_id = tids[0]
        tomo_id = _np.uint32(tomo_id)
        mask = (self.tomo_id == tomo_id)
        if not mask.any():
            raise ValueError('tomo_id %d not found in the particles.'%tomo_id)

        apix = float(tomograms.pix_size[tomograms.get_cix(tomo_id)])

        # Angstroms from the tomogram centre -> voxels from its corner.
        pos = self.export_positions(tomograms,ref_cix,tomo_id)

        eu  = _np.rad2deg(self.ali_eu[ref_cix,mask,:])
        wrap = lambda a: ((a + 180.0) % 360.0) - 180.0
        phi = wrap(eu[:,2] - 90.0)
        the = eu[:,1]
        psi = wrap(eu[:,0] + 90.0)

        cc      = self.ali_cc[ref_cix,mask]
        half_id = self.half_id[mask]
        clss    = self.ref_cix[mask] + 1
        extras  = [ (name,values[mask]) for name,values in extra_cols ]

        with open(filename,'w') as fp:
            fp.write('\t'.join(('pos_x','pos_y','pos_z',
                                'shift_x','shift_y','shift_z',
                                'phi','the','psi',
                                'cc','half_id','class','tomo_id')
                               + tuple(name for name,_ in extras)) + '\n')
            for i in range(pos.shape[0]):
                fp.write('%.4f\t%.4f\t%.4f\t'%(pos[i,0],pos[i,1],pos[i,2]))
                fp.write('0\t0\t0\t')
                fp.write('%.4f\t%.4f\t%.4f\t'%(phi[i],the[i],psi[i]))
                fp.write('%.6f\t%d\t%d\t%d'%(cc[i],half_id[i],clss[i],tomo_id))
                for _,values in extras:
                    # Shortest representation that reads back as the same
                    # float32: whole numbers stay clean, fractions stay exact.
                    fp.write('\t' + _np.format_float_positional(values[i],
                                        precision=None,unique=True,trim='-'))
                fp.write('\n')

        print('[Particles.export_artiax] %d particles from tomogram %d saved to %s.'
              %(pos.shape[0],tomo_id,filename))
        if len(extras) > 0:
            print('  Extra columns: ' + ', '.join(name for name,_ in extras) + '.')
        print('  In ArtiaX, set the origin pixel size to %.3f angstroms.'%apix)

    @staticmethod
    def import_artiax(tomograms, filename, tomo_id=None,
                      extra1_name='extra_1', extra2_name='extra_2') -> Particles:
        """Create a Particles object from an ArtiaX *Generic Particle List*.

        Inverse of :meth:`export_artiax`.  Reads the tab-separated ``.tsv``
        list (a comma-separated file is also accepted) and rebuilds the
        particles, seeding the per-particle defocus from *tomograms* at the
        imported positions, so particles moved or added inside ChimeraX get
        correct CTF parameters.

        Angles are converted back from ArtiaX's ZXZ to SUSAN's ZYZ::

            α = radians(psi) - π/2    β = radians(the)    γ = radians(phi) + π/2

        and canonicalised into SUSAN's ranges (``β ∈ [0,π]``, ``α,γ ∈ (-π,π]``)
        using ``(α, β, γ) ≡ (α+π, -β, γ+π)``, so lists produced by other tools
        are handled too.

        The positions are read as ``pos_* + shift_*``, which assumes the two
        pixel sizes in ArtiaX were both set to the tomogram's pixel size (the
        files written by :meth:`export_artiax` carry zero shifts, so this is
        automatic for a round trip).

        The optional columns ``cc``, ``half_id``, ``class`` and ``tomo_id`` are
        read when present.  Particles picked inside ArtiaX carry zeros in all
        of them: their ``tomo_id`` is filled in as described below, their class
        becomes reference 0, and their half-set is assigned automatically,
        which is reported through a warning since it does not preserve any
        earlier half-set split.

        The columns named by *extra1_name* and *extra2_name* are loaded into
        :attr:`extra_1` and :attr:`extra_2` when present.  They are always
        parsed as floats: ArtiaX rewrites every column that way, so a value
        exported as ``7`` comes back as ``7.0``.

        Parameters
        ----------
        tomograms : Tomograms or str
            Tomogram metadata, or the path to a ``.tomostxt`` file.
        filename : str
            Path to the particle list.
        tomo_id : int, optional
            Tomogram the particles belong to.  Takes precedence over the
            ``tomo_id`` column, and any disagreement is reported.  Required
            when the file has no usable ``tomo_id`` column.
        extra1_name, extra2_name : str, optional
            Columns to load into :attr:`extra_1` / :attr:`extra_2`.  Default
            ``'extra_1'`` and ``'extra_2'``, matching :meth:`export_artiax`;
            pass the names used there if they were customised.  Missing columns
            are ignored.

        Returns
        -------
        Particles

        Examples
        --------
        >>> ptcls = susan.data.Particles.import_artiax('tomos.tomostxt',
        ...                                           'tomo003.tsv')
        """
        if isinstance(tomograms,str):
            tomograms = _tomodef.Tomograms(tomograms)
        if not isinstance(tomograms,_tomodef.Tomograms):
            raise ValueError('Tomograms must be a Tomograms object or the path '
                             'to a .tomostxt file.')

        required = ('pos_x','pos_y','pos_z','shift_x','shift_y','shift_z',
                    'phi','the','psi')
        with open(filename,newline='') as fp:
            header = fp.readline()
            if   '\t' in header: delimiter = '\t'
            elif ',' in header:  delimiter = ','
            else:
                raise ValueError('Cannot determine the delimiter of '+filename
                                 +': expected a tab- or comma-separated header.')
            fp.seek(0)
            reader = _csv.DictReader(fp,delimiter=delimiter)
            fields = list(reader.fieldnames or [])
            missing = [ f for f in required if f not in fields ]
            if len(missing) > 0:
                raise ValueError('Missing columns in ' + filename + ': '
                                 + ', '.join(missing))
            rows = list(reader)

        N = len(rows)
        if N == 0:
            raise ValueError('No particles in ' + filename)

        def column(name,dtype=_np.float32):
            if name not in fields:
                return None
            return _np.array([ float(r[name]) for r in rows ],dtype=dtype)

        # Positions: voxels from the corner (the shift is folded back in).
        pos = _np.zeros((N,3),_np.float32)
        for i,c in enumerate('xyz'):
            pos[:,i] = column('pos_'+c) + column('shift_'+c)

        # Tomogram: the argument wins, the column fills in, zeros are new picks.
        file_ids = column('tomo_id',_np.uint32)
        seen = _np.unique(file_ids[file_ids > 0]) if file_ids is not None \
               else _np.zeros(0,_np.uint32)
        if tomo_id is None:
            if seen.size == 0:
                raise ValueError('No usable tomo_id column in ' + filename
                                 + '; set tomo_id.')
            if seen.size > 1:
                raise ValueError('Several tomo_id values in ' + filename + ' ('
                                 + ','.join(str(t) for t in seen)
                                 + '); set tomo_id to select one.')
            tomo_id = seen[0]
            print('[Particles.import_artiax] tomo_id %d read from the file.'%tomo_id)
        else:
            tomo_id = _np.uint32(tomo_id)
            if seen.size > 0 and not (seen.size == 1 and seen[0] == tomo_id):
                print('[Particles.import_artiax] the file reports tomo_id '
                      + ','.join(str(t) for t in seen)
                      + ', using %d as requested.'%tomo_id)
        if not tomograms.has_tomo(tomo_id):
            raise ValueError('tomo_id %d not found in the tomograms.'%tomo_id)
        tomos_id = _np.full(N,tomo_id,_np.uint32)

        ptcls = Particles.import_data(tomograms,pos,tomos_id,
                                      ptcls_id=_np.arange(N))

        # import_data sorts: ptcl_id[i] is the file row of stored particle i.
        row = ptcls.ptcl_id.astype(_np.int64)

        # ZXZ (degrees) -> ZYZ (radians), then into SUSAN's ranges.
        wrap = lambda a: ((a + _np.pi) % (2*_np.pi)) - _np.pi
        alpha = _np.deg2rad(column('psi')[row]) - _np.pi/2
        beta  = wrap(_np.deg2rad(column('the')[row]))
        gamma = _np.deg2rad(column('phi')[row]) + _np.pi/2
        flip  = beta < 0
        beta [flip] = -beta[flip]
        alpha[flip] +=  _np.pi
        gamma[flip] +=  _np.pi
        ptcls.ali_eu[0,:,0] = wrap(alpha)
        ptcls.ali_eu[0,:,1] = beta
        ptcls.ali_eu[0,:,2] = wrap(gamma)

        cc = column('cc')
        if cc is not None:
            ptcls.ali_cc[0,:] = cc[row]

        half_id = column('half_id',_np.uint32)
        if half_id is not None:
            half_id = half_id[row]
            valid   = (half_id > 0)
            ptcls.half_id[valid] = half_id[valid]
            if not valid.all():
                _warnings.warn('%d particles without half_id (picked in ArtiaX?): '
                               'their half-sets were assigned automatically and do '
                               'not follow any previous split.'%int((~valid).sum()))

        clss = column('class',_np.uint32)
        if clss is not None:
            cur = _np.maximum(clss[row],1) - 1
            for _ in range(int(cur.max())):
                Particles.MRA.duplicate(ptcls,0)
            ptcls.ref_cix[:] = cur

        loaded = []
        for name,field in ((extra1_name,'extra_1'),(extra2_name,'extra_2')):
            values = column(name)
            if values is not None:
                getattr(ptcls,field)[:] = values[row]
                loaded.append('%s -> %s'%(name,field))
        if len(loaded) > 0:
            print('[Particles.import_artiax] extra columns: '
                  + ', '.join(loaded) + '.')

        print('[Particles.import_artiax] %d particles loaded from %s '
              '(tomogram %d, %d references).'
              %(N,filename,tomo_id,ptcls.n_refs))
        return ptcls





