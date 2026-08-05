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

from typing import TYPE_CHECKING as _TYPE_CHECKING

import numpy as _np
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.utils import rotm_euZYZ as _rotm_euZYZ

if _TYPE_CHECKING:
    from .Particles import Particles

class PtclsMRA:
    """Multi-reference alignment helpers for Particles.

    All methods are static and operate on a Particles instance.
    Accessible as ``Particles.MRA``.
    """
    
    @staticmethod
    def duplicate(ptcls, ref_idx=0) -> None:
        """Append a copy of one or more reference slots to the alignment arrays.

        Adds a new reference entry (or entries) to ``ali_eu``, ``ali_t``,
        ``ali_cc``, and ``ali_w`` by copying from the specified slot(s).
        Modifies ``ptcls`` in-place.

        Parameters
        ----------
        ptcls : Particles
            Modified in-place; ``n_refs`` increases by one (or more).
        ref_idx : int or array-like of int, optional
            Index (or indices) of the reference slot(s) to duplicate.
            Default 0.
        """
        idx = _np.array(ref_idx)
        if idx.ndim == 0:
            ptcls.ali_eu = _np.concatenate((ptcls.ali_eu,ptcls.ali_eu[ref_idx][_np.newaxis,:,:]))
            ptcls.ali_t  = _np.concatenate((ptcls.ali_t ,ptcls.ali_t [ref_idx][_np.newaxis,:,:]))
            ptcls.ali_cc = _np.concatenate((ptcls.ali_cc,ptcls.ali_cc[ref_idx][_np.newaxis,:]  ))
            ptcls.ali_w  = _np.concatenate((ptcls.ali_w ,ptcls.ali_w [ref_idx][_np.newaxis,:]  ))
        else:
            ptcls.ali_eu = _np.concatenate((ptcls.ali_eu,ptcls.ali_eu[ref_idx,:,:]))
            ptcls.ali_t  = _np.concatenate((ptcls.ali_t ,ptcls.ali_t [ref_idx,:,:]))
            ptcls.ali_cc = _np.concatenate((ptcls.ali_cc,ptcls.ali_cc[ref_idx,:]))
            ptcls.ali_w  = _np.concatenate((ptcls.ali_w ,ptcls.ali_w [ref_idx,:]))
    
    @staticmethod
    def select_ref(ptcls, ref_idx) -> Particles:
        """Select particles assigned to specific reference(s) and keep only those slots.

        Filters ``ptcls`` to particles whose ``ref_cix`` matches ``ref_idx``
        and trims the alignment arrays to only the requested reference(s).
        ``ref_cix`` values in the result are remapped to 0-based indices.
        Returns a new Particles object; the original is unchanged.

        Parameters
        ----------
        ptcls : Particles
        ref_idx : int or array-like of int
            Reference index (or indices) to retain.

        Returns
        -------
        Particles
        """
        idx = _np.array(ref_idx)
        if idx.ndim == 0:
            rslt = ptcls.select( ptcls.ref_cix == idx )
            rslt.ali_eu = rslt.ali_eu[idx,:,:][_np.newaxis,:,:]
            rslt.ali_t  = rslt.ali_t [idx,:,:][_np.newaxis,:,:]
            rslt.ali_cc = rslt.ali_cc[idx,:]  [_np.newaxis,:]
            rslt.ali_w  = rslt.ali_w [idx,:]  [_np.newaxis,:]
            rslt.ref_cix[:] = 0
        else:
            mask = _np.zeros( ptcls.n_ptcl, bool )
            for i in range(idx.shape[0]):
                mask = mask | (ptcls.ref_cix == idx[i])
            rslt = ptcls.select( mask )
            rslt.ali_eu = rslt.ali_eu[idx,:,:]
            rslt.ali_t  = rslt.ali_t [idx,:,:]
            rslt.ali_cc = rslt.ali_cc[idx,:]
            rslt.ali_w  = rslt.ali_w [idx,:]
            orig = rslt.ref_cix.copy()
            for i in range(idx.shape[0]):
                rslt.ref_cix[ orig==idx[i] ] = i
        return rslt

        
        
        
        
        
        
        
        
        
        
