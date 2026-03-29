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
from numba import jit as _jit
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.utils import rotm_euZYZ as _rotm_euZYZ

class PtclsGeom:
    """Geometry operations on Particles alignment data.

    All methods are static and operate on a Particles instance in-place or
    return a new one.  Accessible as ``Particles.Geom``.
    """
    
###############################################################################
    @staticmethod
    def _validate_single_rotation(eZYZdeg,R):
        if eZYZdeg is not None and R is not None:
            raise ValueError('Set either eZYZdeg or R, not both.')
        
        if eZYZdeg is None and R is None:
            return None
        
        if eZYZdeg is not None:
        
            eu = _np.deg2rad(_np.array(eZYZdeg,dtype=_np.float32))
            if eu.size != 3:
                raise ValueError('eZYZdeg must be a 3-element array/vector.')
            R = _np.zeros((3,3),_np.float32)
            _euZYZ_rotm(R,eu)
        
        else:
            if R.ndim != 2 or R.shape[0] != 3 or R.shape[1] != 3:
                raise ValueError('R must be a 3x3 matrix.')
        
        return _np.float32(R)
    
    @staticmethod
    def _validate_single_translation(t):
        if t is None:
            t = _np.zeros(3,_np.float32)
        else:
            t = _np.array(t,_np.float32)
            if t.size != 3:
                raise ValueError('t must be a 3-element array/vector.')
        return t
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _inplace_shift(ali_eZYZ,ali_t,t):
        R = _np.zeros((3,3),_np.float32)
        for i in range(ali_eZYZ.shape[0]):
            _euZYZ_rotm(R,ali_eZYZ[i])
            tout = R@t
            ali_t[i,:] = ali_t[i,:] + tout
            
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _inplace_rot_shift(ali_eZYZ,ali_t,R,t):
        R_in = _np.zeros((3,3),_np.float32)
        Rout = _np.zeros((3,3),_np.float32)
        for i in range(ali_eZYZ.shape[0]):
            _euZYZ_rotm(R_in,ali_eZYZ[i])
            Rout = (R@R_in.transpose()).transpose()
            _rotm_euZYZ(ali_eZYZ[i],Rout)
            tout = Rout@t
            ali_t[i,:] = ali_t[i,:] + tout
    
    @staticmethod
    def rot_shift(ptcls, eZYZdeg=None, R=None, t=None, ref_idx=0):
        """Apply a single rotation and/or translation to all particles in-place.

        Modifies ``ali_eu[ref_idx]`` and ``ali_t[ref_idx]`` directly.
        Supply either ``eZYZdeg`` or ``R``, not both.

        Parameters
        ----------
        ptcls : Particles
        eZYZdeg : array-like of float (3,), optional
            ZYZ Euler angles in degrees.
        R : ndarray, float32 (3, 3), optional
            Rotation matrix.
        t : array-like of float (3,), optional
            Translation in Ångströms. Default zeros.
        ref_idx : int, optional
            Reference alignment slot to modify. Default 0.
        """
        R = PtclsGeom._validate_single_rotation(eZYZdeg,R)
        t = PtclsGeom._validate_single_translation(t)
                
        if R is None:
            PtclsGeom._inplace_shift(ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],t)
        else:
            PtclsGeom._inplace_rot_shift(ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],R,t)
    
###############################################################################
    @staticmethod
    def _validate_multiple_rotations(eZYZdeg,R):
        if eZYZdeg is not None and R is not None:
            raise ValueError('Set either eZYZdeg or R, not both.')
        
        if eZYZdeg is None and R is None:
            return None
        
        if eZYZdeg is not None:
        
            eu = _np.deg2rad(_np.array(eZYZdeg,dtype=_np.float32))
            if eu.ndim == 1 and eu.size == 3:
                R = _np.zeros((1,3,3),_np.float32)
                _euZYZ_rotm(R[0],eu)
            elif eu.ndim == 2 and eu.shape[1] == 3:
                R = _np.zeros((eu.shape[0],3,3),_np.float32)
                for i in range(eu.shape[0]):
                    _euZYZ_rotm(R[i],eu[i])
            else:
                raise ValueError('eZYZdeg must be a 3-element array/vector or a stack of them.')
            
        
        else:
            if R.ndim < 2 or R.ndim > 3:
                raise ValueError('R must be a 3-by-3 matrix or a stack of them.')
            elif R.ndim == 2:
                R = R[_np.newaxis,:,:]
        
        return R

    @staticmethod
    def _validate_multiple_translations(t):
        if t is not None:
            t = _np.array(t,_np.float32)
            if t.ndim == 1:
                t = t[_np.newaxis,:]
            elif t.ndim != 2:
                raise ValueError('t must be a 1D or 2D matrix.')
        return t

    @staticmethod
    def _validate_multiple_inputs(R,t):
        if R is None and t is None:
            raise ValueError('Set the angles or the shifts...')
        
        if R is not None and t is not None:
            if R.shape[0] != t.shape[0]:
                raise ValueError('Number of angles do not match the number of shifts.')
        
        if R is None and t is not None:
            if t.shape[1] != 3:
                raise ValueError('t is not a N-by-3 matrix.')
        
        if R is not None and t is None:
            if R.shape[1] != 3 or R.shape[2] != 3:
                raise ValueError('R is not a N-by-3-by-3 matrix.')
            t = _np.zeros((R.shape[0],3),_np.float32)
        
        return R,t
    
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _outplace_shift(out_ali_t,in_ali_eZYZ,in_ali_t,t):
        R = _np.zeros((3,3),_np.float32)
        out_ix = 0
        for i in range(in_ali_eZYZ.shape[0]):
            _euZYZ_rotm(R,in_ali_eZYZ[i])
            for j in range(t.shape[0]):
                tout = R@t[j]
                out_ali_t[out_ix,:] = in_ali_t[i,:] + tout
                out_ix = out_ix + 1

    @staticmethod
    @_jit(nopython=True,cache=True)
    def _outplace_rot_shift(out_ali_eZYZ,out_ali_t,in_ali_eZYZ,in_ali_t,R,t):
        R_in = _np.zeros((3,3),_np.float32)
        Rout = _np.zeros((3,3),_np.float32)
        out_ix = 0
        for i in range(in_ali_eZYZ.shape[0]):
            _euZYZ_rotm(R_in,in_ali_eZYZ[i])
            for j in range(t.shape[0]):
                Rout = (R[j]@R_in.transpose()).transpose()
                tout = Rout@t[j]
                out_ali_t[out_ix,:] = in_ali_t[i,:] + tout
                _rotm_euZYZ(out_ali_eZYZ[out_ix],Rout)
                out_ix = out_ix + 1

    @staticmethod
    def expand_by_rot_shift(ptcls, eZYZdeg=None, R=None, t=None, ref_idx=0):
        """Expand a particle list by applying multiple rotations/translations.

        For each particle, produces one output copy per supplied
        rotation/translation, resulting in ``n_ptcl × n_transforms`` particles.
        Useful for symmetry expansion.  Returns a new Particles object;
        the original is unchanged.

        Parameters
        ----------
        ptcls : Particles
        eZYZdeg : array-like, shape (K, 3) or (3,), optional
            ZYZ Euler angles in degrees for each transform.
        R : ndarray, float32, shape (K, 3, 3) or (3, 3), optional
            Rotation matrices. Mutually exclusive with eZYZdeg.
        t : array-like, shape (K, 3) or (3,), optional
            Translations in Ångströms for each transform. Default zeros.
        ref_idx : int, optional
            Reference alignment slot to use as input and output. Default 0.

        Returns
        -------
        Particles
        """
        R   = PtclsGeom._validate_multiple_rotations(eZYZdeg,R)
        t   = PtclsGeom._validate_multiple_translations(t)
        R,t = PtclsGeom._validate_multiple_inputs(R,t)
        
        idx_expand = _np.tile(_np.arange(ptcls.n_ptcl),(t.shape[0],1)).transpose().flatten()
        ptcls_out = ptcls.select(idx_expand)
        
        if R is None:
            PtclsGeom._outplace_shift(ptcls_out.ali_t[ref_idx],ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],t)
        else:
            PtclsGeom._outplace_rot_shift(ptcls_out.ali_eu[ref_idx],ptcls_out.ali_t[ref_idx],ptcls.ali_eu[ref_idx],ptcls.ali_t[ref_idx],R,t)
        
        ptcls_out.update_position(ref_idx)
        return ptcls_out
        
###############################################################################
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _enable_by_tilt(prj_w,tomo_cix,prj_eu,prj_wgt,tilt_min,tilt_max):
        for p in range(prj_w.shape[0]):
            t_id = tomo_cix[p]
            for k in range(prj_w.shape[1]):
                if prj_wgt[t_id,k] > 0:
                    tilt = _np.abs(prj_eu[t_id,k,1])
                    prj_w[p,k] = (tilt<tilt_max)&(tilt>=tilt_min)
                else:
                    prj_w[p,k] = 0
    
    @staticmethod
    def enable_by_tilt(ptcls, tomos, tilt_deg_max, tilt_deg_min=0):
        """Set per-projection weights based on tilt angle range.

        Projections whose absolute tilt angle falls within
        [tilt_deg_min, tilt_deg_max) are set to weight 1; all others are
        set to 0.  Projections already excluded in the Tomograms metadata
        (``proj_wgt == 0``) remain excluded.

        Parameters
        ----------
        ptcls : Particles
            Modified in-place (``prj_w`` updated).
        tomos : Tomograms
        tilt_deg_max : float
            Maximum absolute tilt angle to include (degrees).
        tilt_deg_min : float, optional
            Minimum absolute tilt angle to include (degrees). Default 0.
        """
        tilt_max = _np.abs(tilt_deg_max)
        tilt_min = _np.abs(tilt_deg_min)
        PtclsGeom._enable_by_tilt(ptcls.prj_w,ptcls.tomo_cix,tomos.proj_eZYZ,tomos.proj_wgt,tilt_min,tilt_max)

    @staticmethod
    @_jit(nopython=True,cache=True)
    def _enable_by_tilt_range(prj_w,tomo_cix,prj_eu,prj_wgt,tilt_min,tilt_max):
        R  = _np.zeros((3,3),_np.float32)
        eu = _np.zeros(3,    _np.float32)
        for p in range(prj_w.shape[0]):
            t_id = tomo_cix[p]
            for k in range(prj_w.shape[1]):
                if prj_wgt[t_id,k] > 0:
                    eu[0] = prj_eu[t_id,k,0] * _np.pi / 180.0
                    eu[1] = prj_eu[t_id,k,1] * _np.pi / 180.0
                    eu[2] = prj_eu[t_id,k,2] * _np.pi / 180.0
                    _euZYZ_rotm(R, eu)
                    # Beam direction is the third column of R.
                    # Project it onto the tilt direction [cos(eu0), sin(eu0), 0]
                    # (perpendicular to the tilt axis) to get the signed
                    # horizontal component; atan2 with R[2,2] gives the signed
                    # tilt angle robustly in [-pi, pi].
                    c0    = _np.cos(eu[0])
                    s0    = _np.sin(eu[0])
                    horiz = c0*R[0,2] + s0*R[1,2]
                    tilt  = _np.arctan2(horiz, R[2,2])
                    prj_w[p,k] = (tilt >= tilt_min) & (tilt < tilt_max)
                else:
                    prj_w[p,k] = 0

    @staticmethod
    def enable_by_tilt_range(ptcls, tomos, tilt_deg_min, tilt_deg_max):
        """Set per-projection weights based on a signed tilt-angle range.

        The tilt angle is derived from the full ZYZ rotation matrix of each
        projection (not from ``proj_eZYZ[:,1]`` directly), making it robust
        to non-canonical Euler-angle storage.  The signed angle is defined as
        the angle between the beam direction and the tomogram Z axis, positive
        in the direction of increasing stage tilt.

        Projections whose signed tilt falls within ``[tilt_deg_min,
        tilt_deg_max)`` are set to weight 1; all others are set to 0.
        Projections already excluded in the Tomograms metadata
        (``proj_wgt == 0``) remain excluded.

        Unlike :meth:`enable_by_tilt`, both bounds are signed, so asymmetric
        ranges such as ``(-20, 40)`` are supported.

        Parameters
        ----------
        ptcls : Particles
            Modified in-place (``prj_w`` updated).
        tomos : Tomograms
        tilt_deg_min : float
            Lower bound of the signed tilt range in degrees (inclusive).
        tilt_deg_max : float
            Upper bound of the signed tilt range in degrees (exclusive).
        """
        if tilt_deg_min >= tilt_deg_max:
            raise ValueError(
                f'tilt_deg_min ({tilt_deg_min}) must be less than tilt_deg_max ({tilt_deg_max}).')
        if tilt_deg_min < -180.0 or tilt_deg_max > 180.0:
            raise ValueError(
                f'Tilt range [{tilt_deg_min}, {tilt_deg_max}) exceeds [-180, 180] degrees.')
        tilt_min = _np.float32(_np.deg2rad(tilt_deg_min))
        tilt_max = _np.float32(_np.deg2rad(tilt_deg_max))
        PtclsGeom._enable_by_tilt_range(ptcls.prj_w,ptcls.tomo_cix,tomos.proj_eZYZ,tomos.proj_wgt,tilt_min,tilt_max)

###############################################################################
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _disable_closer(w_mask,pos,sort_ix,dist2):
        N = sort_ix.size
        for i_cur in range(N):
            idx_cur = sort_ix[i_cur]
            if w_mask[idx_cur]:
                for i_nxt in range(i_cur+1,N):
                    idx_nxt = sort_ix[i_nxt]
                    if w_mask[idx_nxt]:
                        vec = pos[idx_cur] - pos[idx_nxt]
                        d   = (vec*vec).sum()
                        if  d < dist2:
                            w_mask[idx_nxt] = False
        
    @staticmethod
    def discard_closer(ptcls, min_dist_angs, ref_idx=0, verbose=False):
        """Remove duplicate/overlapping particles closer than a minimum distance.

        Within each tomogram, particles are sorted by descending ``ali_cc``
        and greedily kept; any particle within ``min_dist_angs`` of an already
        kept particle is discarded.  Returns a new Particles object.

        Parameters
        ----------
        ptcls : Particles
        min_dist_angs : float
            Minimum allowed inter-particle distance in Ångströms.
        ref_idx : int, optional
            Reference index used to compute effective positions
            (position + ali_t[ref_idx]). Default 0.
        verbose : bool, optional
            Print per-tomogram particle counts. Default False.

        Returns
        -------
        Particles
        """
        t_id = _np.unique( ptcls.tomo_cix )
        mask = _np.ones(ptcls.tomo_cix.shape,bool)
        dist = min_dist_angs*min_dist_angs
        
        if verbose:
            print('%d particles in %d tomograms. Processing:'%(ptcls.n_ptcl,t_id.size))
            
        for tid in t_id:
            t_mask  = ptcls.tomo_cix == tid
            cur_cc  = ptcls.ali_cc[ref_idx,t_mask]
            sort_ix = _np.argsort(cur_cc)[::-1]
            pos     = ptcls.position[t_mask] + ptcls.ali_t[ref_idx,t_mask]
            w_mask  = mask[t_mask]
            PtclsGeom._disable_closer(w_mask,pos,sort_ix,dist)
            mask[t_mask] = w_mask
            if verbose:
                print('\tTomogram index %3d: from %7d to %7d particles.'%(tid,sort_ix.size,w_mask.sum()))
        
        if verbose:
            print('Remaining particles: %d'%(mask.sum()))
        
        return ptcls.select( mask )
    
###############################################################################
    @staticmethod
    @_jit(nopython=True,cache=True)
    def _get_min_dist(dist,pos):
        N = pos.shape[0]
        for idx_cur in range(N):
            min_d   = _np.inf
            for idx_wrk in range(N):
                if idx_cur != idx_wrk:
                    vec = pos[idx_cur] - pos[idx_wrk]
                    d   = _np.sqrt( (vec*vec).sum() )
                    if d < min_d:
                        min_d = d
            dist[idx_cur] = min_d
    
    @staticmethod
    def get_min_distance(ptcls, ref_idx=0):
        """Return the distance to the nearest neighbour for every particle.

        Computed per tomogram using effective positions
        (position + ali_t[ref_idx]).

        Parameters
        ----------
        ptcls : Particles
        ref_idx : int, optional
            Reference index for the translation offset. Default 0.

        Returns
        -------
        ndarray, float32, shape (M,)
            Nearest-neighbour distance in Ångströms for each particle.
        """
        t_id = _np.unique( ptcls.tomo_cix )
        dist = _np.zeros(ptcls.tomo_cix.shape,_np.float32)
        
        for tid in t_id:
            t_mask  = ptcls.tomo_cix == tid
            pos     = ptcls.position[t_mask] + ptcls.ali_t[ref_idx,t_mask]
            d_mask  = dist[t_mask]
            PtclsGeom._get_min_dist(d_mask,pos)
            dist[t_mask] = d_mask
        
        return dist
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
