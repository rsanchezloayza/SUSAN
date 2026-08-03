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
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.data._ptclsgeom_core import (
    _inplace_shift,
    _inplace_rot_shift,
    _outplace_shift,
    _outplace_rot_shift,
    _enable_by_tilt,
    _enable_by_tilt_range,
    _disable_closer,
    _get_min_dist,
)

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

        return _np.ascontiguousarray(R, dtype=_np.float32)
    
    @staticmethod
    def _validate_single_translation(t):
        if t is None:
            t = _np.zeros(3,_np.float32)
        else:
            t = _np.array(t,_np.float32)
            if t.size != 3:
                raise ValueError('t must be a 3-element array/vector.')
        return t
    
    _inplace_shift     = staticmethod(_inplace_shift)
    _inplace_rot_shift = staticmethod(_inplace_rot_shift)
    
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

        return _np.ascontiguousarray(R, dtype=_np.float32)

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
    
    _outplace_shift     = staticmethod(_outplace_shift)
    _outplace_rot_shift = staticmethod(_outplace_rot_shift)

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
    _enable_by_tilt = staticmethod(_enable_by_tilt)

    @staticmethod
    def _enable_by_tilt_nominal(ptcls, tomos, tilt_deg_min, tilt_deg_max, signed):
        cix    = tomos.get_cix(ptcls.tomo_id)
        n_proj = ptcls.prj_w.shape[1]
        tilts  = tomos.nominal_tilt_angles[cix, :n_proj]
        wgts   = tomos.proj_wgt[cix, :n_proj]
        if not signed:
            tilts = _np.abs(tilts)
        cond = (tilts >= tilt_deg_min) & (tilts < tilt_deg_max) & (wgts > 0)
        ptcls.prj_w[:, :n_proj] = cond.astype(_np.float32)
        if ptcls.prj_w.shape[1] > n_proj:
            ptcls.prj_w[:, n_proj:] = 0.0

    @staticmethod
    def enable_by_tilt(ptcls, tomos, tilt_deg_max, tilt_deg_min=0, use_nominal=False):
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
        use_nominal : bool, optional
            If True, use ``tomos.nominal_tilt_angles`` instead of the Y
            component of ``proj_eZYZ``.  Default False.
        """
        tilt_max = _np.abs(tilt_deg_max)
        tilt_min = _np.abs(tilt_deg_min)
        if use_nominal:
            PtclsGeom._enable_by_tilt_nominal(ptcls, tomos, tilt_min, tilt_max, signed=False)
        else:
            cix = tomos.get_cix(ptcls.tomo_id).astype(_np.uint32)
            PtclsGeom._enable_by_tilt(ptcls.prj_w,cix,tomos.proj_eZYZ,tomos.proj_wgt,tilt_min,tilt_max)

    _enable_by_tilt_range = staticmethod(_enable_by_tilt_range)

    @staticmethod
    def enable_by_tilt_range(ptcls, tomos, tilt_deg_min, tilt_deg_max, use_nominal=False):
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
        use_nominal : bool, optional
            If True, use ``tomos.nominal_tilt_angles`` directly as the
            signed stage tilt instead of deriving it from ``proj_eZYZ``.
            Default False.
        """
        if tilt_deg_min >= tilt_deg_max:
            raise ValueError(
                f'tilt_deg_min ({tilt_deg_min}) must be less than tilt_deg_max ({tilt_deg_max}).')
        if tilt_deg_min < -180.0 or tilt_deg_max > 180.0:
            raise ValueError(
                f'Tilt range [{tilt_deg_min}, {tilt_deg_max}) exceeds [-180, 180] degrees.')
        if use_nominal:
            PtclsGeom._enable_by_tilt_nominal(ptcls, tomos, tilt_deg_min, tilt_deg_max, signed=True)
        else:
            tilt_min = _np.float32(_np.deg2rad(tilt_deg_min))
            tilt_max = _np.float32(_np.deg2rad(tilt_deg_max))
            cix = tomos.get_cix(ptcls.tomo_id).astype(_np.uint32)
            PtclsGeom._enable_by_tilt_range(ptcls.prj_w,cix,tomos.proj_eZYZ,tomos.proj_wgt,tilt_min,tilt_max)

###############################################################################
    _disable_closer = staticmethod(_disable_closer)
        
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
        t_id = _np.unique( ptcls.tomo_id )
        mask = _np.ones(ptcls.tomo_id.shape,bool)
        dist = min_dist_angs*min_dist_angs
        
        if verbose:
            print('%d particles in %d tomograms. Processing:'%(ptcls.n_ptcl,t_id.size))
            
        for tid in t_id:
            t_mask  = ptcls.tomo_id == tid
            cur_cc  = ptcls.ali_cc[ref_idx,t_mask]
            sort_ix = _np.ascontiguousarray(_np.argsort(cur_cc)[::-1])
            pos     = ptcls.position[t_mask] + ptcls.ali_t[ref_idx,t_mask]
            w_mask  = _np.array(mask[t_mask], dtype=_np.uint8)
            PtclsGeom._disable_closer(w_mask,pos,sort_ix,dist)
            mask[t_mask] = w_mask.astype(bool)
            if verbose:
                print('\tTomogram index %3d: from %7d to %7d particles.'%(tid,sort_ix.size,w_mask.sum()))
        
        if verbose:
            print('Remaining particles: %d'%(mask.sum()))

        return ptcls.select( mask )

###############################################################################
    @staticmethod
    def discard_oversampled_views(ptcls, bin_size_deg=5.0, k_per_bin=1,
                                  ref_idx=0, weight_mask=True, verbose=False):
        """Flatten preferential orientation by keeping the best particles per view bin.

        Particles are binned by view direction on an equal-area "ring" grid:
        180°/bin_size_deg latitude rings, each split into a longitude count
        proportional to sin(latitude) so cells stay compact and equal-area
        from pole to equator.  Within each bin, up to ``k_per_bin`` particles
        with the highest ``ali_cc`` are kept; the rest are discarded.  Returns
        a new Particles object.

        Parameters
        ----------
        ptcls : Particles
        bin_size_deg : float, optional
            Angular resolution of the equal-area grid in degrees. Default 5.0.
        k_per_bin : int, optional
            Maximum number of particles kept per bin. Default 1.
        ref_idx : int, optional
            Reference index used to read angles and cc. Default 0.
        weight_mask : bool, optional
            If True, particles with ``ali_w[ref_idx] <= 0`` are never kept.
            Default True.
        verbose : bool, optional
            Print how many particles were kept. Default False.

        Returns
        -------
        Particles
        """
        if bin_size_deg <= 0:
            raise ValueError('bin_size_deg must be positive.')
        if k_per_bin < 1:
            raise ValueError('k_per_bin must be >= 1.')

        n_rings = int(round(180.0/bin_size_deg))

        e0 = ptcls.ali_eu[ref_idx,:,0]
        e1 = ptcls.ali_eu[ref_idx,:,1]
        cc = ptcls.ali_cc[ref_idx,:].astype(_np.float64,copy=True)

        # View-direction unit vector (works for either ZYZ-range convention).
        vx  = _np.cos(e0)*_np.sin(e1)
        vy  = _np.sin(e0)*_np.sin(e1)
        vz  = _np.cos(e1)
        theta = _np.arccos(_np.clip(vz,-1.0,1.0))   # colatitude, [0,pi]
        lon   = _np.arctan2(vy,vx)                  # azimuth,    [-pi,pi]

        if weight_mask:
            cc[ptcls.ali_w[ref_idx,:] <= 0] = -_np.inf

        # Equal-area "ring" grid: equal-width latitude rings, each split into
        # a longitude count proportional to sin(theta). Cell area stays ~dtheta^2
        # and cells stay compact from pole to equator (no polar slivers).
        ring_edges  = _np.linspace(0.0,_np.pi,n_rings+1)
        ring_center = 0.5*(ring_edges[:-1]+ring_edges[1:])
        n_lon_ring  = _np.maximum(1,
            _np.round(2*n_rings*_np.sin(ring_center)).astype(_np.int64))
        ring_offset = _np.concatenate(([0],_np.cumsum(n_lon_ring)))

        r    = _np.clip(_np.digitize(theta,ring_edges)-1,0,n_rings-1)
        frac = (lon+_np.pi)/(2*_np.pi)
        c    = _np.clip((frac*n_lon_ring[r]).astype(_np.int64),0,n_lon_ring[r]-1)
        cell = ring_offset[r] + c

        # Sort by (cell asc, cc desc); take the first k_per_bin per cell.
        order       = _np.lexsort((-cc,cell))
        cell_sorted = cell[order]
        new_cell    = _np.empty_like(cell_sorted,dtype=bool)
        new_cell[0] = True
        new_cell[1:] = cell_sorted[1:] != cell_sorted[:-1]
        rank = _np.arange(cell_sorted.size) - _np.maximum.accumulate(
            _np.where(new_cell,_np.arange(cell_sorted.size),0))

        mask = _np.zeros(ptcls.n_ptcl,dtype=bool)
        keep = order[(rank < k_per_bin) & (cc[order] > -_np.inf)]
        mask[keep] = True

        if verbose:
            print('Kept %d / %d particles (%d filled bins of %d).'%(
                mask.sum(),ptcls.n_ptcl,
                _np.unique(cell[mask]).size,int(ring_offset[-1])))

        return ptcls.select(mask)

###############################################################################
    _get_min_dist = staticmethod(_get_min_dist)
    
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
        t_id = _np.unique( ptcls.tomo_id )
        dist = _np.zeros(ptcls.tomo_id.shape,_np.float32)
        
        for tid in t_id:
            t_mask  = ptcls.tomo_id == tid
            pos     = ptcls.position[t_mask] + ptcls.ali_t[ref_idx,t_mask]
            d_mask  = dist[t_mask]
            PtclsGeom._get_min_dist(d_mask,pos)
            dist[t_mask] = d_mask
        
        return dist
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
