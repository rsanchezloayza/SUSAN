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

import warnings as _warnings
import numpy as _np
from susan.utils._functions import _renamed_kw

if _TYPE_CHECKING:
    from .Particles import Particles

class PtclsMRA:
    """Multi-reference alignment helpers for Particles.

    All methods are static and operate on a Particles instance.
    Accessible as ``Particles.MRA``.
    """
    
    @staticmethod
    @_renamed_kw(ref_idx='ref_cix')
    def duplicate(ptcls, ref_cix=0) -> None:
        """Append a copy of one or more reference slots to the alignment arrays.

        Adds a new reference entry (or entries) to ``ali_eu``, ``ali_t``,
        ``ali_cc``, and ``ali_w`` by copying from the specified slot(s).
        Modifies ``ptcls`` in-place.

        Parameters
        ----------
        ptcls : Particles
            Modified in-place; ``n_refs`` increases by one (or more).
        ref_cix : int or array-like of int, optional
            Index (or indices) of the reference slot(s) to duplicate.
            Default 0.
        """
        idx = _np.array(ref_cix)
        if idx.ndim == 0:
            ptcls.ali_eu = _np.concatenate((ptcls.ali_eu,ptcls.ali_eu[ref_cix][_np.newaxis,:,:]))
            ptcls.ali_t  = _np.concatenate((ptcls.ali_t ,ptcls.ali_t [ref_cix][_np.newaxis,:,:]))
            ptcls.ali_cc = _np.concatenate((ptcls.ali_cc,ptcls.ali_cc[ref_cix][_np.newaxis,:]  ))
            ptcls.ali_w  = _np.concatenate((ptcls.ali_w ,ptcls.ali_w [ref_cix][_np.newaxis,:]  ))
        else:
            ptcls.ali_eu = _np.concatenate((ptcls.ali_eu,ptcls.ali_eu[ref_cix,:,:]))
            ptcls.ali_t  = _np.concatenate((ptcls.ali_t ,ptcls.ali_t [ref_cix,:,:]))
            ptcls.ali_cc = _np.concatenate((ptcls.ali_cc,ptcls.ali_cc[ref_cix,:]))
            ptcls.ali_w  = _np.concatenate((ptcls.ali_w ,ptcls.ali_w [ref_cix,:]))

    @staticmethod
    def expand_to_n_references(ptcls, n=None, ref_cix=None, assign=None, scope='all', cc_ref=0,
                               per_tomo=False, reset_halves=False, seed=None) -> None:
        """Add reference slots (:meth:`duplicate`) until there are ``n``, optionally reassigning the classes.

        Modifies ``ptcls`` in-place.

        Parameters
        ----------
        ptcls : Particles
        n : int, optional
            Final number of slots; must be larger than ``ptcls.n_refs``.
            Required unless ``ref_cix`` is a list.
        ref_cix : None, int or list of int
            Source of the new slots.  ``None``: the existing slots in turn
            (new slot k copies slot ``k mod n_refs``).  int: every new slot
            copies that slot.  list: the full layout, ``ref_cix[k]`` is the
            source of slot k; the existing slots map to themselves
            (``ref_cix[k] == k``) and ``n`` defaults to its length.
        assign : None, ``'random'`` or ``'cc'``
            ``None``: the classes (``ptcls.ref_cix``) are unchanged.
            ``'random'``: uniform random class.  ``'cc'``: quantile bins of
            the CC (equal counts), the highest CC to the first class of the
            group and the lowest to the last one.
        scope : ``'all'`` or ``'copies'``
            ``'all'``: every particle can go to any of the ``n`` classes; the
            CC of ``'cc'`` is ``ali_cc[cc_ref]``.  ``'copies'``: a particle
            moves only between its class and the slots copied from it (its
            class first); the CC is the one of its class.
        cc_ref : int
            Slot of the CC for ``assign='cc'`` with ``scope='all'``.  Default 0.
        per_tomo : bool
            ``assign='cc'``: quantiles per tomogram instead of global.
        reset_halves : bool
            Reassign the half-sets, balanced per class and tomogram
            (:meth:`Particles.halfsets_randomize`).  Default False: only
            safe before the first alignment, since a pose refined against one
            half-map links the halves once the particle changes half.
        seed : int, optional
            Seed of the random generator (``assign='random'``, ``reset_halves``).

        Notes
        -----
        Particles whose own slot has ``ali_w == 0`` keep their class.
        """
        n0 = ptcls.n_refs
        if ref_cix is None or _np.ndim(ref_cix) == 0:
            if n is None:
                raise ValueError('Set n (or give ref_cix as the full slot layout).')
            n = int(n)
            if n <= n0:
                raise ValueError('n (%d) must be larger than the current number of references (%d).' % (n,n0))
            src = _np.arange(n) % n0 if ref_cix is None else _np.r_[_np.arange(n0),_np.full(n-n0,int(ref_cix))]
        else:
            src = _np.asarray(ref_cix,dtype=_np.int64).ravel()
            if n is not None and int(n) != src.size:
                raise ValueError('n (%d) differs from the length of ref_cix (%d).' % (n,src.size))
            n = src.size
            if n <= n0:
                raise ValueError('ref_cix must be longer than the current number of references (%d).' % n0)
            if not _np.array_equal(src[:n0],_np.arange(n0)):
                raise ValueError('The existing slots must map to themselves: ref_cix[k] == k for k < %d.' % n0)
        if _np.any(src < 0) or _np.any(src >= n0):
            raise ValueError('The sources must be existing slots (0 to %d).' % (n0-1))
        if assign not in (None,'random','cc'):
            raise ValueError('assign must be None, "random" or "cc".')
        if scope not in ('all','copies'):
            raise ValueError('scope must be "all" or "copies".')
        if not 0 <= cc_ref < n:
            raise ValueError('cc_ref must be a slot (0 to %d).' % (n-1))

        PtclsMRA.duplicate(ptcls,src[n0:])
        rng = _np.random.default_rng(seed)

        if assign is not None:
            cls = ptcls.ref_cix.astype(_np.int64)
            active = ptcls.ali_w[cls,_np.arange(ptcls.n_ptcl)] > 0
            if scope == 'all':
                groups = [(_np.arange(n),_np.flatnonzero(active),cc_ref)]
            else:
                groups = [(_np.r_[s,_np.flatnonzero(src[n0:] == s) + n0],_np.flatnonzero(active & (cls == s)),s)
                          for s in range(n0)]
            for fam, idx, cc_slot in groups:
                if len(fam) < 2 or idx.size == 0:
                    continue
                if assign == 'random':
                    ptcls.ref_cix[idx] = fam[rng.integers(0,len(fam),idx.size)]
                    continue
                parts = [idx[ptcls.tomo_id[idx] == t] for t in _np.unique(ptcls.tomo_id[idx])] if per_tomo else [idx]
                for p in parts:
                    if p.size < len(fam):
                        _warnings.warn('%d particles%s for %d classes: some classes stay empty.' % (
                            p.size,' in tomogram %d' % ptcls.tomo_id[p[0]] if per_tomo else '',len(fam)),stacklevel=2)
                    order = _np.argsort(-ptcls.ali_cc[cc_slot,p],kind='stable')
                    rank = _np.empty(p.size,dtype=_np.int64)
                    rank[order] = _np.arange(p.size)
                    ptcls.ref_cix[p] = fam[(rank*len(fam))//p.size]

        if reset_halves:
            ptcls.halfsets_randomize(seed=seed)

    @staticmethod
    def append_as_reference(ptcls_a, ptcls_b, id_offset=10**5, tomograms=None) -> Particles:
        """Merge two particle sets, the classes of ``ptcls_b`` after those of ``ptcls_a``.

        Setup for a 2D alignment or CTF refinement with several species
        (experimental: multi-class 2D/CTF has not been validated on data).
        The result has ``n_refs(a) + n_refs(b)`` slots; the classes of
        ``ptcls_b`` are shifted by ``n_refs(a)``, so the references file must
        list the maps of ``ptcls_a`` and then those of ``ptcls_b``, in slot
        order.  Each particle's own slot (``ali_*`` at its class) is copied to
        the slots of the other set; ``prj_*``, defocus, ``half_id`` and
        ``extra_*`` are kept.  The inputs are unchanged.

        Parameters
        ----------
        ptcls_a, ptcls_b : Particles
            Same ``n_proj``.
        id_offset : int
            Added to the ``ptcl_id`` of ``ptcls_b`` (with a warning); must be
            larger than the largest ``ptcl_id`` of ``ptcls_a``.  Default 10^5.
        tomograms : Tomograms or str, optional
            When given, every ``tomo_id`` must be in it and ``tomo_cix`` is
            recomputed.

        Returns
        -------
        Particles
        """
        if ptcls_a.n_proj != ptcls_b.n_proj:
            raise ValueError('Different number of projections (%d and %d).' % (ptcls_a.n_proj,ptcls_b.n_proj))
        id_offset = int(id_offset)
        if ptcls_a.n_ptcl > 0 and id_offset <= int(ptcls_a.ptcl_id.max()):
            raise ValueError('id_offset (%d) must be larger than the largest ptcl_id of the first set (%d).' % (
                id_offset,int(ptcls_a.ptcl_id.max())))
        if ptcls_b.n_ptcl > 0 and int(ptcls_b.ptcl_id.max()) + id_offset > _np.iinfo(_np.uint32).max:
            raise ValueError('id_offset (%d) makes the ptcl_id of the second set overflow uint32.' % id_offset)

        na, nb = ptcls_a.n_refs, ptcls_b.n_refs
        a, b = ptcls_a.copy(), ptcls_b.copy()
        b.ptcl_id += _np.uint32(id_offset)
        b.ref_cix += _np.uint32(na)
        _warnings.warn('The ptcl_id of the second set are renamed: +%d.' % id_offset,stacklevel=2)

        def widen(p, n_before, n_after):
            own = p.ref_cix.astype(_np.int64) - n_before
            i = _np.arange(p.n_ptcl)
            for name in ('ali_eu','ali_t','ali_cc','ali_w'):
                v = getattr(p,name)
                fill = v[own,i]
                setattr(p,name,_np.concatenate((_np.broadcast_to(fill,(n_before,)+fill.shape),v,
                                                _np.broadcast_to(fill,(n_after,)+fill.shape))).astype(v.dtype))
        widen(a,0,nb)
        widen(b,na,0)
        a.append_ptcls(b)
        if tomograms is not None:
            if isinstance(tomograms,str):
                from susan.data.Tomograms import Tomograms as _Tomograms
                tomograms = _Tomograms(tomograms)
            a.update_tomo_cix(tomograms)
        return a

    @staticmethod
    @_renamed_kw(ref_idx='ref_cix')
    def select_ref(ptcls, ref_cix) -> Particles:
        """Select particles assigned to specific reference(s) and keep only those slots.

        Filters ``ptcls`` to particles whose class (``ptcls.ref_cix``) is in ``ref_cix``
        and trims the alignment arrays to only the requested reference(s).
        ``ref_cix`` values in the result are remapped to 0-based indices.
        Returns a new Particles object; the original is unchanged.

        Parameters
        ----------
        ptcls : Particles
        ref_cix : int or array-like of int
            Reference index (or indices) to retain.

        Returns
        -------
        Particles
        """
        idx = _np.array(ref_cix)
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

        
        
        
        
        
        
        
        
        
        
