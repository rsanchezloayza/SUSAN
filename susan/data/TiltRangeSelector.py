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

import os as _os
import numpy as _np

import susan.io.mrc as _mrc
from susan.data.Tomograms import Tomograms as _Tomograms
from susan.data.Tomograms import lookup_cix as _lookup_cix
from susan.data.Particles import Particles as _Particles
from susan.utils import euZYZ_rotm as _euZYZ_rotm
from susan.utils import rotm_euZYZ as _rotm_euZYZ


class TiltRangeSelector:
    """Build per-tomogram projection masks for a signed tilt range and emit
    reduced :class:`~susan.data.Tomograms` / :class:`~susan.data.Particles`
    on the surviving projections.

    The selector is built once from a source :class:`Tomograms` and a signed
    tilt range ``[tilt_deg_min, tilt_deg_max)``.  For each tomogram it
    computes the indices of projections whose canonicalised signed tilt
    falls within that range *and* that are currently active
    (``proj_wgt > 0``).  The same selector can then be applied to multiple
    inputs that share the projection layout — typically a b1 and a b2
    :class:`Tomograms` produced via :meth:`~susan.data.Tomograms.bin` — so
    the index arithmetic only happens once.

    The reduced output uses a single rectangular per-projection axis of
    length ``new_n_projs = max_t(kept_count[t])``; tomograms keeping fewer
    projections pad the trailing slots with zeros (``proj_wgt = 0``).  Each
    rewritten stack MRC contains only that tomogram's kept slices, so
    ``stack_size[t, 2] = kept_count[t]``.

    .. rubric:: Tilt convention

    ``Rmat_eZYZ`` (the canonical decomposition used throughout SUSAN) places
    β in ``[0, π]``, so a stage tilt of, e.g., −60° is stored as roughly
    ``(±π, 60°, ±π)``.  The selector reads ``proj_eZYZ``, decomposes the
    rotation matrix, and switches to the equivalent representation
    ``(α∓π, −β, γ∓π)`` whenever ``|α| > π/2``.  This yields a signed tilt
    β ∈ [−90°, 90°] with the in-plane component near zero, which is what
    one usually thinks of as "the tilt angle".  The canonicalised angles
    are used only for the filter test; the emitted ``new.proj_eZYZ`` keeps
    SUSAN's original convention.

    .. note::
       For lightweight filtering that does not need on-disk reduction,
       :meth:`susan.data.Particles.Geom.enable_by_tilt_range` zeros the
       relevant ``prj_w`` slots without touching the projection axis or
       the MRC stacks.

    .. rubric:: Attributes

    .. attribute:: tilt_deg_min
       :type: float

       Signed lower bound of the kept range (inclusive), in degrees.

    .. attribute:: tilt_deg_max
       :type: float

       Signed upper bound of the kept range (exclusive), in degrees.

    .. attribute:: new_n_projs
       :type: int

       Per-projection axis length of the reduced
       :class:`Tomograms` / :class:`Particles`.

    .. attribute:: tag
       :type: str

       Filename tag derived from the bounds, e.g. ``'tlt_m60p60'`` for
       ``[-60, 60)`` and ``'tlt_m45d5p45d5'`` for ``[-45.5, 45.5)``.
       ``m`` / ``p`` mark sign; ``d`` replaces the decimal point.
    """

    def __init__(self, tomograms, tilt_deg_min, tilt_deg_max,
                 angle_source='eZYZ_Y'):
        """Build the selection masks from a source tomograms object.

        Parameters
        ----------
        tomograms : Tomograms or str
            Source :class:`Tomograms` instance, or a path to a
            ``.tomostxt`` file.  Only the projection metadata is consulted;
            the source object is not retained.
        tilt_deg_min : float
            Lower bound of the signed tilt range in degrees (inclusive).
        tilt_deg_max : float
            Upper bound of the signed tilt range in degrees (exclusive).
        angle_source : {'eZYZ_Y', 'nominal'}, optional
            * ``'eZYZ_Y'`` (default) — canonicalised Y component of
              ``proj_eZYZ`` (post-alignment, signed in [-90°, 90°]).
            * ``'nominal'`` — ``nominal_tilt_angles`` (stage angle, signed).
        """
        if tilt_deg_min >= tilt_deg_max:
            raise ValueError(
                'tilt_deg_min (%g) must be less than tilt_deg_max (%g).'
                % (float(tilt_deg_min), float(tilt_deg_max)))
        if angle_source not in ('eZYZ_Y', 'nominal'):
            raise ValueError(
                "angle_source must be 'eZYZ_Y' or 'nominal' (got %r)."
                % (angle_source,))

        tomos = _Tomograms(tomograms) if isinstance(tomograms, str) else tomograms
        if not isinstance(tomos, _Tomograms):
            raise TypeError('tomograms must be a Tomograms instance or a filename.')

        self.tilt_deg_min = float(tilt_deg_min)
        self.tilt_deg_max = float(tilt_deg_max)
        self.angle_source = angle_source

        # Snapshot just the shape/identity info we need to validate later inputs.
        self._n_tomos      = int(tomos.n_tomos)
        self._src_n_projs  = int(tomos.n_projs)
        self._tomo_ids     = tomos.tomo_id.copy()
        self._src_num_proj = tomos.num_proj.copy()

        self._kept = []
        for t in range(self._n_tomos):
            P = int(self._src_num_proj[t])
            if P == 0:
                self._kept.append(_np.zeros(0, _np.int32))
                continue
            if angle_source == 'nominal':
                signed = tomos.nominal_tilt_angles[t, :P].astype(_np.float32, copy=True)
            else:
                signed = _np.empty(P, _np.float32)
                for p in range(P):
                    signed[p] = TiltRangeSelector._signed_tilt_deg(tomos.proj_eZYZ[t, p])
            mask  = (signed >= self.tilt_deg_min) & (signed < self.tilt_deg_max)
            mask &= (tomos.proj_wgt[t, :P] > 0)
            self._kept.append(_np.flatnonzero(mask).astype(_np.int32))

        self._kept_count = _np.array([k.size for k in self._kept], _np.uint32)
        self.new_n_projs = int(self._kept_count.max()) if self._kept_count.size else 0

    @staticmethod
    def _signed_tilt_deg(eZYZ_deg):
        """Canonicalised signed tilt β in degrees, with β ∈ [-90, 90] for
        physical tilts.  Picks the ZYZ equivalent branch with in-plane
        ``|α| ≤ π/2`` so the sign of β reflects the stage rotation."""
        R = _np.zeros((3, 3), _np.float32)
        _euZYZ_rotm(R, _np.deg2rad(eZYZ_deg).astype(_np.float32))
        e = _np.zeros(3, _np.float32)
        _rotm_euZYZ(e, R)
        a = float(e[0]); b = float(e[1])
        if abs(a) > _np.pi / 2.0:
            b = -b
        return float(_np.rad2deg(b))

    @property
    def kept_indices(self) -> list[_np.ndarray]:
        """List of length ``n_tomos``; each entry holds the original
        projection indices kept for that tomogram, as ``int32`` arrays."""
        return self._kept

    @property
    def kept_count(self) -> _np.ndarray:
        """``uint32`` array of length ``n_tomos`` with the per-tomogram
        kept-projection count."""
        return self._kept_count

    @property
    def tag(self) -> str:
        return 'tlt_%s%s' % (TiltRangeSelector._fmt_angle(self.tilt_deg_min),
                             TiltRangeSelector._fmt_angle(self.tilt_deg_max))

    @staticmethod
    def _fmt_angle(v):
        s = 'm' if v < 0 else 'p'
        return s + ('%g' % abs(v)).replace('.', 'd')

    def _validate_tomograms(self, tomos):
        if int(tomos.n_tomos) != self._n_tomos:
            raise ValueError(
                'Tomograms has %d entries, selector was built for %d.'
                % (int(tomos.n_tomos), self._n_tomos))
        if not _np.array_equal(tomos.tomo_id, self._tomo_ids):
            raise ValueError(
                'Tomograms tomo_id order does not match the source used to build the selector.')
        for t in range(self._n_tomos):
            if int(tomos.num_proj[t]) < int(self._src_num_proj[t]):
                raise ValueError(
                    'Tomogram %d has num_proj=%d, smaller than source num_proj=%d.'
                    % (t, int(tomos.num_proj[t]), int(self._src_num_proj[t])))

    def _validate_particles(self, ptcls):
        if int(ptcls.n_proj) < int(self._src_n_projs):
            raise ValueError(
                'Particles has n_proj=%d, smaller than source n_projs=%d.'
                % (int(ptcls.n_proj), self._src_n_projs))
        if ptcls.n_ptcl > 0:
            missing = _np.unique(ptcls.tomo_id[~_lookup_cix(self._tomo_ids, ptcls.tomo_id)[1]])
            if missing.size > 0:
                raise ValueError(
                    'Particle tomo_id=%s is not in the source tomograms.'
                    % ','.join(str(m) for m in missing))

    def to_tomograms(self, tomograms, write_stacks=True, in_subfolder=True,
                     filename=None) -> _Tomograms:
        """Emit a reduced :class:`Tomograms` on the kept projections.

        The selector's kept-index list is applied to ``tomograms`` —
        which may differ from the source used to build the selector, as
        long as the projection layout matches (same ``n_tomos``, same
        ``tomo_id`` ordering, and ``num_proj[t]`` at least as large as
        the source's).  This lets the same selector reduce a b1 and a b2
        :class:`Tomograms` consistently.

        Parameters
        ----------
        tomograms : Tomograms or str
            Tomograms to reduce, or a path to a ``.tomostxt`` file.
        write_stacks : bool, optional
            If True (default), each input stack MRC is read and a new MRC
            containing only the kept projections is written.  If False,
            the new ``stack_file`` entries keep the input paths and
            ``stack_size[:, 2]`` is left equal to the kept count (the
            caller is responsible for slicing at read time).
        in_subfolder : bool, optional
            If True (default), each rewritten stack is placed in a
            ``<tag>/`` sibling directory next to the input stack; if
            False, written alongside with the tag inserted into the stem.
            Matches the behaviour of :meth:`Tomograms.bin`.
        filename : str, optional
            If given, also save the reduced :class:`Tomograms` to this
            ``.tomostxt`` file.

        Returns
        -------
        Tomograms
            New :class:`Tomograms` with per-projection arrays of length
            ``new_n_projs``.
        """
        tomos = _Tomograms(tomograms) if isinstance(tomograms, str) else tomograms
        if not isinstance(tomos, _Tomograms):
            raise TypeError('tomograms must be a Tomograms instance or a filename.')
        self._validate_tomograms(tomos)

        tag = self.tag
        new = _Tomograms(n_tomo=self._n_tomos, n_proj=max(self.new_n_projs, 1))

        new.tomo_id[:]    = tomos.tomo_id
        new.tomo_size[:]  = tomos.tomo_size
        new.tomo_position[:] = tomos.tomo_position
        new.pix_size[:]   = tomos.pix_size
        new.voltage[:]    = tomos.voltage
        new.sph_aber[:]   = tomos.sph_aber
        new.amp_cont[:]   = tomos.amp_cont
        new.handedness[:] = tomos.handedness

        for t in range(self._n_tomos):
            k = self._kept[t]
            K = int(k.size)

            new.num_proj[t]      = K
            new.stack_size[t, 0] = tomos.stack_size[t, 0]
            new.stack_size[t, 1] = tomos.stack_size[t, 1]
            new.stack_size[t, 2] = K

            if K > 0:
                new.proj_eZYZ          [t, :K, :] = tomos.proj_eZYZ          [t, k, :]
                new.proj_shift         [t, :K, :] = tomos.proj_shift         [t, k, :]
                new.proj_wgt           [t, :K   ] = tomos.proj_wgt           [t, k   ]
                new.doses              [t, :K   ] = tomos.doses              [t, k   ]
                new.nominal_tilt_angles[t, :K   ] = tomos.nominal_tilt_angles[t, k   ]
                new.def_U              [t, :K   ] = tomos.def_U              [t, k   ]
                new.def_V              [t, :K   ] = tomos.def_V              [t, k   ]
                new.def_ang            [t, :K   ] = tomos.def_ang            [t, k   ]
                new.def_phas           [t, :K   ] = tomos.def_phas           [t, k   ]
                new.def_Bfct           [t, :K   ] = tomos.def_Bfct           [t, k   ]
                new.def_ExFl           [t, :K   ] = tomos.def_ExFl           [t, k   ]
                new.def_mres           [t, :K   ] = tomos.def_mres           [t, k   ]
                new.def_scor           [t, :K   ] = tomos.def_scor           [t, k   ]
                new.ctf_scale_factor   [t, :K   ] = tomos.ctf_scale_factor   [t, k   ]

            in_path = tomos.stack_file[t]
            if write_stacks:
                in_dir   = _os.path.dirname(in_path)
                in_base  = _os.path.basename(in_path)
                stem, ext = _os.path.splitext(in_base)
                out_base = '%s_%s%s' % (stem, tag, ext if ext else '.mrc')
                if in_subfolder:
                    out_dir = _os.path.join(in_dir, tag) if in_dir else tag
                    _os.makedirs(out_dir, exist_ok=True)
                else:
                    out_dir = in_dir
                out_path = _os.path.join(out_dir, out_base) if out_dir else out_base

                stk_in, _ = _mrc.read(in_path)
                if K > 0:
                    stk_out = _np.ascontiguousarray(stk_in[k], dtype=_np.float32)
                else:
                    stk_out = _np.empty((0, int(stk_in.shape[1]), int(stk_in.shape[2])),
                                        dtype=_np.float32)
                _mrc.write(stk_out, out_path, apix=float(new.pix_size[t]))
                new.stack_file[t] = out_path
            else:
                new.stack_file[t] = in_path

        if filename is not None:
            new.save(filename)

        return new

    def to_particles(self, particles, filename=None) -> _Particles:
        """Emit a reduced :class:`Particles` on the kept projections.

        Slices ``prj_eu``, ``prj_t``, ``prj_cc``, ``prj_w`` and the per-
        particle defocus arrays along the per-projection axis, using each
        particle's ``tomo_id`` to look up the matching kept-index list.
        Non-projection fields (positions, alignments, identifiers,
        half-sets) are copied verbatim.

        Defocus values are *not* recomputed.  If you want them rederived
        from the reduced tomograms, call
        :meth:`Particles.update_defocus(new_tomos) <susan.data.Particles.update_defocus>`
        on the returned object.

        Parameters
        ----------
        particles : Particles or str
            Particles to reduce, or a path to a ``.ptclsraw`` file.
        filename : str, optional
            If given, also save the reduced :class:`Particles` to this
            ``.ptclsraw`` file.

        Returns
        -------
        Particles
            New :class:`Particles` with per-projection arrays of length
            ``new_n_projs``.
        """
        ptcls = _Particles(particles) if isinstance(particles, str) else particles
        if not isinstance(ptcls, _Particles):
            raise TypeError('particles must be a Particles instance or a filename.')
        self._validate_particles(ptcls)

        out_n_projs = max(self.new_n_projs, 1)
        out = _Particles(n_ptcl=ptcls.n_ptcl, n_proj=out_n_projs, n_refs=ptcls.n_refs)

        if ptcls.n_ptcl == 0:
            return out

        out.ptcl_id [:]    = ptcls.ptcl_id
        out.tomo_id [:]    = ptcls.tomo_id
        out.tomo_cix[:]    = ptcls.tomo_cix # deprecated: kept as loaded
        out.position[:, :] = ptcls.position
        out.ref_cix [:]    = ptcls.ref_cix
        out.half_id [:]    = ptcls.half_id
        out.extra_1 [:]    = ptcls.extra_1
        out.extra_2 [:]    = ptcls.extra_2

        out.ali_eu[:, :, :] = ptcls.ali_eu
        out.ali_t [:, :, :] = ptcls.ali_t
        out.ali_cc[:, :]    = ptcls.ali_cc
        out.ali_w [:, :]    = ptcls.ali_w

        cix = _lookup_cix(self._tomo_ids, ptcls.tomo_id)[0]
        for m in range(ptcls.n_ptcl):
            t = int(cix[m])
            k = self._kept[t]
            K = int(k.size)
            if K == 0:
                continue
            out.prj_eu  [m, :K, :] = ptcls.prj_eu  [m, k, :]
            out.prj_t   [m, :K, :] = ptcls.prj_t   [m, k, :]
            out.prj_cc  [m, :K   ] = ptcls.prj_cc  [m, k   ]
            out.prj_w   [m, :K   ] = ptcls.prj_w   [m, k   ]
            out.def_U   [m, :K   ] = ptcls.def_U   [m, k   ]
            out.def_V   [m, :K   ] = ptcls.def_V   [m, k   ]
            out.def_ang [m, :K   ] = ptcls.def_ang [m, k   ]
            out.def_phas[m, :K   ] = ptcls.def_phas[m, k   ]
            out.def_Bfct[m, :K   ] = ptcls.def_Bfct[m, k   ]
            out.def_ExFl[m, :K   ] = ptcls.def_ExFl[m, k   ]
            out.def_mres[m, :K   ] = ptcls.def_mres[m, k   ]
            out.def_scor[m, :K   ] = ptcls.def_scor[m, k   ]

        if filename is not None:
            out.save(filename)

        return out
