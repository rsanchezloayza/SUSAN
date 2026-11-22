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
import copy as _copy
import numpy as _np
from scipy.spatial.transform import Rotation as _Rot
from susan.data.Tomograms import Tomograms as _Tomograms
from susan.data.Tomograms import lookup_cix as _lookup_cix
from susan.data.Tomograms import tilt_eZYZ as _tilt_eZYZ
from susan.data.Particles import Particles as _Particles
from susan.utils import euZYZ_to_rotm as _euZYZ_to_rotm
from susan.utils import rotm_to_euZYZ as _rotm_to_euZYZ

_MIN_PTCLS   = 3       # particles (prj_w > 0) a projection needs to be refitted (similarity + 1 redundant)
_MIN_SIGMA   = 10      # without half-sets: particles for a projection's own residual scale, fewer: pooled
_MIN_HALF    = 2       # particles of each half for the half-set difference of a projection
_MIN_OWN     = 10      # particles per half for a tomogram's own tau / noise; fewer: the median of the others
_EM_ITERS    = 500
_IRLS_ITERS  = 10
_CAUCHY_C    = 2.385   # Cauchy tuning constant, in units of the robust sd
_CHI2_MEDIAN = 1.1774  # median of a chi distribution with 2 dof (2D residual norm)
_CHI3_MEDIAN = 1.5382  # median of a chi distribution with 3 dof (rotation angle)
_TOL_DEG     = 1e-4    # a tomostxt matches the source geometry: rotation (deg),
_TOL_A       = 1e-3    # shift and tomogram position (A)
_TOL_SCALE   = 1e-6    # and magnification


class TomogramGeometryUpdater:
    """Move the per-projection geometry of a 2D refinement into the tomograms.

    After the 2D solver, the part of the per-projection corrections that is common to the particles of a projection
    (shift, in-plane rotation, magnification, tilt) sits in every particle's ``prj_t`` / ``prj_eu``. The updater
    fits it per tomogram and projection and moves it into ``proj_eZYZ`` / ``proj_shift`` / ``proj_scale``:

    * tilt and tilt-axis tilt from the orientations: ``Q`` = robust mean of the particles' ``R(prj_eu)``;
    * in-plane rotation, magnification and shift from the projected positions: a robust similarity fit of the
      particles' crop positions on ``(Q R x)_xy``; the new rotation is ``Rz(phi) Q R``. The in-plane angle and the
      log magnification change of each projection are shrunk by their uncertainty (few or clustered particles)
      against their spread over the tomogram (``tau``, empirical Bayes), and the shift is refitted with them. The
      noise comes from the difference of the fits of the two half-sets (independent only with independent half-set
      refinement); a projection without 2 particles in each half keeps its in-plane angle and magnification only
      when other projections of the tomogram give the noise level, and none moves without any. A tomogram with
      fewer than 10 particles in a half takes ``tau`` and the noise level from the median of the other tomograms. Without half-sets
      it comes from the fit residuals (an underestimate for particles of a model-based solver such as Solver2D);
    * a shift or rotation common to all the projections of a tomogram is a motion of the tomogram, not a geometry
      error: it is fitted, kept out of the new geometry (the tomogram frame does not move) and left in the particles.

    Projections with fewer than 3 particles (``prj_w > 0``) keep their geometry; defocus is not changed. The fit
    uses all the particles (both half-sets: the tomograms are shared); each half keeps its own remainder.

    :meth:`to_tomograms` writes the new geometry into any tomostxt of the same tomograms (one call per binning: the
    geometry is in Angstroms and degrees), :meth:`to_particles` converts any particle set built on the source
    geometry so that crop positions and orientations are unchanged: ``prj_t += [s (R x)_xy + t] - [s' (R' x)_xy +
    t']`` and ``R(prj_eu) <- R(prj_eu) R R'^T``, with ``x`` the particle in its own class (``ref_cix``).

    Example::

        upd = TomogramGeometryUpdater('tomo_b1.tomostxt', 'ptcls.ptclsraw')
        print(upd.summary())
        upd.to_tomograms('tomo_b1.tomostxt', 'tomo_b1_upd.tomostxt')
        upd.to_tomograms('tomo_b8.tomostxt', 'tomo_b8_upd.tomostxt')
        upd.to_particles('ptcls.ptclsraw', 'ptcls_upd.ptclsraw')

    .. attribute:: report
       :type: list[dict]

       One entry per tomogram: ``tomo_id``; per projection ``n`` (particles used), ``fitted``, ``rot`` (deg, the
       geometry change as a rotation vector in the detector frame: x tilt-axis tilt, y tilt, z in-plane), ``scale``
       (new / old), ``shift`` (A, new - old), ``rms_before`` / ``rms_after`` (A, rms of the fitted particles'
       ``prj_t``) and ``stretch`` (anisotropy of a full affine fit, reported only), ``shrink`` (fraction of the magnification and
       in-plane change kept); ``tau`` (their spread over the tomogram: log scale, deg); ``gauge_shift`` (A) and
       ``gauge_rot`` (deg, tomogram frame), the common motion left in the particles.
    """

    def __init__(self, tomograms, particles):
        """Fit the new geometry.

        Parameters
        ----------
        tomograms : Tomograms or str
            The geometry the particles were refined on (object or ``.tomostxt``).
        particles : Particles or str
            The refined particles (object or ``.ptclsraw``).
        """
        tomos = _load_tomograms(tomograms)
        ptcls = _load_particles(particles)
        self._ids  = tomos.tomo_id.copy()
        self._K    = tomos.num_proj.astype(int)
        self._pos  = tomos.tomo_position.astype(_np.float64)
        self._R0   = _euZYZ_to_rotm(tomos.proj_eZYZ.astype(_np.float64), degrees=True)
        self._s0   = tomos.proj_scale.astype(_np.float64)
        self._t0   = tomos.proj_shift.astype(_np.float64)
        self._R    = self._R0.copy()
        self._s    = self._s0.copy()
        self._t    = self._t0.copy()
        self.report = []

        cix = tomos.get_cix(ptcls.tomo_id)
        x = ptcls.pos(None).astype(_np.float64) - self._pos[cix]
        hid = ptcls.half_id.ravel()
        halves = _np.isin(hid, 1).any() and _np.isin(hid, 2).any()
        n_own = [min(((cix == c) & (hid == h)).sum() for h in (1, 2)) if halves else (cix == c).sum() // 2
                 for c in range(tomos.n_tomos)]
        self.report = [None] * tomos.n_tomos
        pooled = None
        for big in (True, False):
            for c in range(tomos.n_tomos):
                if (n_own[c] >= _MIN_OWN) != big:
                    continue
                sel = cix == c
                self.report[c] = self._fit_tomogram(c, x[sel], ptcls.prj_eu[sel], ptcls.prj_t[sel], ptcls.prj_w[sel],
                                                    ptcls.half_id[sel], pooled)
            if big:
                own = [r for r in self.report if r is not None and _np.isfinite(r['sig2']).all() and r['fitted'].any()]
                if own:
                    pooled = (_np.median([r['tau2'] for r in own], 0), _np.median([r['sig2'] for r in own], 0))

    # ------------------------------------------------------------------ fit
    def _fit_tomogram(self, c, x, prj_eu, prj_t, prj_w, half, pooled=None):
        """pooled: (tau^2, noise sigma^2) of the in-plane / log-scale change, used instead of the tomogram's own"""
        K = int(self._K[c])
        rep = dict(tomo_id=int(self._ids[c]), n=_np.zeros(K, int), fitted=_np.zeros(K, bool), shrink=_np.zeros((K, 2)),
                   tau=_np.zeros(2), tau2=_np.zeros(2), sig2=_np.full(2, _np.inf), pooled=pooled is not None,
                   rot=_np.zeros((K, 3)), scale=_np.ones(K), shift=_np.zeros((K, 2)), rms_before=_np.zeros(K),
                   rms_after=_np.zeros(K), stretch=_np.zeros(K), gauge_shift=_np.zeros(3), gauge_rot=_np.zeros(3))
        crop, fit = {}, {}
        for k in range(K):
            sel = prj_w[:, k] > 0
            rep['n'][k] = int(sel.sum())
            if rep['n'][k] < _MIN_PTCLS:
                continue
            R0, xs = self._R0[c, k], x[sel]
            p = self._s0[c, k] * (xs @ R0.T)[:, :2] + self._t0[c, k] + prj_t[sel, k]
            QR = _robust_mean_rotation(_euZYZ_to_rotm(prj_eu[sel, k].astype(_np.float64))) @ R0
            q = (xs @ QR.T)[:, :2]
            fit[k] = (QR, q, p) + _fit_similarity(q, p)
            sv = _np.linalg.svd(_fit_affine(q, p, fit[k][5]), compute_uv=False)
            rep['fitted'][k], rep['stretch'][k] = True, sv[0] / sv[1] - 1
            rep['rms_before'][k] = _np.sqrt((prj_t[sel, k].astype(_np.float64) ** 2).sum(1).mean())
            crop[k] = (sel, p)
        halves = _np.isin(half, 1).any() and _np.isin(half, 2).any()
        if fit:
            # in-plane angle and log magnification change of each projection, shrunk by its uncertainty against their
            # spread over the tomogram (tau^2, empirical Bayes); the shift is refitted with them
            ks = sorted(fit)
            dl, g = _np.zeros((len(ks), 2)), _np.zeros((len(ks), 2))
            for i, k in enumerate(ks):
                dl[i], g[i] = _change(fit[k][3], fit[k][7], self._s0[c, k])
            if pooled is not None:
                sig2 = pooled[1]
                var = sig2[None] * g
            elif halves:
                # noise from the two half-sets (independent solutions): (d1 - d2)^2 ~ sigma^2 (g1 + g2)
                ratio = []
                for k in ks:
                    sel, (q, p) = crop[k][0], fit[k][1:3]
                    hk = half[sel]
                    if min((hk == 1).sum(), (hk == 2).sum()) < _MIN_HALF:
                        continue
                    d_, g_ = zip(*(_change(*_fit_similarity(q[hk == h], p[hk == h])[0:5:4], self._s0[c, k]) for h in (1, 2)))
                    ratio.append((d_[0] - d_[1]) ** 2 / (g_[0] + g_[1]))
                sig2 = _np.mean(ratio, 0) if ratio else _np.full(2, _np.inf)  # mean, like tau^2 (heavy tails)
                var = sig2[None] * g
            else:
                dof = _np.array([2 * len(fit[k][1]) - 4 for k in ks], float)
                rss = _np.array([fit[k][6] for k in ks])
                s2 = _np.where(rep['n'][ks] >= _MIN_SIGMA, rss / _np.maximum(dof, 1.0), rss.sum() / max(dof.sum(), 1.0))
                var = s2[:, None] * g
                sig2 = _np.full(2, rss.sum() / max(dof.sum(), 1.0))
            ok = _np.isfinite(var).all(1)
            if pooled is not None:
                tau2 = pooled[0]
            else:
                tau2 = _em_tau2(dl[ok], var[ok]) if ok.any() else _np.zeros(2)
            rep['tau2'], rep['sig2'] = tau2, sig2
            f_ = _np.where(ok[:, None], tau2 / (tau2 + _np.where(ok[:, None], var, 1.0)), 0.0)
            for i, k in enumerate(ks):
                QR, q, p, _, _, w, _ = fit[k][:7]
                ls, phi = f_[i] * dl[i]
                a = self._s0[c, k] * _np.exp(ls) * complex(_np.cos(phi), _np.sin(phi))
                aq = _np.column_stack([a.real * q[:, 0] - a.imag * q[:, 1], a.imag * q[:, 0] + a.real * q[:, 1]])
                t = (w[:, None] * (p - aq)).sum(0) / w.sum()
                Rz = _np.array([[_np.cos(phi), -_np.sin(phi), 0], [_np.sin(phi), _np.cos(phi), 0], [0, 0, 1]])
                self._R[c, k], self._s[c, k], self._t[c, k] = Rz @ QR, abs(a), t
                rep['shrink'][k] = f_[i]
            rep['tau'] = _np.sqrt(tau2) * [1.0, _np.degrees(1.0)]
        f = _np.flatnonzero(rep['fitted'])
        if f.size:
            # common rotation: R' R^T ~ R G R^T for every projection, rotation vector c_k ~ R_k g
            ck = _Rot.from_matrix(self._R[c, f] @ _np.swapaxes(self._R0[c, f], 1, 2)).as_rotvec()
            g = _np.einsum('kji,kj->i', self._R0[c, f], ck) / f.size
            self._R[c, f] = self._R[c, f] @ _Rot.from_rotvec(g).as_matrix().T
            # common shift: t' - t ~ s (R d)_xy
            A = (self._s0[c, f, None, None] * self._R0[c, f, :2, :]).reshape(-1, 3)
            d = _np.linalg.lstsq(A, (self._t[c, f] - self._t0[c, f]).reshape(-1), rcond=None)[0]
            self._t[c, f] -= self._s[c, f, None] * (self._R[c, f, :2, :] @ d)
            rep['gauge_rot'], rep['gauge_shift'] = _np.degrees(g), d
            rep['rot'][f] = _np.degrees(_Rot.from_matrix(self._R[c, f] @ _np.swapaxes(self._R0[c, f], 1, 2)).as_rotvec())
            rep['scale'][f] = self._s[c, f] / self._s0[c, f]
            rep['shift'][f] = self._t[c, f] - self._t0[c, f]
            for k, (sel, p) in crop.items():
                r = p - (self._s[c, k] * (x[sel] @ self._R[c, k].T)[:, :2] + self._t[c, k])
                rep['rms_after'][k] = _np.sqrt((r ** 2).sum(1).mean())
        return rep

    # --------------------------------------------------------------- output
    def summary(self) -> str:
        """One line per tomogram: projections refitted, rms of the tilt / in-plane / tilt-axis changes, magnification
        range, rms shift change, rms of ``prj_t`` before and after, largest affine stretch and the common motion."""
        out = []
        for r in self.report:
            f = r['fitted']
            if not f.any():
                out.append('tomo %d: no projection with %d particles' % (r['tomo_id'], _MIN_PTCLS))
                continue
            rms = lambda v: float(_np.sqrt(_np.mean(v ** 2)))
            out.append('tomo %d: %d / %d projections | tilt %.3f, in-plane %.3f, axis %.3f deg rms | scale %.4f..%.4f '
                       '| kept %.2f / %.2f (tau %.4f, %.3f deg) | shift %.2f A rms | prj_t %.2f -> %.2f A rms | stretch max %.2f%% '
                       '| common shift %s A, rotation %s deg' % (r['tomo_id'], f.sum(), f.size, rms(r['rot'][f, 1]),
                       rms(r['rot'][f, 2]), rms(r['rot'][f, 0]), r['scale'][f].min(), r['scale'][f].max(),
                       _np.median(r['shrink'][f, 0]), _np.median(r['shrink'][f, 1]), r['tau'][0], r['tau'][1], rms(r['shift'][f]),
                       rms(r['rms_before'][f]), rms(r['rms_after'][f]), 100 * r['stretch'][f].max(),
                       _np.round(r['gauge_shift'], 2), _np.round(r['gauge_rot'], 3)))
        return '\n'.join(out)

    def to_tomograms(self, tomograms, filename=None) -> _Tomograms:
        """The new geometry in a tomostxt of the same tomograms (any binning).

        Tomograms are matched by ``tomo_id``; ``proj_eZYZ``, ``proj_shift`` and ``proj_scale`` are replaced, every
        other field (stacks, pixel size, sizes, defocus, ...) is kept.

        Parameters
        ----------
        tomograms : Tomograms or str
            Object (not modified) or ``.tomostxt``.
        filename : str, optional
            Also save the result to this ``.tomostxt``.

        Returns
        -------
        Tomograms

        Raises
        ------
        ValueError
            If a tomogram is not in the source, its number of projections differs, or its geometry is not the
            one the fit started from (a stale file would not match the particles).
        """
        out = _copy.deepcopy(_load_tomograms(tomograms))
        for j in range(out.n_tomos):
            c, K = self._source_index(out.tomo_id[j]), int(out.num_proj[j])
            if K != self._K[c]:
                raise ValueError('tomo_id %d: %d projections, the source has %d.' % (out.tomo_id[j], K, self._K[c]))
            R = _euZYZ_to_rotm(out.proj_eZYZ[j, :K].astype(_np.float64), degrees=True)
            dev = (_geodesic_deg(R, self._R0[c, :K]).max(),
                   _np.abs(out.proj_shift[j, :K] - self._t0[c, :K]).max(),
                   _np.abs(out.tomo_position[j] - self._pos[c]).max(),
                   _np.abs(out.proj_scale[j, :K] - self._s0[c, :K]).max())
            if dev[0] > _TOL_DEG or dev[1] > _TOL_A or dev[2] > _TOL_A or dev[3] > _TOL_SCALE:
                raise ValueError('tomo_id %d: the geometry is not the one of the fit (rotation %.2e deg, shift %.2e A, '
                                 'position %.2e A, scale %.2e).' % ((out.tomo_id[j],) + dev))
            out.proj_eZYZ [j, :K] = _tilt_eZYZ(_rotm_to_euZYZ(self._R[c, :K], degrees=True))
            out.proj_shift[j, :K] = self._t[c, :K]
            out.proj_scale[j, :K] = self._s[c, :K]
        if filename is not None:
            out.save(filename)
        return out

    def to_particles(self, particles, filename=None) -> _Particles:
        """Particles converted to the new geometry (crop positions and orientations unchanged).

        Any particle set built on the source geometry (earlier iterations, the reconstruction particles,
        template-matching sets). Exact for each particle's own class (``ref_cix``); for its other classes the
        positions differ by ``(s R - s' R') (ali_t[r] - ali_t[ref_cix])``.

        Parameters
        ----------
        particles : Particles or str
            Object (not modified) or ``.ptclsraw``.
        filename : str, optional
            Also save the result to this ``.ptclsraw``.

        Returns
        -------
        Particles
        """
        P = _load_particles(particles).copy()
        cix, valid = _lookup_cix(self._ids, P.tomo_id)
        if not bool(_np.all(valid)):
            raise ValueError('tomo_id %s is not in the source tomograms.' % ','.join(str(t) for t in _np.unique(P.tomo_id[~valid])))
        cix = _np.asarray(cix, int)
        x = P.pos(None).astype(_np.float64) - self._pos[cix]
        for c in _np.unique(cix):
            sel, K = cix == c, int(self._K[c])
            if K > P.n_proj:
                raise ValueError('Particles have %d projections, tomo_id %d has %d.' % (P.n_proj, self._ids[c], K))
            old = self._s0[c, None, :K, None] * _np.einsum('kij,nj->nki', self._R0[c, :K, :2], x[sel]) + self._t0[c, :K]
            new = self._s [c, None, :K, None] * _np.einsum('kij,nj->nki', self._R [c, :K, :2], x[sel]) + self._t [c, :K]
            P.prj_t[sel, :K] = (P.prj_t[sel, :K].astype(_np.float64) + old - new).astype(_np.float32)
            D = self._R0[c, :K] @ _np.swapaxes(self._R[c, :K], 1, 2)
            R2 = _euZYZ_to_rotm(P.prj_eu[sel, :K].astype(_np.float64)) @ D
            P.prj_eu[sel, :K] = _rotm_to_euZYZ(R2).astype(_np.float32)
        if filename is not None:
            P.save(filename)
        return P

    def _source_index(self, tomo_id):
        cix, valid = _lookup_cix(self._ids, tomo_id)
        if not bool(_np.all(valid)):
            raise ValueError('tomo_id %d is not in the source tomograms.' % int(tomo_id))
        return int(cix)


def _load_tomograms(t):
    t = _Tomograms(t) if isinstance(t, str) else t
    if not isinstance(t, _Tomograms):
        raise TypeError('tomograms must be a Tomograms instance or a filename.')
    return t


def _load_particles(p):
    p = _Particles(p) if isinstance(p, str) else p
    if not isinstance(p, _Particles):
        raise TypeError('particles must be a Particles instance or a filename.')
    return p


def _geodesic_deg(A, B):
    """Angle (deg) between rotation stacks, stable near 0: 2 asin(|A - B|_F / sqrt 8)."""
    return _np.degrees(2 * _np.arcsin(_np.minimum(_np.linalg.norm(A - B, axis=(-2, -1)) / _np.sqrt(8), 1.0)))


def _cauchy(r, chi_median):
    s = _np.median(r) / chi_median
    return 1.0 / (1.0 + (r / (_CAUCHY_C * s)) ** 2) if s > 0 else _np.ones_like(r)


def _robust_mean_rotation(R):
    """Weighted chordal mean of a rotation stack, Cauchy weights on the angle to the mean."""
    w = _np.ones(len(R))
    for _ in range(_IRLS_ITERS):
        Q = _Rot.from_matrix(R).mean(weights=w).as_matrix()
        w = _cauchy(_Rot.from_matrix(Q.T @ R).magnitude(), _CHI3_MEDIAN)
    return Q


def _fit_similarity(q, p):
    """Robust p ~ a q + t, ``a`` complex (magnification and in-plane rotation). Returns (a, t, weights, weighted
    residual sum of squares, (X^T W X)^-1 of (Re a, Im a, t))."""
    n = len(q)
    X = _np.zeros((2 * n, 4))
    X[0::2] = _np.column_stack([q[:, 0], -q[:, 1], _np.ones(n), _np.zeros(n)])
    X[1::2] = _np.column_stack([q[:, 1],  q[:, 0], _np.zeros(n), _np.ones(n)])
    y = p.reshape(-1)
    w = _np.ones(n)
    for _ in range(_IRLS_ITERS):
        sw = _np.repeat(_np.sqrt(w), 2)
        b = _np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        w = _cauchy(_np.linalg.norm((y - X @ b).reshape(n, 2), axis=1), _CHI2_MEDIAN)
    sw = _np.repeat(_np.sqrt(w), 2)
    b = _np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    r2 = ((y - X @ b) ** 2 * sw ** 2).sum()
    return complex(b[0], b[1]), b[2:], w, float(r2), _np.linalg.pinv((X * sw[:, None]).T @ (X * sw[:, None]))


def _change(a, Cinv, s0):
    """log magnification and in-plane angle change of a similarity ``a`` from ``s0``, and their variance factors
    (variance / residual variance) from (X^T W X)^-1 of the fit"""
    u = _np.array([a.real, a.imag]) / abs(a)
    v = _np.array([-u[1], u[0]])
    C = Cinv[:2, :2]
    return _np.array([_np.log(abs(a) / s0), _np.angle(a)]), _np.array([u @ C @ u, v @ C @ v]) / abs(a) ** 2


def _em_tau2(d, v):
    """Per column, the variance tau^2 of the true values behind estimates d with sampling variances v (maximum
    marginal likelihood, EM)."""
    tau2 = _np.maximum((d ** 2).mean(0), 1e-12)
    for _ in range(_EM_ITERS):
        f = tau2 / (tau2 + v)
        new = ((f * d) ** 2 + f * v).mean(0)
        if _np.all(_np.abs(new - tau2) <= 1e-6 * tau2):
            tau2 = new
            break
        tau2 = new
    return tau2


def _fit_affine(q, p, w):
    """Weighted p ~ A q + t; returns the 2x2 A."""
    X = _np.column_stack([q, _np.ones(len(q))]) * _np.sqrt(w)[:, None]
    return _np.linalg.lstsq(X, p * _np.sqrt(w)[:, None], rcond=None)[0][:2].T
