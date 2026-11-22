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

import math as _math
import os as _os

import numpy as _np
import scipy.sparse as _sp

import susan.data as _ssa_data
import susan.io.ali_stats as _ali_stats
import susan.utils as _ssa_utils

__all__ = ['Solver2D']

_CHI2_2_MEDIAN = 1.3863
_F = {f: i for i, f in enumerate(_ali_stats.FIELDS)}
_TINY = 1e-12
_MIN_OBS = 2           # usable particle peaks of a projection (shift and in-plane rotation determined)
_MIN_KAPPA = 10        # particle peaks for a projection's own residual scale; fewer: the tomogram's pooled scale
_MIN_OWN = 10          # particles of a tomogram (half) to estimate its own priors; fewer: the median of the others
_KAPPA_RANGE = 10      # a projection's residual scale is at least the tomogram's median / range
_SHIFT_PREC = 1e-8     # shift prior precision, relative to the data scale (projections without data in a CV fold)
_ROT_PREC = (1.0, 1e12)  # rotation prior precision range (1/rad^2): sd between 1e-6 rad and 1 rad (the start)
_OBS_PER_DOF = 4       # observations per field parameter of the finest level
_MAX_LEVEL = 4
_LAM_RANGE = 1e4       # strengths within [1/range, range] of the data scale
_LAM_RTOL = 1e-2
_LAM_OUTER = 12
_LAM_DAMP = 0.5
_BASE_OUTER = 3
_CV_OUTER = 8
_CV_SCALES = (1.0, 1e-3)


class Solver2D:
    """2D refinement from the statistics of a 2D alignment (``output_type='alignment_stats'``).

    Per tomogram (and per half-set when the groups of the statistics are per half), every parameter estimated from
    the data.  Model of the peak of particle j on projection k (detector px, from the input pose):
    t_k + w_k [-y, x] + [c_yk z, -c_xk z] + f_k(x, y) + P_k delta_j, with the shift t_k, the rotations about the beam
    w_k and about the detector x and y c_k (z the depth from the median depth), a multi-scale field f_k (hierarchical
    bilinear grids, without the shift and the rotation), and one 3D offset delta_j per particle (prior covariance from
    the even/odd projection split).  The group peaks (2x2 to 5x5) enter as observations of the same model at their
    centroids with the precision of their half-split reproducibility; the particles with their curvature on the scale of
    their residuals (their own with at least 10 particle peaks, else the tomogram's; at least a tenth of the
    tomogram's median, so that no projection takes over the 3D offsets).  Robust (Cauchy) weights; the
    projections are independent given the offsets.  A projection needs 2 usable particle peaks.  A tomogram (half)
    with fewer than 10 particles does not estimate its rotation prior, kappa and offset prior: it takes the median of
    those of the tomograms (of the same half) that do.

    1. base fit: one strength per level of the field and one zero-mean prior per rotation axis, fixed points (pooled
       over the projections of the tomogram); the rotations of projections that the data determine poorly (few
       particles, little spread in depth or in the plane) shrink towards none; the finest level from the observations
       per parameter;
    2. per projection, by cross-validation on held-out particle and group peaks: no field, or the field up to each
       level, at the strength of step 1 or 1000 times weaker, with the rigid part free or kept from the fit without
       field;
    3. the rotations of each projection are those of the base fit; its shift those of the model of step 2.

    Usage: align the particles in 2D with ``output_type='alignment_stats'`` (offset step 0.5 px, no angular search;
    with independent half-sets the groups are per half), then :meth:`solve` gives the particles of the next alignment;
    :class:`SolverHighRes` makes those of the reference reconstruction from the same statistics.

    Attributes
    ----------
    n_fold : int
        Cross-validation folds.  Default ``5``.
    seed : int
        Seed of the folds.  Default ``0``.
    edge_fraction : float
        Summary: suggest a wider offset range when more than this fraction of the peaks of a projection sit at the edge
        of the offset search.  Default ``0.1``.
    max_offset_step : float
        Summary: suggest a smaller offset step above this many pixels (0.5 px gave a consistent gain on
        EMPIAR-10064).  Default ``0.5``.
    verbosity : int
        ``0`` silent, ``1`` a summary (corrections, convergence, suggested search changes), ``2`` also one line per
        tomogram and half.  Default ``1``.
    """

    def __init__(self):
        self.n_fold          = 5
        self.seed            = 0
        self.edge_fraction   = 0.1
        self.max_offset_step = 0.5
        self.verbosity       = 1
        self.solved          = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, ptcls_in, stats, tomos, aligner=None):
        """Particles for the next alignment.

        Parameters
        ----------
        ptcls_in : str or :class:`susan.data.Particles`
            The *input* particles of the alignment that wrote ``stats`` (its offsets are relative to their prj_t).
        stats : str or dict
            The ``.stats.npz`` (or the aligner output ``.ptclsraw``), or :func:`susan.io.ali_stats.read` of it.
        tomos : str or :class:`susan.data.Tomograms`
        aligner : :class:`susan.modules.Aligner`, optional
            The aligner of the iteration: with it the summary suggests changes of its offset search.

        Returns
        -------
        :class:`susan.data.Particles`
            prj_t = input prj_t + solved shift; prj_eu = the solved rotation applied to the input prj_eu; prj_w
            unchanged (an unsolved projection keeps its pose).  :attr:`solved` (n_ptcl, K) marks the solved
            projections.  The uncertainty of what was applied is added to the statistics (the file when ``stats`` is a
            path, else the dict): ``projections/post_cov`` (n_ptcl, K, 3) float32, the posterior covariance
            [xx, xy, yy] (px^2) of the shift, the projection of the covariance of the 3D offset of the particle, NaN
            where unsolved; ``projections/ang_var`` (n_ptcl, K, 3) float32, the variance (deg^2) of the applied
            rotation about x, y and the beam, NaN where no rotation was applied.
        """
        P = _ssa_data.Particles(ptcls_in) if isinstance(ptcls_in, str) else ptcls_in.copy()
        S = _ali_stats.read(stats) if isinstance(stats, str) else stats
        T = _ssa_data.Tomograms(tomos) if isinstance(tomos, str) else tomos
        st = S['projections/stats']
        if st.shape[0] != P.n_ptcl:
            raise ValueError('The statistics have %d particles, the particles %d: give the input of the alignment.' % (
                st.shape[0], P.n_ptcl))
        halves = int(S.get('halves', 1))
        gsizes = sorted(int(k[len('grouped'):-len('/stats')]) for k in S if k.startswith('grouped') and k.endswith('/stats'))
        stomo = [int(t) for t in S['tomo_id']]
        tid = P.tomo_id.flatten().astype(int)
        hid = P.half_id.flatten().astype(int)
        pos = P.pos()
        prj_t_in, prj_eu_in = P.prj_t.copy(), P.prj_eu.copy()
        self.solved = _np.zeros(P.prj_w.shape, bool)
        post_cov = _np.full(P.prj_w.shape + (3,), _np.nan, _np.float32)
        ang_var = _np.full(P.prj_w.shape + (3,), _np.nan, _np.float32)
        ang_ss, ang_n = 0.0, 0
        ang_noise = []
        span = min(aligner.offset.span[:2]) if aligner is not None else None
        edge = []
        jobs = []
        for t in _np.unique(tid):
            cix, ti = int(T.get_cix(t)), stomo.index(int(t))
            K = int(T.num_proj[cix])
            apix = float(T.pix_size[cix])
            st_ = _np.where(tid == t)[0]
            xyz_t = self._projected_positions(T, cix, pos[st_], K)
            Pk = _np.stack([_ssa_utils.euZYZ_to_rotm(T.proj_eZYZ[cix, k], degrees=True)[:2].astype(_np.float64) *
                            float(T.proj_scale[cix, k]) for k in range(K)])
            for h in ((1, 2) if halves == 2 else (0,)):
                m_ = (hid[st_] == h) if h else _np.ones(len(st_), bool)
                sel = st_[m_]
                if len(sel) == 0:
                    continue
                xyz = xyz_t[m_]
                grp = {}
                for gs in gsizes:
                    G, o = gs * gs, (h - 1) * gs * gs if h else 0
                    n_g = S['grouped%d/stats' % gs].shape[1] // (2 if halves == 2 else 1)
                    if n_g != G:
                        continue
                    grp[gs] = (S['grouped%d/stats' % gs][ti][o:o + G, :K], S['grouped%d/label' % gs][sel] - o,
                               S['grouped%d/half' % gs][ti][o:o + G, :K])
                valid_k = P.prj_w[sel][:, :K].max(0) > 0
                edge += [(int(t), int(k)) for k in self._at_edge(st[sel][:, :K], valid_k, span, float(S.get('off_step', 0.5)))]
                jobs.append((t, h, K, apix, sel, xyz, Pk, grp, valid_k))
        # tomograms with enough particles estimate their own priors; the others get the median of those (per half)
        own, pooled = {}, {}
        for big in (True, False):
            for t, h, K, apix, sel, xyz, Pk, grp, valid_k in jobs:
                if (len(sel) >= _MIN_OWN) != big:
                    continue
                prior = None if big else pooled.get(h)
                if self.verbosity > 1:
                    print('    Tomogram %d%s%s:' % (t, (', half %d' % h) if h else '', ' (pooled priors)' if prior else ''), end='', flush=True)
                r = self._solve_tomo(st[sel][:, :K], xyz[..., :2], xyz[..., 2], Pk, grp, apix, valid_k, prior)
                ok = _np.isfinite(r['shift'][..., 0])
                for k in range(K):
                    i = sel[ok[:, k]]
                    if len(i) == 0:
                        continue
                    P.prj_t[i, k] = prj_t_in[i, k] + r['shift'][ok[:, k], k] * apix
                    post_cov[i, k] = r['cov'][ok[:, k], k]
                    self.solved[i, k] = True
                rv = _np.broadcast_to(r['rv'][None], (len(sel), K, 3))
                var = _np.broadcast_to(r['var'][None], (len(sel), K, 3))
                vk = r['var'][_np.isfinite(r['rv']).all(-1)]
                ang_noise.append(_math.sqrt(_np.median(vk[vk > 0])) if (vk > 0).any() else _np.nan)
                m = _np.isfinite(rv).all(-1) & (P.prj_w[sel][:, :K] > 0)
                ii, kk = sel[_np.where(m)[0]], _np.where(m)[1]
                D = _rotvec_rotm(rv[m])
                P.prj_eu[ii, kk] = _ssa_utils.rotm_to_euZYZ(D @ _ssa_utils.euZYZ_to_rotm(prj_eu_in[ii, kk]))
                ang_var[ii, kk] = var[m]
                a = _np.linalg.norm(rv[m], axis=-1)
                ang_ss += float(_np.sum(a[a > 0] ** 2))
                ang_n += int((a > 0).sum())
                if big and 'prior' in r:
                    own.setdefault(h, []).append(r['prior'])
            if big:
                pooled = {h: _pool_priors(v) for h, v in own.items()}
        self._append_variances(stats, S, post_cov, ang_var)
        if self.verbosity:
            self._summary(P, prj_t_in, T, S, post_cov, (ang_ss, ang_n), ang_noise, sorted(set(edge)), aligner)
        return P

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _at_edge(self, stats, valid_k, span, step):
        """projections with more than edge_fraction of their peaks at the edge of the offset search (half-width span px,
        step px; none without the span)"""
        p = _np.abs(stats[..., [_F['px'], _F['py']]].astype(_np.float64))
        fin = _np.isfinite(p[..., 0])
        if span is None or not fin.any():
            return []
        at = fin & (p >= span - step / 2).any(-1)
        frac = at.sum(0) / _np.maximum(fin.sum(0), 1)
        return list(_np.where(valid_k & (frac > self.edge_fraction))[0])

    def _summary(self, P, prj_t_in, T, S, post_cov, ang, ang_noise, edge, aligner):
        """Corrections applied to the dataset; [converged] when none is beyond its noise (rms angle change below the
        median sd of the per-tilt estimates, rms shift change below the median posterior sd); search changes
        suggested by the data.  ang: sum of squares and count of the applied rotations (deg); edge: (tomogram,
        projection) with peaks at the edge of the offset search."""
        on = self.solved
        tid = P.tomo_id.flatten().astype(int)
        apix = _np.array([float(T.pix_size[int(T.get_cix(t))]) for t in tid])[:, None] * _np.ones(on.shape)
        d = _np.linalg.norm(P.prj_t - prj_t_in, axis=-1)
        d_rms = _math.sqrt(_np.mean(d[on] ** 2)) if on.any() else 0.0
        d_px = _math.sqrt(_np.mean((d[on] / apix[on]) ** 2)) if on.any() else 0.0
        noise = _np.median(_np.sqrt((post_cov[on][:, 0] + post_cov[on][:, 2]) / 2) * apix[on]) if on.any() else 0.0
        a_on = ang[1] > 0
        a_rms = _math.sqrt(ang[0] / ang[1]) if a_on else 0.0
        tags = []
        if on.any() and (not a_on or a_rms <= _np.nanmedian(ang_noise)) and d_rms <= noise:
            tags.append('[converged]')
        sug = []
        if edge:
            sug.append('Increase offset range (peaks at the search edge on tomogram:projection %s)' % ' '.join(
                '%d:%d' % e for e in edge))
        elif aligner is not None and on.any():
            r = min(aligner.offset.span[:2])
            off = (P.prj_t - prj_t_in) / apix[..., None]
            for k in range(on.shape[1]):
                if on[:, k].any() and _np.median(_np.linalg.norm(off[on[:, k], k], axis=-1)) > r / 2:
                    sug.append('Increase offset range')
                    break
        if float(S.get('off_step', 0)) > self.max_offset_step:
            sug.append('Reduce offset step')
        if sug:
            tags.append('[Suggestion: %s.]' % ', '.join(sug))
        print('    Solving 2D alignments.%s' % ''.join(' ' + t for t in tags))
        print('    Corrections: shifts %.2f Å rms (%.2f px), angles %.3f° rms. Using %.0f%% of projections.' % (
            d_rms, d_px, a_rms, 100 * on.sum() / max((P.prj_w > 0).sum(), 1)), flush=True)

    @staticmethod
    def _append_variances(stats, S, post_cov, ang_var):
        """Add ``projections/post_cov`` and ``projections/ang_var`` to the dict S, and to the file when stats is a path."""
        S['projections/post_cov'] = post_cov
        S['projections/ang_var'] = ang_var
        if isinstance(stats, str):
            out = _ali_stats.path(stats) if stats.endswith('.ptclsraw') else stats
            _np.savez(out + '.tmp.npz', **S)
            _os.replace(out + '.tmp.npz', out)

    @staticmethod
    def _projected_positions(T, cix, pos, K):
        """(n, K, 3) projected positions of the particles pos (n, 3) of tomogram cix: x, y px from the frame centre
        (crop geometry, without prj_t) and the depth px."""
        p = T.to_tomo(cix, pos).astype(_np.float64)
        out = _np.empty((len(p), K, 3))
        for k in range(K):
            out[:, k] = T.project(cix, p, k) / float(T.pix_size[cix])
        return out

    def _solve_tomo(self, stats, xpx, zpx, Pk, groups, apix, valid_k, prior=None):
        """stats (n, K, NF), xpx (n, K, 2) and zpx (n, K) px, Pk (K, 2, 3) the px shift on each projection of a 3D offset
        in px, groups {gs: (gstats (G, K, NF), label (n,), half (G, K, 2))} -> dict(shift (n, K, 2) px, cov (n, K, 3)
        px^2, rv (K, 3) and var (K, 3) the rotation vector (deg, detector frame) and its variance of each projection,
        prior: the rotation prior, kappa and offset prior it estimated). With ``prior`` (of :func:`_pool_priors`) those
        are fixed instead of estimated."""
        F = _F
        n, K, _ = stats.shape
        I2, I3 = _np.eye(2), _np.eye(3)
        obs = stats[..., [F['px'], F['py']]].astype(_np.float64)
        Ip = _info(stats)
        fin = _np.isfinite(xpx[..., 0]) & _np.isfinite(zpx)
        validp = _np.isfinite(obs[..., 0])
        vp = validp & _plausible(Ip) & (_np.nan_to_num(stats[..., F['cc']]) > 0) & valid_k[None] & fin
        lo, hi = _np.nanmin(xpx.reshape(-1, 2), 0), _np.nanmax(xpx.reshape(-1, 2), 0)
        xyz = _np.concatenate([xpx, zpx[..., None]], -1)
        shift = _np.full((n, K, 2), _np.nan)
        cov = _np.full((n, K, 3), _np.nan)
        rv, var = _np.full((K, 3), _np.nan), _np.full((K, 3), _np.nan)
        out = dict(shift=shift, cov=cov, rv=rv, var=var)

        # group peaks: precision from the half-split reproducibility; the groups that do not reproduce are dropped
        G_ = {}
        for gs, (gst, lab, half) in groups.items():
            if gs < 2:
                continue
            cen = _group_centroids(xyz, lab, validp, gst.shape[0])
            go = gst[..., [F['px'], F['py']]].astype(_np.float64)
            d2 = (half.astype(_np.float64) ** 2).sum(-1)
            gv = _np.isfinite(go[..., 0]) & _np.isfinite(cen[..., 0]) & _np.isfinite(d2)
            med = _np.array([_np.median(d2[gv[:, k], k]) if gv[:, k].any() else _np.nan for k in range(K)])
            sig2 = med / (_CHI2_2_MEDIAN * 4)
            Ig = _np.where(_np.isfinite(sig2), 1.0 / _np.where(_np.isfinite(sig2), _np.maximum(sig2, 1e-6), 1.0), 0.0)
            G_[gs] = dict(o=go, I=Ig, c=cen[..., :2], cz=cen[..., 2], ok=gv & (d2 <= 9 * med[None]))

        act = _np.array([k for k in _np.where(valid_k)[0] if vp[:, k].sum() >= _MIN_OBS], int)
        Ka = len(act)
        if Ka == 0:
            return out

        # observations of the active projections, sorted by projection
        nobs = _np.median(vp[:, act].sum(0))
        L = 0
        while L < _MAX_LEVEL and 2 * (2 ** (L + 1) + 1) ** 2 - 3 <= nobs / _OBS_PER_DOF:
            L += 1
        Gn = 2 ** L + 1
        M = Gn * Gn
        Q, lvl = _ms_space(L, lo, hi)
        q = Q.shape[1]
        nl = _np.bincount(lvl)
        nlev = L + 1
        ar = _np.arange(5, 5 + q)
        B = 5 + q
        Tm = _np.zeros((5 + 2 * M, B))
        Tm[:5, :5] = _np.eye(5)
        Tm[5:, 5:] = Q
        z0 = _np.array([_np.median(zpx[vp[:, k], k]) if vp[:, k].any() else 0.0 for k in range(K)])
        Wall, Wl, Xl, Zl, Yl, Il, Jl, Dl = [], [], [], [], [], [], [], []
        for a, k in enumerate(act):
            W = _np.zeros((n, M))
            W[fin[:, k]] = _field_basis(xpx[fin[:, k], k], Gn, lo, hi)
            Wall.append(_sp.csr_matrix(W))
            pi = _np.where(vp[:, k])[0]
            Dk, Xk, Zk, Yk, Ik, Jk = [W[pi]], [xpx[pi, k]], [zpx[pi, k] - z0[k]], [obs[pi, k]], [Ip[pi, k]], [pi]
            for g in G_.values():
                gi = _np.where(g['ok'][:, k])[0]
                Dk.append(_field_basis(g['c'][gi, k], Gn, lo, hi))
                Xk.append(g['c'][gi, k])
                Zk.append(g['cz'][gi, k] - z0[k])
                Yk.append(g['o'][gi, k])
                Ik.append(_np.broadcast_to(g['I'][k] * I2, (len(gi), 2, 2)))
                Jk.append(_np.full(len(gi), -1))
            Dl.append(_np.concatenate(Dk))
            Wl.append(_sp.csr_matrix(Dl[-1]))
            Xl.append(_np.concatenate(Xk))
            Zl.append(_np.concatenate(Zk))
            Yl.append(_np.concatenate(Yk))
            Il.append(_np.concatenate(Ik))
            Jl.append(_np.concatenate(Jk))
        cnt = _np.array([len(j) for j in Jl])
        off = _np.r_[0, _np.cumsum(cnt)]
        X, Z, Y, I0, jid = [_np.concatenate(v) for v in (Xl, Zl, Yl, Il, Jl)]
        ka = _np.repeat(_np.arange(Ka), cnt)
        N = len(Z)
        isp = jid >= 0
        kt = act[ka]
        Wflat = _sp.block_diag(Wl, format='csr')
        WfT = Wflat.T.tocsr()
        Dn = _np.concatenate(Dl)
        nodes = _np.argsort(-Dn, 1)[:, :4]
        wts = _np.take_along_axis(Dn, nodes, 1)
        pidx = (ka[:, None, None] * M * M + nodes[:, :, None] * M + nodes[:, None, :]).reshape(N, 16)
        pw = (wts[:, :, None] * wts[:, None, :]).reshape(N, 16)
        del Dl, Dn
        Rfl = _np.zeros((N, 2, 5))
        Rfl[:, 0, 0] = 1
        Rfl[:, 1, 1] = 1
        Rfl[:, 0, 3], Rfl[:, 0, 4] = Z, -X[:, 1]
        Rfl[:, 1, 2], Rfl[:, 1, 4] = -Z, X[:, 0]
        PkN = Pk[kt[isp]]
        dense_r = _np.zeros((n, K, 2))
        dense_I = _np.zeros((n, K, 2, 2))
        dense_u = _np.zeros((n, K), bool)

        def predict(u):
            f = u[:, 5:] @ Q.T
            pf = _np.stack([Wflat @ f[:, :M].ravel(), Wflat @ f[:, M:].ravel()], -1)
            return _np.einsum('nai,ni->na', Rfl, u[ka, :5]) + pf

        def quad(r, I):
            return _np.einsum('na,nab,nb->n', r, I, r)

        def scale(st):
            """precision on the absolute scale of the residuals: the particles divided by their kappa"""
            return I0 * _np.where(isp, 1.0 / st['kappa'][ka], 1.0)[:, None, None]

        def info(st):
            return scale(st) * st['w'][:, None, None]

        def proj_offsets(delta):
            pj = _np.zeros((N, 2))
            pj[isp] = _np.einsum('naj,nj->na', PkN, delta[jid[isp]])
            return pj

        def assemble(Ie, ydel):
            """data part of the normal equations per projection, in the reduced parameters"""
            IR = _np.einsum('nab,nbj->naj', Ie, Rfl)
            Iy = _np.einsum('nab,nb->na', Ie, ydel)
            Hn = _np.zeros((Ka, 5 + 2 * M, 5 + 2 * M))
            Hn[:, :5, :5] = _np.add.reduceat(_np.einsum('nai,naj->nij', Rfl, IR), off[:-1])
            Hrf = (WfT @ IR.reshape(N, 10)).reshape(Ka, M, 2, 5)
            Hn[:, 5:5 + M, :5], Hn[:, 5 + M:, :5] = Hrf[:, :, 0], Hrf[:, :, 1]
            Hn[:, :5, 5:] = _np.swapaxes(Hn[:, 5:, :5], 1, 2)
            for (i, j), v in (((0, 0), Ie[:, 0, 0]), ((0, 1), Ie[:, 0, 1]), ((1, 1), Ie[:, 1, 1])):
                Hij = _np.bincount(pidx.ravel(), (pw * v[:, None]).ravel(), Ka * M * M).reshape(Ka, M, M)
                Hn[:, 5 + i * M:5 + (i + 1) * M, 5 + j * M:5 + (j + 1) * M] = Hij
                if i != j:
                    Hn[:, 5 + j * M:5 + (j + 1) * M, 5 + i * M:5 + (i + 1) * M] = _np.swapaxes(Hij, 1, 2)
            gf = (WfT @ Iy).reshape(Ka, M, 2)
            gn = _np.concatenate([_np.add.reduceat(_np.einsum('nai,na->ni', Rfl, Iy), off[:-1]), gf[:, :, 0], gf[:, :, 1]], 1)
            return Tm.T @ Hn @ Tm, gn @ Tm

        rot_prec = _np.full(3, _ROT_PREC[0]) if prior is None else prior['rot_prec'].copy()
        c_est = []

        def blocks(D, lv):
            """normal matrix with the field strengths lv (Ka, q) and the zero-mean prior on the rotations"""
            A = D.copy()
            A[:, ar, ar] += lv
            A[:, [0, 1], [0, 1]] += _SHIFT_PREC * s_[0]
            A[:, [2, 3, 4], [2, 3, 4]] += rot_prec
            return A

        def to_dense(r0, Ie):
            dense_r.fill(0)
            dense_I.fill(0)
            dense_u.fill(False)
            jp, kp = jid[isp], kt[isp]
            dense_r[jp, kp] = r0[isp]
            dense_I[jp, kp] = Ie[isp]
            dense_u[jp, kp] = True

        def offset_prior(r0, Ie):
            """covariance of the 3D offsets from the even/odd projection split"""
            if prior is not None:
                return prior['C'] / apix ** 2
            to_dense(r0, Ie / (1 + quad(r0, Ie) / 4)[:, None, None])
            C = _np.zeros((3, 3))
            qual = (dense_u[:, 0::2].sum(1) >= 8) & (dense_u[:, 1::2].sum(1) >= 8)
            if qual.sum() >= 10:
                De = []
                for par in (0, 1):
                    m_ = _np.zeros(K, bool)
                    m_[par::2] = True
                    IP = dense_I[qual] * m_[None, :, None, None] @ Pk
                    H = _np.einsum('kai,nkaj->nij', Pk, IP)
                    g = _np.einsum('nkaj,nka->nj', IP, dense_r[qual])
                    De.append(_np.linalg.solve(H + 1e-9 * _np.trace(H, axis1=1, axis2=2)[:, None, None] * I3, g[..., None])[..., 0])
                De = [d - d.mean(0) for d in De]
                C = (De[0].T @ De[1] + De[1].T @ De[0]) / (2 * qual.sum())
                ev, V = _np.linalg.eigh(C)
                C = (V * _np.clip(ev, 0, None)) @ V.T
                c_est[:] = [True]
            return C

        def offsets(r0, Ie, C):
            """posterior of the 3D offsets from the residuals r0 of the field"""
            to_dense(r0, Ie)
            IP = dense_I @ Pk
            H = _np.einsum('kai,nkaj->nij', Pk, IP)
            g = _np.einsum('nkaj,nka->nj', IP, dense_r)
            Mx = I3 + H @ C
            delta = _np.einsum('ij,nj->ni', C, _np.linalg.solve(Mx, g[..., None])[..., 0])
            Cp = C @ _np.linalg.inv(Mx)
            return delta, (Cp + _np.swapaxes(Cp, 1, 2)) / 2

        def update(st, u):
            """offsets, kappa and robust weights at the solution u"""
            st = dict(st, u=u)
            Ie = info(st)
            r0 = Y - predict(u)
            st['C'] = offset_prior(r0, Ie)
            st['delta'], st['Cp'] = offsets(r0, Ie, st['C'])
            r = r0 - proj_offsets(st['delta'])
            st['kappa'] = residual_scale(quad(r, I0))
            st['w'] = 1 / (1 + quad(r, scale(st)) / 4)
            return st

        big_k = _np.array([isp[off[a_]:off[a_ + 1]].sum() >= _MIN_KAPPA for a_ in range(Ka)])
        big_o = _np.repeat(big_k, cnt) & isp
        kap0 = []

        def residual_scale(m2):
            """kappa per projection: the median of its particle residuals; with few particles the median over the
            projections with enough of them, else over the residuals about each projection's median (the first call:
            fitted residuals of a few peaks shrink with the fit)"""
            if not kap0:
                kap0.append(max(_np.median(m2[isp]) / _CHI2_2_MEDIAN, 1e-12))
            pooled = max(_np.median(m2[big_o]) / _CHI2_2_MEDIAN, 1e-12) if big_o.any() else (
                kap0[0] if prior is None else prior['kappa'])
            kap = _np.full(Ka, pooled)
            for a_ in _np.where(big_k)[0]:
                sl = slice(off[a_], off[a_ + 1])
                kap[a_] = max(_np.median(m2[sl][isp[sl]]) / _CHI2_2_MEDIAN, 1e-12)
            return _np.maximum(kap, _np.median(kap) / _KAPPA_RANGE)

        def step(st, lam, upd):
            """one outer iteration of the base fit: solve at the strengths lam (nlev,), and their fixed-point update"""
            D, g = assemble(info(st), Y - proj_offsets(st['delta']))
            Sg = _np.linalg.inv(blocks(D, lam[lvl][None]))
            u = _np.einsum('kij,kj->ki', Sg, g)
            st = update(st, u)
            st['Sc'] = _np.einsum('kii->ki', Sg[:, 2:5, 2:5])
            lam_n = lam
            if upd:
                sdiag = _np.bincount(lvl, _np.einsum('aqq->q', Sg[:, 5:, 5:]), nlev)
                lam_n = (nl * Ka - lam * sdiag) / _np.maximum(_np.bincount(lvl, (u[:, 5:] ** 2).sum(0), nlev), _TINY)
                lam_n = _np.clip(lam ** (1 - _LAM_DAMP) * _np.maximum(lam_n, _TINY) ** _LAM_DAMP, s_[0] / _LAM_RANGE, s_[0] * _LAM_RANGE)
            if upd and prior is None:
                # rotations: precision 1/tau^2 per axis from the same evidence fixed point (shrinks the poorly
                # determined projections, e.g. few particles or little depth spread, towards no rotation)
                rp = (Ka - rot_prec * st['Sc'].sum(0)) / _np.maximum((u[:, 2:5] ** 2).sum(0), _TINY)
                rot_prec[:] = _np.clip(rot_prec ** (1 - _LAM_DAMP) * _np.maximum(rp, _TINY) ** _LAM_DAMP, *_ROT_PREC)
            return st, lam_n

        def dsolve(D, g, lv, frz=None, ur=None):
            """solution at the strengths lv (Ka, q); frz: projections whose rigid part (5 parameters) stays at ur"""
            A = blocks(D, lv)
            u = _np.linalg.solve(A, g[..., None])[..., 0]
            if frz is not None and frz.any():
                Af = A[frz]
                u[frz, :5] = ur[frz, :5]
                u[frz, 5:] = _np.linalg.solve(Af[:, 5:, 5:], (g[frz, 5:] - _np.einsum('kij,kj->ki', Af[:, 5:, :5], ur[frz, :5]))[..., None])[..., 0]
            return u

        # 1. base fit
        st = dict(kappa=_np.ones(Ka), w=_np.ones(N), delta=_np.zeros((n, 3)), C=_np.zeros((3, 3)), Cp=_np.zeros((n, 3, 3)))
        r = Y.copy()
        for a in range(Ka):
            sl = slice(off[a], off[a + 1])
            r[sl] -= _np.median(Y[sl][isp[sl]], 0)
        st['kappa'] = residual_scale(quad(r, I0))
        s_ = [float(I0[:, 0, 0].sum() + I0[:, 1, 1].sum()) / 2 / (2 * M - 3)]
        for _ in range(_BASE_OUTER):
            st = step(st, s_[0] * _np.ones(nlev), False)[0]
        s_[0] = float(info(st)[:, [0, 1], [0, 1]].sum() / 2) / (2 * M - 3)
        lam = s_[0] * _np.ones(nlev)
        for _ in range(_LAM_OUTER):
            rp0 = rot_prec.copy()
            st, lam_n = step(st, lam, True)
            done = bool((_np.abs(lam_n - lam) <= _LAM_RTOL * lam).all() and (_np.abs(rot_prec - rp0) <= _LAM_RTOL * rp0).all())
            lam = lam_n
            if done:
                break
        c_base, Sc = st['u'][:, 2:5].copy(), st['Sc']

        # 2. per projection model by cross-validation
        big = s_[0] * _LAM_RANGE * 1e3
        cands = [(-1, 1.0, False)] + [(l_, sc_, fz) for l_ in range(nlev) for sc_ in _CV_SCALES for fz in (False, True)]
        lvc = [_np.broadcast_to(_np.where(lvl <= l_, lam[lvl] * sc_, big)[None], (Ka, q)) for l_, sc_, _ in cands]
        Ie, Is = info(st), scale(st)
        Yd = Y - proj_offsets(st['delta'])
        fold = _np.random.default_rng(self.seed).integers(0, self.n_fold, N)
        scr = _np.zeros((len(cands), Ka))
        for fo in range(self.n_fold):
            te = fold == fo
            D, g = assemble(Ie * (~te)[:, None, None], Yd)
            ur = dsolve(D, g, lvc[0])
            for c_, (_, _, fz) in enumerate(cands):
                u_ = ur if c_ == 0 else dsolve(D, g, lvc[c_], _np.full(Ka, fz), ur)
                r_ = Yd[te] - predict(u_)[te]
                scr[c_] += _np.bincount(ka[te], -_np.log1p(quad(r_, Is[te]) / 4), Ka)
        best = _np.argmax(scr, 0)
        lv = _np.stack([lvc[best[a]][a] for a in range(Ka)])
        frz = _np.array([cands[best[a]][2] for a in range(Ka)])
        for _ in range(_CV_OUTER):
            D, g = assemble(info(st), Y - proj_offsets(st['delta']))
            st = update(st, dsolve(D, g, lv, frz, dsolve(D, g, lvc[0])))

        # 3. outputs: the shift of the model of step 2, the rotations of the base fit
        u = st['u']
        f = u[:, 5:] @ Q.T
        field = _np.full((n, K, 2), _np.nan)
        for a, k in enumerate(act):
            ok = fin[:, k]
            xk, zk = xpx[ok, k], zpx[ok, k] - z0[k]
            c = u[a, 2:5]
            rg = _np.stack([u[a, 0] + c[1] * zk - c[2] * xk[:, 1], u[a, 1] + c[2] * xk[:, 0] - c[0] * zk], -1)
            field[ok, k] = rg + _np.stack([Wall[a][ok] @ f[a, :M], Wall[a][ok] @ f[a, M:]], -1)
        shift[:] = field + _np.einsum('kaj,nj->nka', Pk, st['delta'])
        PC = _np.einsum('kai,nij->nkaj', Pk, st['Cp'])
        for m_, (a_, b_) in enumerate(((0, 0), (0, 1), (1, 1))):
            cov[..., m_] = _np.einsum('nkj,kj->nk', PC[:, :, a_], Pk[:, b_])
        bad = ~_np.isfinite(field[..., 0])
        shift[bad] = _np.nan
        cov[bad] = _np.nan
        deg = _math.degrees(1)
        rv[act] = c_base * deg
        var[act] = Sc * deg ** 2
        if self.verbosity > 1:
            nf = _np.bincount([cands[b][0] + 1 for b in best], minlength=nlev + 1)
            print(' finest level %d (%dx%d); per projection: %s; median kappa %.2f; offset prior sd %s A; rotation prior sd %s deg' % (
                L, Gn, Gn, ', '.join('%s %d' % ('rigid' if l_ == 0 else 'level %d' % (l_ - 1), c) for l_, c in enumerate(nf) if c),
                _np.median(st['kappa']), _np.round(_np.sqrt(_np.diag(st['C'])) * apix, 2), _np.round(deg / _np.sqrt(rot_prec), 3)), flush=True)
        out['prior'] = dict(rot_prec=rot_prec.copy(), kappa=float(_np.median(st['kappa'])),
                            C=st['C'] * apix ** 2 if c_est else None)
        return out


def _pool_priors(priors):
    """median over tomograms of the rotation prior variance, kappa and offset prior (A^2, PSD projected)"""
    C = [p['C'] for p in priors if p['C'] is not None]
    if C:
        C = _np.median(_np.stack(C), 0)
        ev, V = _np.linalg.eigh((C + C.T) / 2)
        C = (V * _np.clip(ev, 0, None)) @ V.T
    else:
        C = _np.zeros((3, 3))
    tau2 = _np.median(_np.stack([1.0 / p['rot_prec'] for p in priors]), 0)
    return dict(rot_prec=_np.clip(1.0 / tau2, *_ROT_PREC), kappa=float(_np.median([p['kappa'] for p in priors])), C=C)


def _field_basis(x, G, lo, hi):
    """(n, G*G) bilinear weights of the control-point grid."""
    u = (x - lo) / _np.maximum(hi - lo, 1e-9) * (G - 1)
    u = _np.clip(u, 0, G - 1 - 1e-9)
    i0 = _np.floor(u).astype(int)
    f = u - i0
    W = _np.zeros((len(x), G * G))
    for dy in (0, 1):
        for dx in (0, 1):
            W[_np.arange(len(x)), (i0[:, 1] + dy) * G + (i0[:, 0] + dx)] += (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
    return W


def _ms_space(L, lo, hi):
    """orthonormal-in-coefficients hierarchical basis Q (2 G^2, 2 G^2 - 3) on the nodes of the G = 2**L + 1 grid, the level of
    each column; level 0 without the constant x and y shifts and the rotation about the centre"""
    G = 2 ** L + 1
    gx, gy = _np.linspace(lo[0], hi[0], G), _np.linspace(lo[1], hi[1], G)
    nx, ny = _np.meshgrid(gx, gy)
    nx, ny = nx.ravel(), ny.ravel()
    M = G * G
    t = _np.linspace(0, 1, G)
    cols, lvl = [], []
    for l in range(L + 1):
        Gl = 2 ** l + 1
        h = 1.0 / (Gl - 1)
        hat = _np.clip(1 - _np.abs(t[None] - _np.arange(Gl)[:, None] * h) / h, 0, None)
        new = _np.ones((Gl, Gl), bool)
        if l > 0:
            new[::2, ::2] = False
        H = _np.einsum('iy,jx->ijyx', hat, hat).reshape(Gl, Gl, M)[new.T]
        Nl = H.shape[0]
        if l == 0:
            cx, cy = nx[[0, G - 1, M - G, M - 1]], ny[[0, G - 1, M - G, M - 1]]
            Nc = _np.zeros((8, 3))
            Nc[:4, 0] = 1
            Nc[4:, 1] = 1
            Nc[:4, 2] = -(cy - ny.mean())
            Nc[4:, 2] = cx - nx.mean()
            Cl = _np.linalg.qr(Nc, mode='complete')[0][:, 3:]
            Bx = _np.zeros((8, 2 * M))
            Bx[:4, :M], Bx[4:, M:] = H, H
        else:
            Cl = _np.eye(2 * Nl)
            Bx = _np.zeros((2 * Nl, 2 * M))
            Bx[:Nl, :M], Bx[Nl:, M:] = H, H
        cols.append(Cl.T @ Bx)
        lvl += [l] * Cl.shape[1]
    return _np.concatenate(cols).T, _np.array(lvl, int)


def _rotvec_rotm(rv):
    """rotation vectors (..., 3) deg -> rotation matrices (..., 3, 3), float32"""
    r = _np.radians(_np.asarray(rv, _np.float64))
    th = _np.linalg.norm(r, axis=-1)[..., None]
    k = r / _np.maximum(th, 1e-12)
    K = _np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - _np.swapaxes(K, -1, -2)
    s, c = _np.sin(th)[..., None], _np.cos(th)[..., None]
    return (_np.eye(3) + s * K + (1 - c) * (K @ K)).astype(_np.float32)


def _group_centroids(xpx, label, valid, G):
    """mean projected position (px) of the members of each group per projection -> (G, K, columns) (NaN when empty)."""
    n, K, nc = xpx.shape
    out = _np.full((G, K, nc), _np.nan)
    for g in range(G):
        m = label == g
        for k in range(K):
            v = m & valid[:, k]
            if v.any():
                out[g, k] = xpx[v, k].mean(0)
    return out


def _info(stats_kn):
    """precision of the peak positions, -H z^2 / cc (the CC curvature in units of the map noise), psd -> (..., 2, 2)."""
    F = _F
    H = _np.zeros(stats_kn.shape[:-1] + (2, 2))
    H[..., 0, 0] = stats_kn[..., F['h_xx']]
    H[..., 0, 1] = H[..., 1, 0] = stats_kn[..., F['h_xy']]
    H[..., 1, 1] = stats_kn[..., F['h_yy']]
    sc = _np.nan_to_num(stats_kn[..., F['z_peak']]) ** 2 / _np.maximum(_np.nan_to_num(stats_kn[..., F['cc']]), 1e-12)
    I = _np.nan_to_num(-H) * sc[..., None, None]
    ev, V = _np.linalg.eigh(I.reshape(-1, 2, 2))
    ev = _np.clip(ev, 1e-12, None)
    return _np.einsum('nij,nj,nkj->nik', V, ev, V).reshape(I.shape)


def _plausible(I):
    """precisions with a sane condition number and scale (rank-deficient fits give absurd curvatures; a peak without
    curvature, NaN in the statistics, gives a vanishing one)."""
    ev = _np.linalg.eigvalsh(_np.nan_to_num(I.reshape(-1, 2, 2))).reshape(I.shape[:-2] + (2,))
    big = ev[..., 1]
    med = _np.nanmedian(_np.where(_np.isfinite(I).all((-1, -2)), big, _np.nan))
    return _np.isfinite(I).all((-1, -2)) & (ev[..., 0] > big * 1e-4) & (big < 100 * med) & (big > 1e-4 * med)
