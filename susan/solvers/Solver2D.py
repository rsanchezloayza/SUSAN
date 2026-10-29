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
import warnings as _warnings

import numpy as _np

import susan.data as _ssa_data
import susan.io.ali_stats as _ali_stats
import susan.utils as _ssa_utils
from susan.solvers._solvers_core import fit_robust as _fit_robust_core

__all__ = ['Solver2D']

_CHI2_2_MEDIAN = 1.3863
_F = {f: i for i, f in enumerate(_ali_stats.FIELDS)}


class Solver2D:
    """2D refinement from the statistics of a 2D alignment (``output_type='alignment_stats'``).

    Per tomogram (and per half-set when the groups of the statistics are per half), every parameter estimated from
    the data:

    1. group peaks: precision from the half-split reproducibility; groups that do not reproduce are dropped;
    2. rigid correction per projection (shift + in-plane rotation) from the finest usable grouping (3x3, 2x2, else
       the 1x1 shift);
    3. particles: reliability P(real | z_peak, r2, projection signal) from a logistic model of their agreement with the
       rigid prediction, and their precision on the absolute scale of the residuals;
    4. per projection, the field model (rigid; rigid + G x G grid; the grid with the rigid part kept) by
       cross-validation on held-out particle and group peaks;
    5. cross-tilt shrinkage of the non-rigid part toward the similar projections, its strength by cross-validation;
    6. per particle posterior of the observation and the field;
    7. the orientation of each projection (rotations about the detector x, y and the beam) from the 1x1 and 2x2
       group peaks over the sequential angular search: per projection, plus a linear field over the 2x2 groups, each
       shrunk by its reproducibility (split halves of the groups).

    Usage: align the particles in 2D with ``output_type='alignment_stats'`` (offset step 0.5 px; the cone and in-plane
    search define the sequential angular search, none: the orientations are kept; with independent half-sets the
    groups are per half), then :meth:`solve` gives the particles of the next alignment; :class:`SolverHighRes` makes
    those of the reference reconstruction from the same statistics.

    Attributes
    ----------
    models : tuple of int
        Grid sizes G of the non-rigid field tried per projection.  Default ``(2, 3, 4, 5)``.
    n_fold : int
        Cross-validation folds.  Default ``5``.
    seed : int
        Seed of the folds.  Default ``0``.
    verbosity : int
        ``0`` silent, ``1`` one line per tomogram and half.  Default ``1``.
    """

    def __init__(self):
        self.models    = (2, 3, 4, 5)
        self.n_fold    = 5
        self.seed      = 0
        self.verbosity = 1
        self.solved    = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, ptcls_in, stats, tomos):
        """Particles for the next alignment.

        Parameters
        ----------
        ptcls_in : str or :class:`susan.data.Particles`
            The *input* particles of the alignment that wrote ``stats`` (its offsets are relative to their prj_t).
        stats : str or dict
            The ``.stats.npz`` (or the aligner output ``.ptclsraw``), or :func:`susan.io.ali_stats.read` of it.
        tomos : str or :class:`susan.data.Tomograms`

        Returns
        -------
        :class:`susan.data.Particles`
            prj_t = input prj_t + solved shift; prj_eu = the solved rotation (step 7) applied to the input prj_eu;
            prj_w unchanged (an unsolved projection keeps its pose and is retried in the next round).
            :attr:`solved` (n_ptcl, K) marks the solved projections.  The uncertainty of what was applied is added to
            the statistics (the file when ``stats`` is a path, else the dict): ``projections/post_cov`` (n_ptcl, K, 3)
            float32, the posterior covariance [xx, xy, yy] (px^2) of the shift, inv(I_obs + I / s^2) with the robustly
            reweighted observation (s^2 I for the particles that keep the field), NaN where unsolved;
            ``projections/ang_var`` (n_ptcl, K, 3) float32, the variance (deg^2) of the applied rotation about x, y and
            the beam: per tilt f1 v1 of the shrunk 1x1 estimate plus f2 (v2 + v1) of the 2x2 field (v2 averaged over
            the groups, the regression of the field not counted: an approximation), 0 for a component without
            estimate, NaN where no rotation was applied.
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
        xpx = self._projected_positions(P, T)
        prj_t_in, prj_eu_in = P.prj_t.copy(), P.prj_eu.copy()
        self.solved = _np.zeros(P.prj_w.shape, bool)
        post_cov = _np.full(P.prj_w.shape + (3,), _np.nan, _np.float32)
        ang_var = _np.full(P.prj_w.shape + (3,), _np.nan, _np.float32)
        for t in _np.unique(tid):
            cix, ti = int(T.get_cix(t)), stomo.index(int(t))
            K = int(T.num_proj[cix])
            apix = float(T.pix_size[cix])
            for h in ((1, 2) if halves == 2 else (0,)):
                sel = _np.where((tid == t) & ((hid == h) if h else True))[0]
                if len(sel) == 0:
                    continue
                grp = {}
                for gs in gsizes:
                    G, o = gs * gs, (h - 1) * gs * gs if h else 0
                    n_g = S['grouped%d/stats' % gs].shape[1] // (2 if halves == 2 else 1)
                    if n_g != G:
                        continue
                    grp[gs] = (S['grouped%d/stats' % gs][ti][o:o + G, :K], S['grouped%d/label' % gs][sel] - o,
                               S['grouped%d/half' % gs][ti][o:o + G, :K])
                valid_k = P.prj_w[sel][:, :K].max(0) > 0
                if self.verbosity:
                    print('    Tomogram %d%s:' % (t, (', half %d' % h) if h else ''), end='', flush=True)
                r = self._solve_tomo(st[sel][:, :K], xpx[sel][:, :K], grp, apix, valid_k)
                ok = _np.isfinite(r['shift'][..., 0])
                for k in range(K):
                    i = sel[ok[:, k]]
                    if len(i) == 0:
                        continue
                    P.prj_t[i, k] = prj_t_in[i, k] + r['shift'][ok[:, k], k] * apix
                    post_cov[i, k] = r['cov'][ok[:, k], k]
                    self.solved[i, k] = True
                rv = self._solve_angles(S, ti, h, sel, xpx[sel][:, :K])
                if rv is not None:
                    rv, var = rv
                    m = _np.isfinite(rv[..., 0]) & (P.prj_w[sel][:, :K] > 0)
                    ii, kk = sel[_np.where(m)[0]], _np.where(m)[1]
                    D = _rotvec_rotm(rv[m])
                    P.prj_eu[ii, kk] = _ssa_utils.rotm_to_euZYZ(D @ _ssa_utils.euZYZ_to_rotm(prj_eu_in[ii, kk]))
                    ang_var[ii, kk] = var[m]
        self._append_variances(stats, S, post_cov, ang_var)
        return P

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
    def _projected_positions(P, T):
        """(n, K, 2) projected particle positions, px from the frame centre (crop geometry, without prj_t)."""
        out = _np.full(P.prj_t.shape[:2] + (2,), _np.nan)
        tid = P.tomo_id.flatten().astype(int)
        pt = P.pos()
        for t in _np.unique(tid):
            cix = int(T.get_cix(t))
            sel = _np.where(tid == t)[0]
            p = T.to_tomo(cix, pt[sel]).astype(_np.float64)
            for k in range(int(T.num_proj[cix])):
                out[sel, k] = T.project(cix, p, k)[:, :2] / float(T.pix_size[cix])
        return out

    def _solve_tomo(self, stats, xpx, groups, apix, valid_k):
        """solve_auto of poc/susan_poc/pool2d.py: stats (n, K, NF), xpx (n, K, 2) px, groups {gs: (gstats (G, K, NF),
        label (n,), half (G, K, 2))} -> dict(shift (n, K, 2) px, angle (n, K) deg, ...)."""
        F = _F
        models, n_fold = tuple(self.models), self.n_fold
        n, K, _ = stats.shape
        rng = _np.random.default_rng(self.seed)
        obs = stats[..., [F['px'], F['py']]].astype(_np.float64)
        Ip = _info(stats)
        zp = _np.nan_to_num(stats[..., F['z_peak']])
        r2p = _np.nan_to_num(stats[..., F['r2']], nan=0.0)
        vp = _np.isfinite(obs[..., 0]) & _plausible(Ip) & (_np.nan_to_num(stats[..., F['cc']]) > 0) & valid_k[None]
        validp = _np.isfinite(obs[..., 0])
        lo, hi = _np.nanmin(xpx.reshape(-1, 2), 0), _np.nanmax(xpx.reshape(-1, 2), 0)
        I2 = _np.eye(2)
        # 1. groups: reproducibility
        G_ = {}
        for gs, (gst, lab, half) in groups.items():
            cen = _group_centroids(xpx, lab, validp, gst.shape[0])
            go = gst[..., [F['px'], F['py']]].astype(_np.float64)
            d2 = (half.astype(_np.float64) ** 2).sum(-1)
            gv = _np.isfinite(go[..., 0]) & _np.isfinite(cen[..., 0]) & _np.isfinite(d2)
            med = _np.array([_np.median(d2[gv[:, k], k]) if gv[:, k].any() else _np.nan for k in range(K)])
            sig2 = med / (_CHI2_2_MEDIAN * 4)
            rep_ok = gv & (d2 <= 9 * med[None])
            if gst.shape[0] == 1:
                m1 = _np.nanmedian(d2[0])
                rep_ok = gv & (d2 <= 9 * m1)
                sig2 = _np.where(_np.isfinite(d2[0]), _np.maximum(d2[0], 1e-6) / 4, _np.nan)
            Ig = _np.where(_np.isfinite(sig2)[None, :, None, None],
                           I2[None, None] / _np.where(_np.isfinite(sig2), _np.maximum(sig2, 1e-6), 1.0)[None, :, None, None], 0.0)
            G_[gs] = dict(o=go, I=_np.broadcast_to(Ig, (gst.shape[0], K, 2, 2)).copy(), c=cen, ok=rep_ok,
                          z=_np.nan_to_num(gst[..., F['z_peak']]))

        def rigid_eval(t, w, x):
            return t[None] + w * _np.stack([-x[:, 1], x[:, 0]], -1)

        # 2. rigid part from the finest usable grouping
        rt, rw, rl = _np.full((K, 2), _np.nan), _np.full(K, _np.nan), _np.full(K, -1)
        for gs in G_:
            sk = _np.array([1.0 / G_[gs]['I'][G_[gs]['ok'][:, k], k, 0, 0].mean() if G_[gs]['ok'][:, k].any() else _np.nan
                            for k in range(K)])
            G_[gs]['use'] = _np.isfinite(sk) & (sk <= 9 * _np.nanmedian(sk))
        for k in _np.where(valid_k)[0]:
            for gs in (3, 2, 1):
                if gs not in G_ or not G_[gs]['use'][k]:
                    continue
                g = G_[gs]
                u = g['ok'][:, k]
                if gs == 1:
                    rt[k], rw[k], rl[k] = g['o'][0, k], 0.0, 1
                    break
                if u.sum() >= min(4, gs * gs):
                    th, _ = _fit_robust(_rigid_rows(g['c'][u, k]), g['I'][u, k], g['o'][u, k], _np.ones(u.sum()), _np.zeros(3))
                    rt[k], rw[k], rl[k] = th[:2], th[2], gs
                    break
        # 3. particles: reliability and absolute precision
        res = _np.full((n, K), _np.nan)
        for k in _np.where(_np.isfinite(rt[:, 0]))[0]:
            res[:, k] = _np.sqrt(((obs[:, k] - rigid_eval(rt[k], rw[k], xpx[:, k])) ** 2).sum(-1))
        lab_ok = vp & _np.isfinite(res)
        zk = G_[1]['z'][0] if 1 in G_ else _np.nanmedian(_np.where(vp, zp, _np.nan), 0)
        zkp = _np.broadcast_to(_np.nan_to_num(zk)[None], zp.shape)
        confp = lab_ok & (zp >= _np.quantile(zp[lab_ok], 0.75)) & (r2p <= _np.quantile(r2p[lab_ok], 0.25))
        r_agree = 1.5 * _np.quantile(res[confp], 0.9)
        X = _np.stack([_np.ones(lab_ok.sum()), zp[lab_ok], r2p[lab_ok], zkp[lab_ok]], 1)
        beta = _logistic(X, (res[lab_ok] <= r_agree).astype(float))
        Pp = _np.where(vp, 1 / (1 + _np.exp(-_np.clip(beta[0] + beta[1] * zp + beta[2] * r2p + beta[3] * zkp, -30, 30))), 0.0)
        Ia = _np.zeros_like(Ip)
        for k in _np.where(_np.isfinite(rt[:, 0]))[0]:
            v = Pp[:, k] > 1e-3
            if not v.any():
                continue
            r = obs[v, k] - rigid_eval(rt[k], rw[k], xpx[v, k])
            m2 = _np.einsum('na,nab,nb->n', r, Ip[v, k], r)
            kap_k = max(_wmedian(m2, Pp[v, k]) / _CHI2_2_MEDIAN, 1e-12)
            Ia[v, k] = Ip[v, k] / kap_k
        # 4. per projection model by cross-validation
        fine = [gs for gs in (3, 4, 5) if gs in G_]
        LAT = _np.stack(_np.meshgrid(_np.linspace(lo[0], hi[0], 7), _np.linspace(lo[1], hi[1], 7)), -1).reshape(-1, 2)
        DLAT = _rigid_rows(LAT).reshape(-1, 3)
        PL = _np.linalg.solve(DLAT.T @ DLAT + 1e-12 * _np.eye(3), DLAT.T)
        L5 = _np.stack(_np.meshgrid(_np.linspace(lo[0], hi[0], 5), _np.linspace(lo[1], hi[1], 5)), -1).reshape(-1, 2)
        DL = {(nm, G): _design(x_, G, lo, hi) for nm, x_ in (('lat', LAT), ('l5', L5)) for G in models}

        def apply(D, th):
            return (D.reshape(-1, D.shape[2]) @ th).reshape(-1, 2)

        def predict(G, th, frozen, x, D=None):
            if G == 0:
                return rigid_eval(th[:2], th[2], x)
            if D is None:
                D = _design(x, G, lo, hi)
            if not frozen:
                return apply(D, th)
            node = _np.zeros_like(th)
            node[3:] = th[3:]
            pr = PL @ apply(DL[('lat', G)], node).ravel()
            return rigid_eval(th[:2], th[2], x) + apply(D, node) - rigid_eval(pr[:2], pr[2], x)

        def score(O, II, Pw, pred):
            r = O - pred
            return float(_np.sum(Pw * -_np.log1p(_np.einsum('na,nab,nb->n', r, II, r) / 4)))

        field = _np.full((n, K, 2), _np.nan)
        model = _np.full(K, -99)
        fits = {}
        for k in _np.where(valid_k & _np.isfinite(rt[:, 0]))[0]:
            ip = _np.where(Pp[:, k] > 1e-3)[0]
            O, II, Xo, Pw = [obs[ip, k]], [Ia[ip, k]], [xpx[ip, k]], [Pp[ip, k]]
            for gs in fine:
                g = G_[gs]
                ig = _np.where(g['ok'][:, k])[0]
                O.append(g['o'][ig, k])
                II.append(g['I'][ig, k])
                Xo.append(g['c'][ig, k])
                Pw.append(_np.ones(len(ig)))
            O, II, Xo, Pw = _np.concatenate(O), _np.concatenate(II), _np.concatenate(Xo), _np.concatenate(Pw)
            fold = rng.integers(0, n_fold, len(O))
            th0r = _np.array([rt[k, 0], rt[k, 1], rw[k]])
            best, per_fold = None, {}
            cands = [(0, True)] + [(Gm, fz) for Gm in models for fz in ((True,) if rl[k] == 1 else (False, True))]
            for Gm, fz in cands:
                D = _design(Xo, Gm, lo, hi) if Gm else _rigid_rows(Xo)
                th0 = _np.zeros(D.shape[2])
                th0[:3] = th0r
                sc, pf = 0.0, []
                for f in range(n_fold):
                    tr, te = fold != f, fold == f
                    if Gm and Pw[tr].sum() < 2 * D.shape[2]:
                        sc = -_np.inf
                        break
                    th = th0 if Gm == 0 else _fit_robust(D[tr], II[tr], O[tr], Pw[tr], th0, frozen=fz)[0]
                    pf.append((th, 1.0))
                    sc += score(O[te], II[te], Pw[te], predict(Gm, th, fz, Xo[te], D[te]))
                if best is None or sc > best[0]:
                    best = (sc, Gm, fz)
                per_fold[(Gm, fz)] = pf
            _, Gm, fz = best
            D = _design(Xo, Gm, lo, hi) if Gm else _rigid_rows(Xo)
            th0 = _np.zeros(D.shape[2])
            th0[:3] = th0r
            th = th0 if Gm == 0 else _fit_robust(D, II, O, Pw, th0, frozen=fz)[0]
            model[k] = 0 if Gm == 0 else (-Gm if fz else Gm)
            fits[k] = dict(G=Gm, fz=fz, th=th, folds=per_fold[(Gm, fz)], II=II, O=O, Pw=Pw, Xo=Xo, fold=fold, D=D)

        def split(fk, th, x, D=None):
            G = fk['G']
            pr = PL @ predict(G, th, fk['fz'], LAT, DL.get(('lat', G))).ravel()
            return pr, predict(G, th, fk['fz'], x, D) - rigid_eval(pr[:2], pr[2], x)

        # 5. cross-tilt shrinkage of the non-rigid part
        VK = sorted(fits)
        alpha = _np.zeros(K)
        if VK:
            vec = _np.stack([split(fits[k], fits[k]['th'], L5, DL.get(('l5', fits[k]['G'])))[1].ravel() for k in VK])
            dm = ((vec[:, None] - vec[None]) ** 2).sum(-1)
            h2 = _np.median(dm[dm > 0]) if _np.any(dm > 0) else 1.0
            for a_, k in enumerate(VK):
                wv = _np.exp(-dm[a_] / h2)
                wv[a_] = 0
                wv = wv / max(wv.sum(), 1e-12)
                fk = fits[k]

                def target(x):
                    out_ = _np.zeros((len(x), 2))
                    Dx = {}
                    for b_, l in enumerate(VK):
                        if wv[b_] > 1e-3:
                            G = fits[l]['G']
                            if G and G not in Dx:
                                Dx[G] = _design(x, G, lo, hi)
                            out_ += wv[b_] * split(fits[l], fits[l]['th'], x, Dx.get(G))[1]
                    return out_

                Tk = target(fk['Xo'])
                best = None
                for al in (0.0, 0.25, 0.5, 0.75, 1.0):
                    sc = 0.0
                    for f, (th_f, _) in enumerate(fk['folds']):
                        te = fk['fold'] == f
                        pr, nr = split(fk, th_f, fk['Xo'][te], fk['D'][te] if fk['G'] else None)
                        sc += score(fk['O'][te], fk['II'][te], fk['Pw'][te],
                                    rigid_eval(pr[:2], pr[2], fk['Xo'][te]) + (1 - al) * nr + al * Tk[te])
                    if best is None or sc > best[0]:
                        best = (sc, al)
                alpha[k] = best[1]
                pr, nr = split(fk, fk['th'], xpx[:, k])
                field[:, k] = rigid_eval(pr[:2], pr[2], xpx[:, k]) + (1 - alpha[k]) * nr + alpha[k] * target(xpx[:, k])
        # 6. per particle posterior
        conf = (Pp >= 0.9) & _np.isfinite(field[..., 0])
        rr = (obs - field)[conf]
        Iinv = _np.linalg.inv(Ip[conf])
        keep = (rr ** 2).sum(-1) < _np.quantile((rr ** 2).sum(-1), 0.9)
        rr, Iinv = rr[keep], Iinv[keep]
        kap0 = _np.median(_np.einsum('na,nab,nb->n', rr, _np.linalg.inv(Iinv), rr)) / _CHI2_2_MEDIAN
        v0 = _np.median((rr ** 2).sum(-1)) / _CHI2_2_MEDIAN
        kg, sg = kap0 * _np.logspace(-4, 1, 101)[:, None, None], v0 * _np.logspace(-4, 1, 101)[None, :, None]
        c00, c11, c01 = kg * Iinv[None, None, :, 0, 0] + sg, kg * Iinv[None, None, :, 1, 1] + sg, kg * Iinv[None, None, :, 0, 1]
        det = c00 * c11 - c01 ** 2
        ll = -0.5 * _np.sum(_np.log(det) + (c11 * rr[:, 0] ** 2 - 2 * c01 * rr[:, 0] * rr[:, 1] + c00 * rr[:, 1] ** 2) / det, -1)
        ik, is_ = _np.unravel_index(_np.argmax(ll), ll.shape)
        kap, s2 = kg[ik, 0, 0], sg[0, is_, 0]
        shift = field.copy()
        ii, kk = _np.where((Pp > 1e-3) & _np.isin(_np.arange(K), VK)[None])
        Io = Ip[ii, kk] / kap
        d = obs[ii, kk] - field[ii, kk]
        Ic = _np.linalg.inv(_np.linalg.inv(Io) + I2 * s2)
        Io = Io * (Pp[ii, kk] / (1 + _np.einsum('na,nab,nb->n', d, Ic, d) / 4))[:, None, None]
        shift[ii, kk] = _np.linalg.solve(Io + I2 / s2, ((Io @ obs[ii, kk][..., None])[..., 0] + field[ii, kk] / s2)[..., None])[..., 0]
        cov = _np.full((n, K, 3), _np.nan)
        cov[_np.isfinite(field[..., 0])] = [s2, 0.0, s2]
        Cp = _np.linalg.inv(Io + I2 / s2)
        cov[ii, kk] = _np.stack([Cp[:, 0, 0], Cp[:, 0, 1], Cp[:, 1, 1]], 1)
        if self.verbosity:
            names = {0: 'rigid'}
            for m in models:
                names[m], names[-m] = '%dx%d' % (m, m), '%dx%d frozen rigid' % (m, m)
            print(' rigid from %s; models %s; median alpha %.2f; s_res %.2f A' % (
                {('%dx%d' % (g, g) if g > 0 else 'none'): int((rl[valid_k] == g).sum()) for g in sorted(set(rl[valid_k]))},
                {names[m]: int((model == m).sum()) for m in sorted(set(model[VK]))}, _np.median(alpha[VK]) if VK else 0.0,
                _math.sqrt(s2) * apix), flush=True)
        return dict(field=field, shift=shift, cov=cov, rigid=(rt, rw, rl), model=model, alpha=alpha,
                    s_res=_math.sqrt(s2), kap=kap, P=Pp)

    def _solve_angles(self, S, ti, h, sel, xpx):
        """7. rotation vectors (deg, detector frame, (n, K, 3)) of the particles sel of tomogram index ti and half h and
        their variances (deg^2; per tilt f1 v1 of the shrunk 1x1 estimate, plus f2 (v2 + v1) of the 2x2 field with v2
        averaged over the groups, the regression of the field not counted), xpx (n, K, 2) their projected positions;
        None without the sequential angular search."""
        A = S.get('angles')
        if A is None or A.shape[0] < 2 or 'grouped1/ang_cc' not in S:
            return None
        A = A.astype(_np.float64)
        n, K = xpx.shape[:2]
        est = {}
        for gs in (1, 2):
            if 'grouped%d/ang_cc' % gs not in S:
                continue
            G = gs * gs
            o = (h - 1) * G if h else 0
            y = S['grouped%d/ang_cc' % gs][ti][o:o + G, :K]
            yh = S['grouped%d/ang_half' % gs][ti][o:o + G, :K]
            e = _vertex(y, A)
            est[gs] = (e, (_vertex(yh[..., 0], A) - _vertex(yh[..., 1], A)) ** 2 / 4)
        e1, v1 = est[1][0][0], est[1][1][0]
        f1 = _shrink(e1, v1)
        rv = _np.broadcast_to(f1 * _np.nan_to_num(e1), (n, K, 3)).copy()
        var = _np.broadcast_to(f1 * v1, (n, K, 3)).copy()
        f2 = _np.zeros(3)
        if 2 in est:
            e2, v2 = est[2]
            dev = e2 - e1[None]
            f2 = _shrink(dev, v2 + v1[None])
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore', RuntimeWarning)
                var += f2 * (_np.nanmean(v2, 0) + v1)[None]
            lab = S['grouped2/label'][sel] - ((h - 1) * 4 if h else 0)
            cen = _group_centroids(xpx, lab, _np.isfinite(xpx[..., 0]), 4)
            for k in range(K):
                for c in _np.where(f2 > 0)[0]:
                    u = _np.isfinite(dev[:, k, c]) & _np.isfinite(cen[:, k, 0])
                    if u.sum() < 3:
                        continue
                    X = _np.c_[_np.ones(u.sum()), cen[u, k]]
                    th = _np.linalg.lstsq(X, f2[c] * dev[u, k, c], rcond=None)[0]
                    rv[:, k, c] += th[0] + xpx[:, k] @ th[1:]
        lim = _np.abs(A).max(0)
        rv = _np.clip(rv, -lim, lim)
        var = _np.where(_np.isfinite(var), var, 0.0)
        if self.verbosity:
            rms = _np.sqrt(_np.nanmean(rv[_np.isfinite(rv[..., 0])].reshape(-1, 3) ** 2, 0))
            print('      angles (x, y, in-plane): rms %.3f %.3f %.3f deg; shrinkage 1x1 %s, 2x2 %s' % (
                tuple(rms) + (_np.round(f1, 2), _np.round(f2, 2))), flush=True)
        return rv, var


# ----------------------------------------------------------------------
# Helpers of the field fit
# ----------------------------------------------------------------------

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


def _design(x, G, lo, hi):
    """(n, 2, P) design of the correction field: params [sx, sy, w, nodes_x (G*G), nodes_y (G*G)]."""
    n = len(x)
    Wn = _field_basis(x, G, lo, hi) if G > 0 else _np.zeros((n, 0))
    m = Wn.shape[1]
    D = _np.zeros((n, 2, 3 + 2 * m))
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 0, 2] = -x[:, 1]
    D[:, 1, 2] = x[:, 0]
    D[:, 0, 3:3 + m] = Wn
    D[:, 1, 3 + m:] = Wn
    return D


def _rigid_rows(x):
    """(m, 2, 3) design of the rigid correction t + w J x."""
    D = _np.zeros((len(x), 2, 3))
    D[:, 0, 0] = 1
    D[:, 1, 1] = 1
    D[:, 0, 2] = -x[:, 1]
    D[:, 1, 2] = x[:, 0]
    return D


def _vertex(y, A):
    """peak of the CC over the sequential angular search: y (..., A) at the rotation vectors A (A, 3) deg -> (..., 3):
    the cone points (and the input) by a 2D quadratic in (x, y), the in-plane points by a 1D quadratic; NaN when there
    are too few points or the fit has no maximum; at most the scanned extent."""
    out = _np.full(y.shape[:-1] + (3,), _np.nan)
    y = y.astype(_np.float64)
    cone = _np.r_[0, _np.where((_np.abs(A[:, 2]) < 1e-3) & (_np.hypot(A[:, 0], A[:, 1]) > 1e-3))[0]]
    inp = _np.r_[0, _np.where((_np.hypot(A[:, 0], A[:, 1]) < 1e-3) & (_np.abs(A[:, 2]) > 1e-3))[0]]
    if len(cone) >= 6:
        u, v = A[cone, 0], A[cone, 1]
        c = y[..., cone] @ _np.linalg.pinv(_np.c_[_np.ones_like(u), u, v, u * u, u * v, v * v]).T
        h00, h01, h11 = 2 * c[..., 3], c[..., 4], 2 * c[..., 5]
        det = h00 * h11 - h01 ** 2
        ok = (h00 < 0) & (det > 0)
        det = _np.where(ok, det, 1.0)
        x = -(h11 * c[..., 1] - h01 * c[..., 2]) / det
        z = -(-h01 * c[..., 1] + h00 * c[..., 2]) / det
        r = _np.hypot(x, z)
        sc = _np.minimum(1.0, _np.hypot(u, v).max() / _np.maximum(r, 1e-12))
        out[..., 0] = _np.where(ok, x * sc, _np.nan)
        out[..., 1] = _np.where(ok, z * sc, _np.nan)
    if len(inp) >= 3:
        w = A[inp, 2]
        c = y[..., inp] @ _np.linalg.pinv(_np.c_[_np.ones_like(w), w, w * w]).T
        ok = c[..., 2] < 0
        out[..., 2] = _np.where(ok, _np.clip(-c[..., 1] / (2 * _np.where(ok, c[..., 2], -1.0)), w.min(), w.max()), _np.nan)
    return out


def _shrink(e, v):
    """per component (last axis), the fraction of the spread of the estimates e beyond their noise variance v"""
    out = _np.zeros(e.shape[-1])
    for c in range(e.shape[-1]):
        m = _np.isfinite(e[..., c]) & _np.isfinite(v[..., c])
        if m.sum() > 2:
            tau2 = max(_np.mean(e[..., c][m] ** 2) - _np.mean(v[..., c][m]), 0.0)
            out[c] = tau2 / (tau2 + _np.mean(v[..., c][m])) if tau2 > 0 else 0.0
    return out


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
    """mean projected position (px) of the members of each group per projection -> (G, K, 2) (NaN when empty)."""
    n, K, _ = xpx.shape
    out = _np.full((G, K, 2), _np.nan)
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
    """precisions with a sane condition number and scale (rank-deficient fits give absurd curvatures)."""
    ev = _np.linalg.eigvalsh(_np.nan_to_num(I.reshape(-1, 2, 2))).reshape(I.shape[:-2] + (2,))
    big = ev[..., 1]
    med = _np.nanmedian(_np.where(_np.isfinite(I).all((-1, -2)), big, _np.nan))
    return _np.isfinite(I).all((-1, -2)) & (ev[..., 0] > big * 1e-4) & (big < 100 * med)


def _logistic(X, y, iters=30, l2=1e-3):
    """logistic regression by Newton (IRLS), X (n, p) -> beta (p,)."""
    beta = _np.zeros(X.shape[1])
    for _ in range(iters):
        p_ = 1 / (1 + _np.exp(-_np.clip(X @ beta, -30, 30)))
        W = p_ * (1 - p_) + 1e-9
        H = (X * W[:, None]).T @ X + l2 * _np.eye(X.shape[1])
        beta = beta + _np.linalg.solve(H, X.T @ (y - p_) - l2 * beta)
    return beta


def _wmedian(v, w):
    if len(v) == 0 or w.sum() <= 0:
        return 1.0
    o = _np.argsort(v)
    c = _np.cumsum(w[o])
    return float(v[o][_np.searchsorted(c, 0.5 * c[-1])])


def _fit_robust(D, I, O, P, th0, iters=6, ridge=1e-3, frozen=False):
    """robust (Cauchy) weighted least squares from th0; P (m,) prior weights; frozen: the rigid part (first 3
    parameters) stays at th0 and only the nodes are fitted -> th, kappa."""
    c = _np.ascontiguousarray
    return _fit_robust_core(c(D, dtype=_np.float64), c(I, dtype=_np.float64), c(O, dtype=_np.float64),
                            c(P, dtype=_np.float64), c(th0, dtype=_np.float64), iters, ridge, frozen)
