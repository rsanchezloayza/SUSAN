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

import numpy as _np

import susan.data as _ssa_data
import susan.io.ctf_stats as _ctf_stats
import susan.utils as _ssa_utils

__all__ = ['SolverCTF']

_CHI2_1_MEDIAN = 0.4549
_CHI2_2_MEDIAN = 1.3863


class SolverCTF:
    """CTF correction from the statistics of the CTF refiner (``output_type='refinement_stats'``).

    Per tomogram (and per half-set when the groups of the statistics are per half), every parameter estimated from
    the data:

    1. every group profile ``X(v)`` (``v = s^2 D(phi) / D_ref``) is fitted as ``A(v) CTF(v; D_ref + delta) W(v)``,
       ``A`` a smooth free amplitude (6 linear hats in the radius), the amplitude contrast phase fixed, the weights
       from the pooled noise ``sum |F|^2 / count``; ``delta`` on a grid, refined by a parabola;
    2. per-projection defocus offset from the 1x1 group (all the particles);
    3. global geometry (tilt angle offset, tilt axis rotation; depth patterns linearized at the 3x3 group centroids,
       the per-projection offset eliminated), robust least squares with per-projection weights from the half-split
       spread; used only when the half-split cross-validation (fit on one half, score on the other) beats the
       offset alone;
    4. per particle depth error ``dz`` along the beam (pattern ``R_k[2, 2]``): maximum over ``z`` of the sum over
       projections of the particle's own CC curves at the surface + ``R_k[2, 2] z``, shrunk by ``s^2 / (s^2 + noise^2)``
       from the even / odd projection split (0: nothing applied);
    5. per-projection translation bias from the imaginary part of the profiles, against its parity-split noise.  When
       refiner of :meth:`solve` has an offset search and the bias is significant, it is subtracted from ``prj_t`` of
       every particle of the projection; otherwise it is only reported.

    The correction (offset + geometry + ``R_k[2, 2] dz``) is added to ``def_U`` and ``def_V`` alike: the astigmatism
    and the phase shift are not refined.

    Usage: run the CTF refiner with ``output_type='refinement_stats'`` (the defocus span of the refiner is the
    range of the per particle curves), then :meth:`solve` gives the particles of the next round; with independent
    half-sets the groups are per half.

    Attributes
    ----------
    fit_range : float
        Range (Angstroms) of the profile fits, ``-fit_range .. fit_range``.  Default ``2000``.
    fit_step : float
        Grid step (Angstroms) of the profile fits.  Default ``5``.
    dz_range : float
        Range (Angstroms) of the particle depth error, ``-dz_range .. dz_range``.  Default ``600``.
    dz_step : float
        Grid step (Angstroms) of the particle depth error.  Default ``5``.
    shift_sigma : float
        A translation bias is significant when it exceeds this many standard deviations (of the parity-split noise,
        pooled over the projections).  Default ``3``.
    edge_fraction : float
        Summary: suggest a wider defocus range when more than this fraction of the per-tilt offsets lie beyond half
        the range of the per-particle curves.  Default ``0.1``.
    step_fraction : float
        Summary: suggest a smaller defocus step when the step exceeds this fraction of the half-maximum width of the
        mean per-particle CC curve.  Default ``0.1``.
    verbosity : int
        ``0`` silent, ``1`` a summary (corrections, convergence, suggested search changes), ``2`` also one line per
        tomogram and half.  Default ``1``.
    report : dict
        After :meth:`solve`, per (tomogram id, half-set) the diagnostics of the last solve: ``offset`` (K),
        ``geometry`` (tilt offset, axis rotation; deg), ``cv``, ``dz`` (per particle), ``shrink``, ``dz_sd``,
        ``shift_bias`` (K, 2; px, the error of ``prj_t``), ``shift_noise`` (K; px) and ``shift_applied`` (K).
    """

    def __init__(self):
        self.fit_range     = 2000.0
        self.fit_step      = 5.0
        self.dz_range      = 600.0
        self.dz_step       = 5.0
        self.shift_sigma   = 3.0
        self.edge_fraction = 0.1
        self.step_fraction = 0.1
        self.verbosity     = 1
        self.report        = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, ptcls_in, stats, tomos, refiner=None):
        """Particles for the next round.

        Parameters
        ----------
        ptcls_in : str or :class:`susan.data.Particles`
            The *input* particles of the refinement that wrote ``stats`` (the rows of the statistics).
        stats : str or dict
            The ``.ctfstats.npz`` (or the refiner output ``.ptclsraw``), or :func:`susan.io.ctf_stats.read` of it.
        tomos : str or :class:`susan.data.Tomograms`
        refiner : :class:`susan.modules.CtfRefiner`, optional
            The refiner of the iteration.  With an offset search (its translations are being refined) the significant
            translation bias of a projection (the error of its ``prj_t``) is subtracted from ``prj_t`` (Angstroms,
            detector frame) of all the particles of the projection; otherwise, or without it, it is only reported.
            With it the summary also suggests changes of its offset search.

        Returns
        -------
        :class:`susan.data.Particles`
            def_U, def_V = input + the correction (0 where not solved); prj_t = input - the significant translation
            bias when the refiner has an offset search.
        """
        P = _ssa_data.Particles(ptcls_in) if isinstance(ptcls_in, str) else ptcls_in.copy()
        S = _ctf_stats.read(stats) if isinstance(stats, str) else stats
        T = _ssa_data.Tomograms(tomos) if isinstance(tomos, str) else tomos
        if S['curves'].shape[0] != P.n_ptcl:
            raise ValueError('The statistics have %d particles, the particles %d: give the input of the refinement.' % (
                S['curves'].shape[0], P.n_ptcl))
        if 'group1/x' not in S or 'group3/x' not in S:
            raise ValueError('The statistics need the group sizes 1 and 3 (stats_groups of the CTF refiner).')
        halves = int(S['halves'])
        tid = P.tomo_id.flatten().astype(int)
        hid = P.half_id.flatten().astype(int)
        pos = P.pos().astype(_np.float64)
        offset_search = refiner is not None and max(refiner.offset.span[:2]) > 0
        def_in = P.def_U.copy()
        self.report = {}
        n_on, n_used = 0, 0
        shifts = []
        for t, ti in enumerate(S['tomo_id'].astype(int)):
            cix = int(T.get_cix(ti))
            K = int(T.num_proj[cix])
            for h in ((1, 2) if halves == 2 else (0,)):
                sel = _np.where((tid == ti) & ((hid == h) if h else True))[0]
                if len(sel) == 0:
                    continue
                if self.verbosity > 1:
                    print('    Tomogram %d%s:' % (ti, (', half %d' % h) if h else ''), end='', flush=True)
                r = self._solve_tomo(S, t, h, sel, T, cix, K, T.to_tomo(cix, pos[sel]))
                on = P.prj_w[sel, :K] > 0
                n_on += int(on.sum())
                n_used += int((on & r['solved'][None]).sum())
                P.def_U[sel, :K] += r['corr']
                P.def_V[sel, :K] += r['corr']
                apix = float(T.pix_size[cix])
                shifts.append(_np.linalg.norm(r['shift_bias'][r['shift_applied']], axis=-1) * apix)
                if offset_search:
                    for k in _np.where(r['shift_applied'])[0]:
                        i = sel[P.prj_w[sel, k] > 0]
                        P.prj_t[i, k] -= (r['shift_bias'][k] * apix).astype(P.prj_t.dtype)
                self.report[(ti, h)] = {k: r[k] for k in ('offset', 'offset_noise', 'geometry', 'cv', 'dz', 'shrink',
                                                          'dz_sd', 'shift_bias', 'shift_noise', 'shift_applied')}
        if self.verbosity:
            self._summary(P.def_U - def_in, P.prj_w > 0, _np.concatenate(shifts or [_np.zeros(0)]), S, refiner,
                          offset_search, 100 * n_used / max(n_on, 1))
        return P

    def _summary(self, dd, on, sb, S, refiner, offset_search, used):
        """Corrections applied to the dataset; [converged] when none is beyond its noise (rms per-tilt offset below twice
        the median parity-split sd, geometry rejected, depth shrinkage at most 1/2: the applied depth within its
        posterior sd, no significant shift bias); search changes suggested by the data (offsets beyond half the curve
        range, step against the width of the mean CC curve)."""
        rr = list(self.report.values())
        off = _np.concatenate([r['offset'] for r in rr])
        nv = _np.concatenate([r['offset_noise'] for r in rr])
        m = _np.isfinite(nv)
        off_sig = m.sum() > 2 and _math.sqrt(_np.mean(off[m] ** 2)) > 2 * _math.sqrt(_np.median(nv[m]))
        geo = [r['geometry'] for r in rr if r['cv'] > 0]
        depth = [r for r in rr if r['shrink'] > 0.5]
        dz = _np.concatenate([r['dz'] for r in rr])
        rms = lambda x: _math.sqrt(_np.mean(_np.square(x))) if _np.size(x) else 0.0
        tags = []
        if not (off_sig or geo or depth or sb.size):
            tags.append('[converged]')
        delta = S['delta'].astype(_np.float64)
        sug = []
        if off.size and _np.mean(_np.abs(off) > _np.abs(delta).max() / 2) > self.edge_fraction:
            sug.append('Increase defocus range')
        if refiner is not None and sb.size and not offset_search:
            sug.append('Increase offset range')
        with _np.errstate(all='ignore'):
            c = _np.nanmean(S['curves'].astype(_np.float64).reshape(-1, len(delta)), 0)
        c = c - _np.nanmin(c)
        if _np.isfinite(c).all() and c.max() > 0:
            half = delta[c >= c.max() / 2]
            if delta[1] - delta[0] > self.step_fraction * (half.max() - half.min() + delta[1] - delta[0]):
                sug.append('Reduce defocus step')
        if sug:
            tags.append('[Suggestion: %s.]' % ', '.join(sug))
        print('    Solving CTF.%s' % ''.join(' ' + t for t in tags))
        print('    Corrections: shifts %s. Using %.0f%% of projections.' % (
            ('%.2f Å rms in %d tilts%s' % (rms(sb), sb.size, '' if offset_search else ', not applied')) if sb.size else 'none', used))
        print('                 defocus %.0f Å rms (per tilt %.0f Å, depth %.0f Å), tilt geometry %s' % (
            rms(dd[on]), rms(off), rms(dz),
            ('%.2f° / axis %.2f°' % tuple(_np.abs(geo).max(0))) if geo else 'none'), flush=True)

    def tilt_amplitudes(self, stats, ptcls, tomos):
        """Per-projection CTF signal scale and relative amplitude B-factor.

        Per tomogram, half-set and projection, the 1x1 group profile is fitted as in :meth:`solve`; the smooth
        amplitude ``A(s)`` (the model divided by the CTF), with the exposure filter applied by the refiner divided
        out, is fitted by the weighted line ``ln A = ln a - B s^2 / 4`` over the fitted band.  The noise comes from
        the fits of the two index parities; ``B`` is shrunk toward the weighted median over the projections by the
        spread beyond the noise, and made relative (the median is 0).

        Parameters
        ----------
        stats : str or dict
            The ``.ctfstats.npz`` (or the refiner output ``.ptclsraw``), or :func:`susan.io.ctf_stats.read` of it.
        ptcls : str or :class:`susan.data.Particles`
            The particles of the statistics (rows; ``def_ExFl`` is the exposure filter of the refinement).
        tomos : str or :class:`susan.data.Tomograms`

        Returns
        -------
        dict
            Per (tomogram id, half-set; 0 when the statistics are not per half) a dict of arrays (K, NaN where not
            estimated): ``scale`` (a, normalized energy units), ``scale_noise`` (sd of ln a), ``B`` (Angstroms^2,
            relative, shrunk), ``B_noise`` (sd of the unshrunk estimate), ``B_raw`` (relative, unshrunk),
            ``exp_filt`` (the exposure filter divided out), and ``shrink`` (mean shrinkage factor).
        """
        P = _ssa_data.Particles(ptcls) if isinstance(ptcls, str) else ptcls
        S = _ctf_stats.read(stats) if isinstance(stats, str) else stats
        T = _ssa_data.Tomograms(tomos) if isinstance(tomos, str) else tomos
        if S['curves'].shape[0] != P.n_ptcl:
            raise ValueError('The statistics have %d particles, the particles %d.' % (S['curves'].shape[0], P.n_ptcl))
        halves = int(S['halves'])
        tid = P.tomo_id.flatten().astype(int)
        hid = P.half_id.flatten().astype(int)
        out = {}
        for t, ti in enumerate(S['tomo_id'].astype(int)):
            K = int(T.num_proj[int(T.get_cix(ti))])
            for h in ((1, 2) if halves == 2 else (0,)):
                sel = _np.where((tid == ti) & ((hid == h) if h else True))[0]
                if len(sel) == 0:
                    continue
                on = P.prj_w[sel, :K] > 0
                exf = _np.array([_np.median(P.def_ExFl[sel, k][on[:, k]]) if on[:, k].any() else 0.0 for k in range(K)])
                out[(ti, h)] = self._amplitudes_tomo(S, t, max(h - 1, 0), K, exf)
                if self.verbosity > 1:
                    r = out[(ti, h)]
                    print('    Tomogram %d%s: B %d projections, median noise %.0f A^2, shrinkage %.2f; ExpFilt %.1f-%.1f A^2' % (
                        ti, (', half %d' % h) if h else '', _np.isfinite(r['B']).sum(), _np.nanmedian(r['B_noise']),
                        r['shrink'], exf.min(), exf.max()), flush=True)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ctf_consts(kv, cs, ac):
        v = kv * 1e3
        lam = 12.2639 / _math.sqrt(v * (1 + 0.97845e-6 * v))
        return dict(LambdaPi=_math.pi * lam, CsLambda3PiH=lam ** 3 * cs * 1e7 * _math.pi / 2, AC=ac, CA=_math.sqrt(1 - ac * ac))

    def _fit_profile(self, x, w, s, c, v, dref, cst, DG, apix, NP, model=False, amp=False):
        """delta (A) of a profile (NaN without data), and the model m(v) = A(v) CTF(v) (per unit |P|^2) when model=True;
        with amp=True the amplitude A(v) and the CTF(v) of the fit."""
        ok = (w > 0) & (c > 0)
        if ok.sum() < 20 or dref <= 0:
            return (_np.nan, None, None) if amp else (_np.nan, None) if model else _np.nan
        x, w, v = x[ok], w[ok], v[ok]
        sig2 = s[ok] / c[ok]
        wt = 1 / _np.maximum(sig2 * w, 1e-30)
        r = _np.sqrt(v) * NP * apix
        kn = _np.linspace(r.min(), r.max(), 6)
        B = _np.clip(1 - _np.abs(r[None] - kn[:, None]) / (kn[1] - kn[0]), 0, None)            # (6, n)
        g = cst['LambdaPi'] * (dref + DG[:, None]) * v[None] - cst['CsLambda3PiH'] * v[None] ** 2
        ct = cst['CA'] * _np.sin(g) + cst['AC'] * _np.cos(g)                                    # (nD, n)
        M = B[None] * (ct * w[None])[:, None]                                                   # (nD, 6, n)
        Gm = _np.einsum('dip,djp->dij', M * wt, M) + 1e-12 * _np.eye(6)[None] * _np.einsum('dip,dip->d', M * wt, M)[:, None, None]
        beta = _np.linalg.solve(Gm, _np.einsum('dip,p->di', M * wt, x))
        res = _np.einsum('dp,p->d', (x[None] - _np.einsum('di,dip->dp', beta, M)) ** 2, wt)
        j = int(_np.argmin(res))
        d = DG[j]
        if 0 < j < len(DG) - 1:
            den = res[j - 1] - 2 * res[j] + res[j + 1]
            if den > 0:
                d = DG[j] + 0.5 * (res[j - 1] - res[j + 1]) / den * (DG[1] - DG[0])
        if not (model or amp):
            return d
        m = _np.full(ok.shape, _np.nan)
        m[ok] = (beta[j] @ B) * (1 if amp else ct[j])
        if amp:
            c = _np.full(ok.shape, _np.nan)
            c[ok] = ct[j]
            return d, m, c
        return d, m

    @staticmethod
    def _depth_patterns(eu_deg, p, eps=0.01):
        """d depth / d(tilt angle offset), d depth / d(axis rotation) (per deg) and R22, at the positions p (n, 3) -> (n, K) x 2, (K)."""
        e = _np.asarray(eu_deg, _np.float64)
        de = _np.zeros_like(e)
        de[:, 1] = eps
        R0 = _ssa_utils.euZYZ_to_rotm(e, degrees=True).astype(_np.float64)
        Rp = _ssa_utils.euZYZ_to_rotm(e + de, degrees=True).astype(_np.float64)
        Rm = _ssa_utils.euZYZ_to_rotm(e - de, degrees=True).astype(_np.float64)
        Z = _ssa_utils.euZYZ_to_rotm(_np.array([[eps, 0, 0], [-eps, 0, 0]]), degrees=True).astype(_np.float64)
        Jt = p @ ((Rp[:, 2] - Rm[:, 2]).T) / (2 * eps)
        Jp = p @ (((R0 @ Z[0])[:, 2] - (R0 @ Z[1])[:, 2]).T) / (2 * eps)
        return Jt, Jp, R0[:, 2, 2]

    @staticmethod
    def _geo_rows(d, okg, wk, Jt, Jp):
        rows, rhs, ww = [], [], []
        for k in range(d.shape[1]):
            u = okg[:, k]
            if u.sum() < 3:
                continue
            for g in _np.where(u)[0]:
                rows.append((Jt[g, k] - Jt[u, k].mean(), Jp[g, k] - Jp[u, k].mean()))
                rhs.append(d[g, k] - d[u, k].mean())
                ww.append(wk[k])
        return _np.array(rows).reshape(-1, 2), _np.array(rhs), _np.array(ww)

    @staticmethod
    def _geo_fit(A, b, w, iters=8):
        ab = _np.zeros(2)
        if len(b) < 3:
            return ab
        for _ in range(iters):
            r2 = (b - A @ ab) ** 2 * w
            sw = _np.sqrt(w / (1 + r2 / (4 * max(_np.median(r2) / _CHI2_1_MEDIAN, 1e-12))))
            ab, *_ = _np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        return ab

    @staticmethod
    def _cauchy(A, b, w, ab, scale):
        return float(_np.sum(_np.log1p((b - A @ ab) ** 2 * w / (4 * scale))))

    @staticmethod
    def _interp(C, q, delta):
        """Rows of C (n, ND) on the uniform grid delta, linearly interpolated at q (n, m); 0 outside."""
        u = (q - delta[0]) / (delta[1] - delta[0])
        i0 = _np.floor(u).astype(int)
        f = u - i0
        inside = (i0 >= 0) & (i0 < len(delta) - 1)
        i0 = _np.clip(i0, 0, len(delta) - 2)
        a = _np.take_along_axis(C, i0, 1)
        b = _np.take_along_axis(C, i0 + 1, 1)
        return _np.where(inside, a + f * (b - a), 0.0)

    def _shift_bias(self, S, t, hh, k, m, NP):
        """Translation bias (px) of projection k from the imaginary part, its parity-split noise (px) and the
        unit-noise scale of the estimate: Im sum ~ 2 pi / NP m(v) (dx wxx + dy wxy)."""
        sh = [S['shift/%s' % c][t, hh, :, k] for c in ('ix', 'iy', 'wxx', 'wyy', 'wxy')]      # (2, NB) each
        f = 2 * _np.pi / NP * _np.nan_to_num(m)

        def lsq(ix, iy, wxx, wyy, wxy):
            A2 = _np.concatenate([_np.stack([f * wxx, f * wxy], 1), _np.stack([f * wxy, f * wyy], 1)])
            b2 = _np.concatenate([ix, iy])
            okb = _np.isfinite(A2).all(1) & _np.isfinite(b2)
            if okb.sum() <= 10:
                return None, None
            return _np.linalg.lstsq(A2[okb], b2[okb], rcond=None)[0], A2[okb]

        d, A2 = lsq(*[z.sum(0) for z in sh])
        if d is None:
            return None
        d0, _ = lsq(*[z[0] for z in sh])
        d1, _ = lsq(*[z[1] for z in sh])
        try:
            cov = _np.trace(_np.linalg.inv(A2.T @ A2)) / 2
        except _np.linalg.LinAlgError:
            return None
        s2 = _np.nan if d0 is None or d1 is None else _np.sum((d0 - d1) ** 2) / 8
        return d, s2, cov

    def _solve_tomo(self, S, t, h, sel, T, cix, K, p):
        """One tomogram (and half-set): sel (n,) particle rows, p (n, 3) positions from the tomogram centre (A)."""
        n = len(sel)
        vmax, NB, NP = float(S['vmax']), int(S['nb']), int(S['box'])
        v = (_np.arange(NB) + 0.5) * vmax / NB
        DG = _np.arange(-self.fit_range, self.fit_range + 1e-6, self.fit_step)
        delta = S['delta'].astype(_np.float64)
        cst = self._ctf_consts(float(S['kv'][t]), float(S['cs'][t]), float(S['ac'][t]))
        apix = float(S['apix'][t])
        dref = S['dref'][t, :K].astype(_np.float64)
        eu = _np.asarray(T.proj_eZYZ[cix, :K], _np.float64)
        hh = max(h - 1, 0)
        Jt, Jp, R22 = self._depth_patterns(eu, p)

        def gfit(gs, l, par=None, model=False):
            o = hh * gs * gs
            a = [S['group%d/%s' % (gs, c)][t, o + l, :, :K] for c in 'xwsc']                 # (2, K, NB) each
            a = [(z.sum(0) if par is None else z[par]) for z in a]
            out, mods = _np.full(K, _np.nan), []
            for k in range(K):
                r = self._fit_profile(a[0][k], a[1][k], a[2][k], a[3][k], v, dref[k], cst, DG, apix, NP, model=model)
                if model:
                    out[k] = r[0]
                    mods.append(r[1])
                else:
                    out[k] = r
            return (out, mods) if model else out

        off, mods = gfit(1, 0, model=True)
        solved = _np.isfinite(off)
        off = _np.nan_to_num(off)
        off_noise = (gfit(1, 0, 0) - gfit(1, 0, 1)) ** 2 / 4
        lab3 = S['group3/label'][sel] - hh * 9
        cen = _np.array([p[lab3 == g].mean(0) if _np.any(lab3 == g) else _np.full(3, _np.nan) for g in range(9)])
        okc = _np.isfinite(cen[:, 0])
        Jtc, Jpc, _ = self._depth_patterns(eu, _np.nan_to_num(cen))
        dF = _np.array([gfit(3, g) for g in range(9)])
        dA = _np.array([gfit(3, g, 0) for g in range(9)])
        dB = _np.array([gfit(3, g, 1) for g in range(9)])
        dh = dA - dB
        good = _np.isfinite(dF) & _np.isfinite(dh) & okc[:, None]
        with _np.errstate(all='ignore'):
            m2 = _np.array([_np.median(dh[good[:, k], k] ** 2) if good[:, k].sum() >= 3 else _np.nan for k in range(K)])
        okg = good & (dh ** 2 <= 9 * _np.nan_to_num(m2, nan=0.0)[None])
        wk = 1 / _np.maximum(_np.nan_to_num(m2, nan=_np.inf) / 4, 1.0)
        cv = 0.0
        for tr, te in ((dA, dB), (dB, dA)):
            At, bt, wt = self._geo_rows(_np.nan_to_num(tr), okg, wk / 2, Jtc, Jpc)
            Ae, be, we = self._geo_rows(_np.nan_to_num(te), okg, wk / 2, Jtc, Jpc)
            if len(be) < 10:
                cv = -1.0
                break
            sc = max(_np.median(be ** 2 * we) / _CHI2_1_MEDIAN, 1e-12)
            cv += self._cauchy(Ae, be, we, _np.zeros(2), sc) - self._cauchy(Ae, be, we, self._geo_fit(At, bt, wt), sc)
        A, b, w = self._geo_rows(_np.nan_to_num(dF), okg, wk, Jtc, Jpc)
        ab = self._geo_fit(A, b, w) if cv > 0 else _np.zeros(2)
        gp = Jt * ab[0] + Jp * ab[1]
        surf = off[None] + gp - gp.mean(0, keepdims=True)                                   # (n, K)
        # per particle dz
        C = S['curves'][sel][:, :K].astype(_np.float64)
        ZG = _np.arange(-self.dz_range, self.dz_range + 1e-6, self.dz_step)
        SS = _np.zeros((2, n, len(ZG)))
        for k in range(K):
            ck = C[:, k]
            okp = _np.isfinite(ck[:, 0])
            if okp.any():
                SS[k % 2, okp] += self._interp(ck[okp], surf[okp, k][:, None] + R22[k] * ZG[None], delta)
        za, zb, zf = ZG[_np.argmax(SS[0], 1)], ZG[_np.argmax(SS[1], 1)], ZG[_np.argmax(SS.sum(0), 1)]
        s2, lam = 0.0, 0.0
        if n >= 4:
            s2 = max(_np.cov(za, zb)[0, 1], 0.0)
            v2 = _np.var(za - zb) / 4
            lam = s2 / (s2 + v2) if s2 + v2 > 0 else 0.0
        dz = lam * (zf - _np.median(zf))
        corr = surf + R22[None] * dz[:, None]
        # translation bias from Im, with the parity-split noise pooled over the projections
        sb, noise, applied = _np.full((K, 2), _np.nan), _np.full(K, _np.nan), _np.zeros(K, bool)
        est = [None if mods[k] is None else self._shift_bias(S, t, hh, k, mods[k], NP) for k in range(K)]
        unit = [e[1] / e[2] for e in est if e is not None and _np.isfinite(e[1])]
        u2 = _np.median(unit) / (_CHI2_2_MEDIAN / 2) if len(unit) >= 3 else _np.nan
        for k, e in enumerate(est):
            if e is None:
                continue
            sb[k] = e[0]
            noise[k] = _math.sqrt(u2 * e[2]) if _np.isfinite(u2) else _np.nan
            applied[k] = _np.isfinite(noise[k]) and _np.linalg.norm(e[0]) > self.shift_sigma * noise[k]
        if self.verbosity > 1:
            fb = sb[_np.isfinite(sb)]
            print(' %d particles; per-tilt offset median abs %.0f A (max %.0f); geometry CV gain %+.2f -> %s; '
                  'dz signal sd %.0f A, shrinkage %.2f; shift bias rms %.3f px (max %.3f), %d significant' % (
                      n, _np.median(_np.abs(off)), _np.abs(off).max(), cv,
                      ('tilt %+.2f deg, axis %+.2f deg' % tuple(ab)) if cv > 0 else 'offset only', _math.sqrt(s2), lam,
                      _np.sqrt(_np.mean(fb ** 2)) if fb.size else _np.nan, _np.abs(fb).max() if fb.size else _np.nan, applied.sum()), flush=True)
        return dict(corr=corr, offset=off, geometry=ab, cv=cv, dz=dz, shrink=lam, dz_sd=_math.sqrt(s2),
                    shift_bias=sb, shift_noise=noise, shift_applied=applied, solved=solved, offset_noise=off_noise)

    def _amplitudes_tomo(self, S, t, hh, K, exf):
        """tilt_amplitudes of one tomogram index t and half hh (0-based); exf (K,) the exposure filter of the data."""
        NB, NP = int(S['nb']), int(S['box'])
        v = (_np.arange(NB) + 0.5) * float(S['vmax']) / NB
        DG = _np.arange(-self.fit_range, self.fit_range + 1e-6, self.fit_step)
        cst = self._ctf_consts(float(S['kv'][t]), float(S['cs'][t]), float(S['ac'][t]))
        apix = float(S['apix'][t])
        dref = S['dref'][t, :K].astype(_np.float64)
        fits = _np.full((3, K, 4), _np.nan)                                                     # sum, parity 0, parity 1
        for k in range(K):
            a = [S['group1/%s' % c][t, hh, :, k] for c in 'xwsc']
            d = _np.nan
            for j, par in enumerate((None, 0, 1)):
                z = [q.sum(0) if par is None else q[par] for q in a]
                r = self._fit_profile(z[0], z[1], z[2], z[3], v, dref[k], cst, DG if par is None else _np.array([d]), apix, NP, amp=True)
                if par is None:
                    d = r[0]
                if r[1] is None:
                    break
                A, ct = r[1], r[2]
                m = _np.isfinite(A) & (A > 0) & (z[2] > 0) & (z[3] > 0) & (z[1] > 0)
                if m.sum() < 5:
                    break
                wl = z[1][m] * ct[m] ** 2 * A[m] ** 2 * z[3][m] / z[2][m]
                X = _np.stack([_np.ones(m.sum()), -v[m] / 4], 1)
                C = _np.linalg.inv(X.T @ (X * wl[:, None]))
                th = C @ (X.T @ (wl * (_np.log(A[m]) + v[m] * exf[k] / 4)))
                fits[j, k] = (th[0], th[1], C[1, 1], C[0, 0])
        lna, B, cB, cl = fits[0].T
        r = dict(scale=_np.full(K, _np.nan), scale_noise=_np.full(K, _np.nan), B=_np.full(K, _np.nan),
                 B_noise=_np.full(K, _np.nan), B_raw=_np.full(K, _np.nan), exp_filt=exf.astype(_np.float64), shrink=0.0)
        both = _np.isfinite(fits[:, :, 1]).all(0)
        if both.sum() < 3:
            return r
        u2B = _np.median(((fits[1, :, 1] - fits[2, :, 1]) ** 2 / 4)[both] / cB[both]) / _CHI2_1_MEDIAN
        u2a = _np.median(((fits[1, :, 0] - fits[2, :, 0]) ** 2 / 4)[both] / cl[both]) / _CHI2_1_MEDIAN
        ok = _np.isfinite(B) & (cB > 0)
        vB = u2B * cB[ok]
        med = _wmedian(B[ok], 1 / vB)
        tau2 = max(_np.mean((B[ok] - med) ** 2) - _np.mean(vB), 0.0)
        f = tau2 / (tau2 + vB)
        r['scale'][ok] = _np.exp(lna[ok])
        r['scale_noise'][ok] = _np.sqrt(u2a * cl[ok])
        r['B_raw'][ok] = B[ok] - med
        r['B'][ok] = f * (B[ok] - med)
        r['B_noise'][ok] = _np.sqrt(vB)
        r['shrink'] = float(_np.mean(f))
        return r


def _wmedian(v, w):
    o = _np.argsort(v)
    c = _np.cumsum(w[o])
    return float(v[o][_np.searchsorted(c, 0.5 * c[-1])])
