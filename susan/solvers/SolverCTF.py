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
from scipy.linalg import cho_factor as _cho_factor, cho_solve as _cho_solve
from scipy.optimize import minimize as _minimize

import susan.data as _ssa_data
import susan.io.ctf_stats as _ctf_stats
import susan.utils as _ssa_utils

__all__ = ['SolverCTF']

_CHI2_1_MEDIAN = 0.4549
_MIN_BINS      = 20
_OUTLIER       = 9.0
_MIN_TILTS     = 4
_PRIOR_STARTS  = ((-1.0, -1.0, 1.0), (-3.0, 0.0, 0.0), (0.0, -4.0, -1.0))


def _spd_solve(G, b):
    """Solve G y = b for a batch of symmetric positive definite matrices G (n, m, m), b (n, m): Cholesky over the batch."""
    m = G.shape[1]
    L = _np.zeros_like(G)
    for j in range(m):
        d = G[:, j, j] - (L[:, j, :j] ** 2).sum(1)
        L[:, j, j] = _np.sqrt(_np.maximum(d, 1e-300))
        for i in range(j + 1, m):
            L[:, i, j] = (G[:, i, j] - (L[:, i, :j] * L[:, j, :j]).sum(1)) / L[:, j, j]
    y = _np.zeros_like(b)
    for i in range(m):
        y[:, i] = (b[:, i] - (L[:, i, :i] * y[:, :i]).sum(1)) / L[:, i, i]
    for i in range(m - 1, -1, -1):
        y[:, i] = (y[:, i] - (L[:, i + 1:, i] * y[:, i + 1:]).sum(1)) / L[:, i, i]
    return y


class SolverCTF:
    """CTF correction from the statistics of the CTF refiner (``output_type='refinement_stats'``).

    Per half-set (the halves solved independently when the groups of the statistics are per half), every parameter
    estimated from the data:

    1. per tomogram and projection, the 1x1 group profile ``X(v)`` (``v = s^2 D(phi) / D_ref``; the sum and the two
       index parities) is fitted as ``A(v) CTF(v; D_ref + delta) W(v)``, ``A`` a smooth free amplitude (6 linear hats in
       the radius), the amplitude contrast phase fixed, the weights from the pooled noise ``sum |F|^2 / count``;
       ``delta`` on a grid, refined by a parabola.  Its noise variance is ``u / curvature`` of the fit residual at the
       minimum, ``u`` pooled over the projections of the half from the parity split; a fit whose parity split exceeds
       9 times that keeps the parity variance;
    2. per-projection defocus offset: the posterior mean of the fits under a prior along the acquisition order, per
       tomogram a free mean + an AR(1) process over the dose order (``doses`` of the tomograms, else ``|tilt|``) with
       variance ``a + b (tilt / max tilt)^2``; ``a``, ``b`` and the correlation by restricted maximum likelihood over
       the tomograms of the half;
    3. per particle depth error ``dz`` along the beam (pattern ``R_k[2, 2]``): maximum over ``z`` (the ``delta`` grid)
       of the particle's depth curve, the sum over projections of its own CC at ``R_k[2, 2] z`` written by the refiner
       (the offsets of step 2 are applied in the next round), shrunk by ``s^2 / (s^2 + noise^2)`` from the even / odd
       projection split (0: nothing applied).

    The correction (offset + ``R_k[2, 2] dz``) is added to ``def_U`` and ``def_V`` alike: the astigmatism and the phase
    shift are not refined.

    Usage: run the CTF refiner with ``output_type='refinement_stats'`` (the defocus span of the refiner is the range of
    the per particle depth curves), then :meth:`solve` gives the particles of the next round; with independent half-sets the
    groups are per half.

    Attributes
    ----------
    fit_range : float
        Range (Angstroms) of the profile fits, ``-fit_range .. fit_range``.  Default ``2000``.
    fit_step : float
        Grid step (Angstroms) of the profile fits.  Default ``5``.
    edge_fraction : float
        Summary: suggest a wider defocus range when more than this fraction of the per-tilt fits lie beyond half the
        range of the per-particle depth curves.  Default ``0.1``.
    step_fraction : float
        Summary: suggest a smaller defocus step when the step exceeds this fraction of the half-maximum width of the
        mean per-particle depth curve.  Default ``0.1``.
    verbosity : int
        ``0`` silent, ``1`` a summary (corrections, convergence, suggested search changes), ``2`` also one line per
        half-set and per tomogram.  Default ``1``.
    report : dict
        After :meth:`solve`, per (tomogram id, half-set) the diagnostics of the last solve (Angstroms): ``offset`` (K,
        applied), ``offset_raw`` (K, the fit), ``offset_noise`` (K, sd of the fit), ``offset_sd`` (K, posterior sd),
        ``prior`` (a, b, correlation of the half), ``dz`` (per particle), ``shrink`` and ``dz_sd``.
    """

    def __init__(self):
        self.fit_range     = 2000.0
        self.fit_step      = 5.0
        self.edge_fraction = 0.1
        self.step_fraction = 0.1
        self.verbosity     = 1
        self.report        = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, ptcls_in, stats, tomos):
        """Particles for the next round.

        Parameters
        ----------
        ptcls_in : str or :class:`susan.data.Particles`
            The *input* particles of the refinement that wrote ``stats`` (the rows of the statistics).
        stats : str or dict
            The ``.ctfstats.npz`` (or the refiner output ``.ptclsraw``), or :func:`susan.io.ctf_stats.read` of it.
        tomos : str or :class:`susan.data.Tomograms`

        Returns
        -------
        :class:`susan.data.Particles`
            def_U, def_V = input + the correction.
        """
        P = _ssa_data.Particles(ptcls_in) if isinstance(ptcls_in, str) else ptcls_in.copy()
        S = _ctf_stats.read(stats) if isinstance(stats, str) else stats
        T = _ssa_data.Tomograms(tomos) if isinstance(tomos, str) else tomos
        if _rows(S) != P.n_ptcl:
            raise ValueError('The statistics have %d particles, the particles %d: give the input of the refinement.' % (
                _rows(S), P.n_ptcl))
        if 'group1/x' not in S:
            raise ValueError('The statistics need the group size 1 (stats_groups of the CTF refiner).')
        halves = int(S['halves'])
        tid = P.tomo_id.flatten().astype(int)
        hid = P.half_id.flatten().astype(int)
        def_in = P.def_U.copy()
        self.report = {}
        n_on, n_used = 0, 0
        hs = (1, 2) if halves == 2 else (0,)
        by_half = {h: [] for h in hs}
        for t, ti in enumerate(S['tomo_id'].astype(int)):
            sels = {h: _np.where((tid == ti) & ((hid == h) if h else True))[0] for h in hs}
            for h, f in self._fit_tomo(S, t, {h: q for h, q in sels.items() if len(q)}, T, int(T.get_cix(ti))).items():
                by_half[h].append(f)
        for h, fits in by_half.items():
            if not fits:
                continue
            prior = self._pool(fits)
            if self.verbosity > 1 and prior is not None:
                print('    Half %d: offset prior sd %.0f A (zero tilt) to %.0f A (max tilt), correlation %.2f' % (
                    h, _math.sqrt(prior[0]), _math.sqrt(prior[0] + prior[1]), prior[2]), flush=True)
            for f in fits:
                r = self._finish(S, f, prior)
                sel, K = f['sel'], f['K']
                on = P.prj_w[sel, :K] > 0
                n_on += int(on.sum())
                n_used += int((on & r['solved'][None]).sum())
                P.def_U[sel, :K] += r['corr']
                P.def_V[sel, :K] += r['corr']
                self.report[(f['ti'], h)] = {k: r[k] for k in r if k not in ('corr', 'solved')}
                if self.verbosity > 1:
                    ok = r['solved']
                    print('    Tomogram %d%s: %d particles; per-tilt offset %.0f A rms (fit %.0f A, posterior sd %.0f A); '
                          'dz signal sd %.0f A, shrinkage %.2f' % (
                              f['ti'], (', half %d' % h) if h else '', len(sel), _rms(r['offset'][ok]),
                              _rms(r['offset_raw'][ok]), _np.nanmedian(r['offset_sd']) if ok.any() else _np.nan,
                              r['dz_sd'], r['shrink']), flush=True)
        if self.verbosity:
            self._summary(P.def_U - def_in, P.prj_w > 0, S, 100 * n_used / max(n_on, 1))
        return P

    def _summary(self, dd, on, S, used):
        """Corrections applied to the dataset; [converged] when none is beyond its noise (per-tilt offsets within their
        posterior sd, depth shrinkage at most 1/2: the applied depth within its posterior sd); search changes suggested
        by the data (fits beyond half the curve range, step against the width of the mean CC curve)."""
        rr = list(self.report.values())
        off = _np.concatenate([r['offset'] for r in rr])
        raw = _np.concatenate([r['offset_raw'] for r in rr])
        sd = _np.concatenate([r['offset_sd'] for r in rr])
        dz = _np.concatenate([r['dz'] for r in rr])
        m = _np.isfinite(sd)
        tags = []
        if _rms(off[m]) <= _rms(sd[m]) and all(r['shrink'] <= 0.5 for r in rr):
            tags.append('[converged]')
        delta = S['delta'].astype(_np.float64)
        sug = []
        raw = raw[_np.isfinite(raw)]
        if raw.size and _np.mean(_np.abs(raw) > _np.abs(delta).max() / 2) > self.edge_fraction:
            sug.append('Increase defocus range')
        with _np.errstate(all='ignore'):
            c = (_np.nanmean(S['depth'].reshape(-1, len(delta)), 0) if 'depth' in S else
                 _np.nanmean(S['curves'].reshape(-1, len(delta)), 0)).astype(_np.float64)
        c = c - _np.nanmin(c)
        if _np.isfinite(c).all() and c.max() > 0:
            half = delta[c >= c.max() / 2]
            if delta[1] - delta[0] > self.step_fraction * (half.max() - half.min() + delta[1] - delta[0]):
                sug.append('Reduce defocus step')
        if sug:
            tags.append('[Suggestion: %s.]' % ', '.join(sug))
        print('    Solving CTF.%s' % ''.join(' ' + t for t in tags))
        print('    Corrections: defocus %.0f Å rms (per tilt %.0f Å, posterior sd %.0f Å; depth %.0f Å). '
              'Using %.0f%% of projections.' % (_rms(dd[on]), _rms(off[m]), _rms(sd[m]), _rms(dz), used), flush=True)

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
        if _rows(S) != P.n_ptcl:
            raise ValueError('The statistics have %d particles, the particles %d.' % (_rows(S), P.n_ptcl))
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

    @staticmethod
    def _ctf_table(v, dref, cst, DG):
        """CTF(v) at the defocus dref + DG and its square, (nD, len(v)) each."""
        ct = _np.sin(cst['LambdaPi'] * (dref + DG[:, None]) * v[None] - cst['CsLambda3PiH'] * v[None] ** 2 + _math.asin(cst['AC']))
        return ct, ct * ct

    @staticmethod
    def _residuals(x, w, s, c, v, apix, NP, ct, ct2):
        """Residual (m, nD) of the profiles x, w, s, c (m, nb) fitted as (beta B) CTF w over the table ct, ct2 (nD, nb)
        (NaN for a profile with fewer than _MIN_BINS valid bins), the amplitudes beta (m, nD, 6), the hats B (m, 6, nb)
        and the valid bins (m, nb)."""
        m, nb = x.shape
        nD = ct.shape[0]
        ok = (w > 0) & (c > 0)
        valid = ok.sum(1) >= _MIN_BINS
        wt = _np.zeros_like(x)
        wt[ok] = 1 / _np.maximum(s[ok] / c[ok] * w[ok], 1e-30)
        xo = _np.where(ok, x, 0)
        r = _np.sqrt(v) * NP * apix
        lo = _np.where(valid, _np.where(ok, r, _np.inf).min(1), 0)
        st = _np.where(valid, (_np.where(ok, r, -_np.inf).max(1) - lo) / 5, 1)
        st = _np.maximum(st, 1e-12)
        B = _np.clip(1 - _np.abs(r - (lo[:, None] + st[:, None] * _np.arange(6))[..., None]) / st[:, None, None], 0, None)
        BB = (B[:, :, None] * B[:, None]).reshape(m, 36, nb) * (w * w * wt)[:, None]
        G = (ct2 @ BB.reshape(m * 36, nb).T).reshape(nD, m, 6, 6).transpose(1, 0, 2, 3)
        b = (ct @ (B * (w * wt * xo)[:, None]).reshape(m * 6, nb).T).reshape(nD, m, 6).transpose(1, 0, 2)
        G = G + 1e-12 * _np.trace(G, axis1=2, axis2=3)[..., None, None] * _np.eye(6)
        beta = _spd_solve(G.reshape(-1, 6, 6), b.reshape(-1, 6)).reshape(m, nD, 6)
        res = (wt * xo * xo).sum(1)[:, None] - (beta * b).sum(2)
        res[~valid] = _np.nan
        return res, beta, B, ok

    @staticmethod
    def _minimum(res, DG):
        """Minimum of the residuals res (m, nD) on the grid DG refined by a parabola, and the curvature there (NaN at
        the edge of the grid or without data)."""
        d, cu = _np.full(len(res), _np.nan), _np.full(len(res), _np.nan)
        ok = _np.isfinite(res).all(1)
        if not ok.any():
            return d, cu
        r = res[ok]
        j = _np.argmin(r, 1)
        if len(DG) < 3:
            d[ok] = DG[j]
            return d, cu
        jc = _np.clip(j, 1, len(DG) - 2)
        i = _np.arange(len(j))
        a, b, c = r[i, jc - 1], r[i, jc], r[i, jc + 1]
        den = a - 2 * b + c
        inner = (j == jc) & (den > 0)
        st = DG[1] - DG[0]
        den = _np.where(inner, den, 1.0)
        d[ok] = DG[j] + _np.where(inner, 0.5 * (a - c) / den * st, 0.0)
        cu[ok] = _np.where(inner, den / st ** 2, _np.nan)
        return d, cu

    def _fit_profile(self, x, w, s, c, v, dref, cst, DG, apix, NP):
        """delta (A) of one profile on the grid DG (NaN without data), its amplitude A(v) and CTF(v) at the grid minimum
        (NaN outside the valid bins; None without data)."""
        if not dref > 0:
            return _np.nan, None, None
        ct, ct2 = self._ctf_table(v, dref, cst, DG)
        res, beta, B, ok = self._residuals(x[None], w[None], s[None], c[None], v, apix, NP, ct, ct2)
        if not _np.isfinite(res).all():
            return _np.nan, None, None
        d = self._minimum(res, DG)[0][0]
        j = int(_np.argmin(res[0]))
        ok = ok[0]
        A, cc = _np.full(len(x), _np.nan), _np.full(len(x), _np.nan)
        A[ok] = beta[0, j] @ B[0][:, ok]
        cc[ok] = ct[j, ok]
        return d, A, cc

    def _fit_tomo(self, S, t, sels, T, cix):
        """The 1x1 fits of tomogram index t per half h of sels (h: particle rows): delta and curvature (3, K) of the sum
        and the parities."""
        if not sels:
            return {}
        K = int(T.num_proj[cix])
        NB, NP = int(S['nb']), int(S['box'])
        v = (_np.arange(NB) + 0.5) * float(S['vmax']) / NB
        DG = _np.arange(-self.fit_range, self.fit_range + 1e-6, self.fit_step)
        cst = self._ctf_consts(float(S['kv'][t]), float(S['cs'][t]), float(S['ac'][t]))
        apix = float(S['apix'][t])
        dref = S['dref'][t, :K].astype(_np.float64)
        hh = [max(h - 1, 0) for h in sels]
        a = [S['group1/%s' % c][t, hh, :, :K].astype(_np.float64) for c in 'xwsc']             # (H, 2, K, NB) each
        a = [_np.stack([q.sum(1), q[:, 0], q[:, 1]], 1).reshape(-1, K, NB) for q in a]          # (H * 3, K, NB)
        d, cu = _np.full((len(hh) * 3, K), _np.nan), _np.full((len(hh) * 3, K), _np.nan)
        for k in range(K):
            if dref[k] > 0:
                d[:, k], cu[:, k] = self._minimum(self._residuals(*[q[:, k] for q in a], v, apix, NP,
                                                                  *self._ctf_table(v, dref[k], cst, DG))[0], DG)
        eu = _np.asarray(T.proj_eZYZ[cix, :K], _np.float64)
        tilt = eu[:, 1]
        dose = _np.asarray(T.doses[cix, :K], _np.float64)
        key = dose if _np.ptp(dose) > 0 else _np.abs(tilt) - 1e-3 * _np.sign(tilt)
        order = _np.argsort(_np.argsort(key, kind='stable')).astype(_np.float64)
        u2, lag = (tilt / max(_np.abs(tilt).max(), 1e-9)) ** 2, _np.abs(order[:, None] - order[None])
        R22 = _ssa_utils.euZYZ_to_rotm(eu, degrees=True)[:, 2, 2].astype(_np.float64)
        return {h: dict(t=t, ti=int(S['tomo_id'][t]), sel=sel, K=K, d=d[3 * i:3 * i + 3], cu=cu[3 * i:3 * i + 3], u2=u2,
                        lag=lag, R22=R22) for i, (h, sel) in enumerate(sels.items())}

    def _pool(self, fits):
        """Noise of the fits (pooled unit of the half) and the offset prior (a, b, correlation; None without data)."""
        d = _np.concatenate([f['d'] for f in fits], 1)
        q = (d[1] - d[2]) ** 2 / 4 * _np.concatenate([f['cu'][0] for f in fits])
        q = q[_np.isfinite(q)]
        u = _np.median(q) / _CHI2_1_MEDIAN if q.size >= 3 else _np.nan
        data = []
        for f in fits:
            par = (f['d'][1] - f['d'][2]) ** 2 / 4
            with _np.errstate(all='ignore'):
                n = u / f['cu'][0]
                n = _np.where(par > _OUTLIER * n, par, n)
            f['n'] = _np.where(_np.isfinite(n) | ~_np.isfinite(f['d'][0]), n, par + self.fit_range ** 2 / 3)
            ok = _np.isfinite(f['d'][0]) & _np.isfinite(f['n'])
            if ok.sum() >= _MIN_TILTS:
                data.append((f['d'][0, ok], f['n'][ok], f['u2'][ok], f['lag'][_np.ix_(ok, ok)]))
        if not data:
            return None
        s0 = max(_np.median(_np.concatenate([y - _np.median(y) for y, *_ in data]) ** 2), 1.0)

        def nll(q):
            th = _prior_of(q, s0)
            return sum(_gls(th, *x)[0] for x in data)

        best = min((_minimize(nll, q0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
                    for q0 in _PRIOR_STARTS), key=lambda r: r.fun)
        return _prior_of(best.x, s0)

    def _finish(self, S, f, prior):
        """Offsets (posterior), per particle depth and the correction (n, K) of one fit of :meth:`_fit_tomo`."""
        K, sel, R22 = f['K'], f['sel'], f['R22']
        y, N = f['d'][0], f['n']
        ok = _np.isfinite(y) & _np.isfinite(N)
        off, sd = _np.zeros(K), _np.full(K, _np.nan)
        if prior is not None and ok.sum() >= _MIN_TILTS:
            off, sd = _posterior(prior, y, N, ok, f['u2'], f['lag'])
        n = len(sel)
        ZG = S['delta'].astype(_np.float64)
        SS = _depth(S, sel, K, off, R22)
        za, zb, zf = ZG[_np.argmax(SS[0], 1)], ZG[_np.argmax(SS[1], 1)], ZG[_np.argmax(SS.sum(0), 1)]
        s2, lam = 0.0, 0.0
        if n >= 4:
            s2 = max(_np.cov(za, zb)[0, 1], 0.0)
            v2 = _np.var(za - zb) / 4
            lam = s2 / (s2 + v2) if s2 + v2 > 0 else 0.0
        dz = lam * (zf - _np.median(zf))
        return dict(corr=off[None] + R22[None] * dz[:, None], solved=ok, offset=off, offset_raw=y, offset_noise=_np.sqrt(N),
                    offset_sd=sd, prior=prior, dz=dz, shrink=lam, dz_sd=_math.sqrt(s2))

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
            a = [S['group1/%s' % c][t, hh, :, k].astype(_np.float64) for c in 'xwsc']
            d = _np.nan
            for j, par in enumerate((None, 0, 1)):
                z = [q.sum(0) if par is None else q[par] for q in a]
                r = self._fit_profile(z[0], z[1], z[2], z[3], v, dref[k], cst, DG if par is None else _np.array([d]), apix, NP)
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


def _rows(S):
    return S['depth' if 'depth' in S else 'curves'].shape[0]


def _depth(S, sel, K, off, R22):
    """Even / odd projection sums (2, n, ND) of the CC curves of the particles sel at D_ref + R22 z, z on the delta grid:
    the refiner's depth curves (written at offset 0), or from the per-projection curves at the offsets off."""
    if 'depth' in S:
        return _np.nan_to_num(S['depth'][sel].transpose(1, 0, 2).astype(_np.float32))
    delta = S['delta'].astype(_np.float64)
    C = _np.nan_to_num(S['curves'][sel, :K], copy=False)
    SS = _np.zeros((2, len(sel), len(delta)), _np.float32)
    for k in range(K):
        u = (off[k] + R22[k] * delta - delta[0]) / (delta[1] - delta[0])
        i0 = _np.floor(u).astype(int)
        inside = (i0 >= 0) & (i0 < len(delta) - 1)
        i0 = _np.clip(i0, 0, len(delta) - 2)
        fr = _np.where(inside, u - i0, 0.0).astype(_np.float32)
        wa = _np.where(inside, 1 - fr, 0.0).astype(_np.float32)
        SS[k % 2] += C[:, k, i0] * wa + C[:, k, i0 + 1] * fr
    return SS


def _rms(x):
    return _math.sqrt(_np.mean(_np.square(x))) if _np.size(x) else 0.0


def _prior_of(q, s0):
    """(a, b, correlation) of the optimizer variables (log a / s0, log b / s0, logit correlation)."""
    q = _np.clip(q, -20, 20)
    return _math.exp(q[0]) * s0, _math.exp(q[1]) * s0, 1 / (1 + _math.exp(-q[2]))


def _prior_cov(th, u2, lag):
    sd = _np.sqrt(th[0] + th[1] * u2)
    return sd[:, None] * sd[None] * th[2] ** lag


def _gls(th, y, N, u2, lag):
    """Restricted negative log likelihood of y = mean + o + e, o ~ N(0, prior), e ~ N(0, diag N), flat mean; and the
    factor, the mean and Sigma^-1 (1, r)."""
    cf = _cho_factor(_prior_cov(th, u2, lag) + _np.diag(N), lower=True, check_finite=False)
    Si1, Siy = _cho_solve(cf, _np.ones_like(y), check_finite=False), _cho_solve(cf, y, check_finite=False)
    A = Si1.sum()
    mu = Siy.sum() / A
    Sir = Siy - mu * Si1
    return _np.log(_np.diag(cf[0])).sum() + 0.5 * (y - mu) @ Sir + 0.5 * _math.log(A), cf, mu, Si1, Sir, A


def _posterior(th, y, N, ok, u2, lag):
    """Posterior mean and sd (K) of the offsets (mean estimated) from the fits y (K) with noise N at ok."""
    _, cf, mu, Si1, Sir, A = _gls(th, y[ok], N[ok], u2[ok], lag[_np.ix_(ok, ok)])
    Co = _prior_cov(th, u2, lag)[:, ok]                                                         # (K, n_ok)
    H = 1 - Co @ Si1
    var = th[0] + th[1] * u2 - _np.einsum('ij,ji->i', Co, _cho_solve(cf, Co.T, check_finite=False)) + H * H / A
    return mu + Co @ Sir, _np.sqrt(_np.maximum(var, 0))


def _wmedian(v, w):
    o = _np.argsort(v)
    c = _np.cumsum(w[o])
    return float(v[o][_np.searchsorted(c, 0.5 * c[-1])])
