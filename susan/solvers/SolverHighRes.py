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
import warnings as _warnings

import numpy as _np

import susan.data as _ssa_data
import susan.io.ali_stats as _ali_stats
import susan.io.ctf_stats as _ctf_stats
from susan.solvers.SolverCTF import SolverCTF as _SolverCTF

__all__ = ['SolverHighRes']

_F = {f: i for i, f in enumerate(_ali_stats.FIELDS)}
_MRES_SIGMA = 0.187


class SolverHighRes:
    """Reconstruction particles from the latest statistics.

    The alignment solvers (:class:`Solver2D`, :class:`SolverCTF`) leave geometrically correct particles (poses, shifts,
    defocus) for the next alignment.  This solver only sets the reconstruction fields of a copy of them, which never
    feed the alignment, and selects the particles to reconstruct:

    1. ``prj_w``: the input ``prj_w`` times a per-tilt scale, normalized to mean 1 over all the valid projections (the
       differences between tomograms are kept).  The scale is the square of a per-tilt correlation (the SNR of a small
       correlation coefficient; the energy-normalized scores are comparable between projections).  Source
       ``weights='2d'``: the median over the particles of the CC above its baseline (the null, or the median second
       CC peak), from the 2D statistics; ``'ctf'``: the amplitude of the CTF signal of the projection
       (:meth:`SolverCTF.tilt_amplitudes`) with pooled CTF statistics, the same CC scale as ``'2d'`` (second-peak
       baseline) at the refined CTF with ``'refinement_covar'`` evidence.  EMPIAR-10064 against PDB 7O7Y: the squared scale +0.016 / +0.006
       map-to-model FSC at 50 / 77 fpix over the linear one (the half-map FSC ranks reweightings wrongly).  The projections the 2D solver did not solve get 0;
    2. ``def_Bfct`` = B_tilt + B_shift (Angstroms^2, the B-factor convention of the reconstruction,
       ``exp(-s^2 B / 4)``).  B_tilt (only with :attr:`b_tilt`): the amplitude B-factor of the projection relative to
       the median of its tomogram and half-set (:meth:`SolverCTF.tilt_amplitudes`, pooled CTF statistics); EMPIAR-10064: it
       trades the low against the high frequencies of the map-to-model FSC (no net gain at any gain factor).  B_shift = 8 pi^2 sigma^2, sigma^2 the
       variance of the shift (posterior covariance ``projections/post_cov``, mean of the two axes) of the latest
       shift estimate: the 2D statistics (Solver2D or ``'alignment_covar'``), or a later ``'refinement_covar'`` CTF
       refinement (it refines the shifts too; see ``latest``);
    3. ``def_mres`` from the angular uncertainty of the 2D solver (``projections/ang_var``, mean of the components
       that were estimated, in rad^2): the displacement of a point at the particle radius ``sigma_d = radius sigma_theta`` and
       its resolution ``sigma_d / 0.187`` (the frequency where ``exp(-2 pi^2 sigma_d^2 k^2) = 0.5``); 0 without the
       angular search.  For a splatting reconstruction it sets the kernel width; with KB gridding it is a hard
       cut-off: set ``set_max_res = False`` there;
    4. selection: with ``cc_threshold``, the fraction of the particles kept per class and half-set, by ali_cc.

    The statistics are recognized by their content (:func:`susan.io.ali_stats.kind`): ``stats_2d`` from
    ``'alignment_stats'`` (with Solver2D) or ``'alignment_covar'``; ``stats_ctf`` the pooled ``.ctfstats.npz`` of
    ``'refinement_stats'`` or the ``.stats.npz`` of ``'refinement_covar'`` (shift posterior and CC at the refined CTF;
    no B_tilt).  The defocus variance of ``'refinement_covar'`` (``projections/ctf_var``) is not used: the
    reconstruction has no defocus-error envelope.

    Poses, shifts and defocus are not changed.  Each statistics input is used only when its rows match the particle
    list (otherwise a warning); without it, the fields it feeds are not set (the weights, and def_mres, keep the
    input values; def_Bfct gets only the other part).

    Attributes
    ----------
    weights : str
        Source of the per-tilt scale, ``'2d'`` or ``'ctf'`` (without CTF statistics: ``'2d'``, with a warning).
        Default ``'2d'``.
    set_max_res : bool
        Write def_mres.  Default ``True``.
    b_tilt : bool
        Add B_tilt to def_Bfct.  Default ``False``.
    set_bfactor : bool
        Write def_Bfct (``False``: the input values are kept).  Default ``True``.
    verbosity : int
        ``0`` silent, ``1`` a summary line, ``2`` also the lines of :meth:`SolverCTF.tilt_amplitudes`.  Default ``0``.
    used : dict
        After :meth:`solve`, what was applied: ``weights`` (source), ``b_tilt``, ``b_shift`` (``'2d'``, ``'ctf'`` or
        ``False``) and ``max_res``.
    """

    def __init__(self):
        self.weights      = '2d'
        self.set_max_res  = True
        self.set_bfactor  = True
        self.b_tilt       = False
        self.verbosity    = 0
        self.used         = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, ptcls, stats_2d=None, stats_ctf=None, tomos=None, cc_threshold=None, null_baseline=None, radius=None,
              latest='2d'):
        """Reconstruction particles.

        Parameters
        ----------
        ptcls : str or :class:`susan.data.Particles`
            The solved particles (the output of the alignment solvers).
        stats_2d : str or dict, optional
            The ``.stats.npz`` of a 2D alignment (with ``projections/post_cov`` and ``projections/ang_var`` of
            :meth:`Solver2D.solve`), or :func:`susan.io.ali_stats.read` of it; its rows are the particles.
        stats_ctf : str or dict, optional
            The ``.ctfstats.npz`` of a ``'refinement_stats'`` CTF refinement (or :func:`susan.io.ctf_stats.read` of
            it), or the ``.stats.npz`` of a ``'refinement_covar'`` one with ``projections/post_cov`` (or
            :func:`susan.io.ali_stats.read` of it); its rows are the particles.
        tomos : str or :class:`susan.data.Tomograms`, optional
            Needed with pooled CTF statistics.
        cc_threshold : float, optional
            Fraction of the particles kept per class and half-set, by ali_cc.  ``None``: all the particles.
        null_baseline : array (n_ptcl, K), optional
            CC baseline without matching signal per particle and projection (from a null alignment), for
            ``weights='2d'``.  Default: one constant per half, the median height of the second CC peak, which matches
            the null baseline (it is the noise maximum of the search; EMPIAR-10064: within 1.3 % median of the
            null-based weights).
        radius : float, optional
            Particle radius (Angstroms), to convert the angular uncertainty to a displacement.  Without it def_mres
            is kept.
        latest : str
            ``'2d'`` or ``'ctf'``: the statistics of the later iteration; B_shift (and the projections left unsolved)
            come from it when it has a shift posterior.  Default ``'2d'``.

        Returns
        -------
        :class:`susan.data.Particles`
            The selected particles (a copy) with prj_w, def_Bfct and def_mres set.
        """
        P = _ssa_data.Particles(ptcls) if isinstance(ptcls, str) else ptcls.copy()
        if latest not in ('2d', 'ctf'):
            raise ValueError("latest must be '2d' or 'ctf'")
        S = self._read(stats_2d, _ali_stats, 'projections/stats', P.n_ptcl, 'The 2D statistics')
        if S is not None and _ali_stats.kind(S) == 'refinement_covar':
            raise ValueError('stats_2d: statistics of a CTF refinement (refinement_covar); pass them as stats_ctf.')
        C = self._read_ctf(stats_ctf, P.n_ptcl)
        E = None
        if C is not None and 'projections/covar' in C:
            E, C = C, None
            if 'projections/post_cov' not in E:
                _warnings.warn('The refinement_covar statistics have no projections/post_cov (made by '
                               'susan.io.ali_stats.ctf_posterior): not used.', RuntimeWarning, stacklevel=2)
                E = None
        if C is not None and tomos is None:
            raise ValueError('The tomograms are needed with the CTF statistics.')
        n, K = P.prj_w.shape
        tid = P.tomo_id.flatten().astype(int)
        hid = P.half_id.flatten().astype(int)
        src = self.weights
        if src not in ('2d', 'ctf'):
            raise ValueError("weights must be '2d' or 'ctf'")
        if src == 'ctf' and C is None and E is None:
            _warnings.warn("weights='ctf' without usable CTF statistics: using '2d'.", RuntimeWarning, stacklevel=2)
            src = '2d'
        if self.set_bfactor and self.b_tilt and C is None and E is not None:
            _warnings.warn('B_tilt needs the pooled CTF statistics (refinement_stats): skipped.', RuntimeWarning, stacklevel=2)
        self.used = dict(weights=None, b_tilt=False, b_shift=False, max_res=False)
        amp = {}
        if C is not None and (src == 'ctf' or (self.set_bfactor and self.b_tilt)):
            ctf = _SolverCTF()
            ctf.verbosity = 2 if self.verbosity > 1 else 0
            amp = ctf.tilt_amplitudes(C, P, tomos)
        T = E if (E is not None and (latest == 'ctf' or S is None)) else S
        post_cov = None if T is None else T.get('projections/post_cov')
        solved = None if post_cov is None else _np.isfinite(post_cov[:, :K, 0])
        f = None
        if src == 'ctf':
            f = self._scale_ctf(amp, tid, hid, n, K) if C is not None else self._scale_2d(P, E, None, tid, hid, n, K)
        elif S is not None:
            f = self._scale_2d(P, S, null_baseline, tid, hid, n, K)
        if f is not None:
            self.used['weights'] = src
            P.prj_w[:] = self._normalize(P.prj_w.astype(_np.float64), f, solved)
        elif solved is not None:
            P.prj_w[~solved] = 0
        if self.set_bfactor:
            B = _np.zeros((n, K))
            if amp:
                B += self._tilt_b(amp, tid, hid, n, K)
                self.used['b_tilt'] = True
            if T is not None:
                if post_cov is None:
                    _warnings.warn('The 2D statistics have no projections/post_cov (made by Solver2D.solve): B_shift '
                                   'skipped.', RuntimeWarning, stacklevel=2)
                else:
                    B += self._shift_b(post_cov[:, :K], T, tid, n)
                    self.used['b_shift'] = 'ctf' if T is E else '2d'
            if self.used['b_tilt'] or self.used['b_shift']:
                P.def_Bfct[:, :K] = B
        if self.set_max_res and S is not None:
            if 'projections/ang_var' not in S:
                _warnings.warn('The 2D statistics have no projections/ang_var (made by Solver2D.solve): max_res '
                               'skipped.', RuntimeWarning, stacklevel=2)
            elif radius is None:
                _warnings.warn('No particle radius: max_res skipped.', RuntimeWarning, stacklevel=2)
            else:
                with _np.errstate(invalid='ignore'), _warnings.catch_warnings():
                    _warnings.simplefilter('ignore', RuntimeWarning)
                    var = _np.nanmean(S['projections/ang_var'][:, :K].astype(_np.float64), -1) * _math.radians(1) ** 2
                    P.def_mres[:, :K] = _np.nan_to_num(radius * _np.sqrt(var) / _MRES_SIGMA)
                self.used['max_res'] = True
        if cc_threshold is not None:
            keep = hid > 0
            for i in range(P.n_refs):
                sel = (P.ref_cix == i).flatten()
                for h in (1, 2):
                    m = sel & (hid == h)
                    if m.any():
                        keep[m & (P.ali_cc[i] < _np.quantile(P.ali_cc[i, m], 1 - cc_threshold))] = False
            P = P[keep]
        if self.verbosity:
            print('    High resolution particles: %d, weights %s, B_tilt %s, B_shift %s, max_res %s.' % (
                P.n_ptcl, self.used['weights'] or 'input', 'yes' if self.used['b_tilt'] else 'no',
                self.used['b_shift'] or 'no', 'yes' if self.used['max_res'] else 'no'))
        return P

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read(stats, io, key, n, name):
        """The statistics as a dict, None when absent or when their rows are not the n particles."""
        if stats is None:
            return None
        S = io.read(stats) if isinstance(stats, str) else stats
        rows = S[key or ('depth' if 'depth' in S else 'curves')].shape[0]
        if rows != n:
            _warnings.warn('%s have %d rows, the particles %d: not used.' % (name, rows, n), RuntimeWarning, stacklevel=3)
            return None
        return S

    @classmethod
    def _read_ctf(cls, stats, n):
        """Pooled CTF statistics (``.ctfstats.npz``) or ``'refinement_covar'`` evidence (``.stats.npz``), by content."""
        if stats is None:
            return None
        if isinstance(stats, str):
            covar = not stats.endswith('.ctfstats.npz')
            stats = (_ali_stats if covar else _ctf_stats).read(stats)
        if 'projections/covar' in stats:
            if _ali_stats.kind(stats) != 'refinement_covar':
                raise ValueError('stats_ctf: statistics of a %s alignment, not of a CTF refinement.' % _ali_stats.kind(stats))
            return cls._read(stats, _ali_stats, 'projections/stats', n, 'The CTF statistics')
        return cls._read(stats, _ctf_stats, None, n, 'The CTF statistics')

    @staticmethod
    def _scale_2d(P, S, null_baseline, tid, hid, n, K):
        """(n, K) squared median of the CC above the baseline per (tomogram, tilt, half-set), NaN where not estimated."""
        st = S['projections/stats'][:, :K]
        cc, r2 = st[..., _F['cc']], st[..., _F['r2']]
        valid = _np.isfinite(cc) & (cc > 0)
        f = _np.full((n, K), _np.nan)
        for h in _np.unique(hid):
            idx = _np.where(hid == h)[0]
            if null_baseline is None:
                v = valid[idx] & _np.isfinite(r2[idx])
                b = _np.full((len(idx), K), _np.median((cc[idx] * r2[idx])[v]) if v.any() else 0.0)
            else:
                b = _np.asarray(null_baseline)[idx, :K]
            for t in _np.unique(tid[idx]):
                r = _np.where(tid[idx] == t)[0]
                for k in range(K):
                    g = r[valid[idx[r], k] & _np.isfinite(b[r, k])]
                    if len(g) >= 10:
                        f[idx[g], k] = max(_np.median(cc[idx[g], k] - b[g, k]), 0.0) ** 2
        return f

    @staticmethod
    def _lookup(amp, t, h):
        return amp.get((t, h), amp.get((t, 0)))

    def _scale_ctf(self, amp, tid, hid, n, K):
        """(n, K) squared amplitude of the CTF signal of the projection of each particle, NaN where not estimated."""
        f = _np.full((n, K), _np.nan)
        for t, h in set(zip(tid, hid)):
            r = self._lookup(amp, t, h)
            if r is not None:
                f[(tid == t) & (hid == h), :len(r['scale'])] = _np.square(r['scale'][:K])
        return f

    def _tilt_b(self, amp, tid, hid, n, K):
        """(n, K) relative B of the projection of each particle (0 where not estimated)."""
        B = _np.zeros((n, K))
        for t, h in set(zip(tid, hid)):
            r = self._lookup(amp, t, h)
            if r is not None:
                B[(tid == t) & (hid == h), :len(r['B'])] = _np.nan_to_num(r['B'][:K])
        return B

    @staticmethod
    def _shift_b(post_cov, S, tid, n):
        """(n, K) 8 pi^2 sigma^2 of the applied shift (Angstroms^2), 0 where it was not solved."""
        apix = _np.zeros(n)
        for t, a in zip(S['tomo_id'], S['apix']):
            apix[tid == int(t)] = a
        s2 = 0.5 * (post_cov[..., 0].astype(_np.float64) + post_cov[..., 2]) * apix[:, None] ** 2
        return _np.nan_to_num(8 * _math.pi ** 2 * s2)

    @staticmethod
    def _normalize(w, f, solved):
        """w times the scale f (negative: 0; NaN: the median), mean 1 over all the valid projections (the signal
        differences between tomograms are kept); 0 where unsolved."""
        on = w > 0
        f = _np.where(_np.isfinite(f), _np.maximum(f, 0), _np.nanmedian(f[on]) if _np.isfinite(f[on]).any() else 1.0)
        w = w * f / (max(_np.mean(f[on]), 1e-12) if on.any() else 1.0)
        if solved is not None:
            w[~solved] = 0
        return w
