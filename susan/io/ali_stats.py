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

"""2D alignment statistics: the ``<output stem>.stats.npz`` written by the aligner with
``output_type='alignment_stats'`` or ``'alignment_covar'`` (:class:`susan.modules.Aligner`).

Keys (numpy arrays):

``projections/stats`` (n_ptcl, K, NF)
    Per particle (rows of the aligner's input particles) and projection, the fields :data:`FIELDS` of the raw CC
    (no projection weight, no priors) over the offsets at the input orientation; NaN where undefined.
``projections/fields`` (NF,)
    The field names.
``grouped<gs>/stats`` (n_tomo, G, K, NF)
    The same fields for the mean CC map of each group at the input orientation; groups: gs x gs equal-count bins of the particle x, y per tomogram, G = halves * gs * gs.
    With ``halves`` 2 (aligned with independent half-sets) the bins are drawn per half and the groups of half h are
    ``(h-1)*gs*gs .. h*gs*gs - 1``: no group mixes the halves.
``grouped<gs>/label`` (n_ptcl,)
    Group of each particle (-1: none), an index into G.
``grouped<gs>/half`` (n_tomo, G, K, 2)
    Peak (x, y) difference between two halves of the particles of each group (particle index parity; a consistency
    check, unrelated to the half-sets).
``angles`` (A, 3)
    The sequential angular search: rotation vectors (degrees, detector frame, applied on the left of the projection
    orientation, as ``prj_eu``) of the input orientation (first, zero), the cone points and the in-plane points.
``grouped<gs>/ang_cc`` (n_tomo, G, K, A), ``grouped<gs>/ang_half`` (n_tomo, G, K, A, 2)
    Groups up to 2 x 2 with more than one angle: peak of the group's mean CC map at every angle, and of the two
    halves of its particles (particle index parity).
``halves`` ()
    1, or 2 when the groups are per half-set.
``tomo_id`` (n_tomo,), ``apix`` (n_tomo,), ``off_step`` ()
    Tomograms, their pixel size and the offset step (pixels).

With ``'alignment_covar'`` there are no groups and one angle; ``projections/stats`` is at the solution of the search
(the best orientation of each projection, offsets from the search centre), and:

``projections/covar`` (n_ptcl, K, NC)
    The fields :data:`COVAR_FIELDS`: ``lambda``, the CC to log-likelihood factor (1 / kappa, kappa = A / sd_cc^2 from
    the CC noise ring, as the aligner priors), and the curvature (CC / deg^2) of the best CC over the offsets about
    detector x, y and z at the solution (NaN for an axis that was not searched).
``covar_steps`` (3,)
    The angular steps (degrees) of those curvatures; 0 for an axis that was not searched.

:func:`posterior` turns them into ``projections/post_cov`` and ``projections/ang_var`` (also written by
:meth:`susan.solvers.Solver2D.solve`), which :class:`susan.solvers.SolverHighRes` uses.

The CTF refiner with ``output_type='refinement_covar'`` writes the same layout (``projections/stats`` at the refined
CTF), with the fields :data:`CTF_COVAR_FIELDS`: ``lambda`` and the curvature of the best CC over the offsets along the
mean defocus (U = V, CC / A^2), the astigmatism (U = -V, CC / A^2), the astigmatism angle (CC / deg^2) and the phase
shift (CC / deg^2); ``covar_steps`` (4,): A, A, deg, deg.  :func:`kind` tells the files apart, :func:`ctf_posterior`
turns them into ``projections/post_cov`` and ``projections/ctf_var``.
"""

__all__ = ['FIELDS', 'COVAR_FIELDS', 'CTF_COVAR_FIELDS', 'path', 'read', 'kind', 'posterior', 'ctf_posterior', 'append']

import numpy as _np

FIELDS = ('cc', 'px', 'py', 'z_peak', 'h_xx', 'h_xy', 'h_yy', 'r2')
COVAR_FIELDS = ('lambda', 'curv_x', 'curv_y', 'curv_z')
CTF_COVAR_FIELDS = ('lambda', 'curv_def', 'curv_ast', 'curv_ang', 'curv_ph')


def path(ptcls_out):
    """Statistics file of an aligner output: the ``.ptclsraw`` extension replaced by ``.stats.npz``."""
    base = ptcls_out[:-len('.ptclsraw')] if ptcls_out.endswith('.ptclsraw') else ptcls_out
    return base + '.stats.npz'


def read(filename):
    """Load a ``.stats.npz`` (or the statistics of a ``.ptclsraw`` output) into a dict of arrays (keys above); the
    stats of a file with other fields (older aligners) are reordered to :data:`FIELDS`."""
    if filename.endswith('.ptclsraw'):
        filename = path(filename)
    with _np.load(filename) as f:
        S = {k: f[k] for k in f.files}
    names = tuple(str(v) for v in S.get('projections/fields', FIELDS))
    if names != FIELDS:
        idx = [names.index(n) for n in FIELDS]
        for k in [k for k in S if k == 'projections/stats' or (k.startswith('grouped') and k.endswith('/stats'))]:
            S[k] = S[k][..., idx]
        S['projections/fields'] = _np.array(FIELDS)
    return S


def kind(S):
    """``'alignment_stats'``, ``'alignment_covar'`` or ``'refinement_covar'`` (the output that wrote ``S``)."""
    if 'projections/covar' not in S:
        return 'alignment_stats'
    names = tuple(str(v) for v in S.get('projections/covar_fields', COVAR_FIELDS))
    return 'refinement_covar' if names == CTF_COVAR_FIELDS else 'alignment_covar'


def _shift_cov(S, offset_sigma):
    """(n_ptcl, K, 3) [xx, xy, yy] posterior covariance (px^2) of the shift, and lambda."""
    st = S['projections/stats'].astype(_np.float64)
    lam = S['projections/covar'][..., 0].astype(_np.float64)
    f = {n: i for i, n in enumerate(FIELDS)}
    with _np.errstate(invalid='ignore', divide='ignore'):
        a, b, c = -st[..., f['h_xx']] / lam, -st[..., f['h_xy']] / lam, -st[..., f['h_yy']] / lam
        p0 = 1.0 / offset_sigma ** 2 if offset_sigma > 0 else 0.0
        a, c = a + p0, c + p0
        det = a * c - b * b
        ok = _np.isfinite(det) & (det > 0) & (a > 0)
        cov = _np.stack([c / det, -b / det, a / det], -1)
    cov[~ok] = [offset_sigma ** 2, 0.0, offset_sigma ** 2] if offset_sigma > 0 else _np.nan
    return cov, lam


def _axis_var(S, lam, prior_var):
    """Variance of each curvature axis: 1 / (-curvature / lambda + 1 / prior_var), the prior alone (NaN without one)
    where the CC gives no usable curvature, NaN for an axis that was not searched."""
    curv = S['projections/covar'][..., 1:].astype(_np.float64)
    pv = _np.asarray(prior_var, _np.float64)
    q0 = _np.where(pv > 0, 1.0 / _np.where(pv > 0, pv, 1.0), 0.0)
    with _np.errstate(invalid='ignore', divide='ignore'):
        prec = -curv / lam[..., None] + q0
        var = _np.where(_np.isfinite(prec) & (prec > 0), 1.0 / prec, _np.where(pv > 0, pv, _np.nan))
    steps = _np.asarray(S.get('covar_steps', _np.ones(curv.shape[-1])), _np.float64)
    var[..., ~(steps > 0)] = _np.nan
    return var


def posterior(S, offset_sigma=0.0, angle_sigma=0.0):
    """Posterior uncertainty of an ``'alignment_covar'`` alignment.

    The log-likelihood is ``CC / lambda`` (the convention of the aligner priors), so the precision of the shift is
    ``-H / lambda + I / offset_sigma^2`` (H the CC Hessian of the peak, px) and of each searched angle
    ``-curvature / lambda + 1 / angle_sigma^2`` (deg).  Where the CC gives no usable curvature the prior alone (NaN
    without a prior).

    Parameters
    ----------
    S : dict
        :func:`read` of the statistics.
    offset_sigma, angle_sigma : float
        The aligner priors (pixels, degrees); 0: none.

    Returns
    -------
    post_cov : ndarray (n_ptcl, K, 3) float32
        [xx, xy, yy] of the shift (px^2).
    ang_var : ndarray (n_ptcl, K, 3) float32
        Variance (deg^2) of the rotation about detector x, y and z; NaN for an axis that was not searched.
    """
    if kind(S) != 'alignment_covar':
        raise ValueError('posterior needs the statistics of an "alignment_covar" alignment (got %s).' % kind(S))
    cov, lam = _shift_cov(S, offset_sigma)
    var = _axis_var(S, lam, [angle_sigma ** 2] * 3)
    return cov.astype(_np.float32), var.astype(_np.float32)


def ctf_posterior(S, offset_sigma=0.0, defocus_sigma=0.0, phase_sigma_deg=0.0):
    """Posterior uncertainty of a ``'refinement_covar'`` CTF refinement (see :func:`posterior`).

    The defocus prior of the refiner, sigma on each of dU and dV, is sigma^2 / 2 on the mean defocus (U + V) / 2 and
    on the astigmatism (U - V) / 2; the astigmatism angle has no prior.

    Parameters
    ----------
    S : dict
        :func:`read` of the statistics.
    offset_sigma, defocus_sigma, phase_sigma_deg : float
        The CTF refiner priors (pixels, Angstroms, degrees); 0: none.

    Returns
    -------
    post_cov : ndarray (n_ptcl, K, 3) float32
        [xx, xy, yy] of the shift (px^2).
    ctf_var : ndarray (n_ptcl, K, 4) float32
        Variance of the mean defocus (A^2), the astigmatism (A^2), the astigmatism angle (deg^2) and the phase shift
        (deg^2); NaN for an axis that was not searched.
    """
    if kind(S) != 'refinement_covar':
        raise ValueError('ctf_posterior needs the statistics of a "refinement_covar" CTF refinement (got %s).' % kind(S))
    cov, lam = _shift_cov(S, offset_sigma)
    d2 = 0.5 * defocus_sigma ** 2
    var = _axis_var(S, lam, [d2, d2, 0.0, phase_sigma_deg ** 2])
    return cov.astype(_np.float32), var.astype(_np.float32)


def append(stats, S, **arrays):
    """Add arrays to the statistics dict ``S`` and rewrite the file ``stats`` (a ``.stats.npz`` or the ``.ptclsraw``
    output it belongs to) with them."""
    S.update(arrays)
    out = path(stats) if stats.endswith('.ptclsraw') else stats
    _np.savez(out + '.tmp.npz', **S)
    import os as _os
    _os.replace(out + '.tmp.npz', out)
