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

"""Diagnostic tools for subtomogram averaging projects.

Functions here are read-only: they never modify project files.
"""

import numpy as _np

import susan.data    as _ssa_data
import susan.io      as _ssa_io
import susan.modules as _ssa_modules

from susan.project.SubtomoAvg import SubtomoAvgBase as _SubtomoAvgBase


def bandpass_shell_sweep(
    sta: _SubtomoAvgBase,
    ite: int,
    apix: float,
    shell_hw: int = 5,
    threshold: float = 0.01,
    shell_pick: str = 'first',
    use_halfsets: bool = False,
    save_cc: str = None,
    fpix_max: int = None,
):
    """Measure per-particle, per-projection signal as a function of resolution.

    For each bandpass shell centred on successive Fourier-pixel frequencies, a
    2-D alignment is run with no angular or translational search to collect
    the per-projection CC scores.  The result is a ``(n_ptcl, n_proj)`` array
    of the resolution of the shell selected by *shell_pick* among those in
    which each projection shows signal above *threshold*.

    Parameters
    ----------
    sta : SubtomoAvgBase (or subclass)
        Project object.  Only :attr:`~SubtomoAvgBase.box_size`,
        :attr:`~SubtomoAvg.list_gpus_ids`, :meth:`~SubtomoAvgBase.path_refstxt`,
        :meth:`~SubtomoAvgBase.path_ptcls`, and
        :attr:`~SubtomoAvg.tomogram_file` are read.
    ite : int
        Iteration whose particles and reference are used.
    apix : float
        Pixel size in Å.
    shell_hw : int, optional
        Half-width of each bandpass shell in Fourier pixels.  Default: ``5``.
    threshold : float, optional
        Fraction of the global maximum positive CC used as the "above noise"
        cut-off.  The absolute threshold is computed as
        ``threshold * max(prj_cc[prj_cc > 0])``.  Default: ``0.1``.
    shell_pick : {'first', 'last'}, optional
        Which threshold crossing defines the cut-off, with shells ordered from
        low to high frequency.  ``'first'`` (default) reports the last shell
        of the leading above-threshold run — it stops at the first shell that
        drops below the cut-off, so an isolated high-frequency spike cannot
        pull the estimate out; ``'last'`` reports the highest-frequency shell
        above the cut-off anywhere in the sweep, ignoring any dips in between.
        The two agree when the CC decays monotonically.
    use_halfsets : bool, optional
        Whether to respect half-set assignments during alignment.
        Default: ``False``.
    save_cc : str, optional
        If given, write the full ``(n_shells, n_ptcl, n_proj)`` CC array to
        this MRC file path.
    fpix_max : int, optional
        Maximum Fourier-pixel shell centre.  Defaults to ``box_size // 2 - 1``.

    Returns
    -------
    max_res_A : numpy.ndarray, shape (n_ptcl, n_proj)
        Resolution (in Å) of the selected shell in which each projection
        carries signal above *threshold*.  Particles/projections with no
        signal in any shell are assigned the lowest-resolution shell value.
    """
    if shell_pick not in ('first', 'last'):
        raise ValueError("shell_pick must be 'first' or 'last', got %r"
                         % (shell_pick,))

    ali = _ssa_modules.Aligner()
    ali.dimensionality   = 2
    ali.list_gpus_ids    = sta.list_gpus_ids
    ali.verbosity        = 0

    ali.cone.span        = 0
    ali.inplane.span     = 0
    ali.offset.span      = (0, 0, 0)
    ali.ctf_correction   = 'on_reference'
    ali.normalize_type   = 'zero_mean_one_std'
    ali.cc_type          = 'cfsc'
    ali.use_halfsets     = use_halfsets

    box_size = sta.box_size
    rolloff  = 2

    if fpix_max is None:
        fpix_max = box_size // 2 - 1
    shell_centres = _np.arange(shell_hw + 1, fpix_max - shell_hw, shell_hw)

    refstxt  = sta.path_refstxt(ite)
    ptcls    = sta.path_ptcls(ite)
    tomofile = sta.tomogram_file

    prj_cc_shells = []   # (n_shells, n_ptcl, n_proj)

    for k, cen in enumerate(shell_centres):
        ali.bandpass.highpass = float(cen - shell_hw)
        ali.bandpass.lowpass  = float(cen + shell_hw)
        ali.bandpass.rolloff  = rolloff

        print(ali.bandpass)

        out_file = f'ali_shell_{k:03d}.ptclsraw'
        ali.align(out_file, refstxt, tomofile, ptcls, box_size)

        p = _ssa_data.Particles(out_file)
        prj_cc_shells.append(p.prj_cc.copy())

    prj_cc_shells = _np.array(prj_cc_shells)   # (n_shells, n_ptcl, n_proj)
    shell_res_A   = (box_size * apix) / shell_centres

    if save_cc is not None:
        _ssa_io.mrc.write(prj_cc_shells, save_cc)

    pos_cc = prj_cc_shells[prj_cc_shells > 0]
    cc_max = pos_cc.max() if pos_cc.size > 0 else 1.0
    above  = prj_cc_shells > threshold * cc_max

    # For each (ptcl, proj): the shell at the chosen threshold crossing.
    if shell_pick == 'first':
        # Last shell of the leading above-threshold run: stop at the first
        # shell that drops below the cut-off.
        below      = ~above
        n_shells   = above.shape[0]
        first_below = _np.where(below.any(axis=0),
                                _np.argmax(below, axis=0),
                                n_shells)
        sel = _np.maximum(first_below - 1, 0)
    else:
        # Highest-frequency shell above the cut-off, dips ignored.
        sel = (above.shape[0] - 1) - _np.argmax(above[::-1], axis=0)
    sel[~above.any(axis=0)] = 0

    max_res_A = shell_res_A[sel]   # (n_ptcl, n_proj) in Å
    return max_res_A


###########################################################################
# Inter-iteration delta / sigma estimation
#
# These measure how much each alignment quantity *moved* between iterations
# ``ite-1`` and ``ite``.  The ``estimate_delta_*`` helpers return the raw
# per-particle (or per-particle-per-projection) magnitudes; the
# ``estimate_sigma_*`` helpers reduce those to a single Gaussian width
# suitable as the prior (``offset_sigma`` / ``angle_sigma`` /
# ``defocus_sigma``) for the *next* iteration.
#
# Alignment: the ``estimate_delta_*`` arrays are aligned to iteration
# ``ite``'s particles in file order — their leading (particle) axis matches
# ``get_ptcls(ite)``, so they can be used directly to index ``prj_w`` etc.
# Particles present in ``ite`` but absent from ``ite-1`` are ``NaN``.
# Particles are matched on the composite ``(tomo_id, ptcl_id)`` key, since
# ``ptcl_id`` alone is not unique across tomograms.  The ``estimate_sigma_*``
# reductions ignore the ``NaN`` entries.
#
# Units: offset helpers default to pixels (``pixels=True``, using
# ``sta.pix_size``; pass ``pixels=False`` for Ångströms); angle helpers
# default to degrees (``degrees=True``; pass ``degrees=False`` for radians).
# These defaults match what the engine's priors expect.  Defocus is always
# in Ångströms (no pixel/degree conversion applies).
#
# IMPORTANT: these quantify inter-iteration *step size*, which shrinks
# toward zero as a search converges.  They are the right quantity for "how
# wide should the next search window be", but they UNDER-estimate the true
# residual alignment error near convergence.  Scale the result (``scale``
# argument, e.g. 1.5–2.0) before using it as a prior, or compare two
# independent half-set alignments instead of consecutive iterations.
###########################################################################


def _zyz_to_R(eu):
    """Vectorised ZYZ Euler (radians) -> rotation matrix.

    Uses SUSAN's convention ``R = Rz(eu0)·Ry(eu1)·Rz(eu2)`` (see the C++
    engine, ``math_cpu.h``).  ``eu`` has shape ``(..., 3)``; returns
    ``(..., 3, 3)``.
    """
    a, b, g = eu[..., 0], eu[..., 1], eu[..., 2]
    ca, sa = _np.cos(a), _np.sin(a)
    cb, sb = _np.cos(b), _np.sin(b)
    cg, sg = _np.cos(g), _np.sin(g)
    R = _np.empty(eu.shape[:-1] + (3, 3), dtype=_np.float64)
    R[..., 0, 0] = ca*cb*cg - sa*sg; R[..., 0, 1] = -ca*cb*sg - sa*cg; R[..., 0, 2] = ca*sb
    R[..., 1, 0] = sa*cb*cg + ca*sg; R[..., 1, 1] = -sa*cb*sg + ca*cg; R[..., 1, 2] = sa*sb
    R[..., 2, 0] = -sb*cg;           R[..., 2, 1] = sb*sg;             R[..., 2, 2] = cb
    return R


def _cone_inplane(R, degrees=True):
    """Cone (out-of-plane) and in-plane (twist about z) deviation angles.

    ``R`` is a deviation rotation of shape ``(..., 3, 3)``.  Matches the
    decomposition the engine's orientation prior penalises:
    ``cone = acos(R[2,2])`` and the swing-twist
    ``inplane = atan2(R[1,0]-R[0,1], R[0,0]+R[1,1])``.  Returned in degrees
    when *degrees* is ``True`` (default), otherwise in radians.
    """
    cone = _np.arccos(_np.clip(R[..., 2, 2], -1.0, 1.0))
    inpl = _np.abs(_np.arctan2(R[..., 1, 0] - R[..., 0, 1],
                               R[..., 0, 0] + R[..., 1, 1]))
    if degrees:
        cone = _np.degrees(cone)
        inpl = _np.degrees(inpl)
    return cone, inpl


def _geodesic(R, degrees=True):
    """Total rotation (geodesic) angle of a deviation rotation.

    ``R`` has shape ``(..., 3, 3)``.  Returns the single axis-angle
    magnitude ``acos((trace(R) - 1) / 2)`` — the full rotation carrying the
    previous pose onto the current one, folding cone and in-plane together.
    Returned in degrees when *degrees* is ``True`` (default), else radians.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    ang = _np.arccos(_np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    return _np.degrees(ang) if degrees else ang


def _match_pair(sta, ite):
    """Load iterations ``ite`` and ``ite-1`` and match particles.

    Matching uses the composite ``(tomo_id, ptcl_id)`` key — ``ptcl_id``
    alone is not unique across tomograms, so matching on it would collapse
    duplicates and pair the wrong particles.

    Returns ``(cur, prv, idx_c, idx_p)`` where ``idx_c`` / ``idx_p`` are row
    indices into ``cur`` / ``prv`` for the particles present in *both*
    iterations, aligned to each other.  ``idx_c`` indexes ``cur`` in its
    original (file) order positions, so it can be used to scatter results
    back into a current-iteration-aligned array (see :func:`_to_current`).
    """
    if ite < 1:
        raise ValueError('iteration delta requires ite >= 1 (needs ite-1).')
    cur = sta.get_ptcls(ite)
    prv = sta.get_ptcls(ite - 1)
    mult = int(max(int(cur.ptcl_id.max(initial=0)),
                   int(prv.ptcl_id.max(initial=0)))) + 1
    kc = cur.tomo_id.astype(_np.int64) * mult + cur.ptcl_id
    kp = prv.tomo_id.astype(_np.int64) * mult + prv.ptcl_id
    common, ic, ip = _np.intersect1d(kc, kp, return_indices=True)
    if common.size == 0:
        raise ValueError('iterations %d and %d share no particles.'
                         % (ite, ite - 1))
    return cur, prv, ic, ip


def _to_current(values, idx_c, n_cur):
    """Scatter per-matched-particle *values* into a current-aligned array.

    Returns a float array whose first axis has length *n_cur* (the current
    iteration's particle count, in file order); rows for particles absent
    from the previous iteration are left as ``NaN``.  ``values`` is indexed
    on its first axis and may carry trailing dimensions (e.g. projections).
    """
    out = _np.full((n_cur,) + _np.shape(values)[1:], _np.nan, dtype=_np.float64)
    out[idx_c] = values
    return out


def _rms_sigma(delta, dof, clip_pct=5.0, scale=1.0):
    """Reduce a folded/positive delta sample to a Gaussian σ.

    For a deviation with ``dof`` independent zero-mean Gaussian components
    of width σ, the magnitude δ satisfies ``E[δ²] = dof · σ²``; this returns
    ``scale · sqrt(mean(δ²) / dof)`` after discarding the top *clip_pct*
    percent (ref-switchers / junk that would otherwise inflate σ).  NaNs
    (excluded projections) are ignored.
    """
    d = _np.asarray(delta, dtype=_np.float64).ravel()
    d = d[_np.isfinite(d)]
    if d.size == 0:
        return 0.0
    if clip_pct and clip_pct > 0:
        hi = _np.percentile(d, 100.0 - clip_pct)
        d = d[d <= hi]
        if d.size == 0:
            return 0.0
    return float(scale * _np.sqrt(_np.mean(d * d) / dof))


# --------------------------------------------------------------------------
# Per-particle / per-projection deltas between ite-1 and ite
# --------------------------------------------------------------------------

def estimate_delta_3D_offset(sta, ite, ref=0, pixels=True):
    """Per-particle 3-D translation change ``‖Δt‖`` between ``ite-1`` and ``ite``.

    Parameters
    ----------
    pixels : bool, optional
        If ``True`` (default) the result is returned in pixels, using
        ``sta.pix_size``; if ``False`` it is returned in Ångströms.

    Returns
    -------
    numpy.ndarray, shape (n_ptcl,)
        Shift-change magnitude (pixels or Å), aligned to iteration *ite*'s
        particles (file order).  Particles absent from ``ite-1`` are ``NaN``.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    dt = cur.ali_t[ref][ic] - prv.ali_t[ref][ip]      # (Nc, 3) Å
    d  = _np.linalg.norm(dt, axis=1)
    if pixels:
        d = d / sta.pix_size
    return _to_current(d, ic, cur.ptcl_id.size)


def estimate_delta_2D_offset(sta, ite, pixels=True):
    """Per-particle, per-projection 2-D shift change ``‖Δt‖``.

    Parameters
    ----------
    pixels : bool, optional
        If ``True`` (default) the result is returned in pixels, using
        ``sta.pix_size``; if ``False`` it is returned in Ångströms.

    Returns
    -------
    numpy.ndarray, shape (n_ptcl, n_proj)
        Shift-change magnitude (pixels or Å), aligned to iteration *ite*'s
        particles (file order).  ``NaN`` for particles absent from ``ite-1``
        and for projections disabled (``prj_w<=0``) in either iteration.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    dt = cur.prj_t[ic] - prv.prj_t[ip]                # (Nc, P, 2) Å
    d  = _np.linalg.norm(dt, axis=-1).astype(_np.float64)
    if pixels:
        d = d / sta.pix_size
    mask = (cur.prj_w[ic] > 0) & (prv.prj_w[ip] > 0)
    d[~mask] = _np.nan
    return _to_current(d, ic, cur.ptcl_id.size)


def estimate_delta_3D_angle(sta, ite, ref=0, degrees=True):
    """Per-particle orientation change, decomposed into cone and in-plane.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the angles are returned in degrees; if
        ``False`` in radians.

    Returns
    -------
    numpy.ndarray, shape (2, n_ptcl)
        ``[0]`` cone (out-of-plane), ``[1]`` in-plane (twist), units per
        *degrees*, aligned to iteration *ite*'s particles (file order).
        Particles absent from ``ite-1`` are ``NaN``.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    Rd = _zyz_to_R(cur.ali_eu[ref][ic]) @ \
         _np.swapaxes(_zyz_to_R(prv.ali_eu[ref][ip]), -1, -2)
    cone, inpl = _cone_inplane(Rd, degrees=degrees)
    n = cur.ptcl_id.size
    return _np.stack([_to_current(cone, ic, n), _to_current(inpl, ic, n)], axis=0)


def estimate_delta_2D_angle(sta, ite, degrees=True):
    """Per-particle, per-projection orientation change (cone + in-plane).

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the angles are returned in degrees; if
        ``False`` in radians.

    Returns
    -------
    numpy.ndarray, shape (2, n_ptcl, n_proj)
        ``[0]`` cone, ``[1]`` in-plane (units per *degrees*), aligned to
        iteration *ite*'s particles (file order).  ``NaN`` for particles
        absent from ``ite-1`` and for disabled projections.  In 2-D
        per-projection alignment the cone component is usually weakly
        constrained — the in-plane slice is the meaningful one for
        ``angle_sigma``.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    Rd = _zyz_to_R(cur.prj_eu[ic]) @ \
         _np.swapaxes(_zyz_to_R(prv.prj_eu[ip]), -1, -2)
    cone, inpl = _cone_inplane(Rd, degrees=degrees)
    mask = (cur.prj_w[ic] > 0) & (prv.prj_w[ip] > 0)
    cone[~mask] = _np.nan
    inpl[~mask] = _np.nan
    n = cur.ptcl_id.size
    return _np.stack([_to_current(cone, ic, n), _to_current(inpl, ic, n)], axis=0)


def estimate_delta_3D_total_angle(sta, ite, ref=0, degrees=True):
    """Per-particle total (geodesic) orientation change between iterations.

    Unlike :func:`estimate_delta_3D_angle`, this returns a *single* angle per
    particle — the full rotation magnitude carrying the previous pose onto
    the current one, combining cone and in-plane into one value.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the angle is returned in degrees; if ``False``
        in radians.

    Returns
    -------
    numpy.ndarray, shape (n_ptcl,)
        Total rotation magnitude (units per *degrees*), aligned to iteration
        *ite*'s particles (file order).  Particles absent from ``ite-1`` are
        ``NaN``.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    Rd = _zyz_to_R(cur.ali_eu[ref][ic]) @ \
         _np.swapaxes(_zyz_to_R(prv.ali_eu[ref][ip]), -1, -2)
    return _to_current(_geodesic(Rd, degrees=degrees), ic, cur.ptcl_id.size)


def estimate_delta_2D_total_angle(sta, ite, degrees=True):
    """Per-particle, per-projection total (geodesic) orientation change.

    Single-angle counterpart of :func:`estimate_delta_2D_angle`.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the angle is returned in degrees; if ``False``
        in radians.

    Returns
    -------
    numpy.ndarray, shape (n_ptcl, n_proj)
        Total rotation magnitude (units per *degrees*), aligned to iteration
        *ite*'s particles (file order).  ``NaN`` for particles absent from
        ``ite-1`` and for disabled projections.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    Rd = _zyz_to_R(cur.prj_eu[ic]) @ \
         _np.swapaxes(_zyz_to_R(prv.prj_eu[ip]), -1, -2)
    d = _geodesic(Rd, degrees=degrees)
    mask = (cur.prj_w[ic] > 0) & (prv.prj_w[ip] > 0)
    d[~mask] = _np.nan
    return _to_current(d, ic, cur.ptcl_id.size)


def estimate_delta_defocus(sta, ite):
    """Per-particle, per-projection defocus change ``√(ΔU² + ΔV²)``.

    Returns
    -------
    numpy.ndarray, shape (n_ptcl, n_proj)
        Defocus-change magnitude in Å, aligned to iteration *ite*'s particles
        (file order).  ``NaN`` for particles absent from ``ite-1`` and for
        disabled projections.
    """
    cur, prv, ic, ip = _match_pair(sta, ite)
    dU = cur.def_U[ic].astype(_np.float64) - prv.def_U[ip]
    dV = cur.def_V[ic].astype(_np.float64) - prv.def_V[ip]
    d  = _np.sqrt(dU*dU + dV*dV)                       # (Nc, P) Å
    mask = (cur.prj_w[ic] > 0) & (prv.prj_w[ip] > 0)
    d[~mask] = _np.nan
    return _to_current(d, ic, cur.ptcl_id.size)


# --------------------------------------------------------------------------
# Sigma estimates for the next iteration's priors
# --------------------------------------------------------------------------

def estimate_sigma_3D_offset(sta, ite, ref=0, pixels=True, clip_pct=5.0, scale=1.0):
    """Estimate ``offset_sigma`` for 3-D alignment.

    Reduces :func:`estimate_delta_3D_offset` (3 translational DOF) to a
    per-axis Gaussian width.

    Parameters
    ----------
    pixels : bool, optional
        If ``True`` (default) the result is in pixels (using
        ``sta.pix_size``) — the unit the engine's ``offset_sigma`` expects;
        if ``False`` it is in Ångströms.
    """
    d = estimate_delta_3D_offset(sta, ite, ref=ref, pixels=pixels)
    return _rms_sigma(d, dof=3, clip_pct=clip_pct, scale=scale)


def estimate_sigma_2D_offset(sta, ite, pixels=True, clip_pct=5.0, scale=1.0):
    """Estimate ``offset_sigma`` for 2-D per-projection alignment.

    Reduces :func:`estimate_delta_2D_offset` (2 translational DOF) to a
    per-axis Gaussian width.

    Parameters
    ----------
    pixels : bool, optional
        If ``True`` (default) the result is in pixels (using
        ``sta.pix_size``); if ``False`` it is in Ångströms.
    """
    d = estimate_delta_2D_offset(sta, ite, pixels=pixels)
    return _rms_sigma(d, dof=2, clip_pct=clip_pct, scale=scale)


def estimate_sigma_3D_angle(sta, ite, ref=0, degrees=True, clip_pct=5.0, scale=1.0):
    """Estimate the angular prior width for 3-D alignment.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the result is in degrees — the unit the
        engine's ``angle_sigma`` expects; if ``False`` in radians.

    Returns
    -------
    (float, float)
        ``(sigma_cone, sigma_inplane)`` (units per *degrees*).  ``angle_sigma``
        is a single engine knob applied to both axes; pass whichever you
        prefer (or their combination — e.g. ``max``) to the next iteration.
    """
    d = estimate_delta_3D_angle(sta, ite, ref=ref, degrees=degrees)   # (2, N)
    sigma_cone = _rms_sigma(d[0], dof=2, clip_pct=clip_pct, scale=scale)
    sigma_inpl = _rms_sigma(d[1], dof=1, clip_pct=clip_pct, scale=scale)
    return sigma_cone, sigma_inpl


def estimate_sigma_2D_angle(sta, ite, degrees=True, clip_pct=5.0, scale=1.0):
    """Estimate the angular prior width for 2-D alignment.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the result is in degrees; if ``False`` in
        radians.

    Returns
    -------
    (float, float)
        ``(sigma_cone, sigma_inplane)`` (units per *degrees*), pooled over all
        projections.  The in-plane value is normally the relevant one for
        ``angle_sigma`` in 2-D mode.
    """
    d = estimate_delta_2D_angle(sta, ite, degrees=degrees)            # (2, N, P)
    sigma_cone = _rms_sigma(d[0], dof=2, clip_pct=clip_pct, scale=scale)
    sigma_inpl = _rms_sigma(d[1], dof=1, clip_pct=clip_pct, scale=scale)
    return sigma_cone, sigma_inpl


def estimate_sigma_3D_total_angle(sta, ite, ref=0, degrees=True,
                                  clip_pct=5.0, scale=1.0):
    """Estimate a single combined angular prior width for 3-D alignment.

    Reduces :func:`estimate_delta_3D_total_angle` (the geodesic rotation,
    3 rotational DOF: cone + in-plane) to one Gaussian width — a single value
    usable directly as ``angle_sigma``, without choosing between the cone and
    in-plane components.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the result is in degrees — the unit the
        engine's ``angle_sigma`` expects; if ``False`` in radians.

    Returns
    -------
    float
        Combined angular width (units per *degrees*).
    """
    d = estimate_delta_3D_total_angle(sta, ite, ref=ref, degrees=degrees)
    return _rms_sigma(d, dof=3, clip_pct=clip_pct, scale=scale)


def estimate_sigma_2D_total_angle(sta, ite, degrees=True,
                                  clip_pct=5.0, scale=1.0):
    """Estimate a single combined angular prior width for 2-D alignment.

    Single-value counterpart of :func:`estimate_sigma_2D_angle`, reducing the
    geodesic rotation (:func:`estimate_delta_2D_total_angle`, 3 rotational
    DOF) pooled over all projections.

    Parameters
    ----------
    degrees : bool, optional
        If ``True`` (default) the result is in degrees; if ``False`` in
        radians.

    Returns
    -------
    float
        Combined angular width (units per *degrees*).
    """
    d = estimate_delta_2D_total_angle(sta, ite, degrees=degrees)
    return _rms_sigma(d, dof=3, clip_pct=clip_pct, scale=scale)


def estimate_sigma_defocus(sta, ite, clip_pct=5.0, scale=1.0):
    """Estimate ``defocus_sigma`` (Å) for CTF refinement.

    Reduces :func:`estimate_delta_defocus` (2 defocus DOF, ``dU``/``dV``)
    to a Gaussian width.  For non-astigmatism refinement (``dV = dU``) the
    result is still a usable scale, though the engine's non-astigmatic
    prior uses a slightly different normalisation.
    """
    d = estimate_delta_defocus(sta, ite)              # Å
    return _rms_sigma(d, dof=2, clip_pct=clip_pct, scale=scale)
